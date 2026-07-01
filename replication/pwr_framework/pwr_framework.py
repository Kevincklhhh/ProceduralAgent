#!/usr/bin/env python3
"""Plan-Watch-Recover framework implementation.

This module implements the core factorization from the PWR paper:

    p(dt, ut, Pt | ot, Pt-1)
      = p_theta(dt, ut | ot, Pt-1) * p_phi(Pt | dt, ut, ot, Pt-1)

The duplex interaction model runs at every observation and decides whether to
stay silent or interrupt. The background planner is invoked only when the duplex
interrupts; on silent turns the plan is carried forward unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


DUPLEX_SYSTEM_PROMPT = (
    "You are a helpful, proactive AI assistant guiding a user through a "
    "step-by-step procedural task. You observe the user's activity through "
    "egocentric video from their perspective.\n\n"
    "Responsibilities:\n"
    "1. Track which step the user is currently performing.\n"
    "2. Detect when a step is completed and proactively announce the next step.\n"
    "3. Detect mistakes or deviations from the expected procedure, including "
    "skipped steps, wrong ordering, technique errors, substitutions, or "
    "improvisations.\n"
    "4. When a deviation is detected, suggest a recovery action.\n"
    "5. If no guidance is needed, stay silent.\n\n"
    "Use the cached procedural plan and current-step visual cues as the main "
    "decision criteria. The plan tells you what should be happening; the frames "
    "tell you whether it is happening.\n\n"
    "Return JSON only, exactly this schema:\n"
    '{"decision":"silent|interrupt",'
    '"interrupt_type":"step_complete|deviation|reactive|reminder|none",'
    '"completed_step_id":"<step id or null>",'
    '"event_class":"precondition_violation|parameter_violation|execution_error|none",'
    '"event_subtype":"order|missing_step|technique|preparation|measurement|temperature|timing|none",'
    '"event_id":"<id or none>",'
    '"utterance":"<brief assistant text, empty if silent>",'
    '"evidence":"<one short visual reason>"}'
)


PLANNER_SYSTEM_PROMPT = (
    "You are the background planner in a Plan-Watch-Recover procedural "
    "assistant. You never speak directly to the user. You receive the prior "
    "structured plan, the duplex interaction model's interrupt decision and "
    "utterance, and the most recent egocentric video clip.\n\n"
    "Your job is to revise the cached procedural plan only as needed. If a step "
    "completed, mark it completed and select the next current step. If the user "
    "deviated from the expected procedure, keep or revise the current step and "
    "write a concrete recovery note. Generate visual cues that are strictly "
    "visible in egocentric video.\n\n"
    "Return JSON only, exactly this schema:\n"
    '{"mode":"start_proactive|update_proactive|recover|end_proactive",'
    '"completed_step_ids":["<step id>","..."],'
    '"current_step_id":"<step id or null>",'
    '"recovery":"<recovery note or empty>",'
    '"step_complete_cues":["<visible cue>","..."],'
    '"step_incomplete_cues":["<visible cue>","..."],'
    '"reason":"<one short reason>"}'
)


@dataclass
class PWRConfig:
    tick_s: float = 0.5
    clip_window_s: float = 8.0
    frames_per_clip: int = 8
    max_clips: int = 15
    max_dim: int = 768
    jpeg_quality: int = 85
    max_seconds: float | None = None
    mock_step_s: float = 20.0


@dataclass
class ModelCall:
    normalized: dict[str, Any]
    raw: str
    error: str | None
    latency_s: float


@dataclass
class ClipBatch:
    jpegs: list[bytes]
    meta: list[dict[str, Any]]


@dataclass
class PWRPlanState:
    task: dict[str, Any]
    steps: list[dict[str, Any]]
    completed: list[Any] = field(default_factory=list)
    current_step_id: Any = None
    complete_cues: list[str] = field(default_factory=list)
    incomplete_cues: list[str] = field(default_factory=list)
    recovery: str = ""
    update_times: list[float] = field(default_factory=lambda: [0.0])

    def __post_init__(self) -> None:
        self.by_id = {sid_s(step["step_id"]): step for step in self.steps}
        self.valid = {sid_s(step["step_id"]): step["step_id"] for step in self.steps}
        self.ranks = step_ranks(self.steps)
        if self.current_step_id is None:
            frontier = self.next_frontier()
            self.current_step_id = frontier[0]["step_id"] if frontier else None
        self.refresh_cues()

    def completed_set(self) -> set[str]:
        return {sid_s(step_id) for step_id in self.completed}

    def mark_completed(self, step_id: Any) -> bool:
        step_id = normalize_step_id(step_id, self.valid)
        if step_id is None:
            return False
        if sid_s(step_id) not in self.completed_set():
            self.completed.append(step_id)
            return True
        return False

    def next_frontier(self) -> list[dict[str, Any]]:
        done = self.completed_set()
        out = []
        for step in self.steps:
            step_id = sid_s(step["step_id"])
            if step_id in done:
                continue
            preds = {sid_s(pred) for pred in step.get("preconditions", [])}
            if preds.issubset(done):
                out.append(step)
        return sorted(out, key=lambda step: (self.ranks.get(sid_s(step["step_id"]), 0), step.get("order", 0)))

    def set_current(self, step_id: Any) -> None:
        step_id = normalize_step_id(step_id, self.valid)
        if step_id is None or sid_s(step_id) in self.completed_set():
            frontier = self.next_frontier()
            self.current_step_id = frontier[0]["step_id"] if frontier else None
        else:
            self.current_step_id = step_id
        self.refresh_cues()

    def refresh_cues(self, complete: list[str] | None = None, incomplete: list[str] | None = None) -> None:
        step = self.by_id.get(sid_s(self.current_step_id))
        if not step:
            self.complete_cues = []
            self.incomplete_cues = []
            return
        default_complete, default_incomplete = default_cues(step)
        complete = [str(item) for item in (complete or []) if str(item).strip()]
        incomplete = [str(item) for item in (incomplete or []) if str(item).strip()]
        self.complete_cues = complete or default_complete
        self.incomplete_cues = incomplete or default_incomplete

    def apply_planner_update(self, update: dict[str, Any], t_s: float) -> bool:
        before = sid_s(self.current_step_id)
        for step_id in update.get("completed_step_ids") or []:
            self.mark_completed(step_id)
        self.set_current(update.get("current_step_id"))
        self.recovery = str(update.get("recovery") or "")
        self.refresh_cues(update.get("step_complete_cues"), update.get("step_incomplete_cues"))
        changed = before != sid_s(self.current_step_id) or bool(self.recovery)
        if changed and (not self.update_times or abs(self.update_times[-1] - t_s) > 1e-6):
            self.update_times.append(float(t_s))
        return changed

    def as_text(self) -> str:
        current_id = sid_s(self.current_step_id)
        done = self.completed_set()
        rows = []
        for step in self.steps:
            step_id = sid_s(step["step_id"])
            if step_id in done:
                tag = "completed"
            elif step_id == current_id:
                tag = "current"
            else:
                preds = {sid_s(pred) for pred in step.get("preconditions", [])}
                tag = "next" if preds.issubset(done) else "remaining"
            rows.append(f"- [{tag}] step_id={step['step_id']}: {step.get('instruction', '')}")
        complete = "\n".join(f"- {cue}" for cue in self.complete_cues) or "- none"
        incomplete = "\n".join(f"- {cue}" for cue in self.incomplete_cues) or "- none"
        recovery = f"\nRecovery note: {self.recovery}" if self.recovery else ""
        return (
            f"Task: {task_title(self.task)}\n"
            "Procedural plan:\n"
            + "\n".join(rows)
            + "\n\nCurrent-step complete cues:\n"
            + complete
            + "\n\nCurrent-step incomplete cues:\n"
            + incomplete
            + recovery
        )


class QwenVideoBackend:
    """OpenAI-compatible multimodal chat backend used for Qwen video servers."""

    def __init__(self, timeout: float = 180.0, retries: int = 2, backoff_s: float = 2.0):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + (
            "" if base.endswith("/chat/completions")
            else "/chat/completions" if base.endswith("/v1")
            else "/v1/chat/completions"
        )
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, jpegs: list[bytes], system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()},
            }
            for jpeg in jpegs
        ]
        content.append({"type": "text", "text": user_prompt})
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        last = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]["content"]
                if isinstance(message, list):
                    return "".join(part.get("text", "") for part in message if isinstance(part, dict))
                return str(message)
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last


class MockVisionBackend:
    model = "mock-pwr-framework"

    def call(self, jpegs: list[bytes], system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        return "{}"


class DuplexInteractionModel:
    """User-facing PWR duplex model p_theta(dt, ut | ot, Pt-1)."""

    def __init__(self, backend: Any, mode: str = "qwen", mock_step_s: float = 20.0):
        self.backend = backend
        self.mode = mode
        self.mock_step_s = mock_step_s

    @property
    def model_name(self) -> str:
        return getattr(self.backend, "model", self.mode)

    def decide(self, jpegs: list[bytes], prompt: str, plan_state: PWRPlanState, t_s: float) -> ModelCall:
        if self.mode == "egoproactive_gold":
            raw_obj = self._gold_decision(plan_state, t_s)
            return ModelCall(raw_obj, json.dumps(raw_obj), None, 0.0)
        if self.mode == "mock":
            raw_obj = self._mock_decision(plan_state, t_s)
            return ModelCall(raw_obj, json.dumps(raw_obj), None, 0.0)
        start = time.time()
        try:
            raw = self.backend.call(jpegs, DUPLEX_SYSTEM_PROMPT, prompt)
            error = None
        except Exception as exc:  # pragma: no cover - network path
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        latency = time.time() - start
        return ModelCall(normalize_duplex(parse_duplex_response(raw), plan_state), raw, error, latency)

    def _mock_decision(self, plan_state: PWRPlanState, t_s: float) -> dict[str, Any]:
        last_update = plan_state.update_times[-1] if plan_state.update_times else 0.0
        should_interrupt = plan_state.current_step_id is not None and (t_s - last_update) >= self.mock_step_s
        if not should_interrupt:
            return {
                "decision": "silent",
                "interrupt_type": "none",
                "completed_step_id": None,
                "event_class": "none",
                "event_subtype": "none",
                "event_id": "none",
                "utterance": "",
                "evidence": "mock normal progress",
            }
        return {
            "decision": "interrupt",
            "interrupt_type": "step_complete",
            "completed_step_id": plan_state.current_step_id,
            "event_class": "none",
            "event_subtype": "none",
            "event_id": "mock_step_complete",
            "utterance": "Move to the next step.",
            "evidence": "mock elapsed-step completion",
        }

    def _gold_decision(self, plan_state: PWRPlanState, t_s: float) -> dict[str, Any]:
        point = None
        for candidate in plan_state.task.get("egoproactive_decision_points", []):
            start, end = candidate.get("interval_s", [None, None])
            if start is None or end is None:
                continue
            if float(start) <= float(t_s) < float(end):
                point = candidate
                break
        if point is None:
            return {
                "decision": "silent",
                "interrupt_type": "none",
                "completed_step_id": None,
                "event_class": "none",
                "event_subtype": "none",
                "event_id": "none",
                "utterance": "",
                "evidence": "no EgoProactive decision interval covers this observation",
            }
        decision = point.get("decision", "silent")
        answer = point.get("answer", "")
        utterance = re.sub(r"^\$(?:interrupt|silent)\$", "", answer).strip()
        return {
            "decision": decision if decision in {"interrupt", "silent"} else "silent",
            "interrupt_type": "step_complete" if decision == "interrupt" else "none",
            "completed_step_id": point.get("step_id") if decision == "interrupt" else None,
            "event_class": "none",
            "event_subtype": "none",
            "event_id": f"egoproactive_{point.get('index', 'unknown')}",
            "utterance": utterance if decision == "interrupt" else "",
            "evidence": f"EgoProactive gold interval {point.get('index')} {point.get('interval_s')}",
        }


class BackgroundPlanner:
    """Background PWR planner p_phi(Pt | dt, ut, ot, Pt-1)."""

    def __init__(self, backend: Any, mode: str = "qwen"):
        self.backend = backend
        self.mode = mode

    @property
    def model_name(self) -> str:
        return getattr(self.backend, "model", self.mode)

    def update(self, jpegs: list[bytes], prompt: str, plan_state: PWRPlanState, duplex: dict[str, Any]) -> ModelCall:
        if self.mode in {"mock", "egoproactive_gold"}:
            update = normalize_planner({}, plan_state, duplex)
            return ModelCall(update, json.dumps(update), None, 0.0)
        start = time.time()
        try:
            raw = self.backend.call(jpegs, PLANNER_SYSTEM_PROMPT, prompt)
            error = None
        except Exception as exc:  # pragma: no cover - network path
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        latency = time.time() - start
        return ModelCall(normalize_planner(parse_planner_response(raw), plan_state, duplex), raw, error, latency)


class PlanAnchoredClipSampler:
    def __init__(self, cap: Any, fps: float, config: PWRConfig):
        self.cap = cap
        self.fps = fps
        self.config = config

    def collect(self, t_s: float, plan_state: PWRPlanState, planner_only: bool = False) -> ClipBatch:
        if planner_only:
            anchors = [("recent", float(t_s))]
        else:
            anchors = [("recent", float(t_s))]
            prior = [u for u in plan_state.update_times if u < t_s - 1e-6]
            for update_t in sorted(prior, reverse=True)[: max(0, self.config.max_clips - 1)]:
                anchors.append((f"plan_update@{update_t:.1f}", float(update_t)))
        all_jpegs: list[bytes] = []
        meta: list[dict[str, Any]] = []
        for label, end_t in anchors[: self.config.max_clips]:
            jpegs, times = self._sample_clip(end_t)
            if not jpegs:
                continue
            first = len(all_jpegs)
            all_jpegs.extend(jpegs)
            meta.append({
                "label": label,
                "clip_end_s": round(float(end_t), 3),
                "frame_times_s": times,
                "image_range": [first, len(all_jpegs) - 1],
            })
        return ClipBatch(all_jpegs, meta)

    def _sample_clip(self, end_t: float) -> tuple[list[bytes], list[float]]:
        start_t = max(0.0, float(end_t) - self.config.clip_window_s)
        if self.config.frames_per_clip <= 1:
            times = [float(end_t)]
        else:
            times = np.linspace(start_t, max(start_t, float(end_t)), self.config.frames_per_clip)
        deduped = []
        seen = set()
        for ts in times:
            key = round(float(ts), 3)
            if key not in seen:
                seen.add(key)
                deduped.append(float(ts))
        jpegs: list[bytes] = []
        used: list[float] = []
        for ts in deduped:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(ts * self.fps))))
            ok, frame = self.cap.read()
            if not ok:
                continue
            jpeg = jpeg_bytes(frame, max_dim=self.config.max_dim, jpeg_q=self.config.jpeg_quality)
            if jpeg is not None:
                jpegs.append(jpeg)
                used.append(round(ts, 3))
        return jpegs, used


class PWRFramework:
    """End-to-end PWR inference loop."""

    def __init__(self, duplex: DuplexInteractionModel, planner: BackgroundPlanner, config: PWRConfig):
        self.duplex = duplex
        self.planner = planner
        self.config = config

    def run(
        self,
        video_path: str | Path,
        task: dict[str, Any],
        rid: str | None = None,
        trace_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        steps = task_steps(task)
        plan_state = PWRPlanState(task=task, steps=steps)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        raw_duration = frame_count / fps if fps else 0.0
        duration = raw_duration
        if task.get("egoproactive_decision_points") and task.get("duration_in_sec"):
            duration = float(task["duration_in_sec"])
            fps = frame_count / duration if duration > 0 else fps
        end_t = min(duration, self.config.max_seconds) if self.config.max_seconds else duration
        sampler = PlanAnchoredClipSampler(cap, fps, self.config)

        trace_fh = None
        if trace_dir is not None and rid is not None:
            debug_dir = Path(trace_dir) / "pwr_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            trace_fh = (debug_dir / f"{rid}.jsonl").open("w")

        timeline = [(0.0, plan_state.current_step_id)]
        interventions: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        dialogue: list[dict[str, Any]] = []
        costs = {
            "duplex_calls": 0,
            "planner_calls": 0,
            "frames_sent": 0,
            "duplex_latency_s": 0.0,
            "planner_latency_s": 0.0,
            "parse_failures": 0,
        }

        tick_times = self._tick_times(task, end_t)
        for t_s in tick_times:
            batch = sampler.collect(t_s, plan_state, planner_only=False)
            if not batch.jpegs:
                break
            costs["duplex_calls"] += 1
            costs["frames_sent"] += len(batch.jpegs)
            duplex_prompt = build_duplex_prompt(plan_state, batch.meta, dialogue)
            duplex_call = self.duplex.decide(batch.jpegs, duplex_prompt, plan_state, t_s)
            costs["duplex_latency_s"] += duplex_call.latency_s
            if duplex_call.raw and safe_json(duplex_call.raw) is None and not duplex_call.raw.lstrip().startswith("$"):
                costs["parse_failures"] += 1
            duplex = duplex_call.normalized
            write_trace(trace_fh, {
                "kind": "duplex",
                "t_s": round(t_s, 3),
                "plan_current": plan_state.current_step_id,
                "clip_meta": batch.meta,
                "n_images": len(batch.jpegs),
                "system_prompt": DUPLEX_SYSTEM_PROMPT,
                "user_prompt": duplex_prompt,
                "raw": duplex_call.raw,
                "normalized": duplex,
                "error": duplex_call.error,
                "latency_s": round(duplex_call.latency_s, 2),
            })

            if duplex["decision"] == "interrupt":
                interventions.append({
                    "t_s": round(t_s, 2),
                    "type": duplex.get("interrupt_type", "none"),
                    "step_id": duplex.get("completed_step_id") or plan_state.current_step_id,
                    "event_class": duplex.get("event_class", "none"),
                    "event_subtype": duplex.get("event_subtype", "none"),
                    "utterance": duplex.get("utterance", ""),
                    "evidence": duplex.get("evidence", ""),
                })
                events.append({
                    "t": round(t_s, 2),
                    "class": duplex.get("event_class", "none"),
                    "subtype": duplex.get("event_subtype", "none"),
                    "id": duplex.get("event_id") or f"pwr_{duplex.get('interrupt_type', 'interrupt')}",
                    "message": duplex.get("utterance", ""),
                })
                dialogue.append({
                    "t_s": float(t_s),
                    "assistant": duplex.get("utterance", ""),
                    "evidence": duplex.get("evidence", ""),
                })

                planner_batch = sampler.collect(t_s, plan_state, planner_only=True)
                costs["planner_calls"] += 1
                costs["frames_sent"] += len(planner_batch.jpegs)
                planner_prompt = build_planner_prompt(plan_state, duplex, planner_batch.meta)
                planner_call = self.planner.update(planner_batch.jpegs, planner_prompt, plan_state, duplex)
                costs["planner_latency_s"] += planner_call.latency_s
                if planner_call.raw and safe_json(planner_call.raw) is None:
                    costs["parse_failures"] += 1
                old_current = plan_state.current_step_id
                update = planner_call.normalized
                plan_state.apply_planner_update(update, t_s)
                if sid_s(old_current) != sid_s(plan_state.current_step_id):
                    timeline.append((float(t_s), plan_state.current_step_id))
                if (update.get("mode") == "recover" or duplex.get("interrupt_type") == "deviation") and events:
                    events[-1]["message"] = update.get("recovery") or events[-1].get("message", "")
                write_trace(trace_fh, {
                    "kind": "planner",
                    "t_s": round(t_s, 3),
                    "old_current": old_current,
                    "new_current": plan_state.current_step_id,
                    "clip_meta": planner_batch.meta,
                    "n_images": len(planner_batch.jpegs),
                    "system_prompt": PLANNER_SYSTEM_PROMPT,
                    "user_prompt": planner_prompt,
                    "raw": planner_call.raw,
                    "normalized": update,
                    "error": planner_call.error,
                    "latency_s": round(planner_call.latency_s, 2),
                })

            if verbose:
                print(
                    f"  t={t_s:6.1f}s decision={duplex['decision']} "
                    f"current={plan_state.current_step_id} planners={costs['planner_calls']}"
                )

        cap.release()
        if trace_fh is not None:
            trace_fh.close()

        cost_out = {
            "vlm_calls": costs["duplex_calls"] + costs["planner_calls"],
            "duplex_calls": costs["duplex_calls"],
            "planner_calls": costs["planner_calls"],
            "frames_sent": costs["frames_sent"],
            "vlm_latency_total_s": round(costs["duplex_latency_s"] + costs["planner_latency_s"], 2),
            "duplex_latency_s": round(costs["duplex_latency_s"], 2),
            "planner_latency_s": round(costs["planner_latency_s"], 2),
            "compute_s": 0.0,
        }
        return {
            "stage_intervals": foreground_intervals(timeline, end_t),
            "events": events,
            "interventions": interventions,
            "escalation_requests": [],
            "cost": cost_out,
            "replication_manifest": self.manifest(),
            "_meta": {
                "duplex_model": self.duplex.model_name,
                "planner_model": self.planner.model_name,
                "evaluated_s": round(end_t, 2),
                "video_duration_s": round(duration, 2),
                "raw_video_duration_s": round(raw_duration, 2),
                "parse_failures": costs["parse_failures"],
                "task_id": task.get("task_id") or (task.get("task") or {}).get("task_id"),
                "final_plan": plan_state.as_text(),
                "plan_update_times": [round(x, 2) for x in plan_state.update_times],
                "trace": trace_dir is not None,
            },
        }

    def _tick_times(self, task: dict[str, Any], end_t: float) -> list[float]:
        if task.get("egoproactive_decision_points"):
            return [
                float(point["interval_s"][0])
                for point in task["egoproactive_decision_points"]
                if float(point["interval_s"][0]) <= end_t + 1e-6
            ]
        times = []
        t_s = 0.0
        while t_s <= end_t + 1e-6:
            times.append(round(t_s, 6))
            t_s += self.config.tick_s
        return times

    def manifest(self) -> dict[str, Any]:
        return {
            "claim": "PWR framework implementation following the paper factorization; trained checkpoints are replaced by configured backends.",
            "paper_mapping": [
                "Duplex interaction model consumes cached plan plus plan-anchored clips at every observation.",
                "Planner is invoked only when duplex emits interrupt; silent turns carry Pt-1 forward unchanged.",
                "Planner consumes the duplex output and the most recent clip, not the full video history.",
                "Plan state stores completed/current/remaining steps and current-step visual cues.",
            ],
            "config": {
                "tick_s": self.config.tick_s,
                "clip_window_s": self.config.clip_window_s,
                "frames_per_clip": self.config.frames_per_clip,
                "max_clips": self.config.max_clips,
            },
        }


def safe_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def parse_duplex_response(raw: str | None) -> dict[str, Any]:
    parsed = safe_json(raw)
    if parsed is not None:
        return parsed
    text = (raw or "").strip()
    if text.startswith("$interrupt$"):
        return {"decision": "interrupt", "utterance": text[len("$interrupt$"):].strip()}
    if text.startswith("$silent$"):
        return {"decision": "silent", "utterance": ""}
    return {}


def parse_planner_response(raw: str | None) -> dict[str, Any]:
    parsed = safe_json(raw)
    if parsed is not None:
        return parsed
    text = raw or ""
    if "<|mode|>" not in text:
        return {}
    out: dict[str, Any] = {}
    mode_match = re.search(r"<\|mode\|>\s*([^<\n]+)", text)
    current_match = re.search(r"\(Current\)\s*([^\n]+)", text)
    if mode_match:
        out["mode"] = mode_match.group(1).strip().split()[0]
    if current_match:
        out["current_step_id"] = current_match.group(1).strip()
    complete_match = re.search(r"<\|step_complete_cues\|>\s*([^<]+)", text)
    incomplete_match = re.search(r"<\|step_incomplete_cues\|>\s*([^<]+)", text)
    if complete_match:
        out["step_complete_cues"] = [complete_match.group(1).strip()]
    if incomplete_match:
        out["step_incomplete_cues"] = [incomplete_match.group(1).strip()]
    return out


def sid_s(step_id: Any) -> str | None:
    if step_id is None:
        return None
    value = str(step_id).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def normalize_step_id(raw: Any, valid: dict[str, Any]) -> Any:
    raw_s = sid_s(raw)
    if raw_s is None:
        return None
    return valid.get(raw_s)


def task_title(task: dict[str, Any]) -> str:
    nested = task.get("task") if isinstance(task.get("task"), dict) else {}
    return str(task.get("title") or nested.get("title") or task.get("task_id") or nested.get("task_id") or "unknown")


def task_steps(task: dict[str, Any]) -> list[dict[str, Any]]:
    if "steps" in task:
        return sorted(task["steps"], key=lambda step: (step.get("order", 0), sid_s(step.get("step_id"))))
    if "nodes" in task:
        steps = []
        for idx, node in enumerate(task.get("nodes", []), 1):
            steps.append({
                "step_id": node["step_id"],
                "order": node.get("order", idx),
                "instruction": node.get("instruction", ""),
                "preconditions": node.get("preconditions", []),
                "checks": node.get("checks", []),
            })
        return sorted(steps, key=lambda step: (step.get("order", 0), sid_s(step.get("step_id"))))
    graph = task.get("graph", {})
    steps = []
    for step in graph.get("steps", []):
        steps.append({
            "step_id": step.get("cc4d_step_id", step["step_id"]),
            "order": step.get("order", len(steps) + 1),
            "instruction": step.get("instruction", ""),
            "preconditions": step.get("requires", []),
        })
    for block in graph.get("step_blocks", []):
        for member in block.get("members", []):
            steps.append({
                "step_id": member.get("cc4d_step_id", member["step_id"]),
                "order": block.get("order", len(steps) + 1),
                "instruction": member.get("instruction", ""),
                "preconditions": block.get("requires", []),
            })
    return sorted(steps, key=lambda step: (step.get("order", 0), sid_s(step.get("step_id"))))


def step_ranks(steps: list[dict[str, Any]]) -> dict[str, int]:
    if steps and all("rank" in step for step in steps):
        return {sid_s(step["step_id"]): int(step["rank"]) for step in steps}
    by_id = {sid_s(step["step_id"]): step for step in steps}
    if any(step.get("preconditions") for step in steps):
        memo: dict[str, int] = {}

        def rank(step_id: Any) -> int:
            step_id_s = sid_s(step_id)
            if step_id_s in memo:
                return memo[step_id_s]
            step = by_id.get(step_id_s)
            preds = [sid_s(pred) for pred in (step.get("preconditions") or []) if sid_s(pred) in by_id] if step else []
            memo[step_id_s] = 0 if not preds else 1 + max(rank(pred) for pred in preds)
            return memo[step_id_s]

        return {sid_s(step["step_id"]): rank(step["step_id"]) for step in steps}
    return {sid_s(step["step_id"]): int(step.get("order", 0)) for step in steps}


def default_cues(step: dict[str, Any]) -> tuple[list[str], list[str]]:
    instruction = str(step.get("instruction", "")).strip() or f"step {step.get('step_id')}"
    return (
        [
            f"Visible evidence that this is finished: {instruction}.",
            "Hands move away from the main object or begin preparing the next action.",
        ],
        [
            f"The user is still doing or setting up: {instruction}.",
            "The required object state for the completed action is not yet visible.",
        ],
    )


def build_duplex_prompt(plan_state: PWRPlanState, clip_meta: list[dict[str, Any]], dialogue: list[dict[str, Any]]) -> str:
    clip_rows = "\n".join(
        f"- images {meta['image_range'][0]}..{meta['image_range'][1]}: {meta['label']}, "
        f"clip_end={meta['clip_end_s']}s, frame_times={meta['frame_times_s']}"
        for meta in clip_meta
    ) or "- no readable frames"
    recent = "\n".join(
        f"- t={turn['t_s']:.1f}s assistant={turn['assistant']} evidence={turn.get('evidence', '')}"
        for turn in dialogue[-5:]
    ) or "- none"
    return (
        plan_state.as_text()
        + "\n\nPlan-anchored visual context:\n"
        + clip_rows
        + "\n\nRecent dialogue/interventions:\n"
        + recent
        + "\n\nDecide whether to stay silent or interrupt now."
    )


def build_planner_prompt(plan_state: PWRPlanState, duplex: dict[str, Any], clip_meta: list[dict[str, Any]]) -> str:
    clip_rows = "\n".join(
        f"- images {meta['image_range'][0]}..{meta['image_range'][1]}: {meta['label']}, "
        f"clip_end={meta['clip_end_s']}s, frame_times={meta['frame_times_s']}"
        for meta in clip_meta
    ) or "- no readable frames"
    return (
        plan_state.as_text()
        + "\n\nDuplex output just given to the user:\n"
        + json.dumps(duplex, ensure_ascii=True)
        + "\n\nMost recent visual context for planner:\n"
        + clip_rows
        + "\n\nUpdate the cached plan Pt."
    )


def normalize_duplex(parsed: dict[str, Any] | None, plan_state: PWRPlanState) -> dict[str, Any]:
    parsed = parsed if isinstance(parsed, dict) else {}
    decision = str(parsed.get("decision", "silent")).strip().lower()
    if decision not in {"silent", "interrupt"}:
        decision = "silent"
    interrupt_type = str(parsed.get("interrupt_type", "none")).strip().lower()
    if interrupt_type not in {"step_complete", "deviation", "reactive", "reminder", "none"}:
        interrupt_type = "none" if decision == "silent" else "step_complete"
    completed = normalize_step_id(parsed.get("completed_step_id"), plan_state.valid)
    if decision == "interrupt" and interrupt_type == "step_complete" and completed is None:
        completed = plan_state.current_step_id
    event_class = str(parsed.get("event_class") or "none").strip()
    event_subtype = str(parsed.get("event_subtype") or "none").strip()
    if decision == "silent":
        interrupt_type = "none"
        completed = None
        event_class = "none"
        event_subtype = "none"
    elif interrupt_type == "deviation" and event_class == "none":
        event_class = "execution_error"
        event_subtype = "technique"
    return {
        "decision": decision,
        "interrupt_type": interrupt_type,
        "completed_step_id": completed,
        "event_class": event_class,
        "event_subtype": event_subtype,
        "event_id": parsed.get("event_id") or f"pwr_{interrupt_type}",
        "utterance": parsed.get("utterance") or "",
        "evidence": parsed.get("evidence") or "",
    }


def normalize_planner(parsed: dict[str, Any] | None, plan_state: PWRPlanState, duplex: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed if isinstance(parsed, dict) else {}
    completed = []
    for step_id in parsed.get("completed_step_ids") or []:
        normalized = normalize_step_id(step_id, plan_state.valid)
        if normalized is not None:
            completed.append(normalized)
    if duplex.get("completed_step_id") is not None:
        completed.append(duplex["completed_step_id"])
    seen = set()
    completed_unique = []
    for step_id in completed:
        key = sid_s(step_id)
        if key not in seen:
            seen.add(key)
            completed_unique.append(step_id)

    current = normalize_step_id(parsed.get("current_step_id"), plan_state.valid)
    if current is None:
        temp_done = plan_state.completed_set() | {sid_s(step_id) for step_id in completed_unique}
        candidates = []
        for step in plan_state.steps:
            step_id = sid_s(step["step_id"])
            if step_id in temp_done:
                continue
            preds = {sid_s(pred) for pred in step.get("preconditions", [])}
            if preds.issubset(temp_done):
                candidates.append(step)
        current = candidates[0]["step_id"] if candidates else None

    mode = str(parsed.get("mode") or "update_proactive").strip()
    if duplex.get("interrupt_type") == "deviation":
        mode = "recover"
    if mode not in {"start_proactive", "update_proactive", "recover", "end_proactive"}:
        mode = "update_proactive"
    return {
        "mode": mode,
        "completed_step_ids": completed_unique,
        "current_step_id": current,
        "recovery": parsed.get("recovery") or (duplex.get("utterance") if mode == "recover" else ""),
        "step_complete_cues": parsed.get("step_complete_cues") or [],
        "step_incomplete_cues": parsed.get("step_incomplete_cues") or [],
        "reason": parsed.get("reason") or "",
    }


def jpeg_bytes(frame: Any, max_dim: int = 768, jpeg_q: int = 85) -> bytes | None:
    height, width = frame.shape[:2]
    scale = max_dim / float(max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return buf.tobytes() if ok else None


def foreground_intervals(timeline: list[tuple[float, Any]], duration: float) -> list[dict[str, Any]]:
    timeline = sorted(timeline, key=lambda item: item[0])
    out = []
    for idx, (start, step_id) in enumerate(timeline):
        end = timeline[idx + 1][0] if idx + 1 < len(timeline) else duration
        if step_id is not None and end > start + 1e-6:
            out.append({"stage": step_id, "start_s": round(start, 2), "end_s": round(end, 2)})
    return out


def write_trace(trace_fh: Any, record: dict[str, Any]) -> None:
    if trace_fh is None:
        return
    trace_fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    trace_fh.flush()
