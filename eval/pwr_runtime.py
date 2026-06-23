#!/usr/bin/env python3
"""Plan-Watch-Recover style runtime for CC4D replay experiments.

The PWR paper does not release its trained duplex/planner checkpoints. This file
implements the system framework described in the paper:

  * a cached structured procedural plan with current/completed/next steps and
    visual cues,
  * plan-anchored clip selection for the duplex interaction call,
  * a user-facing duplex call that emits silent/interrupt,
  * a background planner call only after interrupts, using the most recent clip,
  * scorer-compatible replay output and optional JSONL traces.

The Qwen backend uses the same OpenAI-compatible video endpoint convention used
elsewhere in this repo. The mock backend exercises the control flow offline.

Examples:
  python3 eval/pwr_runtime.py --video data/videos_360p/8_16.mp4 \
    --task tasks/cc4d/spicedhotchocolate.json --backend mock \
    --tick 10 --max-seconds 80 --trace

  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
  QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python3 eval/pwr_runtime.py --video data/videos_360p/8_16.mp4 \
    --task tasks/cc4d/spicedhotchocolate.json --backend qwen --tick 10
"""

import argparse
import base64
import glob
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

BASE = Path(__file__).resolve().parent.parent
ANN = BASE / "data/cc4d/annotations/annotation_json"

DUPLEX_SYSTEM_PROMPT = (
    "You are the duplex interaction model in a Plan-Watch-Recover procedural "
    "assistant. You observe egocentric video clips, a cached structured plan, "
    "and recent dialogue. Decide whether to stay silent or interrupt now.\n\n"
    "Responsibilities:\n"
    "1. Track whether the current plan step is still in progress.\n"
    "2. Interrupt when the current step appears complete and give the next action.\n"
    "3. Interrupt when the user deviates from the expected procedure and give a "
    "brief recovery action.\n"
    "4. Interrupt when a task-specific reminder/check is visibly due.\n"
    "5. Stay silent when the user is progressing normally.\n\n"
    "Use the plan's visual cues as the main decision criteria. Base the decision "
    "on the attached frames, not on likely recipe order alone. When a listed "
    "CC4D reminder/check is the reason for an interrupt, copy its class, subtype, "
    "and id into the event fields.\n\n"
    "Return JSON only, exactly this schema:\n"
    '{"decision": "silent|interrupt", '
    '"interrupt_type": "step_complete|deviation|reactive|reminder|none", '
    '"completed_step_id": "<step id or null>", '
    '"event_class": "precondition_violation|parameter_violation|execution_error|none", '
    '"event_subtype": "order|missing_step|technique|preparation|measurement|temperature|timing|none", '
    '"event_id": "<task reminder/check id or none>", '
    '"utterance": "<assistant text, empty if silent>", '
    '"evidence": "<one short visual reason>"}'
)

PLANNER_SYSTEM_PROMPT = (
    "You are the background planner in a Plan-Watch-Recover procedural assistant. "
    "You never speak directly to the user. You receive the prior structured plan, "
    "the duplex interrupt decision and utterance, and recent egocentric frames. "
    "Update the procedural plan only as needed.\n\n"
    "If a step completed, mark it completed and choose the next current step. If "
    "there is an out-of-plan deviation, keep or revise the current step and write "
    "a concrete recovery note. Generate visual cues that are strictly visible in "
    "egocentric video.\n\n"
    "Return JSON only, exactly this schema:\n"
    '{"mode": "update_proactive|recover|end_proactive", '
    '"completed_step_ids": ["<step id>", "..."], '
    '"current_step_id": "<step id or null>", '
    '"recovery": "<recovery note or empty>", '
    '"step_complete_cues": ["<visual cue>", "..."], '
    '"step_incomplete_cues": ["<visual cue>", "..."], '
    '"reason": "<one short reason>"}'
)


def safe_json(text):
    for candidate in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def sid_s(step_id):
    if step_id is None:
        return None
    return str(step_id)


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def recipe_stem_by_recording():
    ann = json.load(open(ANN / "complete_step_annotations.json"))
    norm2stem = {}
    for f in glob.glob(str(BASE / "tasks/cc4d/*.json")):
        stem = os.path.basename(f)[:-5]
        if "." not in stem:
            norm2stem[_alnum(stem)] = stem
    out = {}
    for rid, rec in ann.items():
        stem = norm2stem.get(_alnum(rec["activity_name"]))
        if stem:
            out[rid] = stem
    return out


def task_title(task):
    nested = task.get("task") if isinstance(task.get("task"), dict) else {}
    return task.get("title") or nested.get("title") or task.get("task_id") or nested.get("task_id") or "unknown"


def task_steps(task):
    if "steps" in task:
        return sorted(task["steps"], key=lambda s: (s.get("order", 0), sid_s(s.get("step_id"))))
    if "nodes" in task:
        steps = []
        for i, node in enumerate(task.get("nodes", []), 1):
            steps.append({
                "step_id": node["step_id"],
                "order": node.get("order", i),
                "instruction": node.get("instruction", ""),
                "preconditions": node.get("preconditions", []),
                "duration_constraint_s": node.get("duration_constraint_s"),
                "sensing_role": node.get("sensing_role"),
                "checks": node.get("checks", []),
            })
        return sorted(steps, key=lambda s: (s.get("order", 0), sid_s(s.get("step_id"))))
    graph = task.get("graph", {})
    steps = []
    for s in graph.get("steps", []):
        steps.append({
            "step_id": s.get("cc4d_step_id", s["step_id"]),
            "order": s.get("order", len(steps) + 1),
            "instruction": s.get("instruction", ""),
            "preconditions": s.get("requires", []),
            "duration_constraint_s": s.get("duration_constraint_s"),
        })
    for block in graph.get("step_blocks", []):
        for m in block.get("members", []):
            steps.append({
                "step_id": m.get("cc4d_step_id", m["step_id"]),
                "order": block.get("order", len(steps) + 1),
                "instruction": m.get("instruction", ""),
                "preconditions": block.get("requires", []),
            })
    return sorted(steps, key=lambda s: (s.get("order", 0), sid_s(s.get("step_id"))))


def step_ranks(steps):
    if steps and all("rank" in s for s in steps):
        return {sid_s(s["step_id"]): int(s["rank"]) for s in steps}
    by_id = {sid_s(s["step_id"]): s for s in steps}
    if any(s.get("preconditions") for s in steps):
        memo = {}

        def rank(step_id):
            step_id = sid_s(step_id)
            if step_id in memo:
                return memo[step_id]
            s = by_id.get(step_id)
            preds = [sid_s(p) for p in (s.get("preconditions") or []) if sid_s(p) in by_id] if s else []
            memo[step_id] = 0 if not preds else 1 + max(rank(p) for p in preds)
            return memo[step_id]

        return {sid_s(s["step_id"]): rank(s["step_id"]) for s in steps}
    return {sid_s(s["step_id"]): int(s.get("order", 0)) for s in steps}


def normalize_step_id(raw, valid):
    if raw is None:
        return None
    raw_s = sid_s(raw)
    return valid.get(raw_s)


def default_cues(step):
    instr = step.get("instruction", "").strip()
    if not instr:
        instr = f"step {step.get('step_id')}"
    complete = [
        f"Visible evidence that this is finished: {instr}.",
        "Hands move away from the main object or begin preparing the next action.",
    ]
    incomplete = [
        f"The user is still doing or setting up: {instr}.",
        "The required object state for the completed action is not yet visible.",
    ]
    return complete, incomplete


def task_checks_text(task):
    rows = []
    for r in task.get("reminders", []):
        cls = r.get("class", "none")
        subtype = r.get("subtype", "none")
        rid = r.get("reminder_id", "reminder")
        rows.append(
            f"- id={rid} step_id={r.get('step_id')} class={cls} subtype={subtype}: "
            f"trigger={r.get('trigger', '')}; message={r.get('message', '')}"
        )
    for n in task.get("nodes", []):
        for chk in n.get("checks", []):
            subtype = chk.get("reminder", "none")
            cls = "parameter_violation" if subtype == "timing" else "execution_error"
            rid = f"{n.get('step_id')}_{subtype}"
            rows.append(
                f"- id={rid} step_id={n.get('step_id')} class={cls} subtype={subtype}: "
                f"detector={chk.get('detector', '')}; criteria={chk.get('detection_criteria', '')}"
            )
    return "\n".join(rows) if rows else "- none"


@dataclass
class PWRPlanState:
    task: dict
    steps: list
    completed: list = field(default_factory=list)
    current_step_id: object = None
    complete_cues: list = field(default_factory=list)
    incomplete_cues: list = field(default_factory=list)
    recovery: str = ""
    update_times: list = field(default_factory=lambda: [0.0])

    def __post_init__(self):
        self.by_id = {sid_s(s["step_id"]): s for s in self.steps}
        self.valid = {sid_s(s["step_id"]): s["step_id"] for s in self.steps}
        self.ranks = step_ranks(self.steps)
        if self.current_step_id is None:
            self.current_step_id = self.next_frontier()[0]["step_id"] if self.next_frontier() else None
        self.refresh_cues()

    def completed_set(self):
        return {sid_s(s) for s in self.completed}

    def mark_completed(self, step_id):
        step_id = normalize_step_id(step_id, self.valid)
        if step_id is None:
            return False
        if sid_s(step_id) not in self.completed_set():
            self.completed.append(step_id)
            return True
        return False

    def next_frontier(self):
        done = self.completed_set()
        out = []
        for s in self.steps:
            sid = sid_s(s["step_id"])
            if sid in done:
                continue
            preds = {sid_s(p) for p in s.get("preconditions", [])}
            if preds.issubset(done):
                out.append(s)
        return sorted(out, key=lambda s: (self.ranks.get(sid_s(s["step_id"]), 0), s.get("order", 0)))

    def remaining_steps(self):
        done = self.completed_set()
        return [s for s in self.steps if sid_s(s["step_id"]) not in done]

    def set_current(self, step_id):
        step_id = normalize_step_id(step_id, self.valid)
        if step_id is None or sid_s(step_id) in self.completed_set():
            frontier = self.next_frontier()
            self.current_step_id = frontier[0]["step_id"] if frontier else None
        else:
            self.current_step_id = step_id
        self.refresh_cues()

    def refresh_cues(self, complete=None, incomplete=None):
        step = self.by_id.get(sid_s(self.current_step_id))
        if not step:
            self.complete_cues, self.incomplete_cues = [], []
            return
        default_complete, default_incomplete = default_cues(step)
        complete = [str(x) for x in (complete or []) if str(x).strip()]
        incomplete = [str(x) for x in (incomplete or []) if str(x).strip()]
        self.complete_cues = complete or default_complete
        self.incomplete_cues = incomplete or default_incomplete

    def apply_planner_update(self, update, t):
        before = sid_s(self.current_step_id)
        for step_id in update.get("completed_step_ids") or []:
            self.mark_completed(step_id)
        if update.get("current_step_id") is not None:
            self.set_current(update.get("current_step_id"))
        else:
            self.set_current(None)
        self.recovery = update.get("recovery") or ""
        self.refresh_cues(update.get("step_complete_cues"), update.get("step_incomplete_cues"))
        changed = before != sid_s(self.current_step_id) or self.recovery
        should_record = changed or not self.update_times
        if should_record and (not self.update_times or abs(self.update_times[-1] - t) > 1e-6):
            self.update_times.append(float(t))
        return changed

    def plan_text(self):
        current_s = sid_s(self.current_step_id)
        done = self.completed_set()
        rows = []
        for s in self.steps:
            sid = sid_s(s["step_id"])
            if sid in done:
                tag = "completed"
            elif sid == current_s:
                tag = "current"
            else:
                preds = {sid_s(p) for p in s.get("preconditions", [])}
                tag = "next" if preds.issubset(done) else "remaining"
            rows.append(f"- [{tag}] step_id={s['step_id']}: {s.get('instruction', '')}")
        complete = "\n".join(f"- {c}" for c in self.complete_cues) or "- none"
        incomplete = "\n".join(f"- {c}" for c in self.incomplete_cues) or "- none"
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


class QwenBackend:
    def __init__(self, timeout=180, retries=2, backoff_s=2.0):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + ("" if base.endswith("/chat/completions")
                           else "/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, jpegs, system_prompt, user_prompt):
        content = [{
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()},
        } for j in jpegs]
        content.append({"type": "text", "text": user_prompt})
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                r.raise_for_status()
                c = r.json()["choices"][0]["message"]["content"]
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                return c
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last


class MockBackend:
    model = "mock-pwr-framework"

    def call(self, jpegs, system_prompt, user_prompt):
        return "{}"


class EgoProactiveGoldBackend:
    model = "egoproactive-gold-replay"

    def call(self, jpegs, system_prompt, user_prompt):
        return "{}"


def jpeg_bytes(frame, max_dim=768, jpeg_q=85):
    h, w = frame.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return buf.tobytes() if ok else None


def sample_clip_frames(cap, fps, end_t, window_s, n_frames, max_dim=768):
    start = max(0.0, float(end_t) - float(window_s))
    if n_frames <= 1:
        times = [float(end_t)]
    else:
        times = np.linspace(start, max(start, float(end_t)), n_frames)
    # Avoid sending exact duplicate frames for clips that collapse near t=0.
    deduped = []
    seen = set()
    for ts in times:
        key = round(float(ts), 3)
        if key not in seen:
            seen.add(key)
            deduped.append(float(ts))

    jpegs, used_times = [], []
    for ts in deduped:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(ts * fps))))
        ok, frame = cap.read()
        if not ok:
            continue
        jb = jpeg_bytes(frame, max_dim=max_dim)
        if jb is not None:
            jpegs.append(jb)
            used_times.append(round(ts, 3))
    return jpegs, used_times


def plan_anchored_clips(cap, fps, t, plan_state, args, planner_only=False):
    if planner_only:
        anchors = [("recent", float(t))]
    else:
        anchors = [("recent", float(t))]
        prior = [u for u in plan_state.update_times if u < t - 1e-6]
        for u in sorted(prior, reverse=True)[: max(0, args.max_clips - 1)]:
            anchors.append((f"plan_update@{u:.1f}", float(u)))

    all_jpegs, meta = [], []
    for label, end_t in anchors[: args.max_clips]:
        jpegs, times = sample_clip_frames(
            cap, fps, end_t, args.clip_window_s, args.frames_per_clip, max_dim=args.max_dim)
        if not jpegs:
            continue
        idx0 = len(all_jpegs)
        all_jpegs.extend(jpegs)
        meta.append({
            "label": label,
            "clip_end_s": round(end_t, 3),
            "frame_times_s": times,
            "image_range": [idx0, len(all_jpegs) - 1],
        })
    return all_jpegs, meta


def build_duplex_prompt(plan_state, clip_meta, dialogue):
    clip_rows = "\n".join(
        f"- images {m['image_range'][0]}..{m['image_range'][1]}: {m['label']}, "
        f"clip_end={m['clip_end_s']}s, frame_times={m['frame_times_s']}"
        for m in clip_meta
    ) or "- no readable frames"
    recent = "\n".join(
        f"- t={h['t_s']:.1f}s assistant={h['assistant']} evidence={h.get('evidence', '')}"
        for h in dialogue[-5:]
    ) or "- none"
    return (
        plan_state.plan_text()
        + "\n\nTask-specific reminder/check context:\n"
        + task_checks_text(plan_state.task)
        + "\n\nPlan-anchored visual context:\n"
        + clip_rows
        + "\n\nRecent dialogue/interventions:\n"
        + recent
        + "\n\nDecide whether to output silent or interrupt now."
    )


def build_planner_prompt(plan_state, duplex, clip_meta):
    clip_rows = "\n".join(
        f"- images {m['image_range'][0]}..{m['image_range'][1]}: {m['label']}, "
        f"clip_end={m['clip_end_s']}s, frame_times={m['frame_times_s']}"
        for m in clip_meta
    ) or "- no readable frames"
    return (
        plan_state.plan_text()
        + "\n\nDuplex output:\n"
        + json.dumps(duplex, ensure_ascii=True)
        + "\n\nRecent visual context for planner:\n"
        + clip_rows
        + "\n\nUpdate the cached plan."
    )


def normalize_duplex(parsed, plan_state):
    parsed = parsed if isinstance(parsed, dict) else {}
    decision = str(parsed.get("decision", "silent")).strip().lower()
    if decision not in {"silent", "interrupt"}:
        decision = "silent"
    itype = str(parsed.get("interrupt_type", "none")).strip().lower()
    if itype not in {"step_complete", "deviation", "reactive", "reminder", "none"}:
        itype = "none" if decision == "silent" else "step_complete"
    completed = normalize_step_id(parsed.get("completed_step_id"), plan_state.valid)
    if decision == "interrupt" and itype == "step_complete" and completed is None:
        completed = plan_state.current_step_id
    event_class = str(parsed.get("event_class") or "none").strip()
    event_subtype = str(parsed.get("event_subtype") or "none").strip()
    if itype == "deviation" and event_class == "none":
        event_class, event_subtype = "execution_error", "technique"
    if itype == "step_complete" and event_class == "none":
        event_subtype = "none"
    return {
        "decision": decision,
        "interrupt_type": itype,
        "completed_step_id": completed,
        "event_class": event_class,
        "event_subtype": event_subtype,
        "event_id": parsed.get("event_id") or f"pwr_{itype}",
        "utterance": parsed.get("utterance") or "",
        "evidence": parsed.get("evidence") or "",
    }


def normalize_planner(parsed, plan_state, duplex):
    parsed = parsed if isinstance(parsed, dict) else {}
    completed = []
    for sid in parsed.get("completed_step_ids") or []:
        norm = normalize_step_id(sid, plan_state.valid)
        if norm is not None:
            completed.append(norm)
    if duplex.get("completed_step_id") is not None:
        completed.append(duplex["completed_step_id"])
    # Preserve order while deduplicating.
    seen, completed2 = set(), []
    for sid in completed:
        if sid_s(sid) not in seen:
            seen.add(sid_s(sid))
            completed2.append(sid)

    current = normalize_step_id(parsed.get("current_step_id"), plan_state.valid)
    if current is None:
        temp_done = plan_state.completed_set() | {sid_s(s) for s in completed2}
        candidates = []
        for s in plan_state.steps:
            sid = sid_s(s["step_id"])
            if sid in temp_done:
                continue
            preds = {sid_s(p) for p in s.get("preconditions", [])}
            if preds.issubset(temp_done):
                candidates.append(s)
        current = candidates[0]["step_id"] if candidates else None

    mode = str(parsed.get("mode") or "update_proactive").strip()
    if duplex.get("interrupt_type") == "deviation":
        mode = "recover"
    if mode not in {"update_proactive", "recover", "end_proactive"}:
        mode = "update_proactive"

    return {
        "mode": mode,
        "completed_step_ids": completed2,
        "current_step_id": current,
        "recovery": parsed.get("recovery") or (duplex.get("utterance") if mode == "recover" else ""),
        "step_complete_cues": parsed.get("step_complete_cues") or [],
        "step_incomplete_cues": parsed.get("step_incomplete_cues") or [],
        "reason": parsed.get("reason") or "",
    }


class PWRCaller:
    def __init__(self, backend, mock_step_s=20.0):
        self.backend = backend
        self.mock_step_s = mock_step_s

    @property
    def model(self):
        return self.backend.model

    def duplex(self, jpegs, prompt, plan_state, t):
        if isinstance(self.backend, EgoProactiveGoldBackend):
            point = None
            for cand in plan_state.task.get("egoproactive_decision_points", []):
                start, end = cand.get("interval_s", [None, None])
                if start is None or end is None:
                    continue
                if float(start) <= float(t) < float(end):
                    point = cand
                    break
            if point is None:
                raw_obj = {
                    "decision": "silent",
                    "interrupt_type": "none",
                    "completed_step_id": None,
                    "utterance": "",
                    "evidence": "no EgoProactive decision interval covers this tick",
                }
            else:
                decision = point.get("decision", "silent")
                answer = point.get("answer", "")
                utterance = re.sub(r"^\$(?:interrupt|silent)\$", "", answer).strip()
                raw_obj = {
                    "decision": decision if decision in {"interrupt", "silent"} else "silent",
                    "interrupt_type": "step_complete" if decision == "interrupt" else "none",
                    "completed_step_id": point.get("step_id") if decision == "interrupt" else None,
                    "event_class": "none",
                    "event_subtype": "none",
                    "event_id": f"egoproactive_{point.get('index', 'unknown')}",
                    "utterance": utterance if decision == "interrupt" else "",
                    "evidence": f"EgoProactive gold interval {point.get('index')} {point.get('interval_s')}",
                }
            return raw_obj, json.dumps(raw_obj), None, 0.0
        if isinstance(self.backend, MockBackend):
            last_update = plan_state.update_times[-1] if plan_state.update_times else 0.0
            should_interrupt = plan_state.current_step_id is not None and (t - last_update) >= self.mock_step_s
            if should_interrupt:
                raw_obj = {
                    "decision": "interrupt",
                    "interrupt_type": "step_complete",
                    "completed_step_id": plan_state.current_step_id,
                    "utterance": "Move to the next step.",
                    "evidence": "mock elapsed-step completion",
                }
            else:
                raw_obj = {
                    "decision": "silent",
                    "interrupt_type": "none",
                    "completed_step_id": None,
                    "utterance": "",
                    "evidence": "mock normal progress",
                }
            return raw_obj, json.dumps(raw_obj), None, 0.0
        t0 = time.time()
        try:
            raw, err = self.backend.call(jpegs, DUPLEX_SYSTEM_PROMPT, prompt), None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        latency = time.time() - t0
        return normalize_duplex(safe_json(raw), plan_state), raw, err, latency

    def planner(self, jpegs, prompt, plan_state, duplex):
        if isinstance(self.backend, MockBackend):
            temp = dict(duplex)
            update = normalize_planner({}, plan_state, temp)
            raw = json.dumps(update)
            return update, raw, None, 0.0
        t0 = time.time()
        try:
            raw, err = self.backend.call(jpegs, PLANNER_SYSTEM_PROMPT, prompt), None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        latency = time.time() - t0
        return normalize_planner(safe_json(raw), plan_state, duplex), raw, err, latency


def write_trace(trace_fh, rec):
    if trace_fh is None:
        return
    trace_fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
    trace_fh.flush()


def foreground_intervals(timeline, duration):
    timeline = sorted(timeline, key=lambda x: x[0])
    out = []
    for i, (t, sid) in enumerate(timeline):
        end = timeline[i + 1][0] if i + 1 < len(timeline) else duration
        if sid is not None and end > t + 1e-6:
            out.append({"stage": sid, "start_s": round(t, 2), "end_s": round(end, 2)})
    return out


def pwr_manifest(args):
    return {
        "arm": args.arm,
        "claim": "PWR-style framework implementation; not an exact reproduction of unreleased trained checkpoints.",
        "paper_mapping": [
            "duplex interaction model receives cached plan and plan-anchored visual clips",
            "planner is invoked only when duplex emits interrupt",
            "planner receives only the most recent visual clip plus cached plan",
            "plan state stores completed/current/remaining steps and visual cues",
        ],
        "approximations": [
            "zero-shot Qwen or mock backend replaces the unreleased trained PWR models",
            "visual cues are heuristic unless the planner backend refreshes them",
            "CC4D replay has no live user utterances, so reactive turns are not simulated",
            "tick can be set to 0.5s for the paper's 2 fps decision cadence, but sparse ticks are practical for Qwen replay",
        ],
        "params": {
            "tick_s": args.tick,
            "clip_window_s": args.clip_window_s,
            "frames_per_clip": args.frames_per_clip,
            "max_clips": args.max_clips,
        },
    }


def run_one(video, task, caller, args, rid=None, trace_dir=None, verbose=True):
    steps = task_steps(task)
    plan_state = PWRPlanState(task=task, steps=steps)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    raw_video_duration = duration
    if isinstance(caller.backend, EgoProactiveGoldBackend) and task.get("duration_in_sec"):
        duration = float(task["duration_in_sec"])
        fps = frame_count / duration if duration > 0 else fps
    end_t = min(duration, args.max_seconds) if args.max_seconds else duration

    trace_fh = None
    if trace_dir is not None and rid is not None:
        (Path(trace_dir) / "pwr_debug").mkdir(parents=True, exist_ok=True)
        trace_fh = open(Path(trace_dir) / "pwr_debug" / f"{rid}.jsonl", "w")

    timeline = [(0.0, plan_state.current_step_id)]
    interventions, events, dialogue = [], [], []
    duplex_calls = planner_calls = frames_sent = parse_failures = 0
    duplex_latency = planner_latency = 0.0

    tick_times = None
    tick_i = 0
    if isinstance(caller.backend, EgoProactiveGoldBackend) and task.get("egoproactive_decision_points"):
        tick_times = [float(dp["interval_s"][0]) for dp in task["egoproactive_decision_points"]
                      if float(dp["interval_s"][0]) <= end_t + 1e-6]
    t = tick_times[0] if tick_times else 0.0
    while t <= end_t + 1e-6:
        jpegs, clip_meta = plan_anchored_clips(cap, fps, t, plan_state, args, planner_only=False)
        if not jpegs:
            break
        frames_sent += len(jpegs)
        duplex_calls += 1
        prompt = build_duplex_prompt(plan_state, clip_meta, dialogue)
        duplex, raw, err, lat = caller.duplex(jpegs, prompt, plan_state, t)
        duplex_latency += lat
        if raw and safe_json(raw) is None and not isinstance(caller.backend, MockBackend):
            parse_failures += 1

        trace_rec = {
            "kind": "duplex",
            "t_s": round(t, 3),
            "plan_current": plan_state.current_step_id,
            "clip_meta": clip_meta,
            "n_images": len(jpegs),
            "system_prompt": DUPLEX_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "raw": raw,
            "normalized": duplex,
            "error": err,
            "latency_s": round(lat, 2),
        }
        write_trace(trace_fh, trace_rec)

        if duplex["decision"] == "interrupt":
            interventions.append({
                "t_s": round(t, 2),
                "type": duplex["interrupt_type"],
                "step_id": duplex.get("completed_step_id") or plan_state.current_step_id,
                "event_class": duplex.get("event_class", "none"),
                "event_subtype": duplex.get("event_subtype", "none"),
                "utterance": duplex.get("utterance", ""),
                "evidence": duplex.get("evidence", ""),
            })
            events.append({
                "t": round(t, 2),
                "class": duplex.get("event_class", "none"),
                "subtype": duplex.get("event_subtype", "none"),
                "id": duplex.get("event_id") or f"pwr_{duplex['interrupt_type']}",
                "message": duplex.get("utterance", ""),
            })
            dialogue.append({
                "t_s": float(t),
                "assistant": duplex.get("utterance", ""),
                "evidence": duplex.get("evidence", ""),
            })

            pjpegs, pmeta = plan_anchored_clips(cap, fps, t, plan_state, args, planner_only=True)
            frames_sent += len(pjpegs)
            planner_calls += 1
            pprompt = build_planner_prompt(plan_state, duplex, pmeta)
            update, praw, perr, plat = caller.planner(pjpegs, pprompt, plan_state, duplex)
            planner_latency += plat
            if praw and safe_json(praw) is None and not isinstance(caller.backend, MockBackend):
                parse_failures += 1
            old_current = plan_state.current_step_id
            plan_state.apply_planner_update(update, t)
            if sid_s(old_current) != sid_s(plan_state.current_step_id):
                timeline.append((float(t), plan_state.current_step_id))
            if (update.get("mode") == "recover" or duplex.get("interrupt_type") == "deviation") and events:
                events[-1]["message"] = update.get("recovery") or events[-1].get("message", "")

            write_trace(trace_fh, {
                "kind": "planner",
                "t_s": round(t, 3),
                "old_current": old_current,
                "new_current": plan_state.current_step_id,
                "clip_meta": pmeta,
                "n_images": len(pjpegs),
                "system_prompt": PLANNER_SYSTEM_PROMPT,
                "user_prompt": pprompt,
                "raw": praw,
                "normalized": update,
                "error": perr,
                "latency_s": round(plat, 2),
            })

        if verbose:
            print(
                f"  t={t:6.1f}s decision={duplex['decision']} "
                f"current={plan_state.current_step_id} planners={planner_calls}"
            )
        if tick_times is not None:
            tick_i += 1
            if tick_i >= len(tick_times):
                break
            t = tick_times[tick_i]
        else:
            t += args.tick

    cap.release()
    if trace_fh is not None:
        trace_fh.close()

    return {
        "stage_intervals": foreground_intervals(timeline, end_t),
        "events": events,
        "interventions": interventions,
        "escalation_requests": [],
        "cost": {
            "vlm_calls": duplex_calls + planner_calls,
            "duplex_calls": duplex_calls,
            "planner_calls": planner_calls,
            "frames_sent": frames_sent,
            "vlm_latency_total_s": round(duplex_latency + planner_latency, 2),
            "duplex_latency_s": round(duplex_latency, 2),
            "planner_latency_s": round(planner_latency, 2),
            "compute_s": 0.0,
        },
        "replication_manifest": pwr_manifest(args),
        "_meta": {
            "model": caller.model,
            "evaluated_s": round(end_t, 2),
            "video_duration_s": round(duration, 2),
            "raw_video_duration_s": round(raw_video_duration, 2),
            "parse_failures": parse_failures,
            "task_id": task.get("task_id") or (task.get("task") or {}).get("task_id"),
            "final_plan": plan_state.plan_text(),
            "plan_update_times": [round(x, 2) for x in plan_state.update_times],
            "trace": trace_dir is not None,
        },
    }


def make_backend(args):
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "egoproactive_gold":
        return EgoProactiveGoldBackend()
    return QwenBackend(timeout=args.timeout, retries=args.retries)


def output_dir(args):
    outd = BASE / args.out_dir / args.arm
    outd.mkdir(parents=True, exist_ok=True)
    return outd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--task")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--video-dir", default="data/videos_360p")
    ap.add_argument("--out-dir", default="experiments/pwr_runtime")
    ap.add_argument("--arm", default="pwr_framework")
    ap.add_argument("--backend", choices=["mock", "qwen", "egoproactive_gold"], default="mock")
    ap.add_argument("--tick", type=float, default=0.5,
                    help="duplex decision period in seconds; paper uses 0.5s (2 fps)")
    ap.add_argument("--clip-window-s", type=float, default=8.0)
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=15)
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--mock-step-s", type=float, default=20.0)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    if args.tick <= 0:
        raise SystemExit("--tick must be positive")
    if args.frames_per_clip <= 0:
        raise SystemExit("--frames-per-clip must be positive")
    if args.max_clips <= 0:
        raise SystemExit("--max-clips must be positive")

    backend = make_backend(args)
    caller = PWRCaller(backend, mock_step_s=args.mock_step_s)
    outd = output_dir(args)
    trace_dir = outd if args.trace else None

    if args.corpus:
        rmap = recipe_stem_by_recording()
        todo = [(rid, stem) for rid, stem in sorted(rmap.items())
                if (BASE / args.video_dir / f"{rid}.mp4").exists()]
        print(f"corpus: {len(todo)} recordings with task+video")
        for i, (rid, stem) in enumerate(todo, 1):
            outp = outd / f"{rid}.json"
            if outp.exists():
                continue
            task = json.loads((BASE / "tasks/cc4d" / f"{stem}.json").read_text())
            print(f"[{i}/{len(todo)}] {rid} ({stem})")
            res = run_one(BASE / args.video_dir / f"{rid}.mp4", task, caller, args,
                          rid=rid, trace_dir=trace_dir, verbose=False)
            res["recording"], res["arm"] = rid, args.arm
            outp.write_text(json.dumps(res, indent=1))
    else:
        if not (args.video and args.task):
            raise SystemExit("single mode needs --video and --task (or use --corpus)")
        rid = Path(args.video).stem
        task = json.loads(Path(args.task).read_text())
        res = run_one(args.video, task, caller, args, rid=rid,
                      trace_dir=trace_dir, verbose=True)
        res["recording"], res["arm"] = rid, args.arm
        outp = outd / f"{rid}.json"
        outp.write_text(json.dumps(res, indent=1))
        print(
            f"wrote {outp} ({res['cost']['duplex_calls']} duplex calls, "
            f"{res['cost']['planner_calls']} planner calls, "
            f"{res['_meta']['parse_failures']} parse-fails)"
        )


if __name__ == "__main__":
    main()
