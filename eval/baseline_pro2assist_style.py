#!/usr/bin/env python3
"""Pro2Assist-style CC4D replication baseline.

This is a prior-work-style arm, not the proposed detector-graph system. It
approximates the Pro2Assist architecture with the signals available in CC4D:
RGB video, CC4D task graphs, reminders, and causal history. Hardware-specific
pieces such as head IMU and hand-motion sensors are simulated with video-only
proxies, and every run records that provenance in `replication_manifest`.

Writes scorer-compatible records:
  <out-dir>/<arm>/<rid>.json
    {stage_intervals, events, escalation_requests, cost,
     replication_manifest, _meta}

Examples:
  python eval/baseline_pro2assist_style.py \
    --video data/videos_360p/8_16.mp4 \
    --task tasks/cc4d/spicedhotchocolate.json \
    --backend mock --max-seconds 60 --trace

  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
  QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_pro2assist_style.py --corpus --backend qwen
"""

import argparse
import base64
import glob
import json
import math
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import requests

BASE = Path(__file__).resolve().parent.parent
ANN = BASE / "data/cc4d/annotations/annotation_json"

SCORED_KEYS = {
    ("precondition_violation", "order"),
    ("precondition_violation", "missing_step"),
    ("execution_error", "technique"),
    ("execution_error", "preparation"),
    ("execution_error", "measurement"),
    ("execution_error", "temperature"),
    ("parameter_violation", "timing"),
}

STATUSES = {"Just start", "In progress", "About to finish", "Step transition"}
ROI_DEFAULT = (0.30, 0.72, 0.34, 0.66)  # lower-center mug/work-surface proxy

PROFILES = {
    "qwen_replay": {
        "period_s": 10.0,
        "window_s": 10.0,
        "candidate_interval_s": 1.0,
        "max_frames": 10,
        "max_gap_s": 10.0,
        "consistency_w": 3,
        "consistency_tau": 0.5,
        "cooldown_s": 15.0,
        "max_dim": 768,
        "feature_width": 320,
        "roi_frac": ROI_DEFAULT,
        "ego_motion_threshold_px": 1.5,
        "frame_diff_threshold": 8.0,
        "roi_motion_threshold_px": 0.8,
        "hsv_change_threshold": 0.16,
    },
    "faithful_sim": {
        "period_s": 1.0,
        "window_s": 10.0,
        "candidate_interval_s": 0.5,
        "max_frames": 10,
        "max_gap_s": 1.0,
        "consistency_w": 6,
        "consistency_tau": 0.5,
        "cooldown_s": 15.0,
        "max_dim": 768,
        "feature_width": 320,
        "roi_frac": ROI_DEFAULT,
        "ego_motion_threshold_px": 1.5,
        "frame_diff_threshold": 8.0,
        "roi_motion_threshold_px": 0.8,
        "hsv_change_threshold": 0.16,
    },
}

SYSTEM_PROMPT = (
    "You are a Pro2Assist-style proactive assistant for procedural cooking tasks. "
    "Track the user's current step and execution status from egocentric RGB frames, "
    "task-graph context, completed-step history, and simulated motion cues. Decide "
    "whether proactive assistance is needed now. Stay silent unless one of the listed "
    "scored reminder classes is clearly supported by the current evidence.\n\n"
    "Return JSON only, exactly this schema, no prose:\n"
    '{"step_id": "<one listed step_id or other>", '
    '"status": "Just start|In progress|About to finish|Step transition", '
    '"proactive_trigger": true, '
    '"class": "precondition_violation|execution_error|parameter_violation|none", '
    '"subtype": "order|missing_step|technique|preparation|measurement|temperature|timing|none", '
    '"response": "<assistant text or empty>", '
    '"evidence": "<one short sentence>"}'
)

PRO2ASSIST_SYSTEM_PROMPT = (
    "You are a Pro^2Assist-style proactive procedural assistant. "
    "Follow the guideline, historical context, sensory context, and hand-motion cues "
    "to infer the user's current procedural action and progress status, then decide "
    "whether a proactive assistance response is required. Return JSON only; no prose. "
    "Use one of the listed step identifiers when possible, or other when the action is outside the guideline.\n\n"
    "Return JSON exactly in this schema:\n"
    '{"current_action": "<one listed step_id or other>", '
    '"progress_status": "Just start|In progress|About to finish|Step transition", '
    '"proactive_trigger": true, '
    '"response": "<assistant text or empty>", '
    '"evidence": "<short visual/context evidence>"}'
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


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def recipe_stem_by_recording():
    ann = json.load(open(ANN / "complete_step_annotations.json"))
    norm2stem = {}
    for f in glob.glob(str(BASE / "tasks/cc4d/*.json")):
        stem = os.path.basename(f)[:-5]
        if "." in stem:
            continue
        norm2stem[_alnum(stem)] = stem
    out = {}
    for rid, rec in ann.items():
        stem = norm2stem.get(_alnum(rec["activity_name"]))
        if stem:
            out[rid] = stem
    return out


def task_steps(task):
    if "steps" in task:
        return sorted(task["steps"], key=lambda s: s.get("order", 0))
    graph = task.get("graph", {})
    steps = []
    for s in graph.get("steps", []):
        steps.append({
            "step_id": s["cc4d_step_id"] if "cc4d_step_id" in s else s["step_id"],
            "order": s.get("order", len(steps) + 1),
            "instruction": s.get("instruction", ""),
            "preconditions": s.get("requires", []),
            "duration_constraint_s": s.get("duration_constraint_s"),
        })
    for block in graph.get("step_blocks", []):
        for m in block.get("members", []):
            steps.append({
                "step_id": m["cc4d_step_id"] if "cc4d_step_id" in m else m["step_id"],
                "order": block.get("order", len(steps) + 1),
                "instruction": m.get("instruction", ""),
                "preconditions": block.get("requires", []),
            })
    return sorted(steps, key=lambda s: (s.get("order", 0), str(s["step_id"])))


def sid_s(step_id):
    return str(step_id)


def canonical_maps(steps):
    valid = {sid_s(s["step_id"]): s["step_id"] for s in steps}
    by_id = {sid_s(s["step_id"]): s for s in steps}
    return valid, by_id


def frontier_steps(steps, completed, active):
    done = {sid_s(s) for s in completed}
    out = []
    for s in steps:
        sid = sid_s(s["step_id"])
        if sid in done:
            continue
        pres = {sid_s(p) for p in s.get("preconditions", [])}
        if pres.issubset(done) or (active is not None and sid == sid_s(active)):
            out.append(s)
    return out[:8] or [s for s in steps if sid_s(s["step_id"]) not in done][:8]


def reminders_for_prompt(task):
    rows = []
    for r in task.get("reminders", []):
        cls = r.get("class")
        subtype = r.get("subtype")
        if (cls, subtype) not in SCORED_KEYS:
            continue
        rows.append(
            f"  - {cls}/{subtype} for step_id={r.get('step_id')}: "
            f"{r.get('trigger', '')} -> {r.get('message', '')}"
        )
    return "\n".join(rows) if rows else "  (none listed for this task)"


def build_user_prompt(task, steps, completed, active, recent, motion_cue, io_style="cc4d_scored"):
    step_rows = []
    for s in steps:
        pres = ",".join(sid_s(p) for p in s.get("preconditions", [])) or "none"
        dur = f", duration_s={s['duration_constraint_s']}" if s.get("duration_constraint_s") else ""
        step_rows.append(
            f"  order={s.get('order')} step_id={s['step_id']} preconditions={pres}{dur}: "
            f"{s.get('instruction', '')}"
        )
    completed_txt = ", ".join(sid_s(s) for s in completed) or "(none)"
    frontier = frontier_steps(steps, completed, active)
    frontier_txt = "\n".join(
        f"  step_id={s['step_id']}: {s.get('instruction', '')}" for s in frontier
    )
    recent_txt = "\n".join(
        f"  t={h['t']:.1f}s step_id={h['step_id']} status={h['status']} "
        f"trigger={h['class']}/{h['subtype']} evidence={h['evidence']}"
        for h in recent[-5:]
    ) or "  (none)"
    active_txt = sid_s(active) if active is not None else "(none yet)"

    if io_style == "pro2assist":
        guideline_txt = "\n".join(step_rows)
        return (
            f"Guideline: Structured guideline / expert procedural knowledge for "
            f"{task.get('title', task.get('task_id', 'unknown'))}:\n"
            f"{guideline_txt}\n\n"
            f"Historical Context: Consistency-verified completed actions are {completed_txt}. "
            f"Current action belief is {active_txt}. Recent action/status predictions:\n{recent_txt}\n\n"
            f"Candidate Next Actions: These are graph-frontier actions whose prerequisites are satisfied or currently active:\n"
            f"{frontier_txt}\n\n"
            f"Sensory Context: The attached egocentric RGB images are a causal observation window ending now. "
            f"Infer only from these images and the context above; do not use future steps as observations.\n\n"
            f"Hand Motion Cues: {motion_cue}\n\n"
            "Output the current_action, progress_status, proactive_trigger, response, and evidence. "
            "Only set proactive_trigger=true when the response should be shown now; otherwise use an empty response."
        )

    return (
        f"Task: {task.get('title', task.get('task_id', 'unknown'))}\n"
        f"Structured CC4D task graph / expert knowledge:\n" + "\n".join(step_rows) + "\n\n"
        f"Completed steps verified by consistency checking: {completed_txt}\n"
        f"Current active step belief: {active_txt}\n"
        f"Current graph frontier candidates:\n{frontier_txt}\n\n"
        f"Scored proactive reminder classes allowed now:\n{reminders_for_prompt(task)}\n\n"
        f"Recent reasoner outputs:\n{recent_txt}\n\n"
        f"Simulated motion cues from causal RGB window:\n{motion_cue}\n\n"
        "Use the images as the sensory context ending now. Pick one step_id, decide "
        "status, and trigger only if a scored reminder is clearly due."
    )
def resize_for_feature(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(round(h * scale))))


def jpeg_bytes(frame, max_dim=768, jpeg_q=85):
    h, w = frame.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return buf.tobytes() if ok else None


def crop_frac(frame, roi_frac):
    r0, r1, c0, c1 = roi_frac
    h, w = frame.shape[:2]
    return frame[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def hsv_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def hist_chi2(a, b):
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def skin_frac(frame):
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    mask = (y > 80) & (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
    return float(mask.mean())


def direction_name(dx, dy, eps=0.12):
    mag = math.hypot(float(dx), float(dy))
    if mag < eps:
        return "stationary"
    horiz = "right" if dx > eps else "left" if dx < -eps else ""
    vert = "down" if dy > eps else "up" if dy < -eps else ""
    if horiz and vert:
        return f"{vert}-{horiz}"
    return horiz or vert or "stationary"


def describe_level(x, low, high):
    if x >= high:
        return "high"
    if x >= low:
        return "moderate"
    return "low"


def finite_float(x, default=0.0):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_frame(cap, fps, ts):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(ts * fps))))
    ok, frame = cap.read()
    return frame if ok else None


def observation_window(cap, fps, t_end, profile):
    t0 = time.perf_counter()
    window_s = profile["window_s"]
    dt = profile["candidate_interval_s"]
    start = max(0.0, t_end - window_s)
    times = list(np.arange(start, t_end + 1e-6, dt))
    if not times or times[-1] < t_end - 1e-6:
        times.append(t_end)

    frames = []
    for ts in times:
        frame = read_frame(cap, fps, ts)
        if frame is not None:
            frames.append((float(ts), frame))
    if not frames:
        return [], [], "No frames were readable in the causal window.", {
            "window_start_s": start, "window_end_s": t_end, "selected": [],
            "features_finite": True, "compute_s": time.perf_counter() - t0,
        }

    feats = []
    selected = set()
    feature_width = int(profile["feature_width"])
    roi_frac = tuple(profile["roi_frac"])
    prev_gray = prev_roi = prev_roi_hist = None

    for idx, (ts, frame) in enumerate(frames):
        small = resize_for_feature(frame, feature_width)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        roi = crop_frac(small, roi_frac)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hist = hsv_hist(roi)
        entry = {
            "idx": idx,
            "t_s": round(ts, 3),
            "ego_motion_px": 0.0,
            "frame_diff": 0.0,
            "roi_motion_px": 0.0,
            "roi_dx": 0.0,
            "roi_dy": 0.0,
            "hsv_change": 0.0,
            "skin_frac": finite_float(skin_frac(roi)),
            "score": 0.0,
            "reasons": [],
        }
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            med = np.median(flow.reshape(-1, 2), axis=0)
            ego_mag = float(np.hypot(med[0], med[1]))
            diff = float(cv2.absdiff(prev_gray, gray).mean())

            roi_flow = cv2.calcOpticalFlowFarneback(
                prev_roi, roi_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            roi_res = roi_flow - med
            roi_dx = float(np.median(roi_res[..., 0]))
            roi_dy = float(np.median(roi_res[..., 1]))
            roi_mag = float(np.sqrt(roi_res[..., 0] ** 2 + roi_res[..., 1] ** 2).mean())
            hsv_delta = hist_chi2(prev_roi_hist, roi_hist)

            entry.update({
                "ego_motion_px": finite_float(ego_mag),
                "frame_diff": finite_float(diff),
                "roi_motion_px": finite_float(roi_mag),
                "roi_dx": finite_float(roi_dx),
                "roi_dy": finite_float(roi_dy),
                "hsv_change": finite_float(hsv_delta),
            })
            if ego_mag >= profile["ego_motion_threshold_px"]:
                entry["reasons"].append("simulated_head_imu_high")
            if diff >= profile["frame_diff_threshold"]:
                entry["reasons"].append("frame_difference")
            if roi_mag >= profile["roi_motion_threshold_px"]:
                entry["reasons"].append("roi_hand_motion_proxy")
            if hsv_delta >= profile["hsv_change_threshold"]:
                entry["reasons"].append("roi_state_change")
            score_parts = [
                ego_mag / max(profile["ego_motion_threshold_px"], 1e-6),
                diff / max(profile["frame_diff_threshold"], 1e-6),
                roi_mag / max(profile["roi_motion_threshold_px"], 1e-6),
                hsv_delta / max(profile["hsv_change_threshold"], 1e-6),
            ]
            entry["score"] = finite_float(max(score_parts))
            if entry["reasons"]:
                selected.add(idx)
        feats.append(entry)
        prev_gray, prev_roi, prev_roi_hist = gray, roi_gray, roi_hist

    latest_idx = len(frames) - 1
    if not selected:
        selected.add(latest_idx)
        feats[latest_idx]["reasons"].append("fallback_no_key_moment")
    elif frames[latest_idx][0] - frames[max(selected)][0] >= profile["max_gap_s"]:
        selected.add(latest_idx)
        feats[latest_idx]["reasons"].append("max_gap_fallback")
    else:
        selected.add(latest_idx)
        feats[latest_idx]["reasons"].append("latest_context")

    max_frames = int(profile["max_frames"])
    if len(selected) > max_frames:
        keep = {latest_idx}
        ranked = sorted(selected, key=lambda i: feats[i]["score"], reverse=True)
        for i in ranked:
            keep.add(i)
            if len(keep) >= max_frames:
                break
        selected = keep

    selected_infos, jpegs = [], []
    for i in sorted(selected):
        ts, frame = frames[i]
        jb = jpeg_bytes(frame, max_dim=profile["max_dim"])
        if jb is None:
            continue
        jpegs.append(jb)
        info = dict(feats[i])
        info["image_bytes"] = len(jb)
        selected_infos.append(info)

    if feats:
        best = max(feats, key=lambda x: x["score"])
        ego_max = max(f["ego_motion_px"] for f in feats)
        roi_max = max(f["roi_motion_px"] for f in feats)
        hsv_max = max(f["hsv_change"] for f in feats)
        skin_max = max(f["skin_frac"] for f in feats)
        cue = (
            f"Simulated head-motion proxy is "
            f"{describe_level(ego_max, profile['ego_motion_threshold_px'] * 0.5, profile['ego_motion_threshold_px'])} "
            f"(max median global flow {ego_max:.2f}px). "
            f"ROI hand-motion proxy is "
            f"{describe_level(roi_max, profile['roi_motion_threshold_px'] * 0.5, profile['roi_motion_threshold_px'])} "
            f"(max residual ROI flow {roi_max:.2f}px), dominant direction "
            f"{direction_name(best['roi_dx'], best['roi_dy'])}. "
            f"ROI state-change proxy is "
            f"{describe_level(hsv_max, profile['hsv_change_threshold'] * 0.5, profile['hsv_change_threshold'])} "
            f"(HSV chi-square {hsv_max:.2f}); skin-motion diagnostic max skin fraction {skin_max:.2f}."
        )
    else:
        cue = "No motion features were available."

    meta = {
        "window_start_s": round(start, 3),
        "window_end_s": round(t_end, 3),
        "candidate_times_s": [round(ts, 3) for ts, _ in frames],
        "selected": selected_infos,
        "features_finite": all(
            math.isfinite(finite_float(v))
            for f in feats
            for k, v in f.items()
            if k not in {"idx", "reasons"}
        ),
        "compute_s": round(time.perf_counter() - t0, 4),
    }
    return jpegs, selected_infos, cue, meta


class QwenBackend:
    def __init__(self, timeout=120, retries=2, backoff_s=2.0):
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

    def call(self, jpegs, user_prompt, call_ctx=None):
        system_prompt = (call_ctx or {}).get("system_prompt", SYSTEM_PROMPT)
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
    def __init__(self, steps, parse_fail_every=0):
        self.steps = steps
        self.calls = 0
        self.model = "mock-pro2assist-style"
        self.parse_fail_every = parse_fail_every

    def call(self, jpegs, user_prompt, call_ctx=None):
        self.calls += 1
        if self.parse_fail_every and self.calls % self.parse_fail_every == 0:
            return "MOCK_PARSE_FAILURE"
        idx = min((self.calls - 1) // 2, max(0, len(self.steps) - 1))
        step = self.steps[idx]["step_id"]
        trigger = False
        cls = subtype = "none"
        response = ""
        # Repeated timing trigger for duplicate-suppression smoke coverage.
        if self.steps[idx].get("duration_constraint_s") and self.calls in (3, 4):
            trigger = True
            cls, subtype = "parameter_violation", "timing"
            response = "The recipe timing may be off; check this step."
        return json.dumps({
            "step_id": step,
            "status": "In progress",
            "proactive_trigger": trigger,
            "class": cls,
            "subtype": subtype,
            "response": response,
            "evidence": "mock prediction from ordered task context",
        })


class Consistency:
    def __init__(self, window, tau):
        self.window = deque(maxlen=window)
        self.tau = tau
        self.active = None
        self.completed = []
        self.transitions = []

    def update(self, step, t):
        if step is None:
            return self.active, False, "parse_fail"
        completed_s = {sid_s(s) for s in self.completed}
        if sid_s(step) in completed_s:
            return self.active, False, "ignored_completed_regression"
        self.window.append(step)
        counts = Counter(sid_s(s) for s in self.window)
        dom_s, count = counts.most_common(1)[0]
        threshold = max(1, int(math.ceil(self.tau * len(self.window))))
        if count < threshold:
            return self.active, False, "candidate_waiting"
        dom = next(s for s in self.window if sid_s(s) == dom_s)
        if self.active is None:
            self.active = dom
            self.transitions.append({"t": t, "step_id": dom, "reason": "initial_consensus"})
            return self.active, True, "initial_consensus"
        if sid_s(dom) != sid_s(self.active):
            if sid_s(self.active) != "other":
                self.completed.append(self.active)
            self.active = dom
            self.transitions.append({"t": t, "step_id": dom, "reason": "sliding_window_consensus"})
            return self.active, True, "sliding_window_consensus"
        return self.active, False, "unchanged"


def normalize_prediction(parsed, valid):
    if not isinstance(parsed, dict):
        return None, None
    sid_raw = parsed.get("step_id", parsed.get("current_action"))
    sid = valid.get(sid_s(sid_raw))
    if sid is None and sid_s(sid_raw).lower() == "other":
        sid = "other"
    if sid is None:
        return None, None
    status_raw = parsed.get("status", parsed.get("progress_status"))
    status = status_raw if status_raw in STATUSES else "In progress"
    cls = parsed.get("class") or "none"
    subtype = parsed.get("subtype") or "none"
    if (cls, subtype) not in SCORED_KEYS:
        if cls != "none" or subtype != "none":
            cls, subtype = "none", "none"
    return sid, {
        "step_id": sid,
        "status": status,
        "proactive_trigger": bool(parsed.get("proactive_trigger")),
        "class": cls,
        "subtype": subtype,
        "response": parsed.get("response") or "",
        "evidence": parsed.get("evidence") or "",
    }


def replication_manifest(profile_name, profile, backend_name):
    return {
        "arm": "pro2assist_style_cc4d",
        "claim": "Approximate Pro2Assist-style prior-work baseline on CC4D; not an exact reproduction.",
        "follows_pro2assist": [
            "online current-step and execution-status reasoning",
            "completed-step history as long-term procedural context",
            "motion-cue text injected into VLM prompt",
            "step-aware consistency checking for historical context update",
            "duplicate proactive response suppression",
        ],
        "simulated_proxies": [
            {
                "component": "head_imu_adaptive_sampling",
                "replacement": "median global optical-flow vector magnitude plus frame-difference energy",
                "label": "ours_empirical_proxy",
            },
            {
                "component": "hand_motion_cues",
                "replacement": "lower-center ROI residual optical flow, 8-way dominant direction, HSV state change, skin+motion diagnostic",
                "label": "ours_empirical_proxy",
            },
            {
                "component": "motion_key_moments",
                "replacement": "select high proxy-score frames and force latest/max-gap fallback",
                "label": "pro2assist_inspired_empirical",
            },
        ],
        "ours_empirical": {
            "backend": backend_name,
            "profile": profile_name,
            "params": profile,
            "notes": [
                "fixed lower-center ROI is used because no per-recording hand/ROI annotations are available",
                "thresholds are engineering defaults and must be tuned only on a declared validation split before publication",
                "qwen_replay uses sparse calls because current local Qwen latency is not live-real-time",
            ],
        },
        "omitted": [
            {
                "component": "LoRA_distillation_training",
                "reason": "Pro2Assist training labels and distillation data are unavailable",
                "label": "omitted_unavailable",
            },
            {
                "component": "real_AR_head_IMU_and_display",
                "reason": "CC4D replay has RGB/audio files but no live RayNeo-style IMU/display loop",
                "label": "omitted_not_in_replay_eval",
            },
            {
                "component": "exact_paper_thresholds",
                "reason": "paper hardware parameters do not directly transfer and not all values are public",
                "label": "omitted_unavailable",
            },
        ],
    }


def save_trace(trace_fh, frames_root, rid, call_rec, jpegs):
    frame_files = []
    if frames_root is not None:
        frames_root.mkdir(parents=True, exist_ok=True)
        for k, jb in enumerate(jpegs):
            fn = f"t{int(round(call_rec['t'])):05d}_{k}.jpg"
            (frames_root / fn).write_bytes(jb)
            frame_files.append(fn)
    rec = dict(call_rec)
    rec["frame_files"] = frame_files
    trace_fh.write(json.dumps(rec) + "\n")
    trace_fh.flush()


def run_one(video, task, backend, profile_name, profile, max_seconds=None,
            trace_dir=None, rid=None, verbose=True, io_style="cc4d_scored"):
    steps = task_steps(task)
    valid, _ = canonical_maps(steps)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end_t = min(duration, max_seconds) if max_seconds else duration

    consistency = Consistency(profile["consistency_w"], profile["consistency_tau"])
    stage_intervals, events, recent = [], [], []
    latencies = []
    frames_sent = parse_failures = 0
    motion_compute_s = 0.0
    system_prompt = PRO2ASSIST_SYSTEM_PROMPT if io_style == "pro2assist" else SYSTEM_PROMPT
    last_event_t = -1e9
    delivered = set()

    trace_fh = None
    frames_root = None
    if trace_dir is not None and rid is not None:
        (Path(trace_dir) / "vlm_debug").mkdir(parents=True, exist_ok=True)
        frames_root = Path(trace_dir) / "frames" / rid
        trace_fh = open(Path(trace_dir) / "vlm_debug" / f"{rid}.jsonl", "w")

    t = profile["period_s"]
    while t <= end_t + 1e-6:
        jpegs, selected_infos, motion_cue, motion_meta = observation_window(cap, fps, t, profile)
        motion_compute_s += float(motion_meta.get("compute_s", 0.0))
        if not jpegs:
            break
        if len(jpegs) > profile["max_frames"]:
            raise AssertionError("selected frame count exceeds profile max_frames")
        if any(not s.get("reasons") for s in selected_infos):
            raise AssertionError("selected frame missing selection reason")

        frames_sent += len(jpegs)
        prompt = build_user_prompt(task, steps, consistency.completed, consistency.active,
                                   recent, motion_cue, io_style=io_style)
        call_ctx = {
            "t": t,
            "completed": consistency.completed,
            "active": consistency.active,
            "system_prompt": system_prompt,
            "io_style": io_style,
        }
        t_call = time.time()
        try:
            raw, err = backend.call(jpegs, prompt, call_ctx=call_ctx), None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        latency = time.time() - t_call
        latencies.append(latency)

        parsed = safe_json(raw) if raw else None
        step, norm_pred = normalize_prediction(parsed, valid)
        if norm_pred is None:
            parse_failures += 1
        active, changed, consistency_reason = consistency.update(step, t)
        active_for_stage = active if active is not None else "other"
        stage_intervals.append({
            "stage": active_for_stage,
            "start_s": round(max(0.0, t - profile["period_s"]), 2),
            "end_s": round(min(t, end_t), 2),
        })

        emitted_event = None
        if norm_pred is not None:
            sig = (
                sid_s(active_for_stage),
                norm_pred["status"],
                norm_pred["class"],
                norm_pred["subtype"],
            )
            due = (
                norm_pred["proactive_trigger"]
                and (norm_pred["class"], norm_pred["subtype"]) in SCORED_KEYS
                and (t - last_event_t) >= profile["cooldown_s"]
                and sig not in delivered
            )
            if due:
                emitted_event = {
                    "t": round(t, 2),
                    "class": norm_pred["class"],
                    "subtype": norm_pred["subtype"],
                    "id": f"{norm_pred['class']}_{norm_pred['subtype']}_{sid_s(active_for_stage)}",
                    "message": norm_pred["response"],
                }
                events.append(emitted_event)
                delivered.add(sig)
                last_event_t = t
            recent.append({
                "t": t,
                "step_id": norm_pred["step_id"],
                "status": norm_pred["status"],
                "class": norm_pred["class"],
                "subtype": norm_pred["subtype"],
                "evidence": norm_pred["evidence"],
            })

        if trace_fh is not None:
            save_trace(trace_fh, frames_root, rid, {
                "call": len(latencies),
                "t": round(t, 2),
                "request": {
                    "model": backend.model,
                    "system_prompt": system_prompt,
                    "io_style": io_style,
                    "user_prompt": prompt,
                    "motion_cue": motion_cue,
                    "motion_meta": motion_meta,
                    "n_images": len(jpegs),
                },
                "response": {
                    "raw": raw,
                    "parsed": parsed,
                    "normalized": norm_pred,
                    "error": err,
                    "latency_s": round(latency, 2),
                },
                "consistency": {
                    "active_step": active_for_stage,
                    "changed": changed,
                    "reason": consistency_reason,
                    "completed_steps": consistency.completed,
                    "window": list(consistency.window),
                },
                "emitted_event": emitted_event,
            }, jpegs)

        if verbose:
            ev = " event" if emitted_event else ""
            st = norm_pred["status"] if norm_pred else "PARSE_FAIL"
            print(f"  t={t:6.1f}s step={active_for_stage} ({st}, {consistency_reason}){ev}")
        t += profile["period_s"]

    cap.release()
    if trace_fh is not None:
        trace_fh.close()

    return {
        "stage_intervals": stage_intervals,
        "events": events,
        "escalation_requests": [],
        "cost": {
            "vlm_calls": len(latencies),
            "frames_sent": frames_sent,
            "vlm_latency_total_s": round(float(np.sum(latencies)), 2) if latencies else 0.0,
            "compute_s": round(motion_compute_s, 2),
        },
        "replication_manifest": replication_manifest(profile_name, profile, backend.model),
        "_meta": {
            "model": backend.model,
            "profile": profile_name,
            "io_style": io_style,
            "evaluated_s": round(end_t, 2),
            "video_duration_s": round(duration, 2),
            "parse_failures": parse_failures,
            "status_labels_logged": True,
            "trace": trace_dir is not None,
            "consistency_transitions": consistency.transitions,
        },
    }


def make_backend(args, steps):
    if args.backend == "mock":
        return MockBackend(steps, parse_fail_every=args.mock_parse_fail_every)
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
    ap.add_argument("--out-dir", default="experiments/pro2assist_style")
    ap.add_argument("--arm", default="pro2assist_style_cc4d")
    ap.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="qwen_replay")
    ap.add_argument("--io-style", choices=["cc4d_scored", "pro2assist"], default="cc4d_scored",
                    help="VLM prompt/output interface: scorer-oriented CC4D JSON or Pro^2Assist-style guideline/history/sensory/response JSON")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=None,
                    help="override profile cooldown_s")
    ap.add_argument("--mock-parse-fail-every", type=int, default=0,
                    help="mock-only: return invalid JSON every N calls (0 disables)")
    args = ap.parse_args()

    profile = dict(PROFILES[args.profile])
    if args.cooldown is not None:
        profile["cooldown_s"] = args.cooldown
    outd = output_dir(args)
    trace_dir = outd if args.trace else None

    if args.corpus:
        rmap = recipe_stem_by_recording()
        todo = [(rid, stem) for rid, stem in sorted(rmap.items())
                if (BASE / args.video_dir / f"{rid}.mp4").exists()]
        print(f"corpus: {len(todo)} recordings with raw task+video")
        for i, (rid, stem) in enumerate(todo, 1):
            outp = outd / f"{rid}.json"
            if outp.exists():
                continue
            task = json.loads((BASE / "tasks/cc4d" / f"{stem}.json").read_text())
            steps = task_steps(task)
            backend = make_backend(args, steps)
            print(f"[{i}/{len(todo)}] {rid} ({stem})")
            res = run_one(BASE / args.video_dir / f"{rid}.mp4", task, backend,
                          args.profile, profile, max_seconds=args.max_seconds,
                          trace_dir=trace_dir, rid=rid, verbose=False,
                          io_style=args.io_style)
            res["recording"], res["arm"] = rid, args.arm
            outp.write_text(json.dumps(res, indent=1))
    else:
        if not (args.video and args.task):
            raise SystemExit("single mode needs --video and --task (or use --corpus)")
        rid = Path(args.video).stem
        task = json.loads(Path(args.task).read_text())
        steps = task_steps(task)
        backend = make_backend(args, steps)
        res = run_one(args.video, task, backend, args.profile, profile,
                      max_seconds=args.max_seconds, trace_dir=trace_dir,
                      rid=rid, verbose=True, io_style=args.io_style)
        res["recording"], res["arm"] = rid, args.arm
        outp = outd / f"{rid}.json"
        outp.write_text(json.dumps(res, indent=1))
        print(f"wrote {outp} ({res['cost']['vlm_calls']} calls, "
              f"{res['_meta']['parse_failures']} parse-fails)")


if __name__ == "__main__":
    main()
