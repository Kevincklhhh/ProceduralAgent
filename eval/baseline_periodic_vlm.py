#!/usr/bin/env python3
"""Periodic VLM baseline: call a VLM every N seconds on a cooking video and
decide current step, completion status, hazards, and assistant action.

Usage:
  python baseline_periodic_vlm.py --video cook.mp4 --task task_pan_fried_egg.json \
      --backend gemini --interval 10 --out runs/run_001

Backends (env vars):
  gemini : GOOGLE_API_KEY (or GEMINI_API_KEY), optional GEMINI_MODEL
  qwen   : QWEN_VIDEO_SERVER_URL (OpenAI-compatible), optional QWEN_VIDEO_MODEL,
           QWEN_VIDEO_API_KEY   (same names as IR/qwen_video_api.py)

Outputs in --out: calls.jsonl (one line per VLM call) and summary.json.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

SYSTEM_PROMPT = """You are observing a cooking task from a camera. Use only visible evidence in the provided frames and the recipe context. The frames are consecutive snapshots ending at the current time.

Decide:
1. Which recipe step is most likely happening now (use exactly one step_id from the list, or "other" if the user is not doing any listed step).
2. Whether that step appears not_started, in_progress, complete, or uncertain.
3. Whether any visible mistake or safety issue appears.
4. Whether the assistant should act now: none, reminder, warning, or ask_confirmation. Only act when one of the listed reminder triggers is clearly met and it was not already issued recently. Otherwise use "none".

If the frames do not contain enough evidence, use status "uncertain" and action "none".

Return JSON only, exactly this schema, no prose:
{"step_id": "...", "status": "...", "confidence": 0.0, "evidence": ["..."], "hazard": null, "action": {"type": "none", "message": "", "reason": "..."}}"""


def build_user_prompt(task, t, history, last_action):
    steps = "\n".join(f"  {s['order']}. {s['step_id']}: {s['instruction']}" for s in task["steps"])
    rems = "\n".join(f"  - [{r['type']}] when {r['trigger']} -> \"{r['message']}\"" for r in task["reminders"])
    hist = "\n".join(f"  t={h['timestamp_s']:.0f}s: {h['step_id']} (conf {h['confidence']:.2f})" for h in history) or "  (none yet)"
    la = f"t={last_action['timestamp_s']:.0f}s: {last_action['type']}" if last_action else "(none yet)"
    return (f"Recipe: {task['title']}\nSteps:\n{steps}\nReminder/warning triggers:\n{rems}\n"
            f"Current video time: {t:.0f}s\nRecent step history:\n{hist}\nLast assistant action: {la}")


def safe_extract_json(text):
    """direct parse -> strip code fences -> greedy {...} substring."""
    for candidate in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def sample_frames(cap, fps, t_end, window_s, n_frames, max_dim=768, jpeg_q=85):
    """n_frames jpegs evenly spaced over [t_end - window_s, t_end]."""
    times = np.linspace(max(0.0, t_end - window_s), t_end, n_frames)
    jpegs = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok:
            jpegs.append(buf.tobytes())
    return jpegs


# ---------------- backends ----------------

class GeminiBackend:
    def __init__(self):
        from google import genai
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def call(self, jpegs, user_prompt):
        from google.genai import types
        parts = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in jpegs]
        parts.append(types.Part.from_text(text=user_prompt))
        resp = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return resp.text


class QwenBackend:
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.url = base
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, jpegs, user_prompt):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": user_prompt})
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 2000,
                   "response_format": {"type": "json_object"},  # vLLM guided JSON; reasoning models otherwise truncate
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}]}
        r = requests.post(self.url, json=payload, headers=self.headers, timeout=120)
        r.raise_for_status()
        c = r.json()["choices"][0]["message"]["content"]
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c


# ---------------- main loop ----------------

def run(args):
    task = json.loads(Path(args.task).read_text())
    valid_steps = {s["step_id"] for s in task["steps"]} | {"other"}
    backend = GeminiBackend() if args.backend == "gemini" else QwenBackend()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end_t = min(duration, args.max_seconds) if args.max_seconds else duration

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calls_f = open(out / "calls.jsonl", "w")

    video_id = Path(args.video).stem
    history, latencies = [], []
    current_step, pending_step, pending_count = None, None, 0
    stage_timeline, events = [], []
    last_action, parse_failures, frames_sent = None, 0, 0

    t = args.interval
    while t <= end_t:
        jpegs = sample_frames(cap, fps, t, args.window, args.frames_per_call)
        if not jpegs:
            break
        frames_sent += len(jpegs)
        prompt = build_user_prompt(task, t, history[-3:], last_action)

        t0 = time.time()
        try:
            raw = backend.call(jpegs, prompt)
            err = None
        except Exception as e:  # log and continue: one bad call must not kill the run
            raw, err = "", f"{type(e).__name__}: {e}"
        latency = time.time() - t0
        latencies.append(latency)

        parsed = safe_extract_json(raw) if raw else None
        if not parsed or parsed.get("step_id") not in valid_steps:
            parse_failures += 1
            parsed = None

        rec = {"call_id": f"{video_id}_t{int(t):04d}", "timestamp_s": round(t, 1),
               "n_frames": len(jpegs), "latency_s": round(latency, 2),
               "raw": raw, "parsed": parsed, "error": err}
        calls_f.write(json.dumps(rec) + "\n")
        calls_f.flush()

        if parsed:
            step = parsed["step_id"]
            conf = float(parsed.get("confidence", 0.5))
            history.append({"timestamp_s": t, "step_id": step, "confidence": conf})

            # smoothing: switch only after the same new step in K consecutive calls
            if step == current_step:
                pending_step, pending_count = None, 0
            elif step == pending_step:
                pending_count += 1
                if pending_count >= args.k_consecutive:
                    if current_step is not None:
                        stage_timeline[-1]["end_s"] = round(t, 1)
                    stage_timeline.append({"step_id": step, "start_s": round(t, 1), "end_s": round(t, 1)})
                    current_step, pending_step, pending_count = step, None, 0
            else:
                pending_step, pending_count = step, 1
                if current_step is None:  # first prediction seeds the timeline
                    stage_timeline.append({"step_id": step, "start_s": round(t, 1), "end_s": round(t, 1)})
                    current_step, pending_step, pending_count = step, None, 0

            action = parsed.get("action") or {}
            a_type = action.get("type", "none")
            cooled = last_action is None or (t - last_action["timestamp_s"]) >= args.cooldown
            if a_type != "none" and a_type in task["allowed_assistant_actions"] and cooled:
                ev = {"timestamp_s": round(t, 1), "action_type": a_type,
                      "message": action.get("message", ""), "step_id": step}
                events.append(ev)
                last_action = {"timestamp_s": t, "type": a_type}
                print(f"  t={t:6.0f}s  {step:16s} -> {a_type.upper()}: {action.get('message', '')}")
            else:
                print(f"  t={t:6.0f}s  {step:16s} ({parsed.get('status')}, conf {conf:.2f})")
        else:
            print(f"  t={t:6.0f}s  PARSE FAIL{' / ' + err if err else ''}")

        t += args.interval

    if stage_timeline:
        stage_timeline[-1]["end_s"] = round(min(t, end_t), 1)
    calls_f.close()
    cap.release()

    n_calls = len(latencies)
    summary = {
        "video_id": video_id, "task_id": task["task_id"],
        "baseline": {"mode": "periodic_vlm", "interval_s": args.interval,
                     "frames_per_call": args.frames_per_call, "window_s": args.window,
                     "backend": args.backend, "model": backend.model,
                     "k_consecutive": args.k_consecutive, "cooldown_s": args.cooldown},
        "video_duration_s": round(duration, 1), "evaluated_seconds": round(end_t, 1),
        "stage_timeline": stage_timeline, "events": events,
        "cost_log": {
            "num_vlm_calls": n_calls, "frames_sent": frames_sent,
            "mean_latency_s": round(float(np.mean(latencies)), 2) if latencies else None,
            "p95_latency_s": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
            "parse_failure_rate": round(parse_failures / n_calls, 3) if n_calls else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {n_calls} calls, {len(events)} assistant actions, "
          f"parse failures {parse_failures}. Wrote {out}/summary.json and {out}/calls.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--backend", choices=["gemini", "qwen"], default="gemini")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between VLM calls")
    p.add_argument("--frames-per-call", type=int, default=3)
    p.add_argument("--window", type=float, default=2.0, help="seconds covered by the frames of one call")
    p.add_argument("--k-consecutive", type=int, default=2, help="calls needed to confirm a stage switch")
    p.add_argument("--cooldown", type=float, default=15.0, help="min seconds between assistant actions")
    p.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds")
    p.add_argument("--out", default="runs/latest")
    run(p.parse_args())
