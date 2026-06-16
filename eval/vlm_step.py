"""vlm_step -- the VLM arm of the procedure-monitor runtime.

The monitor calls this only where audio cannot decide: a C-none step/block is
polled periodically (cadence + cost capped by the plan's vlm_policy) to label its
silent members, and a B-trigger candidate fires exactly one confirming call.
Reuses eval/run_step_baseline.py wholesale (the Qwen-on-saltyfish client, 1 fps
frame sampling, JSON parsing, {step_id,status,evidence} schema).

Modes:
  - "qwen": real Qwen on saltyfish (needs QWEN_VIDEO_SERVER_URL + a video file).
  - "mock": no server; returns a deterministic verdict so the polling/cost path
    is exercisable offline. The mock walks the block's members in order.

poll(video, t, plan, unit) -> (verdict|None, latency_s, n_frames)
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))

_SYSTEM_PROMPT = (
    "You are tracking a user's progress through a cooking task from recent egocentric "
    "frames and the recipe context. Identify the procedural step happening now and its "
    "status. The frames are consecutive snapshots ending at the current moment.\n\n"
    "Always commit to exactly ONE step_id from the candidate list — the single most likely "
    "step the user is currently performing (or has most recently performed) — even when "
    'uncertain. Never answer "other". Use status to express uncertainty.\n\n'
    "Return JSON only, exactly this schema, no prose:\n"
    '{"step_id": <one step_id from the candidates>, '
    '"status": "<Just start|In progress|About to finish|Step transition>", '
    '"evidence": "<short visual justification>"}'
)


class VLMArm:
    def __init__(self, mode="mock", trace_dir=None):
        self.mode = mode
        self.trace_dir = trace_dir
        self._poll_counts = {}
        self._backend = None
        self._trace = {}
        if trace_dir:
            os.makedirs(os.path.join(trace_dir, "traces"), exist_ok=True)

    # ---- prompt for the silent adds/mix region ----
    def _build_prompt(self, unit, completed):
        cands = "\n".join(f"  step_id={m['step_id']}: {m['instruction']}"
                          for m in unit.members)
        hist = "; ".join(completed) if completed else "(none yet)"
        return (f"The user is in the 'adds + mix' phase of a recipe. The frames are the "
                f"most recent snapshots ending now.\nCandidate steps:\n{cands}\n\n"
                f"Already done: {hist}\n\n"
                "Pick the single candidate step the user is doing now (or most recently "
                "did) — always choose one, never \"other\". Return JSON only, exactly: "
                '{"step_id": <one step_id above>, '
                '"status": "<Just start|In progress|About to finish|Step transition>", '
                '"evidence": "<short visual justification>"}')

    def _qwen_call(self, jpegs, prompt):
        """One Qwen-on-saltyfish call. max_tokens=2000 (Qwen3.6 is a reasoning model
        and truncates below that); response_format json_object; temp 0."""
        import base64
        import requests
        base = os.getenv("QWEN_VIDEO_SERVER_URL",
                         "http://saltyfish.eecs.umich.edu:8000").rstrip("/")
        url = (base if base.endswith("/chat/completions")
               else base + ("/chat/completions" if base.endswith("/v1")
                            else "/v1/chat/completions"))
        model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": prompt})
        payload = {"model": model, "temperature": 0.0, "max_tokens": 2000,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                                {"role": "user", "content": content}]}
        headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"
        r = requests.post(url, json=payload, headers=headers, timeout=180)
        r.raise_for_status()
        c = r.json()["choices"][0]["message"]["content"]
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c

    def _map_step(self, unit, sid_raw):
        sid = str(sid_raw)
        for m in unit.members:
            if sid == str(m["step_id"]) or sid == str(m.get("cc4d_step_id")):
                return m["step_id"]
        return None

    def poll(self, video, t, plan, unit, completed=None):
        pol = plan.vlm_policy
        n_frames = int(pol.get("n_frames", 10))
        window = float(pol.get("window_s", 10.0))
        completed = completed or []

        if self.mode == "mock":
            k = self._poll_counts.get(unit.uid, 0)
            self._poll_counts[unit.uid] = k + 1
            m = unit.members[min(k, len(unit.members) - 1)]
            verdict = {"step_id": m["step_id"], "status": "In progress",
                       "confidence": 0.5, "evidence": "mock"}
            self._log(unit, t, [], verdict, 0.0, None)
            return verdict, 0.0, n_frames

        # ---- real Qwen on saltyfish ----
        import cv2
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from run_step_baseline import sample_frames_1fps, safe_json
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        jpegs = sample_frames_1fps(cap, fps, t, window, max_frames=n_frames)
        cap.release()
        prompt = self._build_prompt(unit, completed)
        t0 = time.time()
        try:
            raw, err = self._qwen_call(jpegs, prompt), None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        parsed = safe_json(raw) or {}
        step = self._map_step(unit, parsed.get("step_id"))
        verdict = None
        if step:
            verdict = {"step_id": step, "status": parsed.get("status"),
                       "confidence": parsed.get("confidence", 1.0),
                       "evidence": parsed.get("evidence")}
        self._log(unit, t, jpegs, verdict, lat, err)
        return verdict, lat, len(jpegs)

    def _log(self, unit, t, jpegs, verdict, lat, err):
        if not self.trace_dir:
            return
        fh = self._trace.get(unit.uid)
        if fh is None:
            fh = open(os.path.join(self.trace_dir, "traces", f"{unit.uid}.jsonl"), "a")
            self._trace[unit.uid] = fh
        fh.write(json.dumps({"t": round(t, 1), "block": unit.uid,
                             "n_frames": len(jpegs), "verdict": verdict,
                             "latency_s": round(lat, 2), "error": err}) + "\n")
        fh.flush()
