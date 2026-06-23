"""proposed_vlm_arm -- the VLM arm of the procedure-monitor runtime (proposed system).

The monitor calls this only where audio cannot decide: a C-none step/block (or any
active step that carries a recipe-derived check) gets ONE periodic merged call --
cadence + cost capped by the plan's vlm_policy. That single call does both jobs at
once: (1) identify which candidate step is happening now, and (2) for that step,
decide whether to SPEAK UP (emit a reminder) or STAY SILENT (user on track). It never
fires a second call for the checks. Reuses eval/baseline_t1_step.py for the
Qwen-on-saltyfish client, 1 fps frame sampling, and JSON parsing.

Modes:
  - "qwen": real Qwen on saltyfish (needs QWEN_VIDEO_SERVER_URL + a video file).
  - "mock": no server; returns a deterministic verdict (walks members, never emits a
    reminder) so the polling/cost path is exercisable offline.

poll_and_check(video, t, plan, unit, completed) -> (verdict|None, latency_s, n_frames)
  verdict = {step_id, status, evidence, reminders: [{reminder, message, observed}]}
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))

# Merged recognition + proactive-reminder prompt. The VLM is framed as an assistant that
# decides whether to speak: it stays silent when the user is on track, and emits a reminder
# only on a clear, listed deviation for the step actually underway.
_SYSTEM_PROMPT = (
    "You are a proactive cooking assistant watching a user through recent egocentric frames "
    "(consecutive snapshots ending now). Two jobs in one answer:\n"
    "1) RECOGNIZE: commit to exactly ONE step_id from the candidate list — the step the user "
    'is performing now (or most recently performed). Never answer "other"; use status for '
    "uncertainty.\n"
    "2) DECIDE — speak up or stay silent. For the step you picked, each candidate lists the "
    "recipe expectations to verify. STAY SILENT (reminders: []) when the user is on track — "
    "this is the normal case. Emit a reminder ONLY when the frames CLEARLY show the user "
    "deviating from a listed expectation for the step that is actually underway. If the step "
    "is not clearly happening yet (e.g. status 'Just start', still fetching tools), or you are "
    "unsure, STAY SILENT.\n\n"
    "Return JSON only, exactly this schema, no prose:\n"
    '{"step_id": <one candidate step_id>, '
    '"status": "<Just start|In progress|About to finish|Step transition>", '
    '"evidence": "<short visual justification>", '
    '"reminders": [{"reminder": "<a listed reminder tag for this step>", '
    '"message": "<short corrective nudge to the user>", "observed": "<what you saw>"}]}'
)


class VLMArm:
    def __init__(self, mode="mock", trace_dir=None, recording=None):
        self.mode = mode
        self.trace_dir = trace_dir
        self.recording = recording
        self._poll_counts = {}
        self._backend = None
        self._trace = {}
        if trace_dir:
            os.makedirs(os.path.join(trace_dir, "traces"), exist_ok=True)

    # ---- candidates for a unit: block members, or the single step itself ----
    def _candidates(self, unit):
        if unit.is_block:
            return [{"step_id": m["step_id"], "instruction": m.get("instruction", ""),
                     "checks": m.get("checks", [])} for m in unit.members]
        return [{"step_id": unit.uid, "instruction": unit.instruction,
                 "checks": getattr(unit, "checks", [])}]

    # ---- merged recognition + checks prompt ----
    def _build_prompt(self, unit, cands, completed):
        lines = []
        for c in cands:
            lines.append(f"  step_id={c['step_id']}: {c['instruction']}")
            for chk in c["checks"]:
                if chk.get("detector") == "VLM":
                    lines.append(f"      expectation[{chk.get('reminder')}]: "
                                 f"{chk.get('detection_criteria', '')}")
        cand_block = "\n".join(lines)
        hist = "; ".join(completed) if completed else "(none yet)"
        return (f"The user is mid-recipe. The frames are the most recent snapshots ending now.\n"
                f"Candidate steps (pick the ONE happening now), each with the expectations to "
                f"verify for it:\n{cand_block}\n\n"
                f"Already done: {hist}\n\n"
                "Pick the current step. For THAT step only, check its listed expectations and "
                "decide whether to speak. Stay silent (reminders: []) if the user is on track "
                "or the step is not clearly underway; emit a reminder only on a clear "
                "deviation. Return JSON only per the schema.")

    def _qwen_call(self, jpegs, prompt, system=_SYSTEM_PROMPT):
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
                   "messages": [{"role": "system", "content": system},
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

    def _map_step(self, cands, sid_raw):
        sid = str(sid_raw)
        for c in cands:
            if sid == str(c["step_id"]):
                return c["step_id"]
        return None

    def poll_and_check(self, video, t, plan, unit, completed=None):
        """One merged call: recognize the active step + decide speak/silent for its checks."""
        pol = plan.vlm_policy
        window = float(pol.get("window_s", 10.0))
        sample_fps = float(pol.get("fps", 1.0))
        n_frames = max(1, int(round(window * sample_fps)))
        completed = completed or []
        cands = self._candidates(unit)

        if self.mode == "mock":
            k = self._poll_counts.get(unit.uid, 0)
            self._poll_counts[unit.uid] = k + 1
            c = cands[min(k, len(cands) - 1)]
            verdict = {"step_id": c["step_id"], "status": "In progress",
                       "confidence": 0.5, "evidence": "mock", "reminders": []}
            self._log(unit, t, [], verdict, 0.0, None)
            return verdict, 0.0, n_frames

        # ---- real Qwen on saltyfish ----
        import cv2
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from baseline_t1_step import sample_frames_1fps, safe_json
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        jpegs = sample_frames_1fps(cap, fps, t, window, max_frames=n_frames, sample_fps=sample_fps)
        cap.release()
        prompt = self._build_prompt(unit, cands, completed)
        t0 = time.time()
        try:
            raw, err = self._qwen_call(jpegs, prompt), None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        parsed = safe_json(raw) or {}
        step = self._map_step(cands, parsed.get("step_id"))
        verdict = None
        if step:
            # keep only reminders whose tag is a real listed expectation for the named step
            valid = {chk.get("reminder") for c in cands if c["step_id"] == step
                     for chk in c["checks"] if chk.get("detector") == "VLM"}
            rems = [r for r in (parsed.get("reminders") or [])
                    if isinstance(r, dict) and r.get("reminder") in valid]
            verdict = {"step_id": step, "status": parsed.get("status"),
                       "confidence": parsed.get("confidence", 1.0),
                       "evidence": parsed.get("evidence"), "reminders": rems}
        self._log(unit, t, jpegs, verdict, lat, err)
        return verdict, lat, len(jpegs)

    def _log(self, unit, t, jpegs, verdict, lat, err):
        if not self.trace_dir:
            return
        fh = self._trace.get(unit.uid)
        if fh is None:
            fh = open(os.path.join(self.trace_dir, "traces", f"{unit.uid}.jsonl"), "a")
            self._trace[unit.uid] = fh
        fh.write(json.dumps({"recording": self.recording, "t": round(t, 1),
                             "unit": unit.uid, "n_frames": len(jpegs),
                             "verdict": verdict, "latency_s": round(lat, 2),
                             "error": err}) + "\n")
        fh.flush()
