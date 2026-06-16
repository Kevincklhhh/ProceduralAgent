#!/usr/bin/env python3
"""Tier-1 baseline: ONLINE CURRENT-STEP RECOGNITION (T1 only), RGB-only, Pro2Assist-style.

Per call (every --interval s): sample frames at 1 fps over the last interval, send
[system prompt + recipe step guideline + completed-steps history + frames], ask for the
current step + status. Replicates Pro2Assist's step / status / historical-context / system
prompt. DROPPED on purpose (per design 2026-06-15): proactive trigger, consistency-check
smoothing, hand-motion cues, fine-tuning. Model is zero-shot Qwen3.6 on saltyfish.

The predicted step labels the window [t-interval, t); successive windows tile the timeline
-> stage_intervals. No events are emitted (T2 is out of scope for Tier 1).

Writes the unified per-recording JSON that eval/score_corpus.py reads:
  <out-dir>/<arm>/<rid>.json  {stage_intervals, events:[], escalation_requests:[], cost}

Usage (single video):
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/run_step_baseline.py --video data/videos_360p/8_11.mp4 \
     --task tasks/cc4d/spicedhotchocolate.json --interval 10 --arm qwen36_i10
Usage (corpus over all recordings that have a task JSON + a 360p video):
  ... python eval/run_step_baseline.py --corpus --interval 10 --arm qwen36_i10
"""
import argparse, base64, json, os, re, time, glob
from pathlib import Path
import cv2, numpy as np, requests

BASE = Path(__file__).resolve().parent.parent
ANN = BASE / 'data/cc4d/annotations/annotation_json'

SYSTEM_PROMPT = (
    "You are a proactive assistant for procedural cooking tasks, tracking the user's progress "
    "through real-time egocentric camera frames and procedural context. Based on these contexts, "
    "understand the user's actions and identify the current procedural step and its execution "
    "status. The frames are consecutive snapshots ending at the current moment.\n\n"
    "Always commit to exactly ONE step_id from the guideline — the single most likely current "
    'step — even when uncertain. Never answer "other". If the user is between steps, choose the '
    "step they are transitioning into (or the one just finished) and reflect that in status.\n\n"
    "Return JSON only, exactly this schema, no prose:\n"
    '{"step_id": <the single most likely step_id from the guideline>, '
    '"status": "<Just start|In progress|About to finish|Step transition>", '
    '"evidence": "<short visual justification>"}'
)


def norm(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def build_user_prompt(task, history):
    """history: list of prior VLM responses {t, step, status, evidence} (most recent last)."""
    steps = "\n".join(f"  {s['order']}. step_id={s['step_id']}: {s['instruction']}"
                      for s in task['steps'])
    if history:
        obs = "\n".join(
            f"  t={h['t']:.0f}s: step_id={h['step']} ({h['status']}) — {h['evidence']}"
            for h in history)
    else:
        obs = "  (none yet)"
    return (f"Guideline (recipe: {task['title']}). Steps in nominal order:\n{steps}\n\n"
            f"Your previous observations (your own recent predictions, most recent last):\n{obs}\n\n"
            f"Sensory Context: the images are the most recent frames. "
            f"Identify the current step and status now.")


def safe_json(text):
    for c in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())):
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def sample_frames_1fps(cap, fps, t_end, interval, max_frames=16, max_dim=768, jpeg_q=85):
    """One frame per second over [t_end - interval, t_end] (capped)."""
    n = min(max_frames, max(1, int(round(interval))))
    times = np.linspace(max(0.0, t_end - interval), t_end, n)
    jpegs = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        sc = max_dim / max(h, w)
        if sc < 1.0:
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok:
            jpegs.append(buf.tobytes())
    return jpegs


class Qwen:
    def __init__(self):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + ("" if base.endswith("/chat/completions")
                           else "/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, jpegs, user_prompt):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": user_prompt})
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 1200,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}]}
        r = requests.post(self.url, json=payload, headers=self.headers, timeout=120)
        r.raise_for_status()
        c = r.json()["choices"][0]["message"]["content"]
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c


def run_one(video, task, backend, interval, max_seconds=None, verbose=True,
            trace_dir=None, rid=None, history_k=5):
    valid = {str(s['step_id']): s['step_id'] for s in task['steps']}    # model-out -> canonical
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end_t = min(duration, max_seconds) if max_seconds else duration

    # trace = per-call VLM context window (frames + prompt + prediction) for the visualizer
    trace, frames_root = None, None
    if trace_dir is not None and rid is not None:
        frames_root = Path(trace_dir) / "frames" / rid
        frames_root.mkdir(parents=True, exist_ok=True)
        (Path(trace_dir) / "traces").mkdir(parents=True, exist_ok=True)
        trace = open(Path(trace_dir) / "traces" / f"{rid}.jsonl", "w")

    stage_intervals, latencies = [], []
    frames_sent = parse_fail = 0
    history = []                          # prior VLM responses {t, step, status, evidence}

    t = interval
    while t <= end_t + 1e-6:
        jpegs = sample_frames_1fps(cap, fps, t, interval)
        if not jpegs:
            break
        frames_sent += len(jpegs)
        ctx = history if history_k <= 0 else history[-history_k:]   # prior responses shown
        prompt = build_user_prompt(task, ctx)
        t0 = time.time()
        try:
            raw = backend.call(jpegs, prompt); err = None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        latencies.append(lat)

        parsed = safe_json(raw)
        sid_raw = str(parsed.get("step_id")) if parsed else None
        if sid_raw in valid:
            step = valid[sid_raw]
        elif sid_raw == "other":
            step = "other"
        else:
            step = "other"; parse_fail += 1

        stage_intervals.append({"stage": step,
                                "start_s": round(t - interval, 1), "end_s": round(min(t, end_t), 1)})

        if trace is not None:                          # dump the call's context window
            frame_files = []
            for k, j in enumerate(jpegs):
                fn = f"t{int(round(t)):05d}_{k}.jpg"
                (frames_root / fn).write_bytes(j)
                frame_files.append(fn)
            trace.write(json.dumps({
                "t": round(t, 1), "start_s": round(t - interval, 1), "end_s": round(min(t, end_t), 1),
                "frame_files": frame_files, "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt,
                "completed_step_ids": list(completed), "pred_step": step,
                "pred_status": (parsed or {}).get("status"), "pred_evidence": (parsed or {}).get("evidence"),
                "raw": raw, "error": err, "latency_s": round(lat, 2)}) + "\n")
            trace.flush()

        if step != "other" and step != active:        # immediate completed-step update
            if active is not None and active != "other":
                completed.append(active)
            active = step
        if verbose:
            st = parsed.get("status") if parsed else "PARSE_FAIL"
            print(f"  t={t:6.0f}s  step={step}  ({st})")
        t += interval

    cap.release()
    if trace is not None:
        trace.close()
    n = len(latencies)
    return {
        "stage_intervals": stage_intervals, "events": [], "escalation_requests": [],
        "cost": {"vlm_calls": n, "frames_sent": frames_sent,
                 "vlm_latency_total_s": round(float(np.sum(latencies)), 2) if latencies else 0.0,
                 "compute_s": 0.0},
        "_meta": {"model": backend.model, "interval_s": interval, "fps_sampling": 1.0,
                  "evaluated_s": round(end_t, 1), "parse_failures": parse_fail,
                  "traced": trace_dir is not None},
    }


def _alnum(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def recipe_stem_by_recording():
    """recording_id -> recipe stem. Recipe stems ('spicedhotchocolate') vs activity names
    ('Spiced Hot Chocolate') match only after stripping non-alphanumerics."""
    ann = json.load(open(ANN / 'complete_step_annotations.json'))
    norm2stem = {_alnum(os.path.basename(f)[:-5]): os.path.basename(f)[:-5]
                 for f in glob.glob(str(BASE / 'tasks/cc4d/*.json'))}
    out = {}
    for rid, r in ann.items():
        s = norm2stem.get(_alnum(r['activity_name']))
        if s:
            out[rid] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--task")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--arm", default="qwen36")
    ap.add_argument("--out-dir", default="experiments/t1_baseline")
    ap.add_argument("--video-dir", default="data/videos_360p")
    ap.add_argument("--trace", action="store_true",
                    help="dump per-call frames+prompt+prediction for the visualizer")
    a = ap.parse_args()

    backend = Qwen()
    outd = BASE / a.out_dir / a.arm
    outd.mkdir(parents=True, exist_ok=True)
    tdir = outd if a.trace else None

    if a.corpus:
        rmap = recipe_stem_by_recording()
        todo = [(rid, stem) for rid, stem in sorted(rmap.items())
                if (BASE / a.video_dir / f"{rid}.mp4").exists()]
        print(f"corpus: {len(todo)} recordings with task+video")
        for i, (rid, stem) in enumerate(todo, 1):
            outp = outd / f"{rid}.json"
            if outp.exists():
                continue
            task = json.loads((BASE / 'tasks/cc4d' / f"{stem}.json").read_text())
            print(f"[{i}/{len(todo)}] {rid} ({stem})")
            res = run_one(BASE / a.video_dir / f"{rid}.mp4", task, backend, a.interval,
                          a.max_seconds, verbose=False, trace_dir=tdir, rid=rid)
            res["recording"], res["arm"] = rid, a.arm
            outp.write_text(json.dumps(res, indent=1))
    else:
        if not (a.video and a.task):
            raise SystemExit("single mode needs --video and --task (or use --corpus)")
        rid = Path(a.video).stem
        task = json.loads(Path(a.task).read_text())
        res = run_one(a.video, task, backend, a.interval, a.max_seconds,
                      trace_dir=tdir, rid=rid)
        res["recording"], res["arm"] = rid, a.arm
        (outd / f"{rid}.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {outd / (rid + '.json')}  ({res['cost']['vlm_calls']} calls, "
              f"{res['_meta']['parse_failures']} parse-fails)")


if __name__ == "__main__":
    main()
