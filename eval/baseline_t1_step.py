#!/usr/bin/env python3
"""Tier-1 baseline: ONLINE CURRENT-STEP RECOGNITION (T1 only), RGB-only, Pro2Assist-style.

Per call (every --interval s): sample frames at 1 fps over the last interval, send
[system prompt + recipe step guideline + the list of already-completed steps + the model's
own recent responses (timestamp + predicted step + evidence, last --history of them) +
frames], ask for the current step + status. DROPPED on purpose (per design 2026-06-15):
proactive trigger, consistency-check smoothing, hand-motion cues, fine-tuning. Model is
zero-shot Qwen3.6 on saltyfish.

Order follows the recipe's PARTIAL order, taken from each step's `preconditions` (the DAG),
NOT a linear chain. Steps that share a rank are UNORDERED -- e.g. the three hot-chocolate adds,
which each only require the first microwave -- and are shown to the model as one [ANY ORDER]
group. Only CROSS-group regression is clamped: a prediction that drops to an earlier rank group
is clamped forward, but within a group the adds may be predicted in any order (both raw and
enforced labels are kept in the trace/debug). The predicted step labels the window
[t-interval, t); successive windows tile the timeline -> stage_intervals. No events are
emitted (T2 is out of scope for Tier 1).

--debug records every VLM call's full input/output to <out>/<arm>/vlm_debug/<rid>.jsonl.

Writes the unified per-recording JSON that eval/eval_score_corpus.py reads:
  <out-dir>/<arm>/<rid>.json  {stage_intervals, events:[], escalation_requests:[], cost}

(Also reused as a library by eval/proposed_vlm_arm.py for the Qwen client + frame sampling.)

Usage (single video):
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_t1_step.py --video data/videos_360p/8_11.mp4 \
     --task tasks/cc4d/spicedhotchocolate.json --interval 10 --arm qwen36_i10
Usage (corpus over all recordings that have a task JSON + a 360p video):
  ... python eval/baseline_t1_step.py --corpus --interval 10 --arm qwen36_i10
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
    "The evidence field MUST be a single short sentence of at most 20 words. Do NOT enumerate "
    "timer digits, frame-by-frame details, or lists; summarize the visual cue in one clause.\n\n"
    "Return JSON only, exactly this schema, no prose:\n"
    '{"step_id": <the single most likely step_id from the guideline>, '
    '"status": "<Just start|In progress|About to finish|Step transition>", '
    '"evidence": "<one short sentence, <=20 words>"}'
)


def norm(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def step_ranks(steps):
    """step_id -> rank (a PARTIAL order). Prefer an explicit `rank`; else longest-path over
    `preconditions`; else fall back to `order`. Steps that SHARE a rank are UNORDERED
    (interchangeable) -- e.g. the three spice/chocolate adds in hot chocolate, which each only
    require the first microwave while `mix` requires all three. Cross-rank order is still
    enforced; only within-rank order is free."""
    if steps and all('rank' in s for s in steps):
        return {s['step_id']: s['rank'] for s in steps}
    by_id = {s['step_id']: s for s in steps}
    if any(s.get('preconditions') for s in steps):
        memo = {}
        def r(sid):
            if sid in memo:
                return memo[sid]
            s = by_id.get(sid)
            preds = [p for p in (s.get('preconditions') or []) if p in by_id] if s else []
            memo[sid] = 0 if not preds else 1 + max(r(p) for p in preds)
            return memo[sid]
        return {s['step_id']: r(s['step_id']) for s in steps}
    return {s['step_id']: s.get('order', 0) for s in steps}


def build_user_prompt(task, history, completed_steps):
    """history: prior VLM responses {t, step, status, evidence} (most recent last).
    completed_steps: step dicts in fully-completed RANK GROUPS — shown so the model never
    regresses to an earlier group. Steps sharing a rank are interchangeable (any order)."""
    rank = step_ranks(task['steps'])
    groups = {}
    for s in sorted(task['steps'], key=lambda s: (rank[s['step_id']], s.get('order', 0))):
        groups.setdefault(rank[s['step_id']], []).append(s)
    blocks = []
    for rk in sorted(groups):
        grp = groups[rk]
        if len(grp) == 1:
            blocks.append(f"  step_id={grp[0]['step_id']}: {grp[0]['instruction']}")
        else:
            inner = "\n".join(f"      step_id={s['step_id']}: {s['instruction']}" for s in grp)
            blocks.append(f"  [ANY ORDER — do all of these before moving on:\n{inner}\n  ]")
    steps = "\n".join(blocks)
    if completed_steps:
        done = "\n".join(f"  step_id={s['step_id']}: {s['instruction']}" for s in completed_steps)
        order_rule = (f"\n\nAlready completed (do NOT pick any of these again):\n{done}\n"
                      "Follow the groups top-to-bottom; steps inside an [ANY ORDER] group can be "
                      "done in any order. Pick a step from the current or a later group, never earlier.")
    else:
        order_rule = ("\n\nNothing is completed yet. Follow the groups top-to-bottom; steps inside "
                      "an [ANY ORDER] group can be done in any order. Begin at the first group.")
    if history:
        obs = "\n".join(
            f"  t={h['t']:.0f}s: step_id={h['step']} ({h['status']}) — {h['evidence']}"
            for h in history)
    else:
        obs = "  (none yet)"
    return (f"Guideline (recipe: {task['title']}). Steps grouped by stage:\n{steps}"
            f"{order_rule}\n\n"
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


def sample_frames_1fps(cap, fps, t_end, interval, max_frames=16, max_dim=768, jpeg_q=85,
                       sample_fps=1.0):
    """`sample_fps` frames per second over [t_end - interval, t_end], capped at max_frames.
    Default sample_fps=1.0 == one frame per second (the T1 baseline behavior)."""
    n = min(max_frames, max(1, int(round(interval * sample_fps))))
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
    def __init__(self, timeout=120, retries=2, backoff_s=2.0):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + ("" if base.endswith("/chat/completions")
                           else "/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.timeout = timeout            # per-attempt read timeout (s)
        self.retries = retries            # extra attempts after the first on transient errors
        self.backoff_s = backoff_s        # linear backoff between attempts
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
        # Retry on transient server/network errors (chiefly ReadTimeout, which otherwise
        # leaves a dead 'other' window). Backoff between attempts; raise the last error.
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


def run_one(video, task, backend, interval, max_seconds=None, verbose=True,
            trace_dir=None, rid=None, history_k=5, debug_dir=None):
    valid = {str(s['step_id']): s['step_id'] for s in task['steps']}    # model-out -> canonical
    rank = step_ranks(task['steps'])                                    # canonical -> partial-order rank
    steps_sorted = sorted(task['steps'], key=lambda s: s['order'])
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

    # debug = full per-call VLM input (system+user prompt, image metadata) + output (raw+enforced)
    dbg = None
    if debug_dir is not None and rid is not None:
        (Path(debug_dir) / "vlm_debug").mkdir(parents=True, exist_ok=True)
        dbg = open(Path(debug_dir) / "vlm_debug" / f"{rid}.jsonl", "w")

    stage_intervals, latencies = [], []
    frames_sent = parse_fail = 0
    history = []                          # prior VLM responses {t, step, status, evidence}
    high_water_rank = -1                  # highest rank GROUP reached; no regress across groups
    last_step = None                      # most recent accepted step (clamp target on regression)

    t = interval
    while t <= end_t + 1e-6:
        jpegs = sample_frames_1fps(cap, fps, t, interval)
        if not jpegs:
            break
        frames_sent += len(jpegs)
        ctx = history if history_k <= 0 else history[-history_k:]   # prior responses shown
        completed = [s for s in steps_sorted if rank[s['step_id']] < high_water_rank]  # whole groups done
        prompt = build_user_prompt(task, ctx, completed)
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

        # Enforce the recipe's PARTIAL order: no regression to an earlier rank group, but steps
        # within a group (e.g. the three adds) are interchangeable. A backward prediction is
        # clamped to the most recent accepted step; same-group or later predictions advance.
        step_raw = step
        clamped = False
        if step in rank:
            rr = rank[step]
            if rr < high_water_rank:
                step = last_step if last_step is not None else step
                clamped = True
            else:
                high_water_rank = rr
                last_step = step

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
                "prev_responses": ctx, "completed_steps": [s['step_id'] for s in completed],
                "pred_step_raw": step_raw, "pred_step": step, "clamped": clamped,
                "high_water_rank": high_water_rank,
                "pred_status": (parsed or {}).get("status"), "pred_evidence": (parsed or {}).get("evidence"),
                "raw": raw, "error": err, "latency_s": round(lat, 2)}) + "\n")
            trace.flush()

        if dbg is not None:                            # full VLM call I/O for debugging
            dbg.write(json.dumps({
                "call": len(latencies), "t": round(t, 1),
                "request": {"model": backend.model, "temperature": 0.0,
                            "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt,
                            "n_images": len(jpegs), "image_bytes": [len(j) for j in jpegs]},
                "response": {"raw": raw, "error": err, "latency_s": round(lat, 2),
                             "parsed": parsed, "step_raw": step_raw, "step_enforced": step,
                             "clamped": clamped, "high_water_rank": high_water_rank,
                             "status": (parsed or {}).get("status"),
                             "evidence": (parsed or {}).get("evidence")}}) + "\n")
            dbg.flush()

        history.append({"t": round(t, 1), "step": step,
                        "status": (parsed or {}).get("status") or "?",
                        "evidence": (parsed or {}).get("evidence") or ""})
        if verbose:
            st = parsed.get("status") if parsed else "PARSE_FAIL"
            print(f"  t={t:6.0f}s  step={step}  ({st})")
        t += interval

    cap.release()
    if trace is not None:
        trace.close()
    if dbg is not None:
        dbg.close()
    n = len(latencies)
    return {
        "stage_intervals": stage_intervals, "events": [], "escalation_requests": [],
        "cost": {"vlm_calls": n, "frames_sent": frames_sent,
                 "vlm_latency_total_s": round(float(np.sum(latencies)), 2) if latencies else 0.0,
                 "compute_s": 0.0},
        "_meta": {"model": backend.model, "interval_s": interval, "fps_sampling": 1.0,
                  "evaluated_s": round(end_t, 1), "parse_failures": parse_fail,
                  "traced": trace_dir is not None, "debugged": debug_dir is not None},
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
    ap.add_argument("--history", type=int, default=5,
                    help="# of prior VLM responses (timestamp+step+evidence) shown in the prompt; 0=all")
    ap.add_argument("--debug", action="store_true",
                    help="record every VLM call's full input (system+user prompt, image metadata) and "
                         "output (raw+parsed+enforced step) to <out>/<arm>/vlm_debug/<rid>.jsonl")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-attempt VLM read timeout (s)")
    ap.add_argument("--retries", type=int, default=2,
                    help="extra VLM attempts after the first on transient errors (timeout/conn/HTTP); 0=off")
    a = ap.parse_args()

    backend = Qwen(timeout=a.timeout, retries=a.retries)
    outd = BASE / a.out_dir / a.arm
    outd.mkdir(parents=True, exist_ok=True)
    tdir = outd if a.trace else None
    ddir = outd if a.debug else None

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
                          a.max_seconds, verbose=False, trace_dir=tdir, rid=rid, history_k=a.history,
                          debug_dir=ddir)
            res["recording"], res["arm"] = rid, a.arm
            outp.write_text(json.dumps(res, indent=1))
    else:
        if not (a.video and a.task):
            raise SystemExit("single mode needs --video and --task (or use --corpus)")
        rid = Path(a.video).stem
        task = json.loads(Path(a.task).read_text())
        res = run_one(a.video, task, backend, a.interval, a.max_seconds,
                      trace_dir=tdir, rid=rid, history_k=a.history, debug_dir=ddir)
        res["recording"], res["arm"] = rid, a.arm
        (outd / f"{rid}.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {outd / (rid + '.json')}  ({res['cost']['vlm_calls']} calls, "
              f"{res['_meta']['parse_failures']} parse-fails)")


if __name__ == "__main__":
    main()
