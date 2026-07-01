#!/usr/bin/env python3
"""Tier-1 baseline VARIANT: COMPLETION-CRITERIA STATE MACHINE (T1 only), RGB-only.

The counterpart to eval/baseline_t1_step.py (current-step RECOGNITION). Instead of asking
"which step is happening now?" every tick, this holds a POINTER at the step the user should be
on and asks one yes/no question: "has the CURRENT step been completed?" The pointer advances
by at most one step per tick, and ONLY when the gate fires `done=true`. The predicted step for
the window [t-interval, t) is whatever the pointer is on during that window.

This is the textbook completion-gated state machine whose weakness the head-to-head is meant to
measure: a single missed completion (false `done=false`) is an ABSORBING error -- the pointer
never advances and every later window is mislabeled (a "stall"). It also follows the recipe's
LINEAR `order` chain (not the DAG partial order the recognition arm uses), so a user who does
steps out of order stalls the gate -- by design, so the cost of rigidity is visible, not hidden.

One VLM call per tick (cost-parity with the recognition arm). `--lookahead K` relaxes the pure
machine: if the focal step's gate is `done=false`, also probe the next K steps and jump the
pointer to the furthest one reported complete (an "escape hatch"; default 0 = pure machine).

Metrics added to _meta: reached_end, final_step_idx, n_steps, advances, max_dwell_ticks,
stalled (reached_end is False) -- the stall rate is the number that confirms or kills the
"recognition beats criteria" judgement.

Writes the same unified per-recording JSON eval/eval_score_corpus.py reads:
  <out-dir>/<arm>/<rid>.json  {stage_intervals, events:[], escalation_requests:[], cost}

Usage mirrors baseline_t1_step.py (--video/--task or --corpus, --interval, --arm, --trace,
--debug, --history). Example:
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_t1_criteria.py --corpus --interval 10 --arm criteria_i10
"""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np

# Reuse the recognition baseline's client, frame sampler, JSON parsing, and corpus mapping so
# the two arms differ ONLY in the prompt/control loop -- a controlled comparison.
from baseline_t1_step import (
    BASE, Qwen, sample_frames_1fps, safe_json, recipe_stem_by_recording,
)

CRIT_DIR = BASE / 'tasks/cc4d_probe'


def load_completion_criteria(stem):
    """step_id -> completion CLAIM, from the generated criteria artifact
    tasks/cc4d_probe/<stem>.generated.criteria.json (Stage-1 node form; firewall-clean:
    recipe DAG + step text only). The `completion.claim` is the explicit done-condition the
    gate checks. Missing file / node -> empty dict; the gate then falls back to the step
    instruction's end-state."""
    f = CRIT_DIR / f"{stem}.generated.criteria.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out = {}
    for n in d.get('nodes', []):
        claim = (n.get('completion') or {}).get('claim')
        if claim:
            out[n['step_id']] = claim
    return out


SYSTEM_PROMPT = (
    "You are a procedural-cooking progress monitor running a step-by-step checklist. At each "
    "moment you are TOLD the single current step the user is expected to be working on. Your only "
    "job is to decide whether THAT current step has been COMPLETED in the frames -- judge the "
    "visible end-state, not the user's intent and not any other step. The frames are consecutive "
    "snapshots ending at the current moment.\n\n"
    "Answer done=true ONLY if the current step's described end-state is clearly achieved in the "
    "frames. Answer done=false if it has not been started or is still in progress.\n\n"
    "The evidence field MUST be a single short sentence of at most 20 words.\n\n"
    "Return JSON only, exactly this schema, no prose:\n"
    '{"done": <true|false>, "evidence": "<one short sentence, <=20 words>"}'
)


def build_user_prompt(task, steps_seq, ptr, history, crit=None, lookahead=0):
    """steps_seq: steps in linear `order`. ptr: index of the focal (current) step.
    history: prior gate decisions {t, step, done, evidence} (most recent last).
    crit: step_id -> completion CLAIM (the generated done-condition); falls back to the
    step instruction's end-state when absent."""
    crit = crit or {}
    lines = []
    for i, s in enumerate(steps_seq):
        if i < ptr:
            mark = "[done]"
        elif i == ptr:
            mark = "[CURRENT]"
        elif lookahead and i <= ptr + lookahead:
            mark = "[next]"
        else:
            mark = ""
        lines.append(f"  step_id={s['step_id']}: {s['instruction']} {mark}".rstrip())
    steps_block = "\n".join(lines)
    focal = steps_seq[ptr]
    if history:
        obs = "\n".join(
            f"  t={h['t']:.0f}s: current=step_id={h['step']} done={h['done']} — {h['evidence']}"
            for h in history)
    else:
        obs = "  (none yet)"
    focal_crit = crit.get(focal['step_id'])
    crit_line = (f"\nCompletion criterion: {focal_crit}" if focal_crit
                 else "\nCompletion criterion: the step's described end-state is visibly achieved.")
    ask = (f"CURRENT step to check (step_id={focal['step_id']}): {focal['instruction']}{crit_line}\n"
           f"Has THIS step been completed -- the completion criterion above satisfied -- in the "
           f"most recent frames? Set done=true only if finished; done=false if not yet started or "
           f"still in progress. Judge only this step, not earlier or later ones.")
    if lookahead:
        nxt = [steps_seq[ptr + k] for k in range(1, lookahead + 1) if ptr + k < len(steps_seq)]
        if nxt:
            extra = "; ".join(f"step_id={s['step_id']} ({s['instruction']})" for s in nxt)
            ask += (f"\n\nIf and only if the CURRENT step is already done, you may instead report "
                    f"the furthest of these later steps that is ALSO already complete by setting "
                    f"done=true (we will advance to it): {extra}. Otherwise judge only the current step.")
    return (f"Checklist (recipe: {task['title']}). Steps in order:\n{steps_block}\n\n"
            f"Your previous checks (most recent last):\n{obs}\n\n"
            f"Sensory Context: the images are the most recent frames. {ask}")


class CritQwen(Qwen):
    """Same HTTP client as the recognition arm but with the completion-criteria system prompt."""
    def call(self, jpegs, user_prompt):
        import base64, os, requests
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": user_prompt})
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 1200,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}]}
        if os.getenv("QWEN_NO_THINK"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["max_tokens"] = 256
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
            trace_dir=None, rid=None, history_k=5, debug_dir=None, lookahead=0, crit=None):
    crit = crit or {}
    steps_seq = sorted(task['steps'], key=lambda s: s['order'])   # LINEAR order chain (the gate's spine)
    n_steps = len(steps_seq)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end_t = min(duration, max_seconds) if max_seconds else duration

    trace, frames_root = None, None
    if trace_dir is not None and rid is not None:
        frames_root = Path(trace_dir) / "frames" / rid
        frames_root.mkdir(parents=True, exist_ok=True)
        (Path(trace_dir) / "traces").mkdir(parents=True, exist_ok=True)
        trace = open(Path(trace_dir) / "traces" / f"{rid}.jsonl", "w")
    dbg = None
    if debug_dir is not None and rid is not None:
        (Path(debug_dir) / "vlm_debug").mkdir(parents=True, exist_ok=True)
        dbg = open(Path(debug_dir) / "vlm_debug" / f"{rid}.jsonl", "w")

    stage_intervals, latencies = [], []
    frames_sent = parse_fail = 0
    history = []                  # prior gate decisions {t, step, done, evidence}
    ptr = 0                       # index into steps_seq -- the state-machine pointer
    advances = 0
    dwell = 0                     # consecutive ticks the pointer has sat on this focal step
    max_dwell = 0

    t = interval
    while t <= end_t + 1e-6:
        jpegs = sample_frames_1fps(cap, fps, t, interval)
        if not jpegs:
            break
        frames_sent += len(jpegs)
        done_machine = ptr >= n_steps
        focal_idx = min(ptr, n_steps - 1)
        focal = steps_seq[focal_idx]
        # Window is labeled with the step the pointer is on DURING the window.
        win_step = focal['step_id']

        parsed, raw, err, lat = None, "", None, 0.0
        gate_done = False
        jump_to = None
        if not done_machine:
            ctx = history if history_k <= 0 else history[-history_k:]
            prompt = build_user_prompt(task, steps_seq, ptr, ctx, crit, lookahead)
            t0 = time.time()
            try:
                raw = backend.call(jpegs, prompt); err = None
            except Exception as e:
                raw, err = "", f"{type(e).__name__}: {e}"
            lat = time.time() - t0
            latencies.append(lat)
            parsed = safe_json(raw)
            if parsed is None:
                parse_fail += 1
            gate_done = bool(parsed.get("done")) if parsed else False
            # --lookahead: if a later [next] step is reported complete, jump the pointer to it.
            if lookahead and parsed:
                jt = parsed.get("jump_to_step_id")
                if jt is not None:
                    for k in range(1, lookahead + 1):
                        if ptr + k < n_steps and str(steps_seq[ptr + k]['step_id']) == str(jt):
                            jump_to = ptr + k
        else:
            prompt = "(procedure complete — pointer past last step; no gate call)"

        stage_intervals.append({"stage": win_step,
                                "start_s": round(t - interval, 1), "end_s": round(min(t, end_t), 1)})

        # Advance logic: one step per tick on `done`, or jump on lookahead.
        advanced = False
        if not done_machine:
            if jump_to is not None and jump_to > ptr:
                ptr = jump_to + 1; advances += 1; advanced = True; dwell = 0
            elif gate_done:
                ptr += 1; advances += 1; advanced = True; dwell = 0
            else:
                dwell += 1; max_dwell = max(max_dwell, dwell)

        if trace is not None:
            frame_files = []
            for k, j in enumerate(jpegs):
                fn = f"t{int(round(t)):05d}_{k}.jpg"
                (frames_root / fn).write_bytes(j)
                frame_files.append(fn)
            trace.write(json.dumps({
                "t": round(t, 1), "start_s": round(t - interval, 1), "end_s": round(min(t, end_t), 1),
                "frame_files": frame_files, "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt,
                "prev_responses": (history if history_k <= 0 else history[-history_k:]),
                "focal_step": focal['step_id'], "pred_step": win_step, "pred_step_raw": win_step,
                "gate_done": gate_done, "advanced": advanced, "dwell_ticks": dwell, "ptr": ptr,
                "pred_status": ("done" if gate_done else "in progress"),
                "pred_evidence": (parsed or {}).get("evidence"),
                "raw": raw, "error": err, "latency_s": round(lat, 2)}) + "\n")
            trace.flush()
        if dbg is not None:
            dbg.write(json.dumps({
                "call": len(latencies), "t": round(t, 1),
                "request": {"model": backend.model, "temperature": 0.0,
                            "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt,
                            "n_images": len(jpegs), "image_bytes": [len(j) for j in jpegs]},
                "response": {"raw": raw, "error": err, "latency_s": round(lat, 2),
                             "parsed": parsed, "focal_step": focal['step_id'], "win_step": win_step,
                             "gate_done": gate_done, "advanced": advanced, "ptr": ptr,
                             "evidence": (parsed or {}).get("evidence")}}) + "\n")
            dbg.flush()

        history.append({"t": round(t, 1), "step": win_step,
                        "done": gate_done,
                        "evidence": (parsed or {}).get("evidence") or ""})
        if verbose:
            tag = "DONE→advance" if advanced else ("(machine complete)" if done_machine else "wait")
            print(f"  t={t:6.0f}s  focal={win_step}  {tag}  (ptr={ptr}/{n_steps})")
        t += interval

    cap.release()
    if trace is not None:
        trace.close()
    if dbg is not None:
        dbg.close()
    n = len(latencies)
    reached_end = ptr >= n_steps
    return {
        "stage_intervals": stage_intervals, "events": [], "escalation_requests": [],
        "cost": {"vlm_calls": n, "frames_sent": frames_sent,
                 "vlm_latency_total_s": round(float(np.sum(latencies)), 2) if latencies else 0.0,
                 "compute_s": 0.0},
        "_meta": {"model": backend.model, "interval_s": interval, "fps_sampling": 1.0,
                  "evaluated_s": round(end_t, 1), "parse_failures": parse_fail,
                  "mode": "completion_criteria", "lookahead": lookahead,
                  "criteria_steps": sum(1 for s in steps_seq if s['step_id'] in crit),
                  "n_steps": n_steps, "final_step_idx": ptr, "advances": advances,
                  "reached_end": reached_end, "stalled": not reached_end,
                  "max_dwell_ticks": max_dwell,
                  "traced": trace_dir is not None, "debugged": debug_dir is not None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--task")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--arm", default="criteria")
    ap.add_argument("--out-dir", default="experiments/t1_baseline")
    ap.add_argument("--video-dir", default="data/videos_360p")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--history", type=int, default=5,
                    help="# of prior gate decisions shown in the prompt; 0=all")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="escape hatch: also probe the next K steps and jump to the furthest "
                         "reported complete; 0 = pure sequential state machine")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    backend = CritQwen(timeout=a.timeout, retries=a.retries)
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
            crit = load_completion_criteria(stem)
            print(f"[{i}/{len(todo)}] {rid} ({stem})  criteria for {len(crit)} steps")
            res = run_one(BASE / a.video_dir / f"{rid}.mp4", task, backend, a.interval,
                          a.max_seconds, verbose=False, trace_dir=tdir, rid=rid,
                          history_k=a.history, debug_dir=ddir, lookahead=a.lookahead, crit=crit)
            res["recording"], res["arm"] = rid, a.arm
            outp.write_text(json.dumps(res, indent=1))
    else:
        if not (a.video and a.task):
            raise SystemExit("single mode needs --video and --task (or use --corpus)")
        rid = Path(a.video).stem
        task = json.loads(Path(a.task).read_text())
        crit = load_completion_criteria(Path(a.task).stem)
        res = run_one(a.video, task, backend, a.interval, a.max_seconds,
                      trace_dir=tdir, rid=rid, history_k=a.history, debug_dir=ddir,
                      lookahead=a.lookahead, crit=crit)
        res["recording"], res["arm"] = rid, a.arm
        (outd / f"{rid}.json").write_text(json.dumps(res, indent=1))
        m = res["_meta"]
        print(f"wrote {outd / (rid + '.json')}  ({res['cost']['vlm_calls']} calls, "
              f"{m['parse_failures']} parse-fails, reached_end={m['reached_end']}, "
              f"final {m['final_step_idx']}/{m['n_steps']}, max_dwell={m['max_dwell_ticks']})")


if __name__ == "__main__":
    main()
