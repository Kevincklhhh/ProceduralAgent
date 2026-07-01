#!/usr/bin/env python3
"""Qualcomm zero-shot BASELINE arm — faithful to the paper's Appendix F protocol.

Reproduces the way the Qualcomm Interactive Cooking paper (NeurIPS'25) ran turn-based MLLMs
as baselines: a fixed 5 s tick, and at each tick a TWO-STAGE gated query anchored to the
current recipe instruction:
  1. completion check  (verbatim Appendix-F prompt)  -> yes/no
  2. mistake check     (verbatim Appendix-F prompt, seeded with a per-step "mistake summary")
The [mistake summary] role is filled by OUR probe-derived criteria checks
(tasks/cc4d_probe/<stem>.generated.criteria.json) -- the analog of the paper's Qwen-generated
candidate-mistake list (see memory: qualcomm-paper-baseline-protocol).

Two modes (same tick loop; only the current-step pointer differs):
  --mode streaming   pointer advances when the completion check says "yes" (SELF-TRACKED) ->
                     completion errors propagate. (paper Tables 3-4)
  --mode turnbased   pointer = the GT step active at t (ORACLE step boundaries); no completion
                     gating -> each step is judged with the correct instruction, isolating
                     mistake detection. (paper Table 5)

Emits the canonical unified arm format (docs/REMINDER_EVALUATION.md §4) so it scores through
BOTH eval/eval_score_corpus.py and the Qualcomm profile (eval/qualcomm_eval.py):
  <out>/<arm>/<rid>.json = {recording, arm, stage_intervals, events:[{t,class,subtype,message}],
                            escalation_requests:[], cost, _meta}

Subtype note: the Appendix mistake prompt returns free-text yes/feedback (untyped, as in the
paper). We tag each fired event with a subtype by matching the feedback against the step's
checks (fallback: the step's first check subtype). Qualcomm scoring is untyped so this only
affects our FA-1 typed view, not the Qualcomm numbers.

Usage:
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_qualcomm_zeroshot.py --mode streaming --rids 8_50,10_50 \
    --out-dir experiments/qualcomm_run --arm qwen36_zs
"""
import argparse, json, os, sys, time, base64
import cv2, requests, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_t1_step as t1  # reuse sample_frames_1fps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC4D = os.path.join(BASE, "tasks", "cc4d")
PROBE = os.path.join(BASE, "tasks", "cc4d_probe")
STEP_ANN = os.path.join(BASE, "data/cc4d/annotations/annotation_json/step_annotations.json")
TIMELINE = os.path.join(BASE, "data/qualcomm_interactive_cooking/qualcomm_timeline.json")
SUBSET = {"technique", "preparation", "measurement", "temperature", "timing"}

# ---- verbatim Appendix-F prompts (Gemini/Qwen variant) -------------------------------------
COMPLETION_PROMPT = (
    "You are an expert cooking assistant helping a person cook. The person is provided with an "
    "instruction and your task is to check if the instruction has been completed.\n\n"
    "##INSTRUCTIONS:\n"
    "The person has been instructed to: {instruction}.\n"
    'If the person has completed the instruction answer "yes" else answer "no". '
    "DO NOT OUTPUT ANY OTHER TEXT.")

MISTAKE_PROMPT = (
    "You are an expert cooking assistant who is observing a person who is provided with step by "
    "step instructions for cooking. You should look out for mistakes made by the person.\n\n"
    "##INSTRUCTIONS:\n"
    "The person is trying to complete the following instruction: {instruction}.\n"
    "This is how you can check for mistakes: {mistake_summary}.\n"
    "Your task is to check if the person has already made a mistake.\n"
    "Note that the person may not have completed the provided instruction, that is, the person "
    "may have only partially completed the provided instruction.\n"
    'The answer should be "yes" or "no". In case of yes, please provide a concise feedback to '
    "the person describing the mistake (i.e. Yes. <feedback>.). Directly address the person.")

# ---- closed-set variant: the mistake check is reformulated as a multiple-choice classification
# over the step's known candidate mistakes (the generated criteria). The model picks the
# number(s) of any mistake it observes, or 0 for none. The subtype is then read DIRECTLY from
# the chosen criterion (no post-hoc token-overlap guess) -> a clean type-matched prediction.
CLOSEDSET_PROMPT = (
    "You are an expert cooking assistant observing a person who is following step-by-step "
    "cooking instructions. You should look out for mistakes made by the person.\n\n"
    "##INSTRUCTIONS:\n"
    "The person is currently trying to: {instruction}.\n"
    "Below is a numbered list of the specific mistakes that can happen at this step:\n"
    "{menu}\n"
    "Look at the frames and decide which, if any, of these listed mistakes the person has made. "
    "Note the person may have only partially completed the instruction.\n"
    'Answer with the number(s) of every listed mistake that has occurred, comma-separated '
    '(e.g. "1" or "1,3"). If none of the listed mistakes has occurred, answer "0". '
    "DO NOT OUTPUT ANY OTHER TEXT.")


def closedset_menu(checks):
    return "\n".join(f"{i+1}. {claim}" for i, (sub, claim) in enumerate(checks))


def parse_choices(txt, n):
    """Extract chosen 1..n indices from the model's answer; ignore 0 and out-of-range."""
    import re
    nums = {int(x) for x in re.findall(r"\d+", txt or "")}
    return sorted(i for i in nums if 1 <= i <= n)


class QwenText:
    """Same endpoint as baseline_t1_step.Qwen but WITHOUT forced json_object, so the verbatim
    yes/no Appendix prompts can be used and parsed as plain text."""
    def __init__(self, timeout=120, retries=2, backoff_s=2.0, enable_thinking=False,
                 max_tokens=None):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + ("" if base.endswith("/chat/completions")
                           else "/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.timeout, self.retries, self.backoff_s = timeout, retries, backoff_s
        self.enable_thinking = enable_thinking
        # thinking needs room for the reasoning trace before the answer; no-think stays tight
        self.max_tokens = max_tokens if max_tokens is not None else (2048 if enable_thinking else 120)
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, jpegs, prompt):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in jpegs]
        content.append({"type": "text", "text": prompt})
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens,
                   "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                   "messages": [{"role": "user", "content": content}]}
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                c = msg.get("content")
                txt = ("".join(p.get("text", "") for p in c if isinstance(p, dict))
                       if isinstance(c, list) else (c or ""))
                # when thinking is on, strip the reasoning trace so digit/yes parsing sees only
                # the final answer (some servers inline <think>...</think> in content)
                if "</think>" in txt:
                    txt = txt.rsplit("</think>", 1)[-1]
                return txt
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last


def parse_yes(txt):
    return (txt or "").strip().lower().lstrip('"\'*. ').startswith("yes")


def load_checks(stem):
    """step_id -> [(subtype, claim)] from the generated criteria (probe-derived)."""
    f = os.path.join(PROBE, stem + ".generated.criteria.json")
    out = {}
    if not os.path.exists(f):
        return out
    for n in json.load(open(f))["nodes"]:
        out[n["step_id"]] = [(c["reminder"], c["claim"]) for c in n.get("checks", [])
                             if c["reminder"] in SUBSET]
    return out


def mistake_summary(checks):
    if not checks:
        return "watch for any deviation from the instruction (wrong amount, wrong technique, wrong timing)"
    return "; ".join(f"{i+1}. ({sub}) {claim}" for i, (sub, claim) in enumerate(checks))


def cls_of(sub):
    return "parameter_violation" if sub == "timing" else "execution_error"


def pick_subtype(feedback, checks):
    """Tag a fired mistake with a subtype by token overlap with the checks (fallback: first)."""
    if not checks:
        return ("execution_error", "technique")
    fb = (feedback or "").lower()
    best, score = checks[0], -1
    for sub, claim in checks:
        s = sum(1 for w in set(claim.lower().split()) if len(w) > 4 and w in fb)
        if s > score:
            best, score = (sub, claim), s
    sub = best[0]
    cls = "parameter_violation" if sub == "timing" else "execution_error"
    return cls, sub


def recipe_steps(stem):
    task = json.load(open(os.path.join(CC4D, f"{stem}.json")))
    return sorted(task["steps"], key=lambda s: s.get("order", 0))


def oracle_spans(rid):
    ann = json.load(open(STEP_ANN))
    r = ann.get(rid, {})
    return [(s["step_id"], float(s["start_time"]), float(s["end_time"]))
            for s in r.get("steps", []) if s["start_time"] >= 0 and s["end_time"] >= 0]


GT_EXEC = os.path.join(BASE, "data", "cc4d_proactive")


def gt_reminders(rid):
    """GT execution reminders for a recording (ORACLE side — used only by --gate)."""
    p = os.path.join(GT_EXEC, rid + ".json")
    return json.load(open(p)).get("reminders", []) if os.path.exists(p) else []


def run_one(rid, stem, mode, video, interval, sample_fps, max_frames, prompt_style="freetext",
            window=None, enable_thinking=False, gate="none", gate_window=30.0, backend="qwen",
            verbose=True):
    window = window if window is not None else interval   # visual context per call (decoupled from tick)
    steps = recipe_steps(stem)
    by_id = {s["step_id"]: s for s in steps}
    checks_by_id = load_checks(stem)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    # thinking can take far longer than the 120s default read timeout -> bump it
    if backend == "gpt54":
        from gpt_client import GPT54Text
        client = GPT54Text(timeout=300)        # reasoning model; gpt-5.4 endpoint
    else:
        client = QwenText(timeout=300 if enable_thinking else 120, enable_thinking=enable_thinking)

    spans = oracle_spans(rid) if mode == "turnbased" else None
    rems = gt_reminders(rid) if gate != "none" else []        # ORACLE gate input
    error_steps = {r.get("anchor_step") for r in rems}        # GT steps that carry an error

    stage_intervals, events, calls = [], [], []
    fired_steps = set()
    n_calls = n_frames = 0
    lat = 0.0
    t0 = time.time()

    def frames_window(t_end, win):
        # frames = sample_fps over the window, but at least max_frames and at most 32 (a wide
        # gate window subsamples instead of sending 60+ frames -> keeps calls server-friendly)
        wf = min(32, max(max_frames, int(round(win * sample_fps))))
        return t1.sample_frames_1fps(cap, fps, t_end, win, max_frames=wf, sample_fps=sample_fps)

    def probe(sid, jpegs, t_event, ws, we):
        """Run the mistake check for step `sid` over `jpegs`; append events + a call-trace
        record; return the list of fired events. Honors prompt_style."""
        nonlocal n_calls, lat
        instr = by_id.get(sid, {}).get("instruction", f"step {sid}")
        checks = checks_by_id.get(sid) or []
        fired = []
        tc = time.time()
        if prompt_style == "closedset":
            if not checks:
                return fired
            ans = client.call(jpegs, CLOSEDSET_PROMPT.format(instruction=instr,
                                                             menu=closedset_menu(checks)))
            n_calls += 1; dt = time.time() - tc; lat += dt
            chosen = parse_choices(ans, len(checks)); seen = set()
            for idx in chosen:
                sub, claim = checks[idx - 1]
                if sub in seen:
                    continue
                seen.add(sub)
                fired.append({"t": round(t_event, 1), "class": cls_of(sub), "subtype": sub,
                              "message": claim[:300], "step_id": sid, "choice": idx,
                              "raw": ans.strip()[:60]})
        else:  # freetext
            ans = client.call(jpegs, MISTAKE_PROMPT.format(instruction=instr,
                                                           mistake_summary=mistake_summary(checks)))
            n_calls += 1; dt = time.time() - tc; lat += dt
            if parse_yes(ans):
                cls, sub = pick_subtype(ans, checks)
                fired.append({"t": round(t_event, 1), "class": cls, "subtype": sub,
                              "message": ans.strip()[:300], "step_id": sid})
        calls.append({"t": round(t_event, 1), "kind": "mistake", "step_id": sid,
                      "win_start": round(max(0.0, ws), 1), "win_end": round(we, 1),
                      "n_frames": len(jpegs), "latency_s": round(dt, 1),
                      "fired": [e["subtype"] for e in fired], "answer": ans.strip()[:80]})
        return fired

    # ---- GATE: reminders -- one probe per GT reminder, frames centered on the moment ----------
    if gate == "reminders":
        for r in sorted(rems, key=lambda x: x["t"]):
            sid = r.get("anchor_step"); te = float(r["t"])
            ws, we = te - gate_window, min(dur, te + gate_window)
            jpegs = frames_window(we, 2 * gate_window); n_frames += len(jpegs)
            for e in probe(sid, jpegs, te, ws, we):
                events.append(e)
            if verbose:
                print(f"  reminder @{te:6.1f} step {sid} [{r.get('subtype')}] -> "
                      f"{[e['subtype'] for e in events if e['step_id']==sid][-3:]}")
        # one event per (step, subtype): drop duplicate probes' repeats (would be self-FP)
        seen = {}
        for e in events:
            seen.setdefault((e["step_id"], e["subtype"]), e)
        events = sorted(seen.values(), key=lambda e: e["t"])
        seg = [{"stage": s, "start_s": round(a, 1), "end_s": round(b, 1)}
               for (s, a, b) in (spans or [])]
        cap.release()
        return _pack(rid, seg, events, calls, n_calls, n_frames, lat, mode, prompt_style,
                     interval, window, enable_thinking, sample_fps, gate, gate_window, t0, backend)

    # ---- streaming / turnbased tick loop (gate none or errorsteps) ----------------------------
    def current_step(t, ptr):
        if mode == "streaming":
            return steps[ptr]["step_id"] if ptr < len(steps) else None
        active = [sid for (sid, s, e) in spans if s <= t < e]   # oracle
        return active[-1] if active else None

    ptr = 0
    cur_open = None  # (step_id, start_s)
    t = interval
    while t <= min(dur, 10_000) + 1e-6:
        sid = current_step(t, ptr)
        if sid is None:
            t += interval
            continue
        # maintain stage_intervals (start = 0.0 for the very first, else the boundary time t)
        if cur_open is None:
            cur_open = (sid, 0.0 if not stage_intervals else t)
        elif cur_open[0] != sid:
            stage_intervals.append({"stage": cur_open[0], "start_s": round(cur_open[1], 1),
                                    "end_s": round(t, 1)})
            cur_open = (sid, t)
        instr = by_id[sid].get("instruction", f"step {sid}")
        jpegs = frames_window(t, window)
        n_frames += len(jpegs)

        advanced = False
        if mode == "streaming":
            tc = time.time()
            comp = client.call(jpegs, COMPLETION_PROMPT.format(instruction=instr)); n_calls += 1
            dtc = time.time() - tc; lat += dtc
            yes = parse_yes(comp)
            calls.append({"t": round(t, 1), "kind": "completion", "step_id": sid,
                          "win_start": round(max(0.0, t - window), 1), "win_end": round(t, 1),
                          "n_frames": len(jpegs), "latency_s": round(dtc, 1),
                          "fired": ["complete"] if yes else [], "answer": (comp or "").strip()[:40]})
            if yes:
                stage_intervals.append({"stage": sid, "start_s": round(cur_open[1], 1), "end_s": round(t, 1)})
                ptr += 1
                cur_open = (steps[ptr]["step_id"], t) if ptr < len(steps) else None
                advanced = True
                if verbose:
                    print(f"  t={t:6.1f} step {sid} COMPLETE -> advance")
        # mistake check (skip the tick where we just completed; once per step occurrence)
        gated_out = gate == "errorsteps" and sid not in error_steps   # oracle: only error steps
        if (not advanced and not gated_out and sid not in fired_steps
                and checks_by_id.get(sid) is not None):
            fired = probe(sid, jpegs, t, max(0.0, t - window), t)
            if fired:
                fired_steps.add(sid)
                events.extend(fired)
                if verbose:
                    print(f"  t={t:6.1f} step {sid} MISTAKE [{','.join(e['subtype'] for e in fired)}]")
        t += interval

    if cur_open is not None:
        stage_intervals.append({"stage": cur_open[0], "start_s": round(cur_open[1], 1),
                                "end_s": round(min(dur, t), 1)})
    cap.release()
    return _pack(rid, stage_intervals, events, calls, n_calls, n_frames, lat, mode, prompt_style,
                 interval, window, enable_thinking, sample_fps, gate, gate_window, t0, backend)


def _pack(rid, seg, events, calls, n_calls, n_frames, lat, mode, prompt_style, interval, window,
          enable_thinking, sample_fps, gate, gate_window, t0, backend="qwen"):
    return {
        "recording": rid, "arm": None,
        "stage_intervals": seg, "events": events, "escalation_requests": [], "calls": calls,
        "cost": {"vlm_calls": n_calls, "frames_sent": n_frames,
                 "vlm_latency_total_s": round(lat, 1), "compute_s": 0.0},
        "_meta": {"backend": backend, "mode": mode, "prompt_style": prompt_style, "interval_s": interval,
                  "window_s": window, "enable_thinking": enable_thinking, "sample_fps": sample_fps,
                  "gate": gate, "gate_window_s": gate_window if gate == "reminders" else None,
                  "prompt_source": "closedset_criteria" if prompt_style == "closedset"
                  else "qualcomm_appendix_F", "mistake_summary_source": "criteria.generated",
                  "wall_s": round(time.time() - t0, 1)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["streaming", "turnbased"], required=True)
    ap.add_argument("--prompt-style", choices=["freetext", "closedset"], default="freetext",
                    help="freetext = paper Appendix-F yes/no; closedset = multiple-choice over "
                         "the step's candidate mistakes (clean type-matched prediction)")
    ap.add_argument("--rids", help="comma list of recording_ids; default = all test split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--video-dir", default=os.path.join(BASE, "data/videos_360p"))
    ap.add_argument("--out-dir", default="experiments/qualcomm_run")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--interval", type=float, default=5.0, help="tick seconds (paper: 5)")
    ap.add_argument("--window", type=float, default=None,
                    help="seconds of visual context sampled per call (default = interval); "
                         "raise (e.g. 12) for longer context without changing tick cadence")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable Qwen reasoning (slower; max_tokens 2048, timeout 300s)")
    ap.add_argument("--gate", choices=["none", "errorsteps", "reminders"], default="none",
                    help="ORACLE call-gating for fast recall probes (precision becomes dishonest): "
                         "errorsteps = mistake-check only on GT steps that carry an error; "
                         "reminders = one probe per GT reminder, frames centered on its moment")
    ap.add_argument("--gate-window", type=float, default=30.0,
                    help="half-width (s) of the clip centered on each reminder for --gate reminders")
    ap.add_argument("--backend", choices=["qwen", "gpt54"], default="qwen",
                    help="gpt54 = Azure-NAIRR GPT-5.4 (messi endpoint, api_key from .env)")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true", help="re-run even if output json exists")
    args = ap.parse_args()

    tl = json.load(open(TIMELINE))
    stem_of = __import__("qualcomm_adapter")._RECIPE_STEM
    if args.rids:
        rids = args.rids.split(",")
    elif args.split == "all":               # no train/test split — evaluate on every recording
        rids = list(tl.keys())
    else:
        rids = [v for v, d in tl.items() if d["split"] == args.split]
    out = os.path.join(BASE, args.out_dir, args.arm)
    os.makedirs(out, exist_ok=True)

    done = err = 0
    for i, rid in enumerate(rids):
        dst = os.path.join(out, rid + ".json")
        vid = os.path.join(args.video_dir, rid + ".mp4")
        if rid not in tl or not os.path.exists(vid):
            print(f"skip {rid} (no timeline/video)"); continue
        if not args.overwrite and os.path.exists(dst):     # resume: skip already-done
            print(f"[{i+1}/{len(rids)}] {rid} already done, skip"); done += 1; continue
        stem = stem_of.get(tl[rid]["recipe"], tl[rid]["recipe"])
        print(f"[{i+1}/{len(rids)}] {rid} ({stem}) mode={args.mode}", flush=True)
        try:                                                # fault-tolerant over long runs
            res = run_one(rid, stem, args.mode, vid, args.interval, args.sample_fps,
                          args.max_frames, prompt_style=args.prompt_style,
                          window=args.window, enable_thinking=args.enable_thinking,
                          gate=args.gate, gate_window=args.gate_window, backend=args.backend)
        except Exception as e:
            err += 1; print(f"    !! FAILED {rid}: {type(e).__name__}: {e}", flush=True); continue
        res["arm"] = args.arm
        json.dump(res, open(dst, "w"), indent=1)
        done += 1
        print(f"    -> {len(res['events'])} mistakes, {res['cost']['vlm_calls']} calls, "
              f"{res['_meta']['wall_s']}s", flush=True)
    print(f"wrote -> {out}  ({done} done, {err} failed)")


if __name__ == "__main__":
    main()
