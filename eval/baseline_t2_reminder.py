#!/usr/bin/env python3
"""T2 baseline — proactive-reminder ERROR DETECTION via a per-step VLM call.

Question this answers: given the recipe's anticipated checks, can the VLM SEE an execution
error when it occurs, and at what overhead? Metric starts binary (detected-or-not).

NOT a generalization result, by construction:
  * The window provider is `oracle-step-span`: frames are sampled over the GROUND-TRUTH step
    start->end. A deployed system has neither the step identity nor its boundaries for free;
    those must come from online step recognition (T1) or a cheap detector firing, with noise.
    So this measures the DETECTION CEILING + per-step overhead with segmentation error removed.
    The window provider is swappable (--window oracle | <future: t1 | detector>) so the
    generalization-capable version drops in without touching the prompt or scoring.
  * The criteria checks are probe-derived from these same recordings, so recall here is
    IN-SAMPLE. A clean eval would hold out recordings.

Runtime: one VLM call per (recording, step) that has checks; ALL of that step's checks are
merged into one prompt; strict-JSON out {subtype: {violated, evidence}}.

GT (answer key): data/cc4d_family_a/<rec>.json events, classes execution_error +
parameter_violation/timing (+ execution_error/temperature). order/missing_step excluded
(structural, runtime-handled). Per-step spans: CC4D step_annotations.json.

Usage:
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_t2_reminder.py --recipe spicedhotchocolate --vlm qwen --sample-fps 0.5 --max-frames 8
  python eval/baseline_t2_reminder.py --recipe spicedhotchocolate --vlm mock   # offline plumbing test
"""
import argparse, json, os, sys, time, glob, collections
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_t1_step as t1  # Qwen, sample_frames_1fps, safe_json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_ANN = os.path.join(BASE, "data/cc4d/annotations/annotation_json/step_annotations.json")
FAMILY_A = os.path.join(BASE, "data/cc4d_proactive")
CHECK_SUBS = {"measurement", "technique", "preparation", "timing", "temperature"}
STRUCTURAL = {"order", "missing_step"}

SYS = ("You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe and a "
       "list of checks (possible mistakes) for that step. For EACH check decide if the mistake is "
       "visible in the frames. Be conservative: only flag 'violated': true when you actually see "
       "evidence. Reply with a single JSON object mapping each check's subtype to "
       '{"violated": true|false, "evidence": "<short>"}.')


def load_criteria(recipe):
    f = os.path.join(BASE, "tasks/cc4d_probe", recipe + ".generated.criteria.json")
    d = json.load(open(f))
    out = {}  # step_id -> {instruction, checks:[{subtype,claim}]}
    for n in d["nodes"]:
        out[n["step_id"]] = {"instruction": n["instruction"],
                             "checks": [{"subtype": c["reminder"], "claim": c["claim"]}
                                        for c in n.get("checks", [])]}
    return out


def oracle_spans():
    """recording_id -> {step_id: (start,end)} from CC4D step annotations."""
    ann = json.load(open(STEP_ANN))
    out = {}
    for rid, r in ann.items():
        out[rid] = {s["step_id"]: (s["start_time"], s["end_time"]) for s in r["steps"]}
    return out


def load_gt(rid):
    """set of (step_id, subtype) execution-error targets for this recording (order/missing excluded)."""
    f = os.path.join(FAMILY_A, rid + ".json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    gt = set()
    # cc4d_proactive schema: flat `reminders` (execution mistakes only; order/missing already
    # excluded upstream). subtype is the scored key; anchor_step ties it to the step window.
    for e in d.get("reminders", d.get("events", [])):
        sub = e.get("subtype")
        if sub not in STRUCTURAL and sub in CHECK_SUBS and e.get("anchor_step") is not None:
            gt.add((e["anchor_step"], sub))
    return gt


def recordings_for(recipe):
    stem_by = t1.recipe_stem_by_recording()
    return sorted(rid for rid, s in stem_by.items() if s == recipe)


def build_prompt(step, checks):
    lines = [f"STEP: {step['instruction']}", "", "CHECKS:"]
    for c in checks:
        lines.append(f"- {c['subtype']}: {c['claim']}")
    lines.append("")
    lines.append('Return JSON: {"<subtype>": {"violated": bool, "evidence": str}, ...} for the subtypes above.')
    return "\n".join(lines)


SYS_TARGETED = ("You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe "
                "and ONE specific thing to watch for. Decide whether that mistake actually occurred "
                "in the frames. Be conservative: only say it is a mistake if you see evidence. "
                'Reply with a single JSON object {"is_mistake": true|false, "evidence": "<short>"}.')


def run_targeted(args, criteria, spans, recs, out_dir):
    """Easiest upper-bound: one VLM call per (recording, step, subtype) case, naming the single
    mistake to watch for; VLM says correct/mistake. Positives = GT errors (-> recall);
    matched negatives = same (step,subtype) where GT has no such error (-> false-alarm rate)."""
    client = t1.Qwen() if args.vlm == "qwen" else None
    if client:
        t1.SYSTEM_PROMPT = SYS_TARGETED
    # claim text per (step,subtype) from criteria; gather GT per recording
    claim = {(sid, c["subtype"]): c["claim"] for sid, node in criteria.items() for c in node["checks"]}
    gt = {rid: (load_gt(rid) or set()) for rid in recs}
    cells = sorted({(sid, sub) for rid in recs for (sid, sub) in gt[rid]})  # (step,subtype) that ever erred
    cases = []  # (rid, step, subtype, is_positive)
    for (sid, sub) in cells:
        pos = [rid for rid in recs if (sid, sub) in gt[rid]]
        neg = [rid for rid in recs if (sid, sub) not in gt[rid] and sid in spans.get(rid, {})]
        for rid in pos:
            cases.append((rid, sid, sub, True))
        for rid in neg[:args.neg_per_cell]:
            cases.append((rid, sid, sub, False))

    per = collections.defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    calls = []
    n_frames = 0
    t_start = time.time()
    caps = {}
    for (rid, sid, sub, is_pos) in cases:
        vid = os.path.join(args.video_dir, rid + ".mp4")
        if not os.path.exists(vid):
            continue
        if rid not in caps:
            caps[rid] = cv2.VideoCapture(vid)
        cap = caps[rid]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start, end = spans[rid][sid]
        jpegs = t1.sample_frames_1fps(cap, fps, t_end=end, interval=max(0.5, end - start),
                                      max_frames=args.max_frames, sample_fps=args.sample_fps)
        n_frames += len(jpegs)
        watch = claim.get((sid, sub), f"a {sub} mistake on this step")
        prompt = (f"STEP: {criteria[sid]['instruction']}\n\nWATCH FOR ({sub}): {watch}\n\n"
                  'Did THIS specific mistake occur in the frames? Reply {"is_mistake": bool, "evidence": str}.')
        if client:
            t0 = time.time()
            try:
                obj = t1.safe_json(client.call(jpegs, prompt)) or {}
            except Exception as e:
                obj = {}
            lat = time.time() - t0
        else:
            obj = {"is_mistake": is_pos, "evidence": "mock"}; lat = 0.0  # mock = perfect, for plumbing
        flag = bool(obj.get("is_mistake", False))
        if is_pos:
            per[sub]["TP" if flag else "FN"] += 1
        else:
            per[sub]["FP" if flag else "TN"] += 1
        calls.append({"rid": rid, "step_id": sid, "subtype": sub, "is_positive": is_pos,
                      "pred_mistake": flag, "n_frames": len(jpegs), "latency_s": round(lat, 2)})
    for c in caps.values():
        c.release()

    summary = {"recipe": args.recipe, "arm": args.vlm, "mode": "targeted", "window": args.window,
               "n_recordings": len(recs), "n_calls": len(calls), "n_frames": n_frames,
               "wall_s": round(time.time() - t_start, 1), "sample_fps": args.sample_fps,
               "max_frames": args.max_frames, "neg_per_cell": args.neg_per_cell,
               "per_subtype": {}, "_caveats": "EASIEST upper bound: oracle step window + single named GT "
               "mistake to verify (heavy GT leak by design). Measures pure VLM perception, not deployable."}
    tot = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for sub, c in sorted(per.items()):
        rec = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else None
        far = c["FP"] / (c["FP"] + c["TN"]) if (c["FP"] + c["TN"]) else None
        summary["per_subtype"][sub] = {**c, "recall": rec, "false_alarm_rate": far}
        for k in tot:
            tot[k] += c[k]
    R = tot["TP"] / (tot["TP"] + tot["FN"]) if (tot["TP"] + tot["FN"]) else None
    FAR = tot["FP"] / (tot["FP"] + tot["TN"]) if (tot["FP"] + tot["TN"]) else None
    summary["overall"] = {**tot, "recall": R, "false_alarm_rate": FAR}
    json.dump(calls, open(os.path.join(out_dir, "calls.jsonl"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json + calls.jsonl")


SYS_BINARY = ("You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe. "
              "Decide whether the person made ANY mistake while performing this step (e.g. wrong "
              "ingredient/container, wrong amount, poor technique/spill, wrong timing, wrong "
              "heat/power). Be conservative: only say mistake if you see evidence. Reply with a "
              'single JSON object {"is_mistake": true|false, "evidence": "<short>"}.')


def run_binary(args, criteria, spans, recs, out_dir):
    """Head-to-head control for `targeted`: generic 'is there ANY error in this step?' (no specific
    mistake named, no leak), one call per (rid,step), scored at STEP level. Uses the EXACT (rid,step)
    units the targeted run covered so only the QUESTION differs. Also reports the targeted run's
    step-level numbers (OR over its per-subtype calls) for a direct same-unit comparison."""
    tgt = os.path.join(BASE, "experiments/baseline_t2", f"{args.recipe}_qwen_targeted", "calls.jsonl")
    if not os.path.exists(tgt):
        raise SystemExit(f"need the targeted run first: {tgt} missing")
    tcalls = json.load(open(tgt))
    units = {}  # (rid,sid) -> {gt, tgt_pred}
    for c in tcalls:
        k = (c["rid"], c["step_id"])
        u = units.setdefault(k, {"gt": False, "tgt_pred": False})
        u["gt"] = u["gt"] or c["is_positive"]
        u["tgt_pred"] = u["tgt_pred"] or c["pred_mistake"]

    client = t1.Qwen() if args.vlm == "qwen" else None
    if client:
        t1.SYSTEM_PROMPT = SYS_BINARY
    calls = []
    n_frames = 0
    t0all = time.time()
    caps = {}
    for (rid, sid), u in units.items():
        vid = os.path.join(args.video_dir, rid + ".mp4")
        if not os.path.exists(vid):
            continue
        if rid not in caps:
            caps[rid] = cv2.VideoCapture(vid)
        cap = caps[rid]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start, end = spans[rid][sid]
        jpegs = t1.sample_frames_1fps(cap, fps, t_end=end, interval=max(0.5, end - start),
                                      max_frames=args.max_frames, sample_fps=args.sample_fps)
        n_frames += len(jpegs)
        prompt = (f"STEP: {criteria[sid]['instruction']}\n\n"
                  'Did the person make ANY mistake while performing this step? '
                  'Reply {"is_mistake": bool, "evidence": str}.')
        if client:
            t1c = time.time()
            try:
                obj = t1.safe_json(client.call(jpegs, prompt)) or {}
            except Exception:
                obj = {}
            lat = time.time() - t1c
        else:
            obj = {"is_mistake": u["gt"]}; lat = 0.0
        flag = bool(obj.get("is_mistake", False))
        calls.append({"rid": rid, "step_id": sid, "gt": u["gt"], "binary_pred": flag,
                      "targeted_pred": u["tgt_pred"], "n_frames": len(jpegs), "latency_s": round(lat, 2)})
    for c in caps.values():
        c.release()

    def score(pred_key):
        TP = sum(1 for c in calls if c["gt"] and c[pred_key])
        FN = sum(1 for c in calls if c["gt"] and not c[pred_key])
        FP = sum(1 for c in calls if not c["gt"] and c[pred_key])
        TN = sum(1 for c in calls if not c["gt"] and not c[pred_key])
        rec = TP / (TP + FN) if (TP + FN) else None
        far = FP / (FP + TN) if (FP + TN) else None
        prec = TP / (TP + FP) if (TP + FP) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        return {"TP": TP, "FN": FN, "FP": FP, "TN": TN, "recall": rec, "false_alarm_rate": far,
                "precision": prec, "f1": f1}

    summary = {"recipe": args.recipe, "arm": args.vlm, "mode": "binary_vs_targeted_steplevel",
               "n_step_units": len(calls), "n_pos": sum(c["gt"] for c in calls),
               "n_binary_calls": len(calls), "n_frames": n_frames, "wall_s": round(time.time() - t0all, 1),
               "binary_generic": score("binary_pred"), "targeted_named": score("targeted_pred"),
               "_note": "same (rid,step) units + same oracle windows; binary = 'any error?' (1 call/step, "
               "no leak); targeted = OR of per-subtype named calls (from the targeted run, ~k calls/step). "
               "Only the QUESTION differs; targeted spends more calls."}
    json.dump(calls, open(os.path.join(out_dir, "calls.jsonl"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json + calls.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="spicedhotchocolate")
    ap.add_argument("--vlm", choices=["qwen", "mock"], default="mock")
    ap.add_argument("--mode", choices=["survey", "targeted", "binary"], default="survey")
    ap.add_argument("--window", choices=["oracle"], default="oracle")
    ap.add_argument("--video-dir", default=os.path.join(BASE, "data/videos_360p"))
    ap.add_argument("--sample-fps", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--neg-per-cell", type=int, default=2)
    ap.add_argument("--recs", default=None, help="comma list to restrict recordings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    criteria = load_criteria(args.recipe)
    spans = oracle_spans()
    recs = args.recs.split(",") if args.recs else recordings_for(args.recipe)
    out_dir = args.out or os.path.join(BASE, "experiments/baseline_t2", f"{args.recipe}_{args.vlm}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    if args.mode == "targeted":
        return run_targeted(args, criteria, spans, recs, out_dir)
    if args.mode == "binary":
        return run_binary(args, criteria, spans, recs, out_dir)
    client = t1.Qwen() if args.vlm == "qwen" else None
    # patch system prompt for the call
    if client:
        t1.SYSTEM_PROMPT = SYS

    calls = []
    n_frames_total = 0
    t_start = time.time()
    pred = {}  # (rid, step, subtype) -> violated bool
    no_check = collections.Counter()  # GT subtypes with no authored check on that step

    for rid in recs:
        vid = os.path.join(args.video_dir, rid + ".mp4")
        rspans = spans.get(rid, {})
        if not os.path.exists(vid):
            print(f"  skip {rid}: no video"); continue
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for step_id, node in criteria.items():
            if not node["checks"]:
                continue
            span = rspans.get(step_id)
            if not span:
                continue  # step not in this recording's annotation
            start, end = span
            jpegs = t1.sample_frames_1fps(cap, fps, t_end=end, interval=max(0.5, end - start),
                                          max_frames=args.max_frames, sample_fps=args.sample_fps)
            n_frames_total += len(jpegs)
            prompt = build_prompt(node, node["checks"])
            if client:
                t0 = time.time()
                try:
                    raw = client.call(jpegs, prompt)
                    obj = t1.safe_json(raw) or {}
                except Exception as e:
                    obj = {}; raw = f"ERROR {e}"
                lat = time.time() - t0
            else:  # mock: flag nothing (plumbing test)
                obj = {c["subtype"]: {"violated": False, "evidence": "mock"} for c in node["checks"]}
                raw = "mock"; lat = 0.0
            for c in node["checks"]:
                v = bool((obj.get(c["subtype"]) or {}).get("violated", False))
                pred[(rid, step_id, c["subtype"])] = v
            calls.append({"rid": rid, "step_id": step_id, "n_frames": len(jpegs),
                          "latency_s": round(lat, 2), "pred": {c["subtype"]: bool((obj.get(c["subtype"]) or {}).get("violated", False)) for c in node["checks"]}})
        cap.release()

    # ---- score ----
    by = lambda: {"TP": 0, "FN": 0, "FP": 0}
    per = collections.defaultdict(by)
    authored = {(rid, sid, c["subtype"]) for rid in recs for sid, node in criteria.items() for c in node["checks"]}
    gt_all = set()
    for rid in recs:
        gt = load_gt(rid)
        if gt is None:
            continue
        for (sid, sub) in gt:
            gt_all.add((rid, sid, sub))
            if (rid, sid, sub) not in authored:
                no_check[sub] += 1  # GT error we never asked about (criteria coverage gap)
                per[sub]["FN"] += 1
            elif pred.get((rid, sid, sub)):
                per[sub]["TP"] += 1
            else:
                per[sub]["FN"] += 1
    # FP: predicted violated where no GT error of that subtype on that step
    for (rid, sid, sub), v in pred.items():
        if v and (rid, sid, sub) not in gt_all:
            per[sub]["FP"] += 1

    summary = {"recipe": args.recipe, "arm": args.vlm, "window": args.window,
               "n_recordings": len(recs), "n_calls": len(calls), "n_frames": n_frames_total,
               "wall_s": round(time.time() - t_start, 1),
               "sample_fps": args.sample_fps, "max_frames": args.max_frames,
               "per_subtype": {}, "no_authored_check": dict(no_check),
               "_caveats": "oracle step-span windowing (detection ceiling, not deployable); in-sample (criteria probe-derived from these recordings)."}
    tot = by()
    for sub, c in sorted(per.items()):
        rec = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else None
        prec = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else None
        summary["per_subtype"][sub] = {**c, "recall": rec, "precision": prec}
        for k in tot: tot[k] += c[k]
    R = tot["TP"] / (tot["TP"] + tot["FN"]) if (tot["TP"] + tot["FN"]) else None
    summary["overall"] = {**tot, "recall": R}

    json.dump(calls, open(os.path.join(out_dir, "calls.jsonl"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json + calls.jsonl")


if __name__ == "__main__":
    main()
