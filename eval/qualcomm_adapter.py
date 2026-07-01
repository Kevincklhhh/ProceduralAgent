#!/usr/bin/env python3
"""Adapter: our unified arm output  ->  Qualcomm Interactive Cooking prediction format.

We keep the Qualcomm paper's OWN evaluator
(`replication/qualcomm_interactive_cooking_eval/eval.py`) UNMODIFIED and only translate the
prediction shape, so the numbers are theirs (exact reproduction). Run it via
`eval/qualcomm_eval.py`.

================================================================================
THE QUALCOMM SUBSET — a strict subset of our T2 reminder task
================================================================================
Our T2 scores 7 reminder subtypes in 3 classes (see docs/BENCHMARK_INDEX.md §2):
    execution_error      : technique, preparation, measurement, temperature
    parameter_violation  : timing
    precondition_violation : order, missing_step          <-- OUR ADDITIONS

The Qualcomm benchmark scores **only the 5 mistake subtypes** and has **no order /
missing_step** at all: their dataset "annotate[s] for all mistake categories except order
error and missing steps ... not used in our experiments" (paper Appendix B). So:

    QUALCOMM SUBSET  =  our T2  minus  precondition_violation/{order, missing_step}
                     =  {technique, preparation, measurement, temperature, timing}

This adapter ENFORCES that subset on the prediction side: any arm event whose subtype is
order/missing_step (or `other`) is dropped before emission (it would otherwise be an
all-false-positive against a GT that never contains those). Dropped counts are logged so the
exclusion is never silent.

================================================================================
TWO EVALUATION MODES (mirror the paper's two settings)
================================================================================
--mode streaming   T1+T2. Instruction/Success come from the ARM's self-tracked step timeline
                   (`stage_intervals`); a missed/late completion shifts later segments
                   (error propagation), exactly like the paper's Tables 3-4. Feedback from T2.

--mode turnbased   T2 ISOLATED. Instruction/Success come from the ORACLE GT step boundaries
                   (the Qualcomm GT instruction timeline), so each step is scored
                   independently with correct segmentation and no propagation -- the paper's
                   Table 5 setting. Feedback (mistakes) still from the arm, placed into the
                   GT segment by timestamp. This isolates mistake-detection quality.

--oracle           Build a near-perfect prediction from the GT timeline (harness smoke test);
                   mode is irrelevant (instructions are GT either way).

Channel mapping (both modes): Instruction <- step start; Success <- step end; Feedback <- T2
event. Tie-order at a step boundary (Success_k.t == Instruction_{k+1}.t): Feedback < Success
< Instruction so Success stays in segment k.

Output: their format, per recording: {"video_id","pred_texts","pred_timestamps"} where every
text is prefixed "Instruction: " / "Feedback: " / "Success: " and pred_texts[0] is an
instruction.

Usage (build predictions; eval/qualcomm_eval.py wraps build+score):
  python eval/qualcomm_adapter.py --oracle --split test --out /tmp/pred_oracle.json
  python eval/qualcomm_adapter.py --mode streaming --results-dir experiments/t1_baseline \
         --arm qwen36_i10_hist --split test --out /tmp/pred_stream.json
  python eval/qualcomm_adapter.py --mode turnbased --results-dir experiments/baseline_t2 \
         --arm qwen --split test --out /tmp/pred_turn.json
"""
import argparse, json, os, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC4D = os.path.join(BASE, "tasks", "cc4d")
TIMELINE = os.path.join(BASE, "data", "qualcomm_interactive_cooking", "qualcomm_timeline.json")

# The Qualcomm subset: the 5 mistake subtypes their benchmark scores. order/missing_step/other
# are excluded (not in their GT) -- this constant IS the subset definition, enforced below.
SUBSET_SUBTYPES = {"technique", "preparation", "measurement", "temperature", "timing"}

# tie-break priority at equal timestamps (see module docstring)
_PRI = {"Feedback": 0, "Success": 1, "Instruction": 2}


def _flatten(items):
    """items: list of (timestamp, prefix, text) -> (pred_texts, pred_timestamps), chronological
    with the step-boundary tie-ordering."""
    items = sorted(items, key=lambda it: (it[0], _PRI[it[1]]))
    return ([f"{pfx}: {txt}" for (_t, pfx, txt) in items],
            [float(t) for (t, _pfx, _txt) in items])


def _subtype(e):
    return e.get("subtype")


def feedback_items(events):
    """Arm events -> Feedback items, RESTRICTED to the Qualcomm subset. Returns (items, n_dropped)."""
    items, dropped = [], 0
    for e in events:
        if _subtype(e) not in SUBSET_SUBTYPES:        # drop order/missing_step/other
            dropped += 1
            continue
        msg = e.get("message") or f"{e.get('class','')}/{_subtype(e)} issue"
        items.append((e["t"], "Feedback", msg))
    return items, dropped


def steps_by_id(recipe):
    task = json.load(open(os.path.join(CC4D, f"{recipe}.json")))
    return {s["step_id"]: s.get("instruction", "") for s in task["steps"]}


def runs_from_intervals(stage_intervals):
    """Collapse consecutive same-stage intervals into runs [(stage, start_s, end_s)].
    'other'/None stages carry no instruction and are dropped."""
    runs = []
    for iv in stage_intervals:
        st = iv.get("stage")
        if st is None or st == "other":
            continue
        if runs and runs[-1][0] == st and abs(runs[-1][2] - iv["start_s"]) < 1e-6:
            runs[-1] = (st, runs[-1][1], iv["end_s"])
        else:
            runs.append((st, iv["start_s"], iv["end_s"]))
    return runs


def coaching_from_arm(uni, instr_by_id):
    """STREAMING coaching channel: Instruction/Success from the arm's self-tracked timeline."""
    items = []
    for stage, start, end in runs_from_intervals(uni.get("stage_intervals", [])):
        instr = instr_by_id.get(stage, instr_by_id.get(str(stage), f"step {stage}"))
        items.append((start, "Instruction", instr))
        items.append((end, "Success", f"Completed: {instr}"))
    return items


def coaching_from_gt(tl):
    """TURN-BASED coaching channel: Instruction/Success from the ORACLE GT step timeline."""
    items = [(i["t"], "Instruction", i["text"]) for i in tl.get("instructions", [])]
    items += [(s["t"], "Success", s["text"]) for s in tl.get("successes", [])]
    return items


def prediction_from_unified(rid, uni, instr_by_id, mode, tl):
    """One arm record -> Qualcomm prediction dict, per mode. Returns (pred, n_dropped)."""
    coaching = coaching_from_gt(tl) if mode == "turnbased" else coaching_from_arm(uni, instr_by_id)
    fb, dropped = feedback_items(uni.get("events", []))
    texts, stamps = _flatten(coaching + fb)
    return {"video_id": rid, "pred_texts": texts, "pred_timestamps": stamps}, dropped


def prediction_from_gt(vid, tl):
    """ORACLE smoke test: rebuild from GT timeline (mistakes already subset-only)."""
    items = coaching_from_gt(tl)
    items += [(m["t"], "Feedback", m["text"]) for m in tl.get("mistakes", [])
              if m.get("class") in SUBSET_SUBTYPES]
    texts, stamps = _flatten(items)
    return {"video_id": vid, "pred_texts": texts, "pred_timestamps": stamps}


def build_predictions(mode, split, oracle=False, results_dir=None, arm=None):
    """Return (predictions list, total feedback dropped)."""
    timeline = json.load(open(TIMELINE))

    def in_split(vid):
        if split == "all":
            return True
        rec_split = timeline.get(vid, {}).get("split")
        return rec_split in ("train", "validation") if split == "train" else rec_split == split

    preds, dropped_total = [], 0
    if oracle:
        for vid, tl in timeline.items():
            if in_split(vid):
                preds.append(prediction_from_gt(vid, tl))
    else:
        if not (results_dir and arm):
            raise SystemExit("non-oracle mode needs --results-dir and --arm")
        instr_cache = {}
        for f in sorted(glob.glob(os.path.join(results_dir, arm, "*.json"))):
            rid = os.path.splitext(os.path.basename(f))[0]
            if rid not in timeline or not in_split(rid):
                continue
            uni = json.load(open(f))
            stem = _RECIPE_STEM.get(timeline[rid].get("recipe"), timeline[rid].get("recipe"))
            if stem not in instr_cache:
                try:
                    instr_cache[stem] = steps_by_id(stem)
                except FileNotFoundError:
                    instr_cache[stem] = {}
            pred, dropped = prediction_from_unified(rid, uni, instr_cache[stem], mode, timeline[rid])
            preds.append(pred)
            dropped_total += dropped
    return preds, dropped_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="GT-derived prediction (smoke test)")
    ap.add_argument("--mode", choices=["streaming", "turnbased"], default="streaming")
    ap.add_argument("--results-dir", help="dir holding <arm>/<rid>.json unified arm outputs")
    ap.add_argument("--arm", help="arm subdir under --results-dir")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds, dropped = build_predictions(args.mode, args.split, args.oracle,
                                       args.results_dir, args.arm)
    json.dump(preds, open(args.out, "w"), indent=1)
    n_fb = sum(sum("Feedback" in t for t in p["pred_texts"]) for p in preds)
    tag = "oracle" if args.oracle else args.mode
    print(f"wrote {len(preds)} predictions -> {args.out}  [{tag}, split={args.split}]  "
          f"{n_fb} Feedback msgs kept, {dropped} non-subset (order/missing/other) dropped")


# human activity_name (timeline) -> recipe stem (tasks/cc4d/<stem>.json)
_RECIPE_STEM = {
    "spiced hot chocolate": "spicedhotchocolate", "ramen": "ramen", "coffee": "coffee",
    "microwave egg sandwich": "microwaveeggsandwich", "dressed up meatballs": "dressedupmeatballs",
    "microwave mug pizza": "microwavemugpizza", "pan fried tofu": "panfriedtofu", "mug cake": "mugcake",
    "microwave french toast": "microwavefrenchtoast", "pinwheels": "pinwheels", "tomato chutney": "tomatochutney",
    "spicy tuna avocado wraps": "spicytunaavocadowraps", "caprese bruschetta": "capresebruschetta",
    "sauted mushrooms": "sautedmushrooms", "scrambled eggs": "scrambledeggs",
    "blender banana pancakes": "blenderbananapancakes", "herb omelet with fried tomatoes": "herbomeletwithfriedtomatoes",
    "broccoli stir fry": "broccolistirfry", "tomato mozzarella salad": "tomatomozzarellasalad",
    "butter corn cup": "buttercorncup", "cucumber raita": "cucumberraita", "zoodles": "zoodles",
    "cheese pimiento": "cheesepimiento", "breakfast burritos": "breakfastburritos",
}


if __name__ == "__main__":
    main()
