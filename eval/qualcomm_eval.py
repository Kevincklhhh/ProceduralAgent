#!/usr/bin/env python3
"""Qualcomm-subset evaluator (entry point).

Scores an arm on the **Qualcomm subset** (5 mistake subtypes; order/missing_step excluded --
see eval/qualcomm_adapter.py) by building predictions and running the Qualcomm paper's OWN
`eval.py` UNMODIFIED. Two settings, mirroring the paper:

  streaming  T1+T2  -- self-tracked completion (error propagation); paper Tables 3-4.
  turnbased  T2 iso -- oracle GT step segmentation; isolates mistake detection; paper Table 5.

This wrapper only orchestrates (build predictions -> shell out to their scorer in the `qual`
conda env). The metric numbers are produced entirely by their code.

Usage:
  python eval/qualcomm_eval.py --oracle --split test --both            # harness smoke test
  python eval/qualcomm_eval.py --mode streaming --results-dir experiments/t1_baseline --arm qwen36_i10_hist
  python eval/qualcomm_eval.py --both --results-dir experiments/proposed_system/results --arm proposed
Env: QUAL_PY (default ~/miniconda3/envs/qual/bin/python), QUAL_EVAL_DIR (their repo).
"""
import argparse, json, os, re, subprocess, sys, tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualcomm_adapter as qa

QUAL_PY = os.environ.get("QUAL_PY", os.path.expanduser("~/miniconda3/envs/qual/bin/python"))
QUAL_EVAL_DIR = os.environ.get(
    "QUAL_EVAL_DIR", os.path.join(BASE, "replication", "qualcomm_interactive_cooking_eval"))


def run_scorer(pred_path, plan_set, split):
    """Invoke their unmodified eval.py; return (text_block, parsed_counts) from the final
    (corpus-cumulative) block. parsed_counts: {ic_acc, tp, fp, tn, fn}."""
    if not os.path.exists(QUAL_PY):
        raise SystemExit(f"qual env python not found: {QUAL_PY} (set QUAL_PY)")
    cmd = [QUAL_PY, "eval.py", "--plan_set", plan_set, "--split", split,
           "--predictions_file_path", pred_path]
    env = {**os.environ, "PYTHONPATH": "./"}
    p = subprocess.run(cmd, cwd=QUAL_EVAL_DIR, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
        raise SystemExit(f"their eval.py failed (rc={p.returncode})")
    out = p.stdout
    block = out[out.rfind("DETECTION_WINDOW"):] if "DETECTION_WINDOW" in out else out[-800:]
    ic = re.search(r"IC-Acc:\s*([\d.]+)", block)
    cm = re.search(r"mistake_tp:(\d+),\s*mistake_fp:(\d+),\s*mistake_tn:(\d+),\s*mistake_fn:(\d+)", block)
    parsed = {"ic_acc": float(ic.group(1)) if ic else None,
              "tp": int(cm.group(1)), "fp": int(cm.group(2)),
              "tn": int(cm.group(3)), "fn": int(cm.group(4))} if cm else {}
    return block.strip(), parsed


def _prf(tp, fp, tn, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return pr, rc, f1, acc


def _score_split(mode, scoring_split, args, tmpdir):
    """Build subset predictions for one scoring split and run their eval.py. Returns parsed."""
    preds, dropped = qa.build_predictions(
        mode, scoring_split, oracle=args.oracle, results_dir=args.results_dir, arm=args.arm)
    pred_path = os.path.join(tmpdir, f"pred_{'oracle' if args.oracle else mode}_{scoring_split}.json")
    json.dump(preds, open(pred_path, "w"), indent=1)
    n_fb = sum(sum("Feedback" in t for t in p["pred_texts"]) for p in preds)
    block, parsed = run_scorer(pred_path, args.plan_set, scoring_split)
    print(f"  [{scoring_split}] {len(preds)} preds, {n_fb} subset-Feedback, {dropped} dropped")
    print("   " + block.replace("\n", "\n   "))
    parsed["n_preds"] = len(preds)
    return parsed


def evaluate(mode, args, tmpdir):
    label = "oracle" if args.oracle else mode
    print(f"\n{'='*72}\n[{label}]  plan_set={args.plan_set}  scope={args.split}")
    # "all" (no train/test split): their loader's 'train'(=train+val,275) + 'test'(109) = 384,
    # disjoint, so summing their printed counts is the exact corpus number.
    splits = ["train", "test"] if args.split == "all" else [args.split]
    parts = [_score_split(mode, s, args, tmpdir) for s in splits]
    parts = [p for p in parts if p.get("tp") is not None]
    if len(splits) > 1 and parts:
        tp = sum(p["tp"] for p in parts); fp = sum(p["fp"] for p in parts)
        tn = sum(p["tn"] for p in parts); fn = sum(p["fn"] for p in parts)
        pr, rc, f1, acc = _prf(tp, fp, tn, fn)
        # IC-Acc: recording-weighted across splits (denominators not printed; proxy)
        w = sum(p["n_preds"] for p in parts) or 1
        ic = sum((p["ic_acc"] or 0) * p["n_preds"] for p in parts) / w
        print(f"\n  >>> CORPUS ({label}, all 384) <<<")
        print(f"      IC-Acc≈{ic:.1f} (recording-weighted)  |  mistake "
              f"TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"      Precision={pr:.3f}  Recall={rc:.3f}  F1={f1:.3f}  Acc={acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--mode", choices=["streaming", "turnbased"], default="streaming")
    ap.add_argument("--both", action="store_true", help="run streaming AND turnbased")
    ap.add_argument("--results-dir")
    ap.add_argument("--arm")
    ap.add_argument("--plan_set", default="main", choices=["main", "advanced_planning"])
    ap.add_argument("--split", default="all", choices=["train", "validation", "test", "all"])
    ap.add_argument("--keep", help="dir to keep prediction jsons (default: temp)")
    args = ap.parse_args()

    modes = ["streaming", "turnbased"] if args.both else [args.mode]
    tmpdir = args.keep or tempfile.mkdtemp(prefix="qualeval_")
    os.makedirs(tmpdir, exist_ok=True)
    for m in modes:
        evaluate(m, args, tmpdir)
    if not args.keep:
        print(f"\n(prediction jsons in {tmpdir})")


if __name__ == "__main__":
    main()
