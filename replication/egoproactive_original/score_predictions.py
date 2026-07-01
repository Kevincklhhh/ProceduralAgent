#!/usr/bin/env python3
"""Score original EgoProactive predictions against interrupt/silent labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_tag(response: str) -> str:
    return "interrupt" if str(response).lstrip().startswith("$interrupt$") else "silent"


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    int_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    int_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    int_f1 = 2 * int_p * int_r / (int_p + int_r) if (int_p + int_r) > 0 else 0.0

    sil_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    sil_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sil_f1 = 2 * sil_p * sil_r / (sil_p + sil_r) if (sil_p + sil_r) > 0 else 0.0

    return {
        "macro_f1": round((int_f1 + sil_f1) / 2, 4),
        "gmean_f1": round(math.sqrt(int_f1 * sil_f1), 4),
        "interrupt_precision": round(int_p, 4),
        "interrupt_recall": round(int_r, 4),
        "interrupt_f1": round(int_f1, 4),
        "silent_precision": round(sil_p, 4),
        "silent_recall": round(sil_r, 4),
        "silent_f1": round(sil_f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": tp + fp + tn + fn,
    }


def score(gold: list[dict], pred: list[dict]) -> dict:
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} sessions but pred has {len(pred)}")

    totals = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    per_task_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    per_row = []
    skipped_chunks = 0

    for i, (g, p) in enumerate(zip(gold, pred)):
        gold_answers = g.get("answers", [])
        pred_answers = p.get("answers", [])
        n = min(len(gold_answers), len(pred_answers))
        skipped_chunks += abs(len(gold_answers) - len(pred_answers))
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        tags = []
        for j in range(n):
            gold_tag = parse_tag(gold_answers[j])
            pred_tag = parse_tag(pred_answers[j])
            tags.append({"gold": gold_tag, "pred": pred_tag})
            if gold_tag == "interrupt" and pred_tag == "interrupt":
                counts["tp"] += 1
            elif gold_tag == "silent" and pred_tag == "interrupt":
                counts["fp"] += 1
            elif gold_tag == "silent" and pred_tag == "silent":
                counts["tn"] += 1
            else:
                counts["fn"] += 1
        task = str(g.get("task", "unknown"))
        for key in totals:
            totals[key] += counts[key]
            per_task_counts[task][key] += counts[key]
        per_row.append({
            "index": i,
            "video_path": g.get("video_path", ""),
            "task": task,
            "num_chunks": len(gold_answers),
            "tags": tags,
            "counts": counts,
        })

    return {
        "overall": binary_metrics(**totals),
        "per_task": {
            task: binary_metrics(**counts)
            for task, counts in sorted(per_task_counts.items())
        },
        "total_sessions": len(gold),
        "skipped_chunks": skipped_chunks,
        "per_row": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = score(load_jsonl(args.gold), load_jsonl(args.pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    overall = results["overall"]
    print(f"Sessions: {results['total_sessions']}")
    print(f"Chunks scored: {overall['support']}")
    print(
        "Interrupt: "
        f"P={overall['interrupt_precision']:.3f} "
        f"R={overall['interrupt_recall']:.3f} "
        f"F1={overall['interrupt_f1']:.3f}"
    )
    print(
        "Silent: "
        f"P={overall['silent_precision']:.3f} "
        f"R={overall['silent_recall']:.3f} "
        f"F1={overall['silent_f1']:.3f}"
    )
    print(f"Macro F1: {overall['macro_f1']:.4f}")
    print(f"G-mean F1: {overall['gmean_f1']:.4f}")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()

