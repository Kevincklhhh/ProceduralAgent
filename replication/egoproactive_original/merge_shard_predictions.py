#!/usr/bin/env python3
"""Merge EgoProactive shard prediction JSONLs in original input order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original ordered gold/input JSONL.")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory containing pred_shard_*.jsonl files.")
    parser.add_argument("--output", type=Path, required=True, help="Merged prediction JSONL.")
    parser.add_argument("--missing-output", type=Path, help="Optional JSONL of gold rows not yet predicted.")
    args = parser.parse_args()

    gold_rows = load_jsonl(args.input)
    predictions: dict[str, dict] = {}
    duplicate_count = 0
    shard_paths = sorted(args.shard_dir.glob("pred_shard_*.jsonl"))
    for shard_path in shard_paths:
        for line_no, line in enumerate(shard_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_path = str(row.get("video_path", ""))
            if not video_path:
                continue
            if video_path in predictions:
                duplicate_count += 1
            predictions[video_path] = row

    ordered = []
    missing = []
    for gold in gold_rows:
        video_path = str(gold.get("video_path", ""))
        pred = predictions.get(video_path)
        if pred is None:
            missing.append(gold)
        else:
            ordered.append(pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out_f:
        for row in ordered:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.missing_output is not None:
        args.missing_output.parent.mkdir(parents=True, exist_ok=True)
        with args.missing_output.open("w") as out_f:
            for row in missing:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_rows": len(gold_rows),
        "shard_files": len(shard_paths),
        "predicted_rows": len(ordered),
        "missing_rows": len(missing),
        "duplicates_seen": duplicate_count,
        "output": str(args.output),
        "missing_output": str(args.missing_output) if args.missing_output else None,
    }, indent=2))


if __name__ == "__main__":
    main()
