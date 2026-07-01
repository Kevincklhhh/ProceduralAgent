#!/usr/bin/env python3
"""Create a small EgoProactive JSONL containing rows with downloaded videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ANNOTATION = Path(
    "/home/kailaic/NeuroTrace/pro/wearable_ai_annotations/egoproactive/"
    "wearable_ai_2026_egoproactive_val_700.jsonl"
)
DEFAULT_VIDEO_DIR = Path("/home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val")


def normalize_video_id(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value).stem


def load_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out", type=Path, default=Path("replication/egoproactive_original/output/local_subset.jsonl"))
    parser.add_argument("--video-id", help="Optional video id or filename to keep exactly one row.")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    wanted_id = normalize_video_id(args.video_id)
    available = {p.name for p in args.video_dir.glob("*.mp4")}
    rows = []
    for row in load_rows(args.annotation):
        video_name = str(row.get("video_path", ""))
        if wanted_id is not None and Path(video_name).stem != wanted_id:
            continue
        if video_name not in available:
            continue
        rows.append(row)
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if wanted_id is not None and not rows:
        raise SystemExit(f"No downloaded video row matched --video-id {args.video_id!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "annotation": str(args.annotation),
        "video_dir": str(args.video_dir),
        "out": str(args.out),
        "rows": len(rows),
        "videos": [r.get("video_path") for r in rows],
    }, indent=2))


if __name__ == "__main__":
    main()

