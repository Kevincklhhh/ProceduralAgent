#!/usr/bin/env python3
"""Launch resumable EgoProactive Qwen server-backed shard workers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/egoproactive/hf_wearable_ai_starter_kit/egoproactive/"
    "wearable_ai_2026_egoproactive_val_700.jsonl"
)
DEFAULT_VIDEO_DIR = Path("data/egoproactive/hf_wearable_ai_starter_kit/egoproactive/val")
DEFAULT_RUNNER = Path("replication/egoproactive_original/run_qwen_original.py")


def parse_shard_indices(raw: str | None, num_shards: int) -> list[int]:
    if not raw:
        return list(range(num_shards))
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if i < 0 or i >= num_shards)
    if bad:
        raise SystemExit(f"shard indices out of range for {num_shards} shards: {bad}")
    return sorted(out)


def build_cmd(args: argparse.Namespace, shard_index: int, output_path: Path, trace_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.runner),
        "--input",
        str(args.input),
        "--video-dir",
        str(args.video_dir),
        "--output",
        str(output_path),
        "--trace-dir",
        str(trace_dir),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(args.num_shards),
        "--max-frames",
        str(args.max_frames),
        "--frames-per-interval",
        str(args.frames_per_interval),
        "--max-history-turns",
        str(args.max_history_turns),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-dim",
        str(args.max_dim),
        "--jpeg-quality",
        str(args.jpeg_quality),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--backoff-s",
        str(args.backoff_s),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.max_samples_per_shard is not None:
        cmd.extend(["--max-samples", str(args.max_samples_per_shard)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--run-root", type=Path, default=Path("replication/egoproactive_original/output/qwen36_full_32f"))
    parser.add_argument("--server-url", default=os.getenv("QWEN_VIDEO_SERVER_URL"))
    parser.add_argument("--model", default=os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B"))
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-indices", help="Comma/range list, e.g. 0,2,5-7. Default: all shards.")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples-per-shard", type=int)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frames-per-interval", type=int, default=16)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-dim", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--backoff-s", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    if not args.server_url:
        raise SystemExit("Set QWEN_VIDEO_SERVER_URL or pass --server-url")
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    if args.parallelism <= 0:
        raise SystemExit("--parallelism must be positive")

    selected = parse_shard_indices(args.shard_indices, args.num_shards)
    shard_dir = args.run_root / "shards"
    trace_root = args.run_root / "traces"
    log_dir = args.run_root / "logs"
    for path in (shard_dir, trace_root, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QWEN_VIDEO_SERVER_URL"] = args.server_url
    env["QWEN_VIDEO_MODEL"] = args.model
    env["PYTHONUNBUFFERED"] = "1"

    print(
        f"Launching {len(selected)} shard(s), num_shards={args.num_shards}, "
        f"parallelism={args.parallelism}, run_root={args.run_root}"
    )

    pending = list(selected)
    active: list[tuple[int, subprocess.Popen, object]] = []
    failures: list[tuple[int, int]] = []

    def launch(shard_index: int) -> None:
        output_path = shard_dir / f"pred_shard_{shard_index:03d}.jsonl"
        trace_dir = trace_root / f"shard_{shard_index:03d}"
        log_path = log_dir / f"shard_{shard_index:03d}.log"
        cmd = build_cmd(args, shard_index, output_path, trace_dir)
        print(f"shard {shard_index:03d}: {' '.join(cmd)} > {log_path}")
        if args.dry_run:
            return
        log_f = log_path.open("a", buffering=1)
        log_f.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        active.append((shard_index, proc, log_f))

    while pending or active:
        while pending and len(active) < args.parallelism:
            launch(pending.pop(0))
        if args.dry_run:
            continue
        time.sleep(5)
        still_active = []
        for shard_index, proc, log_f in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((shard_index, proc, log_f))
                continue
            log_f.write(f"=== exit {rc} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.close()
            print(f"shard {shard_index:03d} exited rc={rc}")
            if rc != 0:
                failures.append((shard_index, rc))
                if args.stop_on_failure:
                    pending.clear()
        active = still_active

    if failures:
        print(f"Failures: {failures}", file=sys.stderr)
        raise SystemExit(1)
    print("All selected shards completed.")


if __name__ == "__main__":
    main()
