#!/usr/bin/env python3
"""Run the explicit Plan-Watch-Recover framework implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pwr_framework import (  # noqa: E402
    BackgroundPlanner,
    DuplexInteractionModel,
    MockVisionBackend,
    PWRConfig,
    PWRFramework,
    QwenVideoBackend,
)


def make_backend(kind: str, timeout: float, retries: int):
    if kind in {"mock", "egoproactive_gold"}:
        return MockVisionBackend()
    if kind == "qwen":
        return QwenVideoBackend(timeout=timeout, retries=retries)
    raise SystemExit(f"unknown backend: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", default="experiments/pwr_runtime")
    parser.add_argument("--arm", default="pwr_framework")
    parser.add_argument(
        "--backend",
        choices=["mock", "qwen", "egoproactive_gold"],
        default=None,
        help="Convenience shorthand for both duplex and planner. egoproactive_gold uses gold duplex + mock planner.",
    )
    parser.add_argument("--duplex-backend", choices=["mock", "qwen", "egoproactive_gold"], default="mock")
    parser.add_argument("--planner-backend", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--clip-window-s", type=float, default=8.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=15)
    parser.add_argument("--max-dim", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--mock-step-s", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.backend is not None:
        args.duplex_backend = args.backend
        args.planner_backend = "mock" if args.backend in {"mock", "egoproactive_gold"} else args.backend

    if args.tick <= 0:
        raise SystemExit("--tick must be positive")
    if args.frames_per_clip <= 0:
        raise SystemExit("--frames-per-clip must be positive")
    if args.max_clips <= 0:
        raise SystemExit("--max-clips must be positive")

    task = json.loads(Path(args.task).read_text())
    config = PWRConfig(
        tick_s=args.tick,
        clip_window_s=args.clip_window_s,
        frames_per_clip=args.frames_per_clip,
        max_clips=args.max_clips,
        max_dim=args.max_dim,
        jpeg_quality=args.jpeg_quality,
        max_seconds=args.max_seconds,
        mock_step_s=args.mock_step_s,
    )
    duplex_backend = make_backend(args.duplex_backend, args.timeout, args.retries)
    planner_backend = make_backend(args.planner_backend, args.timeout, args.retries)
    framework = PWRFramework(
        duplex=DuplexInteractionModel(duplex_backend, mode=args.duplex_backend, mock_step_s=args.mock_step_s),
        planner=BackgroundPlanner(planner_backend, mode=args.planner_backend),
        config=config,
    )

    out_dir = Path(args.out_dir) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir if args.trace else None
    rid = Path(args.video).stem
    result = framework.run(args.video, task, rid=rid, trace_dir=trace_dir, verbose=not args.quiet)
    result["recording"] = rid
    result["arm"] = args.arm
    out_path = out_dir / f"{rid}.json"
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote {out_path} "
        f"({result['cost']['duplex_calls']} duplex calls, "
        f"{result['cost']['planner_calls']} planner calls, "
        f"{result['_meta']['parse_failures']} parse-fails)"
    )


if __name__ == "__main__":
    main()
