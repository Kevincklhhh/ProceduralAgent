# PWR Framework Replication

This directory contains an explicit implementation of the Plan-Watch-Recover
(PWR) framework from `related_work/01_core_methods/plan_watch_recover_2606.04970.pdf`.

The important code-level pieces are:

- `DuplexInteractionModel`: the user-facing model. At every observation it sees
  the cached plan plus plan-anchored video clips and emits `silent` or
  `interrupt` with an utterance.
- `BackgroundPlanner`: the background model. It is called only when duplex emits
  `interrupt`; it updates completed/current/remaining steps and visual cues.
- `PWRFramework`: the inference loop that enforces the paper factorization:
  silent turns carry the prior plan forward unchanged, interrupt turns invoke
  the planner.
- `PlanAnchoredClipSampler`: gives duplex the recent 8s clip plus clips anchored
  at prior plan-update timestamps; gives planner only the most recent clip.

The trained checkpoints from the paper are not public here, so the implementation
uses configurable backends. `qwen` calls the local OpenAI-compatible Qwen video
server; `mock` exercises the control flow; `egoproactive_gold` replays downloaded
EgoProactive labels through the duplex side.

## Run A Local EgoProactive PWR Demo

Gold replay smoke test:

```bash
BACKEND=egoproactive_gold \
replication/pwr_framework/run_egoproactive_pwr_demo.sh cc4ac34272e4c3e4
```

Qwen run:

```bash
QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
replication/pwr_framework/run_egoproactive_pwr_demo.sh cc4ac34272e4c3e4
```

Outputs are written under `experiments/pwr_runtime/<arm>/`, with optional traces
in `pwr_debug/<video_id>.jsonl`.

## Direct CLI

```bash
python3 replication/pwr_framework/run_pwr.py \
  --video /path/to/video.mp4 \
  --task /path/to/pwr_task.json \
  --duplex-backend qwen \
  --planner-backend qwen \
  --trace \
  --arm pwr_framework_demo
```

For EgoProactive-converted tasks, the runner automatically ticks at the original
annotation decision-point starts and uses `duration_in_sec` to map annotation
seconds to video frames.
