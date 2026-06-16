# eval/ — main evaluation scripts

| Script | Role |
| --- | --- |
| `periodic_vlm.py` | Periodic-VLM baseline: every N s, send frames + task context to the VLM, strict-JSON step/action output, K-consecutive smoothing, action cooldown. Writes `calls.jsonl` (every call: frames sent, latency, raw + parsed response) + `summary.json` (smoothed stage timeline, assistant actions, cost log). |
| `engine.py` | Detector-only replay arm: event-driven task-graph state machine over the frozen audio detectors in `../detectors/detectors_lib.py`. Writes unified results to `../experiments/replay_v1/results/detector_replay/`. |
| `run_escalation.py` | Escalation arm: services each detector-arm escalation request with exactly one Qwen call (10 frames spanning `[0, t_esc]`) to verify added ingredients; merges into `detector_plus_escalation` results. |
| `convert_periodic_vlm.py` | Converts `periodic_vlm.py` run outputs into the unified per-recording result format (`periodic_vlm_qwen` arm). |
| `score.py` | Referee: per-second stage accuracy (coarse/fine), reminder P/R vs the truth table, cost ledger. Writes `../experiments/replay_v1/results/scores.json` and `../experiments/replay_v1/REPORT.md`. |

Run order for the full experiment: `engine.py` → `periodic_vlm.py` (×6 recordings) + `convert_periodic_vlm.py` → `run_escalation.py` → `score.py`.

## periodic_vlm.py usage

```bash
# Qwen3.6-27B on saltyfish (OpenAI-compatible, no key) — the VLM we use; do not use Gemini
export QWEN_VIDEO_SERVER_URL="http://saltyfish.eecs.umich.edu:8000"
export QWEN_VIDEO_MODEL="Qwen/Qwen3.6-27B"
python3 periodic_vlm.py --video ../data/videos_480p/8_16.mp4 \
  --task ../tasks/task_spiced_hot_chocolate_cc4d.json --backend qwen \
  --interval 10 --frames-per-call 3 --out ../experiments/replay_v1/runs/8_16
```

Key knobs: `--interval` (seconds between calls), `--frames-per-call`, `--window` (seconds the frames span), `--k-consecutive` (stage-switch smoothing), `--cooldown` (min seconds between assistant actions), `--max-seconds`.

Task definitions live in `../tasks/`; ground truth in `../data/gt_activity8.json`; the unified result format and the reminder truth table are documented in `score.py` and `../docs/TASK_DEFINITION.md`.
