# Original EgoProactive Runner

This directory is for running the native EgoProactive task from the Wearable AI
starter kit: watch a streaming first-person video session and, for each annotated
interval, predict either:

- `$interrupt$<assistant utterance>`
- `$silent$`

The scripts keep the original prediction format so the output can be scored with
the same interrupt/silent metrics used by the starter kit.

## Local Data Defaults

The scripts default to the downloaded files on this machine:

```bash
ANNOTATION=/home/kailaic/NeuroTrace/pro/wearable_ai_annotations/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
VIDEO_DIR=/home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val
```

Only a few videos are downloaded locally, so the usual flow starts by creating a
small subset JSONL containing just rows whose videos exist.

## 1. Build A Local Subset

```bash
python3 replication/egoproactive_original/make_local_subset.py \
  --annotation /home/kailaic/NeuroTrace/pro/wearable_ai_annotations/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \
  --video-dir /home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val \
  --out replication/egoproactive_original/output/local_subset.jsonl
```

For the one phone-apps example used in the PWR demo:

```bash
python3 replication/egoproactive_original/make_local_subset.py \
  --video-id cc4ac34272e4c3e4 \
  --out replication/egoproactive_original/output/phone_apps.jsonl
```

## 2. Run The Native EgoProactive Task With Qwen

One-command demo:

```bash
QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
replication/egoproactive_original/run_phone_apps_qwen.sh cc4ac34272e4c3e4
```

Or run the steps manually:

```bash
QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
python3 replication/egoproactive_original/run_qwen_original.py \
  --input replication/egoproactive_original/output/phone_apps.jsonl \
  --video-dir /home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val \
  --output replication/egoproactive_original/output/phone_apps_predictions.jsonl \
  --trace-dir replication/egoproactive_original/output/traces \
  --timeout 180 \
  --retries 1
```

This follows the original task framing: at chunk `j`, the model sees visual
evidence up to chunk `j`, the high-level user query, and prior assistant turns.
It then emits one answer string for that chunk.

## 3. Score Predictions

```bash
python3 replication/egoproactive_original/score_predictions.py \
  --gold replication/egoproactive_original/output/phone_apps.jsonl \
  --pred replication/egoproactive_original/output/phone_apps_predictions.jsonl \
  --out replication/egoproactive_original/output/phone_apps_results.json
```

## Optional: Run The PWR Adapter On The Same Row

This converts one EgoProactive row into the plan-shaped task expected by
`eval/pwr_runtime.py`, then runs the PWR loop at the original EgoProactive
decision points.

```bash
QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 \
QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
replication/egoproactive_original/run_pwr_adapter_demo.sh cc4ac34272e4c3e4
```

