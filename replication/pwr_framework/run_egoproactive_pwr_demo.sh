#!/usr/bin/env bash
set -euo pipefail

VIDEO_ID="${1:-cc4ac34272e4c3e4}"
BACKEND="${BACKEND:-qwen}"
ARM="${ARM:-pwr_framework_egoproactive_${BACKEND}_${VIDEO_ID}}"

ANNOTATION="${ANNOTATION:-/home/kailaic/NeuroTrace/pro/wearable_ai_annotations/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val}"
TASK_DIR="${TASK_DIR:-replication/pwr_framework/output/egoproactive_tasks}"
OUT_DIR="${OUT_DIR:-experiments/pwr_runtime}"

mkdir -p "$TASK_DIR"

python3 scripts/egoproactive_to_pwr_task.py \
  --jsonl "$ANNOTATION" \
  --video-path "${VIDEO_ID}.mp4" \
  --out "$TASK_DIR/${VIDEO_ID}.task.json"

MAX_SECONDS="$(python3 -c "import json; print(json.load(open('$TASK_DIR/${VIDEO_ID}.task.json'))['duration_in_sec'])")"

python3 replication/pwr_framework/run_pwr.py \
  --video "$VIDEO_DIR/${VIDEO_ID}.mp4" \
  --task "$TASK_DIR/${VIDEO_ID}.task.json" \
  --backend "$BACKEND" \
  --max-seconds "$MAX_SECONDS" \
  --frames-per-clip "${FRAMES_PER_CLIP:-4}" \
  --max-clips "${MAX_CLIPS:-4}" \
  --trace \
  --out-dir "$OUT_DIR" \
  --arm "$ARM" \
  --timeout "${TIMEOUT:-180}" \
  --retries "${RETRIES:-1}"
