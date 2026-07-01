#!/usr/bin/env bash
set -euo pipefail

VIDEO_ID="${1:-cc4ac34272e4c3e4}"
ROOT="${ROOT:-replication/egoproactive_original}"
OUT_DIR="${OUT_DIR:-$ROOT/output}"
ANNOTATION="${ANNOTATION:-/home/kailaic/NeuroTrace/pro/wearable_ai_annotations/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val}"

if [[ -z "${QWEN_VIDEO_SERVER_URL:-}" ]]; then
  echo "Set QWEN_VIDEO_SERVER_URL before running this script." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

GOLD="$OUT_DIR/${VIDEO_ID}.jsonl"
PRED="$OUT_DIR/${VIDEO_ID}_qwen_predictions.jsonl"
RESULTS="$OUT_DIR/${VIDEO_ID}_qwen_results.json"
TRACE_DIR="$OUT_DIR/traces"

python3 "$ROOT/make_local_subset.py"   --annotation "$ANNOTATION"   --video-dir "$VIDEO_DIR"   --video-id "$VIDEO_ID"   --out "$GOLD"

python3 "$ROOT/run_qwen_original.py"   --input "$GOLD"   --video-dir "$VIDEO_DIR"   --output "$PRED"   --trace-dir "$TRACE_DIR"   --timeout "${TIMEOUT:-180}"   --retries "${RETRIES:-1}"   --max-frames "${MAX_FRAMES:-32}"   --frames-per-interval "${FRAMES_PER_INTERVAL:-16}"

python3 "$ROOT/score_predictions.py"   --gold "$GOLD"   --pred "$PRED"   --out "$RESULTS"
