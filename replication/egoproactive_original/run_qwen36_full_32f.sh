#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROCEDURAL_AGENT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-replication/egoproactive_original/output/qwen36_full_32f}"
mkdir -p "$RUN_ROOT"

exec python3 replication/egoproactive_original/launch_qwen_shards.py \
  --server-url "${QWEN_VIDEO_SERVER_URL:-http://saltyfish.eecs.umich.edu:8000}" \
  --model "${QWEN_VIDEO_MODEL:-Qwen/Qwen3.6-27B}" \
  --num-shards "${NUM_SHARDS:-8}" \
  --parallelism "${PARALLELISM:-2}" \
  --run-root "$RUN_ROOT" \
  --max-frames "${MAX_FRAMES:-32}" \
  --frames-per-interval "${FRAMES_PER_INTERVAL:-16}" \
  --max-history-turns "${MAX_HISTORY_TURNS:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-dim "${MAX_DIM:-768}" \
  --jpeg-quality "${JPEG_QUALITY:-85}" \
  --timeout "${TIMEOUT:-600}" \
  --retries "${RETRIES:-0}"
