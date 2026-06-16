#!/usr/bin/env bash
# Start both the video/data server and the frontend server
# (same port layout as self_data/annotator: 4002 data, 3010 frontend).
cd "$(dirname "$0")"

echo "Starting video/data server on :4002..."
node video-server.js &
SERVER_PID=$!

echo "Starting frontend on :3010..."
node frontend-server.js &
FRONTEND_PID=$!

trap "kill $SERVER_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
