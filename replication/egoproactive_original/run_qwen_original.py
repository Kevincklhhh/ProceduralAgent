#!/usr/bin/env python3
"""Run the original EgoProactive streaming interrupt/silent task with Qwen.

This writes predictions in the starter-kit Proactive format:

  {"video_path": "...mp4", "answers": ["$silent$", "$interrupt$...", ...]}

At interval j, the model sees frames from intervals 0..j, capped by
--max-frames, plus the high-level query and the prior dialogue history. This is
the original EgoProactive task setup, not the PWR plan-adapter setup.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests


DEFAULT_VIDEO_DIR = Path("/home/kailaic/NeuroTrace/pro/wearable_ai_sample/egoproactive/val")

SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the user "
    "performing a procedural task. The user has issued a single high-level query. "
    "As the video unfolds you observe a series of short chunks; after each chunk "
    "you decide whether to speak or stay silent.\n\n"
    "Output format as JSON only:\n"
    "{\"answer\":\"$interrupt$<timely assistant utterance>\"}\n"
    "or\n"
    "{\"answer\":\"$silent$\"}\n\n"
    "Speak when the user asks you something, when an earlier action needs "
    "correction, or when you have useful, timely guidance for the next step. "
    "Stay silent when nothing useful needs to be said. Do not mention future "
    "visual events that are not visible yet."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_completed_predictions(path: Path) -> set[str]:
    """Return video_path keys already present in an existing predictions JSONL."""
    if not path.exists():
        return set()
    completed = set()
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_path = row.get("video_path")
            answers = row.get("answers")
            if video_path and isinstance(answers, list):
                completed.add(str(video_path))
    return completed


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    indexed_rows = list(enumerate(rows))
    if args.start_index is not None:
        indexed_rows = [(i, row) for i, row in indexed_rows if i >= args.start_index]
    if args.end_index is not None:
        indexed_rows = [(i, row) for i, row in indexed_rows if i < args.end_index]
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise SystemExit("--num-shards must be positive")
        if args.shard_index is None:
            raise SystemExit("--shard-index is required when --num-shards is set")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise SystemExit("--shard-index must be in [0, --num-shards)")
        indexed_rows = [
            (i, row) for i, row in indexed_rows if i % args.num_shards == args.shard_index
        ]
    if args.max_samples is not None:
        indexed_rows = indexed_rows[: args.max_samples]
    return [row for _, row in indexed_rows]


def safe_json(text: str | None) -> dict | None:
    if not text:
        return None
    for candidate in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def normalize_answer(raw: str | None) -> str:
    text = (raw or "").strip()
    obj = safe_json(text)
    if isinstance(obj, dict):
        text = str(obj.get("answer", "")).strip()
    if text.startswith("$interrupt$"):
        rest = text[len("$interrupt$"):].strip()
        return "$interrupt$" + (rest or "I can help with the next step now.")
    if text.startswith("$silent$"):
        return "$silent$"
    lowered = text.lower()
    if "interrupt" in lowered and "silent" not in lowered[:30]:
        cleaned = re.sub(r"^\$?interrupt\$?", "", text, flags=re.IGNORECASE).strip()
        return "$interrupt$" + (cleaned or "I can help with the next step now.")
    return "$silent$"


def jpeg_bytes(frame, max_dim: int, jpeg_q: int) -> bytes | None:
    h, w = frame.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return buf.tobytes() if ok else None


def sample_interval_frames(
    cap,
    fps: float,
    interval: list[float],
    frames_per_interval: int,
    max_dim: int,
    jpeg_q: int,
) -> list[dict]:
    start, end = float(interval[0]), float(interval[1])
    if frames_per_interval <= 1:
        times = [end]
    else:
        times = np.linspace(start, end, frames_per_interval)
    out = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(ts * fps))))
        ok, frame = cap.read()
        if not ok:
            continue
        encoded = jpeg_bytes(frame, max_dim=max_dim, jpeg_q=jpeg_q)
        if encoded is not None:
            out.append({"t_s": round(float(ts), 3), "jpeg": encoded})
    return out


def load_frames_by_interval(video_path: Path, row: dict, args: argparse.Namespace) -> list[list[dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if row.get("duration_in_sec"):
        duration = float(row["duration_in_sec"])
        fps = frame_count / duration if duration > 0 else fps
    frames = [
        sample_interval_frames(
            cap,
            fps,
            interval,
            frames_per_interval=args.frames_per_interval,
            max_dim=args.max_dim,
            jpeg_q=args.jpeg_quality,
        )
        for interval in row.get("video_intervals", [])
    ]
    cap.release()
    return frames


def cap_cumulative_frames(frames_by_interval: list[list[dict]], end_index: int, max_frames: int) -> list[dict]:
    frames = []
    for idx in range(end_index + 1):
        frames.extend(frames_by_interval[idx])
    if max_frames > 0 and len(frames) > max_frames:
        stride = len(frames) / max_frames
        frames = [frames[int(i * stride)] for i in range(max_frames)]
    return frames


def build_user_prompt(row: dict, chunk_index: int, frame_times: list[float], max_history_turns: int) -> str:
    query = str(row.get("query", ""))
    task = str(row.get("task", ""))
    interval = row.get("video_intervals", [])[chunk_index]
    dialog = row.get("dialog", [])
    turns = dialog[chunk_index][1:] if chunk_index < len(dialog) and len(dialog[chunk_index]) >= 1 else []
    if max_history_turns == 0:
        turns = []
    elif max_history_turns > 0:
        turns = turns[-max_history_turns:]
    history = "\n".join(
        f"- {turn.get('role', 'user')}: {turn.get('text', '')}"
        for turn in turns
        if turn.get("text")
    ) or "- none"
    return (
        f"Task: {task}\n"
        f"User query: {query}\n"
        f"Current chunk: {chunk_index + 1}/{len(row.get('video_intervals', []))}, "
        f"interval={interval}s\n"
        f"Visible frame times from past and current chunks: {frame_times}\n\n"
        f"Prior dialogue before this decision:\n{history}\n\n"
        "Predict the assistant response for this exact chunk. Return JSON with one "
        "field, answer, whose value starts with either $interrupt$ or $silent$."
    )


class QwenClient:
    def __init__(self, timeout: float, retries: int, backoff_s: float):
        base = os.getenv("QWEN_VIDEO_SERVER_URL")
        if not base:
            raise SystemExit("Set QWEN_VIDEO_SERVER_URL")
        base = base.rstrip("/")
        self.url = base + (
            "" if base.endswith("/chat/completions")
            else "/chat/completions" if base.endswith("/v1")
            else "/v1/chat/completions"
        )
        self.model = os.getenv("QWEN_VIDEO_MODEL", "Qwen/Qwen3.6-27B")
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s
        self.headers = {"Content-Type": "application/json"}
        if os.getenv("QWEN_VIDEO_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.getenv('QWEN_VIDEO_API_KEY')}"

    def call(self, frames: list[dict], user_prompt: str, max_tokens: int) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(frame["jpeg"]).decode()
                },
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": user_prompt})
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        last = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                content_obj = response.json()["choices"][0]["message"]["content"]
                if isinstance(content_obj, list):
                    return "".join(p.get("text", "") for p in content_obj if isinstance(p, dict))
                return str(content_obj)
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last


def run_row(row: dict, client: QwenClient, args: argparse.Namespace, trace_dir: Path | None) -> dict:
    video_path = args.video_dir / str(row["video_path"])
    frames_by_interval = load_frames_by_interval(video_path, row, args)
    answers = []
    trace_rows = []
    trace_f = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{Path(row['video_path']).stem}.jsonl"
        trace_f = trace_path.open("w")
    try:
        for chunk_index in range(len(row.get("video_intervals", []))):
            frames = cap_cumulative_frames(frames_by_interval, chunk_index, args.max_frames)
            frame_times = [f["t_s"] for f in frames]
            prompt = build_user_prompt(row, chunk_index, frame_times, args.max_history_turns)
            t0 = time.time()
            error = None
            try:
                raw = client.call(frames, prompt, args.max_new_tokens)
            except Exception as exc:
                raw = ""
                error = f"{type(exc).__name__}: {exc}"
            latency_s = time.time() - t0
            answer = normalize_answer(raw)
            answers.append(answer)
            rec = {
                "video_path": row["video_path"],
                "chunk_index": chunk_index,
                "interval_s": row.get("video_intervals", [])[chunk_index],
                "n_images": len(frames),
                "frame_times_s": frame_times,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompt,
                "raw": raw,
                "answer": answer,
                "error": error,
                "latency_s": round(latency_s, 2),
            }
            trace_rows.append(rec)
            if trace_f is not None:
                trace_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                trace_f.flush()
            print(f"  {row['video_path']} chunk {chunk_index + 1:02d}: {answer[:80]}")
    finally:
        if trace_f is not None:
            trace_f.close()
    return {"video_path": row["video_path"], "answers": answers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true", help="Append and skip video_path rows already present in --output.")
    parser.add_argument("--start-index", type=int, help="0-based inclusive row index in the input JSONL.")
    parser.add_argument("--end-index", type=int, help="0-based exclusive row index in the input JSONL.")
    parser.add_argument("--shard-index", type=int, help="0-based shard id for original input row indices.")
    parser.add_argument("--num-shards", type=int, help="Total number of shards for original input row indices.")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frames-per-interval", type=int, default=16)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-dim", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--backoff-s", type=float, default=2.0)
    args = parser.parse_args()

    rows = select_rows(load_jsonl(args.input), args)
    client = QwenClient(timeout=args.timeout, retries=args.retries, backoff_s=args.backoff_s)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_predictions(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as out_f:
        for i, row in enumerate(rows, 1):
            if str(row.get("video_path")) in completed:
                print(f"[{i}/{len(rows)}] skip completed {row.get('video_path')}")
                continue
            print(f"[{i}/{len(rows)}] {row.get('video_path')} :: {row.get('task')}")
            pred = run_row(row, client, args, args.trace_dir)
            out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()

