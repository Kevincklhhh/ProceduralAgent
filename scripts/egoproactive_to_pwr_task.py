#!/usr/bin/env python3
"""Convert one EgoProactive JSONL row into pwr_runtime.py's task JSON shape."""

import argparse
import json
import re
from pathlib import Path


def clean_answer(answer):
    text = re.sub(r"^\$(?:interrupt|silent)\$", "", answer or "").strip()
    return text or "Stay silent while the user continues the current action."


def label(answer):
    answer = (answer or "").strip().lower()
    if answer.startswith("$interrupt$"):
        return "interrupt"
    if answer.startswith("$silent$"):
        return "silent"
    return "unknown"


def load_row(jsonl_path, video_path):
    with open(jsonl_path) as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            if row.get("video_path") == video_path:
                row["_line_no"] = line_no
                return row
    raise SystemExit(f"video_path not found in annotation JSONL: {video_path}")


def convert(row):
    steps = []
    decision_points = []
    prev_step_id = None
    for i, (interval, answer) in enumerate(zip(row["video_intervals"], row["answers"]), 1):
        decision = label(answer)
        step_id = None
        if decision == "interrupt":
            step_id = len(steps) + 1
            preconditions = [prev_step_id] if prev_step_id is not None else []
            prev_step_id = step_id
            steps.append({
                "step_id": step_id,
                "order": step_id,
                "name": f"egoproactive_interrupt_{step_id:02d}",
                "instruction": clean_answer(answer),
                "preconditions": preconditions,
                "egoproactive_interval_s": interval,
                "egoproactive_source_decision_index": i,
            })
        decision_points.append({
            "index": i,
            "interval_s": interval,
            "decision": decision,
            "answer": answer,
            "step_id": step_id,
        })
    return {
        "task_id": Path(row["video_path"]).stem,
        "title": row["task"],
        "source": "facebook/wearable-ai egoproactive",
        "query": row["query"],
        "domain": row["domain"],
        "duration_in_sec": row["duration_in_sec"],
        "source_video_path": row["video_path"],
        "source_jsonl_line": row["_line_no"],
        "steps": steps,
        "egoproactive_decision_points": decision_points,
        "reminders": [],
        "allowed_assistant_actions": ["none", "interrupt", "silent"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row = load_row(args.jsonl, args.video_path)
    task = convert(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out),
        "line": row["_line_no"],
        "video_path": row["video_path"],
        "task": row["task"],
        "n_intervals": len(row["video_intervals"]),
        "n_interrupt_steps": len(task["steps"]),
        "decisions": [label(a) for a in row["answers"]],
    }, indent=2))


if __name__ == "__main__":
    main()
