#!/usr/bin/env python3
"""Convert baseline_periodic_vlm summary.json runs into the unified per-recording
result format under replay/results/periodic_vlm_qwen/{rec}.json."""

import json
import re
import sys
from pathlib import Path

BASE = Path("/home/kailaic/NeuroTrace/ProceduralAgent/experiments/replay_v1")
ARM = "periodic_vlm_qwen"

RECS = ["8_16", "8_3", "8_25", "8_26", "8_31", "8_50"]

# how many periodic_vlm processes shared the vLLM server during each run
CONCURRENCY_NOTES = {
    "8_16": "ran at 2-way concurrency; sequential smoke-test latency 43.6s/call",
    "8_3": "ran solo (1-way) after first attempt was killed at 25/42 calls",
    "8_25": "ran at 2-way concurrency",
    "8_26": "ran at 2-way concurrency",
    "8_31": "ran at 2-way concurrency",
    "8_50": "ran at 2-way concurrency",
}

LATE_STAGES = {"mix", "heat_serve"}


def classify_event(msg, t, stage_at_t, seen_late_stage):
    m = msg.lower()
    # overtime: still running / too long / over a minute
    if re.search(r"still running|too long|over a minute|longer than|exceed", m) or \
       (re.search(r"1 minute|one minute|minute is up", m) and re.search(r"still|running|check the microwave|is up", m)):
        return "overtime_microwave_2" if seen_late_stage else "overtime_microwave"
    if re.search(r"\bdone\b|\bready\b|take (the )?(mug|it) out|take out|finished", m):
        return "microwave_done_prompt"
    if re.search(r"(forgot|didn'?t add|did not add|missing|skipp?ed|haven'?t added|no .*added).*"
                 r"(chocolate|cinnamon|sugar|ingredient)", m) or \
       re.search(r"(chocolate|cinnamon|sugar|ingredient).*(missing|not added|skipped|forgot|went in yet)", m) or \
       re.search(r"did you add", m):
        return "missing_ingredient_before_mix"
    if re.search(r"(mix|stir).*(missing|not done|skipped|forgot|didn'?t|did not|before)", m) or \
       re.search(r"(forgot|missing|skipped).*(mix|stir)", m):
        return "missing_mix_before_heat"
    if re.search(r"\bhot\b|careful|burn|caution", m):
        return "hot_mug_caution"
    return "other"


def convert(rec):
    run_dir = BASE / "runs" / rec
    summary = json.loads((run_dir / "summary.json").read_text())

    # stage_intervals from smoothed stage_timeline (keep fine ids)
    stage_intervals = []
    for seg in summary["stage_timeline"]:
        stage_intervals.append({
            "stage": seg["step_id"],
            "start_s": float(seg["start_s"]),
            "end_s": float(seg["end_s"]),
        })
    # enforce non-overlapping ordered (timeline is sequential by construction)
    for i in range(1, len(stage_intervals)):
        if stage_intervals[i]["start_s"] < stage_intervals[i - 1]["end_s"]:
            stage_intervals[i]["start_s"] = stage_intervals[i - 1]["end_s"]

    # track when the smoothed model first enters a late stage (mix/heat_serve)
    first_late_t = None
    for seg in summary["stage_timeline"]:
        if seg["step_id"] in LATE_STAGES:
            first_late_t = float(seg["start_s"])
            break

    events = []
    for ev in summary["events"]:
        t = float(ev["timestamp_s"])
        seen_late = first_late_t is not None and t >= first_late_t
        eid = classify_event(ev.get("message", ""), t, ev.get("step_id"), seen_late)
        etype = "warning" if ev.get("action_type") == "warning" else "reminder"
        out_ev = {"t": t, "type": etype, "id": eid, "message": ev.get("message", "")}
        events.append(out_ev)

    cost_log = summary["cost_log"]
    n_calls = cost_log["num_vlm_calls"]
    mean_lat = cost_log.get("mean_latency_s") or 0.0
    lat_total = round(n_calls * mean_lat, 1)
    # compute_s = total wall: vlm latency + frame sampling overhead; we use wall
    # time recorded externally if available, else latency total.
    wall_file = run_dir / "wall_s.txt"
    compute_s = float(wall_file.read_text().strip()) if wall_file.exists() else lat_total

    result = {
        "recording": rec,
        "arm": ARM,
        "stage_intervals": stage_intervals,
        "events": events,
        "escalation_requests": [],
        "cost": {
            "vlm_calls": n_calls,
            "frames_sent": cost_log["frames_sent"],
            "vlm_latency_total_s": lat_total,
            "compute_s": round(compute_s, 1),
            "notes": f"interval=10s frames_per_call=3 model={summary['baseline']['model']} "
                     f"parse_failure_rate={cost_log.get('parse_failure_rate')} "
                     f"mean_latency_s={mean_lat}; {CONCURRENCY_NOTES.get(rec, '')}",
        },
    }
    out_dir = BASE / "results" / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}: {len(stage_intervals)} intervals, {len(events)} events")


if __name__ == "__main__":
    recs = sys.argv[1:] or RECS
    for rec in recs:
        convert(rec)
