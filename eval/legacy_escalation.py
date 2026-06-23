#!/usr/bin/env python3
"""legacy_escalation.py (was run_escalation.py) -- ESCALATION arm: detector replay +
exactly the sparse VLM calls the detector arm requested. LEGACY replay_v1 generation.

For each recording, the detector arm emitted one escalation_request at t_esc
("verify which ingredients were added to the mug before mixing/heating").
We extract 10 frames uniformly spanning [0, t_esc], make ONE Qwen call
(same OpenAI-compatible vLLM endpoint usage as QwenBackend in
baseline_periodic_vlm.py), and merge the verdict back into a copy of the
detector-arm result as arm "detector_plus_escalation".
"""

import base64
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path("/home/kailaic/NeuroTrace/ProceduralAgent/experiments/replay_v1")
DET_DIR = ROOT / "results" / "detector_replay"
OUT_DIR = ROOT / "results" / "detector_plus_escalation"
RAW_DIR = OUT_DIR / "raw_vlm"
VID_DIR = Path("/home/kailaic/NeuroTrace/ProceduralAgent/data/videos_480p")

URL = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B"
RECS = ["8_16", "8_3", "8_25", "8_26", "8_31", "8_50"]

SYSTEM_PROMPT = (
    "You are verifying a cooking step from camera snapshots. Use only visible "
    "evidence in the provided frames. Answer with strict JSON only, no prose."
)

USER_PROMPT = (
    "The frames are chronological snapshots of an egocentric video, uniformly "
    "spanning from the start of the recording up to the moment the user starts "
    "mixing/heating a mug of milk for spiced hot chocolate.\n"
    "The recipe requires adding these ingredients to the microwaved milk:\n"
    "  - 2 pieces of chocolate\n"
    "  - 1/5 tsp cinnamon\n"
    "  - 1 tsp sugar\n"
    "Based only on what is visible in the frames, decide for each ingredient "
    "whether the user added it to the mug at any point in the frames shown.\n"
    "Answer strict JSON exactly in this schema:\n"
    '{"chocolate_added": true/false, "cinnamon_added": true/false, '
    '"sugar_added": true/false, "evidence": ["..."], "confidence": 0.0-1.0}'
)

INGREDIENT_KEYS = [("chocolate_added", "chocolate"),
                   ("cinnamon_added", "cinnamon"),
                   ("sugar_added", "sugar")]


def extract_frames(rec, t_esc, n_frames=10, jpeg_q=80):
    """n_frames jpegs uniformly spanning [0, t_esc] (480p source, no resize)."""
    cap = cv2.VideoCapture(str(VID_DIR / f"{rec}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for {rec}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = np.linspace(0.0, t_esc, n_frames)
    jpegs, used_times = [], []
    for ts in times:
        idx = min(int(ts * fps), n_total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok:
            jpegs.append(buf.tobytes())
            used_times.append(round(float(ts), 2))
    cap.release()
    return jpegs, used_times


def call_qwen(jpegs):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
               for j in jpegs]
    content.append({"type": "text", "text": USER_PROMPT})
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": 2000,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": content}]}
    t0 = time.time()
    r = requests.post(URL, json=payload, headers={"Content-Type": "application/json"},
                      timeout=300)
    r.raise_for_status()
    latency = time.time() - t0
    c = r.json()["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c, latency


def process(rec):
    det = json.loads((DET_DIR / f"{rec}.json").read_text())
    t_esc = det["escalation_requests"][0]["t"]

    t0 = time.time()
    jpegs, frame_times = extract_frames(rec, t_esc)
    extract_s = time.time() - t0

    raw, latency = call_qwen(jpegs)
    parsed = None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # ---- merge into a copy of the detector result ----
    res = copy.deepcopy(det)
    res["arm"] = "detector_plus_escalation"
    missing = []
    if parsed:
        for key, name in INGREDIENT_KEYS:
            if parsed.get(key) is False:
                missing.append(name)
                res["events"].append({
                    "t": t_esc, "type": "reminder",
                    "id": "missing_ingredient_before_mix",
                    "message": (f"Did you add the {name}? The recipe adds 2 pieces of "
                                f"chocolate, 1/5 tsp cinnamon and 1 tsp sugar to the "
                                f"microwaved milk before mixing/heating."),
                })
    res["events"].sort(key=lambda e: e["t"])

    note = (f"escalation VLM call at t={t_esc}s; verdict: " +
            ", ".join(f"{n}={'added' if parsed.get(k) else 'MISSING'}"
                      for k, n in INGREDIENT_KEYS) +
            f"; confidence={parsed.get('confidence')}") if parsed else \
           "escalation VLM call: PARSE FAILURE, no reminders emitted"
    res["cost"] = {
        "vlm_calls": 1,
        "frames_sent": len(jpegs),
        "vlm_latency_total_s": round(latency, 2),
        "compute_s": round(det["cost"]["compute_s"] + extract_s, 2),
        "notes": note,
    }

    (OUT_DIR / f"{rec}.json").write_text(json.dumps(res, indent=1))
    (RAW_DIR / f"{rec}.json").write_text(json.dumps({
        "recording": rec, "t_esc": t_esc, "model": MODEL,
        "n_frames": len(jpegs), "frame_times_s": frame_times,
        "jpeg_quality": 80, "latency_s": round(latency, 2),
        "raw_response": raw, "parsed": parsed,
    }, indent=1))
    print(f"{rec}: t_esc={t_esc} latency={latency:.1f}s missing={missing or 'none'} "
          f"conf={parsed.get('confidence') if parsed else None}")
    return rec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for _ in ex.map(process, RECS):
            pass
    print("done")


if __name__ == "__main__":
    main()
