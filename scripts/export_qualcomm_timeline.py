#!/usr/bin/env python3
"""Export the Qualcomm Interactive Cooking layer to JSON for the visualizer.

The Qualcomm layer ships as parquet (one row per recording, parallel arrays of
`output_timestamps` / `output_texts` / `output_types`). The visualizer's
video-server is deliberately dependency-free (plain node:http, no parquet
reader), so we pre-flatten the parquet here into one JSON keyed by `video_id`
(== CC4D `recording_id`, e.g. "10_18").

The per-recording derivation mirrors `eval/probe_prior_work.py:load_qualcomm`
(the canonical reading of this layer), keeping the original message texts so the
timeline can show them:

  instructions  - every `instruction` event: a step-start guidance message.
  completions   - derived completion times: every instruction after the first
                  marks the previous step's completion, and `feedback_finish_all`
                  marks the last. (instr[1:] + finish_all)
  successes     - `feedback_action_aligned_success` confirmations.
  mistakes      - `feedback_action_aligned_mistake_<class>_error` events; the
                  timestamp is when the mistake first becomes visible (per the
                  paper, a median ~8 s before the step ends).
  finish        - the `feedback_finish_all` time (null if absent).

Output: data/qualcomm_interactive_cooking/qualcomm_timeline.json (gitignored
with the rest of data/). Re-run after refreshing the dataset.

Usage: python3 scripts/export_qualcomm_timeline.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QIC = ROOT / "data" / "qualcomm_interactive_cooking" / "main"
OUT = ROOT / "data" / "qualcomm_interactive_cooking" / "qualcomm_timeline.json"


def build():
    out = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(QIC / f"{split}-00000-of-00001.parquet")
        for _, r in df.iterrows():
            ts = [float(t) for t in r.output_timestamps]
            txts = list(r.output_texts)
            tys = list(r.output_types)

            instructions, successes, mistakes = [], [], []
            finish = None
            for t, txt, ty in zip(ts, txts, tys):
                if ty == "instruction":
                    instructions.append({"t": t, "text": txt})
                elif ty == "feedback_action_aligned_success":
                    successes.append({"t": t, "text": txt})
                elif ty == "feedback_finish_all":
                    finish = t
                elif "mistake" in ty:
                    cls = ty.split("mistake_")[1].replace("_error", "")
                    mistakes.append({"t": t, "class": cls, "text": txt})

            # Completion of step k coincides with instruction k+1 (the model
            # only advances guidance once the current step is done); the final
            # step completes at finish_all.
            instr_t = [i["t"] for i in instructions]
            completions = sorted(instr_t[1:] + ([finish] if finish is not None else []))

            out[r.video_id] = {
                "recipe": r.activity_name,
                "split": split,
                "end": max(ts) if ts else 0.0,
                "finish": finish,
                "instructions": instructions,
                "completions": completions,
                "successes": successes,
                "mistakes": mistakes,
            }
    return out


def main():
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)
    n_mist = sum(len(v["mistakes"]) for v in data.values())
    n_instr = sum(len(v["instructions"]) for v in data.values())
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data)} recordings, "
          f"{n_instr} instructions, {n_mist} mistakes")


if __name__ == "__main__":
    main()
