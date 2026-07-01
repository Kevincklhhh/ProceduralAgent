#!/usr/bin/env python3
"""Backfill the per-call VLM trace (`calls`) into turn-based qualcomm_run arm JSONs that were
written before the trace existed, so the visualizer's T1/T2 "VLM calls + context" track works
for them too.

The turn-based tick loop is fully deterministic, so the calls are reconstructible offline (NO
VLM): at each tick t (= interval, 2*interval, ...) the current step is the oracle GT-active step;
a mistake call happens iff that step has a candidate menu and hasn't fired yet; the call's window
is [t-window, t]; and the call FIRED iff an event exists at (t, step). Only `answer`/`latency_s`
are unrecoverable (left blank / null) — everything the timeline needs (when, window, frames,
fired) is exact.

Only handles mode=turnbased (streaming's pointer/completion logic differs). Adds res['calls']
and marks _meta.calls_reconstructed=True; idempotent (skips arms that already have real calls).

Usage: python eval/backfill_calls.py --arms qwen36_zs_turn_cs,qwen36_zs_turn_cs_w12,qwen36_zs_turn
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_qualcomm_zeroshot as B

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QRUN = os.path.join(BASE, "experiments", "qualcomm_run")
TIMELINE = json.load(open(os.path.join(BASE, "data/qualcomm_interactive_cooking/qualcomm_timeline.json")))
STEM_OF = __import__("qualcomm_adapter")._RECIPE_STEM


def reconstruct(rid, res):
    meta = res.get("_meta", {})
    if meta.get("mode") != "turnbased":
        return None                                  # streaming logic differs; skip
    ps = meta.get("prompt_style", "freetext")
    interval = float(meta.get("interval_s", 5.0))
    window = float(meta.get("window_s") or interval)
    sample_fps = float(meta.get("sample_fps", 1.0))
    max_frames = 8                                    # baseline default (not stored in meta)
    n_frames = min(32, max(max_frames, int(round(window * sample_fps))))

    stem = STEM_OF.get(TIMELINE.get(rid, {}).get("recipe"), TIMELINE.get(rid, {}).get("recipe"))
    checks = B.load_checks(stem)                      # step_id -> [(subtype, claim)] (SUBSET only)
    if ps == "closedset":
        callable_steps = {s for s, c in checks.items() if c}      # non-empty menu only
    else:
        callable_steps = set(checks.keys())                       # any criteria node

    spans = B.oracle_spans(rid)                       # [(sid, s, e)]
    if not spans:
        return []
    t_end = max(e for _, _, e in spans)

    def current_step(t):
        active = [sid for (sid, s, e) in spans if s <= t < e]
        return active[-1] if active else None         # last active == the run's choice

    # events grouped by (rounded-tick, step) -> fired subtypes
    ev_at = {}
    for e in res.get("events", []):
        ev_at.setdefault((round(float(e["t"]), 1), e.get("step_id")), []).append(e.get("subtype"))

    calls, fired_steps = [], set()
    t = interval
    while t <= t_end + 1e-6:
        sid = current_step(t)
        if sid is not None and sid in callable_steps and sid not in fired_steps:
            tr = round(t, 1)
            fired = ev_at.get((tr, sid), [])
            if fired:
                fired_steps.add(sid)
            calls.append({"t": tr, "kind": "mistake", "step_id": sid,
                          "win_start": round(max(0.0, t - window), 1), "win_end": tr,
                          "n_frames": n_frames, "latency_s": None,
                          "fired": list(fired), "answer": ""})
        t += interval
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma list of arm dirs under experiments/qualcomm_run")
    ap.add_argument("--force", action="store_true", help="overwrite even real (non-reconstructed) calls")
    args = ap.parse_args()
    for arm in args.arms.split(","):
        d = os.path.join(QRUN, arm)
        if not os.path.isdir(d):
            print(f"skip {arm}: no dir"); continue
        n = skipped = empty = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            p = os.path.join(d, f)
            res = json.load(open(p))
            if res.get("calls") and not res["_meta"].get("calls_reconstructed") and not args.force:
                skipped += 1; continue               # real trace already present
            calls = reconstruct(f[:-5], res)
            if calls is None:
                skipped += 1; continue               # not turn-based
            res["calls"] = calls
            res.setdefault("_meta", {})["calls_reconstructed"] = True
            json.dump(res, open(p, "w"), indent=1)
            n += 1; empty += (len(calls) == 0)
        print(f"{arm}: backfilled {n} recordings ({empty} with 0 calls), skipped {skipped}")


if __name__ == "__main__":
    main()
