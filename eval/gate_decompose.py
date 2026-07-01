#!/usr/bin/env python3
"""Per-reminder failure decomposition for a reminders-GATED arm (one closed-set call per GT
reminder), and optional head-to-head between two arms.

For each GT reminder (anchor_step, subtype, t) we find the gated call at that step+moment and bucket:
  HIT        the call fired the reminder's subtype
  WRONG_TYPE fired something, but not that subtype
  MISS       fired nothing (answered "0")
  NO_CALL    no call landed on this reminder (shouldn't happen for a clean gated run)

Usage:
  python eval/gate_decompose.py experiments/qualcomm_run/gpt54_gate_rem
  python eval/gate_decompose.py experiments/qualcomm_run/cs_gate_rem experiments/qualcomm_run/gpt54_gate_rem
"""
import json, os, sys, glob
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(BASE, "data", "cc4d_proactive")


def gt_reminders(rid):
    p = os.path.join(GT, rid + ".json")
    return json.load(open(p)).get("reminders", []) if os.path.exists(p) else []


def decompose(arm_dir):
    rows = []  # (rid, subtype, bucket)
    for f in sorted(glob.glob(os.path.join(arm_dir, "*.json"))):
        rid = os.path.basename(f)[:-5]
        res = json.load(open(f))
        calls = res.get("calls", [])
        for r in gt_reminders(rid):
            sid, te, sub = r.get("anchor_step"), round(float(r["t"]), 1), r.get("subtype")
            cand = [c for c in calls if c.get("step_id") == sid and abs(c.get("t", -9) - te) <= 0.6]
            if not cand:
                rows.append((rid, sub, "NO_CALL")); continue
            fired = cand[0].get("fired", [])
            if sub in fired:
                rows.append((rid, sub, "HIT"))
            elif fired:
                rows.append((rid, sub, "WRONG_TYPE"))
            else:
                rows.append((rid, sub, "MISS"))
    return rows


def summarize(name, rows):
    n = len(rows)
    by_bucket = defaultdict(int)
    by_sub = defaultdict(lambda: defaultdict(int))
    for _, sub, b in rows:
        by_bucket[b] += 1
        by_sub[sub][b] += 1
        by_sub[sub]["_n"] += 1
    print(f"\n=== {name}  ({n} GT reminders) ===")
    for b in ("HIT", "WRONG_TYPE", "MISS", "NO_CALL"):
        c = by_bucket[b]
        if c:
            print(f"  {b:11s} {c:4d}  ({100*c/n:4.0f}%)")
    # recall(detection-any) and recall(typed) as a quick F1-style read of the ceiling
    hit = by_bucket["HIT"]
    print(f"  -> typed-HIT recall {hit/n:.3f}   any-detection recall {(hit+by_bucket['WRONG_TYPE'])/n:.3f}")
    print("  per-subtype typed-HIT:")
    for sub in sorted(by_sub):
        s = by_sub[sub]
        print(f"    {sub:12s} {s['HIT']:3d}/{s['_n']:<3d} ({100*s['HIT']/s['_n']:3.0f}%)")
    return by_bucket, by_sub


def main():
    arms = sys.argv[1:]
    if not arms:
        print(__doc__); sys.exit(1)
    results = {}
    for a in arms:
        rows = decompose(a)
        results[a] = summarize(os.path.basename(a.rstrip("/")), rows)
    if len(arms) == 2:
        (ba, _), (bb, _) = results[arms[0]], results[arms[1]]
        na, nb = os.path.basename(arms[0].rstrip("/")), os.path.basename(arms[1].rstrip("/"))
        print(f"\n=== head-to-head (typed-HIT) ===")
        print(f"  {na}: {ba['HIT']}   {nb}: {bb['HIT']}")


if __name__ == "__main__":
    main()
