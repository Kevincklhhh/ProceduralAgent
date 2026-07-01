#!/usr/bin/env python3
"""Focused T1 scorer for the hot-chocolate head-to-head: recognition vs completion-criteria.

Active-Step Accuracy (per 1 s tick, the predicted step must be in the GT set of steps active at
that instant) -- the same metric as eval/eval_score_corpus.py's stage_acc -- restricted to the
16 spicedhotchocolate recordings, plus the criteria arm's stall metrics from _meta.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_score_corpus import load_steps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = "8_11 8_15 8_16 8_19 8_20 8_25 8_26 8_3 8_30 8_31 8_33 8_35 8_40 8_44 8_45 8_50".split()
ARMS = sys.argv[1:] or ["recognition_i10", "criteria_i10"]
RES = os.path.join(BASE, "experiments/t1_baseline")


def active_step_acc(segs, ivs):
    """Return (n_ticks, n_correct, n_active_ticks, n_active_correct). 'other' allowed only in gaps."""
    if not segs:
        return 0, 0, 0, 0
    T = int(math.ceil(max(e for _, e, _ in segs)))
    n = nc = ne = nce = 0
    for i in range(T):
        t = i + 0.5
        G = {s[2] for s in segs if s[0] <= t < s[1]}
        phits = [iv for iv in ivs if iv['start_s'] <= t < iv['end_s']]
        pl = max(phits, key=lambda iv: iv['start_s'])['stage'] if phits else 'other'
        ok = (pl in G) if G else (pl == 'other')
        n += 1; nc += ok
        if G:
            ne += 1; nce += ok
    return n, nc, ne, nce


def main():
    steps_by_rec = load_steps()
    for arm in ARMS:
        print(f"\n========== {arm} ==========")
        print(f"{'rid':6} {'actAcc':7} {'acc(excl-other)':16} {'reached_end':11} {'final/N':8} {'maxDwell':8}")
        N = NC = NE = NCE = 0
        stalls = 0; have = 0
        for rid in IDS:
            f = os.path.join(RES, arm, f"{rid}.json")
            if not os.path.exists(f):
                print(f"{rid:6} (missing)")
                continue
            have += 1
            d = json.load(open(f))
            steps = steps_by_rec.get(rid, [])
            segs = [(float(s['start_time']), float(s['end_time']), s['step_id'])
                    for s in steps if s['start_time'] >= 0 and s['end_time'] >= 0]
            n, nc, ne, nce = active_step_acc(segs, d['stage_intervals'])
            N += n; NC += nc; NE += ne; NCE += nce
            m = d.get('_meta', {})
            re_ = m.get('reached_end'); fin = f"{m.get('final_step_idx','-')}/{m.get('n_steps','-')}"
            md = m.get('max_dwell_ticks', '-')
            if re_ is False:
                stalls += 1
            acc = nc / n if n else float('nan')
            acce = nce / ne if ne else float('nan')
            print(f"{rid:6} {acc:6.3f}  {acce:14.3f}   {str(re_):11} {fin:8} {md}")
        pa = NC / N if N else float('nan')
        pae = NCE / NE if NE else float('nan')
        print(f"{'POOL':6} {pa:6.3f}  {pae:14.3f}   "
              f"{'stalled '+str(stalls)+'/'+str(have) if 'criteria' in arm else ''}")


if __name__ == "__main__":
    main()
