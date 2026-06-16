#!/usr/bin/env python3
"""Re-score the A4 sizzle/cook DSP detector against TIGHTENED cook windows.

OLD window = [min start, max end] over ALL cook sub-steps (includes silent adds,
empty-pan 'heat oil', off-heat 'set') -> dilutes coverage.
NEW window = UNION of the ON-HEAT-WITH-FOOD sub-steps only (cook/saute/simmer/
continue-cooking/keep-mixing/toast where food is frying), scored as a union of
intervals (inter-step gaps don't count). This measures: when food is actively
cooking, does the sizzle detector fire?

Same detector (detect_sizzle_runs, frozen 23_5 params), same metric family
(coverage; recall = coverage>=0.40). A4 only (the 'sizzle detector'). CPU."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
COV_MIN = 0.40
SIZZLE = dict(nfft=1024, hop=256, band_lo=1500.0, band_hi=7000.0, block_s=0.5,
              roll_win_s=45.0, baseline_pct=20, level_db=8.0, merge_gap_s=20.0, min_run_s=30.0)
# (name, ALL cook steps [old], ON-HEAT-with-food steps [new])
RECIPES = {
    '25': ('Pan Fried Tofu',     {279,286,292,285,290,277,278,289,281,288}, {281,285,288,289,290,292}),
    '23': ('Broccoli Stir Fry',  {266,258,265,261,253,274,256,264},         {253,264,274}),
    '20': ('Sauteed Mushrooms',  {207,208,209,213,217,218,219,220},         {207,213}),
    '16': ('Scrambled Eggs',     {160,162,165,166,168,170,171,172,175,178,179}, {165,166,170,171}),
    '22': ('Herb Omelet',        {235,238,240,241,245},                     {238,240,245}),
    '15': ('Tomato Chutney',     {139,141,142,144,146,147,148,152,153,157}, {148,152,153}),
    '18': ('Zoodles',            {192,199,201,202,204},                     {192,202}),
    '21': ('Banana Pancakes',    {223,226,228,230,231},                     {226,230}),
    '29': ('Caprese Bruschetta', {352},                                     {352}),
}


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def union_len_overlap(intervals, runs):
    """return (total union length, sizzle-seconds overlapping the union)."""
    if not intervals:
        return 0.0, 0.0
    iv = sorted([list(x) for x in intervals])
    merged = [iv[0]]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    ulen = sum(b - a for a, b in merged)
    cov = sum(ov(ra, rb, a, b) for ra, rb in runs for a, b in merged)
    return ulen, cov


def main():
    hb, _ = dl.load_frozen_params()
    print(f"{'recipe':<20}{'recs':>5}{'OLD_R':>7}{'OLD_cov':>8}{'NEW_R':>7}{'NEW_cov':>8}{'dR':>7}")
    summ = {}
    nold = nnew = 0
    for aidx, (name, allcook, onheat) in RECIPES.items():
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        oh, oc, nh, nc, n = 0, [], 0, [], 0
        for rec in recs:
            steps = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}
            oldw = [(steps[s]['start_time'], steps[s]['end_time']) for s in allcook if s in steps]
            neww = [(steps[s]['start_time'], steps[s]['end_time']) for s in onheat if s in steps]
            if not oldw or not neww:
                continue
            n += 1
            fs16, x16 = dl.load_audio_16k(rec)
            sizz = dl.detect_sizzle_runs(x16, fs16, SIZZLE)
            # OLD: span coverage
            cs, ce = min(a for a, b in oldw), max(b for a, b in oldw)
            covo = sum(ov(a, b, cs, ce) for a, b in sizz) / (ce - cs)
            oc.append(covo); oh += int(covo >= COV_MIN)
            # NEW: on-heat union coverage
            ulen, covov = union_len_overlap(neww, sizz)
            covn = covov / ulen if ulen > 0 else 0.0
            nc.append(covn); nh += int(covn >= COV_MIN)
        ro, rn = oh / n, nh / n
        summ[aidx] = {'recipe': name, 'n': n, 'old_recall': round(ro, 3),
                      'old_cov': round(float(np.mean(oc)), 3), 'new_recall': round(rn, 3),
                      'new_cov': round(float(np.mean(nc)), 3)}
        print(f"{name:<20}{n:>5}{ro:>7.2f}{np.mean(oc):>8.2f}{rn:>7.2f}{np.mean(nc):>8.2f}{rn-ro:>+7.2f}")
    aud = [v for k, v in summ.items() if k in ('25', '23', '20', '16', '22', '18')]
    print(f"\nAUDIBLE-cook recipes mean: OLD R={np.mean([v['old_recall'] for v in aud]):.2f}  "
          f"NEW R={np.mean([v['new_recall'] for v in aud]):.2f}")
    print(f"ALL 9 recipes mean:        OLD R={np.mean([v['old_recall'] for v in summ.values()]):.2f}  "
          f"NEW R={np.mean([v['new_recall'] for v in summ.values()]):.2f}")
    print(f"total testing points (recordings): {sum(v['n'] for v in summ.values())}")
    json.dump(summ, open(f'{os.path.dirname(__file__)}/results_rescore_onheat.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
