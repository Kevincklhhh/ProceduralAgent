#!/usr/bin/env python3
"""Probe: A4 sizzle — test whether a cheap DSP sizzle/fry detector can anchor the
COOK phase of a NO-MICROWAVE stovetop recipe, the way the microwave hum (A1)
anchored Spiced Hot Chocolate. This is the catalog's top-priority untested probe
("gates all stovetop recipes").

Recipe: Broccoli Stir Fry (activity 23), all 16 recordings.
Structure: a long PREP phase (chop/mince/whisk/add-to-bowl) then a COOK phase
(heat oil -> add to skillet -> cook stirring -> pour sauce -> thicken).

Detector (detectors_lib.detect_sizzle_runs): sustained elevation of 1.5-7 kHz band
energy above the per-recording floor AND high spectral flatness (noise-like, not
tonal). Tuned on 23_5, FROZEN, eval on the other 15.

Scoring per recording (present steps only; start<0 = skipped, excluded):
  COOK steps  = {266 heat-oil, 258/265/261 add-to-skillet, 253/274/264 cook,
                 256 pour-sauce}  -> cook_phase = [min start, max end].
  coverage    = sizzle seconds overlapping cook_phase / cook_phase length.
  HIT (cook anchored) if coverage >= COV_MIN.
  false_sizzle_prep_s = sizzle seconds overlapping the PREP phase (should be ~0).
  hum cross-check     = n hum runs (should be 0; there is no microwave).
Outputs results_sizzle.json. Usage: python probe_sizzle.py [--only 23_5]
"""
import json, os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
PROBE_DIR = os.path.join(BASE, 'detectors', 'probes')

# Per-activity cook-step sets (steps where food is in the hot pan -> sizzle).
CONFIG = {
    '23': dict(name='Broccoli Stir Fry',
               cook_steps={266, 258, 265, 261, 253, 274, 256, 264}, tune='23_5'),
    '25': dict(name='Pan Fried Tofu',
               cook_steps={279, 286, 292, 285, 290, 277, 278, 289, 281, 288},
               tune='25_4'),
    # below: same FROZEN 23_5 params, no per-recipe tuning (cross-recipe generalization)
    '16': dict(name='Scrambled Eggs',
               cook_steps={160, 162, 165, 166, 168, 170, 171, 172, 175, 178, 179},
               tune=None),
    '20': dict(name='Sauteed Mushrooms',
               cook_steps={207, 208, 209, 213, 217, 218, 219, 220}, tune=None),
    '22': dict(name='Herb Omelet w/ Fried Tomatoes',
               cook_steps={235, 238, 240, 241, 245}, tune=None),
    '15': dict(name='Tomato Chutney',
               cook_steps={139, 141, 142, 144, 146, 147, 148, 152, 153, 157}, tune=None),
    '18': dict(name='Zoodles',
               cook_steps={192, 199, 201, 202, 204}, tune=None),
    '21': dict(name='Blender Banana Pancakes',
               cook_steps={223, 226, 228, 230, 231}, tune=None),
    '29': dict(name='Caprese Bruschetta (toast)',
               cook_steps={352}, tune=None),
}
COV_MIN = 0.40

# Sizzle params, tuned on 23_5 then FROZEN (16 kHz front-end). v2: SUSTAINED-LEVEL
# (rolling-median) gate -- the discriminator is persistence, not spectrum (grounded
# in EPIC-SOUNDS: sizzling is long-form/background, spectrally inseparable from
# intermittent broadband bursts like water/rustle/whisk).
SIZZLE_PARAMS = dict(
    nfft=1024, hop=256,
    band_lo=1500.0, band_hi=7000.0,
    block_s=0.5,            # block-reduce band energy to ~2 Hz (median per 0.5 s)
    roll_win_s=45.0,        # rolling-median window -> sustained level
    baseline_pct=20,        # per-recording prep sustained floor
    level_db=8.0,           # frying holds rolling-median +17..+30 dB; prep bursts ~0
    merge_gap_s=20.0,       # bridge stir pauses within the cook phase
    min_run_s=30.0,         # long-form: a true fry persists; rejects burst clusters
)

SHC_HUM_STEPS = [89, 83]  # for the cross-check we only need the hum detector to be quiet


def overlap_s(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def present_steps(rec):
    return [s for s in ANN[rec]['steps'] if s['start_time'] >= 0]


def run_one(rec, hb, cook_steps):
    fs16, x16 = dl.load_audio_16k(rec)
    sizz = dl.detect_sizzle_runs(x16, fs16, SIZZLE_PARAMS)
    hums = dl.detect_hum_runs(x16, fs16, hb)

    steps = present_steps(rec)
    cook = [s for s in steps if s['step_id'] in cook_steps]
    if not cook:
        return {'recording': rec, 'cook': 'absent'}
    cs = min(s['start_time'] for s in cook)
    ce = max(s['end_time'] for s in cook)
    cook_len = ce - cs

    sizz_in_cook = sum(overlap_s(a, b, cs, ce) for a, b in sizz)
    coverage = sizz_in_cook / cook_len if cook_len > 0 else 0.0

    # prep = everything that isn't a cook step, as disjoint windows
    prep = [(s['start_time'], s['end_time']) for s in steps
            if s['step_id'] not in cook_steps]
    false_prep = 0.0
    for a, b in sizz:
        for ps, pe in prep:
            false_prep += overlap_s(a, b, ps, pe)

    first_sizz = min((a for a, b in sizz if b >= cs), default=None)
    onset_lat = round(first_sizz - cs, 1) if first_sizz is not None else None

    return {'recording': rec,
            'n_sizzle_runs': len(sizz),
            'sizzle_runs': [[round(a, 1), round(b, 1), round(b - a, 1)] for a, b in sizz],
            'cook_window': [round(cs, 1), round(ce, 1)], 'cook_len_s': round(cook_len, 1),
            'sizzle_in_cook_s': round(sizz_in_cook, 1), 'coverage': round(coverage, 3),
            'cook_hit': bool(coverage >= COV_MIN),
            'onset_latency_s': onset_lat,
            'false_sizzle_prep_s': round(false_prep, 1),
            'n_hum_runs_falsepos': len(hums)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--activity', default='23', choices=list(CONFIG))
    ap.add_argument('--only')
    args = ap.parse_args()
    cfg = CONFIG[args.activity]
    cook_steps, tune = cfg['cook_steps'], cfg['tune']
    hb, _ = dl.load_frozen_params()
    recs = ([args.only] if args.only
            else sorted([k for k in ANN if k.startswith(args.activity + '_')],
                        key=lambda x: int(x.split('_')[1])))
    per = {r: run_one(r, hb, cook_steps) for r in recs}

    scored = [v for v in per.values() if v.get('cook') != 'absent']
    n = len(scored)
    hits = sum(v['cook_hit'] for v in scored)
    mean_cov = sum(v['coverage'] for v in scored) / n if n else 0.0
    tot_false = sum(v['false_sizzle_prep_s'] for v in scored)
    tot_cook = sum(v['cook_len_s'] for v in scored)
    hum_fp = sum(v['n_hum_runs_falsepos'] for v in scored)
    summary = {'recipe': f"{cfg['name']} (act {args.activity})", 'recordings': n,
               'cook_anchor_recall': round(hits / n, 3) if n else None,
               'mean_coverage': round(mean_cov, 3),
               'false_sizzle_prep_s_total': round(tot_false, 1),
               'cook_len_s_total': round(tot_cook, 1),
               'false_rate_prep_vs_cook': round(tot_false / tot_cook, 3) if tot_cook else None,
               'hum_falsepos_runs_total': hum_fp}
    print(json.dumps(summary, indent=1))
    print('\nrec      cook_win          len  cov   hit  onset  falsePrep_s  humFP  sizzle_runs')
    for r in recs:
        v = per[r]
        if v.get('cook') == 'absent':
            print(f'  {r:7s} (no cook steps)'); continue
        runs = ' '.join(f"{d[2]:.0f}s" for d in v['sizzle_runs'][:6])
        mark = '   [TUNE]' if r == tune else ''
        print(f"  {r:7s} {str(v['cook_window']):17s} {v['cook_len_s']:5.0f} "
              f"{v['coverage']:.2f}  {'Y' if v['cook_hit'] else 'MISS':4s} "
              f"{str(v['onset_latency_s']):>6s} {v['false_sizzle_prep_s']:9.0f}    "
              f"{v['n_hum_runs_falsepos']:3d}  [{v['n_sizzle_runs']}] {runs}{mark}")
    if not args.only:
        out = f"results_sizzle_act{args.activity}.json"
        json.dump({'probe': f'sizzle_act{args.activity}', 'params': SIZZLE_PARAMS,
                   'summary': summary, 'per_recording': per},
                  open(os.path.join(PROBE_DIR, out), 'w'), indent=1)
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
