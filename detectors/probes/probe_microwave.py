#!/usr/bin/env python3
"""Probe: A1 hum + A2 beep on a SECOND microwave recipe to find the DSP limits.

The microwave hum+beep DSP stack was validated only on Spiced Hot Chocolate
(activity 8), whose microwave runs are ~60 s (1 min). This probe runs the SAME
FROZEN detectors (detectors_lib, params from results_hum_beep.json, no tuning)
on Microwave Egg Sandwich (activity 1), whose three microwave cycles are much
shorter -- 30 s, 15-30 s, and ~10 s -- the last one BELOW the validated
min_run_s = 20 s floor. Goal: see where the cheap detector holds and where it
breaks.

Scoring per recording (present steps only; start<0 = skipped, excluded):
  microwave steps = {4 (~30s), 8 (15-30s more), 10 (~10s cheese melt)}.
  hum HIT  if a kept hum run overlaps [s-TOL, e+TOL].
  beep HIT if a beep lands within TOL of a microwave-step END.
  false hum run = a kept run overlapping NO microwave-step window.
Outputs results_microwave_act1.json. Usage: python probe_microwave.py [--only 1_7]
"""
import json, os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
PROBE_DIR = os.path.join(BASE, 'detectors', 'probes')

ACTIVITY = '1'
MW_STEPS = [4, 8, 10]   # three microwave cycles (local step ids for activity 1)
TOL = 15.0


def overlaps(run, s, e, tol=TOL):
    return run[0] <= e + tol and run[1] >= s - tol


def step_win(rec, sid):
    steps = {x['step_id']: x for x in ANN[rec]['steps']}
    s = steps.get(sid)
    if not s or s['start_time'] < 0:
        return None
    return (float(s['start_time']), float(s['end_time']))


def run_one(rec, hb):
    fs16, x16 = dl.load_audio_16k(rec)
    hums = dl.detect_hum_runs(x16, fs16, hb)
    beeps = dl.detect_beeps(x16, fs16, hb)
    beep_t = [b['t'] for b in beeps]

    res = {'recording': rec, 'n_hum_runs': len(hums),
           'hum_runs': [[round(a, 1), round(b, 1), round(b - a, 1)] for a, b in hums],
           'n_beeps': len(beeps), 'steps': {}}
    matched = set()
    for sid in MW_STEPS:
        w = step_win(rec, sid)
        if w is None:
            res['steps'][sid] = {'present': False}
            continue
        s, e = w
        hit_runs = [i for i, r in enumerate(hums) if overlaps(r, s, e)]
        matched.update(hit_runs)
        beep_hit = any(abs(bt - e) <= TOL for bt in beep_t)
        res['steps'][sid] = {'present': True, 'win': [round(s, 1), round(e, 1)],
                             'dur_window_s': round(e - s, 1),
                             'hum_hit': bool(hit_runs),
                             'hum_run_dur_s': [round(hums[i][1] - hums[i][0], 1) for i in hit_runs],
                             'beep_hit': bool(beep_hit)}
    res['false_hum_runs'] = len(hums) - len(matched)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    hb, _ = dl.load_frozen_params()
    recs = sorted([r for r in ANN if r.split('_')[0] == ACTIVITY
                   and os.path.exists(f'{dl.AUDIO_DIR}/{r}_16k.wav')],
                  key=lambda r: int(r.split('_')[1]))
    if args.only:
        recs = [args.only]

    out = []
    # aggregate counters per microwave step
    agg = {sid: {'present': 0, 'hum_hit': 0, 'beep_hit': 0} for sid in MW_STEPS}
    false_total = 0
    for rec in recs:
        r = run_one(rec, hb)
        out.append(r)
        false_total += r['false_hum_runs']
        for sid in MW_STEPS:
            st = r['steps'][sid]
            if st.get('present'):
                agg[sid]['present'] += 1
                agg[sid]['hum_hit'] += int(st['hum_hit'])
                agg[sid]['beep_hit'] += int(st['beep_hit'])
        print(f"{rec:8s} hum_runs={r['n_hum_runs']} beeps={r['n_beeps']} "
              f"false={r['false_hum_runs']}  " +
              ' '.join(f"s{sid}:{'H' if r['steps'][sid].get('hum_hit') else '-'}"
                       f"{'B' if r['steps'][sid].get('beep_hit') else '-'}"
                       f"({r['steps'][sid]['dur_window_s']}s)" if r['steps'][sid].get('present')
                       else f"s{sid}:skip" for sid in MW_STEPS))

    summary = {'activity': ACTIVITY, 'recipe': 'Microwave Egg Sandwich',
               'n_recordings': len(recs), 'tol_s': TOL,
               'min_run_s_floor': hb['min_run_s'],
               'false_hum_runs_total': false_total, 'per_step': {}}
    for sid in MW_STEPS:
        a = agg[sid]
        summary['per_step'][sid] = {
            'present': a['present'],
            'hum_recall': round(a['hum_hit'] / a['present'], 3) if a['present'] else None,
            'beep_recall': round(a['beep_hit'] / a['present'], 3) if a['present'] else None}
    print('\n=== SUMMARY (frozen detector, no tuning) ===')
    print(json.dumps(summary, indent=2))
    json.dump({'summary': summary, 'recordings': out},
              open(f'{PROBE_DIR}/results_microwave_act1.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
