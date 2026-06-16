#!/usr/bin/env python3
"""Attribute the false-run jump: run BOTH the original frozen params (min_run=20,
smooth=21) and the optimized short-cycle params (min_run=8, smooth=7) across ALL
microwave recipes, all recordings. Reports fused recall + false/rec per recipe for
each, so we can see exactly what the short-cycle optimization cost in false runs.
Audio loaded once per recording. Also flags how many 'false' runs are FRAGMENTS
(a short run in a recording that has a matched microwave run nearby)."""
import json, os, sys, copy, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
TOL = 15.0
MW = {'1': ('Microwave Egg Sandwich', [4, 8, 10]), '2': ('Dressed Up Meatballs', [20, 25]),
      '3': ('Microwave Mug Pizza', [32]), '4': ('Ramen', [45]),
      '7': ('Breakfast Burritos', [73]), '8': ('Spiced Hot Chocolate', [89, 83]),
      '9': ('Microwave French Toast', [91, 94]), '13': ('Butter Corn Cup', [133, 134]),
      '26': ('Mug Cake', [312]), '27': ('Cheese Pimiento', [318, 323])}


def step_win(rec, sid):
    s = {x['step_id']: x for x in ANN[rec]['steps']}.get(sid)
    return None if (not s or s['start_time'] < 0) else (float(s['start_time']), float(s['end_time']))


def recs_for(a):
    return sorted([r for r in ANN if r.split('_')[0] == a
                   and os.path.exists(f'{dl.AUDIO_DIR}/{r}_16k.wav')],
                  key=lambda r: int(r.split('_')[1]))


def score_set(recs, mwsteps, audio, beeps_by_rec, p):
    agg = collections.Counter()
    false_total = 0
    frag_false = 0   # false run whose recording also has a matched microwave run
    for rec in recs:
        x, fs = audio[rec]
        runs = dl.detect_hum_runs(x, fs, p)
        beeps = beeps_by_rec[rec]
        wins = [step_win(rec, sid) for sid in mwsteps]
        wins = [w for w in wins if w]
        matched = set()
        for w in wins:
            s, e = w
            agg['present'] += 1
            idx = [i for i, r in enumerate(runs) if r[0] <= e + TOL and r[1] >= s - TOL]
            matched.update(idx)
            h = bool(idx); b = any(abs(bt - e) <= TOL for bt in beeps)
            agg['hum'] += h; agg['beep'] += b; agg['fused'] += (h or b)
        nf = len(runs) - len(matched)
        false_total += nf
        if nf and matched:           # recording has both a real hit and extra runs
            frag_false += nf
    return agg, false_total, frag_false


def main():
    opt, _ = dl.load_frozen_params()              # current = optimized
    base = copy.deepcopy(opt)
    base['min_run_s'] = 20.0; base['feat_med_frames'] = 21; base['med_frames'] = 21

    print(f"{'recipe':<24}|{'BASELINE min=20':^22}|{'OPTIMIZED min=8':^22}")
    print(f"{'':<24}|{'fused':>8}{'fls/rec':>9}{'':>5}|{'fused':>8}{'fls/rec':>9}{'frag':>5}")
    G = {'base': collections.Counter(), 'opt': collections.Counter()}
    Gf = {'base': [0, 0], 'opt': [0, 0]}  # [false, frag]
    Grecs = 0
    for aidx, (name, mwsteps) in MW.items():
        recs = recs_for(aidx)
        audio = {r: (lambda fx: (fx[1], fx[0]))(dl.load_audio_16k(r)) for r in recs}
        beeps = {r: [b['t'] for b in dl.detect_beeps(audio[r][0], audio[r][1], opt)] for r in recs}
        ab, fb, frb = score_set(recs, mwsteps, audio, beeps, base)
        ao, fo, fro = score_set(recs, mwsteps, audio, beeps, opt)
        pb, po = ab['present'], ao['present']
        print(f"{name:<24}|{ab['fused']/pb:>8.2f}{fb/len(recs):>9.2f}{'':>5}|"
              f"{ao['fused']/po:>8.2f}{fo/len(recs):>9.2f}{fro:>5}")
        for k in ('present', 'hum', 'beep', 'fused'):
            G['base'][k] += ab[k]; G['opt'][k] += ao[k]
        Gf['base'][0] += fb; Gf['base'][1] += frb
        Gf['opt'][0] += fo; Gf['opt'][1] += fro
        Grecs += len(recs)
    pb, po = G['base']['present'], G['opt']['present']
    print(f"{'OVERALL':<24}|{G['base']['fused']/pb:>8.2f}{Gf['base'][0]/Grecs:>9.2f}{'':>5}|"
          f"{G['opt']['fused']/po:>8.2f}{Gf['opt'][0]/Grecs:>9.2f}{Gf['opt'][1]:>5}")
    print(f"\nbaseline: fused {G['base']['fused']}/{pb}, false {Gf['base'][0]} "
          f"({Gf['base'][1]} in recs w/ a hit)")
    print(f"optimized: fused {G['opt']['fused']}/{po}, false {Gf['opt'][0]} "
          f"({Gf['opt'][1]} in recs w/ a hit = likely fragments/other-appliance)")


if __name__ == '__main__':
    main()
