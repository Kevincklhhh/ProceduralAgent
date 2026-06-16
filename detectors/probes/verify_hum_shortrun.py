#!/usr/bin/env python3
"""Verify the winning short-cycle hum params (min_run=8, smooth=7 frames/1.8s) with
the FULL stack: hum, beep, and hum-or-beep fusion on both recipes, plus a relaxed-beep
variant. Confirms no SHC regression and reports the real fused headline."""
import json, os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
TOL = 15.0
SHC = ['8_11', '8_15', '8_16', '8_19', '8_20', '8_25', '8_26', '8_3',
       '8_30', '8_31', '8_33', '8_35', '8_40', '8_44', '8_45', '8_50']


def step_win(rec, sid):
    s = {x['step_id']: x for x in ANN[rec]['steps']}.get(sid)
    return None if (not s or s['start_time'] < 0) else (float(s['start_time']), float(s['end_time']))


def recs_for(a):
    return sorted([r for r in ANN if r.split('_')[0] == a
                   and os.path.exists(f'{dl.AUDIO_DIR}/{r}_16k.wav')],
                  key=lambda r: int(r.split('_')[1]))


def evaluate(recs, steps, p, relax_beep=False):
    bp = copy.deepcopy(p)
    if relax_beep:
        bp['tone_ratio'] = 0.25; bp['peak_db_min'] = -80.0
    agg = {s: {'p': 0, 'hum': 0, 'beep': 0, 'fused': 0} for s in steps}
    false_total = 0
    for rec in recs:
        fs, x = dl.load_audio_16k(rec)
        runs = dl.detect_hum_runs(x, fs, p)
        beeps = [b['t'] for b in dl.detect_beeps(x, fs, bp)]
        matched = set()
        for sid in steps:
            w = step_win(rec, sid)
            if w is None:
                continue
            s, e = w
            agg[sid]['p'] += 1
            idx = [i for i, r in enumerate(runs) if r[0] <= e + TOL and r[1] >= s - TOL]
            matched.update(idx)
            h = bool(idx); b = any(abs(bt - e) <= TOL for bt in beeps)
            agg[sid]['hum'] += h; agg[sid]['beep'] += b; agg[sid]['fused'] += (h or b)
        false_total += len(runs) - len(matched)
    out = {}
    for s in steps:
        a = agg[s]
        out[s] = {k: round(a[k] / a['p'], 2) if a['p'] else None
                  for k in ('hum', 'beep', 'fused')}
        out[s]['n'] = a['p']
    return out, false_total


def show(title, out, false_total):
    print(f'\n{title}  (false hum runs total = {false_total})')
    print(f'  {"step":<8}{"n":>3}{"hum":>7}{"beep":>7}{"fused":>7}')
    tp = tf = 0
    for s, v in out.items():
        print(f'  {s:<8}{v["n"]:>3}{v["hum"]:>7.2f}{v["beep"]:>7.2f}{v["fused"]:>7.2f}')
        tp += v['n']; tf += round(v['fused'] * v['n'])
    print(f'  {"ALL":<8}{tp:>3}{"":>7}{"":>7}{tf / tp:>7.2f}')


p = dict(dl.load_frozen_params()[0])
p['min_run_s'] = 8; p['feat_med_frames'] = 7; p['med_frames'] = 7

a1, a8 = recs_for('1'), SHC
o1, f1 = evaluate(a1, [4, 8, 10], p)
o8, f8 = evaluate(a8, [89, 83], p)
o1r, f1r = evaluate(a1, [4, 8, 10], p, relax_beep=True)
print('=== OPTIMIZED hum (min_run=8, smooth=1.8s), frozen beep ===')
show('Microwave Egg Sandwich (act 1)', o1, f1)
show('Spiced Hot Chocolate (act 8) [regression check]', o8, f8)
print('\n=== same hum + RELAXED beep (tone>=0.25, peak>=-80dB) on act 1 ===')
show('Microwave Egg Sandwich (act 1)', o1r, f1r)
