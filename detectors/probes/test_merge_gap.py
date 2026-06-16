#!/usr/bin/env python3
"""Test the fragmentation hypothesis: if the optimized detector's extra runs are
FRAGMENTS of long microwave runs (split by reduced smoothing), raising merge_gap_s
should re-merge them -> fewer false runs, same recall. Sweep merge_gap in {5,10,15}
over all microwave recipes with the optimized params otherwise fixed."""
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


def main():
    opt, _ = dl.load_frozen_params()
    gaps = [5.0, 10.0, 15.0]
    print(f"{'recipe':<24}" + ''.join(f"|gap{int(g):>2}: fuse fls/rec" for g in gaps))
    tot = {g: collections.Counter() for g in gaps}
    totf = {g: 0 for g in gaps}
    totrec = 0
    for aidx, (name, mwsteps) in MW.items():
        recs = recs_for(aidx)
        line = f"{name:<24}"
        for g in gaps:
            p = copy.deepcopy(opt); p['merge_gap_s'] = g
            agg = collections.Counter(); false_total = 0
            for rec in recs:
                fs, x = dl.load_audio_16k(rec)
                runs = dl.detect_hum_runs(x, fs, p)
                beeps = [b['t'] for b in dl.detect_beeps(x, fs, opt)]
                matched = set()
                for sid in mwsteps:
                    w = step_win(rec, sid)
                    if w is None:
                        continue
                    s, e = w; agg['present'] += 1
                    idx = [i for i, r in enumerate(runs) if r[0] <= e + TOL and r[1] >= s - TOL]
                    matched.update(idx)
                    agg['fused'] += (bool(idx) or any(abs(bt - e) <= TOL for bt in beeps))
                false_total += len(runs) - len(matched)
            pr = agg['present']
            line += f"|     {agg['fused']/pr:>5.2f}{false_total/len(recs):>8.2f}"
            for k in ('present', 'fused'):
                tot[g][k] += agg[k]
            totf[g] += false_total
        print(line)
        totrec += len(recs)
    line = f"{'OVERALL':<24}"
    for g in gaps:
        line += f"|     {tot[g]['fused']/tot[g]['present']:>5.2f}{totf[g]/totrec:>8.2f}"
    print(line)


if __name__ == '__main__':
    main()
