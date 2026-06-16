#!/usr/bin/env python3
"""Run the OPTIMIZED (now-default) hum+beep detector across ALL microwave recipes,
ALL recordings with audio. Scores per-recipe microwave-RUN recall (hum / beep / fused)
and false hum runs (runs overlapping no microwave window). Curated microwave-RUN step
sets (excludes 'place in microwave-safe bowl' / 'stir the cup' non-run steps; includes
SHC step 83 'heat 1 min' reheat that the keyword filter misses)."""
import json, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
TOL = 15.0
MW = {  # activity_idx -> microwave-RUN step ids, with recipe name
    '1':  ('Microwave Egg Sandwich', [4, 8, 10]),
    '2':  ('Dressed Up Meatballs',   [20, 25]),
    '3':  ('Microwave Mug Pizza',    [32]),
    '4':  ('Ramen',                  [45]),
    '7':  ('Breakfast Burritos',     [73]),
    '8':  ('Spiced Hot Chocolate',   [89, 83]),
    '9':  ('Microwave French Toast', [91, 94]),
    '13': ('Butter Corn Cup',        [133, 134]),
    '26': ('Mug Cake',               [312]),
    '27': ('Cheese Pimiento',        [318, 323]),
}


def step_win(rec, sid):
    s = {x['step_id']: x for x in ANN[rec]['steps']}.get(sid)
    return None if (not s or s['start_time'] < 0) else (float(s['start_time']), float(s['end_time']))


def recs_for(a):
    return sorted([r for r in ANN if r.split('_')[0] == a
                   and os.path.exists(f'{dl.AUDIO_DIR}/{r}_16k.wav')],
                  key=lambda r: int(r.split('_')[1]))


def main():
    hb, _ = dl.load_frozen_params()
    print(f"detector params: min_run_s={hb['min_run_s']} feat_med={hb['feat_med_frames']} "
          f"med={hb['med_frames']}  (optimized)")
    out = []
    grand = collections.Counter()
    print(f"\n{'recipe':<26}{'recs':>5}{'cyc':>5}{'hum':>7}{'beep':>7}{'fused':>7}"
          f"{'false':>7}{'fls/rec':>8}")
    for aidx, (name, mwsteps) in MW.items():
        recs = recs_for(aidx)
        agg = collections.Counter()
        false_total = 0
        per_rec = []
        for rec in recs:
            fs, x = dl.load_audio_16k(rec)
            runs = dl.detect_hum_runs(x, fs, hb)
            beeps = [b['t'] for b in dl.detect_beeps(x, fs, hb)]
            matched = set()
            rinfo = {'rec': rec, 'n_runs': len(runs), 'steps': {}}
            for sid in mwsteps:
                w = step_win(rec, sid)
                if w is None:
                    continue
                s, e = w
                agg['present'] += 1
                idx = [i for i, r in enumerate(runs) if r[0] <= e + TOL and r[1] >= s - TOL]
                matched.update(idx)
                h = bool(idx); b = any(abs(bt - e) <= TOL for bt in beeps)
                agg['hum'] += h; agg['beep'] += b; agg['fused'] += (h or b)
                rinfo['steps'][sid] = {'hum': h, 'beep': b}
            f = len(runs) - len(matched)
            false_total += f
            rinfo['false'] = f
            per_rec.append(rinfo)
        p = agg['present']
        row = {'activity': aidx, 'recipe': name, 'n_recs': len(recs), 'cycles': p,
               'hum_recall': round(agg['hum'] / p, 3) if p else None,
               'beep_recall': round(agg['beep'] / p, 3) if p else None,
               'fused_recall': round(agg['fused'] / p, 3) if p else None,
               'false_runs': false_total,
               'false_per_rec': round(false_total / len(recs), 2) if recs else None,
               'recordings': per_rec}
        out.append(row)
        for k in ('present', 'hum', 'beep', 'fused'):
            grand[k] += agg[k]
        grand['false'] += false_total
        grand['recs'] += len(recs)
        print(f"{name:<26}{len(recs):>5}{p:>5}{row['hum_recall']:>7.2f}"
              f"{row['beep_recall']:>7.2f}{row['fused_recall']:>7.2f}"
              f"{false_total:>7}{row['false_per_rec']:>8.2f}")

    gp = grand['present']
    print(f"\n{'OVERALL':<26}{grand['recs']:>5}{gp:>5}"
          f"{grand['hum']/gp:>7.2f}{grand['beep']/gp:>7.2f}{grand['fused']/gp:>7.2f}"
          f"{grand['false']:>7}{grand['false']/grand['recs']:>8.2f}")
    summary = {'params': {k: hb[k] for k in ('min_run_s', 'feat_med_frames', 'med_frames')},
               'overall': {'n_recs': grand['recs'], 'cycles': gp,
                           'hum_recall': round(grand['hum'] / gp, 3),
                           'beep_recall': round(grand['beep'] / gp, 3),
                           'fused_recall': round(grand['fused'] / gp, 3),
                           'false_runs': grand['false'],
                           'false_per_rec': round(grand['false'] / grand['recs'], 3)},
               'per_recipe': out}
    json.dump(summary, open(f'{os.path.dirname(__file__)}/results_all_microwave.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
