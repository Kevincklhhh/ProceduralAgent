#!/usr/bin/env python3
"""Probe: audio_anchors — validate the cheap audio backbone of the Spiced Hot Chocolate
sensor schedule across ALL 16 SHC recordings (it was previously probed on only 6).

The schedule (data/sensor_schedule_spicedhotchocolate.json) leans on three audio anchors,
all already frozen in detectors/detectors_lib.py (params from the validated probes):
  A1 hum run     -> microwave_initial (step 89) and heat_serve (step 83) completion
  A2 end beep    -> corroborates microwave/heat completion
  A3 clink train -> mix (step 85) completion ('strong' tier)
  A6 pour        -> weak, logged only (fill / serve), NOT scored as an anchor

Scoring per recording (GT step windows from complete_step_annotations.json; present steps
only — skipped step start=-1 excluded, so reordered/short error runs are handled):
  - hum-expected steps = present {89, 83}: HIT if a hum run overlaps [s-TOL, e+TOL].
  - mix step = present {85}: HIT if a STRONG clink train overlaps [s-TOL, e+TOL].
  - beep: HIT if a beep lands within TOL of a hum-expected step END.
  - false hum runs / false strong clinks = detections overlapping no expected window.

Frozen detectors, no tuning here (tuned earlier on 8_16). Outputs results_audio_anchors.json.
Usage: python probe_audio_anchors.py [--only 8_16]
"""
import json, os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
PROBE_DIR = os.path.join(BASE, 'detectors', 'probes')
HUM_STEPS = [89, 83]      # microwave_initial, heat_serve
MIX_STEP = 85
TOL = 15.0
SHC = ['8_11', '8_15', '8_16', '8_19', '8_20', '8_25', '8_26', '8_3',
       '8_30', '8_31', '8_33', '8_35', '8_40', '8_44', '8_45', '8_50']


def overlaps(run, s, e, tol=TOL):
    return run[0] <= e + tol and run[1] >= s - tol


def step_win(rec, sid):
    steps = {x['step_id']: x for x in ANN[rec]['steps']}
    s = steps.get(sid)
    if not s or s['start_time'] < 0:
        return None
    return (float(s['start_time']), float(s['end_time']))


def run_one(rec, hb, pc):
    fs16, x16 = dl.load_audio_16k(rec)
    fs48, x48 = dl.load_audio_48k(rec)
    hums = dl.detect_hum_runs(x16, fs16, hb)
    beeps = dl.detect_beeps(x16, fs16, hb)
    clinks = [c for c in dl.detect_clink_trains(x48, fs48, pc) if c['strong']]

    res = {'recording': rec, 'n_hum_runs': len(hums),
           'hum_runs': [[round(a, 1), round(b, 1), round(b - a, 1)] for a, b in hums],
           'n_beeps': len(beeps), 'n_strong_clinks': len(clinks),
           'hum_steps': {}, 'mix': None}
    matched_hum = set()
    for sid in HUM_STEPS:
        w = step_win(rec, sid)
        if w is None:
            res['hum_steps'][sid] = 'absent'
            continue
        hit = [i for i, r in enumerate(hums) if overlaps(r, *w)]
        matched_hum.update(hit)
        beep_hit = any(abs(b['t'] - w[1]) <= TOL for b in beeps)
        res['hum_steps'][sid] = {'window': [round(w[0], 1), round(w[1], 1)],
                                 'hum_hit': bool(hit), 'beep_at_end': beep_hit}
    res['false_hum_runs'] = len(hums) - len(matched_hum)

    wmix = step_win(rec, MIX_STEP)
    if wmix is None:
        res['mix'] = 'absent'
    else:
        chit = [c for c in clinks if overlaps((c['start'], c['end']), *wmix)]
        res['mix'] = {'window': [round(wmix[0], 1), round(wmix[1], 1)],
                      'clink_hit': bool(chit)}
    res['false_strong_clinks'] = len(clinks) - (1 if (isinstance(res['mix'], dict)
                                                      and res['mix']['clink_hit']) else 0)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only'); args = ap.parse_args()
    hb, pc = dl.load_frozen_params()
    recs = [args.only] if args.only else SHC
    per = {r: run_one(r, hb, pc) for r in recs}

    # aggregate
    hum_exp = hum_hit = beep_exp = beep_hit = mix_exp = mix_hit = 0
    false_hum = false_clink = 0
    for v in per.values():
        for sid, d in v['hum_steps'].items():
            if isinstance(d, dict):
                hum_exp += 1; hum_hit += d['hum_hit']
                beep_exp += 1; beep_hit += d['beep_at_end']
        if isinstance(v['mix'], dict):
            mix_exp += 1; mix_hit += v['mix']['clink_hit']
        false_hum += v['false_hum_runs']; false_clink += v['false_strong_clinks']
    summary = {'recordings': len(recs), 'tol_s': TOL,
               'hum_anchor': {'expected': hum_exp, 'hit': hum_hit,
                              'recall': round(hum_hit / hum_exp, 3) if hum_exp else None,
                              'false_runs': false_hum},
               'beep_at_microwave_end': {'expected': beep_exp, 'hit': beep_hit,
                                         'recall': round(beep_hit / beep_exp, 3) if beep_exp else None},
               'mix_clink_anchor': {'expected': mix_exp, 'hit': mix_hit,
                                    'recall': round(mix_hit / mix_exp, 3) if mix_exp else None,
                                    'false_strong_clinks': false_clink}}
    print(json.dumps(summary, indent=1))
    print('\nrec     hums(dur)                    | hum89 hum83 beep | mix_clink | falseHum falseClink')
    for r in recs:
        v = per[r]
        hd = ' '.join(f"{d[2]:.0f}s" for d in v['hum_runs'])

        def cell(sid):
            d = v['hum_steps'].get(sid)
            return '-' if d == 'absent' else ('Y' if d['hum_hit'] else 'MISS')
        beep = any(isinstance(d, dict) and d['beep_at_end'] for d in v['hum_steps'].values())
        mix = v['mix']
        mc = '-' if mix == 'absent' else ('Y' if mix['clink_hit'] else 'MISS')
        print(f"  {r:6s} [{v['n_hum_runs']}] {hd:24s} | {cell(89):5s} {cell(83):5s} "
              f"{'Y' if beep else 'n':4s} | {mc:9s} | {v['false_hum_runs']}        {v['false_strong_clinks']}"
              f"{'   [TUNE]' if r == '8_16' else ''}")
    if not args.only:
        json.dump({'probe': 'audio_anchors', 'summary': summary, 'per_recording': per},
                  open(os.path.join(PROBE_DIR, 'results_audio_anchors.json'), 'w'), indent=1)
        print('\nwrote results_audio_anchors.json')


if __name__ == '__main__':
    main()
