#!/usr/bin/env python3
"""Test the two UNTESTED audio-transition classes (the remaining frontier):
  (1) appliance-motor on/off  - blender/grinder (loud sustained motor); kettle boil.
  (2) water on/off            - tap rinse/thaw (sustained mid-band flow).
Both are CAUSAL-ish sustained detectors with a SHORT rolling window (low latency,
unlike A4's 45 s). Metric = per target step: detected (run overlaps [s-10,e+10]),
onset latency (run.start - s), offset latency (run.end - e), runs/rec (false sense)."""
import json, os, sys
import numpy as np
from scipy.signal import stft
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
TOL = 10.0


def loud_runs(x, fs, band, level_db, min_run_s, roll_win_s=4.0, merge_s=3.0, block_s=0.5):
    f, t, Z = stft(x, fs=fs, nperseg=1024, noverlap=1024 - 256, window='hann',
                   padded=False, boundary=None)
    P = np.abs(Z) ** 2
    bm = (f >= band[0]) & (f <= band[1])
    bdb = 10 * np.log10(P[bm].sum(axis=0) + 1e-12)
    dt = 256 / fs
    blk = max(1, int(round(block_s / dt)))
    nb = len(bdb) // blk
    if nb < 3:
        return []
    coarse = np.median(bdb[:nb * blk].reshape(nb, blk), axis=1)
    tc = (np.arange(nb) + 0.5) * blk * dt
    dtc = blk * dt
    w = max(1, int(round(roll_win_s / dtc))); half = w // 2
    roll = np.array([np.median(coarse[max(0, i - half):i + half + 1]) for i in range(nb)])
    floor = np.percentile(roll, 20)
    on = roll - floor > level_db
    # runs
    runs, i = [], 0
    while i < nb:
        if on[i]:
            j = i
            while j + 1 < nb and on[j + 1]:
                j += 1
            runs.append([float(tc[i]) - dtc / 2, float(tc[j]) + dtc / 2]); i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_run_s]


# detector configs
APP = dict(band=(200.0, 8000.0), level_db=12.0, min_run_s=5.0)     # loud motor
BOIL = dict(band=(1000.0, 8000.0), level_db=6.0, min_run_s=15.0)   # gentle broadband boil
WAT = dict(band=(1000.0, 8000.0), level_db=8.0, min_run_s=4.0)     # tap flow

# (activity, step, label, config)
TARGETS = [
    ('21', 227, 'Pancakes blender-blitz', APP),
    ('15', 150, 'Chutney blender-puree', APP),
    ('5', 67, 'Coffee grinder', APP),
    ('5', 64, 'Coffee kettle-boil', BOIL),
    ('20', 206, 'Mushrooms rinse', WAT),
    ('12', 125, 'TomatoMozz rinse', WAT),
    ('17', 189, 'Cucumber Raita rinse', WAT),
    ('13', 137, 'ButterCorn thaw-rinse', WAT),
]


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def main():
    print(f"{'target':<26}{'n':>4}{'recall':>8}{'onset':>8}{'offset':>8}{'runs/rec':>9}")
    for aidx, sid, label, cfg in TARGETS:
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        det, on_l, off_l, nruns, n = 0, [], [], 0, 0
        for rec in recs:
            st = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}.get(sid)
            if not st:
                continue
            n += 1
            s, e = float(st['start_time']), float(st['end_time'])
            fs, x = dl.load_audio_16k(rec)
            runs = loud_runs(x, fs, **cfg)
            nruns += len(runs)
            ov = [r for r in runs if r[0] <= e + TOL and r[1] >= s - TOL]
            if ov:
                det += 1
                m = min(ov, key=lambda r: abs(r[0] - s))
                on_l.append(m[0] - s); off_l.append(m[1] - e)
        print(f"{label:<26}{n:>4}{det/n:>8.2f}{str(med(on_l)):>8}{str(med(off_l)):>8}{nruns/n:>9.2f}")
    print("\nrecall = frac of recs where a run overlaps the step. onset/offset = median "
          "signed sec (run start - step start; run end - step end). runs/rec = total "
          "detector runs per recording (high w/o gating = noisy).")


if __name__ == '__main__':
    main()
