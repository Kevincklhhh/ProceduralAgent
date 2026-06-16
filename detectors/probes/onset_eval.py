#!/usr/bin/env python3
"""Test cheap STEP-START cues vs the true frying start (the boundary_eval showed the
sustained detectors are laggy/late at onset). Two cues:

  (1) SIZZLE-ONSET (rising edge): causal short-window detector — 1.5-7 kHz band energy
      vs a TRAILING 30 s median; fire when it jumps +8 dB and holds >=5 s. This is the
      add-food-to-hot-pan burst, the rising edge A4's 45 s centered median smooths away.
  (2) PRE-COOK TRANSIENT cue (stove-on / igniter): A2 detect_beeps and A3 clink/transient
      trains in [cs-60, cs]; how often is there ANY detectable transient just before frying.

Reference cs = on-heat phase start (rescore_onheat windows). Latency = onset - cs
(signed; ~0 or slightly negative = caught the true food-in-pan burst). CPU only."""
import json, os, sys
import numpy as np
from scipy.signal import stft, medfilt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
ONHEAT = {'25': {281,285,288,289,290,292}, '23': {253,264,274}, '20': {207,213},
          '16': {165,166,170,171}, '22': {238,240,245}, '15': {148,152,153},
          '18': {192,202}, '21': {226,230}, '29': {352}}
NAME = {'25':'Pan Fried Tofu','23':'Broccoli Stir Fry','20':'Sauteed Mushrooms',
        '16':'Scrambled Eggs','22':'Herb Omelet','15':'Tomato Chutney',
        '18':'Zoodles','21':'Banana Pancakes','29':'Caprese Bruschetta'}
# onset detector params
BAND = (1500.0, 7000.0); BLOCK_S = 0.5; TRAIL_S = 30.0; MIN_HIST_S = 8.0
LEVEL_DB = 8.0; HOLD_S = 5.0; ASSOC = 45.0


def sizzle_onsets(x, fs):
    f, t, Z = stft(x, fs=fs, nperseg=1024, noverlap=1024 - 256, window='hann',
                   padded=False, boundary=None)
    P = np.abs(Z) ** 2
    band = (f >= BAND[0]) & (f <= BAND[1])
    bdb = 10 * np.log10(P[band].sum(axis=0) + 1e-12)
    dt = 256 / fs
    blk = max(1, int(round(BLOCK_S / dt)))
    nb = len(bdb) // blk
    coarse = np.median(bdb[:nb * blk].reshape(nb, blk), axis=1)
    tc = (np.arange(nb) + 0.5) * blk * dt
    dtc = blk * dt
    trail = int(TRAIL_S / dtc); hist = int(MIN_HIST_S / dtc); hold = int(HOLD_S / dtc)
    onsets, i = [], hist
    while i < nb - hold:
        base = np.median(coarse[max(0, i - trail):i])
        if coarse[i] - base > LEVEL_DB and np.mean(coarse[i:i + hold]) - base > LEVEL_DB - 2:
            onsets.append(float(tc[i]))
            i += hold + trail  # past this rise
        else:
            i += 1
    return onsets


def phase(rec, aidx):
    steps = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}
    w = [(steps[s]['start_time'], steps[s]['end_time']) for s in ONHEAT[aidx] if s in steps]
    return (min(a for a, b in w), max(b for a, b in w)) if w else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def main():
    hb, pc = dl.load_frozen_params()
    print(f"{'recipe':<20}{'n':>4}{'onset_lat':>10}{'|lat|':>7}{'det±15':>8}"
          f"{'precook_A2':>11}{'precook_A3':>11}")
    for aidx in ['25', '23', '20', '16', '22', '15', '18', '21', '29']:
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        lat, det, a2, a3, n = [], 0, 0, 0, 0
        for rec in recs:
            p = phase(rec, aidx)
            if not p:
                continue
            cs, ce = p; n += 1
            fs16, x16 = dl.load_audio_16k(rec)
            ons = sizzle_onsets(x16, fs16)
            near = [o for o in ons if cs - ASSOC <= o <= cs + ASSOC]
            if near:
                o = min(near, key=lambda z: abs(z - cs))
                lat.append(o - cs); det += int(abs(o - cs) <= 15)
            else:
                lat.append(None)
            # pre-cook transient cues in [cs-60, cs]
            beeps = dl.detect_beeps(x16, fs16, hb)
            if any(cs - 60 <= b['t'] <= cs for b in beeps):
                a2 += 1
            fs48, x48 = dl.load_audio_48k(rec)
            cl = dl.detect_clink_trains(x48, fs48, pc)
            if any(c['start'] <= cs and c['end'] >= cs - 60 for c in cl):
                a3 += 1
        print(f"{NAME[aidx]:<20}{n:>4}{str(med(lat)):>10}"
              f"{str(med([abs(x) for x in lat if x is not None])):>7}{det/n:>8.2f}"
              f"{a2/n:>11.2f}{a3/n:>11.2f}")
    print("\nonset_lat = median signed sec (onset - frying start; ~0/neg = caught the "
          "food-in-pan burst). det±15 = frac within 15 s. precook_A2/A3 = frac of recs "
          "with a beep / transient-train in [cs-60,cs] (stove-on cue).")


if __name__ == '__main__':
    main()
