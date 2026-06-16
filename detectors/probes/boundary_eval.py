#!/usr/bin/env python3
"""Boundary-detection eval (the metric that matches our setting, NOT coverage).

For each recording the frying phase = [cs, ce] = union span of the ON-HEAT-with-food
sub-steps. We measure, for A4 (DSP, 45 s window) and AL (CNN, 2 s window):
  ONSET  latency = detected_cook_start - cs   (signed; + = late)
  OFFSET latency = detected_cook_end   - ce   (signed; + = late)
  detect rate     = a cook interval overlaps [cs, ce]
  false_trigger/rec = detector-positive intervals lying entirely in prep (would
                      fire a needless VLM call)
A4 intervals = detect_sizzle_runs. AL intervals = (cook_prob>th) frames merged over
<=5 s gaps, >=4 s long (light, keeps 2 s responsiveness; NO 20 s bridge).
This exposes A4's window lag: its onset cannot precede ~half its 45 s median."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
AL_TH, MATCH = 0.1, 30.0
COOK_IDX = [367, 490, 456, 296]
SIZZLE = dict(nfft=1024, hop=256, band_lo=1500.0, band_hi=7000.0, block_s=0.5,
              roll_win_s=45.0, baseline_pct=20, level_db=8.0, merge_gap_s=20.0, min_run_s=30.0)
# on-heat-with-food sub-steps (from rescore_onheat.py)
ONHEAT = {'25': {281,285,288,289,290,292}, '23': {253,264,274}, '20': {207,213},
          '16': {165,166,170,171}, '22': {238,240,245}, '15': {148,152,153},
          '18': {192,202}, '21': {226,230}, '29': {352}}
NAME = {'25':'Pan Fried Tofu','23':'Broccoli Stir Fry','20':'Sauteed Mushrooms',
        '16':'Scrambled Eggs','22':'Herb Omelet','15':'Tomato Chutney',
        '18':'Zoodles','21':'Banana Pancakes','29':'Caprese Bruschetta'}


def phase(rec, aidx):
    steps = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}
    w = [(steps[s]['start_time'], steps[s]['end_time']) for s in ONHEAT[aidx] if s in steps]
    return (min(a for a, b in w), max(b for a, b in w)) if w else None


def to_intervals(on_t):  # list of frame times that are 'on' -> merged intervals
    if not len(on_t):
        return []
    iv, s, p = [], on_t[0], on_t[0]
    for t in on_t[1:]:
        if t - p <= 5.0:
            p = t
        else:
            iv.append([s, p]); s = p = t
    iv.append([s, p])
    return [[a, b] for a, b in iv if b - a >= 4.0]


def boundary(intervals, cs, ce):
    """onset/offset error (signed) for the cook interval overlapping [cs,ce] near it;
    plus count of intervals fully in prep (false triggers)."""
    matched = [iv for iv in intervals if iv[0] <= ce + MATCH and iv[1] >= cs - MATCH]
    false_trig = sum(1 for iv in intervals if iv[1] < cs - MATCH or iv[0] > ce + MATCH)
    if not matched:
        return None, None, false_trig
    onset = min(iv[0] for iv in matched)
    offset = max(iv[1] for iv in matched)
    return onset - cs, offset - ce, false_trig


def load_model():
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    m.to('cuda:0').eval()
    return m, 'cuda:0'


def al_intervals(model, dev, rec):
    import torch
    fs, x = wavfile.read(f'{BASE}/data/audio/{rec}_48k.wav')
    x = x.astype(np.float64) / 32768.0
    x32 = resample_poly(x, 2, 3).astype(np.float32)
    W, H = 64000, 32000
    starts = list(range(0, max(1, len(x32) - W), H))
    probs = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            segs = np.stack([x32[s:s + W] for s in starts[i:i + 64]])
            out = model(torch.from_numpy(segs).to(dev), None)['clipwise_output'].cpu().numpy()
            probs.append(out[:, COOK_IDX].max(axis=1))
    sig = np.concatenate(probs)
    t = np.array(starts[:len(sig)]) / 32000.0 + 1.0
    return to_intervals(t[sig > AL_TH])


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def main():
    model, dev = load_model()
    print(f"{'recipe':<20}{'n':>4} | {'A4 onset':>9}{'A4 offset':>10}{'A4 det':>7}{'A4 ftrg':>8}"
          f" | {'AL onset':>9}{'AL offset':>10}{'AL det':>7}{'AL ftrg':>8}")
    for aidx in ['25', '23', '20', '16', '22', '15', '18', '21', '29']:
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        A = {'on': [], 'off': [], 'det': 0, 'ft': 0}
        L = {'on': [], 'off': [], 'det': 0, 'ft': 0}
        n = 0
        for rec in recs:
            p = phase(rec, aidx)
            if not p:
                continue
            cs, ce = p
            n += 1
            fs16, x16 = dl.load_audio_16k(rec)
            a4 = [list(r) for r in dl.detect_sizzle_runs(x16, fs16, SIZZLE)]
            on, off, ft = boundary(a4, cs, ce)
            A['on'].append(on); A['off'].append(off); A['det'] += int(on is not None); A['ft'] += ft
            al = al_intervals(model, dev, rec)
            on, off, ft = boundary(al, cs, ce)
            L['on'].append(on); L['off'].append(off); L['det'] += int(on is not None); L['ft'] += ft
        print(f"{NAME[aidx]:<20}{n:>4} | {str(med(A['on'])):>9}{str(med(A['off'])):>10}"
              f"{A['det']/n:>7.2f}{A['ft']/n:>8.2f} | {str(med(L['on'])):>9}{str(med(L['off'])):>10}"
              f"{L['det']/n:>7.2f}{L['ft']/n:>8.2f}", flush=True)
    print("\n(onset/offset = median signed seconds vs GT cook start/end; + = late. "
          "det = fraction with a matched cook interval; ftrg = prep false-trigger intervals/rec.)")


if __name__ == '__main__':
    main()
