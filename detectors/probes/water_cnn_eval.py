#!/usr/bin/env python3
"""Can the CNN (PANNs CNN14, AudioSet) detect water/tap on/off better than the DSP
level gate (which had recall 0.76-1.00 but 9-14 runs/rec = noisy)? Uses AudioSet water
classes {288 Water, 370 Water tap/faucet, 371 Sink, 444 Liquid}. Same 2 s/1 s windowing
as AL-cook. Metric per tap-rinse step: recall (interval overlaps), onset/offset latency,
and runs/rec (precision — the DSP's weakness)."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
WATER_IDX = [288, 370, 371, 444]
THS = [0.05, 0.1]
TOL = 10.0
TARGETS = [('20', 206, 'Mushrooms rinse'), ('12', 125, 'TomatoMozz rinse'),
           ('17', 189, 'Cucumber Raita rinse'), ('13', 137, 'ButterCorn thaw-rinse')]


def load_model():
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    m.to('cuda:0').eval()
    return m, 'cuda:0'


def water_sig(model, dev, rec):
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
            probs.append(out[:, WATER_IDX].max(axis=1))
    sig = np.concatenate(probs)
    t = np.array(starts[:len(sig)]) / 32000.0 + 1.0
    return t, sig


def intervals(t, on, gap=3.0, mindur=3.0):
    idx = np.where(on)[0]
    if not len(idx):
        return []
    runs, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            runs.append([t[s], t[a]]); s = b
    runs.append([t[s], t[idx[-1]]])
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [[a, b] for a, b in merged if b - a >= mindur]


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def main():
    model, dev = load_model()
    print("CNN water detector (AudioSet 288/370/371/444) vs DSP baseline (recall .76-1.0, 9-14 runs/rec)\n")
    for th in THS:
        print(f"--- threshold {th} ---")
        print(f"{'target':<24}{'n':>4}{'recall':>8}{'onset':>8}{'offset':>8}{'runs/rec':>9}")
        for aidx, sid, label in TARGETS:
            recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                           and os.path.exists(f'{dl.AUDIO_DIR}/{k}_48k.wav')],
                          key=lambda x: int(x.split('_')[1]))
            det, on_l, off_l, nr, n = 0, [], [], 0, 0
            for rec in recs:
                st = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}.get(sid)
                if not st:
                    continue
                n += 1
                s, e = float(st['start_time']), float(st['end_time'])
                t, sig = water_sig(model, dev, rec)
                ivs = intervals(t, sig > th)
                nr += len(ivs)
                ov = [r for r in ivs if r[0] <= e + TOL and r[1] >= s - TOL]
                if ov:
                    det += 1
                    m = min(ov, key=lambda r: abs(r[0] - s))
                    on_l.append(m[0] - s); off_l.append(m[1] - e)
            print(f"{label:<24}{n:>4}{det/n:>8.2f}{str(med(on_l)):>8}{str(med(off_l)):>8}{nr/n:>9.2f}")
        print()


if __name__ == '__main__':
    main()
