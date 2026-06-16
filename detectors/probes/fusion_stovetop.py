#!/usr/bin/env python3
"""Measure A4 x AL FUSION operating points across all 9 stovetop recipes, same metric
(coverage>=0.40 = recall; prep-positive sec / cook sec = false_rate).
Per recording: A4 = detect_sizzle_runs; AL = CNN14 cook-prob > th, bridged 20s -> runs.
Fusions (interval algebra on the two run-sets):
  UNION        A4 OR AL          - recall ceiling
  INTER        A4 AND AL         - both agree (max precision)
  A4gAL        A4 runs kept only if overlapped by an AL run ("AL vetoes A4 prep FAs",
               keeps full A4 coverage inside confirmed runs)  <-- candidate
AL threshold 0.05 (best-recall point from the bridged sweep)."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
COV_MIN, AL_TH, GAP = 0.40, 0.05, 20.0
COOK_IDX = [367, 490, 456, 296]
SIZZLE = dict(nfft=1024, hop=256, band_lo=1500.0, band_hi=7000.0, block_s=0.5,
              roll_win_s=45.0, baseline_pct=20, level_db=8.0, merge_gap_s=20.0, min_run_s=30.0)
RECIPES = {
    '25': ('Pan Fried Tofu', {279, 286, 292, 285, 290, 277, 278, 289, 281, 288}),
    '23': ('Broccoli Stir Fry', {266, 258, 265, 261, 253, 274, 256, 264}),
    '20': ('Sauteed Mushrooms', {207, 208, 209, 213, 217, 218, 219, 220}),
    '16': ('Scrambled Eggs', {160, 162, 165, 166, 168, 170, 171, 172, 175, 178, 179}),
    '22': ('Herb Omelet', {235, 238, 240, 241, 245}),
    '15': ('Tomato Chutney', {139, 141, 142, 144, 146, 147, 148, 152, 153, 157}),
    '18': ('Zoodles', {192, 199, 201, 202, 204}),
    '21': ('Banana Pancakes', {223, 226, 228, 230, 231}),
    '29': ('Caprese Bruschetta', {352}),
}


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def norm(runs):
    if not runs:
        return []
    runs = sorted([list(r) for r in runs])
    out = [runs[0]]
    for a, b in runs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inter(A, B):
    res = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                res.append([lo, hi])
    return norm(res)


def gated(A, B):  # A runs that overlap any B run, kept whole
    return [list(a) for a in A if any(ov(a[0], a[1], b[0], b[1]) > 0 for b in B)]


def windows(rec, cook_steps):
    steps = [s for s in ANN[rec]['steps'] if s['start_time'] >= 0]
    cook = [s for s in steps if s['step_id'] in cook_steps]
    if not cook:
        return None
    cs = min(s['start_time'] for s in cook); ce = max(s['end_time'] for s in cook)
    prep = [(s['start_time'], s['end_time']) for s in steps if s['step_id'] not in cook_steps]
    return cs, ce, prep


def metrics(runs, cs, ce, prep):
    cov = sum(ov(a, b, cs, ce) for a, b in runs) / (ce - cs)
    fls = sum(ov(a, b, ps, pe) for a, b in runs for ps, pe in prep)
    return cov, fls


def load_model():
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    m.to('cuda:0').eval()
    return m, 'cuda:0'


def al_runs(model, dev, rec):
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
    on = sig > AL_TH
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
        if a - merged[-1][1] <= GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


VARIANTS = ['A4', 'AL', 'UNION', 'INTER', 'A4gAL']


def main():
    model, dev = load_model()
    print(f'fusion (AL th={AL_TH}, bridge {GAP}s)\n')
    hdr = f"{'recipe':<20}{'recs':>5}"
    for v in VARIANTS:
        hdr += f"{v+'_r':>8}{v+'_f':>7}"
    print(hdr)
    allres = {}
    for aidx, (name, cook_steps) in RECIPES.items():
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        acc = {v: {'hit': 0, 'fls': 0.0} for v in VARIANTS}
        cook_tot = 0.0; n = 0
        for rec in recs:
            w = windows(rec, cook_steps)
            if w is None:
                continue
            cs, ce, prep = w
            if ce - cs <= 0:
                continue
            n += 1; cook_tot += ce - cs
            fs16, x16 = dl.load_audio_16k(rec)
            A = norm(dl.detect_sizzle_runs(x16, fs16, SIZZLE))
            B = al_runs(model, dev, rec)
            sets = {'A4': A, 'AL': B, 'UNION': norm(A + B), 'INTER': inter(A, B),
                    'A4gAL': gated(A, B)}
            for v in VARIANTS:
                cov, fls = metrics(sets[v], cs, ce, prep)
                acc[v]['hit'] += int(cov >= COV_MIN); acc[v]['fls'] += fls
        row = {'recipe': name, 'n': n}
        line = f"{name:<20}{n:>5}"
        for v in VARIANTS:
            r = acc[v]['hit'] / n; f = acc[v]['fls'] / cook_tot
            row[v] = {'recall': round(r, 3), 'false_rate': round(f, 3)}
            line += f"{r:>8.2f}{f:>7.2f}"
        print(line, flush=True); allres[aidx] = row
    # corpus-level (recording-weighted) means
    print()
    agg = {v: {'r': [], 'f': []} for v in VARIANTS}
    for row in allres.values():
        for v in VARIANTS:
            agg[v]['r'].append(row[v]['recall']); agg[v]['f'].append(row[v]['false_rate'])
    line = f"{'MEAN(recipe)':<20}{'':>5}"
    for v in VARIANTS:
        line += f"{np.mean(agg[v]['r']):>8.2f}{np.mean(agg[v]['f']):>7.2f}"
    print(line)
    json.dump(allres, open(f'{os.path.dirname(__file__)}/results_fusion_stovetop.json', 'w'), indent=2)
    print('\nwrote results_fusion_stovetop.json')


if __name__ == '__main__':
    main()
