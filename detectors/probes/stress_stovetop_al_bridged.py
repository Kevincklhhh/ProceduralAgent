#!/usr/bin/env python3
"""Fair-comparison addendum: give AL (CNN14) the SAME persistence logic A4 has —
bridge gaps <=20 s between on-frames — then recompute cook coverage/recall/false.
Tests whether the CNN can be a coverage anchor (like A4) once consolidated, or
whether it's intrinsically a sparse precision signal. th in {0.05, 0.1}."""
import json, os, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
COV_MIN = 0.40
THS = [0.05, 0.1]
GAP = 20.0
COOK_IDX = [367, 490, 456, 296]
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


def windows(rec, cook_steps):
    steps = [s for s in ANN[rec]['steps'] if s['start_time'] >= 0]
    cook = [s for s in steps if s['step_id'] in cook_steps]
    if not cook:
        return None
    cs = min(s['start_time'] for s in cook); ce = max(s['end_time'] for s in cook)
    prep = [(s['start_time'], s['end_time']) for s in steps if s['step_id'] not in cook_steps]
    return cs, ce, prep


def on_to_runs(t, on, gap=GAP):
    """contiguous on-frames -> [s,e] runs, then merge runs with gap<=GAP."""
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
    return merged


def load_model():
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    m.to('cuda:0').eval()
    return m, 'cuda:0'


def al_signal(model, dev, rec, win=2.0, hop=1.0):
    import torch
    fs, x = wavfile.read(f'{BASE}/data/audio/{rec}_48k.wav')
    x = x.astype(np.float64) / 32768.0
    x32 = resample_poly(x, 2, 3).astype(np.float32)
    W, H = int(win * 32000), int(hop * 32000)
    starts = list(range(0, max(1, len(x32) - W), H))
    probs = []
    with torch.no_grad():
        for i in range(0, len(starts), 64):
            segs = np.stack([x32[s:s + W] for s in starts[i:i + 64]])
            out = model(torch.from_numpy(segs).to(dev), None)['clipwise_output'].cpu().numpy()
            probs.append(out[:, COOK_IDX].max(axis=1))
    sig = np.concatenate(probs)
    t = np.array(starts[:len(sig)]) / 32000.0 + win / 2
    return t, sig


def main():
    model, dev = load_model()
    print('AL with 20s gap-bridging (fair vs A4 persistence)\n')
    print(f"{'recipe':<20}{'recs':>5}" + ''.join(f"{'br'+str(th)+'r':>9}{'br'+str(th)+'cov':>9}{'br'+str(th)+'f':>9}" for th in THS))
    res = {}
    for aidx, (name, cook_steps) in RECIPES.items():
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        acc = {th: {'hit': 0, 'cov': [], 'fls': 0.0} for th in THS}
        cook_tot = 0.0; n = 0
        for rec in recs:
            w = windows(rec, cook_steps)
            if w is None:
                continue
            cs, ce, prep = w; clen = ce - cs
            if clen <= 0:
                continue
            n += 1; cook_tot += clen
            t, sig = al_signal(model, dev, rec)
            for th in THS:
                runs = on_to_runs(t, sig > th)
                cov = sum(ov(a, b, cs, ce) for a, b in runs) / clen
                acc[th]['cov'].append(cov); acc[th]['hit'] += int(cov >= COV_MIN)
                acc[th]['fls'] += sum(ov(a, b, ps, pe) for a, b in runs for ps, pe in prep)
        row = {'recipe': name, 'n': n}
        line = f"{name:<20}{n:>5}"
        for th in THS:
            r = round(acc[th]['hit'] / n, 3); c = round(float(np.mean(acc[th]['cov'])), 3)
            f = round(acc[th]['fls'] / cook_tot, 3)
            row[f'br{th}'] = {'recall': r, 'cov': c, 'false_rate': f}
            line += f"{r:>9.2f}{c:>9.2f}{f:>9.2f}"
        print(line, flush=True); res[aidx] = row
    json.dump(res, open(f'{os.path.dirname(__file__)}/results_stress_stovetop_bridged.json', 'w'), indent=2)
    print('\nwrote results_stress_stovetop_bridged.json')


if __name__ == '__main__':
    main()
