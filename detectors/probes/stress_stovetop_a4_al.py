#!/usr/bin/env python3
"""Stress-test the stovetop cook detector across ALL 9 stovetop recipes / all recordings,
head-to-head A4 (DSP sustained-level sizzle) vs AL (CNN14 AudioSet cook tagger), with
IDENTICAL cook/prep windows and metrics:
  cook window  = [min start, max end] over present cook steps.
  prep windows = all present NON-cook steps (disjoint).
  recall       = fraction of recordings with coverage(cook) >= 0.40.
  mean_cov     = mean coverage of the cook window.
  false_rate   = detector-positive seconds in prep / total cook seconds.
A4: detect_sizzle_runs (frozen 23_5 params). AL: CNN14 cook-prob (Frying/Sizzle/Boiling/
Steam, 2s win/1s hop), swept at th in {0.05,0.1,0.2} (probs are low-scaled). One model
forward per recording; thresholds applied to the same signal."""
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
AL_THS = [0.05, 0.1, 0.2]
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


def windows(rec, cook_steps):
    steps = [s for s in ANN[rec]['steps'] if s['start_time'] >= 0]
    cook = [s for s in steps if s['step_id'] in cook_steps]
    if not cook:
        return None
    cs = min(s['start_time'] for s in cook); ce = max(s['end_time'] for s in cook)
    prep = [(s['start_time'], s['end_time']) for s in steps if s['step_id'] not in cook_steps]
    return cs, ce, prep


def load_model():
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    dev = 'cuda:0'
    m.to(dev).eval()
    return m, dev


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
    print(f'CNN14 on {dev}\n')
    results = {}
    hdr = f"{'recipe':<20}{'recs':>5}{'A4_rec':>8}{'A4_cov':>8}{'A4_fls':>8}"
    for th in AL_THS:
        hdr += f"{'AL'+str(th)+'r':>8}{'AL'+str(th)+'f':>8}"
    print(hdr)
    for aidx, (name, cook_steps) in RECIPES.items():
        recs = sorted([k for k in ANN if k.startswith(aidx + '_')
                       and os.path.exists(f'{dl.AUDIO_DIR}/{k}_16k.wav')],
                      key=lambda x: int(x.split('_')[1]))
        a4 = {'hit': 0, 'cov': [], 'fls': 0.0, 'cook': 0.0, 'n': 0}
        al = {th: {'hit': 0, 'cov': [], 'fls': 0.0} for th in AL_THS}
        for rec in recs:
            w = windows(rec, cook_steps)
            if w is None:
                continue
            cs, ce, prep = w
            clen = ce - cs
            if clen <= 0:
                continue
            a4['n'] += 1; a4['cook'] += clen
            # ---- A4 ----
            fs16, x16 = dl.load_audio_16k(rec)
            sizz = dl.detect_sizzle_runs(x16, fs16, SIZZLE)
            cov = sum(ov(a, b, cs, ce) for a, b in sizz) / clen
            a4['cov'].append(cov); a4['hit'] += int(cov >= COV_MIN)
            a4['fls'] += sum(ov(a, b, ps, pe) for a, b in sizz for ps, pe in prep)
            # ---- AL ----
            t, sig = al_signal(model, dev, rec)
            inc = (t >= cs) & (t < ce)
            ninc = max(1, inc.sum())
            for th in AL_THS:
                on = sig > th
                cov_al = (inc & on).sum() / ninc
                al[th]['cov'].append(cov_al); al[th]['hit'] += int(cov_al >= COV_MIN)
                fp = 0.0
                ton = t[on]
                for ps, pe in prep:
                    fp += ((ton >= ps) & (ton < pe)).sum()  # 1 s/frame
                al[th]['fls'] += fp
        n = a4['n']
        row = {'recipe': name, 'n': n,
               'a4_recall': round(a4['hit'] / n, 3), 'a4_cov': round(float(np.mean(a4['cov'])), 3),
               'a4_false_rate': round(a4['fls'] / a4['cook'], 3), 'al': {}}
        line = f"{name:<20}{n:>5}{row['a4_recall']:>8.2f}{row['a4_cov']:>8.2f}{row['a4_false_rate']:>8.2f}"
        for th in AL_THS:
            r = round(al[th]['hit'] / n, 3); f = round(al[th]['fls'] / a4['cook'], 3)
            row['al'][th] = {'recall': r, 'cov': round(float(np.mean(al[th]['cov'])), 3), 'false_rate': f}
            line += f"{r:>8.2f}{f:>8.2f}"
        print(line, flush=True)
        results[aidx] = row
    json.dump(results, open(f'{os.path.dirname(__file__)}/results_stress_stovetop.json', 'w'), indent=2)
    print('\nwrote results_stress_stovetop.json')


if __name__ == '__main__':
    main()
