"""Diagnose: hum band dB-above-baseline traces for all recordings, within/around GT microwave steps."""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'
gt = json.load(open('/home/kailaic/NeuroTrace/ProceduralAgent/data/gt_activity8.json'))
recs = ['8_16', '8_26', '8_3', '8_25', '8_31', '8_50']

fig, axes = plt.subplots(len(recs), 1, figsize=(16, 3 * len(recs)))
for ax, rec in zip(axes, recs):
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_16k.wav')
    x = x.astype(np.float64) / 32768.0
    f, t, Z = stft(x, fs=fs, nperseg=8192, noverlap=4096, window='hann', padded=False, boundary=None)
    P = np.abs(Z) ** 2
    hb = 10 * np.log10(P[(f >= 100) & (f < 1000)].sum(axis=0) + 1e-12)
    base = np.percentile(hb, 20)
    rel = hb - base
    # smooth for printing
    k = 21
    sm = np.convolve(rel, np.ones(k) / k, mode='same')
    ax.plot(t, rel, lw=0.4, color='gray')
    ax.plot(t, sm, lw=1.0, color='blue')
    for s in gt[rec]['steps']:
        if s['step_id'] in (89, 83) and s['start_time'] >= 0:
            ax.axvspan(s['start_time'], s['end_time'], color='orange', alpha=0.25)
    for thr in (6, 9, 12):
        ax.axhline(thr, color='red', ls=':', lw=0.7)
    ax.set_ylabel(f'{rec}\ndB>base')
    ax.set_ylim(-5, 40)
    # print compact medians inside each microwave GT step in 5s chunks
    print(f'--- {rec} (base={base:.1f} dB) ---')
    for s in gt[rec]['steps']:
        if s['step_id'] in (89, 83) and s['start_time'] >= 0:
            a, b = s['start_time'], s['end_time']
            vals = []
            tt = a
            while tt < b:
                m = (t >= tt) & (t < tt + 5)
                vals.append(f'{np.median(rel[m]):.0f}' if m.any() else '-')
                tt += 5
            print(f'  step{s["step_id"]} [{a:.0f}-{b:.0f}] 5s-medians: {" ".join(vals)}')
plt.tight_layout()
plt.savefig('/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes/explore_all_humband.png', dpi=70)
print('saved plot')
