"""Exploration on tuning recording 8_16 only: look at pour-band energy and
high-band spectral flux onsets vs GT step windows, to pick thresholds."""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, medfilt

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'
GT = json.load(open('/home/kailaic/NeuroTrace/ProceduralAgent/data/gt_activity8.json'))

REC = '8_16'
sr, x = wavfile.read(f'{AUDIO_DIR}/{REC}_48k.wav')
x = x.astype(np.float64) / 32768.0

steps = {s['step_id']: (s['start_time'], s['end_time']) for s in GT[REC]['steps']}

# ---------------- POUR feature: 1-8 kHz band energy (dB), 20 ms hop ----------
NFFT_P, HOP_P = 2048, 960  # 20 ms hop @48k
f, t_p, Z = stft(x, fs=sr, nperseg=NFFT_P, noverlap=NFFT_P - HOP_P, padded=False, boundary=None)
P = np.abs(Z) ** 2
band = (f >= 1000) & (f <= 8000)
e_band = P[band].sum(axis=0)
db_band = 10 * np.log10(e_band + 1e-12)
# spectral flatness in band (geometric/arithmetic mean) - pour is noisy/flat
logP = np.log(P[band] + 1e-12)
flat = np.exp(logP.mean(axis=0)) / (P[band].mean(axis=0) + 1e-12)
# smooth ~0.3 s
k = 15
db_s = medfilt(db_band, k)
flat_s = medfilt(flat, k)
# rolling median background (31 s)
win = int(31 / 0.02)
bg = np.array([np.median(db_s[max(0, i - win // 2):i + win // 2 + 1]) for i in range(0, len(db_s), 50)])
bg_full = np.interp(np.arange(len(db_s)), np.arange(0, len(db_s), 50), bg)
rel = db_s - bg_full

print('=== POUR band (1-8k) stats per GT step ===')
for sid, (a, b) in sorted(steps.items()):
    m = (t_p >= a) & (t_p <= b)
    print(f'step {sid:3d} [{a:6.1f}-{b:6.1f}] reldB p50={np.percentile(rel[m],50):5.1f} '
          f'p90={np.percentile(rel[m],90):5.1f} p99={np.percentile(rel[m],99):5.1f} '
          f'flat p50={np.percentile(flat_s[m],50):.3f} p90={np.percentile(flat_s[m],90):.3f}')

# contiguous regions above threshold for a few thresholds
def regions(mask, t, min_gap=0.4):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out = []
    s0 = p = idx[0]
    for i in idx[1:]:
        if (i - p) * 0.02 > min_gap:
            out.append((t[s0], t[p]))
            s0 = i
        p = i
    out.append((t[s0], t[p]))
    return out

for thr in [4, 6, 8, 10]:
    regs = regions(rel > thr, t_p)
    regs = [(a, b) for a, b in regs if b - a >= 0.8]
    print(f'--- rel>{thr} dB, dur>=0.8 s: {len(regs)} regions')
    for a, b in regs:
        inside = [sid for sid, (s, e) in steps.items() if a < e and b > s]
        fmean = np.mean(flat_s[(t_p >= a) & (t_p <= b)])
        print(f'   {a:7.1f}-{b:7.1f} ({b-a:5.1f}s) flat={fmean:.3f} steps={inside}')

# ---------------- CLINK: high-band spectral flux onsets ----------------------
NFFT_C, HOP_C = 1024, 480  # 10 ms hop
f2, t_c, Z2 = stft(x, fs=sr, nperseg=NFFT_C, noverlap=NFFT_C - HOP_C, padded=False, boundary=None)
M = np.abs(Z2)
hb = (f2 >= 4000) & (f2 <= 16000)
Mh = M[hb]
flux = np.maximum(Mh[:, 1:] - Mh[:, :-1], 0).sum(axis=0)
flux = np.concatenate([[0], flux])
# adaptive threshold: rolling median + K*MAD over 5 s
wlen = 501
med = medfilt(flux, wlen)
mad = medfilt(np.abs(flux - med), wlen) + 1e-9

for K in [4, 6, 8, 10]:
    above = flux > med + K * mad
    onsets = []
    i = 1
    while i < len(flux) - 1:
        if above[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if not onsets or t_c[i] - onsets[-1] > 0.07:
                onsets.append(t_c[i])
            i += 7
        else:
            i += 1
    onsets = np.array(onsets)
    print(f'=== CLINK K={K}: {len(onsets)} onsets total ===')
    for sid, (a, b) in sorted(steps.items()):
        n = int(((onsets >= a) & (onsets <= b)).sum())
        rate = n / max(b - a, 1)
        print(f'  step {sid:3d} [{a:6.1f}-{b:6.1f}]: {n:4d} onsets ({rate:.2f}/s)')
    out = int(sum(1 for o in onsets if not any(a <= o <= b for a, b in steps.values())))
    print(f'  outside any step: {out}')
