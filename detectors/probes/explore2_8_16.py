"""Closer look: hum spectrum shape, adaptive-threshold band energy, beep candidates on 8_16."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

AUDIO = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio/8_16_16k.wav'
fs, x = wavfile.read(AUDIO)
x = x.astype(np.float64) / 32768.0
NFFT = 8192; HOP = 4096
f, t, Z = stft(x, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP, window='hann', padded=False, boundary=None)
P = np.abs(Z)**2
eps = 1e-12

hum_mask = ((t > 76) & (t < 128))
off_mask = ((t > 180) & (t < 370))
Sh = P[:, hum_mask].mean(axis=1)
So = P[:, off_mask].mean(axis=1)
print('=== mean spectrum hum vs non-hum (dB), 0-1500 Hz in 50Hz steps ===')
for lo in range(0, 1500, 50):
    m = (f >= lo) & (f < lo + 50)
    print(f'{lo:5d}-{lo+50:4d}Hz  hum={10*np.log10(Sh[m].mean()+eps):7.1f}  off={10*np.log10(So[m].mean()+eps):7.1f}  diff={10*np.log10((Sh[m].mean()+eps)/(So[m].mean()+eps)):6.1f}')

# hum band energy with adaptive baseline
hb = P[(f >= 100) & (f < 1000)].sum(axis=0)
hb_db = 10*np.log10(hb + eps)
base = np.percentile(hb_db, 20)
print(f'\nbaseline (20th pct) = {base:.1f} dB')
print('hum-window median above baseline:', np.median(hb_db[hum_mask]) - base)
# how it behaves in second run
m2 = (t > 384) & (t < 446)
print('2nd-run median above baseline:', np.median(hb_db[m2]) - base)
# distribution outside microwave steps
outside = ~(((t >= 70.4) & (t <= 134.5)) | ((t >= 377.4) & (t <= 451.9)))
print('outside: median %.1f  90pct %.1f  99pct %.1f (dB above baseline)' % tuple(
    np.percentile(hb_db[outside] - base, [50, 90, 99])))

# ===== BEEP exploration: narrowband peak in 1-3.5kHz with short STFT =====
NF2 = 2048; H2 = 512  # 128ms window, 32ms hop
f2, t2, Z2 = stft(x, fs=fs, nperseg=NF2, noverlap=NF2-H2, window='hann', padded=False, boundary=None)
P2 = np.abs(Z2)**2
bm = (f2 >= 1000) & (f2 <= 3500)
fb = f2[bm]; Pb = P2[bm]
peak = Pb.max(axis=0)
peakf = fb[Pb.argmax(axis=0)]
# tonality: peak bin power vs band total
tone = peak / (Pb.sum(axis=0) + eps)
# also vs full-spectrum
cand = (tone > 0.3) & (10*np.log10(peak + eps) > -75)
print('\n=== beep candidates (tonality>0.3, peak>-75dB), grouped ===')
idx = np.where(cand)[0]
groups = []
for i in idx:
    if groups and t2[i] - groups[-1][-1][0] < 0.3:
        groups[-1].append((t2[i], peakf[i], tone[i], 10*np.log10(peak[i]+eps)))
    else:
        groups.append([(t2[i], peakf[i], tone[i], 10*np.log10(peak[i]+eps))])
for g in groups:
    ts = [a[0] for a in g]; fr = [a[1] for a in g]
    print(f't={ts[0]:7.2f}-{ts[-1]:7.2f}  n={len(g):3d}  freq={np.median(fr):7.1f}Hz  maxtone={max(a[2] for a in g):.2f}  maxdB={max(a[3] for a in g):.1f}')
