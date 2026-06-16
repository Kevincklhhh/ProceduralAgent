"""Exploratory analysis of microwave hum signature on 8_16 (clean run)."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio/8_16_16k.wav'
fs, x = wavfile.read(AUDIO)
x = x.astype(np.float64) / 32768.0

NFFT = 8192
HOP = 4096
f, t, Z = stft(x, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP, window='hann', padded=False, boundary=None)
P = np.abs(Z)**2  # power spectrogram, shape (freq, time)
print('frames:', P.shape[1], 'freq bins:', P.shape[0], 'df=%.2f Hz' % (f[1]-f[0]))

def band(lo, hi):
    m = (f >= lo) & (f < hi)
    return P[m].sum(axis=0)

bands = {
    'b0_60': band(10, 60),
    'b60_200': band(60, 200),
    'b200_500': band(200, 500),
    'b500_1k': band(500, 1000),
    'b1k_2k': band(1000, 2000),
    'b2k_4k': band(2000, 4000),
    'b4k_8k': band(4000, 8000),
    'total': band(10, 8000),
}
# spectral flatness in 200-2000 (fan noise broadband => high flatness when mic noise dominates)
m = (f >= 200) & (f < 2000)
eps = 1e-12
flat = np.exp(np.mean(np.log(P[m] + eps), axis=0)) / (np.mean(P[m], axis=0) + eps)

# spectral stationarity: correlation of log-spectrum between adjacent frames
L = np.log(P[(f >= 100) & (f < 6000)] + eps)
corr = np.zeros(L.shape[1])
for i in range(1, L.shape[1]):
    a, b = L[:, i-1], L[:, i]
    a = a - a.mean(); b = b - b.mean()
    corr[i] = (a*b).sum() / (np.sqrt((a*a).sum()*(b*b).sum()) + eps)

gt_on = [(70.4, 134.5), (377.4, 451.9)]  # GT step windows (include walking)
inwin = np.zeros(len(t), bool)
for a, b in gt_on:
    inwin |= (t >= a) & (t <= b)

print('\n=== median feature values: inside GT microwave steps vs outside ===')
for name, v in bands.items():
    print(f'{name:10s} in={np.median(v[inwin]):.3e}  out={np.median(v[~inwin]):.3e}  ratio={np.median(v[inwin])/max(np.median(v[~inwin]),1e-15):.2f}')
print(f'flatness   in={np.median(flat[inwin]):.4f}  out={np.median(flat[~inwin]):.4f}')
print(f'statcorr   in={np.median(corr[inwin]):.4f}  out={np.median(corr[~inwin]):.4f}')

# print a downsampled time series of key ratios to see structure (every 4s)
r_low = bands['b60_200'] / (bands['total'] + eps)
r_mid = (bands['b200_500'] + bands['b500_1k']) / (bands['total'] + eps)
logtot = 10*np.log10(bands['total'] + eps)
print('\n t     logtot  r60-200  r200-1k  flat   corr')
for i in range(0, len(t), 16):
    s = slice(i, min(i+16, len(t)))
    mark = '*' if inwin[i] else ' '
    print(f'{t[i]:6.1f}{mark} {np.median(logtot[s]):7.1f} {np.median(r_low[s]):.3f}   {np.median(r_mid[s]):.3f}  {np.median(flat[s]):.3f}  {np.median(corr[s]):.3f}')

# spectrogram plot
fig, ax = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
S = 10*np.log10(P + eps)
im = ax[0].pcolormesh(t, f[:1600], S[:1600], shading='auto', cmap='magma', vmin=-130, vmax=-50)
ax[0].set_ylabel('Hz (0-3.1k)')
for a, b in gt_on:
    for axx in ax: axx.axvline(a, color='cyan', ls='--'); axx.axvline(b, color='cyan', ls='--')
ax[1].plot(t, logtot, lw=0.5, label='total dB')
ax[1].plot(t, 10*np.log10(bands['b60_200']+eps), lw=0.5, label='60-200 dB')
ax[1].legend(); ax[1].set_ylabel('dB')
ax[2].plot(t, corr, lw=0.5, label='stationarity corr')
ax[2].plot(t, flat, lw=0.5, label='flatness 200-2k')
ax[2].legend(); ax[2].set_xlabel('s')
plt.tight_layout()
plt.savefig('/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes/explore_8_16.png', dpi=80)
print('\nsaved plot')
