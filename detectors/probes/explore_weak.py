"""What does the hum look like on weak recordings (8_25, 8_50)? Spectral lines + stationarity."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'

cases = [  # rec, hum window (safe interior), background window
    ('8_25', (85, 145), (240, 300)),
    ('8_50', (105, 155), (30, 60)),
    ('8_16', (76, 128), (200, 290)),
    ('8_26', (60, 170), (230, 310)),
]
for rec, hw, bw in cases:
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_16k.wav')
    x = x.astype(np.float64) / 32768.0
    f, t, Z = stft(x, fs=fs, nperseg=8192, noverlap=4096, window='hann', padded=False, boundary=None)
    P = np.abs(Z) ** 2
    hm = (t >= hw[0]) & (t <= hw[1])
    bm = (t >= bw[0]) & (t <= bw[1])
    Sh = np.median(P[:, hm], axis=1)  # median spectrum robust to transients
    Sb = np.median(P[:, bm], axis=1)
    print(f'=== {rec} median spectrum: hum-bg contrast (dB) per 25Hz, 50-700 Hz ===')
    row = []
    for lo in range(50, 700, 25):
        m = (f >= lo) & (f < lo + 25)
        d = 10 * np.log10((Sh[m].mean() + 1e-14) / (Sb[m].mean() + 1e-14))
        row.append(f'{lo}:{d:.0f}')
    print('  ' + ' '.join(row))
    # strongest narrow lines during hum: peak vs local floor (line contrast), 80-500 Hz
    sel = (f >= 80) & (f <= 520)
    fsel = f[sel]; S = 10 * np.log10(Sh[sel] + 1e-14)
    # local floor: median in +/-15 bins (~30Hz)
    floor = np.array([np.median(S[max(0, i - 15):i + 16]) for i in range(len(S))])
    contrast = S - floor
    top = np.argsort(contrast)[-6:][::-1]
    print('  top hum lines: ' + ', '.join(f'{fsel[i]:.0f}Hz(+{contrast[i]:.0f}dB)' for i in sorted(top, key=lambda i: fsel[i])))
    # stationarity of 100-1000 band level: rolling std (9 frames ~2.3s), median inside hum vs bg
    hb = 10 * np.log10(P[(f >= 100) & (f < 1000)].sum(axis=0) + 1e-12)
    k = 9
    rs = np.array([np.std(hb[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(hb))])
    print(f'  rolling-std(2.3s) of band dB: hum_med={np.median(rs[hm]):.2f}  bg_med={np.median(rs[bm]):.2f}')
    # line-tracking feature: power in 2Hz around 120Hz and around strongest line vs floor power
    def line_feat(fc, half=3):  # half bins of 1.95Hz
        i0 = np.argmin(np.abs(f - fc))
        line = P[i0 - half:i0 + half + 1].sum(axis=0)
        ring = P[i0 - 12:i0 - 5].sum(axis=0) + P[i0 + 5:i0 + 12].sum(axis=0)
        return 10 * np.log10((line + 1e-14) / (ring + 1e-14))
    for fc in (120.0, 240.0, 218.0):
        lf = line_feat(fc)
        print(f'  line {fc:.0f}Hz contrast: hum_med={np.median(lf[hm]):.1f} dB  bg_med={np.median(lf[bm]):.1f} dB')
