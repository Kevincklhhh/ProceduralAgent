"""detectors_lib -- importable wrappers around the FROZEN probe detectors.

Code is a verbatim port of the validated probe implementations:
  - microwave hum runs + beeps : detectors/probes/hum_beep.py   (16 kHz wav)
  - pour bursts + clink trains : detectors/probes/pour_clink.py (48 kHz wav)

Parameters are NOT redefined here; they are loaded from the frozen probe
result files (results_hum_beep.json top-level "params", results_pour_clink.json
"frozen_params") so nothing can drift from the validated probe phase.

Causality / lookahead note (the engine treats decisions at time t as using
audio up to t + lookahead):
  - hum mask: cascaded centered median filters. UPDATED 2026-06-15 for short-cycle
    detection -- feature medfilt 7 frames (+-0.90 s) -> rolling std 17 frames
    (+-2.05 s) -> mask medfilt 7 frames (+-0.90 s); worst-case effective lookahead
    ~3.85 s (now the 17-frame rolling-std dominates). Previously 21-frame medfilts
    gave ~7.2 s; the smaller windows + min_run_s 20->8 let it catch ~10-30 s hums.
  - beep grouping: <= 1.5 s.
  - clink train: 5 s confirmation window.
  - pour: 31 s centered rolling-median background (~15.5 s lookahead) --
    pour is therefore a LOGGED SECONDARY signal only, never gates decisions.
"""
import json

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, medfilt

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'
HUM_BEEP_RESULTS = ('/home/kailaic/NeuroTrace/ProceduralAgent/detectors/'
                    'probes/results_hum_beep.json')
POUR_CLINK_RESULTS = ('/home/kailaic/NeuroTrace/ProceduralAgent/detectors/'
                      'probes/results_pour_clink.json')

HUM_LOOKAHEAD_S = 3.9   # worst case after 2026-06-15 short-cycle retune (was 7.2);
                        # dominant window now the 17-frame rolling std (~4.3 s span)
CLINK_LOOKAHEAD_S = 5.0
BEEP_LOOKAHEAD_S = 1.5


def load_frozen_params():
    """Return (hum_beep_params, pour_clink_params) from the frozen probe runs."""
    with open(HUM_BEEP_RESULTS) as fh:
        hb = json.load(fh)['params']
    with open(POUR_CLINK_RESULTS) as fh:
        pc = json.load(fh)['frozen_params']
    # tuples were serialized as lists
    for k, v in pc.items():
        if isinstance(v, list):
            pc[k] = tuple(v)
    return hb, pc


def load_audio_16k(rec):
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_16k.wav')
    return fs, x.astype(np.float64) / 32768.0


def load_audio_48k(rec):
    fs, x = wavfile.read(f'{AUDIO_DIR}/{rec}_48k.wav')
    return fs, x.astype(np.float64) / 32768.0


# --------------------------------------------------------------------------
# microwave hum + beep (verbatim from probes/hum_beep.py)
# --------------------------------------------------------------------------
def _hum_features(x, fs, p):
    f, t, Z = stft(x, fs=fs, nperseg=p['nfft'], noverlap=p['nfft'] - p['hop'],
                   window='hann', padded=False, boundary=None)
    P = (np.abs(Z) ** 2)
    band = (f >= p['hum_lo']) & (f < p['hum_hi'])
    hb_db = 10.0 * np.log10(P[band].sum(axis=0) + 1e-12)
    f1 = hb_db - np.percentile(hb_db, p['baseline_pct'])
    i0 = int(np.argmin(np.abs(f - p['mains_line_hz'])))
    line = P[i0 - 3:i0 + 4].sum(axis=0)
    ring = P[i0 - 12:i0 - 4].sum(axis=0) + P[i0 + 5:i0 + 13].sum(axis=0)
    f2 = 10.0 * np.log10((line + 1e-14) / (ring + 1e-14))
    f1 = medfilt(f1, kernel_size=p['feat_med_frames'])
    f2 = medfilt(f2, kernel_size=p['feat_med_frames'])
    k = p['rollstd_frames']
    pad = k // 2
    hp = np.pad(f1, pad, mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(hp, k)
    f3 = win.std(axis=1)
    return t, f1, f2, f3


def _runs_from_mask(on, t, dt, p):
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            runs.append([float(t[i]) - dt / 2, float(t[j]) + dt / 2])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < p['merge_gap_s']:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if (b - a) > p['min_run_s']]


def detect_hum_runs(x, fs, p):
    """Kept hum runs (>= min_run_s = 20 s) as [(start_s, end_s), ...]."""
    t, f1, f2, f3 = _hum_features(x, fs, p)
    vote = ((f1 > p['t1_level_db']).astype(int)
            + (f2 > p['t2_line_db']).astype(int)
            + (f3 < p['t3_std_db']).astype(int))
    on = (f1 > p['gate_db']) & (vote >= p['vote_needed'])
    on = medfilt(on.astype(float), kernel_size=p['med_frames']) > 0.5
    return _runs_from_mask(on, t, p['hop'] / fs, p)


def detect_beeps(x, fs, p):
    """Tonal beeps 800-5000 Hz: [{t, dur_s, freq_hz, peak_db}, ...]."""
    f, t, Z = stft(x, fs=fs, nperseg=p['b_nfft'], noverlap=p['b_nfft'] - p['b_hop'],
                   window='hann', padded=False, boundary=None)
    P = (np.abs(Z) ** 2)
    bm = (f >= p['beep_lo']) & (f <= p['beep_hi'])
    fb, Pb = f[bm], P[bm]
    peak = Pb.max(axis=0)
    peakf = fb[Pb.argmax(axis=0)]
    tone = peak / (Pb.sum(axis=0) + 1e-12)
    peak_db = 10.0 * np.log10(peak + 1e-12)
    cand = np.where((tone > p['tone_ratio']) & (peak_db > p['peak_db_min']))[0]

    groups, cur = [], []
    for i in cand:
        if cur and (t[i] - t[cur[-1]]) >= p['group_gap_s']:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)

    beeps = []
    for g in groups:
        ts, fr = t[g], peakf[g]
        if len(g) < p['min_frames'] or (ts[-1] - ts[0]) > p['max_beep_s']:
            continue
        if np.percentile(fr, 75) - np.percentile(fr, 25) > p['max_freq_spread_hz']:
            continue
        beeps.append(dict(t=round(float(0.5 * (ts[0] + ts[-1])), 2),
                          dur_s=round(float(ts[-1] - ts[0]), 2),
                          freq_hz=round(float(np.median(fr)), 1),
                          peak_db=round(float(peak_db[g].max()), 1)))
    return beeps


# --------------------------------------------------------------------------
# pour + clink trains (verbatim from probes/pour_clink.py, params passed in)
# --------------------------------------------------------------------------
def detect_pours(x, sr, p):
    """Broadband pour bursts: [{start, end, flatness, rel_db}, ...].
    SECONDARY signal only (too many false alarms to gate decisions)."""
    nfft, hop = p['pour_nfft'], p['pour_hop']
    f, t, Z = stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                   padded=False, boundary=None)
    Pw = np.abs(Z) ** 2
    band = (f >= p['pour_band'][0]) & (f <= p['pour_band'][1])
    db = 10 * np.log10(Pw[band].sum(axis=0) + 1e-12)
    logp = np.log(Pw[band] + 1e-12)
    flat = np.exp(logp.mean(axis=0)) / (Pw[band].mean(axis=0) + 1e-12)
    db_s = medfilt(db, p['pour_smooth_frames'])
    flat_s = medfilt(flat, p['pour_smooth_frames'])
    hop_s = hop / sr
    win = int(p['pour_bg_win_s'] / hop_s)
    sub = np.arange(0, len(db_s), 50)
    bg = np.array([np.median(db_s[max(0, i - win // 2): i + win // 2 + 1]) for i in sub])
    rel = db_s - np.interp(np.arange(len(db_s)), sub, bg)

    mask = rel > p['pour_rel_db']
    idx = np.where(mask)[0]
    dets = []
    if len(idx):
        s0 = q = idx[0]
        regs = []
        for i in idx[1:]:
            if (i - q) * hop_s > p['pour_gap_s']:
                regs.append((s0, q))
                s0 = i
            q = i
        regs.append((s0, q))
        for a, b in regs:
            dur = (b - a) * hop_s
            if not (p['pour_min_s'] <= dur <= p['pour_max_s']):
                continue
            fl = float(np.mean(flat_s[a:b + 1]))
            if fl < p['pour_flat_min']:
                continue
            dets.append(dict(start=round(float(t[a]), 2), end=round(float(t[b]), 2),
                             flatness=round(fl, 3),
                             rel_db=round(float(np.max(rel[a:b + 1])), 1)))
    return dets


def detect_sizzle_runs(x, fs, p):
    """A4 sizzle/fry: SUSTAINED broadband runs from food in hot oil.

    Returns [(start_s, end_s), ...]. The discriminator is TEMPORAL PERSISTENCE,
    not spectrum: per EPIC-SOUNDS sizzling/boiling is a long-form/background class
    (median > 10 s) acoustically inseparable from intermediate broadband bursts
    (water, rustle, whisk, chop). Those bursts are intermittent, so a rolling
    MEDIAN of the 1.5-7 kHz band energy stays near the prep floor during prep but
    steps up and holds during frying. We gate on that sustained level.

    Implementation: band energy per STFT frame -> block-reduced to ~2 Hz (median
    per 0.5 s) -> rolling median over `roll_win_s` -> threshold `level_db` above
    the per-recording sustained floor (low percentile of the rolling median).
    Lookahead = roll_win_s / 2 (centered rolling median); this is a SLOW signal,
    appropriate for a stage-level anchor, NOT a low-latency reactive cue.
    """
    f, t, Z = stft(x, fs=fs, nperseg=p['nfft'], noverlap=p['nfft'] - p['hop'],
                   window='hann', padded=False, boundary=None)
    P = np.abs(Z) ** 2
    band = (f >= p['band_lo']) & (f <= p['band_hi'])
    band_db = 10.0 * np.log10(P[band].sum(axis=0) + 1e-12)

    dt = p['hop'] / fs
    blk = max(1, int(round(p['block_s'] / dt)))
    nb = len(band_db) // blk
    if nb < 3:
        return []
    coarse = np.median(band_db[:nb * blk].reshape(nb, blk), axis=1)
    tc = (np.arange(nb) + 0.5) * blk * dt
    dtc = blk * dt

    w = max(1, int(round(p['roll_win_s'] / dtc)))
    half = w // 2
    roll = np.array([np.median(coarse[max(0, i - half):i + half + 1])
                     for i in range(nb)])
    floor = np.percentile(roll, p['baseline_pct'])
    on = roll - floor > p['level_db']

    cp = dict(p); cp['min_run_s'] = p['min_run_s']
    return _runs_from_mask(on, tc, dtc, cp)


def detect_clink_trains(x, sr, p):
    """Spoon-clink trains: [{start, end, n_ringy_onsets, density, strong}, ...].
    'strong' tier (>= 8 s, >= 1.5 ringy onsets/s) is the stir/mix event."""
    nfft, hop = p['clink_nfft'], p['clink_hop']
    f, t, Z = stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                   padded=False, boundary=None)
    M = np.abs(Z)
    hb = (f >= p['clink_flux_band'][0]) & (f <= p['clink_flux_band'][1])
    rb = (f >= p['ring_band'][0]) & (f <= p['ring_band'][1])
    flux = np.concatenate([[0.0], np.maximum(M[hb][:, 1:] - M[hb][:, :-1], 0).sum(axis=0)])
    med = medfilt(flux, 501)
    mad = medfilt(np.abs(flux - med), 501) + 1e-9
    above = flux > med + p['clink_mad_k'] * mad

    onsets = []
    i = 1
    while i < len(flux) - 1:
        if above[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if not onsets or t[i] - onsets[-1][0] > p['clink_minsep_s']:
                onsets.append((t[i], i))
            i += 7
        else:
            i += 1

    ringy = []
    Mr = M[rb]
    for tt, i in onsets:
        j0, j1 = i + 2, min(i + 9, M.shape[1])  # 20-80 ms post onset
        if j1 <= j0:
            continue
        spec = Mr[:, j0:j1].mean(axis=1)
        pk = 20 * np.log10(spec.max() / (np.median(spec) + 1e-12) + 1e-12)
        if pk >= p['ring_db']:
            ringy.append(tt)
    ringy = np.array(ringy)

    g = np.arange(0, t[-1], p['train_grid_s'])
    act = np.array([((ringy >= a) & (ringy < a + p['train_win_s'])).sum()
                    >= p['train_min_onsets'] for a in g])
    evs = []
    idx = np.where(act)[0]
    if len(idx):
        s0 = q = idx[0]
        raw = []
        for i in idx[1:]:
            if (i - q) * p['train_grid_s'] > p['train_gap_s']:
                raw.append((g[s0], g[q] + p['train_win_s']))
                s0 = i
            q = i
        raw.append((g[s0], g[q] + p['train_win_s']))
        for a, b in raw:
            if b - a < p['train_min_dur_s']:
                continue
            n = int(((ringy >= a) & (ringy <= b)).sum())
            dens = n / (b - a)
            evs.append(dict(start=round(float(a), 2), end=round(float(b), 2),
                            n_ringy_onsets=n, density=round(float(dens), 2),
                            strong=bool(b - a >= p['strong_min_dur_s']
                                        and dens >= p['strong_min_dens'])))
    return evs
