"""Probe pour_clink: classical audio detectors for POUR (GT step 88) and
SPOON CLINK / STIR trains (GT step 85), CaptainCook4D activity 8.

All thresholds were tuned ONLY on 8_16 (clean run) and are frozen below.
Evaluation runs on the other five recordings with identical parameters.

POUR detector:  1-8 kHz band energy (dB) vs 31 s rolling-median background,
                rel > 4 dB, spectral flatness >= 0.12 (rejects tonal
                scrape/ring segments), gap-merge 0.4 s, duration 0.8-8 s.
CLINK detector: spectral-flux onsets in 4-16 kHz (median + 6*MAD adaptive,
                70 ms min separation); onset is "ringy"/metallic if the
                1.5-12 kHz spectrum 20-80 ms after onset has peak/median
                >= 22 dB; train = >= 3 ringy onsets in any 5 s window
                (0.5 s grid, 3 s gap merge, >= 3 s long).  "Strong" tier:
                duration >= 8 s AND ringy-onset density >= 1.5 /s
                (on 8_16 this isolates exactly the GT mix step).
"""
import json
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, medfilt

AUDIO_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/data/audio'
GT_PATH = '/home/kailaic/NeuroTrace/ProceduralAgent/data/gt_activity8.json'
OUT_PATH = '/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes/results_pour_clink.json'

TUNE_REC = '8_16'
EVAL_RECS = ['8_26', '8_3', '8_25', '8_31', '8_50']

# ----------------------------- frozen parameters -----------------------------
P = dict(
    # pour
    pour_nfft=2048, pour_hop=960,            # 20 ms hop @ 48 kHz
    pour_band=(1000, 8000),
    pour_smooth_frames=15,                   # ~0.3 s median smoothing
    pour_bg_win_s=31.0,                      # rolling median background
    pour_rel_db=4.0,
    pour_flat_min=0.12,
    pour_gap_s=0.4, pour_min_s=0.8, pour_max_s=8.0,
    # clink
    clink_nfft=1024, clink_hop=480,          # 10 ms hop @ 48 kHz
    clink_flux_band=(4000, 16000),
    clink_mad_k=6.0, clink_minsep_s=0.07,
    ring_band=(1500, 12000), ring_db=22.0,   # ringiness gate (metallic)
    train_win_s=5.0, train_min_onsets=3, train_grid_s=0.5,
    train_gap_s=3.0, train_min_dur_s=3.0,
    strong_min_dur_s=8.0, strong_min_dens=1.5,
)


def detect_pour(x, sr):
    nfft, hop = P['pour_nfft'], P['pour_hop']
    f, t, Z = stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                   padded=False, boundary=None)
    Pw = np.abs(Z) ** 2
    band = (f >= P['pour_band'][0]) & (f <= P['pour_band'][1])
    db = 10 * np.log10(Pw[band].sum(axis=0) + 1e-12)
    logp = np.log(Pw[band] + 1e-12)
    flat = np.exp(logp.mean(axis=0)) / (Pw[band].mean(axis=0) + 1e-12)
    db_s = medfilt(db, P['pour_smooth_frames'])
    flat_s = medfilt(flat, P['pour_smooth_frames'])
    hop_s = hop / sr
    win = int(P['pour_bg_win_s'] / hop_s)
    sub = np.arange(0, len(db_s), 50)
    bg = np.array([np.median(db_s[max(0, i - win // 2): i + win // 2 + 1]) for i in sub])
    rel = db_s - np.interp(np.arange(len(db_s)), sub, bg)

    mask = rel > P['pour_rel_db']
    idx = np.where(mask)[0]
    dets = []
    if len(idx):
        s0 = p = idx[0]
        regs = []
        for i in idx[1:]:
            if (i - p) * hop_s > P['pour_gap_s']:
                regs.append((s0, p))
                s0 = i
            p = i
        regs.append((s0, p))
        for a, b in regs:
            dur = (b - a) * hop_s
            if not (P['pour_min_s'] <= dur <= P['pour_max_s']):
                continue
            fl = float(np.mean(flat_s[a:b + 1]))
            if fl < P['pour_flat_min']:
                continue
            dets.append(dict(start=round(float(t[a]), 2), end=round(float(t[b]), 2),
                             flatness=round(fl, 3),
                             rel_db=round(float(np.max(rel[a:b + 1])), 1)))
    return dets


def detect_clink(x, sr):
    nfft, hop = P['clink_nfft'], P['clink_hop']
    f, t, Z = stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop,
                   padded=False, boundary=None)
    M = np.abs(Z)
    hb = (f >= P['clink_flux_band'][0]) & (f <= P['clink_flux_band'][1])
    rb = (f >= P['ring_band'][0]) & (f <= P['ring_band'][1])
    flux = np.concatenate([[0.0], np.maximum(M[hb][:, 1:] - M[hb][:, :-1], 0).sum(axis=0)])
    med = medfilt(flux, 501)
    mad = medfilt(np.abs(flux - med), 501) + 1e-9
    above = flux > med + P['clink_mad_k'] * mad

    onsets = []
    i = 1
    while i < len(flux) - 1:
        if above[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if not onsets or t[i] - onsets[-1][0] > P['clink_minsep_s']:
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
        if pk >= P['ring_db']:
            ringy.append(tt)
    ringy = np.array(ringy)

    # transient trains: >= train_min_onsets ringy onsets in any 5 s window
    g = np.arange(0, t[-1], P['train_grid_s'])
    act = np.array([((ringy >= a) & (ringy < a + P['train_win_s'])).sum()
                    >= P['train_min_onsets'] for a in g])
    evs = []
    idx = np.where(act)[0]
    if len(idx):
        s0 = p = idx[0]
        raw = []
        for i in idx[1:]:
            if (i - p) * P['train_grid_s'] > P['train_gap_s']:
                raw.append((g[s0], g[p] + P['train_win_s']))
                s0 = i
            p = i
        raw.append((g[s0], g[p] + P['train_win_s']))
        for a, b in raw:
            if b - a < P['train_min_dur_s']:
                continue
            n = int(((ringy >= a) & (ringy <= b)).sum())
            dens = n / (b - a)
            evs.append(dict(start=round(float(a), 2), end=round(float(b), 2),
                            n_ringy_onsets=n, density=round(float(dens), 2),
                            strong=bool(b - a >= P['strong_min_dur_s']
                                        and dens >= P['strong_min_dens'])))
    return evs, len(onsets), len(ringy)


def overlap_steps(a, b, steps):
    return sorted(sid for sid, (s, e) in steps.items()
                  if s >= 0 and a < e and b > s)


def evaluate(rec, gt):
    t0 = time.time()
    sr, x = wavfile.read(f'{AUDIO_DIR}/{rec}_48k.wav')
    x = x.astype(np.float64) / 32768.0
    t_load = time.time() - t0

    steps = {s['step_id']: (s['start_time'], s['end_time']) for s in gt['steps']}

    t1 = time.time()
    pours = detect_pour(x, sr)
    t_pour = time.time() - t1
    t2 = time.time()
    clinks, n_onsets, n_ringy = detect_clink(x, sr)
    t_clink = time.time() - t2

    # ---- pour vs step 88
    s88 = steps.get(88, (-1, -1))
    pour_hits = [d for d in pours if s88[0] >= 0 and d['start'] < s88[1] and d['end'] > s88[0]]
    pour_fas = [d for d in pours if d not in pour_hits]
    fa_by_step = {}
    for d in pour_fas:
        ov = overlap_steps(d['start'], d['end'], steps)
        key = ','.join(str(s) for s in ov if s != 88) or 'none'
        fa_by_step[key] = fa_by_step.get(key, 0) + 1

    # ---- clink vs step 85
    s85 = steps.get(85, (-1, -1))
    step85_present = s85[0] >= 0
    cl_hits = [e for e in clinks if step85_present and e['start'] < s85[1] and e['end'] > s85[0]]
    cl_fas = [e for e in clinks if e not in cl_hits]
    cl_fa_by_step = {}
    for e in cl_fas:
        ov = overlap_steps(e['start'], e['end'], steps)
        key = ','.join(str(s) for s in ov if s != 85) or 'none'
        cl_fa_by_step[key] = cl_fa_by_step.get(key, 0) + 1
    strong = [e for e in clinks if e['strong']]
    strong_hits = [e for e in strong if step85_present and e['start'] < s85[1] and e['end'] > s85[0]]

    dur = len(x) / sr
    return dict(
        duration_s=round(dur, 1),
        gt_step88=[round(v, 1) for v in s88],
        gt_step85=[round(v, 1) for v in s85],
        pour=dict(
            hit=bool(pour_hits),
            n_detections=len(pours),
            detections_in_step88=pour_hits,
            n_false_alarms=len(pour_fas),
            fa_per_min_outside_88=round(len(pour_fas) / ((dur - max(0, s88[1] - s88[0])) / 60), 2),
            fa_overlap_steps=fa_by_step,
            all_detections=pours,
        ),
        clink=dict(
            step85_present=step85_present,
            hit=(bool(cl_hits) if step85_present else None),
            n_events=len(clinks),
            n_onsets_total=n_onsets, n_ringy_onsets=n_ringy,
            events_overlapping_step85=cl_hits,
            n_false_alarms=len(cl_fas),
            fa_overlap_steps=cl_fa_by_step,
            strong_tier=dict(
                n_strong_events=len(strong),
                hit=(bool(strong_hits) if step85_present else None),
                n_false_alarms=len(strong) - len(strong_hits),
                events=strong),
            all_events=clinks,
        ),
        wall_time_s=dict(load=round(t_load, 2), pour=round(t_pour, 2),
                         clink=round(t_clink, 2),
                         total=round(t_load + t_pour + t_clink, 2)),
    )


def main():
    gt = json.load(open(GT_PATH))
    out = dict(probe='pour_clink', tuning_recording=TUNE_REC,
               frozen_params={k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in P.items()},
               tuned_on_8_16={}, frozen_eval={})
    out['tuned_on_8_16'][TUNE_REC] = evaluate(TUNE_REC, gt[TUNE_REC])
    for rec in EVAL_RECS:
        print(f'... {rec}')
        out['frozen_eval'][rec] = evaluate(rec, gt[rec])

    with open(OUT_PATH, 'w') as fh:
        json.dump(out, fh, indent=1)

    # compact console summary
    for grp in ['tuned_on_8_16', 'frozen_eval']:
        print(f'===== {grp} =====')
        for rec, r in out[grp].items():
            p, c = r['pour'], r['clink']
            ch = c['hit'] if c['step85_present'] else 'N/A(skipped)'
            sh = c['strong_tier']['hit'] if c['step85_present'] else 'N/A'
            print(f"{rec}: POUR hit={p['hit']} FA={p['n_false_alarms']} "
                  f"({p['fa_per_min_outside_88']}/min) | "
                  f"CLINK hit={ch} FA={c['n_false_alarms']} "
                  f"strong: hit={sh} FA={c['strong_tier']['n_false_alarms']} | "
                  f"wall={r['wall_time_s']['total']}s")


if __name__ == '__main__':
    main()
