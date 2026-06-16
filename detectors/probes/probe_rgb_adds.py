#!/usr/bin/env python3
"""Probe: rgb_adds — can cheap RGB mark each 'adds' step of Spiced Hot Chocolate complete?

Tests the completion criterion in data/sensor_schedule_spicedhotchocolate.json for the adds
stages (chocolate/cinnamon/sugar), over ALL 16 SHC recordings.

FINDING THAT SHAPED THIS PROBE (8_16): the naive R3 hand-gate (absolute YCrCb skin fraction)
FAILS here — skin-colored surfaces (wood counter, milk, beige mug) keep skin% ~0.5 baseline,
so an absolute skin threshold fires the whole window. We therefore make R4 (mug-ROI HSV
state-change) the PRIMARY add detector; skin/motion are kept only as diagnostics.

R4 detector (cheap, no GPU/VLM):
  - tight lower-center mug ROI (stand-in for R1 one-shot grounding; documented limitation)
  - sliding HSV(16x16 H-S) histogram chi-square between mean hist over [t-4,t-1]s vs [t+1,t+4]s
  - peaks = local maxima with chi2 >= STATE_CHI2_MIN, min-separation PEAK_MINSEP_S
  - each peak = "a mug state change" => an add step marked complete

Scoring per recording: a GT add window [s,e] is recalled if a peak lands in [s-TOL, e+TOL]
(greedy one-to-one). Peaks not matching any add window = false adds (over-segmentation).

Protocol: STATE_CHI2_MIN / PEAK_MINSEP_S tuned on 8_16 (clean) ONLY, frozen, eval on other 15.
GT add steps from complete_step_annotations.json; skipped add (start=-1) excluded, so the
expected count is the number of PRESENT add steps (handles error recordings).

Outputs: results_rgb_adds.json. Usage: python probe_rgb_adds.py [--only 8_16]
"""
import json, os, argparse
import cv2
import numpy as np

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
VID = os.path.join(BASE, 'data', 'videos_360p')
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
PROBE_DIR = os.path.join(BASE, 'detectors', 'probes')
ADD_STEPS = {90, 84, 87}
TUNE_REC = '8_16'
SAMPLE_FPS = 3.0
TOL_S = 15.0
PAD_S = 8.0
ROI = (0.30, 0.72, 0.34, 0.66)    # tight lower-center mug crop

# frozen params (tuned on 8_16: recall 3/3, 1 spurious peak)
STATE_CHI2_MIN = 2.5
PEAK_MINSEP_S = 15.0
SKIN_DIAG_MIN = 0.10              # diagnostic only, not used for detection


def skin_frac(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = y[..., 0], y[..., 1], y[..., 2]
    return float(((Y > 80) & (Cr > 133) & (Cr < 173) & (Cb > 77) & (Cb < 127)).mean())


def hsv_hist(bgr):
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = cv2.calcHist([h], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(H, H, 0, 1, cv2.NORM_MINMAX)
    return H.flatten()


def adds_window(rec):
    steps = {s['step_id']: s for s in ANN[rec]['steps']}
    present = [(sid, steps[sid]) for sid in ADD_STEPS
              if sid in steps and steps[sid]['start_time'] >= 0]
    if not present:
        return None, []
    gt = sorted([(sid, float(s['start_time']), float(s['end_time'])) for sid, s in present],
                key=lambda x: x[1])
    return (max(0.0, gt[0][1] - PAD_S), max(e for _, _, e in gt) + PAD_S), gt


def extract(rec, win):
    cache = os.path.join(PROBE_DIR, f'rgb_adds_{rec}.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['t'], d['skin'], d['hists']
    cap = cv2.VideoCapture(os.path.join(VID, f'{rec}.mp4'))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {rec}')
    r0, r1, c0, c1 = ROI
    ts, sk, hs = [], [], []
    t = win[0]
    while t <= win[1]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok:
            break
        H, W = fr.shape[:2]
        roi = fr[int(r0 * H):int(r1 * H), int(c0 * W):int(c1 * W)]
        ts.append(t); sk.append(skin_frac(roi)); hs.append(hsv_hist(roi))
        t += 1.0 / SAMPLE_FPS
    cap.release()
    t, sk, hs = np.array(ts), np.array(sk), np.array(hs)
    np.savez(cache, t=t, skin=sk, hists=hs)
    return t, sk, hs


def chi2_series(t, hists):
    d, g = int(3 * SAMPLE_FPS), int(1 * SAMPLE_FPS)
    chi = np.zeros(len(hists))
    for i in range(len(hists)):
        if i - g <= 0 or i + g + d > len(hists):
            continue
        pre = hists[max(0, i - g - d):i - g].mean(0)
        post = hists[i + g:i + g + d].mean(0)
        chi[i] = 0.5 * np.sum((pre - post) ** 2 / (pre + post + 1e-9))
    return chi


def find_peaks(t, chi):
    order = np.argsort(chi)[::-1]
    peaks = []
    for i in order:
        if chi[i] < STATE_CHI2_MIN:
            break
        if all(abs(t[i] - t[j]) > PEAK_MINSEP_S for j in peaks):
            peaks.append(i)
    return sorted(peaks)


def match(peak_ts, gt_adds, tol):
    used = set(); pairs = []
    for pt in peak_ts:
        best, bd = None, tol + 1
        for k, (sid, s, e) in enumerate(gt_adds):
            if k in used:
                continue
            d = 0.0 if s - tol <= pt <= e + tol else min(abs(pt - s), abs(pt - e))
            if (s - tol <= pt <= e + tol) and d < bd:
                bd, best = d, k
        if best is not None:
            used.add(best)
            pairs.append({'peak_t': round(pt, 1), 'gt_step': gt_adds[best][0],
                          'gt_window': [round(gt_adds[best][1], 1), round(gt_adds[best][2], 1)]})
    return len(used), pairs


def run_one(rec):
    win, gt = adds_window(rec)
    if win is None:
        return {'recording': rec, 'note': 'all adds skipped', 'expected_adds': 0}
    t, skin, hists = extract(rec, win)
    chi = chi2_series(t, hists)
    peaks = find_peaks(t, chi)
    peak_ts = [float(t[i]) for i in peaks]
    tp, pairs = match(peak_ts, gt, TOL_S)
    return {'recording': rec, 'window': [round(win[0], 1), round(win[1], 1)],
            'expected_adds': len(gt), 'n_peaks': len(peaks),
            'matched_adds_TP': tp, 'spurious_peaks': len(peaks) - tp,
            'recall': round(tp / len(gt), 3) if gt else None,
            'precision': round(tp / len(peaks), 3) if peaks else None,
            'chi2_baseline_med': round(float(np.median(chi[chi > 0])), 3) if (chi > 0).any() else None,
            'peak_times': [round(x, 1) for x in peak_ts],
            'gt_add_windows': [[s, round(a, 1), round(b, 1)] for s, a, b in gt],
            'matches': pairs,
            'skin_diag_med': round(float(np.median(skin)), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only'); args = ap.parse_args()
    recs = [args.only] if args.only else \
        ['8_11', '8_15', '8_16', '8_19', '8_20', '8_25', '8_26', '8_3',
         '8_30', '8_31', '8_33', '8_35', '8_40', '8_44', '8_45', '8_50']
    per = {r: run_one(r) for r in recs}
    exp = sum(v.get('expected_adds', 0) for v in per.values())
    tp = sum(v.get('matched_adds_TP', 0) for v in per.values())
    npk = sum(v.get('n_peaks', 0) for v in per.values())
    summary = {'tuned_on': TUNE_REC, 'sample_fps': SAMPLE_FPS, 'roi_frac': ROI, 'tol_s': TOL_S,
               'state_chi2_min': STATE_CHI2_MIN, 'peak_minsep_s': PEAK_MINSEP_S,
               'finding': 'R3 absolute-skin hand-gate FAILS (skin-colored surfaces); R4 HSV '
                          'state-change is the working primitive.',
               'total_expected_adds': exp, 'total_matched_TP': tp, 'total_peaks': npk,
               'pooled_recall': round(tp / exp, 3) if exp else None,
               'pooled_precision': round(tp / npk, 3) if npk else None}
    print(json.dumps(summary, indent=1))
    print('\nrec     exp peaks  TP  recall  prec   skin_med  chi_med')
    for r in recs:
        v = per[r]
        if 'recall' in v:
            print(f"  {r:6s}  {v['expected_adds']}   {v['n_peaks']:2d}   {v['matched_adds_TP']}   "
                  f"{str(v['recall']):5s}  {str(v['precision']):5s}  {v['skin_diag_med']:5}    "
                  f"{v['chi2_baseline_med']}{'   [TUNE]' if r == TUNE_REC else ''}")
        else:
            print(f"  {r:6s}  {v.get('note')}")
    if not args.only:
        json.dump({'probe': 'rgb_adds', 'summary': summary, 'per_recording': per},
                  open(os.path.join(PROBE_DIR, 'results_rgb_adds.json'), 'w'), indent=1)
        print('\nwrote results_rgb_adds.json')


if __name__ == '__main__':
    main()
