#!/usr/bin/env python3
"""
Probe: vision_motion
Cheap RGB motion detector on egocentric (head-mounted GoPro) video.
- Input: 320p @ 5fps downscaled streams (the cheap tier's actual input).
- Per frame: grayscale Farneback optical flow vs previous frame,
  subtract the median flow vector (crude ego-motion compensation),
  record mean residual motion magnitude + spatial concentration.
- Stirring periodicity: sliding 10s windows, FFT of residual-motion series,
  score = peak power in 0.5-2.5 Hz band / median spectral power.
  (Nyquist at 5 fps is 2.5 Hz, so the requested 0.5-3 Hz band is truncated.)
- Threshold tuned ONLY on 8_16 (clean run), frozen, evaluated on 8_26.

Outputs: results_vision_motion.json (+ optional png plots) in this directory.
"""
import json
import os
import time

import cv2
import numpy as np

PROBE_DIR = "/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes"
GT_PATH = "/home/kailaic/NeuroTrace/ProceduralAgent/data/gt_activity8.json"
RECS = ["8_16", "8_26"]
FPS = 5.0
WIN_S = 10.0          # sliding window length (s)
HOP_S = 1.0           # hop (s)
BAND = (0.5, 2.5)     # Hz; Nyquist = 2.5 Hz at 5 fps
STIR_STEP = 85
TOPK_FRAC = 0.05      # spatial concentration: share of residual energy in top 5% pixels

cv2.setNumThreads(4)


# ---------------------------------------------------------------- motion pass
def extract_motion(rec):
    """Farneback flow at 320p/5fps with median-vector ego compensation.
    Returns dict of per-frame series + timing stats. Cached to npz."""
    cache = os.path.join(PROBE_DIR, f"motion_{rec}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    path = os.path.join(PROBE_DIR, f"{rec}_320p5.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")

    resid_mean, resid_conc, ego_mag, flow_ms = [], [], [], []
    prev = None
    t_wall0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            t0 = time.perf_counter()
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            # ego-motion compensation: subtract per-frame median flow vector
            med = np.median(flow.reshape(-1, 2), axis=0)
            res = flow - med
            mag = np.sqrt(res[..., 0] ** 2 + res[..., 1] ** 2)
            flow_ms.append((time.perf_counter() - t0) * 1000.0)

            resid_mean.append(float(mag.mean()))
            ego_mag.append(float(np.hypot(med[0], med[1])))
            # spatial concentration: residual energy share of top 5% pixels
            flat = np.sort(mag.ravel())
            k = max(1, int(len(flat) * TOPK_FRAC))
            tot = flat.sum()
            resid_conc.append(float(flat[-k:].sum() / tot) if tot > 0 else 0.0)
        prev = gray
    cap.release()
    wall = time.time() - t_wall0

    out = dict(
        resid_mean=np.array(resid_mean, dtype=np.float32),
        resid_conc=np.array(resid_conc, dtype=np.float32),
        ego_mag=np.array(ego_mag, dtype=np.float32),
        flow_ms_mean=np.array([np.mean(flow_ms)]),
        flow_ms_p50=np.array([np.median(flow_ms)]),
        flow_ms_p95=np.array([np.percentile(flow_ms, 95)]),
        wall_s=np.array([wall]),
        n_frames=np.array([len(resid_mean) + 1]),
    )
    np.savez(cache, **out)
    return out


# ----------------------------------------------------------- periodicity pass
def periodicity_windows(series, fps=FPS, win_s=WIN_S, hop_s=HOP_S, band=BAND):
    """Sliding-window FFT periodicity score.
    Returns (centers_s, scores, peak_freqs)."""
    n_win = int(round(win_s * fps))
    n_hop = int(round(hop_s * fps))
    hann = np.hanning(n_win)
    freqs = np.fft.rfftfreq(n_win, d=1.0 / fps)
    band_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    pos_idx = np.where(freqs > 0)[0]

    centers, scores, pfreqs = [], [], []
    # series[i] is flow between frame i and i+1 -> time ~ (i+1)/fps
    for s0 in range(0, len(series) - n_win + 1, n_hop):
        x = series[s0:s0 + n_win].astype(np.float64)
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x * hann)) ** 2
        med = np.median(spec[pos_idx])
        pk = band_idx[np.argmax(spec[band_idx])]
        score = spec[pk] / med if med > 0 else 0.0
        centers.append((s0 + n_win / 2.0 + 1.0) / fps)
        scores.append(float(score))
        pfreqs.append(float(freqs[pk]))
    return np.array(centers), np.array(scores), np.array(pfreqs)


def roc_auc(y, s):
    """Rank-based ROC-AUC (Mann-Whitney)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prf(y, pred):
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                tp=tp, fp=fp, fn=fn)


def main():
    gt = json.load(open(GT_PATH))
    results = {"probe": "vision_motion",
               "config": {"input": "320p @ 5 fps grayscale, Farneback flow, median-vector ego compensation",
                          "window_s": WIN_S, "hop_s": HOP_S, "band_hz": list(BAND),
                          "note": "Nyquist at 5 fps is 2.5 Hz; requested 0.5-3 Hz band truncated to 0.5-2.5 Hz",
                          "score": "peak FFT power in band / median spectral power (DC excluded)"},
               "recordings": {}}

    per_rec = {}
    for rec in RECS:
        m = extract_motion(rec)
        resid = m["resid_mean"]
        centers, scores, pfreqs = periodicity_windows(resid)
        steps = {s["step_id"]: (s["start_time"], s["end_time"])
                 for s in gt[rec]["steps"] if s["start_time"] >= 0}
        st85 = steps[STIR_STEP]
        y = (centers >= st85[0]) & (centers <= st85[1])
        per_rec[rec] = dict(m=m, centers=centers, scores=scores, pfreqs=pfreqs,
                            y=y, steps=steps)

    # ---- tune threshold on 8_16 only (max F1 over candidate thresholds)
    tr = per_rec["8_16"]
    cand = np.unique(tr["scores"])
    best_thr, best_f1 = None, -1.0
    for thr in cand:
        f = prf(tr["y"], tr["scores"] >= thr)["f1"]
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    results["tuning"] = {"tuned_on": "8_16", "criterion": "max F1 on step-85 window labels",
                         "frozen_threshold": round(best_thr, 3)}

    total_wall = 0.0
    for rec in RECS:
        d = per_rec[rec]
        m, centers, scores, pfreqs, y = d["m"], d["centers"], d["scores"], d["pfreqs"], d["y"]
        resid = m["resid_mean"]
        t_frame = (np.arange(len(resid)) + 1.0) / FPS

        # per-GT-step mean residual motion (sanity ordering check)
        step_motion = {}
        for sid, (a, b) in sorted(d["steps"].items()):
            sel = (t_frame >= a) & (t_frame <= b)
            if sel.sum() == 0:
                continue
            step_motion[str(sid)] = {
                "interval_s": [round(a, 1), round(b, 1)],
                "mean_resid_motion": round(float(resid[sel].mean()), 4),
                "mean_concentration_top5pct": round(float(m["resid_conc"][sel].mean()), 4),
                "mean_periodicity_score": round(float(scores[(centers >= a) & (centers <= b)].mean()), 3)
                if np.any((centers >= a) & (centers <= b)) else None,
            }
        covered = np.zeros(len(resid), bool)
        for a, b in d["steps"].values():
            covered |= (t_frame >= a) & (t_frame <= b)
        if (~covered).sum():
            step_motion["outside_any_step"] = {
                "mean_resid_motion": round(float(resid[~covered].mean()), 4),
                "mean_concentration_top5pct": round(float(m["resid_conc"][~covered].mean()), 4),
            }

        # periodicity summary for step 85
        auc_period = roc_auc(y, scores)
        auc_motion_only = roc_auc(y, np.interp(centers, t_frame, resid))
        det = prf(y, scores >= best_thr)

        order = np.argsort(scores)[::-1]
        top5 = [{"t_s": round(float(centers[i]), 1), "score": round(float(scores[i]), 2),
                 "peak_hz": round(float(pfreqs[i]), 2), "in_step85": bool(y[i])}
                for i in order[:5]]
        inside = np.where(y)[0]
        worst_in85 = [{"t_s": round(float(centers[i]), 1), "score": round(float(scores[i]), 2),
                       "peak_hz": round(float(pfreqs[i]), 2)}
                      for i in inside[np.argsort(scores[inside])][:3]] if len(inside) else []

        wall = float(m["wall_s"][0]); total_wall += wall
        results["recordings"][rec] = {
            "role": "tuning (threshold fit here)" if rec == "8_16" else "frozen eval",
            "n_frames_320p5": int(m["n_frames"][0]),
            "n_windows": len(scores),
            "step85_interval_s": [round(d["steps"][STIR_STEP][0], 1), round(d["steps"][STIR_STEP][1], 1)],
            "periodicity_score": {
                "mean_inside_step85": round(float(scores[y].mean()), 3),
                "mean_outside_step85": round(float(scores[~y].mean()), 3),
                "roc_auc": round(auc_period, 3),
                "roc_auc_raw_motion_baseline": round(auc_motion_only, 3),
                "median_peak_hz_inside_step85": round(float(np.median(pfreqs[y])), 2) if y.any() else None,
            },
            "detection_at_frozen_threshold": det,
            "best_windows_top5": top5,
            "worst_windows_inside_step85": worst_in85,
            "per_step_motion": step_motion,
            "cost": {
                "flow_ms_per_frame_mean": round(float(m["flow_ms_mean"][0]), 2),
                "flow_ms_per_frame_p95": round(float(m["flow_ms_p95"][0]), 2),
                "wall_s_motion_pass": round(wall, 1),
                "ms_per_processed_frame_incl_decode": round(1000.0 * wall / int(m["n_frames"][0]), 2),
            },
            "ego_motion_median_vector_mag_mean_px": round(float(m["ego_mag"].mean()), 3),
        }

        # optional plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax[0].plot(t_frame, resid, lw=0.5, color="gray")
            ax[0].set_ylabel("mean residual flow (px)")
            ax[1].plot(centers, scores, lw=1.0, color="C0")
            ax[1].axhline(best_thr, color="r", ls="--", lw=0.8, label=f"frozen thr={best_thr:.1f}")
            ax[1].set_ylabel("periodicity score"); ax[1].set_xlabel("time (s)")
            for a in ax:
                for sid, (s0, s1) in d["steps"].items():
                    a.axvspan(s0, s1, alpha=0.25 if sid == STIR_STEP else 0.06,
                              color="C2" if sid == STIR_STEP else "C1")
            ax[1].legend(loc="upper right")
            fig.suptitle(f"{rec}: residual motion + stirring periodicity (green=GT step 85)")
            fig.tight_layout()
            fig.savefig(os.path.join(PROBE_DIR, f"vision_motion_{rec}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:
            print("plot skipped:", e)

    results["total_motion_pass_wall_s"] = round(total_wall, 1)
    out = os.path.join(PROBE_DIR, "results_vision_motion.json")
    json.dump(results, open(out, "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
