#!/usr/bin/env python3
"""
Variant exploration for the vision_motion probe (uses cached motion npz).
All variant SELECTION happens on 8_16 only; the chosen variant + its threshold
are frozen and evaluated on 8_26. Appends a 'variant_exploration' section to
results_vision_motion.json.
"""
import json
import os

import numpy as np

from probe_vision_motion import (PROBE_DIR, GT_PATH, FPS, WIN_S, HOP_S,
                                 STIR_STEP, extract_motion, roc_auc, prf)


def window_scores(series, band, fps=FPS):
    n_win = int(round(WIN_S * fps))
    n_hop = int(round(HOP_S * fps))
    hann = np.hanning(n_win)
    freqs = np.fft.rfftfreq(n_win, d=1.0 / fps)
    band_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    pos_idx = np.where(freqs > 0)[0]
    centers, scores = [], []
    for s0 in range(0, len(series) - n_win + 1, n_hop):
        x = series[s0:s0 + n_win].astype(np.float64)
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x * hann)) ** 2
        med = np.median(spec[pos_idx])
        pk = spec[band_idx].max()
        scores.append(float(pk / med) if med > 0 else 0.0)
        centers.append((s0 + n_win / 2.0 + 1.0) / fps)
    return np.array(centers), np.array(scores)


def window_mean(series, fps=FPS):
    n_win = int(round(WIN_S * fps))
    n_hop = int(round(HOP_S * fps))
    centers, vals = [], []
    for s0 in range(0, len(series) - n_win + 1, n_hop):
        vals.append(float(series[s0:s0 + n_win].mean()))
        centers.append((s0 + n_win / 2.0 + 1.0) / fps)
    return np.array(centers), np.array(vals)


def main():
    gt = json.load(open(GT_PATH))
    data = {}
    for rec in ["8_16", "8_26"]:
        m = extract_motion(rec)  # cached
        st85 = next((s["start_time"], s["end_time"])
                    for s in gt[rec]["steps"] if s["step_id"] == STIR_STEP)
        data[rec] = (m, st85)

    variants = {
        "period_0.5-2.5Hz_resid": lambda m: window_scores(m["resid_mean"], (0.5, 2.5)),
        "period_1.0-2.5Hz_resid": lambda m: window_scores(m["resid_mean"], (1.0, 2.5)),
        "period_0.5-2.5Hz_conc": lambda m: window_scores(m["resid_conc"], (0.5, 2.5)),
        "period_1.0-2.5Hz_conc": lambda m: window_scores(m["resid_conc"], (1.0, 2.5)),
        "mean_concentration": lambda m: window_mean(m["resid_conc"]),
        "conc_over_egomag": lambda m: window_mean(m["resid_conc"] / (1.0 + m["ego_mag"])),
    }

    out = {"selection_on": "8_16", "auc": {}, "frozen_eval_8_26": {}}
    aucs_816 = {}
    cached = {}
    for name, fn in variants.items():
        row = {}
        for rec in ["8_16", "8_26"]:
            m, st85 = data[rec]
            c, s = fn(m)
            y = (c >= st85[0]) & (c <= st85[1])
            cached[(name, rec)] = (c, s, y)
            row[rec] = round(roc_auc(y, s), 3)
        aucs_816[name] = row["8_16"]
        out["auc"][name] = {"8_16_tuning": row["8_16"], "8_26_frozen_eval": row["8_26"]}

    best = max(aucs_816, key=aucs_816.get)
    out["selected_variant"] = best

    # tune threshold for the best variant on 8_16, freeze, eval on 8_26
    c, s, y = cached[(best, "8_16")]
    best_thr, best_f1 = None, -1.0
    for thr in np.unique(s):
        f = prf(y, s >= thr)["f1"]
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    out["frozen_threshold"] = round(best_thr, 4)
    out["tuning_8_16"] = prf(y, s >= best_thr)
    out["tuning_8_16"]["roc_auc"] = aucs_816[best]
    c2, s2, y2 = cached[(best, "8_26")]
    out["frozen_eval_8_26"] = prf(y2, s2 >= best_thr)
    out["frozen_eval_8_26"]["roc_auc"] = out["auc"][best]["8_26_frozen_eval"]
    out["frozen_eval_8_26"]["mean_inside_step85"] = round(float(s2[y2].mean()), 4)
    out["frozen_eval_8_26"]["mean_outside_step85"] = round(float(s2[~y2].mean()), 4)

    res_path = os.path.join(PROBE_DIR, "results_vision_motion.json")
    results = json.load(open(res_path))
    results["variant_exploration"] = out
    json.dump(results, open(res_path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
