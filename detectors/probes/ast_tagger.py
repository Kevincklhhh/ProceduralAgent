#!/usr/bin/env python3
"""Probe: pretrained AudioSet tagger (AST) as medium-tier audio event detector.

Model: MIT/ast-finetuned-audioset-10-10-0.4593 (ASTFeatureExtractor + ASTForAudioClassification)
- 5s windows, 2.5s hop over 16 kHz mono wavs (AST pads internally to 10.24s).
- Threshold-free ROC-AUC of class scores vs GT step membership.
- Threshold for microwave tuned ONLY on 8_16, frozen, evaluated on the other 5.
- Latency benchmarks (GPU b=1, GPU batched, CPU b=1) for the cost ledger.

Outputs:
  results_ast_tagger.json   (compact machine-readable results)
  scores_ast_tagger.npz     (per-window kitchen-class score timelines, auxiliary)
  plots_ast_tagger.png      (microwave score timelines, nice-to-have)
"""
import json
import os
import time

import numpy as np
import torch
from scipy.io import wavfile
from scipy.stats import rankdata

BASE = "/home/kailaic/NeuroTrace/ProceduralAgent/detectors"
AUDIO_DIR = os.path.join(BASE, "audio")
PROBE_DIR = os.path.join(BASE, "probes")
GT_PATH = os.path.join(BASE, "gt_activity8.json")
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

RECS = ["8_16", "8_26", "8_3", "8_25", "8_31", "8_50"]
TUNE_REC = "8_16"
SR = 16000
WIN = 5.0
HOP = 2.5
BATCH = 64

KITCHEN_SUBSTRINGS = [
    "microwave", "beep", "liquid", "pour", "stir", "cutlery", "dishes",
    "water", "boil", "sizzle", "frying", "blender", "speech",
]


def load_wav(rec):
    sr, x = wavfile.read(os.path.join(AUDIO_DIR, f"{rec}_16k.wav"))
    assert sr == SR, f"{rec}: sr={sr}"
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return x


def make_windows(x):
    n = len(x)
    win_n = int(WIN * SR)
    hop_n = int(HOP * SR)
    starts, wins = [], []
    t = 0
    while t < n:
        seg = x[t:t + win_n]
        if len(seg) < win_n // 2:  # drop tail windows shorter than 2.5s
            break
        if len(seg) < win_n:
            seg = np.pad(seg, (0, win_n - len(seg)))
        starts.append(t / SR)
        wins.append(seg)
        t += hop_n
    return np.array(starts), wins


def roc_auc(pos, neg):
    """Mann-Whitney AUC with tie handling."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def step_intervals(gt, rec, step_ids):
    out = []
    for s in gt[rec]["steps"]:
        if s["step_id"] in step_ids and s["start_time"] >= 0:
            out.append((s["start_time"], s["end_time"]))
    return out


def in_intervals(centers, intervals):
    m = np.zeros(len(centers), dtype=bool)
    for a, b in intervals:
        m |= (centers >= a) & (centers <= b)
    return m


def main():
    gt = json.load(open(GT_PATH))
    device = "cuda:0"

    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    print("loading model...", flush=True)
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID)
    model.eval().to(device)
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]

    # --- kitchen-relevant class index lookup by substring ---
    kitchen_matches = {}
    for sub in KITCHEN_SUBSTRINGS:
        kitchen_matches[sub] = [(i, lab) for i, lab in enumerate(labels)
                                if sub in lab.lower()]
    for sub, m in kitchen_matches.items():
        print(f"  substring '{sub}': {[lab for _, lab in m]}")

    def idx_of(label_name):
        return labels.index(label_name)

    # --- inference over all recordings ---
    all_scores = {}   # rec -> (n_win, 527) float32 sigmoid scores
    all_centers = {}  # rec -> window centers (s)
    wall_times = {}   # rec -> {audio_s, wall_s, n_windows}

    for rec in RECS:
        x = load_wav(rec)
        dur = len(x) / SR
        starts, wins = make_windows(x)
        t0 = time.time()
        scores = []
        with torch.no_grad():
            for b in range(0, len(wins), BATCH):
                batch = wins[b:b + BATCH]
                inp = fe([np.asarray(w) for w in batch], sampling_rate=SR,
                         return_tensors="pt")
                logits = model(inp.input_values.to(device)).logits
                scores.append(torch.sigmoid(logits).cpu().numpy())
        torch.cuda.synchronize()
        wall = time.time() - t0
        scores = np.concatenate(scores).astype(np.float32)
        all_scores[rec] = scores
        all_centers[rec] = starts + WIN / 2.0
        wall_times[rec] = {"audio_s": round(dur, 1), "wall_s": round(wall, 2),
                           "n_windows": len(starts)}
        print(f"{rec}: {dur:.0f}s audio, {len(starts)} windows, "
              f"{wall:.1f}s wall", flush=True)

    # --- save compact score timelines for kitchen classes (auxiliary) ---
    track_labels = sorted({lab for m in kitchen_matches.values() for _, lab in m})
    track_idx = [idx_of(l) for l in track_labels]
    np.savez_compressed(
        os.path.join(PROBE_DIR, "scores_ast_tagger.npz"),
        labels=np.array(track_labels),
        **{f"centers_{r}": all_centers[r] for r in RECS},
        **{f"scores_{r}": all_scores[r][:, track_idx] for r in RECS},
    )

    # --- threshold-free ROC-AUC evaluations ---
    MICROWAVE = "Microwave oven"
    eval_specs = {
        "microwave_vs_steps_89_83": {"label": MICROWAVE, "steps": [89, 83]},
        "microwave_vs_step_89_only": {"label": MICROWAVE, "steps": [89]},
        "pour_vs_step_88": {"label": "Pour", "steps": [88]},
        "liquid_vs_step_88": {"label": "Liquid", "steps": [88]},
        "cutlery_vs_step_85": {"label": "Cutlery, silverware", "steps": [85]},
        "dishes_vs_step_85": {"label": "Dishes, pots, and pans", "steps": [85]},
    }
    auc_results = {}
    for name, spec in eval_specs.items():
        li = idx_of(spec["label"])
        per_rec = {}
        pool_pos, pool_neg = [], []
        for rec in RECS:
            iv = step_intervals(gt, rec, spec["steps"])
            if not iv:
                per_rec[rec] = None  # step skipped in this recording
                continue
            mask = in_intervals(all_centers[rec], iv)
            s = all_scores[rec][:, li]
            per_rec[rec] = (round(roc_auc(s[mask], s[~mask]), 3)
                            if mask.any() and (~mask).any() else None)
            pool_pos.append(s[mask])
            pool_neg.append(s[~mask])
        pooled = roc_auc(np.concatenate(pool_pos), np.concatenate(pool_neg))
        auc_results[name] = {"per_rec": per_rec, "pooled": round(pooled, 3)}
        print(name, auc_results[name])

    # --- top-5 classes per GT step segment for 8_16 and 8_26 ---
    top5 = {}
    for rec in ["8_16", "8_26"]:
        top5[rec] = {}
        for s in gt[rec]["steps"]:
            if s["start_time"] < 0:
                continue
            mask = in_intervals(all_centers[rec],
                                [(s["start_time"], s["end_time"])])
            if not mask.any():
                continue
            mean_s = all_scores[rec][mask].mean(axis=0)
            order = np.argsort(mean_s)[::-1][:5]
            top5[rec][f"step_{s['step_id']}"] = {
                "desc": s["description"][:60],
                "span": [round(s["start_time"], 1), round(s["end_time"], 1)],
                "top5": [[labels[i], round(float(mean_s[i]), 3)] for i in order],
            }

    # --- microwave threshold: tune on 8_16 ONLY, freeze, eval on others ---
    mi = idx_of(MICROWAVE)

    def prf(scores, mask, thr):
        pred = scores >= thr
        tp = int((pred & mask).sum())
        fp = int((pred & ~mask).sum())
        fn = int((~pred & mask).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 3), round(r, 3), round(f1, 3)

    s16 = all_scores[TUNE_REC][:, mi]
    m16 = in_intervals(all_centers[TUNE_REC],
                       step_intervals(gt, TUNE_REC, [89, 83]))
    best_thr, best_f1 = None, -1
    for thr in np.unique(s16):
        _, _, f1 = prf(s16, m16, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    tuned = {"threshold": round(best_thr, 4),
             "tune_rec": TUNE_REC,
             "tune_prf": prf(s16, m16, best_thr)}

    frozen = {}
    pool_s, pool_m = [], []
    for rec in RECS:
        if rec == TUNE_REC:
            continue
        s = all_scores[rec][:, mi]
        m = in_intervals(all_centers[rec], step_intervals(gt, rec, [89, 83]))
        frozen[rec] = prf(s, m, best_thr)
        pool_s.append(s)
        pool_m.append(m)
    frozen["pooled_5rec"] = prf(np.concatenate(pool_s),
                                np.concatenate(pool_m), best_thr)

    # --- microwave event segments with frozen threshold (gap-merge 5s) ---
    def events(centers, above, max_gap=5.1, min_dur=5.0):
        evs = []
        cur = None
        for c, a in zip(centers, above):
            if a:
                if cur is None:
                    cur = [c - WIN / 2, c + WIN / 2]
                elif c - WIN / 2 - cur[1] <= max_gap - WIN:  # window gap
                    cur[1] = c + WIN / 2
                else:
                    evs.append(cur)
                    cur = [c - WIN / 2, c + WIN / 2]
            # handled by gap rule on next active window
        if cur is not None:
            evs.append(cur)
        # merge events whose gap <= max_gap
        merged = []
        for e in evs:
            if merged and e[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = e[1]
            else:
                merged.append(e)
        return [[round(a, 1), round(b, 1), round(b - a, 1)]
                for a, b in merged if b - a >= min_dur]

    mw_events = {}
    for rec in RECS:
        s = all_scores[rec][:, mi]
        mw_events[rec] = {
            "detected": events(all_centers[rec], s >= best_thr),
            "gt_steps": [[sid, *map(lambda v: round(v, 1), iv)]
                         for sid in (89, 83)
                         for iv in step_intervals(gt, rec, [sid])],
        }

    # --- latency benchmarks ---
    bench_win = wavfile.read(os.path.join(AUDIO_DIR, "8_16_16k.wav"))[1]
    bench_win = (bench_win.astype(np.float32) / 32768.0)[80 * SR:85 * SR]

    def time_fe(n=10):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fe(bench_win, sampling_rate=SR, return_tensors="pt")
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1000

    feat = fe(bench_win, sampling_rate=SR, return_tensors="pt").input_values

    def time_gpu(batch_size, n=20):
        x = feat.repeat(batch_size, 1, 1).to(device)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1000

    fe_ms = time_fe()
    gpu_b1 = time_gpu(1)
    gpu_b64 = time_gpu(64, n=5)

    model_cpu = model.to("cpu")
    with torch.no_grad():
        model_cpu(feat)  # warmup
        t0 = time.perf_counter()
        model_cpu(feat)
        cpu_b1 = (time.perf_counter() - t0) * 1000
    model.to(device)

    n_win_450 = int((450 - WIN) / HOP) + 1
    latency = {
        "feature_extraction_ms_per_window_cpu": round(fe_ms, 1),
        "gpu_forward_ms_batch1": round(gpu_b1, 1),
        "gpu_forward_ms_per_window_batch64": round(gpu_b64 / 64, 2),
        "cpu_forward_ms_batch1": round(cpu_b1, 1),
        "windows_per_450s_recording": n_win_450,
        "est_total_s_450s_gpu_batched": round(
            n_win_450 * (fe_ms + gpu_b64 / 64) / 1000, 1),
        "est_total_s_450s_cpu": round(n_win_450 * (fe_ms + cpu_b1) / 1000, 1),
        "measured_wall_s_per_recording_gpu_batched": wall_times,
        "reference": {"gemini_vlm_call_s": 2, "local_qwen_vlm_call_s": 45},
    }

    results = {
        "probe": "ast_tagger",
        "model": MODEL_ID,
        "config": {"sr": SR, "window_s": WIN, "hop_s": HOP, "batch": BATCH,
                   "window_label_rule": "window center inside GT step interval",
                   "note_ast_padding": "AST pads 5s input to 10.24s internally"},
        "kitchen_class_matches": {k: [lab for _, lab in v]
                                  for k, v in kitchen_matches.items()},
        "roc_auc": auc_results,
        "top5_classes_per_step": top5,
        "microwave_threshold": {"tuned_on_8_16": tuned,
                                "frozen_eval_prf_other5": frozen},
        "microwave_events_frozen_threshold": mw_events,
        "latency": latency,
    }
    out_path = os.path.join(PROBE_DIR, "results_ast_tagger.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)

    # --- optional plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(RECS), 1, figsize=(12, 14), sharex=False)
        for ax, rec in zip(axes, RECS):
            ax.plot(all_centers[rec], all_scores[rec][:, mi], lw=1,
                    label="Microwave oven score")
            for a, b in step_intervals(gt, rec, [89]):
                ax.axvspan(a, b, color="green", alpha=0.2)
            for a, b in step_intervals(gt, rec, [83]):
                ax.axvspan(a, b, color="orange", alpha=0.2)
            ax.axhline(best_thr, color="red", ls="--", lw=0.8)
            ax.set_title(f"{rec} (green=step89, orange=step83, "
                         f"red=frozen thr {best_thr:.3f})", fontsize=9)
            ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(PROBE_DIR, "plots_ast_tagger.png"), dpi=110)
        print("wrote plot")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
