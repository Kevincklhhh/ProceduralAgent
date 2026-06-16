#!/usr/bin/env python3
"""CLAP zero-shot audio-text matching probe ("promptable audio detector").

Model: laion/clap-htsat-unfused (48 kHz input, repeat-pad to 10 s internally).
Windows: 5 s, hop 2.5 s. Softmax over 7 text prompts per window.

Tuning protocol: thresholds tuned ONLY on 8_16 (clean run), frozen, then
evaluated on the other five recordings. AUCs reported per recording + pooled.
"""
import json
import os
import time

import numpy as np
import torch
from scipy.io import wavfile
from scipy.stats import rankdata
from transformers import ClapModel, ClapProcessor

AUDIO_DIR = "/home/kailaic/NeuroTrace/ProceduralAgent/data/audio"
GT_PATH = "/home/kailaic/NeuroTrace/ProceduralAgent/data/gt_activity8.json"
OUT_DIR = "/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes"
MODEL_ID = "laion/clap-htsat-unfused"

RECS = ["8_16", "8_26", "8_3", "8_25", "8_31", "8_50"]
TUNE_REC = "8_16"
EVAL_RECS = [r for r in RECS if r != TUNE_REC]

WIN_S = 5.0
HOP_S = 2.5
SR = 48000
BATCH = 64

PROMPTS = [
    "a microwave oven running",
    "a microwave beep",
    "pouring liquid into a cup",
    "stirring a drink with a spoon, glass clinking",
    "opening a wrapper or container",
    "a person speaking",
    "a quiet kitchen, refrigerator hum",
]

# event -> (prompt index, set of positive GT step_ids)
EVENTS = {
    "microwave_running": (0, {89, 83}),
    "pouring": (2, {88}),
    "stirring": (3, {85}),
}


def load_wav(rec):
    sr, x = wavfile.read(os.path.join(AUDIO_DIR, f"{rec}_48k.wav"))
    assert sr == SR, f"{rec}: expected {SR} Hz, got {sr}"
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
    win = int(WIN_S * SR)
    hop = int(HOP_S * SR)
    starts, chunks = [], []
    t = 0
    while t + win <= n or (t == 0 and n > 0):
        chunk = x[t:t + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        starts.append(t / SR)
        chunks.append(chunk)
        t += hop
    # cover the tail (last partial window) if remainder > half a window
    if starts and (n - (starts[-1] * SR + win)) > hop * SR:
        pass
    return np.array(starts), chunks


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def prf(pred, labels):
    pred = np.asarray(pred).astype(bool)
    labels = np.asarray(labels).astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def labels_for(centers, steps, pos_ids):
    lab = np.zeros(len(centers), dtype=int)
    for s in steps:
        if s["step_id"] in pos_ids and s["start_time"] >= 0:
            lab |= ((centers >= s["start_time"]) & (centers <= s["end_time"])).astype(int)
    return lab


def main():
    gt = json.load(open(GT_PATH))
    device = "cuda:0"
    t0 = time.time()
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s")

    # text embeddings (computed once)
    with torch.no_grad():
        ti = processor(text=PROMPTS, return_tensors="pt", padding=True)
        ti = {k: v.to(device) for k, v in ti.items()}
        text_emb = model.get_text_features(**ti)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale_a.exp().item()
    print(f"logit_scale_a.exp() = {logit_scale:.2f}")

    per_rec = {}
    lat_gpu_ms_per_window = []
    for rec in RECS:
        x = load_wav(rec)
        dur = len(x) / SR
        starts, chunks = make_windows(x)
        centers = starts + WIN_S / 2.0

        t_fe0 = time.time()
        feats = []
        for i in range(0, len(chunks), BATCH):
            inp = processor(audios=chunks[i:i + BATCH], sampling_rate=SR,
                            return_tensors="pt")
            feats.append(inp["input_features"])
        feats = torch.cat(feats, dim=0)
        t_fe = time.time() - t_fe0

        # warmup once on first recording
        if rec == RECS[0]:
            with torch.no_grad():
                _ = model.get_audio_features(input_features=feats[:4].to(device))
            torch.cuda.synchronize()

        t_g0 = time.time()
        embs = []
        with torch.no_grad():
            for i in range(0, len(feats), BATCH):
                e = model.get_audio_features(
                    input_features=feats[i:i + BATCH].to(device))
                embs.append(e)
            torch.cuda.synchronize()
        t_gpu = time.time() - t_g0
        audio_emb = torch.cat(embs, dim=0)
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)

        sims = (audio_emb @ text_emb.T).cpu().numpy()          # cosine sims
        probs = torch.softmax(torch.tensor(sims) * logit_scale, dim=-1).numpy()

        per_rec[rec] = {
            "duration_s": dur, "n_windows": len(starts), "centers": centers,
            "sims": sims, "probs": probs, "steps": gt[rec]["steps"],
            "t_feature_extract_s": t_fe, "t_gpu_infer_s": t_gpu,
        }
        lat_gpu_ms_per_window.append(t_gpu / len(starts) * 1000)
        print(f"{rec}: {dur:.0f}s, {len(starts)} windows, "
              f"fe {t_fe:.1f}s, gpu {t_gpu:.2f}s")

    # ---------------- AUC evaluation (threshold-free) ----------------
    auc_results = {}
    for ev, (pidx, pos_ids) in EVENTS.items():
        ev_res = {"per_recording": {}, "prompt": PROMPTS[pidx]}
        pooled_s_tune, pooled_l_tune = [], []
        pooled_s_eval, pooled_l_eval = [], []
        for rec in RECS:
            d = per_rec[rec]
            lab = labels_for(d["centers"], d["steps"], pos_ids)
            sc = d["sims"][:, pidx]
            auc = roc_auc(sc, lab)
            ev_res["per_recording"][rec] = {
                "auc": None if auc is None else round(auc, 3),
                "n_pos": int(lab.sum()), "n_neg": int(len(lab) - lab.sum()),
            }
            if rec == TUNE_REC:
                pooled_s_tune.append(sc); pooled_l_tune.append(lab)
            else:
                pooled_s_eval.append(sc); pooled_l_eval.append(lab)
        a_t = roc_auc(np.concatenate(pooled_s_tune), np.concatenate(pooled_l_tune))
        a_e = roc_auc(np.concatenate(pooled_s_eval), np.concatenate(pooled_l_eval))
        a_all = roc_auc(
            np.concatenate(pooled_s_tune + pooled_s_eval),
            np.concatenate(pooled_l_tune + pooled_l_eval))
        ev_res["auc_tune_8_16"] = None if a_t is None else round(a_t, 3)
        ev_res["auc_eval_pooled_5recs"] = None if a_e is None else round(a_e, 3)
        ev_res["auc_pooled_all6"] = None if a_all is None else round(a_all, 3)
        auc_results[ev] = ev_res

    # ---------------- threshold tuning on 8_16 only ----------------
    thresh_results = {}
    for ev, (pidx, pos_ids) in EVENTS.items():
        d = per_rec[TUNE_REC]
        lab = labels_for(d["centers"], d["steps"], pos_ids)
        sc = d["probs"][:, pidx]
        best = {"thresh": None, "f1": -1.0}
        for th in np.unique(np.round(sc, 4)):
            m = prf(sc >= th, lab)
            if m["f1"] > best["f1"]:
                best = {"thresh": float(th), "f1": m["f1"], "metrics": m}
        th = best["thresh"]
        ev_th = {"prompt": PROMPTS[pidx], "threshold_softmax_prob": round(th, 4),
                 "tuned_on_8_16": best["metrics"], "frozen_eval": {}}
        pooled_p, pooled_l = [], []
        for rec in EVAL_RECS:
            d = per_rec[rec]
            lab = labels_for(d["centers"], d["steps"], pos_ids)
            sc = d["probs"][:, pidx]
            ev_th["frozen_eval"][rec] = prf(sc >= th, lab)
            pooled_p.append(sc >= th); pooled_l.append(lab)
        ev_th["frozen_eval"]["pooled_5recs"] = prf(
            np.concatenate(pooled_p), np.concatenate(pooled_l))
        thresh_results[ev] = ev_th

    # ---------------- top prompt per GT segment (8_16, 8_26) ----------------
    top_prompt = {}
    for rec in ["8_16", "8_26"]:
        d = per_rec[rec]
        rows = []
        for s in d["steps"]:
            if s["start_time"] < 0:
                continue
            m = (d["centers"] >= s["start_time"]) & (d["centers"] <= s["end_time"])
            if m.sum() == 0:
                continue
            mean_probs = d["probs"][m].mean(axis=0)
            order = np.argsort(mean_probs)[::-1]
            rows.append({
                "step_id": s["step_id"],
                "t": f"{s['start_time']:.0f}-{s['end_time']:.0f}s",
                "desc": s["description"].split("-", 1)[-1][:50],
                "top_prompt": PROMPTS[order[0]],
                "top_prob": round(float(mean_probs[order[0]]), 3),
                "second_prompt": PROMPTS[order[1]],
                "second_prob": round(float(mean_probs[order[1]]), 3),
            })
        top_prompt[rec] = rows

    # ---------------- latency: CPU single window ----------------
    model_cpu = model.to("cpu").eval()
    one = per_rec[TUNE_REC]
    chunk = load_wav(TUNE_REC)[: int(WIN_S * SR)]
    cpu_fe_ms, cpu_fwd_ms = [], []
    for _ in range(4):
        t0 = time.time()
        inp = processor(audios=[chunk], sampling_rate=SR, return_tensors="pt")
        cpu_fe_ms.append((time.time() - t0) * 1000)
        t0 = time.time()
        with torch.no_grad():
            _ = model_cpu.get_audio_features(input_features=inp["input_features"])
        cpu_fwd_ms.append((time.time() - t0) * 1000)
    cpu_fe_ms, cpu_fwd_ms = sorted(cpu_fe_ms), sorted(cpu_fwd_ms)

    latency = {
        "gpu_batched_ms_per_window_model_forward": round(
            float(np.mean(lat_gpu_ms_per_window)), 2),
        "cpu_single_window_feature_extract_ms": round(cpu_fe_ms[len(cpu_fe_ms)//2], 1),
        "cpu_single_window_model_forward_ms": round(cpu_fwd_ms[len(cpu_fwd_ms)//2], 1),
        "per_recording_total_s": {
            rec: {
                "feature_extraction_s": round(per_rec[rec]["t_feature_extract_s"], 2),
                "gpu_inference_s": round(per_rec[rec]["t_gpu_infer_s"], 2),
                "total_s": round(per_rec[rec]["t_feature_extract_s"]
                                 + per_rec[rec]["t_gpu_infer_s"], 2),
                "audio_duration_s": round(per_rec[rec]["duration_s"], 1),
                "n_windows": per_rec[rec]["n_windows"],
            } for rec in RECS
        },
        "model_load_s": round(load_s, 1),
    }

    results = {
        "probe": "clap_zeroshot",
        "model": MODEL_ID,
        "window_s": WIN_S, "hop_s": HOP_S, "sample_rate": SR,
        "prompts": PROMPTS,
        "label_rule": "window positive if its center falls inside a GT step segment",
        "events": {ev: {"prompt_idx": pidx, "positive_steps": sorted(ids)}
                   for ev, (pidx, ids) in EVENTS.items()},
        "auc": auc_results,
        "thresholded_detection": thresh_results,
        "top_prompt_per_gt_segment": top_prompt,
        "latency": latency,
    }
    out_path = os.path.join(OUT_DIR, "results_clap_zeroshot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", out_path)

    # ---------------- optional plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"microwave_running": "tab:red", "pouring": "tab:blue",
                  "stirring": "tab:green"}
        fig, axes = plt.subplots(len(RECS), 1, figsize=(14, 2.2 * len(RECS)),
                                 sharex=False)
        for ax, rec in zip(axes, RECS):
            d = per_rec[rec]
            for ev, (pidx, pos_ids) in EVENTS.items():
                ax.plot(d["centers"], d["probs"][:, pidx], color=colors[ev],
                        lw=1, label=ev)
                for s in d["steps"]:
                    if s["step_id"] in pos_ids and s["start_time"] >= 0:
                        ax.axvspan(s["start_time"], s["end_time"],
                                   color=colors[ev], alpha=0.12)
            ax.set_title(rec, fontsize=9)
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=7, ncol=3)
        fig.tight_layout()
        plot_path = os.path.join(OUT_DIR, "clap_zeroshot_timelines.png")
        fig.savefig(plot_path, dpi=110)
        print("wrote", plot_path)
    except Exception as e:
        print("plotting skipped:", e)


if __name__ == "__main__":
    main()
