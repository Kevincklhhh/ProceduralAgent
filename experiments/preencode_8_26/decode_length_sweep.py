#!/usr/bin/env python3
"""Latency as a function of OUTPUT token length, input held fixed.

Not the open-vs-bounded framing -- a clean sweep N = output tokens vs trigger latency.
Holds the evidence/prompt fixed (8_26 step 90 windows), prefills ONCE, then runs a single
greedy decode loop out to max(N), timestamping at checkpoints -> the whole curve in one pass.

Reports, per window (= fixed input size):
  prefill_ms                 (input-side cost, produces the 1st-token logits)
  decode_to_N_ms[N]          (decode-loop time to have emitted N tokens, EOS ignored)
  total_trigger_ms[N]        = prefill_ms + decode_to_N_ms[N]
  per-token marginal ms      (slope between checkpoints -> shows KV-growth nonlinearity)

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ~/miniconda3/envs/qwen36/bin/python decode_length_sweep.py
"""
import os, sys, json, time, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = [1, 32, 96, 256, 512, 768, 1024, 1280, 1536, 2048]
MAXN = max(CHECKPOINTS)
TRIALS = 3                      # 3 windows x 3 trials = 9 measured decode passes
WARMUP_TOKENS = 16             # cheap kernel-compile pass, not a measured run


def sync(): torch.cuda.synchronize()


def prefill_full(m, embeds, pos, L):
    """Full-sequence prefill -> (last-token logits, past_key_values)."""
    cp = torch.arange(L, device=m.dev)
    out = m.lm(inputs_embeds=embeds, position_ids=pos, past_key_values=None,
               use_cache=True, cache_position=cp)
    logits = m.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
    return logits, out.past_key_values


def decode_curve(m, embeds, pos, L, maxn=MAXN, cps=frozenset(CHECKPOINTS)):
    """One greedy decode pass to maxn tokens. Returns (prefill_ms, {N: decode_to_N_ms}).
    decode_to_N_ms[N] = wall time from end-of-prefill to having emitted N tokens."""
    sync(); t = time.perf_counter()
    logits, past = prefill_full(m, embeds, pos, L)
    sync(); prefill_ms = (time.perf_counter() - t) * 1e3

    next_pos = pos[:, :, -1:].clone()           # 3D mRoPE; text decode -> +1 each step
    marks = {}
    sync(); t0 = time.perf_counter()
    for i in range(1, maxn + 1):
        tid = int(logits.argmax(-1).item())     # the i-th emitted token
        if i in cps:
            sync(); marks[i] = (time.perf_counter() - t0) * 1e3
        if i == maxn:
            break
        next_pos = next_pos + 1
        emb = m.embed(torch.tensor([[tid]], device=m.dev))
        cp = torch.tensor([L + i - 1], device=m.dev)
        out = m.lm(inputs_embeds=emb, position_ids=next_pos, past_key_values=past,
                   use_cache=True, cache_position=cp)
        past = out.past_key_values
        logits = m.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :]
    return prefill_ms, marks


def main():
    torch.set_grad_enabled(False)
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    R.log("[load] ckpt:", R.CKPT)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        R.CKPT, device_map={"": 0}, attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(R.CKPT)
    mp = 128 * 28 * 28
    proc.image_processor.max_pixels = mp
    if hasattr(proc.image_processor, "size") and isinstance(proc.image_processor.size, dict):
        proc.image_processor.size["longest_edge"] = mp
    m = R.M(model, proc)

    out = {"model": "Qwen3.5-27B-FP8", "checkpoints": CHECKPOINTS, "trials": TRIALS,
           "windows": {}}
    for wname in ["short", "medium", "long"]:
        win = R.WINDOWS[wname]
        frames = R.decode_frames(*win)
        P = m.stage_P(frames)
        V = m.stage_V(P["pixel_values"], P["grid"])
        MG = m.stage_MG(P["input_ids"], V)
        L, pos = P["L"], P["pos"]
        R.log(f"\n===== {wname} {win}  L={L} tok =====")
        decode_curve(m, MG, pos, L, maxn=WARMUP_TOKENS, cps=frozenset([WARMUP_TOKENS]))
        pf, curves = [], {N: [] for N in CHECKPOINTS}
        for _ in range(TRIALS):
            p, marks = decode_curve(m, MG, pos, L)
            pf.append(p)
            for N in CHECKPOINTS:
                curves[N].append(marks[N])
        pf_med = statistics.median(pf)
        rec = {"L": L, "win": win, "prefill_ms": round(pf_med, 1), "by_N": {}}
        prev_t, prev_n = 0.0, 0
        for N in CHECKPOINTS:
            d = statistics.median(curves[N])
            marg = (d - prev_t) / (N - prev_n) if N > prev_n else 0.0
            rec["by_N"][N] = {"decode_ms": round(d, 1),
                              "total_trigger_ms": round(pf_med + d, 1),
                              "marginal_ms_per_tok": round(marg, 2)}
            R.log(f"  N={N:4d}  decode={d:8.1f}ms  total(prefill+decode)={pf_med+d:8.1f}ms  "
                  f"marginal={marg:6.2f} ms/tok")
            prev_t, prev_n = d, N
        out["windows"][wname] = rec

    with open(os.path.join(OUTDIR, "decode_length_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    R.log("\n[done] wrote decode_length_sweep.json")


if __name__ == "__main__":
    main()
