#!/usr/bin/env python3
"""Streaming rolling-prefill prototype on 8_26 step 90.

Instead of batch-prefilling all evidence frames at trigger time (old arm D), feed each
frame into the KV cache AS IT ARRIVES (ViT-encode that one frame + incremental prefill into
the growing cache). By trigger time the cache already holds the procedure + every frame seen,
so the trigger path is just: append the check question + decode one token.

Tests EXACTLY three things, per evidence window (5/10/20 frames @1fps ending 206.2s):
  (1) DECISION-EQUIVALENCE: streamed cache vs one-shot batch prefill -> same verdict?
      (logit diff vs FP8 noise floor; we already know FP8 makes them not bit-exact.)
  (2) REAL-TIME per-frame cost: ViT+prefill for one frame << inter-frame budget
      (1fps=1000ms; also report vs 2fps=500ms, 4fps=250ms).
  (3) TRIGGER FLOOR: trigger->answer latency after streaming N frames, ~133ms regardless of N.

Reuses validated stage primitives from run_preencode_arms.M and the constrained 3-way readout
from followup_constrained.

Run:  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python stream_rolling_prefill.py
"""
import os, sys, json, time, statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R
import followup_constrained as F   # build_class_ids, restricted_answer

OUTDIR = os.path.dirname(os.path.abspath(__file__))
WINDOWS = R.WINDOWS


def sync():
    torch.cuda.synchronize()


def summ(xs):
    xs = sorted(xs)
    return dict(median=statistics.median(xs), p90=xs[min(len(xs) - 1, int(0.9 * len(xs)))],
                p95=xs[min(len(xs) - 1, int(0.95 * len(xs)))], min=xs[0], max=xs[-1], n=len(xs))


def per_frame_pixels(m, P):
    """Split the stacked pixel_values back into one tensor per frame (raw patches)."""
    counts = P["grid"].prod(-1).tolist()
    return list(torch.split(P["pixel_values"], counts, dim=0))


def stream_chunks(m, P, vstart_id):
    """Tile the prefix [0:k] into (preamble) + one chunk per frame, by vision_start/_end."""
    ids = P["input_ids"]
    vs = (ids == vstart_id).nonzero(as_tuple=True)[0].tolist()
    ve = (ids == m.vision_end_id).nonzero(as_tuple=True)[0].tolist()
    chunks = [(0, vs[0], None)]                       # preamble: system+user-open+procedure text
    for i in range(len(vs)):
        chunks.append((vs[i], ve[i] + 1, i))          # frame i: <vision_start>..pads..<vision_end>
    assert chunks[-1][1] == P["k"], (chunks[-1][1], P["k"])
    return chunks


def stream_prefill(m, P, pix_splits, chunks, time_it=True):
    """Roll frames into the cache one chunk at a time. Returns (past, per_frame_ms)."""
    past = None
    per_frame_ms = []
    for (a, b, fi) in chunks:
        ids_chunk = P["input_ids"][a:b]
        if time_it:
            sync(); t0 = time.perf_counter()
        emb = m.embed(ids_chunk[None])
        if fi is not None:                            # ViT-encode THIS frame, scatter into pads
            ie = m.stage_V(pix_splits[fi], P["grid"][fi:fi + 1])
            mask = (ids_chunk[None] == m.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, ie.to(emb.dtype))
        cp = torch.arange(a, b, device=m.dev)
        out = m.lm(inputs_embeds=emb, position_ids=P["pos"][:, :, a:b],
                   past_key_values=past, use_cache=True, cache_position=cp)
        past = out.past_key_values
        if time_it:
            sync(); dt = time.perf_counter() - t0
            if fi is not None:
                per_frame_ms.append(dt * 1e3)
    return past, per_frame_ms


def trigger_logits(m, full_embeds, P, past):
    """At trigger: append the check-question suffix to `past`, decode -> final logits."""
    k, L = P["k"], P["L"]
    past.crop(k)
    return m.stage_SF(full_embeds, P["pos"], k, L, past)[0]


def main():
    torch.set_grad_enabled(False)
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    t = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        R.CKPT, device_map={"": 0}, attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(R.CKPT)
    proc.image_processor.max_pixels = 128 * 28 * 28
    if isinstance(getattr(proc.image_processor, "size", None), dict):
        proc.image_processor.size["longest_edge"] = 128 * 28 * 28
    print(f"[load] {time.perf_counter()-t:.1f}s", flush=True)
    m = R.M(model, proc)
    vstart_id = model.config.vision_start_token_id
    class_ids = F.build_class_ids(m.tok)

    WARM, MEAS = 8, 30
    out = {"recording": "8_26", "step_id": 90, "model": "Qwen3.5-27B-FP8",
           "note": "streaming rolling-prefill: one frame -> cache at a time",
           "windows": {}}

    for wname, win in WINDOWS.items():
        frames = R.decode_frames(*win)
        P = m.stage_P(frames)
        pix_splits = per_frame_pixels(m, P)
        chunks = stream_chunks(m, P, vstart_id)

        k, L = P["k"], P["L"]
        all_vit = m.stage_V(P["pixel_values"], P["grid"])
        full_embeds = m.stage_MG(P["input_ids"], all_vit)

        # helper: fresh batch prefix cache (used once) -> trigger logits. No cache reuse, so
        # no crop-aliasing; this is the clean path for comparing logit VALUES.
        def batch_logits_fresh():
            pb = m.stage_PF(full_embeds, P["pos"], k)
            return m.stage_SF(full_embeds, P["pos"], k, L, pb)[0]

        # ---- warm up streaming kernels (cold = ~10s triton/FLA autotune) ----
        for _ in range(2):
            pb, _ = stream_prefill(m, P, pix_splits, chunks, time_it=False)
            del pb

        # ---- (1) decision-equivalence: streamed cache vs batch ----
        lg_batch = batch_logits_fresh()
        # FP8 run-to-run noise floor: TWO independent fresh batch builds, each used once.
        nf = float((batch_logits_fresh() - batch_logits_fresh()).abs().max())
        past_stream, _ = stream_prefill(m, P, pix_splits, chunks, time_it=False)
        lg_stream = m.stage_SF(full_embeds, P["pos"], k, L, past_stream)[0]
        free_b, _ = R.argmax_token(m, lg_batch); free_s, _ = R.argmax_token(m, lg_stream)
        ans_b, sc_b = F.restricted_answer(lg_batch, class_ids)
        ans_s, sc_s = F.restricted_answer(lg_stream, class_ids)
        stream_vs_batch = float((lg_batch - lg_stream).abs().max())

        # ---- (2) per-frame streaming cost (real-time check) ----
        pf_ms = []
        for _ in range(5):                            # repeat to get a stable per-frame sample
            _, pf = stream_prefill(m, P, pix_splits, chunks, time_it=True)
            pf_ms += pf
        pf_stat = summ(pf_ms)

        # ---- (3) trigger floor after streaming N frames ----
        for _ in range(WARM):
            trigger_logits(m, full_embeds, P, past_stream)
        trig = []
        for _ in range(MEAS):
            sync(); t0 = time.perf_counter()
            _ = trigger_logits(m, full_embeds, P, past_stream)
            sync(); trig.append((time.perf_counter() - t0) * 1e3)
        trig_stat = summ(trig)

        out["windows"][wname] = {
            "win": win, "L": P["L"], "k": P["k"], "n_frames": win[2],
            "decision_equiv": {
                "free_argmax_batch": free_b, "free_argmax_stream": free_s,
                "answer_batch": ans_b, "answer_stream": ans_s,
                "answers_match": ans_b == ans_s,
                "stream_vs_batch_max_abs_logit_diff": stream_vs_batch,
                "fp8_noise_floor": nf,
                "scores_batch": sc_b, "scores_stream": sc_s},
            "per_frame_stream_ms": pf_stat,
            "trigger_floor_ms": trig_stat,
        }
        de = out["windows"][wname]["decision_equiv"]
        print(f"\n[{wname}] N={win[2]} L={P['L']} k={P['k']}", flush=True)
        print(f"  (1) equiv: batch->{ans_b} stream->{ans_s} match={de['answers_match']} "
              f"| stream-vs-batch max|Δlogit|={stream_vs_batch:.3f} (FP8 noise floor={nf:.3f})", flush=True)
        print(f"  (2) per-frame ViT+prefill: median={pf_stat['median']:.1f}ms p90={pf_stat['p90']:.1f} "
              f"max={pf_stat['max']:.1f}  -> 1fps budget 1000ms, 4fps budget 250ms", flush=True)
        print(f"  (3) trigger floor: median={trig_stat['median']:.1f}ms p90={trig_stat['p90']:.1f} "
              f"p95={trig_stat['p95']:.1f}", flush=True)

    json.dump(out, open(os.path.join(OUTDIR, "stream_rolling_prefill.json"), "w"), indent=2)
    print("\n[done] wrote stream_rolling_prefill.json", flush=True)
    print("\n==== SUMMARY ====", flush=True)
    print(f"{'window':8} {'N':>3} {'equiv':>7} {'Δlogit':>8} {'perframe_ms':>12} {'trigger_ms':>11}", flush=True)
    for w in WINDOWS:
        W = out["windows"][w]; de = W["decision_equiv"]
        print(f"{w:8} {W['n_frames']:>3} {str(de['answers_match']):>7} "
              f"{de['stream_vs_batch_max_abs_logit_diff']:>8.2f} "
              f"{W['per_frame_stream_ms']['median']:>12.1f} {W['trigger_floor_ms']['median']:>11.1f}", flush=True)


if __name__ == "__main__":
    main()
