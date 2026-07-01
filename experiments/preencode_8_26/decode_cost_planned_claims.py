#!/usr/bin/env python3
"""Planned claims reduce DECODE: open-ended reminder generation vs bounded claim verification.

Claim under test: anticipating the error as pre-compiled yes/no CLAIMS (from the Stage-1 criteria)
replaces ONE long open-ended generation per step with k single-token verifications (k = #claims
for that step). Since decode is sequential and dominates trigger latency (and the evidence prefill
is shared/pre-encodable across both framings), cutting decoded tokens cuts trigger latency.

Setup mirrors logs/2026-06-24_t2-binary-vs-targeted.md (oracle step windows, criteria claims) but
measures COST (decoded tokens + decode latency), not detection accuracy. Local Qwen3.5-27B-FP8.

Per (recording, step):
  evidence = frames over the oracle step span (shared, pre-encoded ONCE)
  OPEN     : append "describe any mistake" -> greedy-decode until <|im_end|> (cap 96)  -> N_open tokens
  BOUNDED  : for each anticipated claim -> append "Check: <claim>. yes/no?" -> decode 1 token
Report decoded-token counts and decode latency, open vs bounded (summed over a step's claims).

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python decode_cost_planned_claims.py
"""
import os, sys, json, time, statistics
import numpy as np
import torch
import decord
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R

BASE = "/home/kailaic/NeuroTrace/ProceduralAgent"
STEP_ANN = f"{BASE}/data/cc4d/annotations/annotation_json/step_annotations.json"
CRITERIA = f"{BASE}/tasks/cc4d_probe/spicedhotchocolate.generated.criteria.json"
VIDEO_DIR = f"{BASE}/data/videos_360p"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

RECORDINGS = ["8_26", "8_3", "8_11", "8_15", "8_19", "8_30"]
MAX_FRAMES = 16
MAX_NEW = 96

SYS = "You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."
OPEN_Q = ("Watch the frames. If the user made any mistake during this step, describe the corrective "
          "reminder you would give. If everything looks correct, reply 'No reminder needed.'")
def claim_q(claim):
    return f"Check: {claim} Did this specific mistake occur in the frames? Answer yes or no."


def sample_frames(video, start, end, n):
    vr = decord.VideoReader(video, num_threads=2)
    fps = vr.get_avg_fps()
    ts = np.linspace(start, max(start, end - 0.1), n)
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in ts]
    b = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(b[i]) for i in range(b.shape[0])]


def build_ids(m, step_text, frames, question):
    """Return input_ids, pos(3,1,L), pixel_values, grid, k_ev (split after last vision_end)."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content":
            [{"type": "text", "text": f"Step: {step_text}"}]
            + [{"type": "image", "image": im} for im in frames]
            + [{"type": "text", "text": question}]},
    ]
    text = m.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = m.proc(text=[text], images=frames, return_tensors="pt").to(m.dev)
    ids = inp["input_ids"][0]
    grid = inp["image_grid_thw"]
    ve = (ids == m.vision_end_id).nonzero(as_tuple=True)[0]
    k_ev = int(ve[-1].item()) + 1
    am = torch.ones((1, ids.shape[0]), dtype=torch.long, device=m.dev)
    pos, _ = m.get_rope(ids[None], inp["mm_token_type_ids"], image_grid_thw=grid, attention_mask=am)
    return ids, pos, inp["pixel_values"], grid, k_ev


def sync():
    torch.cuda.synchronize()


def greedy(m, past, k_ev, suffix_ids, pos_start, max_new, eos):
    """Prefill suffix on the shared evidence cache, then greedy-decode. Returns
    (gen_ids, prefill_ms, decode_ms)."""
    Ls = suffix_ids.shape[0]
    semb = m.embed(suffix_ids[None])
    spos = torch.arange(pos_start, pos_start + Ls, device=m.dev).view(1, 1, Ls).expand(3, 1, Ls).contiguous()
    sync(); t0 = time.perf_counter()
    logits, _ = m._lm_logits(semb, spos, past=past, cache_position=torch.arange(k_ev, k_ev + Ls, device=m.dev))
    sync(); prefill_ms = (time.perf_counter() - t0) * 1e3
    cur, npos = k_ev + Ls, pos_start + Ls
    gen = []
    sync(); t0 = time.perf_counter()
    for _ in range(max_new):
        tid = int(logits.argmax(-1))
        gen.append(tid)
        if tid in eos:
            break
        emb = m.embed(torch.tensor([[tid]], device=m.dev))
        p1 = torch.full((3, 1, 1), npos, device=m.dev)
        logits, _ = m._lm_logits(emb, p1, past=past, cache_position=torch.arange(cur, cur + 1, device=m.dev))
        cur += 1; npos += 1
    sync(); decode_ms = (time.perf_counter() - t0) * 1e3
    return gen, prefill_ms, decode_ms


def main():
    torch.set_grad_enabled(False)
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        R.CKPT, device_map={"": 0}, attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(R.CKPT)
    proc.image_processor.max_pixels = 128 * 28 * 28
    if isinstance(getattr(proc.image_processor, "size", None), dict):
        proc.image_processor.size["longest_edge"] = 128 * 28 * 28
    m = R.M(model, proc)
    eos = {m.tok.convert_tokens_to_ids("<|im_end|>"), m.tok.eos_token_id}
    eos.discard(None)

    ann = json.load(open(STEP_ANN))
    spans = {rid: {s["step_id"]: (s["start_time"], s["end_time"]) for s in r["steps"]}
             for rid, r in ann.items()}
    crit = json.load(open(CRITERIA))
    steps = {n["step_id"]: {"instr": n["instruction"],
                            "claims": [c["claim"] for c in n["checks"]]} for n in crit["nodes"]}

    # ---- warm up triton/FLA + decode kernels (first call autotunes ~18s; not measured) ----
    wf = sample_frames(f"{VIDEO_DIR}/{RECORDINGS[0]}.mp4", 180.0, 200.0, 8)
    wids, wpos, wpix, wgrid, wk = build_ids(m, "warmup", wf, OPEN_Q)
    wpast = m.stage_PF(m.stage_MG(wids[:wk], m.stage_V(wpix, wgrid)), wpos[:, :, :wk], wk)
    greedy(m, wpast, wk, wids[wk:], int(wpos[:, :, :wk].max()) + 1, 12, eos)
    wpast.crop(wk); greedy(m, wpast, wk, wids[wk:], int(wpos[:, :, :wk].max()) + 1, 1, eos)
    del wpast
    print("[warmup] done", flush=True)

    rows = []
    for rid in RECORDINGS:
        video = f"{VIDEO_DIR}/{rid}.mp4"
        if not os.path.exists(video):
            print(f"[skip] no video {rid}", flush=True); continue
        for sid, node in steps.items():
            if sid not in spans.get(rid, {}):
                continue
            start, end = spans[rid][sid]
            nfr = max(2, min(MAX_FRAMES, int(end - start)))
            frames = sample_frames(video, start, end, nfr)

            # shared evidence prefill (ViT once, prefill once) using the OPEN variant's prefix
            ids_o, pos_o, pix, grid, k_ev = build_ids(m, node["instr"], frames, OPEN_Q)
            vit = m.stage_V(pix, grid)
            ev_emb = m.stage_MG(ids_o[:k_ev], vit)
            sync(); t0 = time.perf_counter()
            past = m.stage_PF(ev_emb, pos_o[:, :, :k_ev], k_ev)
            sync(); ev_prefill_ms = (time.perf_counter() - t0) * 1e3
            max_ev = int(pos_o[:, :, :k_ev].max())

            # OPEN: greedy decode the reminder
            open_suffix = ids_o[k_ev:]
            past.crop(k_ev)
            gen, o_pre, o_dec = greedy(m, past, k_ev, open_suffix, max_ev + 1, MAX_NEW, eos)
            n_open = len(gen)
            open_text = m.tok.decode([g for g in gen if g not in eos], skip_special_tokens=True).strip()

            # BOUNDED: one yes/no per anticipated claim, reusing the evidence cache
            claim_recs = []
            for cl in node["claims"]:
                ids_c, _, _, _, k_ev_c = build_ids(m, node["instr"], frames, claim_q(cl))
                assert k_ev_c == k_ev, (k_ev_c, k_ev)
                csuf = ids_c[k_ev:]
                past.crop(k_ev)
                g, c_pre, c_dec = greedy(m, past, k_ev, csuf, max_ev + 1, 1, eos)
                ans = m.tok.decode(g, skip_special_tokens=True).strip()
                claim_recs.append({"claim": cl[:60], "prefill_ms": round(c_pre, 1),
                                   "decode_ms": round(c_dec, 1), "answer": ans})

            k = len(node["claims"])
            row = {"rec": rid, "step": sid, "n_frames": nfr, "k_ev_tokens": k_ev,
                   "ev_prefill_ms": round(ev_prefill_ms, 1),
                   "open_decode_tokens": n_open, "open_hit_cap": n_open >= MAX_NEW,
                   "open_prefill_ms": round(o_pre, 1),
                   "open_decode_ms": round(o_dec, 1), "open_text": open_text[:120],
                   "n_claims": k, "bounded_decode_tokens": k,
                   "bounded_prefill_ms": round(sum(c["prefill_ms"] for c in claim_recs), 1),
                   "bounded_decode_ms": round(sum(c["decode_ms"] for c in claim_recs), 1),
                   "claims": claim_recs}
            rows.append(row)
            print(f"[{rid} s{sid}] open={n_open}tok/{o_dec:.0f}ms  bounded={k}tok/"
                  f"{row['bounded_decode_ms']:.0f}ms  (evprefill={ev_prefill_ms:.0f}ms)  "
                  f"open='{open_text[:50]}'", flush=True)
            del past

    # ---- aggregate ----
    n_open = [r["open_decode_tokens"] for r in rows]
    o_dec = [r["open_decode_ms"] for r in rows]
    k_cl = [r["n_claims"] for r in rows]
    b_dec = [r["bounded_decode_ms"] for r in rows]
    tot_open_tok = sum(n_open); tot_bound_tok = sum(k_cl)
    per_tok = sum(o_dec) / max(1, sum(n_open))   # ms per decoded token
    agg = {"n_steps": len(rows), "recordings": RECORDINGS,
           "open_decode_tokens": {"median": statistics.median(n_open), "mean": round(statistics.mean(n_open), 1),
                                  "min": min(n_open), "max": max(n_open), "total": tot_open_tok},
           "bounded_decode_tokens": {"median": statistics.median(k_cl), "mean": round(statistics.mean(k_cl), 2),
                                     "total": tot_bound_tok},
           "decode_ms": {"open_median": round(statistics.median(o_dec), 1),
                         "bounded_median": round(statistics.median(b_dec), 1),
                         "open_total": round(sum(o_dec), 0), "bounded_total": round(sum(b_dec), 0)},
           "open_hit_cap_count": sum(1 for r in rows if r["open_hit_cap"]),
           "max_new_cap": MAX_NEW,
           "per_decoded_token_ms": round(per_tok, 1),
           "decode_token_reduction_x": round(tot_open_tok / max(1, tot_bound_tok), 1),
           "decode_time_reduction_x": round(sum(o_dec) / max(1e-9, sum(b_dec)), 1)}
    json.dump({"agg": agg, "rows": rows}, open(os.path.join(OUTDIR, "decode_cost_planned_claims.json"), "w"), indent=2)
    print("\n==== AGGREGATE ====", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    print("\n[done] wrote decode_cost_planned_claims.json", flush=True)


if __name__ == "__main__":
    main()
