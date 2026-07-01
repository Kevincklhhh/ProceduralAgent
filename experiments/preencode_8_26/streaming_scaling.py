#!/usr/bin/env python3
"""Scaling curve: as the streamed evidence window grows (frames -> minutes), how do
(a) trigger->answer latency and (b) cache memory grow? Separates the 16 full-attention
layers (growing KV) from the 48 linear-attention layers (O(1) recurrent state).

No assumption about evidence locality — just the cost of the keep-everything policy.

Streams 8_26 from t=0 at 1 fps out to the full video length, measuring at log-spaced
checkpoints. At each checkpoint, the trigger appends the step-90 verifier question right
after the frames seen so far (positions continued), decodes one token, then crops back.

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python streaming_scaling.py
"""
import os, sys, json, time, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R
import followup_constrained as F

OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_MAX = 440                       # 8_26 is 445.3s; 1 fps
CHECKPOINTS = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 440]
TRIG_WARM, TRIG_MEAS = 6, 15


def sync():
    torch.cuda.synchronize()


def summ(xs):
    xs = sorted(xs)
    return dict(median=statistics.median(xs), p90=xs[min(len(xs) - 1, int(0.9 * len(xs)))],
                min=xs[0], max=xs[-1], n=len(xs))


def cache_bytes(past, layer_types):
    """Sum cache-tensor bytes per layer, split full-attn vs linear-attn."""
    full_b = lin_b = 0
    layers = getattr(past, "layers", None)
    if layers is None:
        return None, None
    for i, L in enumerate(layers):
        b = 0
        for a in ("keys", "values", "conv_states", "recurrent_states",
                  "conv_state", "recurrent_state", "key_cache", "value_cache",
                  "ssm_state"):
            v = getattr(L, a, None)
            if isinstance(v, torch.Tensor):
                b += v.numel() * v.element_size()
        if layer_types[i] == "full_attention":
            full_b += b
        else:
            lin_b += b
    return full_b, lin_b


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
    layer_types = model.config.text_config.layer_types
    vstart_id = model.config.vision_start_token_id
    class_ids = F.build_class_ids(m.tok)

    # ---- build the full N_MAX-frame layout once (positions, per-frame token chunks) ----
    frames = R.decode_frames(0.0, float(N_MAX), N_MAX)
    print(f"[setup] decoded {len(frames)} frames; building layout...", flush=True)
    P = m.stage_P(frames)               # input_ids, pos(3,1,L), pixel_values, grid, k, L
    # suffix = the verifier question + assistant header (everything after last vision_end)
    suffix_ids = P["input_ids"][P["k"]:P["L"]]
    suffix_emb = m.embed(suffix_ids[None])
    Ls = suffix_ids.shape[0]
    # per-frame pixel splits + token chunks
    counts = P["grid"].prod(-1).tolist()
    pix_splits = list(torch.split(P["pixel_values"], counts, dim=0))
    ids = P["input_ids"]
    vs = (ids == vstart_id).nonzero(as_tuple=True)[0].tolist()
    ve = (ids == m.vision_end_id).nonzero(as_tuple=True)[0].tolist()
    chunks = [(0, vs[0], None)] + [(vs[i], ve[i] + 1, i) for i in range(len(vs))]
    print(f"[setup] L={P['L']} k={P['k']} suffix={Ls} frames={len(vs)}", flush=True)

    def trigger_after(past, cur_k):
        """Append suffix right after cur_k cached tokens; decode 1 token; crop back."""
        cur_max = int(P["pos"][:, :, :cur_k].max())
        spos = torch.arange(cur_max + 1, cur_max + 1 + Ls, device=m.dev).view(1, 1, Ls).expand(3, 1, Ls).contiguous()
        cp = torch.arange(cur_k, cur_k + Ls, device=m.dev)
        past.crop(cur_k)
        lg, _ = m._lm_logits(suffix_emb, spos, past=past, cache_position=cp)
        past.crop(cur_k)
        return lg[0]

    # ---- stream frame by frame, measure at checkpoints ----
    past = None
    cur_k = 0
    per_frame_ms = {}
    rows = []
    struct_printed = False
    cp_set = set(CHECKPOINTS)
    for (a, b, fi) in chunks:
        ids_chunk = P["input_ids"][a:b]
        sync(); t0 = time.perf_counter()
        emb = m.embed(ids_chunk[None])
        if fi is not None:
            ie = m.stage_V(pix_splits[fi], P["grid"][fi:fi + 1])
            mask = (ids_chunk[None] == m.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, ie.to(emb.dtype))
        out = m.lm(inputs_embeds=emb, position_ids=P["pos"][:, :, a:b],
                   past_key_values=past, use_cache=True,
                   cache_position=torch.arange(a, b, device=m.dev))
        past = out.past_key_values
        sync(); dt = (time.perf_counter() - t0) * 1e3
        cur_k = b
        nframe = (fi + 1) if fi is not None else 0
        if fi is not None:
            per_frame_ms[nframe] = dt

        if not struct_printed and past is not None:
            layers = getattr(past, "layers", None)
            print(f"[cache] type={type(past).__name__} nlayers={len(layers) if layers else '?'}", flush=True)
            if layers:
                for i in (3, 0):
                    L = layers[i]
                    ts = {a2: tuple(getattr(L, a2).shape) for a2 in dir(L)
                          if isinstance(getattr(L, a2, None), torch.Tensor) and getattr(L, a2).numel()}
                    print(f"   layer{i} ({layer_types[i]}) {type(L).__name__}: {ts}", flush=True)
            struct_printed = True

        if nframe in cp_set:
            full_b, lin_b = cache_bytes(past, layer_types)
            for _ in range(TRIG_WARM):
                trigger_after(past, cur_k)
            tl = []
            for _ in range(TRIG_MEAS):
                sync(); s = time.perf_counter()
                lg = trigger_after(past, cur_k)
                sync(); tl.append((time.perf_counter() - s) * 1e3)
            ans, _ = F.restricted_answer(lg, class_ids)
            _, freetok = R.argmax_token(m, lg)
            tstat = summ(tl)
            row = {"frames": nframe, "ctx_tokens": cur_k,
                   "full_kv_MB": round(full_b / 1e6, 2) if full_b else None,
                   "lin_state_MB": round(lin_b / 1e6, 2) if lin_b else None,
                   "trigger_ms_median": round(tstat["median"], 1),
                   "trigger_ms_p90": round(tstat["p90"], 1),
                   "per_frame_prefill_ms": round(dt, 1),
                   "answer": ans, "free_token": freetok}
            rows.append(row)
            print(f"[ckpt] frames={nframe:>4} ctx={cur_k:>6}tok  fullKV={row['full_kv_MB']:>7}MB "
                  f"linState={row['lin_state_MB']:>6}MB  trigger={row['trigger_ms_median']:>6.1f}ms "
                  f"perframe={dt:>6.1f}ms  ans={ans}", flush=True)

    out = {"recording": "8_26", "model": "Qwen3.5-27B-FP8",
           "full_attn_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
           "n_full": layer_types.count("full_attention"), "n_linear": layer_types.count("linear_attention"),
           "checkpoints": rows}
    json.dump(out, open(os.path.join(OUTDIR, "streaming_scaling.json"), "w"), indent=2)
    print("\n[done] wrote streaming_scaling.json", flush=True)
    # quick linear-fit of trigger latency vs ctx tokens (slope ms / 1k tok)
    xs = [r["ctx_tokens"] for r in rows]; ys = [r["trigger_ms_median"] for r in rows]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0
    print(f"[fit] trigger latency slope = {slope*1000:.3f} ms per 1k ctx-tokens; "
          f"intercept ~{my-slope*mx:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
