#!/usr/bin/env python3
"""Probe the live hybrid cache + measure the memory split (full-attn KV grows vs
linear-attn recurrent state constant), to ground the eviction prototype.

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python probe_eviction.py
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R


def nbytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) and t.numel() else 0


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
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    print("[layers] 64 total |", len(full), "full-attn:", full, flush=True)

    def prefill_n(nframes):
        win = (206.2 - nframes, 206.2, nframes)
        frames = R.decode_frames(*win)
        P = m.stage_P(frames)
        vit = m.stage_V(P["pixel_values"], P["grid"])
        emb = m.stage_MG(P["input_ids"], vit)
        past = m.stage_PF(emb, P["pos"], P["k"])
        return past, P

    # ---- inspect cache structure on a small prefill ----
    past, P = prefill_n(5)
    print("\n[cache] type:", type(past).__name__, flush=True)
    layers = getattr(past, "layers", None)
    if layers is not None:
        print("[cache] .layers len:", len(layers), flush=True)
        for i in [3, 0]:  # a full-attn (3) and a linear-attn (0) layer
            L = layers[i]
            attrs = {a: tuple(getattr(L, a).shape) for a in dir(L)
                     if isinstance(getattr(L, a, None), torch.Tensor) and getattr(L, a).numel()}
            print(f"  layer {i} ({layer_types[i]}) [{type(L).__name__}]: {attrs}", flush=True)

    # ---- memory split vs #frames streamed ----
    print("\n[memory] full-attn KV vs linear-attn state as frames grow:", flush=True)
    print(f"{'frames':>7} {'k_tok':>6} {'fullKV_MB':>10} {'linState_MB':>12} {'fullKV/frame_KB':>16}", flush=True)
    rows = []
    for n in [5, 10, 20, 40]:
        past, P = prefill_n(n)
        layers = past.layers
        full_b, lin_b = 0, 0
        for i, L in enumerate(layers):
            b = sum(nbytes(getattr(L, a)) for a in dir(L)
                    if isinstance(getattr(L, a, None), torch.Tensor))
            if layer_types[i] == "full_attention":
                full_b += b
            else:
                lin_b += b
        kfull = full_b / len(full) / max(P["k"], 1) * 1024  # bytes/token/layer -> approx
        print(f"{n:>7} {P['k']:>6} {full_b/1e6:>10.2f} {lin_b/1e6:>12.2f} {full_b/len(full)/P['k']/1024*1e6:>16.2f}",
              flush=True)
        rows.append({"frames": n, "k_tok": P["k"], "full_kv_MB": round(full_b/1e6, 3),
                     "lin_state_MB": round(lin_b/1e6, 3)})
        del past
    json.dump({"full_attn_layers": full, "rows": rows},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_eviction.json"), "w"), indent=2)
    print("\n[done] wrote probe_eviction.json", flush=True)


if __name__ == "__main__":
    main()
