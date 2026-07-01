#!/usr/bin/env python3
"""Probe: is the 'The user wants me to…' ramble a THINKING-mode artifact?
Test bounded claim + open-ended under enable_thinking True vs False, on 8_26 step 90.

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python probe_thinking.py
"""
import os, sys, json
import numpy as np, torch, decord
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R
import followup_constrained as F

BASE = "/home/kailaic/NeuroTrace/ProceduralAgent"
VIDEO = f"{BASE}/data/videos_360p/8_26.mp4"
SYS = "You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."
STEP = "Add 2 pieces of chocolate to the mug"
CLAIM = "The recipe calls for exactly 2 pieces of chocolate; flag a different count (e.g. 1, 4, or 5 pieces instead of 2)."
CLAIM_Q = f"Check: {CLAIM} Did this specific mistake occur in the frames? Answer yes or no."
OPEN_Q = ("Watch the frames. If the user made any mistake during this step, describe the corrective "
          "reminder you would give. If everything looks correct, reply 'No reminder needed.'")


def frames(start=179.75, end=219.54, n=16):
    vr = decord.VideoReader(VIDEO, num_threads=2); fps = vr.get_avg_fps()
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in np.linspace(start, end - 0.1, n)]
    b = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(b[i]) for i in range(b.shape[0])]


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
    class_ids = F.build_class_ids(m.tok)
    fr = frames()

    def build(question, enable_thinking):
        messages = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                    {"role": "user", "content": [{"type": "text", "text": f"Step: {STEP}"}]
                     + [{"type": "image", "image": im} for im in fr]
                     + [{"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=enable_thinking)
        inp = proc(text=[text], images=fr, return_tensors="pt").to(m.dev)
        return inp, text

    def gen(inp, n):
        out = model.generate(**inp, max_new_tokens=n, do_sample=False)
        return m.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)

    def first_tok_constrained(inp):
        out = model(**inp)
        lg = out.logits[0, -1].float()
        ans, sc = F.restricted_answer(lg, class_ids)
        top = torch.topk(lg, 5)
        toks = [(m.tok.decode([int(i)]), round(float(v), 2)) for i, v in zip(top.indices, top.values)]
        return ans, {k: round(v, 2) for k, v in sc.items()}, toks

    # show how the rendered prompt tail differs
    for et in (True, False):
        _, text = build(CLAIM_Q, et)
        print(f"\n=== enable_thinking={et} | prompt tail ===\n...{repr(text[-80:])}", flush=True)

    print("\n########## BOUNDED CLAIM ##########", flush=True)
    for et in (True, False):
        inp, _ = build(CLAIM_Q, et)
        print(f"\n--- enable_thinking={et}: generate 200 tok ---", flush=True)
        print(repr(gen(inp, 200))[:900], flush=True)
        ans, sc, top = first_tok_constrained(inp)
        print(f"   1st-token constrained: ans={ans} scores={sc}", flush=True)
        print(f"   1st-token top5: {top}", flush=True)

    print("\n########## OPEN-ENDED ##########", flush=True)
    for et in (True, False):
        inp, _ = build(OPEN_Q, et)
        print(f"\n--- enable_thinking={et}: generate 200 tok ---", flush=True)
        print(repr(gen(inp, 200))[:900], flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
