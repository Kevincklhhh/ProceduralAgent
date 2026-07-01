#!/usr/bin/env python3
"""Does the LOCAL HF transformers Qwen3.5 suppress thinking when asked for JSON, the way the
remote vLLM (json_object grammar) did? Test 8_26 step 90 measurement claim under 4 configs.

vLLM json_object enforces grammar from token 1 -> no <think>. HF generate has no such grammar,
so the only local lever is enable_thinking. This checks whether asking for JSON in the PROMPT
suppresses thinking locally (hypothesis: it does NOT).

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python probe_local_json_think.py
"""
import os, sys, numpy as np, torch, decord
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R

VIDEO = "/home/kailaic/NeuroTrace/ProceduralAgent/data/videos_360p/8_26.mp4"
SYS = "You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."
STEP = "Add 2 pieces of chocolate to the mug"
CLAIM = "The recipe calls for exactly 2 pieces of chocolate; flag a different count (e.g. 1, 4, or 5 pieces instead of 2)."
Q_YESNO = f"Check: {CLAIM} Did this specific mistake occur in the frames? Answer yes or no."
Q_JSON = f'Check: {CLAIM} Did this specific mistake occur in the frames? Reply {{"is_mistake": true|false}}.'


def main():
    torch.set_grad_enabled(False)
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        R.CKPT, device_map={"": 0}, attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(R.CKPT); proc.image_processor.max_pixels = 128 * 28 * 28
    m = R.M(model, proc); tok = m.tok
    ct = tok.convert_tokens_to_ids("</think>"); ie = tok.convert_tokens_to_ids("<|im_end|>")

    vr = decord.VideoReader(VIDEO, num_threads=2); fps = vr.get_avg_fps()
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in np.linspace(179.75, 219.4, 16)]
    b = vr.get_batch(idx).asnumpy(); fr = [Image.fromarray(b[i]) for i in range(b.shape[0])]

    def run(q, enable_thinking, cap=600):
        msg = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
               {"role": "user", "content": [{"type": "text", "text": f"Step: {STEP}"}]
                + [{"type": "image", "image": im} for im in fr] + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=enable_thinking)
        inp = proc(text=[text], images=fr, return_tensors="pt").to(m.dev)
        out = model.generate(**inp, max_new_tokens=cap, do_sample=False, eos_token_id=[ie])
        g = out[0, inp["input_ids"].shape[1]:].tolist()
        g = [t for t in g if t != ie]
        closed = (ct in g)
        nat = g.index(ct) if closed else None
        ans = tok.decode(g[(nat + 1):], skip_special_tokens=True).strip() if closed else "(no </think> within cap)"
        head = tok.decode(g[:40], skip_special_tokens=True)
        return len(g), closed, nat, ans, head

    print(f"[ids] </think>={ct}\n", flush=True)
    for tag, q, et in [
        ("A think=ON  yes/no", Q_YESNO, True),
        ("B think=OFF yes/no", Q_YESNO, False),
        ("C think=ON  JSON  ", Q_JSON, True),
        ("D think=OFF JSON  ", Q_JSON, False),
    ]:
        n, closed, nat, ans, head = run(q, et)
        print(f"=== {tag} ===", flush=True)
        print(f"   gen_tokens={n} think_closed={closed} think_len={nat}", flush=True)
        print(f"   first40={head!r}", flush=True)
        print(f"   ANSWER={ans!r}\n", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
