#!/usr/bin/env python3
"""Reasoning-budget vs accuracy (thinking ON, budget-forced): does a pointed bounded claim
reach the correct verdict with a SMALLER forced reasoning budget than open-ended "find any
mistake"? Greedy thinking on this model runs ~1000 tokens (open-ended doesn't even terminate),
so we FORCE-close </think> at budget B and read the verdict.

Per (recording, step):
  evidence frames over the oracle step span (<=16 @1fps); prompt opens <think> (default template)
  decode up to THINK_CAP think tokens once (greedy), then for each B in BUDGETS:
    force "...think[:B]</think>\n\n" -> decode the short answer -> verdict
  OPEN  : reminder vs "No reminder needed"   BOUNDED(per claim): yes/no
Score vs GT (data/cc4d_family_a). Manual KV decode for exact position control + cheap re-use.

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python budget_forcing.py
"""
import os, sys, json, statistics
import numpy as np, torch, decord
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R

BASE = "/home/kailaic/NeuroTrace/ProceduralAgent"
STEP_ANN = f"{BASE}/data/cc4d/annotations/annotation_json/step_annotations.json"
CRITERIA = f"{BASE}/tasks/cc4d_probe/spicedhotchocolate.generated.criteria.json"
FAMILY_A = f"{BASE}/data/cc4d_family_a"
VIDEO_DIR = f"{BASE}/data/videos_360p"
OUT = os.path.dirname(os.path.abspath(__file__))

RECORDINGS = ["8_26"]
MAX_FRAMES = 16
BUDGETS = [0, 64, 128, 256]
THINK_CAP = max(BUDGETS)
CHECK_SUBS = {"measurement", "technique", "preparation", "timing", "temperature"}

SYS = "You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."
OPEN_Q = ("Watch the frames. If the user made any mistake during this step, describe the corrective "
          "reminder you would give. If everything looks correct, reply 'No reminder needed.'")
def claim_q(c): return f"Check: {c} Did this specific mistake occur in the frames? Answer yes or no."


def load_gt(rid):
    f = f"{FAMILY_A}/{rid}.json"
    if not os.path.exists(f): return set()
    return {(e["anchor_step"], e["subtype"]) for e in json.load(open(f)).get("events", [])
            if e.get("cls") in ("execution_error", "parameter_violation")
            and e.get("subtype") in CHECK_SUBS and e.get("anchor_step") is not None}


def frames_for(video, s, e, n):
    vr = decord.VideoReader(video, num_threads=2); fps = vr.get_avg_fps()
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in np.linspace(s, max(s, e - 0.1), n)]
    b = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(b[i]) for i in range(b.shape[0])]


def main():
    torch.set_grad_enabled(False)
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        R.CKPT, device_map={"": 0}, attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(R.CKPT); proc.image_processor.max_pixels = 128 * 28 * 28
    if isinstance(getattr(proc.image_processor, "size", None), dict):
        proc.image_processor.size["longest_edge"] = 128 * 28 * 28
    m = R.M(model, proc); tok = m.tok
    close_ids = [tok.convert_tokens_to_ids("</think>")] + tok.encode("\n\n", add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = {im_end, tok.eos_token_id}; eos.discard(None)

    def fwd(past, emb, pos, cpos):
        out = m.lm(inputs_embeds=emb, position_ids=pos, past_key_values=past, use_cache=True, cache_position=cpos)
        return m.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :], out.past_key_values

    def emb_ids(ids):  # ids: python list -> (1,len,h)
        return m.embed(torch.tensor(ids, device=m.dev)[None])

    def build_prompt(step_text, frames, question):
        messages = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                    {"role": "user", "content": [{"type": "text", "text": f"Step: {step_text}"}]
                     + [{"type": "image", "image": im} for im in frames]
                     + [{"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inp = proc(text=[text], images=frames, return_tensors="pt").to(m.dev)
        ids = inp["input_ids"][0]
        am = torch.ones((1, ids.shape[0]), dtype=torch.long, device=m.dev)
        pos, _ = m.get_rope(ids[None], inp["mm_token_type_ids"], image_grid_thw=inp["image_grid_thw"], attention_mask=am)
        return ids, pos, inp["pixel_values"], inp["image_grid_thw"]

    def verdicts_over_budgets(step_text, frames, question):
        ids, pos, pix, grid = build_prompt(step_text, frames, question)
        Lp = ids.shape[0]; maxp = int(pos.max())
        vit = m.stage_V(pix, grid)
        full_emb = m.stage_MG(ids, vit)                      # image embeds scattered into prompt
        # prefill whole prompt -> cache + first think-token logits
        logits, past = fwd(None, full_emb, pos, torch.arange(Lp, device=m.dev))
        # greedy think decode up to THINK_CAP
        think = []; cur, npos = Lp, maxp + 1
        for _ in range(THINK_CAP):
            t = int(logits.argmax(-1)); think.append(t)
            if t in eos: break
            logits, past = fwd(past, emb_ids([t]), torch.full((3, 1, 1), npos, device=m.dev),
                               torch.arange(cur, cur + 1, device=m.dev))
            cur += 1; npos += 1
        out = {}
        for B in BUDGETS:
            past.crop(Lp)                                    # back to prompt-only
            seq = think[:B] + close_ids
            posx = torch.arange(maxp + 1, maxp + 1 + len(seq), device=m.dev).view(1, 1, -1).expand(3, 1, -1).contiguous()
            logits, past = fwd(past, emb_ids(seq), posx, torch.arange(Lp, Lp + len(seq), device=m.dev))
            cur2, np2, ans = Lp + len(seq), maxp + 1 + len(seq), []
            for _ in range(8):
                t = int(logits.argmax(-1))
                if t in eos: break
                ans.append(t)
                logits, past = fwd(past, emb_ids([t]), torch.full((3, 1, 1), np2, device=m.dev),
                                   torch.arange(cur2, cur2 + 1, device=m.dev))
                cur2 += 1; np2 += 1
            out[B] = tok.decode(ans, skip_special_tokens=True).strip()
        past.crop(Lp)
        return out, len(think)

    ann = json.load(open(STEP_ANN))
    spans = {rid: {s["step_id"]: (s["start_time"], s["end_time"]) for s in r["steps"]} for rid, r in ann.items()}
    crit = json.load(open(CRITERIA))
    steps = {n["step_id"]: {"instr": n["instruction"], "claims": [(c["reminder"], c["claim"]) for c in n["checks"]]}
             for n in crit["nodes"]}

    rows = []
    for rid in RECORDINGS:
        video = f"{VIDEO_DIR}/{rid}.mp4"
        if not os.path.exists(video): continue
        gt = load_gt(rid)
        for sid, node in steps.items():
            if sid not in spans.get(rid, {}): continue
            s, e = spans[rid][sid]; nfr = max(2, min(MAX_FRAMES, int(e - s)))
            fr = frames_for(video, s, e, nfr)
            step_pos = any(st == sid for (st, _) in gt)
            ov, oth = verdicts_over_budgets(node["instr"], fr, OPEN_Q)
            open_b = {B: {"answer": ov[B][:60], "pred": not ("no reminder" in ov[B].lower() or "no mistake" in ov[B].lower()),
                          "correct": (not ("no reminder" in ov[B].lower() or "no mistake" in ov[B].lower())) == step_pos}
                      for B in BUDGETS}
            claims = []
            for (sub, cl) in node["claims"]:
                cv, cth = verdicts_over_budgets(node["instr"], fr, claim_q(cl))
                gtc = (sid, sub) in gt
                per = {B: {"answer": cv[B][:20], "pred": cv[B].lower().startswith("yes"),
                           "correct": cv[B].lower().startswith("yes") == gtc} for B in BUDGETS}
                claims.append({"subtype": sub, "gt": gtc, "think_len_capped": cth, "by_budget": per})
            rows.append({"rec": rid, "step": sid, "gt_pos": step_pos, "open_think_capped": oth,
                         "open_by_budget": open_b, "claims": claims})
            print(f"[{rid} s{sid} gt={step_pos}] OPEN " +
                  " ".join(f"B{B}:{'Y' if open_b[B]['pred'] else 'n'}{'/ok' if open_b[B]['correct'] else '/X'}" for B in BUDGETS) +
                  " || claims " +
                  " | ".join(c["subtype"][:4] + "(gt" + ("Y" if c["gt"] else "n") + ")" +
                             "".join(f" B{B}:{'Y' if c['by_budget'][B]['pred'] else 'n'}" for B in BUDGETS) for c in claims),
                  flush=True)

    def acc_at(B, which):
        preds = []
        for r in rows:
            if which == "open":
                preds.append((r["open_by_budget"][B]["pred"], r["gt_pos"]))
            else:
                for c in r["claims"]:
                    preds.append((c["by_budget"][B]["pred"], c["gt"]))
        tp = sum(1 for p, g in preds if p and g); fp = sum(1 for p, g in preds if p and not g)
        fn = sum(1 for p, g in preds if not p and g); tn = sum(1 for p, g in preds if not p and not g)
        return {"recall": round(tp / max(1, tp + fn), 2), "precision": round(tp / max(1, tp + fp), 2),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    agg = {"recordings": RECORDINGS, "n_steps": len(rows), "budgets": BUDGETS, "think_cap": THINK_CAP,
           "open_accuracy_by_budget": {B: acc_at(B, "open") for B in BUDGETS},
           "bounded_accuracy_by_budget": {B: acc_at(B, "claim") for B in BUDGETS}}
    json.dump({"agg": agg, "rows": rows}, open(f"{OUT}/budget_forcing.json", "w"), indent=2)
    print("\n==== ACCURACY vs REASONING BUDGET ====", flush=True)
    print("budget |", " ".join(f"B{B}" for B in BUDGETS), flush=True)
    print("OPEN recall:   ", [agg["open_accuracy_by_budget"][B]["recall"] for B in BUDGETS], flush=True)
    print("OPEN precision:", [agg["open_accuracy_by_budget"][B]["precision"] for B in BUDGETS], flush=True)
    print("BND  recall:   ", [agg["bounded_accuracy_by_budget"][B]["recall"] for B in BUDGETS], flush=True)
    print("BND  precision:", [agg["bounded_accuracy_by_budget"][B]["precision"] for B in BUDGETS], flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
