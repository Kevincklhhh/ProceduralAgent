#!/usr/bin/env python3
"""Decode cost AT ACCURACY, thinking ON: open-ended "find any mistake" vs bounded per-claim
"did THIS happen? yes/no". The real decode cost is the <think>...</think> reasoning block.
Hypothesis: a pointed pre-compiled claim needs a SHORTER reasoning chain than open-ended,
because the search space is already narrowed -- so planned claims cut reasoning tokens at
equal (or better) accuracy.

Per (recording, step):
  evidence frames over the oracle step span (1 fps, <=16 frames)
  OPEN     : ask for any corrective reminder -> think -> verdict (reminder vs "No reminder needed")
  BOUNDED  : per anticipated claim -> ask yes/no -> think -> yes/no verdict
Measured: reasoning-token count (tokens before </think>), answer tokens, verdict, correctness vs GT.

Model generates from inside <think> (default template). Greedy, stop at <|im_end|>, cap 384.

Run: CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python decode_cost_thinking.py
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
OUTDIR = os.path.dirname(os.path.abspath(__file__))

RECORDINGS = ["8_26", "8_3"]
MAX_FRAMES, MAX_NEW = 16, 384
CHECK_SUBS = {"measurement", "technique", "preparation", "timing", "temperature"}
STRUCTURAL = {"order", "missing_step"}

SYS = "You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."
OPEN_Q = ("Watch the frames. If the user made any mistake during this step, describe the corrective "
          "reminder you would give. If everything looks correct, reply 'No reminder needed.'")
def claim_q(c): return f"Check: {c} Did this specific mistake occur in the frames? Answer yes or no."


def load_gt(rid):
    f = f"{FAMILY_A}/{rid}.json"
    if not os.path.exists(f): return set()
    gt = set()
    for e in json.load(open(f)).get("events", []):
        if e.get("cls") in ("execution_error", "parameter_violation") and \
           e.get("subtype") in CHECK_SUBS and e.get("subtype") not in STRUCTURAL and e.get("anchor_step") is not None:
            gt.add((e["anchor_step"], e["subtype"]))
    return gt


def sample_frames(video, s, e, n):
    vr = decord.VideoReader(video, num_threads=2); fps = vr.get_avg_fps()
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in np.linspace(s, max(s, e - 0.1), n)]
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
    tok = m.tok
    close_think = tok.convert_tokens_to_ids("</think>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [x for x in (im_end, tok.eos_token_id) if x is not None]
    print(f"[ids] </think>={close_think} <|im_end|>={im_end}", flush=True)

    ann = json.load(open(STEP_ANN))
    spans = {rid: {s["step_id"]: (s["start_time"], s["end_time"]) for s in r["steps"]}
             for rid, r in ann.items()}
    crit = json.load(open(CRITERIA))
    steps = {n["step_id"]: {"instr": n["instruction"],
                            "claims": [(c["reminder"], c["claim"]) for c in n["checks"]]} for n in crit["nodes"]}

    def run(step_text, frames, question):
        """Generate from inside <think>; return (think_toks, answer_toks, finished, answer_text, full_text)."""
        messages = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                    {"role": "user", "content": [{"type": "text", "text": f"Step: {step_text}"}]
                     + [{"type": "image", "image": im} for im in frames]
                     + [{"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=True)
        inp = proc(text=[text], images=frames, return_tensors="pt").to(m.dev)
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos_ids)
        g = out[0, inp["input_ids"].shape[1]:].tolist()
        g = [t for t in g if t not in eos_ids]
        if close_think in g:
            ci = g.index(close_think); finished = True
        else:
            ci = len(g); finished = False
        think_toks, ans_ids = ci, g[ci + 1:]
        ans_text = tok.decode(ans_ids, skip_special_tokens=True).strip()
        return think_toks, len(ans_ids), finished, ans_text, tok.decode(g, skip_special_tokens=True)

    rows = []
    for rid in RECORDINGS:
        video = f"{VIDEO_DIR}/{rid}.mp4"
        if not os.path.exists(video):
            print(f"[skip] {rid}", flush=True); continue
        gt = load_gt(rid)
        for sid, node in steps.items():
            if sid not in spans.get(rid, {}): continue
            s, e = spans[rid][sid]
            nfr = max(2, min(MAX_FRAMES, int(e - s)))
            frames = sample_frames(video, s, e, nfr)
            step_pos = any(st == sid for (st, _) in gt)

            # OPEN
            t_th, t_an, fin, atext, _ = run(node["instr"], frames, OPEN_Q)
            open_pred = not ("no reminder" in atext.lower() or "no mistake" in atext.lower())
            open_rec = {"think_tokens": t_th, "answer_tokens": t_an, "finished": fin,
                        "pred_positive": open_pred, "gt_positive": step_pos,
                        "correct": open_pred == step_pos, "answer": atext[:140]}

            # BOUNDED per claim
            claim_recs = []
            for (sub, cl) in node["claims"]:
                tt, ta, f2, at2, _ = run(node["instr"], frames, claim_q(cl))
                low = at2.lower()
                pred = low.startswith("yes") or ("yes" in low[:8] and "no" not in low[:8])
                gtc = (sid, sub) in gt
                claim_recs.append({"subtype": sub, "think_tokens": tt, "answer_tokens": ta,
                                   "finished": f2, "pred_positive": pred, "gt_positive": gtc,
                                   "correct": pred == gtc, "answer": at2[:40]})

            row = {"rec": rid, "step": sid, "n_frames": nfr,
                   "open": open_rec, "claims": claim_recs,
                   "bounded_think_total": sum(c["think_tokens"] for c in claim_recs),
                   "n_claims": len(claim_recs)}
            rows.append(row)
            bt = row["bounded_think_total"]; bclaim = [c["think_tokens"] for c in claim_recs]
            print(f"[{rid} s{sid}] OPEN think={t_th} pred={open_pred}({'ok' if open_rec['correct'] else 'X'}) "
                  f"| BOUNDED k={len(claim_recs)} think/claim={bclaim} sum={bt} "
                  f"verdicts={[(c['subtype'][:4], 'Y' if c['pred_positive'] else 'n', 'ok' if c['correct'] else 'X') for c in claim_recs]}",
                  flush=True)
            del frames

    # aggregate
    open_th = [r["open"]["think_tokens"] for r in rows]
    claim_th = [c["think_tokens"] for r in rows for c in r["claims"]]
    bnd_step = [r["bounded_think_total"] for r in rows]
    def acc(preds):  # (correct, positive, tp, fp, fn)
        tp = sum(1 for p in preds if p["pred_positive"] and p["gt_positive"])
        fp = sum(1 for p in preds if p["pred_positive"] and not p["gt_positive"])
        fn = sum(1 for p in preds if not p["pred_positive"] and p["gt_positive"])
        tn = sum(1 for p in preds if not p["pred_positive"] and not p["gt_positive"])
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": round(tp / max(1, tp + fn), 2), "precision": round(tp / max(1, tp + fp), 2)}
    open_preds = [r["open"] for r in rows]
    claim_preds = [c for r in rows for c in r["claims"]]
    agg = {"n_steps": len(rows), "recordings": RECORDINGS,
           "open_think_tokens": {"median": statistics.median(open_th), "mean": round(statistics.mean(open_th), 1),
                                 "min": min(open_th), "max": max(open_th)},
           "bounded_think_per_claim": {"median": statistics.median(claim_th), "mean": round(statistics.mean(claim_th), 1),
                                       "min": min(claim_th), "max": max(claim_th), "n_claims": len(claim_th)},
           "bounded_think_per_step_sum": {"median": statistics.median(bnd_step), "mean": round(statistics.mean(bnd_step), 1)},
           "open_truncated": sum(1 for r in rows if not r["open"]["finished"]),
           "claim_truncated": sum(1 for c in claim_preds if not c["finished"]),
           "accuracy_open_step_level": acc(open_preds),
           "accuracy_bounded_claim_level": acc(claim_preds),
           "think_reduction_per_claim_vs_open_x": round(statistics.median(open_th) / max(1, statistics.median(claim_th)), 2)}
    json.dump({"agg": agg, "rows": rows}, open(f"{OUTDIR}/decode_cost_thinking.json", "w"), indent=2)
    print("\n==== AGGREGATE ====", flush=True); print(json.dumps(agg, indent=2), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
