#!/usr/bin/env python3
"""Follow-up to run_preencode_arms.py: (1) constrained 3-way decode {yes,no,uncertain}
so the verifier answer is meaningful, and (2) a conclusive negative cache-control using
SUBSTANTIVELY different evidence + a bit-level logit-identity proof of cache binding.

Reuses the validated stage primitives (M, build_pre, stages) from run_preencode_arms.
Latency is NOT re-measured here (already done + verified); this only changes the readout
of the final-position logits, which is latency-neutral.

Run:  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python followup_constrained.py
"""
import os, sys, json, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_preencode_arms as R   # M, build_pre, decode_frames, WINDOWS, argmax_token, native_token, VIDEO

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Surface forms per answer class; we score the FIRST token of each form and take the
# best (max logit) within a class, then argmax across classes. Robust to leading-space /
# capitalization variants without needing an extra forward pass.
CLASS_FORMS = {
    "yes":       ["yes", " yes", "Yes", " Yes", "YES"],
    "no":        ["no", " no", "No", " No", "NO"],
    "uncertain": ["uncertain", " uncertain", "Uncertain", " Uncertain"],
}


def build_class_ids(tok):
    out = {}
    for cls, forms in CLASS_FORMS.items():
        ids = set()
        for f in forms:
            enc = tok.encode(f, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[cls] = sorted(ids)
    return out


def restricted_answer(logits, class_ids):
    """logits: (vocab,) final-position logits. Returns (answer, per-class best logit)."""
    scores = {}
    lg = logits.float()
    for cls, ids in class_ids.items():
        scores[cls] = max(float(lg[i]) for i in ids)
    answer = max(scores, key=scores.get)
    return answer, scores


def full_logits(m, pre):
    """arm-C path: full prefill -> final-position logits (vocab,)."""
    P = pre["P"]
    return m.stage_FF(pre["MG"], P["pos"], P["L"])[0]


def cached_logits(m, pre):
    """arm-D path: cached prefix + suffix forward -> final-position logits (vocab,)."""
    P = pre["P"]; k, L = P["k"], P["L"]
    pre["PF"].crop(k)
    return m.stage_SF(pre["MG"], P["pos"], k, L, pre["PF"])[0]


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
    class_ids = build_class_ids(m.tok)
    print("[class token ids]", class_ids, flush=True)

    out = {"recording": "8_26", "step_id": 90, "model": "Qwen3.5-27B-FP8",
           "decode": "constrained-3way{yes,no,uncertain}", "class_ids": class_ids,
           "windows": {}, "neg_control": {}}

    # ---- (1) constrained answer per window; confirm cached==full at the logit level ----
    for wname, win in R.WINDOWS.items():
        pre, P = R.build_pre(m, win)
        lf = full_logits(m, pre)
        lc = cached_logits(m, pre)
        free_id, free_tok = R.argmax_token(m, lf)
        ans_full, sc_full = restricted_answer(lf, class_ids)
        ans_cached, sc_cached = restricted_answer(lc, class_ids)
        logits_identical = bool(torch.equal(lf, lc))
        max_abs_diff = float((lf - lc).abs().max())
        out["windows"][wname] = {
            "win": win, "L": P["L"], "k": P["k"],
            "free_argmax_token": free_tok, "free_argmax_id": free_id,
            "constrained_answer_full": ans_full, "scores_full": sc_full,
            "constrained_answer_cached": ans_cached, "scores_cached": sc_cached,
            "cached_eq_full_answer": ans_full == ans_cached,
            "cached_eq_full_logits_bitexact": logits_identical,
            "cached_vs_full_max_abs_logit_diff": max_abs_diff,
        }
        print(f"[{wname}] free='{free_tok}'  constrained full={ans_full} cached={ans_cached}  "
              f"match={ans_full==ans_cached}  bitexact_logits={logits_identical} "
              f"(maxdiff={max_abs_diff:.2e})  scores={ {k: round(v,2) for k,v in sc_full.items()} }",
              flush=True)

    # ---- (2) conclusive negative cache-control with SUBSTANTIVELY different evidence ----
    # Correct evidence = step-90 window (chocolate-in-mug).  Alternative evidence = an
    # earlier, visually different step (~30-50s, milk/mug prep, no chocolate count).
    correct_win = R.WINDOWS["long"]                 # 186.2..206.2
    alt_win = (30.0, 50.0, 20)                       # different step, same frame budget
    pre_correct, Pc = R.build_pre(m, correct_win)
    pre_alt, Pa = R.build_pre(m, alt_win)

    # FP8 noise floor: same evidence, same path, computed twice. Any non-zero diff here is
    # pure kernel non-determinism, NOT a real change. The evidence signal must clear this.
    nf_a = full_logits(m, pre_correct)
    nf_b = full_logits(m, pre_correct)
    noise_floor = float((nf_a - nf_b).abs().max())

    # answers from a correctly-rebuilt cache for each evidence set
    lc_correct = cached_logits(m, pre_correct)
    lc_alt = cached_logits(m, pre_alt)
    ans_correct, sc_correct = restricted_answer(lc_correct, class_ids)
    ans_alt, sc_alt = restricted_answer(lc_alt, class_ids)

    # THE FAILURE MODE: evidence changed (correct -> alt) but we REUSE the stale cache built
    # on `correct`. Mechanically the suffix is identical, so the stale path returns exactly
    # the correct-evidence logits -> it silently ignores the new evidence.
    pre_correct["PF"].crop(Pc["k"])
    l_stale = m.stage_SF(pre_correct["MG"], Pc["pos"], Pc["k"], Pc["L"], pre_correct["PF"])[0]
    ans_stale, _ = restricted_answer(l_stale, class_ids)
    stale_eq_correct_bitexact = bool(torch.equal(l_stale, lc_correct))

    signal = float((lc_correct - lc_alt).abs().max())
    out["neg_control"] = {
        "correct_win": correct_win, "alt_win": alt_win,
        "answer_correct_evidence": ans_correct, "scores_correct": sc_correct,
        "answer_alt_evidence": ans_alt, "scores_alt": sc_alt,
        "answer_if_reusing_stale_cache": ans_stale,
        "fp8_noise_floor_max_abs_logit_diff": noise_floor,
        "correct_vs_alt_max_abs_logit_diff": signal,
        "signal_over_noise": round(signal / noise_floor, 1) if noise_floor else None,
        "evidence_changes_argmax_answer": ans_correct != ans_alt,
        "stale_eq_correct_bitexact": stale_eq_correct_bitexact,
    }
    print(f"\n[neg-control] FP8 noise floor (same evidence x2) max|Δlogit| = {noise_floor:.3f}", flush=True)
    print(f"[neg-control] correct vs alt evidence       max|Δlogit| = {signal:.3f}  "
          f"-> {signal/noise_floor:.1f}x noise" if noise_floor else "", flush=True)
    print(f"[neg-control] argmax: correct->{ans_correct}  alt->{ans_alt}  "
          f"(flip={ans_correct != ans_alt}); stale-reuse->{ans_stale}", flush=True)

    json.dump(out, open(os.path.join(OUTDIR, "followup_constrained.json"), "w"), indent=2)
    print("\n[done] wrote followup_constrained.json", flush=True)


if __name__ == "__main__":
    main()
