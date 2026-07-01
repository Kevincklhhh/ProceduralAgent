#!/usr/bin/env python3
"""Controlled pre-encode / KV-cache latency experiment on recording 8_26, step 90.

Model: Qwen3.5-27B-FP8 (LOCAL HF weights), env `qwen36` (transformers 5.13.dev).
Run with GPU 7 only visible:  CUDA_VISIBLE_DEVICES=7 python run_preencode_arms.py

Step 90: "Add 2 pieces of chocolate to the mug"  (span 179.75..219.54s)
GT event 8_26_e4: execution_error/measurement, window [206.2, 234.5].

Fixed evidence cut (plan pretended-selected), all ending at GT visibility 206.2s:
  short : 201.2..206.2,  5 frames @1fps
  medium: 196.2..206.2, 10 frames @1fps
  long  : 186.2..206.2, 20 frames @1fps

Prompt:
  Procedure step: Add 2 pieces of chocolate to the mug.
  Check: Did the user add a number of chocolate pieces other than exactly 2?
  Answer with one token: yes, no, or uncertain.

Four arms, differing ONLY in which stages run BEFORE vs AT trigger.
  Shared stages:  P (decord decode + processor -> pixel_values,input_ids)
                  V (vision tower -> image embeds)
                  MG (scatter image embeds into text embeds -> merged inputs_embeds)
                  PF (LM prefill over prefix [proc+evidence] -> past_key_values)   [arm D only]
                  FF (LM prefill over FULL seq -> logits)                          [arms A/B/C]
                  SF (LM forward of suffix [check] with cached past -> logits)     [arm D only]
  A cold        : trigger = P + V + MG + FF
  B async-prep  : trigger =     V + MG + FF      (P pre)
  C async-vit   : trigger =         FF           (P,V,MG pre)
  D async-prefill: trigger =        SF           (P,V,MG,PF pre)

greedy, max_new_tokens=1.  Reports TTFT (trigger->first token) and critical-path
(trigger->done) -- equal at 1 token by construction, both logged.

Verification:
  (1) latency decomposition A>B>C>D, gap grows with frame count
  (2) token-equivalence: A argmax id == D argmax id over trials
  (3) negative cache-control: perturb one prefix token -> D must change / not silently reuse
"""
import os, sys, glob, json, time, argparse, statistics
import torch
import numpy as np
from PIL import Image
import decord

CKPT = glob.glob("/data/kailaic/hf_cache/models--Qwen--Qwen3.5-27B-FP8/snapshots/*")[0]
VIDEO = "/home/kailaic/NeuroTrace/ProceduralAgent/data/videos_360p/8_26.mp4"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

PROC_STEP = "Add 2 pieces of chocolate to the mug."
SYS = "You are a cooking-procedure monitor."
PROC_TEXT = f"Procedure step: {PROC_STEP}"
CHECK_TEXT = ("Check: Did the user add a number of chocolate pieces other than exactly 2?\n"
              "Answer with one token: yes, no, or uncertain.")

WINDOWS = {
    "short":  (201.2, 206.2, 5),
    "medium": (196.2, 206.2, 10),
    "long":   (186.2, 206.2, 20),
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- frame I/O
def decode_frames(start_s, end_s, n_frames):
    """decord decode of n_frames at 1fps over [start,end]. Returns list[PIL]."""
    vr = decord.VideoReader(VIDEO, num_threads=2)
    fps = vr.get_avg_fps()
    ts = [start_s + i for i in range(n_frames)]           # 1 fps
    ts = [min(t, end_s) for t in ts]
    idx = [min(int(round(t * fps)), len(vr) - 1) for t in ts]
    batch = vr.get_batch(idx).asnumpy()                   # (n,H,W,3)
    return [Image.fromarray(batch[i]) for i in range(batch.shape[0])]


# ---------------------------------------------------------------- model glue
class M:
    """Resolved handles + stage primitives for the loaded model."""
    def __init__(self, model, processor):
        self.model = model
        self.proc = processor
        self.tok = processor.tokenizer
        self.dev = model.device
        cfg = model.config
        self.image_token_id = cfg.image_token_id
        self.vision_end_id = cfg.vision_end_token_id
        # resolve submodules  (Qwen3_5Model = model.model; text = .language_model)
        mm = model.model
        self.inner = mm
        self.get_image_features = mm.get_image_features          # -> BaseModelOutputWithPooling
        self.lm = mm.language_model                              # Qwen3_5TextModel
        self.embed = self.lm.embed_tokens
        self.lm_head = model.lm_head
        self.get_rope = mm.get_rope_index                        # needs mm_token_type_ids
        log("[resolve] visual:", type(mm.visual).__name__,
            "| lm:", type(self.lm).__name__, "| get_rope: yes")

    # ---- P : build processor inputs (cpu->gpu) for a set of PIL frames
    def stage_P(self, frames):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content":
                [{"type": "text", "text": PROC_TEXT}]
                + [{"type": "image", "image": im} for im in frames]
                + [{"type": "text", "text": CHECK_TEXT}]},
        ]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=frames, return_tensors="pt").to(self.dev)
        input_ids = inp["input_ids"][0]
        grid = inp["image_grid_thw"]
        pix = inp["pixel_values"]                        # get_image_features casts dtype internally
        mmtt = inp["mm_token_type_ids"]                  # (1,L) required for M-RoPE
        # split index: right after the LAST vision_end token
        ve = (input_ids == self.vision_end_id).nonzero(as_tuple=True)[0]
        k = int(ve[-1].item()) + 1
        L = input_ids.shape[0]
        am = torch.ones((1, L), dtype=torch.long, device=self.dev)
        pos, _ = self.get_rope(input_ids[None], mmtt, image_grid_thw=grid,
                               attention_mask=am)        # (3,1,L) absolute 3D positions
        return dict(input_ids=input_ids, pixel_values=pix, grid=grid,
                    k=k, L=L, pos=pos, am=am)

    # ---- V : vision tower -> image embeds (matches model.forward's get_image_features+cat)
    def stage_V(self, pix, grid):
        out = self.get_image_features(pix, grid)
        return torch.cat(out.pooler_output, dim=0)

    # ---- MG : merged full inputs_embeds (text embeds with image embeds scattered)
    def stage_MG(self, input_ids, img_embeds):
        emb = self.embed(input_ids[None])                      # (1,L,h)
        mask = (input_ids[None] == self.image_token_id).unsqueeze(-1).expand_as(emb)
        emb = emb.masked_scatter(mask, img_embeds.to(emb.dtype))
        return emb

    def _lm_logits(self, inputs_embeds, position_ids, past=None, cache_position=None):
        out = self.lm(inputs_embeds=inputs_embeds, position_ids=position_ids,
                      past_key_values=past, use_cache=True, cache_position=cache_position)
        h = out.last_hidden_state
        past = out.past_key_values
        logits = self.lm_head(h[:, -1:, :])                    # only need last position
        return logits[:, -1, :], past

    # ---- FF : full prefill over whole seq -> last-token logits  (arms A/B/C)
    def stage_FF(self, full_embeds, pos, L):
        cp = torch.arange(L, device=self.dev)
        logits, _ = self._lm_logits(full_embeds, pos, past=None, cache_position=cp)
        return logits

    # ---- PF : prefill prefix only -> past  (arm D pre-trigger)
    def stage_PF(self, full_embeds, pos, k):
        cp = torch.arange(k, device=self.dev)
        out = self.lm(inputs_embeds=full_embeds[:, :k, :], position_ids=pos[:, :, :k],
                      past_key_values=None, use_cache=True, cache_position=cp)
        return out.past_key_values

    # ---- SF : suffix forward with cached past -> last-token logits  (arm D trigger)
    def stage_SF(self, full_embeds, pos, k, L, past):
        cp = torch.arange(k, L, device=self.dev)
        logits, _ = self._lm_logits(full_embeds[:, k:, :], pos[:, :, k:], past=past, cache_position=cp)
        return logits


def argmax_token(m, logits):
    tid = int(logits.argmax(-1).item())
    return tid, m.tok.decode([tid]).strip()


def native_token(m, frames):
    """Ground-truth reference: the real model.generate path (greedy, 1 token).
    Used to validate that the manual stage decomposition reproduces the model exactly."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content":
            [{"type": "text", "text": PROC_TEXT}]
            + [{"type": "image", "image": im} for im in frames]
            + [{"type": "text", "text": CHECK_TEXT}]},
    ]
    text = m.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = m.proc(text=[text], images=frames, return_tensors="pt").to(m.dev)
    with torch.no_grad():
        out = m.model.generate(**inp, max_new_tokens=1, do_sample=False)
    tid = int(out[0, inp["input_ids"].shape[1]])
    return tid, m.tok.decode([tid]).strip()


# ---------------------------------------------------------------- timing
def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync(); t = time.perf_counter(); r = fn(); sync()
    return r, time.perf_counter() - t


def run_arm(m, arm, frames, pre):
    """Execute one arm once. `pre` = cached pre-trigger artifacts (built per window).
    Returns dict of stage times + trigger_latency + token."""
    rec = {}
    # ---- pre-trigger (not counted in trigger latency); reuse cached `pre` ----
    P = pre["P"]; img_embeds = pre["V"]; full_embeds = pre["MG"]; past = pre["PF"]
    L, k, pos = P["L"], P["k"], P["pos"]

    # ---- at-trigger work depends on arm ----
    sync(); t0 = time.perf_counter()
    if arm == "A":
        frames2 = decode_frames(*pre["win"])               # P: decode
        (P2), tp = timed(lambda: m.stage_P(frames2))
        (ie), tv = timed(lambda: m.stage_V(P2["pixel_values"], P2["grid"]))
        (fe), tm = timed(lambda: m.stage_MG(P2["input_ids"], ie))
        (lg), tf = timed(lambda: m.stage_FF(fe, P2["pos"], P2["L"]))
        rec.update(preprocess_s=tp, vit_encode_s=tv, merge_s=tm, full_prefill_s=tf)
        logits = lg
    elif arm == "B":
        (ie), tv = timed(lambda: m.stage_V(P["pixel_values"], P["grid"]))
        (fe), tm = timed(lambda: m.stage_MG(P["input_ids"], ie))
        (lg), tf = timed(lambda: m.stage_FF(fe, pos, L))
        rec.update(preprocess_s=0.0, vit_encode_s=tv, merge_s=tm, full_prefill_s=tf)
        logits = lg
    elif arm == "C":
        (lg), tf = timed(lambda: m.stage_FF(full_embeds, pos, L))
        rec.update(preprocess_s=0.0, vit_encode_s=0.0, merge_s=0.0, full_prefill_s=tf)
        logits = lg
    elif arm == "D":
        past.crop(k)                                        # untimed: restore prefix-only cache
        (lg), ts = timed(lambda: m.stage_SF(full_embeds, pos, k, L, past))
        rec.update(preprocess_s=0.0, vit_encode_s=0.0, merge_s=0.0, evidence_prefill_s=0.0,
                   suffix_decode_s=ts)
        logits = lg
    sync(); trigger_latency = time.perf_counter() - t0
    tid, tok = argmax_token(m, logits)
    rec.update(trigger_latency_s=trigger_latency, ttft_s=trigger_latency,
               critical_path_s=trigger_latency, token_id=tid, generated_token=tok)
    return rec


def build_pre(m, win):
    """Build all pre-trigger artifacts for a window (timed once, reported)."""
    start, end, n = win
    t_frames = time.perf_counter(); frames = decode_frames(start, end, n); t_frames = time.perf_counter() - t_frames
    P, tp = timed(lambda: m.stage_P(frames))
    V, tv = timed(lambda: m.stage_V(P["pixel_values"], P["grid"]))
    MG, tm = timed(lambda: m.stage_MG(P["input_ids"], V))
    PF, tpf = timed(lambda: m.stage_PF(MG, P["pos"], P["k"]))
    return dict(frames=frames, P=P, V=V, MG=MG, PF=PF, win=win,
                async_decode_s=t_frames, async_preprocess_s=tp, async_vit_s=tv,
                async_merge_s=tm, async_prefill_s=tpf), P


def summarize(xs):
    xs = sorted(xs)
    return dict(median=statistics.median(xs),
                p90=xs[min(len(xs) - 1, int(0.90 * len(xs)))],
                p95=xs[min(len(xs) - 1, int(0.95 * len(xs)))],
                mean=statistics.mean(xs), n=len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measured", type=int, default=30)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=128 * 28 * 28,  # ~128 vis tokens/frame
                    help="cap per-frame pixels fed to the vision tower")
    args = ap.parse_args()

    # FP8 (float8_e4m3fn) is retained on cc>=8.9 (RTX 6000 Ada): ~30GB on one 49GB card,
    # ~18GB free for activations. Single-GPU keeps the latency numbers free of cross-GPU
    # pipeline distortion. (Run with CUDA_VISIBLE_DEVICES=7 -> the only fully-free card.)
    torch.set_grad_enabled(False)   # manual forwards must not build the autograd graph
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    log("[load] ckpt:", CKPT, "attn:", args.attn, "max_pixels:", args.max_pixels)
    t = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        CKPT, device_map={"": 0}, attn_implementation=args.attn).eval()
    proc = AutoProcessor.from_pretrained(CKPT)
    proc.image_processor.max_pixels = args.max_pixels
    if hasattr(proc.image_processor, "size") and isinstance(proc.image_processor.size, dict):
        proc.image_processor.size["longest_edge"] = args.max_pixels
    log(f"[load] done in {time.perf_counter()-t:.1f}s; dtype sample:",
        next(model.parameters()).dtype, "| device map spread:",
        sorted({str(p.device) for p in model.parameters()}))
    m = M(model, proc)

    jsonl = open(os.path.join(OUTDIR, "runs.jsonl"), "w")
    summary = {"recording": "8_26", "step_id": 90, "model": "Qwen3.5-27B-FP8",
               "attn": args.attn, "warmup": args.warmup, "measured": args.measured,
               "windows": {}, "equivalence": {}, "neg_control": {}}

    for wname, win in WINDOWS.items():
        log(f"\n===== window {wname} {win} =====")
        pre, P = build_pre(m, win)
        log(f"[pre {wname}] L={P['L']} k={P['k']} img_tokens={int((P['input_ids']==m.image_token_id).sum())} "
            f"decode={pre['async_decode_s']*1e3:.0f}ms prep={pre['async_preprocess_s']*1e3:.0f}ms "
            f"vit={pre['async_vit_s']*1e3:.0f}ms merge={pre['async_merge_s']*1e3:.0f}ms "
            f"prefill={pre['async_prefill_s']*1e3:.0f}ms")
        summary["windows"][wname] = {"L": P["L"], "k": P["k"], "win": win,
            "async": {kk: pre[kk] for kk in pre if kk.startswith("async")}, "arms": {}}

        # native reference (real model.generate) validates the whole manual decomposition
        nat_id, nat_tok = native_token(m, pre["frames"])
        # token-equivalence: manual-full (C) and cached (D) vs native, over greedy trials
        eq = []
        for _ in range(5):
            ra = run_arm(m, "C", pre["frames"], pre)   # manual full forward
            rd = run_arm(m, "D", pre["frames"], pre)   # cached-prefix forward
            eq.append((ra["token_id"], rd["token_id"]))
        c_match = all(a == nat_id for a, _ in eq)
        d_match = all(b == nat_id for _, b in eq)
        cd_match = all(a == b for a, b in eq)
        summary["equivalence"][wname] = {"native_id": nat_id, "native_token": nat_tok,
            "pairs_cD": eq, "manualfull_eq_native": c_match,
            "cached_eq_native": d_match, "cached_eq_manualfull": cd_match}
        log(f"[equiv {wname}] native='{nat_tok}'({nat_id})  "
            f"manualfull==native:{c_match}  cached==native:{d_match}  cached==manualfull:{cd_match}  pairs={eq}")

        for arm in ["A", "B", "C", "D"]:
            for _ in range(args.warmup):
                run_arm(m, arm, pre["frames"], pre)
            trig, tok = [], None
            stage_acc = {}
            for _ in range(args.measured):
                r = run_arm(m, arm, pre["frames"], pre)
                trig.append(r["trigger_latency_s"]); tok = r["generated_token"]
                for kk, vv in r.items():
                    if kk.endswith("_s"):
                        stage_acc.setdefault(kk, []).append(vv)
                jsonl.write(json.dumps({
                    "recording": "8_26", "step_id": 90,
                    "evidence_start_s": win[0], "evidence_end_s": win[1], "n_frames": win[2],
                    "window": wname, "arm": arm,
                    "preprocess_s": round(r.get("preprocess_s", 0.0), 6),
                    "vit_encode_s": round(r.get("vit_encode_s", 0.0), 6),
                    "merge_s": round(r.get("merge_s", 0.0), 6),
                    "full_prefill_s": round(r.get("full_prefill_s", 0.0), 6),
                    "evidence_prefill_s": round(pre["async_prefill_s"], 6) if arm == "D" else 0.0,
                    "suffix_decode_s": round(r.get("suffix_decode_s", 0.0), 6),
                    "trigger_latency_s": round(r["trigger_latency_s"], 6),
                    "ttft_s": round(r["ttft_s"], 6),
                    "critical_path_s": round(r["critical_path_s"], 6),
                    "generated_token": r["generated_token"],
                }) + "\n")
            s = summarize(trig)
            summary["windows"][wname]["arms"][arm] = {
                "trigger_latency": s, "generated_token": tok,
                "stages": {kk: summarize(vv) for kk, vv in stage_acc.items()}}
            log(f"[{wname} arm {arm}] trig median={s['median']*1e3:.1f}ms "
                f"p90={s['p90']*1e3:.1f} p95={s['p95']*1e3:.1f}  tok='{tok}'")

    # negative cache-control: flip one prefix token id, reuse OLD past -> token may change;
    # the point is we are NOT silently reusing a stale cache for different evidence.
    log("\n===== negative cache-control (long) =====")
    pre = build_pre(m, WINDOWS["long"])[0]
    P = pre["P"]; k, L, pos = P["k"], P["L"], P["pos"]
    base = run_arm(m, "C", pre["frames"], pre)              # correct full forward, reference token
    # corrupt ONE prefix (procedure) token; its effect lives entirely in the prefix region.
    bad_ids = P["input_ids"].clone()
    pos5 = 5; bad_ids[pos5] = int(bad_ids[pos5].item()) + 1000
    bad_embeds = m.stage_MG(bad_ids, pre["V"])
    cp = torch.arange(k, L, device=m.dev)
    # STALE: reuse the CORRECT prefix cache with corrupted-suffix forward -> corruption is
    # invisible (it lived in the cached prefix), so answer == baseline. This is the failure
    # mode we must avoid: a stale cache silently ignores changed evidence.
    correct_past = m.stage_PF(pre["MG"], pos, k)
    logits_stale, _ = m._lm_logits(bad_embeds[:, k:, :], pos[:, :, k:], past=correct_past, cache_position=cp)
    # FRESH: rebuild the prefix cache from corrupted evidence -> reflects the change.
    good_past = m.stage_PF(bad_embeds, pos, k)
    logits_fresh, _ = m._lm_logits(bad_embeds[:, k:, :], pos[:, :, k:], past=good_past, cache_position=cp)
    t_stale = argmax_token(m, logits_stale); t_fresh = argmax_token(m, logits_fresh)
    summary["neg_control"] = {"baseline_token": base["generated_token"],
                              "stale_past_token": t_stale[1], "fresh_past_token": t_fresh[1],
                              "stale_vs_fresh_differ": t_stale[0] != t_fresh[0]}
    log(f"[neg-control] baseline='{base['generated_token']}' "
        f"stale-cache='{t_stale[1]}' fresh-cache='{t_fresh[1]}' "
        f"differ={t_stale[0]!=t_fresh[0]}")

    jsonl.close()
    json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    log("\n[done] wrote runs.jsonl + summary.json to", OUTDIR)

    # final table
    log("\n==== TRIGGER LATENCY (median ms) ====")
    log(f"{'window':8} {'A cold':>9} {'B prep':>9} {'C vit':>9} {'D prefill':>9}   token")
    for wname in WINDOWS:
        a = summary["windows"][wname]["arms"]
        row = " ".join(f"{a[x]['trigger_latency']['median']*1e3:9.1f}" for x in "ABCD")
        log(f"{wname:8} {row}   A='{a['A']['generated_token']}' D='{a['D']['generated_token']}'")


if __name__ == "__main__":
    main()
