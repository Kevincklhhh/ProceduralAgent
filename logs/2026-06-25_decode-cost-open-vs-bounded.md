# Planned claims reduce decode: open-ended reminder vs bounded claim verification (8_*)

- **Date:** 2026-06-25
- **Author:** claude-opus-4-8[1m] (Claude Code session)
- **Scope:** The decode-token-cost comparison only — `decode_cost_planned_claims.py`: one open-ended
  "describe any mistake" generation per step vs the step's bounded per-claim "did THIS happen?"
  checks, sharing a pre-encoded evidence cache. **Measures decoded-token count + decode latency,
  NOT detection accuracy.** Excludes the budget-forcing run and the thinking-control probes (same
  session, separate scope).
- **Code:** `experiments/preencode_8_26/decode_cost_planned_claims.py` @ repo `6562aea` —
  **script UNTRACKED (not committed); working tree dirty.**

## Question
A bounded "did mistake X happen? yes/no" answers in one decode token; an open-ended "what's wrong
here?" must decode a whole sentence. Decode is the sequential, latency-dominating stage (the
pre-encode work showed ~133 ms is decode-bound), and the evidence prefill is shared/pre-encodable
across both framings. So: how much **decode** does anticipating the error as pre-compiled CLAIMS
save versus letting the VLM generate the reminder open-ended — including when a step has multiple
claims?

## Setup
- **Data:** recipe `spicedhotchocolate`, 6 recordings `8_26, 8_3, 8_11, 8_15, 8_19, 8_30`. Anticipated
  claims (1–3/step, 7 steps) from `tasks/cc4d_probe/spicedhotchocolate.generated.criteria.json`.
  Oracle step windows from `data/cc4d/annotations/annotation_json/step_annotations.json`; videos
  `data/videos_360p/8_*.mp4`. **GT not consumed** (cost run, not accuracy).
- **Model:** `Qwen/Qwen3.5-27B-FP8`, local HF weights
  `/data/kailaic/hf_cache/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941…`, env `qwen36`
  (transformers 5.13.0.dev0), single GPU RTX 6000 Ada, FP8 retained, `attn_implementation=sdpa`.
- **Parameters:** ≤16 frames/step (evenly sampled over the oracle span), per-frame pixel cap
  `128·28·28`; greedy; open-ended `max_new_tokens=96`; bounded 1 token/claim. Evidence
  (`system + "Step:…" + frames`) prefilled ONCE per step and reused for the open call and every
  claim. **`enable_thinking` NOT set → default thinking ON** (see caveats). 8-token warmup decode to
  absorb the ~18 s cold triton/FLA autotune. Prompts:
  - SYS: `"You are a cooking-procedure monitor. You are shown frames from ONE step of a recipe."`
  - OPEN: `"Watch the frames. If the user made any mistake during this step, describe the corrective reminder you would give. If everything looks correct, reply 'No reminder needed.'"`
  - BOUNDED: `"Check: {claim} Did this specific mistake occur in the frames? Answer yes or no."`
- **Command:**
  ```bash
  cd experiments/preencode_8_26
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniconda3/envs/qwen36/bin/python -u decode_cost_planned_claims.py
  ```
- **Inputs:** `spicedhotchocolate.generated.criteria.json`, `step_annotations.json`, `data/videos_360p/8_*.mp4`.

## Results
42 (recording, step) units. Numbers from `decode_cost_planned_claims.json`.

| metric | open-ended | bounded (per step, summed over claims) |
|---|---|---|
| decode tokens / step (median, mean) | **96, 96** (all 42 hit the 96 cap) | **2, 2.29** (= #claims) |
| decode latency / step (median) | **9,585 ms** | **212 ms** |
| total decode tokens (42 steps) | 4,032 | 96 |
| total decode latency | 406.1 s | 9.7 s |
| **reduction** | — | **42.0× tokens, 41.8× time** |

Per-decoded-token cost **100.7 ms**. Shared evidence prefill (excluded from the above): median
**1,529 ms/step** (225–5,171 ms), reused by the open call + all claims. Verbatim outputs (8_26 s90):
open = `'The user wants me to monitor a cooking step: "Add 2 pieces of chocolate to the mug".\n\n1.  **Analyze the video:**…'` (capped); both claim answers = `'The'`.

## Interpretation
A bounded check decodes exactly **k tokens** (k = #claims, mean 2.29) versus the open-ended
generation's 96 (cap), and since decode is ~101 ms/token the per-step decode latency drops ~42×
(9.6 s → 0.2 s) — supporting that pointed pre-compiled claims replace a long generation with a few
1-token verdicts at the trigger. The robust, model-independent part is the **k-token bound**; the
absolute 42× is inflated by this model's behavior (below).

## Caveats & limits
- **Outputs are degenerate (thinking ON):** with `enable_thinking` unset, the model emits a reasoning
  preamble — open recited to the 96 cap **42/42 times**, and the bounded 1-token output is `'The'`,
  **not a real yes/no**. So this measures decode-token COUNT, not verdicts/accuracy.
- **42× is an over-estimate:** the open length is the cap, not a natural reminder; with
  `enable_thinking=False` or a one-sentence reminder it would be ~10× (open ~20–30 tok vs k). The
  bounded = k tokens is the reliable claim.
- **Decode-only:** the shared evidence prefill (~1.5 s, pre-encodable to the ~133 ms floor) is
  excluded from the reduction; cold first-call autotune excluded via warmup.
- **Single recipe, in-sample, n=42 steps**; claims probe-derived from these recordings.
- **Script untracked.**

## Artifacts
- `experiments/preencode_8_26/decode_cost_planned_claims.py` — the experiment script.
- `experiments/preencode_8_26/decode_cost_planned_claims.json` — agg + per-(rec,step) rows (open tokens/latency, per-claim tokens/latency/answer).
- `experiments/preencode_8_26/decode_cost.log` — full stdout.
