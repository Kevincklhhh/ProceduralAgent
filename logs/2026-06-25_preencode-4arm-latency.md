# Procedure-compiled evidence cache: 4-arm trigger-latency ablation (8_26, step 90)

- **Date:** 2026-06-25
- **Author:** claude-opus-4-8[1m] (Claude Code session)
- **Scope:** The 4-arm async-pre-encoding latency ablation only — `run_preencode_arms.py`
  (arms cold / async-preprocess / async-ViT / async-prefill, across short/medium/long evidence
  windows). **Excludes** the constrained-decode + evidence-binding follow-up
  (`followup_constrained.py`), which is a separate run and supersedes this run's (inconclusive)
  negative-control — log that separately if wanted.
- **Code:** `experiments/preencode_8_26/run_preencode_arms.py` @ repo `6562aea` — **script is
  untracked (not committed); working tree dirty.**

## Question
When a cheap trigger fires **one** VLM (Vision-Language Model — the model that reads video frames
and answers in text) verifier call, how long is *trigger → answer*, and how much of that delay can
be removed by doing pipeline stages **ahead of the trigger** (while the system is otherwise idle)?
This is the **B-trigger** role in the project's sensing taxonomy (a cheap event wakes one VLM call
instead of running the VLM continuously); the metric is **latency**, not detection accuracy.

## Setup
- **Data:** Recording `8_26`, recipe "Spiced Hot Chocolate". One fixed, oracle-selected case:
  **step 90** ("Add 2 pieces of chocolate to the mug", span 179.75–219.54 s). GT (ground-truth
  answer key) event `8_26_e4` = `execution_error/measurement`, visible window [206.2, 234.5] s,
  from `data/cc4d_family_a/8_26.json`. GT is used only to place the evidence windows; it is not
  fed to the model.
- **Verifier prompt (fixed):** *"Procedure step: Add 2 pieces of chocolate to the mug. Check: Did
  the user add a number of chocolate pieces other than exactly 2? Answer with one token: yes, no,
  or uncertain."*
- **Model:** `Qwen/Qwen3.5-27B-FP8`, local HF weights
  `/data/kailaic/hf_cache/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a`.
  Env `qwen36` (transformers 5.13.0.dev0, torch 2.7.0+cu126). **Single GPU**, RTX 6000 Ada
  (compute capability 8.9). FP8 is **retained, not upcast** (`float8_e4m3fn`, `dequantize=False`,
  **30.4 GB** loaded) — verified by loading and inspecting param dtypes. `attn_implementation=sdpa`.
- **Parameters:** 3 evidence windows, all 1 fps, all ending at GT visibility 206.2 s:
  short = 201.2–206.2 s (5 frames), medium = 196.2–206.2 s (10), long = 186.2–206.2 s (20).
  Per-frame pixel cap `max_pixels = 128·28·28` (≈100 k px → ~110 visual tokens/frame observed).
  Greedy, `max_new_tokens=1`, `torch.set_grad_enabled(False)`. 8 warmup + 30 measured runs per
  (arm, window); `torch.cuda.synchronize()` around every timed region.
- **Command:**
  ```bash
  cd experiments/preencode_8_26
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniconda3/envs/qwen36/bin/python -u run_preencode_arms.py --warmup 8 --measured 30
  ```
- **Inputs:** `data/videos_360p/8_26.mp4` (video), `data/cc4d_family_a/8_26.json` (window/step def).

**Arms** (which pipeline stages run *before* the trigger vs *on* the trigger path):

| Arm | Pre-computed before trigger | On trigger path (counted in latency) |
|---|---|---|
| A — cold | nothing | preprocess + ViT + LM-prefill + decode |
| B — async-preprocess | frame decode/resize | ViT + prefill + decode |
| C — async-ViT | + vision encoding | prefill + decode |
| D — async-prefill | + full procedure+evidence KV cache | append question + decode |

(KV cache = the model's stored internal reading of the context; prefill = building it. Arm D is the
"procedure-compiled evidence cache": because the recipe is known, the step+frames are pre-read.)

## Results

**Trigger latency, median ms (30 runs after 8 warmup):**

| window (frames) | A cold | B async-prep | C async-ViT | **D async-prefill** | A→D speedup |
|---|---|---|---|---|---|
| short (5) | 766.9 | 566.9 | 516.6 | **133.1** | 5.8× |
| medium (10) | 1459.2 | 1081.6 | 971.3 | **132.6** | 11.0× |
| long (20) | 2809.8 | 2236.3 | 1989.3 | **133.1** | 21.1× |

p90/p95 are tight (e.g. long-D p95 = 133.4 ms; long-A p95 = 2905.1 ms). Sequence lengths: short
L=1176 (k-prefix=1139, suffix=37 tok), medium L=2286, long L=4506.

**Marginal cost each stage adds to the trigger path** (arm-median deltas, ms):

| stage | short | medium | long |
|---|---|---|---|
| preprocess (A−B) | 200.0 | 377.6 | 573.5 |
| ViT encode (B−C) | 50.3 | 110.3 | 247.0 |
| **LM prefill (C−D)** | **383.5** | **838.7** | **1856.2** |
| decode floor (D) | 133.1 | 132.6 | 133.1 |

LM prefill is the dominant stage and the **only one that scales with window size**; the arm-D decode
floor is **constant ~133 ms** regardless of 5/10/20 frames.

**Verification checks (all from `summary.json`):**
1. Ordering A>B>C>D holds on every window; the A→C/D gap grows with frame count (634→1327→2677 ms). ✓
2. Token-equivalence: cached (D) == manual-full (C) == native `model.generate`, all argmax token
   `'The'` (id 760), over 5 trials × 3 windows. ✓ (decision-level; see caveats re bit-exactness.)
3. Negative cache-control **inconclusive in this run**: baseline / stale / fresh all = `'The'`,
   `stale_vs_fresh_differ=False` — the degenerate `'The'` output couldn't show cache binding.
   (Resolved in the follow-up run, out of scope here.)

Note: the greedy `max_new_tokens=1` token is `'The'`, not yes/no/uncertain — the prompt doesn't
constrain a one-word answer. Per the run's design ("ignore output quality"), this does not affect the
latency numbers (arms measure the same forward); it only makes the answer and the neg-control
uninformative.

## Interpretation
Pre-computing the procedure+evidence KV cache before the trigger (arm D) collapses trigger→answer
latency to a **constant ~133 ms decode floor, independent of evidence-window length**, versus
767→2810 ms when done cold — a 5.8–21× cut that grows with the window, because the LM prefill it
removes is the only stage that scales with frames. For the B-trigger role this says the verifier call
can be made fast and window-size-agnostic *if* the prefill is moved off the trigger path.

## Caveats & limits
- **Oracle window:** frames are handed to the model at the right time; a real trigger must fire
  itself — not tested here.
- **Decision-equivalent, not bit-exact:** D==C at the argmax, but under FP8 the split prefill rounds
  differently (the follow-up measured ~2.1 logit diff vs a 0.0 same-shape noise floor).
- **Latency only:** no energy measured; A-solve / C-none slices and accuracy out of scope.
- **n=1 case, in-sample:** single recording/step; the model also *missed* this measurement error
  ('no error' answer) — fast ≠ correct.
- **Cold prefill ≈10–12 s** (one-time triton/FLA kernel autotune) is absorbed by warmup, not in the
  reported medians; script is untracked.

## Artifacts
- `experiments/preencode_8_26/run_preencode_arms.py` — the experiment script.
- `experiments/preencode_8_26/summary.json` — per-window/arm median/p90/p95, stage breakdown, equivalence, neg-control.
- `experiments/preencode_8_26/runs.jsonl` — 360 per-run records (4 arms × 3 windows × 30).
- `experiments/preencode_8_26/run.log` — full stdout incl. load, equivalence lines, final table.
