# Streaming rolling-prefill: mechanism validation + keep-everything cost scaling (8_26)

- **Date:** 2026-06-25
- **Author:** claude-opus-4-8[1m] (Claude Code session)
- **Scope:** Two linked streaming runs on recording 8_26: (A) the rolling-prefill prototype
  (`stream_rolling_prefill.py`) validating that incremental per-frame prefill matches batch
  prefill; (B) the keep-everything cost scaling curve (`streaming_scaling.py`) measuring trigger
  latency and cache memory as the streamed window grows to the full video. **Excludes** the
  4-arm latency ablation and the constrained-decode follow-up (already logged / separate).
- **Code:** `experiments/preencode_8_26/stream_rolling_prefill.py`,
  `experiments/preencode_8_26/streaming_scaling.py` @ repo `6562aea` — **both scripts untracked
  (not committed); working tree dirty.**

## Question
Background: the 4-arm ablation showed that pre-filling the procedure+evidence KV cache (the model's
stored internal reading of the context) before the trigger makes trigger→answer latency a constant
~133 ms. (A) Can that prefill be done **incrementally while the video streams** — one frame fed into
the cache as it arrives — and still give the same verdict (so "prefill is free" is realizable, not
just a batch trick)? (B) If so, is keeping an **arbitrarily long** window free, or where does the
cost start growing? This matters for the **B-trigger** sensing role (a cheap event wakes one VLM
call): we need to know the evidence-window budget before latency/memory bite.

## Setup
- **Data:** Recording `8_26` (recipe "Spiced Hot Chocolate"), `data/videos_360p/8_26.mp4`
  (445.3 s). Run A: evidence windows ending at 206.2 s — short 5 / medium 10 / long 20 frames @
  1 fps. Run B: streamed from t=0 at 1 fps out to 440 frames (~7.4 min). GT not consumed (these
  measure latency/memory, not detection).
- **Model:** `Qwen/Qwen3.5-27B-FP8`, local HF weights
  `/data/kailaic/hf_cache/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941…`. Env `qwen36`
  (transformers 5.13.0.dev0, torch 2.7.0+cu126), **single GPU** RTX 6000 Ada (cc 8.9). FP8 retained
  (`float8_e4m3fn`, ~30.4 GB), `attn_implementation=sdpa`. **Hybrid architecture (verified from
  config + live cache):** 64 layers = **16 full-attention** (idx 3,7,…,63; growing KV) + **48
  linear-attention** (GatedDeltaNet; `recurrent_states` 48×128×128 + `conv_states`, O(1)).
- **Parameters:** per-frame pixel cap `max_pixels=128·28·28`; greedy; verifier prompt = the step-90
  check ("…added a number other than exactly 2? yes/no/uncertain"). Run A: 2 warmup streams + 5×N
  per-frame samples; trigger floor 8 warmup + 30 measured. Run B: checkpoints at
  {10,25,50,75,100,150,200,250,300,350,400,440} frames, trigger 6 warmup + 15 measured each;
  `torch.set_grad_enabled(False)`; `torch.cuda.synchronize()` around timed regions.
- **Commands:**
  ```bash
  cd experiments/preencode_8_26
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniconda3/envs/qwen36/bin/python -u stream_rolling_prefill.py     # Run A
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniconda3/envs/qwen36/bin/python -u streaming_scaling.py          # Run B
  ```
- **Inputs:** `data/videos_360p/8_26.mp4`. Reuses stage primitives from `run_preencode_arms.py`
  (M, decode_frames, stage_P/V/MG/PF/SF) and the 3-way readout from `followup_constrained.py`.

## Results

### Run A — rolling (incremental) prefill vs batch prefill

| window | frames | ctx (k tok) | verdict batch→stream | stream-vs-batch Δlogit | FP8 noise floor | per-frame ViT+prefill (med / p90) | trigger floor (med / p90 / p95) |
|---|---|---|---|---|---|---|---|
| short | 5 | 1,139 | no → no ✓ | 1.984 | 0.000 | 146.0 / 156.5 ms | 133.2 / 148.2 / 148.2 ms |
| medium | 10 | 2,249 | no → no ✓ | 2.188 | 0.000 | 145.6 / 146.5 ms | 134.1 / 148.4 / 149.8 ms |
| long | 20 | 4,469 | no → no ✓ | 1.938 | 0.000 | 148.9 / 152.6 ms | 135.0 / 139.5 / 140.7 ms |

Streamed cache gives the **same verdict** as batch on every window; logits differ ~2 (FP8
shape-sensitivity) against a **0.000** same-shape noise floor. Per-frame cost ~147 ms (constant);
trigger floor ~133 ms (constant in N).

### Run B — keep-everything cost vs streamed window length

| frames | ctx tok | ≈time | full-attn KV | linear state | trigger med (p90) | per-frame prefill |
|---|---|---|---|---|---|---|
| 10 | 2,249 | 10 s | 147.4 MB | **79.43 MB** | 132.4 (132.7) ms | 143.7 ms |
| 25 | 5,579 | 25 s | 365.6 MB | **79.43 MB** | 132.0 (132.3) | 153.5 |
| 50 | 11,129 | 50 s | 729.4 MB | **79.43 MB** | 132.9 (133.2) | 167.6 |
| 100 | 22,229 | 1.7 m | 1,456.8 MB | **79.43 MB** | 133.8 (142.2) | 198.6 |
| 150 | 33,329 | 2.5 m | 2,184.3 MB | **79.43 MB** | 134.8 (135.2) | 226.0 |
| 200 | 44,429 | 3.3 m | 2,911.7 MB | **79.43 MB** | 136.4 (147.8) | 255.1 |
| 250 | 55,529 | 4.2 m | 3,639.2 MB | **79.43 MB** | 150.9 (153.5) | 287.7 |
| 300 | 66,629 | 5.0 m | 4,366.6 MB | **79.43 MB** | 178.5 (179.1) | 324.8 |
| 350 | 77,729 | 5.8 m | 5,094.1 MB | **79.43 MB** | 208.5 (209.2) | 366.8 |
| 400 | 88,829 | 6.7 m | 5,821.5 MB | **79.43 MB** | 234.9 (235.5) | 400.6 |
| 440 | 97,709 | 7.4 m | 6,403.5 MB | **79.43 MB** | 254.9 (255.5) | 429.1 |

Linear-attn state **flat at 79.43 MB** across all lengths. Full-attn KV grows linearly
(~65 KB/token ≈ 14.6 MB/frame → 6.4 GB at 7.4 min). Trigger latency flat (~133 ms) to ~200 frames /
44k tok, then linear: fit **1.227 ms per 1k ctx-tokens** (intercept ~110 ms), reaching 255 ms (≈2×)
by 97.7k tokens.

## Interpretation
Incremental per-frame prefill reproduces batch prefill's verdict, so "prefill is free" is realizable
in a real stream: per-frame cost ~147 ms (well under the 1 fps budget) and a constant ~133 ms
trigger floor. But keeping the window is **not** free indefinitely — the 48 linear-attention layers
are O(1) forever (79 MB flat) while the 16 full-attention layers grow KV ~15 MB/frame and push
trigger latency up past ~3.5 min of context. So for a single step / short window, keep everything;
eviction is a later lever and only needs to touch 16 of 64 layers.

## Caveats & limits
- **Verdict is degenerate/incidental:** model answers "no"/"yes" (often wrong); Run B's `ans` flips
  because evidence is t=0…N, not step-90 content — these runs measure cost, not coverage.
- **Decision-equivalent, not bit-exact:** streamed vs batch logits differ ~2 (FP8 shape effect).
- **Single GPU, single recording, oracle frames/trigger:** no energy, no generalization.
- **Latency crossover measured to 97.7k tok only;** beyond 7.4 min is extrapolation (1.23 ms/1k tok).
- **Both scripts untracked;** cold per-window prefill (~10 s triton/FLA autotune) excluded via warmup.

## Artifacts
- `experiments/preencode_8_26/stream_rolling_prefill.py` / `.json` / `stream.log` — Run A.
- `experiments/preencode_8_26/streaming_scaling.py` / `streaming_scaling.json` / `scaling.log` — Run B.
