# Qualcomm zero-shot baseline — turn-based (T2 isolated), all 384, Qwen3.6-27B

- **Date:** 2026-06-28
- **Author:** claude-opus-4-8 (Claude Code session)
- **Scope:** The turn-based (T2-isolated) Qualcomm zero-shot baseline run over all 384 CC4D
  recordings with Qwen3.6-27B, scored on the Qualcomm subset via the paper's own `eval.py`.
  **Excludes** the streaming (T1+T2) pass (still running, not yet scored) and the oracle/latency
  diagnostics.
- **Code:** `eval/baseline_qualcomm_zeroshot.py`, `eval/qualcomm_adapter.py`,
  `eval/qualcomm_eval.py` @ `6562aea` (these three files are **untracked/uncommitted** — new this
  session; working tree dirty).

## Question
Given the current step (oracle) and a per-step "mistake summary" of candidate errors, can a
strong zero-shot MLLM (Qwen3.6-27B) detect procedural mistakes well enough to register on the
Qualcomm Interactive Cooking metric — i.e. what is the *ceiling* of the anticipate-and-check
approach when step identity is free?

## Setup
- **Data:** all 384 CC4D recordings (24 recipes); no train/test split applied for the runs.
  GT for scoring = Qualcomm Interactive Cooking dataset (HuggingFace `qualcomm/qualcomm-
  interactive-cooking-dataset`, loaded by their `eval.py`). Oracle step boundaries from
  `data/cc4d/annotations/annotation_json/step_annotations.json`. Videos: `data/videos_360p/*.mp4`.
- **Model:** `Qwen/Qwen3.6-27B`, served via vLLM on `http://saltyfish.eecs.umich.edu:8000`
  (verified live via `/v1/models`). Reasoning **disabled** (`chat_template_kwargs.enable_thinking=false`)
  — required, else short yes/no prompts burn the token budget (~45 s/call, `content=None`).
- **Parameters:** mode=turnbased (current step = oracle GT-active step; no completion gating),
  tick interval 5 s (paper cadence), sample 1 fps, max 8 frames/call, temperature 0, max_tokens 120.
  Prompts = verbatim Qualcomm Appendix-F (completion + mistake check). `[mistake summary]` =
  per-step probe-derived `criteria.json` checks (the GT-error-derived candidate list).
- **Command(s):**
  ```bash
  # arm (produces unified per-recording output)
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python eval/baseline_qualcomm_zeroshot.py --mode turnbased --split all --interval 5 \
    --out-dir experiments/qualcomm_run --arm qwen36_zs_turn

  # score on the Qualcomm subset via their unmodified eval.py (qual env), all 384 = train+test
  python eval/qualcomm_eval.py --mode turnbased --split all \
    --results-dir experiments/qualcomm_run --arm qwen36_zs_turn
  ```
- **Inputs:** `data/videos_360p/{rid}.mp4`; `tasks/cc4d_probe/{recipe}.generated.criteria.json`
  (24 recipes); `data/cc4d/annotations/annotation_json/step_annotations.json`;
  `data/qualcomm_interactive_cooking/qualcomm_timeline.json`; Qualcomm HF GT (via `eval.py`).

## Results
Arm output: **383/384** recordings (one skipped: `12_6`, no oracle spans/video). 1,755 mistake
events emitted; 38,053 VLM calls; 278,665 frames; ~235 s avg compute/recording (~25 h cumulative).

Scored on the Qualcomm subset (5 mistake subtypes; order/missing excluded — `0 dropped`),
their `eval.py`, 15 s window:

| Split | n | IC-Acc | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| train+val | 274 | 73.2 | 252 | 964 | 2167 | 691 | 0.21 | 0.27 | 0.23 |
| test | 109 | 72.7 | 144 | 389 | 798 | 301 | 0.27 | 0.32 | 0.29 |
| **CORPUS (384)** | **383** | **~73.1** | **396** | **1353** | **2965** | **992** | **0.226** | **0.285** | **0.252** |

Fluency on matched TPs: BERT ~0.47, ROUGE-L ~0.35. (Corpus mistake counts summed exactly across
the two disjoint loader splits; IC-Acc is recording-weighted — denominators not printed.)
Paper reference (turn-based, Table 5): baseline mistake F1 0.06–0.19; LiveMamba 0.19.

## Interpretation
Even with the step given for free and a GT-derived candidate-mistake list, the 27B model reaches
only F1≈0.25 (recall 0.29) — it misses ~70% of mistakes and **over-fires badly** (FP 1353 ≫ TP
396), confirming that mistake detection is the hard, unsolved half of the task and that
anticipation alone does not make detection reliable. This is the optimistic ceiling of the
anticipate-and-check mechanism; the deployable (firewall-clean, self-tracked-step) number will be
lower, so the project's leverage stays on cheap sensing/gating, not on detection accuracy.

## Caveats & limits
- Firewall-RELAXED: mistake summary mined from the GT error annotations (in-sample ceiling, not a clean predictor).
- Oracle step boundaries (turn-based) — step identity given free; not deployable.
- Over-firing dominates (FP 1353 vs TP 396); it also drags IC-Acc to ~73 (spurious feedback breaks the clean-segment check).
- 383/384 (`12_6` skipped); corpus IC-Acc is a recording-weighted proxy.
- Subtype tagging is a keyword heuristic (skews toward `preparation`); irrelevant to the untyped Qualcomm metric, but don't read our typed view off this.

## Artifacts
- `experiments/qualcomm_run/qwen36_zs_turn/*.json` — 383 per-recording unified arm outputs (stage_intervals + events + cost).
- `experiments/qualcomm_run/turn_all.log` — run log (all-384 pass).
- `eval/baseline_qualcomm_zeroshot.py` — the arm (Appendix-F protocol).
- `eval/qualcomm_adapter.py`, `eval/qualcomm_eval.py` — subset adapter + scoring runner.
- (Predictions fed to their eval.py were written to a temp dir; reproducible via the scoring command above.)
