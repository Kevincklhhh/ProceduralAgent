# Benchmark Index — canonical task + evaluator map (2026-06-26)

One page that pins **what the benchmark is**, under a single naming scheme, and reconciles
the parallel vocabularies that grew across docs. Authoritative for naming; defers to the
linked docs for full specs.

## 0. Naming reconciliation (read this first)

Four vocabularies refer to the same structure. Canonical = the **Task** column.

| Canonical task | Also called | Layer it belongs to |
|---|---|---|
| **T1** | "step localization", "Step-Acc", `stage_acc` | a *task* |
| **FA-1** | "T2 / typed streaming reminders", "reminder P/R/F1" | a *task* (reminder family) |
| **FA-2** | "T2 / intervention decision", "G-Mean" | a *task* (reminder family) |
| **FA-3** | "T2 / timing diagnostics", STS / τ | a *task* (reminder family) — **defined, not implemented** |
| — | **Box 1 / Box 2 / Box 3** | *pipeline stages*, orthogonal to tasks (see below) |

- **T2** is the umbrella for the reminder family **{FA-1, FA-2, FA-3}** — not a task itself.
- **Box 1/2/3** are pipeline stages, NOT tasks: Box 1 = GT generation (`eval/gt_build_family_a.py`),
  Box 2 = evaluation (`eval/eval_score_corpus.py`), Box 3 = recipe→sensor predictor. The
  **firewall** keeps Box 1 and Box 3 from sharing inputs beyond the recipe
  (`docs/PIPELINE_THREE_BOXES.md`).

## 1. Tasks

All tasks run on CC4D, causal/online (data ≤ t), every arm gets the same task JSON.

| Task | Question | Output | GT | Metric | Code | Status |
|---|---|---|---|---|---|---|
| **T1** | what step is happening now? | one `step_id` or `"other"` per query (1 Hz) | active-step **set** `G_t` from CC4D step segments | Active-Step Acc (+ excl-other) | `stage_acc()` | ✅ implemented |
| **FA-1** | when + what to remind? (system emits) | `events:[{t,class,subtype,reminder_id}]`, silence default | per-class event **windows** | per-class windowed **P/R/F1** (±15s / ±30s / membership); silence scored | `score_fa1()` | ✅ implemented |
| **FA-2** | speak or stay silent? (system queried) | `interrupt`/`silent` at each decision point | `decision_points` (pos = window start; neg = step-completions-with-nothing-owed + grid) | **G-Mean F1** = √(Interrupt-F1 × Silent-F1) | `score_fa2()` | ✅ implemented |
| **FA-3** | how early / how delayed? | (timing of TPs) | windows / step boundaries | STS = exp(−(t̂−s)/(e−s)); completion delay τ | — | ⚠️ **spec only, not coded** |
| **Cost** (pairs with all) | at what sensing cost? | `cost{vlm_calls,frames_sent,vlm_latency_total_s,compute_s}` | — | totals + per-video-minute | `score_cost()` | ✅ implemented |

FA-1 vs FA-2 are **different task formulations of the same goal**, not one task scored two
ways: FA-1 makes the system *emit* typed reminders on a free timeline (tests localization +
typing + silence); FA-2 *queries* a binary untyped decision at fixed points and balances it
with G-Mean (isolates the speak/quiet call, robust to class imbalance). T1 is separate (what
vs whether-to-speak). Specs: `docs/TASK_T1_STEP_LOCALIZATION.md`, `docs/REMINDER_EVALUATION.md`.

## 2. Reminder taxonomy (FA-1/FA-2 classes) — 3 classes, 7 scored subtypes

Derived mechanically from CC4D tags by `eval/gt_build_family_a.py` (map: `TAG2REMINDER`).
Each subtype's **window-start** rule differs (reactive = anchored to visibility; derived =
anchored to DAG state). Corpus counts over 384 recordings / 2,409 events
(`data/cc4d_family_a/_summary.json`).

| Class | Subtype | CC4D tag | Window-start derivation | Count |
|---|---|---|---|---|
| `execution_error` | technique | Technique | Qualcomm visibility ts (reactive) | 453 |
| | preparation | Preparation | Qualcomm visibility ts | 360 |
| | measurement | Measurement | Qualcomm visibility ts | 325 |
| | temperature | Temperature | Qualcomm ts; else power-level subset → `[step.start, step.end+grace]` | 67 |
| `parameter_violation` | timing | Timing | Qualcomm timing ts, else `step.end` fallback | 178 |
| `precondition_violation` | missing_step | Missing Step | first executed transitive DAG-successor start (derived; ~88% derivable) | 237 |
| | order | Order | the out-of-order step's own start (reactive) | 789 |

**Excluded** (not reminder classes): `safety` (no GT), next-step `guidance` (= T1 byproduct;
kept only as FA-2 negatives), `other` (no actionable subtype). See
`docs/PROACTIVE_REMINDER_GT.md`, `[[reminder-labels-need-existing-gt]]`,
`[[next-step-guidance-not-a-reminder-class]]`.

## 3. Evaluator

`eval/eval_score_corpus.py` is the Box-2 referee over all 384 `data/cc4d_family_a/` recordings.
Scores **T1 + FA-1 + FA-2 + Cost** (FA-3 not yet). Reads each arm at
`<results-dir>/<arm>/<rid>.json` (unified format, `docs/REMINDER_EVALUATION.md` §4). Built-in
**self-test arms**: `oracle` → P=R=F1=1.0; `silent` → recall 0. Splits: Qualcomm
train 213 / val 62 / test 109.

## 4. Qualcomm-comparable profile (a strict subset of our T2)

Reproduces the Qualcomm Interactive Cooking paper (NeurIPS'25) table metrics by running
**their** `eval.py` unmodified on our output.

- **Subset definition (enforced in code, `eval/qualcomm_adapter.py:SUBSET_SUBTYPES`):**
  `QUALCOMM SUBSET = our T2 − precondition_violation/{order, missing_step}`
  = the **5 mistake subtypes** {technique, preparation, measurement, temperature, timing}.
  The Qualcomm dataset has **no order/missing_step at all** (paper Appendix B: annotated for all
  categories *except* order/missing, and unused), so the relationship is clean on both sides —
  the adapter drops any arm event outside the subset (logged, never silent).
- **Metric (theirs):** segment-anchored, ROUGE≥0.8 instruction-gated, 15 s-window — IC-Acc +
  mistake P/R/F1 + BERT/ROUGE.
- **Two settings (mirror the paper):**
  - **streaming (T1+T2):** Instruction/Success from the arm's *self-tracked* timeline
    (`stage_intervals`) → completion errors propagate (paper Tables 3-4). Feedback from FA-1.
  - **turnbased (T2 isolated):** Instruction/Success from *oracle GT* step boundaries → each
    step scored independently, isolates mistake detection (paper Table 5). Feedback from FA-1.
- **Bridge + runner:** `eval/qualcomm_adapter.py` (arm output → their prediction format, with
  `--mode streaming|turnbased`); `eval/qualcomm_eval.py` builds predictions and shells out to
  `replication/qualcomm_interactive_cooking_eval/eval.py` in conda env `qual`
  (`python eval/qualcomm_eval.py --both --results-dir <dir> --arm <arm>`).
- **Status:** ✅ both modes implemented + validated. GT-oracle smoke test = IC-Acc 99.8 /
  mistake F1 0.98; real-arm path + subset-drop + streaming-vs-turnbased verified on a crafted
  arm. ❌ not yet run on a full real arm over the test split. See
  `[[qualcomm-eval-reproduction-harness]]`.

## 5. Known gaps (as of this index)

1. **FA-3 (STS, τ)** — defined in `REMINDER_EVALUATION.md`, not implemented in the referee.
2. **Qualcomm profile** — both modes implemented/validated; not yet run on a full real arm over test.
3. **Arm output format** — a documented convention, not an enforced schema; `baseline_t2_reminder`
   and `baseline_periodic_vlm` don't natively emit it (out of scope here; tracked separately).
