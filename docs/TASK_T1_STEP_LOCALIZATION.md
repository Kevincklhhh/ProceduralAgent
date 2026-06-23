# T1 — Online Current-Step Recognition (step localization; 2026-06-15)

Companion to **T2 — Proactive Reminders** (`REMINDER_EVALUATION.md`). Both run on CC4D.
T1 is built on CC4D's **original step annotation only** (`complete_step_annotations.json`);
it does **not** use error tags or Qualcomm mistake timestamps (those are T2 / Box 1). The
Qualcomm `instructions` field (step-completion times) is a redundant cross-check, not a
required input.

> **Status:** the metric is already implemented as `stage_acc()` in `eval/eval_score_corpus.py`
> (per-second current-step accuracy). This doc promotes it from "T2 substrate / stage
> accuracy reporting line" to a **named, first-class task** and freezes its conventions.

## 1. Task

**Online current-step recognition.** At each query time `t`, predict the single step the
user is currently performing, from video (RGB; audio optional) **up to `t` only** — causal,
no lookahead. This is the task shape Pro2Assist reports as **Step-Acc** (Step Identification
Accuracy: "whether the predicted step matches the ground truth", Pro2Assist §6.1, p.14).
Pro2Assist is the closest published analogue; we adopt its metric shape on CC4D for the
first time (CC4D's own benchmark is offline multi-step localization / TAL, not online).

- **Input per query:** frames (and optionally audio) over `[0, t]`, plus the task JSON
  (ordered step list + text). Same procedure knowledge every arm gets (fairness, as T2).
- **Output per query:** one `step_id` from the recipe's step set, or `"other"` for off-task
  / between-steps / no-step-active time.
- **Causal & online:** a prediction at `t` may use only data ≤ `t`. Arms may declare a
  smoothing lookahead (e.g. T2's K=2-consecutive transition rule); it is reported, not free.

## 2. Ground truth (CC4D original step segments)

GT comes from `annotation_json/complete_step_annotations.json`: per recording, executed
steps `{step_id, start_time, end_time}` (5,700 instances over 384 recordings). The GT at
time `t` is the **set of active steps** `G_t = {step : start ≤ t < end}` — not a single
label. This makes the task overlap-tolerant (see §3).

Frozen conventions:

| Situation | CC4D fact | Convention |
|---|---|---|
| **Skipped step** | `start_time = end_time = -1` (287 of 5,700) | excluded — never active; contributes nothing to `G_t` |
| **Overlapping steps** (multitasking, e.g. prep during a microwave run) | ~9% of consecutive segments overlap (453 of 5,029 gaps) | **both** steps are in `G_t`. A single prediction naming **either** is correct — no foreground/dominant pick is imposed |
| **Gap / before first / after last step** | no segment covers `t` | `G_t = ∅`; correct iff prediction = `"other"` |
| **Repeated step ids** within a recording | ids repeat in execution order | scored by `step_id` (instance identity not required for current-step accuracy) |

Evaluation horizon: `[0, last executed step end]`, sampled at **1 Hz** (`t = i + 0.5`).
1 Hz is a deployment-neutral fine grid; report the rate so it is reproducible.

## 3. Metric — Active-Step Accuracy (overlap-tolerant)

The system outputs **one** step `p_t` per query. It is scored correct iff:

- `p_t ∈ G_t`  (it names *any* currently-active step), or
- `G_t = ∅` **and** `p_t = "other"` (correctly off-task).

This keeps Pro2Assist-style single-prediction current-step recognition but stops punishing
a system during legitimate multitasking: when two steps genuinely overlap, naming either is
right. Only the **GT** side is a set; the prediction stays single-label, so the metric does
not become trivially easier off the overlap regions (89% of time `|G_t| = 1` and it reduces
exactly to Step-Acc).

| Metric | Definition | Role |
|---|---|---|
| **Active-Step Acc** | fraction of sampled `t` scored correct by the rule above, **including** `G_t = ∅` (`"other"`) | headline; reduces to Pro2Assist Step-Acc where `|G_t| = 1` |
| **Active-Step Acc (excl. other)** | same, restricted to `t` where `G_t ≠ ∅` | guards against inflation from long easy `"other"` stretches |
| **Macro Active-Step Acc** *(secondary)* | per-step accuracy averaged over steps | guards against long-step dominance |

Both headline variants are emitted by `stage_acc()` (`accuracy`, `accuracy_excl_other`),
now using the active-set rule. Macro is a small addition if wanted.

## 4. Why T1 exists for *this* project (cost pairing — the actual contribution)

A bare accuracy number is **not** the contribution. Per the project goal (sensor control:
energy/latency saved at equal coverage), T1 is only meaningful **paired with the sensing
cost that achieved it**. Report Step-Acc *alongside* `cost` (`vlm_calls`, `frames_sent`,
`vlm_latency_total_s`, detector `compute_s`) — already logged per arm in `eval_score_corpus.py`.

The intended comparison: a **periodic-VLM** arm (call the VLM every Δ s) vs a
**detector-gated** arm (cheap RGB/audio tracks the step; VLM fires only on ambiguous
transitions). The claim T1 supports is *"same Step-Acc at X% of always-on-VLM cost"* —
never "Step-Acc = N%" in isolation.

## 5. Relationship to T2

T1 is the **substrate** for T2: step transitions are where T2 mints decision points
(`s` = window start) and hard negatives (a step completed, but no reminder is owed —
stay silent). T1 and T2 share GT *source* (CC4D step segments) but score different things:
T1 = "what step is this," T2 = "should I speak now." They stay separable — an arm can be
graded on either independently.

## 6. Open knobs (declared defaults; change here if revisited)

- **Sampling rate:** 1 Hz. (Lower = cheaper to score, coarser; higher = finer.)
- **Overlap rule:** GT is the active-step **set** `G_t`; a single prediction is correct if
  it names any member (Active-Step Accuracy). (Rejected alternative: force a single
  foreground GT via most-recently-started — punishes correct multitasking.)
- **Headline includes `"other"`:** yes, but always report excl-other beside it.
