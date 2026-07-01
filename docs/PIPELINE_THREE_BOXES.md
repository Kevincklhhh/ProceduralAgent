# The Three Boxes (authoritative pipeline map; 2026-06-15)

This is the top-level map. The project has **three separate things**; keep them separate.
Older docs that bundled them (`CONVERSION_AND_EVAL_PROTOCOL.md`) are superseded by this split.

```
┌─ BOX 1 ─ REMINDER GT GENERATION (answer key) ────────────┐
│ in : CC4D error tags + Qualcomm timestamps + DAG          │
│ alg: reactive (Qualcomm ts + text); execution mistakes only│
│ out: data/cc4d_proactive/{rid}.json  {t, content, subtype} │
│ doc: PROACTIVE_REMINDER_GT.md   code: eval/gt_build_proactive.py
└───────────────────────────────────────────────────────────┘
                              │ truth table
                              ▼
┌─ BOX 2 ─ REMINDER EVALUATION (referee) ──────────────────┐
│ in : predicted reminders  +  Box-1 truth table            │
│ alg: windowed per-class P/R/F1; G-Mean F1; silence scored │
│ out: _scores_corpus.json (corpus) / scores.json (pilot)   │
│ doc: REMINDER_EVALUATION.md   code: eval/eval_score_corpus.py │
└───────────────────────────────────────────────────────────┘
                              ▲ predicted reminders
                              │
┌─ BOX 3 ─ RECIPE → SENSOR MAPPING (our predictor) ────────┐
│ in : recipe DAG + step text  ONLY  (never tags / ts)      │
│ alg: 2 hops recipe→criteria→sensorplan; runtime drives the │
│      active-set FSM + duty-cycles VLM; reminders ride state │
│ out: <name>.criteria.json → <name>.sensorplan.json         │
│ doc: tasks/PROCEDURE_MONITOR_COMPILER.md (stage 2) +        │
│      REMINDER_RUNTIME.md (runtime)  code: eval/proposed_*.py │
└───────────────────────────────────────────────────────────┘
```

**T1 vs T2.** The boxes above are **T2** (proactive reminders). **T1** (online current-step
recognition) shares Box 3's predictor — the same `.sensorplan.json` runtime emits the stage
timeline — and is scored by a parallel referee, `TASK_T1_STEP_LOCALIZATION.md`, built on CC4D's
**step annotations only** (no error tags / Qualcomm ts). One predictor, two scored tasks.

## The firewall (the one rule that ties the boxes together)

**Box 1 and Box 3 never share per-recording data.** They share only the recipe (DAG +
step text) — the same procedure knowledge a deployed assistant ships with — and they meet
only inside Box 2 at scoring time. Box 3 (the predictor) must NEVER see the CC4D error
tags, which recordings erred, the Qualcomm mistake timestamps, or execution traces; those
build Box 1 (the answer key). Feeding any Box-1 input into Box 3 leaks test labels and
invalidates every score.

## Box 1 — what counts as GT (mechanical-only, 2026-06-15)

Every scored event's window start comes from one of exactly **three mechanical anchors**:

| Mechanism | Window start `s` | Window end `e` | Classes (counts) |
|---|---|---|---|
| **Reactive** (Qualcomm timestamp) | Qualcomm visibility ts; `step.end` fallback for timing when absent | `step.end + grace` | execution_error {technique 453, preparation 360, measurement 325, temperature 54 (Qualcomm-timed)}, parameter/timing 178 |
| **DAG-derived** | first executed transitive DAG-successor of the skipped step | `s + grace` | precondition/missing_step 237 |
| **Step start** | the step's own `start_time` | `step.end + grace` | precondition/order 789; execution/temperature 13 (power-level, no Qualcomm ts) |

**Total scored: 2,409 events** over 384 recordings; 171 recordings have no scored event
(164 truly clean + 7 all-dropped/unmapped). Silence on clean recordings is scored.

### The order class is scored straight off the CC4D tag (decided 2026-06-15)

`precondition_violation/order` is **scored**, one event per CC4D Order-tagged step (789
events over 117 recordings). There is **no benign/harmful adjudication**: the CC4D Order
Error tag *is* the ground truth, and overriding it with our own "this one's harmless"
verdict would mean CC4D is no longer GT (the move that invalidated the old 8_50/8_45
"harmless early sugar" rationale). Every tagged step is reminder-worthy.

Each order event carries a **`dag_edge_violation`** boolean — a *diagnostic only*, it does
NOT gate scoring. It records whether the reorder breaks a real DAG edge, i.e. whether the
cheap DAG-state detector can catch it: **52% (408/789) yes** (the A-solve recoverable
slice), **48% (381/789) no** (CC4D's canonical sequence is stricter than the DAG's partial
order, or the DAG is missing an edge — a detector-recall gap to report, not a reason to
drop the event).

### Temperature (power-level subset) is now scored (2026-06-21)

The 13 Temperature-tagged steps with no Qualcomm timestamp (the "low instead of high"
power-level subset) are no longer set aside — they are scored with a **step-start anchor**:
a wrong power level is in effect from the moment the step begins, so the window is the whole
active interval `[step.start, step.end + grace]` (flagged `low_confidence_temperature`). There
are **no suspended classes**; every CC4D-tagged class is either scored or excluded by design.

### Also excluded by design (predates the mechanical cut)

- **Safety / unattended** — no CC4D tag, no Qualcomm event, nothing to derive from.
- **Next-step guidance** — its timing GT ≡ step-completion GT, so scoring it just
  re-measures step recognition. Step completions are reused as Box-2 hard negatives.

## Parameters we chose (uniform, declared once — knobs, not labels)

These are applied mechanically to every recording; they are not debatable per-case GT.

| Parameter | Value | Role |
|---|---|---|
| `grace` (`GRACE`) | 15 s | window-end pad `e`; also negative-point dedup radius |
| `δ` join slack | 10 s | how far a Qualcomm ts may fall outside `[step.start, step.end]` and still bind |
| match tolerance | ±15 s (ours) / ±30 s (LiveMamba-comparable) | TP radius for sparse classes in Box 2 |
| FA-2 negative ratio | e.g. 3:1 | silent:interrupt decision points in Box 2 |
| detector floors | per-primitive (e.g. 20 s hum) | a-priori scoring exclusions, declared per rule |

Note: `grace` is doing double duty in `gt_build_family_a.py` (window-end pad AND negative
dedup); tuning one moves the other. Split into two constants if that ever matters.

## Glossary

- **Window `[s, e]`** — `s` = warning trigger (earliest evidence); `e` = deadline/PNR.
- **PNR** (point-of-no-return) — moment after which a reminder is useless; the principled
  target for `e`, currently approximated as `step.end + grace`.
- **Reactive vs preventive** — reactive = fire when the error becomes *visible* (all our
  GT); preventive = fire *before* (not GT-derivable on CC4D; scored as earliness-in-window
  via STS instead).
- **Qualcomm layer** — local interactive-cooking annotations that add sub-step visibility
  timestamps to CC4D's otherwise untimestamped error tags.
- **Co-referential order tags** — one reorder symmetrically tags ~6.8 steps (each named
  relative to the others). We score **one event per tagged step** (789 total), not grouped
  per deviation — the chosen granularity as of 2026-06-15.
- **`dag_edge_violation`** — per-order-event diagnostic: does the reorder break a real DAG
  edge (cheap-detector recoverable, 52%) or only CC4D's stricter canonical sequence (48%)?
  Diagnostic only; does not affect scoring.
- **Transitive successor** — missing-step fires when *any* downstream dependent runs.
- **Silence GT** — the clean recordings, scored for correctly not speaking.
- **G-Mean F1** — `√(Interrupt-F1 × Silent-F1)`; degenerate always-silent/always-speak → 0.
- **STS** — earliness-within-window diagnostic, `exp(−(t̂−s)/(e−s))`.
- **FA-1 / FA-2 / FA-3** — Box-2 task arms: streaming triggering / decision-at-points /
  timing diagnostics (defined in `REMINDER_EVALUATION.md`).
