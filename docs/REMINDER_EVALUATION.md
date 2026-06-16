# BOX 2 — Proactive-Reminder Evaluation (referee; 2026-06-15)

See `PIPELINE_THREE_BOXES.md` for how this fits. This box scores **predicted reminders**
(from any arm) against the **Box-1 truth table** (`data/cc4d_family_a/`). It is the only
place Box 1 (answer key) and Box 3 (predictor) meet. Split out of the former
`CONVERSION_AND_EVAL_PROTOCOL.md` Part 2.

Scored taxonomy is **mechanical-only** (2026-06-15): execution_error (technique /
preparation / measurement / temperature-with-ts), parameter/timing, precondition/missing_step,
and **precondition/order** (scored straight off the CC4D Order tag — one event per tagged
step, 789 total, no benign/harmful adjudication; see `PIPELINE_THREE_BOXES.md`). Only
temperature-power-level (13) remains suspended; safety and next-step guidance are excluded.
See `FAMILY_A_CC4D_AUGMENTATION.md` for derivation. **2,396 scored events total.**

## Protocol (frozen)

1. **Replay setting.** Offline causal replay of recorded RGB+audio. Arms may use sensor
   data up to time `t` plus a declared lookahead (detector smoothing; reported). GT is
   fully derived (Box 1) before any arm runs.
2. **Fairness.** Every arm receives the same task JSON (same procedure knowledge); arms
   differ only in sensing policy (event-driven detectors / periodic VLM / detectors +
   targeted escalation).
3. **Tuning discipline.** Detector parameters are tuned on one designated clean recording
   per recipe, frozen, evaluated on all others. Tuning recordings reported separately.
   Any structural choice made after seeing eval data is logged as a design-leakage
   disclosure.
4. **Unified result format** per recording per arm:
   `{stage_intervals, events:[{t, class, id, message}], escalation_requests,
   cost:{vlm_calls, frames_sent, vlm_latency_total_s, compute_s}}`.
5. **Splits.** Adopt the Qualcomm splits (train 213 / val 62 / test 109) for LiveMamba
   comparability.
6. **Scoring implementation.** `eval/score_corpus.py` is the corpus referee for all arms
   over the full 384-recording `data/cc4d_family_a/` truth table → `_scores_corpus.json`.
   It scores FA-1 per-class P/R/F1 (membership + ±15 s + ±30 s), FA-2 G-Mean F1, stage
   accuracy, and cost; order is scored with a `dag_edge_violation` sub-breakdown (the
   cheap-DAG-detector recoverable share, 52%). Validated by GT-derived `oracle` (→ all
   1.0) and `silent` (→ recall 0, G-Mean 0) reference arms. Real arms are scored via
   `--results-dir <dir> --arms a,b` (unified per-recording JSON: `stage_intervals`,
   `events[{t,class,subtype}]`, `escalation_requests`, `cost`). The older `eval/score.py`
   remains the activity-8 replay pilot (3 arms, hand-built truth, `experiments/replay_v1`).

## The three task arms

### FA-1 — Typed streaming reminder triggering (primary, metric M1)

System watches the stream and emits typed, timestamped reminders; silence is default.

- **Output:** events `{t, class, reminder_id}` over the three scored classes.
- **Scoring:** per-class windowed P/R/F1 vs the truth table; one TP per expected
  `(recording, event)`; any scored-id fire not owed = FP; **clean recordings count**
  (silence scored). Report ±15 s (ours) and ±30 s (LiveMamba/Ego-MC comparable) variants
  for the sparse classes.
- **Per-class always, pooled never** — 84.3% of mistake events are visual-leaning; a
  pooled number would flatter or hide the audio arm. Report temperature as its own sub-row
  (the audio-leaning slice of execution_error).
- **Window-membership for dense events:** where inter-step gaps are tight (median 4.7 s),
  a TP is `t̂ ∈ [s, e]` (membership), not a fixed radius. Fixed radius is kept only for the
  sparse mistake events where it is unambiguous and cross-paper comparable.

### FA-2 — Intervention decision at decision points (metric M2, G-Mean F1)

At each given timestamp `t`, output `interrupt` / `silent` given stream up to `t` + task JSON.

- **Positive points:** one per truth-table event, at window start `s`.
- **Negative points:** (a) every step completion where nothing is owed (the **hard
  negatives** — something happened, but no intervention is due; where over-talkative VLMs
  fail); (b) mid-step samples ≥ tolerance from any window; (c) all of the above on clean
  recordings.
- **Grid, not just events:** negatives are placed on a regular grid + all step boundaries
  (PWR-style), so the mere *presence* of a query carries no signal.
- **Collision rule:** points closer than the tolerance merge into one with the union label
  (handles overlapping/parallel steps).
- **Metric:** G-Mean F1 `= √(Interrupt-F1 × Silent-F1)`; degenerate always-silent /
  always-speak → 0. Expected scale: ~1.6 k positives + negatives at a declared fixed ratio.
- This is the headline VLM-vs-sensor-graph stage (frontier models score low here elsewhere).

### FA-3 — Timing-quality diagnostics (secondary, free once windows exist)

- **STS** `= exp(−(t̂−s)/(e−s))` per TP — earliness within the window (this is how
  "preventiveness" is scored, since no preventive window is GT).
- **Step-completion delay τ** on the step-tracking substrate (Family E reporting).
- **Probe protocol (optional):** query at `s−5..s−2` (expect silent) and `s..s+3` (expect
  fire) — a fair non-streaming comparison point for offline VLMs at near-zero extra cost.

## Metrics summary

| What | Metric |
|---|---|
| Stage tracking | per-second coarse accuracy over `[0, last GT step end]`; fine where supported |
| Reminder decisions (FA-1) | per-class windowed P/R/F1; silence scored |
| Intervention timing (FA-2) | G-Mean F1 |
| Earliness (FA-3) | STS, τ |
| Cost | VLM calls, frames, total VLM latency, detector CPU-s; normalized per video-minute; real-time feasibility stated |

## Reporting rules

- Per-class, never pooled. Temperature reported as its own execution sub-row.
- Report the suspended-order fraction honestly (events that exist as CC4D facts but are not
  scored here) so coverage is not overstated.
- Declare the FA-2 negative-sampling ratio + rule in the release; report sampling-free
  FA-1 window-membership numbers alongside.
- Oracle-stage ablation (PWR-style): main number with self-tracked stages + an optional
  GT-stage upper bound, clearly labeled non-deployable.
