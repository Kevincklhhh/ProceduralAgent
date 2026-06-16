# CC4D → Sensor-Stage Map Conversion, and the Evaluation Protocol (v1)

> 🗂️ **SUPERSEDED (2026-06-15) — split into the three boxes.** This doc bundled three things that are now separate; see `PIPELINE_THREE_BOXES.md` for the map. Its content now lives in: **Box 3 predictor** (Part 1, steps C1–C4/C6) → `SENSOR_GRAPH_COMPILER_PROMPT.md`; **Box 1 GT** (step C5, the truth table) → `FAMILY_A_CC4D_AUGMENTATION.md` + `eval/build_family_a_gt.py` (now mechanical-only — C5's order/temperature/preventive rows are suspended or reactive); **Box 2 evaluation** (Part 2) → `REMINDER_EVALUATION.md`. Kept for history (the activity-8 worked example); do not treat C1–C6 as one pipeline.

> ⛔ **TWO SEPARATE PIPELINES — do not let them share per-recording inputs (firewall, added 2026-06-14).** This doc historically described C1–C6 as one "conversion," but C2–C4 and C5 belong to **different pipelines that must be built independently**:
> - **PREDICTOR (sensor-stage map):** C1, C2–C4, C6 — built from the **recipe alone** (DAG + step text). This is the system's plan; see `SENSOR_GRAPH_COMPILER_PROMPT.md`. It anticipates the error space from structure and must **never** see the error tags, Qualcomm timestamps, or execution traces.
> - **GT (reminder truth table / answer key):** C5 — built from the **CC4D + Qualcomm annotations**; see `FAMILY_A_CC4D_AUGMENTATION.md`. This is what the predictor is scored against.
> - **They share exactly one thing: the recipe (DAG + step text).** Whoever/whatever authors the sensor map must not have seen C5's truth table for the eval recordings, or the scores are leaked. C1 (adopt graph) and C6 (stage map) are shared recipe structure; C5 is the firewall's far side. The `[mech]/[judg]` labels below describe *effort*, not which pipeline a step belongs to — read the pipeline tag too.

Two things are specified here, exactly as exercised on activity 8 (Spiced Hot Chocolate) in `experiments/replay_v1/`:

1. **The conversion**: how a CC4D recipe becomes (a) our executable task JSON ("sensor-stage map", **PREDICTOR** — recipe-only) and, *separately*, (b) the per-recording reminder truth table used for scoring (**GT** — annotation-derived). Each step below is marked **[mech]**/**[judg]** (effort) and belongs to one pipeline; never feed (b) into (a).
2. **The frozen evaluation protocol** all arms are scored under, so larger-scale runs are comparable.

---

## Part 1 — Conversion pipeline

### Inputs (all from `data/cc4d/annotations/`)

| Input | Content | Used for |
| --- | --- | --- |
| `task_graphs/{recipe}.json` | `{"steps": {node_id: text}, "edges": [[from,to]]}` — the DAG | graph structure, preconditions |
| step texts (inside the graph) | parameters in natural language: "for 1 minute", "2 pieces", "1/5 teaspoon" | timer bounds, expensive criteria |
| `annotation_json/complete_step_annotations.json` | per recording: `{step_id, start_time, end_time, has_errors}`; skipped = `-1` | stage-accuracy GT; truth-table support |
| `annotation_json/error_annotations.json` | per recording per step: error tags ∈ {Timing, Missing Step, Order, Temperature, Measurement, Preparation, Technique, Other} + free text | reminder truth table |

### Step C1 — Adopt the graph [mech]

Take CC4D nodes/edges verbatim. Rename numeric node ids to readable `step_id`s (88→`fill_milk`), keep the mapping in `cc4d_step_ids`. Edges become `preconditions` (e.g. `mix` requires all three adds). Nothing is added or removed — the DAG is the dataset's, not ours.

### Step C2 — Classify each step's completion event [judg]

For each step, assign the primary observable completion event using the fixed rubric (event classes: `audio_appliance / audio_transient / audio_texture / rgb_transfer / rgb_motion_periodic / rgb_state_change / logic_timer / vlm_only`). Rule: classify by the cheapest signal that confirms the step **happened**; quantity/identity checks split off (→ C3). This is the main judgment step; the detector-primitive library (`docs/DETECTOR_FEASIBILITY.md` §6, 8 primitives) makes it LLM-automatable later because the answer space is small and closed.

### Step C3 — Bind detectors + tiers [judg, parameter extraction mech]

Each completion event binds to a primitive with parameters:
- class → primitive: `audio_appliance` → `sustained_band(profile)`; `audio_transient(beep)` → `tonal_burst`; stir → `transient_train(clink)`; etc.
- **timer bounds parsed from step text [mech]**: "for 1 minute" → `duration ∈ 60s ± 20s` (tolerance ±33% is a fixed convention, declared once).
- tier: cheap = audio/timer/logic, medium = RGB, expensive = VLM. Unverifiable-cheaply attributes ("2 pieces", "skimmed milk", power level) become `expensive_criteria` with an escalation action.

### Step C4 — Synthesize reminders [judg over a closed template set]

Reminders come from exactly three template families (no free invention):
1. **Parameter violations** [mech once C3 done]: timed step → `overtime_*` (fires DURING the run at upper bound) and `undertime_*` (at run end) warnings.
2. **Graph-precondition violations** [mech]: successor stage starts while a precondition has no completion event → `missing_<precondition>` reminder (e.g. `missing_ingredient_before_mix`, `missing_mix_before_heat`). When the precondition is not cheaply verifiable (ingredient identity), the trigger becomes an **escalation request** instead of a direct reminder.
3. **Safety templates** [judg, small fixed list]: post-heat caution (`hot_mug_caution`), unattended-done (`microwave_done_prompt`).

### Step C5 — Derive the reminder truth table [mech — must stay mechanical for credibility]

Per recording, from `error_annotations` tags ONLY (never from any system output — circularity guard):

| CC4D evidence | Expected reminder |
| --- | --- |
| Timing Error on a timed step, actual > upper bound | `overtime_*` |
| Timing Error on a timed step, actual < lower bound | `undertime_*` |
| Missing Step on ingredient X **and** mix executed | `missing_ingredient_before_mix` (X) |
| Missing Step on mix **and** heat executed | `missing_mix_before_heat` |
| clean recording | none — silence is scored |
| Measurement / Preparation / Technique / Temperature errors | NOT mapped to reminders in v1 (out of scope; listed as future ids) |

**Detectability-floor exclusions are declared a priori, per rule not per result**: e.g. runs shorter than the 20 s hum floor are excluded from timing scoring (8_26's split 8 s heat) and reported in text. Order Errors are not scored as reminders in v1 (8_50's early sugar makes "sugar missing" a false alarm — the truth table encodes that).

### Step C6 — Stage map for scoring [mech]

`step_id → coarse stage`: parallel, audio-indistinguishable steps collapse (the three adds → `adds`); everything else 1:1; unlabeled time = `other`. Fine (7-way) scoring is additionally reported for arms that can distinguish (VLM arm).

### What hand-conversion cost, and what scaling needs

For activity 8: C1/C5/C6 ≈ scriptable in an afternoon (and **should be scripted before scaling** — C5 especially, since a mechanical truth-table extractor is also the credibility argument). C2–C4 took ~1–2 hours of judgment for 7 steps; at ~15 steps/recipe, hand-conversion of the next 4 recipes is ~a day total — acceptable for v2 scale; the LLM compiler (recipe text → task JSON) is the deferred research deliverable, and these hand conversions become its gold references.

---

## Part 2 — Evaluation protocol (frozen, v1)

1. **Replay setting**: offline causal replay of recorded RGB+audio. Systems may only use sensor data up to time t plus a declared lookahead (detector smoothing; ≤ 7.2 s in v1, reported). GT (stages + truth table) is fully derived from CC4D annotations before any arm runs.
2. **Fairness**: every arm receives the same task JSON (same procedure knowledge); arms differ only in sensing policy (event-driven detectors / periodic VLM / detectors + targeted escalation).
3. **Tuning discipline**: detector parameters are tuned on one designated clean recording per recipe (8_16 for activity 8), frozen, and evaluated on all others. Tuning recordings are reported separately at scale. Any structural choice made after seeing eval data is recorded as a design-leakage disclosure (see `detectors/probes/results_hum_beep.json` note).
4. **Unified result format** per recording per arm: `{stage_intervals, events[{t, type, id, message}], escalation_requests, cost{vlm_calls, frames_sent, vlm_latency_total_s, compute_s}}`.
5. **Metrics**:
   - *Stage tracking*: per-second coarse accuracy over `[0, last GT step end]` (and excluding GT-`other` seconds); fine accuracy where supported; anchor boundary deltas as descriptive timing.
   - *Reminder decisions*: precision/recall over the scored ids vs the truth table; one TP per expected (recording, id) pair; any fire of a scored id on a recording where it is not expected = FP (silence on clean recordings is therefore scored); fire timestamp must fall in the relevant GT step window ± 15 s. Descriptive-only ids (`hot_mug_caution`, `microwave_done_prompt`) reported but excluded from P/R.
   - *Cost*: VLM calls, frames, total VLM latency, detector CPU seconds; normalized per video-minute; real-time feasibility stated (a policy whose per-decision latency exceeds its decision interval is marked not-live-runnable).
6. **Scoring implementation**: `eval/score.py` (single referee for all arms) → `experiments/replay_v1/results/scores.json` + `REPORT.md`.

### Scale-up sequence implied by this protocol

1. Script C1/C5/C6 (mechanical extractor: CC4D annotations → stage GT + truth table for ANY recipe).
2. Activity 8, all 16 recordings (10 with errors; we used 6) — same task JSON, zero new judgment.
3. Microwave family (9 more recipes) — hand C2–C4 (~a day), validated primitives reused verbatim.
4. Gate: `texture_dynamics` (sizzle) probe on one stovetop recording → unlocks the 8 stovetop recipes.
5. Cold-assembly recipes deferred until the medium RGB tier exists.
