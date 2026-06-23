# BOX 3 — Proactive-Reminder Runtime (predictor; 2026-06-20)

How the runtime emits proactive reminders (T2). It is a **consumer** of the step-recognition
runtime: it adds no second state machine. The active-set FSM already computes, per tick,
which steps are running; the reminder layer reads that state plus the sensor bus and decides
what to warn about. See `PIPELINE_THREE_BOXES.md` (this is Box 3), `REMINDER_EVALUATION.md`
(Box 2 scores its output), and the active-set model in the step-recognition runtime.

**Firewall.** Every reminder a node can raise is authored from the **recipe text only**
(quantities, durations, named ingredients/techniques). Box-1 GT (CC4D tags, Qualcomm
timestamps) is used *only* to score this runtime's output and never to author checks or tune
thresholds per recording.

**Two axes, kept separate.** A reminder has a *kind* (the CC4D `subtype`: measurement,
technique, preparation, temperature, timing, missing_step, order) and a *sensor* that runs it
(the `detector`: `D1`..`D6`, `VLM`, or `none`). Cost follows the detector, not the kind: a
`none` reminder is free, `D6` is a clock, `VLM` is the expensive sensor. That cost split is
the sensor-control story — it is the same A-solve / B-trigger / C-none tiering from
`tasks/AUDIO_RUNTIME_LIBRARY.md`, not a new vocabulary. Detector names follow that library:
D1..D5 physical sensors, **D6** = `timer — duration & precondition logic`, `VLM` = expensive.

---

## 0. Recap of the state it rides on

From the step-recognition runtime, available every tick `t`:

- `completed` — nodes whose `completion` has fired (latency ≤ 10 s).
- `active` = { n : preconditions(n) ⊆ completed } − completed. **Size ≥ 1**; the fork
  (e.g. the 3 hot-chocolate adds) makes it genuinely multi-membered.
- For each active node, an **activation tick** `t_on(n)` and (once it leaves) a completion
  tick `t_off(n)` — the bounds of its active interval.
- `recognized(n)` — for a silent multi-member active set, which node the VLM poll says is
  happening now (consulted only when `|active| > 1` and the members are sensor-ambiguous).

The reminder layer **never recomputes** any of this. It subscribes.

---

## 1. The three reminder kinds, by sensor

### Precondition (`detector: none`) — FREE, read off the FSM
Covers `precondition_violation/{order, missing_step}` (the single largest chunk of the GT:
25 of ~76 events on spiced hot chocolate).

- **missing_step** — a node never executed whose transitive successor ran; emit anchored at
  the earliest executed successor's start (matches Box-1's derivation).
- **order** *(deferred — see §4)* — a node recognized active while its preconditions are not
  all complete; emit at the offending step's own start.

Cost: **zero marginal sensing** — warnings come out of state the system already maintains.
Detecting that a *silent* step ran requires the VLM poll we are already spending to
disambiguate the fork; on an audio-anchored step (microwave) the onset gives it outright.

### Timing (`detector: D6`) — CHEAP, a clock over an interval
Covers `parameter_violation/timing` (10 events; lands on the two microwave steps, both with
`duration_constraint_s: 60`).

- For a node carrying `duration_constraint_s`, time its active interval: **overtime** when it
  runs past the bound, **undertime** when it finishes short (where a minimum is meaningful).
- The bounds are inherited from whatever settled activation/completion. Microwave steps give
  **crisp** bounds (`cycle_start`/`cycle_end`), so their timing reminders are reliable. Silent
  steps give **fuzzy** bounds (VLM poll granularity) — do not author timing checks there
  unless the interval is sensor-clean.

Cost: **a clock, no VLM.**

### Execution (`detector: VLM`) — EXPENSIVE, ONE merged periodic call
Covers `execution_error/{measurement, technique, preparation}` (~36 events) and, weakly,
`execution_error/temperature`.

- Each node carries a `checks` list of recipe-derived quality questions (e.g. "how many
  chocolate pieces went in? flag if ≠ 2").
- **One call does both jobs.** The active foreground unit that needs the VLM — the C-none
  block, or any active step carrying a VLM check — gets a **single periodic call** (cadence +
  cost capped by `vlm_policy`) that (1) recognizes which candidate step is happening now and
  (2) evaluates *that step's* checks. Recognition and all of the step's checks ride one prompt;
  there is never a second call for the checks.
- **Speak-or-stay-silent.** The prompt frames the VLM as a proactive assistant: stay silent
  when the user is on track (the normal case, `reminders: []`); emit a reminder only on a
  clear, listed deviation for the step *actually underway*. This is also what prevents
  premature firing — a check is judged against the step the VLM itself names as in-progress,
  not at the first tick a step is guessed.
- A given `(node, subtype)` reminder is emitted at most once (dedup), so a persistent
  deviation polled every period is not repeated.

Cost: **VLM, but one duty-cycled call per period** — the checks add no calls beyond the
recognition poll the silent fork already pays for.

---

## 2. The duty-cycle this produces (spiced hot chocolate)

| Node | Checks carried | VLM? |
|---|---|---|
| fill_milk | preparation (skimmed milk) | yes |
| microwave_initial | timing 60 s (`D6`) | **no** |
| add_cinnamon / add_sugar / add_chocolate | measurement, identity | yes (rides fork poll) |
| mix | technique | yes |
| heat_serve | timing 60 s (`D6`) | **no** |

Precondition reminders run every tick at zero cost across all nodes. The VLM is asleep
through **both microwave runs** (audio + the D6 clock cover recognition *and* their timing
reminders) and awake only across the silent prep/adds/mix span — exactly where it is already
awake for step recognition. **~46% of reminders (precondition + timing = 35/76) cost no VLM.**

---

## 3. Output schema

Per emitted reminder:

```json
{ "t": 304.7, "node": "add_chocolate", "class": "execution_error",
  "subtype": "measurement", "detector": "VLM",
  "evidence": "3 pieces observed, recipe states 2", "message": "..." }
```

`subtype` is what Box 2 scores (it matches against `data/cc4d_family_a/` by class, subtype,
window). `detector` is what ran it — the cost split (free / D6 / VLM) is a groupby over this
field, not a stored label.

---

## 4. Latency / causality

All reminders obey the shared ≤ 10 s look-ahead buffer. Precondition fires at the recognition
tick of the offending step; timing at the constraint crossing; execution within the poll
latency of the active node.

---

## 5. Implementation status (eval/proposed_runtime.py, 2026-06-20)

Wired and generic over any sensorplan that follows the template (checks read from the node's
`checks` list; sensor chosen by each check's `detector` field):
- **Execution checks (`VLM`)** — `VLMArm.poll_and_check()` makes ONE merged periodic call per
  active VLM-needing foreground unit (the C-none block or any active step with a VLM check):
  it recognizes the step and evaluates that step's checks in a single prompt, emitting a
  reminder only for the deviations the VLM raises (deduped per `(node, subtype)`).
- **Timing (`D6`)** — `_timing_reminders()` runs each `detector:"D6"`, `reminder:"timing"`
  check via `TimerChecker` over the active interval vs `duration_constraint_s`.
- **missing_step (`none`)** — `_missing_step_reminders()` sweeps the original node DAG.
  **Observability gate:** a silent member is claimable as missing only when the VLM arm
  actually ran; with the VLM off it is unsensed, not skipped, so it is not flagged.

Output: a `reminders` array (§3 schema). `sensor_events` stays the raw released-detector log.

---

## 6. Open issues

- **order** — deferred (2026-06-20 decision): needs the FSM to react to a successor running
  before its predicate, the same machinery order detection wants.
- **temperature (microwave power level)** — "low instead of high". Box 1 now scores the whole
  class (the power-level subset anchored at step start), but there is no clean detector and the
  power dial is often off-screen, so VLM recall here is expected to be low — treat these checks
  as best-effort and report their recall separately.
- **node re-entry / loops** — recipes like dressed-up meatballs repeat microwave→stir. The
  timer must key on the **node instance**, not the step_id, or it double-counts.
- **fuzzy silent-step intervals** — timing on a silent step is unreliable; only author timing
  checks where the interval is sensor-bounded (here: the two microwave steps).
