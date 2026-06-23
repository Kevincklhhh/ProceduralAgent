# Family A on CaptainCook4D: Annotation Review + Augmentation Design (2026-06-12; rev 2026-06-13: GT-backing audit; rev 2026-06-15: mechanical-only cut)

> 📦 **This is BOX 1 — Reminder GT generation.** Top-level map: `PIPELINE_THREE_BOXES.md`. Evaluation lives in `REMINDER_EVALUATION.md` (Box 2); the predictor in `tasks/PROCEDURE_MONITOR_COMPILER.md` + `REMINDER_RUNTIME.md` (Box 3).

> ✂️ **MECHANICAL-ONLY CUT (2026-06-15; temperature folded in 2026-06-21).** Scored GT is now exactly what `eval/gt_build_family_a.py` derives mechanically — window start from a Qualcomm timestamp or a step/DAG-structural rule, nothing needing a human call. **2,409 scored events.**
>
> 🔁 **ORDER UN-SUSPENDED (2026-06-15).** The order class is now **scored** — one event per CC4D Order-tagged step (**789 events**, window start = the out-of-order step's `start_time`). The benign/harmful adjudication is **deleted**: the CC4D Order tag *is* the ground truth and we do not override it (the old 8_50/8_45 "harmless early sugar" rationale was an illegitimate override of the annotation). Each event carries a `dag_edge_violation` diagnostic (52% caught by a real DAG edge / 48% only by CC4D's stricter sequence) — reporting detector recall, *not* gating scoring. The §4 quirk-5 / §4.1 adjudication discussion below is **superseded** and kept only as history.
>
> 🌡️ **TEMPERATURE FULLY SCORED (2026-06-21).** The power-level subset (13 — Temperature tags with no Qualcomm ts) is **no longer suspended**: it scores with a step-start anchor (window `[step.start, step.end+grace]`, flagged `low_confidence_temperature`), since a wrong power level is in effect from step start. **There are no suspended classes.** Safety and next-step guidance remain excluded as before.

> ⛔ **This is the GT pipeline (the answer key) — firewalled from the predictor (added 2026-06-14).** Everything here derives the reminder *truth table* from CC4D + Qualcomm annotations. It must stay **separate from the sensor-map / predictor pipeline** (the Box-3 predictor; see `PIPELINE_THREE_BOXES.md`): the predictor plans from the **recipe (DAG + step text) alone** and must never see the error tags, Qualcomm timestamps, or execution traces used below. The two pipelines share only the recipe and meet only at scoring time. Leaking any GT here into the predictor invalidates every score.

> **Governing principle (rev 2026-06-13): a reminder class is admissible only if its events derive from existing annotation (CC4D tag and/or Qualcomm timestamp).** A class we would have to author from scratch — inventing both the events and the rubric, then scoring ourselves against our own definition — is not ground truth and is excluded. This is the same circularity guard the conversion protocol applies (§4, "never from any system output"); here it is applied to the *taxonomy* itself. Consequence: **Safety/unattended is cut** (no CC4D or Qualcomm annotation exists for it — see §3.0), and the scored taxonomy is **three classes**, not four.

Goal: define **When-to-intervene (proactive timing)** tasks — Family A of `ASSISTANCE_TASK_LANDSCAPE.md` — on CC4D, which no published work has done (landscape §3: PWR explicitly rejected CC4D; Pro2Assist/ProAssist/WTaG/YETI all evaluate elsewhere; LiveMamba's layer is mistake-feedback GT, not intervention-timing decision points). End state: CC4D becomes the first **audio-bearing** dataset with proactive-reminder timing GT. This doc reviews what CC4D's annotations actually provide, what is missing for Family A, and specifies the augmentation layer. Extends `CONVERSION_AND_EVAL_PROTOCOL.md` (v1, activity 8) to all 384 recordings.

## 1. What CC4D's annotation structure provides (verified on disk, `data/cc4d/annotations/`)

| Primitive | File | Content | Family-A use |
|---|---|---|---|
| Step segments | `annotation_json/complete_step_annotations.json` | 384 recordings × steps: `{step_id, start_time, end_time, description, has_errors}`; 5,700 step instances | step-completion timestamps = anchor for every window/decision point |
| Error tags | `annotation_json/error_annotations.json` | per step: `errors: [{tag, description}]`, tag ∈ {Order 795, Technique 502, Preparation 410, Measurement 331, Missing Step 285, Timing 177, Temperature 66, Other 8} = 2,574 instances in 220 error recordings (164 clean) | which recordings owe which reminder (truth-table source) |
| Task graphs | `task_graphs/{recipe}.json` (24) | `{"steps": {node_id: text}, "edges": [[from, to]]}` DAG; step text carries parameters ("for 1 minute", "2 tbsp") | preconditions for order/missing reminders; timer bounds |
| Splits | `data_splits/` | environment/person/recipe/recording × {combined, normal-only} | normal-only splits = one-class training discipline (M3) |
| Metadata | `metadata/video_information.csv` | durations etc. | normalization |
| **Audio** | GoPro MP4s + HoloLens mic (videos, not in annotations repo) | unannotated | our unclaimed modality (landscape §3) |

Plus the local **Qualcomm layer** (`data/qualcomm_interactive_cooking/`, reviewed in landscape §5–5.1): timestamped typed messages over all 384 recordings — 5,263 `instruction` (step-completion times), 4,067 success confirmations, 1,388 typed mistake events with **sub-step visibility timestamps** (median 8 s before step end), `remaining_plan` state, machine-readable recipe DAGs, standardized splits (213/62/109).

### Structure quirks that constrain the design (measured this session)

1. **Skipped steps have `start_time = end_time = -1`** (287 of 5,700 instances). A Missing-Step reminder has no native timestamp; its fire-time must be *derived* — the start of the first executed step whose DAG preconditions include the skipped step. Mechanical, but graph-dependent.
2. **9% of consecutive step segments overlap** (453 of 5,029 gaps negative — multitasking, e.g. prep during microwave runs). "Intervene at the step boundary" must tolerate overlapping/parallel execution; boundary events from different steps can collide in time.
3. **Inter-step gaps are tight**: median 4.7 s, p75 11.6 s. A flat ±15 s matching tolerance spans multiple boundaries → next-step-guidance scoring must be **window-membership** (TP iff t̂ ∈ [s,e] with e clipped at the next step's start + grace), not fixed-radius matching. Fixed radius (±15/30 s) is kept only for sparse mistake events where it is unambiguous (and cross-paper comparable, M1).
4. **Repeated step ids** occur within a recording (`recording_id_step_idx.json` repeats ids in execution order); node references need instance suffixes (`_1/_2`, as Qualcomm's `advanced_planning` does).
5. **Order Error descriptions are templated**: 98% contain "before"/"after"; only **304 unique strings** cover all 795 instances (top string covers 87). Benign-vs-harmful adjudication (the 8_50 problem, v1 §C5) is therefore a **per-cluster** judgment pass over ~304 strings, not 795 case-by-case calls.
6. Error tags are **step-level, untimestamped** — CC4D alone says *which step* erred, not *when it became visible*. The Qualcomm layer fills this for execution errors only (97% of its events re-attach CC4D tags + add timestamps); order/missing remain ours to time (via the DAG derivation in (1) and the out-of-order step's own start_time).

## 2. The gap, precisely

Family A presupposes step-tracking GT, an error inventory, graphs, and splits — all already on disk, which is why CC4D is the right host. What it lacks splits cleanly into **derivable** gaps (existing annotation + DAG + a declared convention produces the label) and a **non-derivable** gap (no annotation exists at all):

| Needed | Status on CC4D | Verdict |
|---|---|---|
| Decision points (PWR-style query times with interrupt/silent GT) | absent, but **derivable** from step segments + the truth table | build it (§4) |
| Intervention validity windows [s,e] per due reminder (Pro2Assist-style) | absent, but **derivable** (v1 built them by hand for activity 8) | build it (§4) |
| Silence GT (clean recordings scored for not-speaking) | **implicit** in the 164 clean recordings; just needs formalizing | build it (§4) |
| Safety/unattended events | **no annotation anywhere** — absent from both CC4D and Qualcomm taxonomies (landscape §5.1) | **EXCLUDE** (§3.0) |

The distinction is the whole point: decision points / windows / silence are *re-expressions of labels that already exist* (a step boundary, an error tag, a clean recording). Safety would be a *new label with no source* — we would define what counts as unsafe, sweep for it ourselves, and then have nothing independent to check our sweep against. That is not ground truth.

## 3. Proposed task suite: three tasks, one GT layer

One annotation layer (§4) feeds three evaluation protocols, matching the metric kit M1–M2 (landscape §4) plus diagnostics. All tasks are **causal**: input is RGB+audio (+task JSON) up to t. All are **modality-neutral by construction** — GT derives from CC4D/Qualcomm annotations and the DAG, never from what our detectors can see (anti-bias rules, landscape §5.1).

### 3.0 What is scored, and what is excluded

The scored taxonomy is **three GT-backed reminder classes**, graded by how much of each label is inherited vs. derived vs. judged:

| Class | Fact GT (did it happen?) | Timing GT (when visible?) | Reminder-worthiness | Net grade |
|---|---|---|---|---|
| **Precondition violation** (missing step, order error) | CC4D Missing-Step (285) / Order (795) tags | **derived** from DAG + step starts | inherited for missing; **judged** for order (benign/harmful, §4 quirk 5) | strong fact; order needs a judgment layer |
| **Parameter violation** (timing) | CC4D Timing tag (177; 90% cite a concrete duration) | **derived** (bound crossing from step text) | inherited | strong — cleanest preventive class |
| **Execution error** (technique, preparation, measurement, **temperature**) | CC4D tags (1,170 + 66 temp) | **Qualcomm visibility timestamps** (external) | inherited | strongest — both fact and timing are externally annotated |

**Excluded — Safety/unattended.** No CC4D tag, no Qualcomm event, nothing to derive from. Building it means authoring the events *and* the rubric and self-scoring — excluded from the benchmark per the governing principle. (It remains a legitimate *deployment* behavior — a fielded assistant should still warn about an unattended microwave — but it cannot be a *scored task* here. If we ever want it scored, it needs an independent annotation effort with its own inter-annotator agreement, treated as a separate contribution, not folded into this layer.)

**Re-homed — Temperature (66).** Previously slotted under "parameter violation (preventive)". 76% are discrete power-level settings ("low instead of high") with no time-bound to cross, so there is no preventive window — it is right-or-wrong from step start and only *reactively* observable. It keeps its CC4D tag (so it stays GT-backed) but moves into the **execution-error** class, scored reactively at the Qualcomm/step-window timestamp, not as a preventive parameter bound.

**Demoted — Next-step guidance.** Not a reminder class (its timing GT ≡ step-completion GT); see FA-1 note below.

### FA-1. Typed Streaming Reminder Triggering (primary; metric M1)

*The v1 protocol scaled up.* System watches the stream and emits typed, timestamped reminders; silence is default.

- **Output**: events `{t, reminder_class, reminder_id}`; **three scored classes**: precondition violation / parameter violation / execution error (§3.0).
- **Scoring**: per-class windowed P/R/F1 against the truth table; one TP per expected (recording, event); any scored-id fire not owed = FP; **clean recordings count** (silence scored). Windows per §4; report ±15 s (ours) and 30 s (LiveMamba/Ego-MC comparable) variants for the sparse classes.
- **Per-class always, pooled never** (84.3% of mistake events are visual-leaning — pooled numbers would hide or flatter the audio arm; landscape §5.1). Within execution error, report temperature as its own sub-row (it is the audio-leaning slice of that class).
- **Next-step guidance is NOT a Family A class (demoted 2026-06-12).** Its timing GT is definitionally identical to step-completion GT, so scoring it as "when-to-intervene" just re-measures step recognition — that's Family E substrate, already reported as stage accuracy + completion delay τ. Its 5,263 events would also drown the ~2–3 k genuine intervention events in any pooled number. The completion≠execution residue (a step done wrong must not trigger "next step", IndustReal) is covered by the execution-error class. Retained uses: (a) step completions become FA-2 **hard negatives** — know the step ended, still stay silent; (b) one appendix table under Qualcomm's own IC-Acc protocol for LiveMamba comparability (their task, their published 23.1–31.5%; no claim it is ours).

### FA-2. Intervention Decision at Decision Points (metric M2, G-Mean F1)

*PWR's task shape, minted on CC4D.* At each given timestamp t, output `interrupt`/`silent` given stream up to t + task JSON.

- **Positive points**: one per truth-table event, placed at window start s (earliest moment the evidence exists — Qualcomm visibility timestamp for execution errors; derived trigger time for order/missing/parameter).
- **Negative points**: (a) every step completion where nothing is owed (the hard negatives — something *happened*, but no intervention is due; this is where over-talkative VLMs fail, WTaG/PWR); (b) mid-step samples ≥ tolerance from any window; (c) all of the above on the 164 clean recordings.
- **Collision rule**: points from different events/boundaries closer than the tolerance merge into one point with the union label (handles quirk (2)/(3)).
- **Metric**: G-Mean F1 = √(Interrupt-F1 × Silent-F1) — degenerate always-silent/always-speak → 0. Expected scale: ~2–3 k positives (§5) + negatives sampled at a declared fixed ratio (e.g. 3:1) → ~10 k points, same order as PWR's EgoProactive (9,935).
- This is the headline VLM-vs-sensor-graph stage: frontier models score ≤ .08–.51 on this metric elsewhere (landscape §0.3).

### FA-3. Timing-quality diagnostics (secondary, free once windows exist)

- **STS** = exp(−(t̂−s)/(e−s)) per TP (Pro2Assist) — earliness within the window.
- **Step-completion delay τ** (IndustReal) on the step-tracking substrate (Family E reporting, not a reminder class).
- Optional **probe protocol** (OmniPro): query the system at s−5..s−2 (expect silent) and s..s+3 (expect fire) — gives offline VLMs a fair non-streaming comparison point at near-zero extra annotation cost (points derive from the same windows).

## 4. The GT layer: derivation per reminder class

Artifact: `data/cc4d_family_a/{recording_id}.json`:

```json
{
  "recording_id": "8_50",
  "events": [
    {"event_id": "8_50_e3", "class": "precondition_violation",
     "reminder_id": "missing_mix_before_heat", "window": [412.3, 442.3],
     "anchor": {"step_instance": "heat_serve_1", "anchor_type": "step_start"},
     "source": "cc4d_order_tag+dag", "adjudication": "harmful", "notes": ""}
  ],
  "decision_points": [
    {"t": 412.3, "label": "interrupt", "event_id": "8_50_e3"},
    {"t": 250.0, "label": "silent", "event_id": null, "kind": "step_completion_negative"}
  ]
}
```

Derivation rules (extends C5; [mech]/[judg] discipline as in the conversion protocol):

| Class | Window [s, e] | Source | Status |
|---|---|---|---|
| 1. Next-step guidance *(demoted — not Family-A-scored, see §3 FA-1)* | s = step_i completion (CC4D end_time, cross-checked vs Qualcomm instruction t); e = next step start + grace, clipped (quirk 3) | step segments / Qualcomm | **[mech]** — still extracted: feeds FA-2 hard negatives, Family E cross-check, IC-Acc appendix |
| 2a. Missing step | s = start of first executed DAG-successor of the skipped step; e = s + grace | CC4D Missing-Step tag (285) + DAG replay | **[mech]** (quirk 1 rule) |
| 2b. Order violation | s = out-of-order step's start_time; e = step end (PNR-capped if irreversible) | CC4D Order tag (795) + DAG | **[judg]** per-cluster: ~304 unique descriptions adjudicated benign/harmful once, then applied mechanically |
| 3. Parameter violation (timing, **reactive** — see §4.2 rev) | s = Qualcomm reactive timestamp (91%); else step.end_time, low-confidence | CC4D Timing tag (177) + Qualcomm timing events (161) | **[mech]**; preventive ±33% window dropped (not GT-derivable, §4.2) |
| 4. Execution error (technique, preparation, measurement, **+ temperature**) | s = Qualcomm visibility timestamp; e = step end + grace | Qualcomm mistake events (1,170) + CC4D Temperature tag (66, reactive — re-homed from class 3, §3.0) | **[mech]** (Qualcomm timestamps adopted as-is); **[judg]** review of the 136 CC4D tags Qualcomm dropped (keep/drop rule documented); temperature events have no Qualcomm timestamp for the power-level subset → anchor at step start, **[judg]** |
| ~~5. Safety/unattended~~ | **EXCLUDED — no GT (§3.0)** | — | — |

Circularity guard unchanged: every window derives from CC4D/Qualcomm annotations + DAG + declared conventions — never from detector output. Audio event inventories stay calibration-only.

### 4.1 What is still missing to produce the labels (critical path)

Most of the layer is mechanical, but three pieces block a complete release and none is fully solved by existing annotation:

1. **Order-error benign/harmful adjudication — the binding constraint.** 795 events have a CC4D *fact* tag but no reminder-worthiness label; without the per-cluster pass (304 strings) none of them can become a precondition reminder, and the 8_50 case proves a naive "every order violation → reminder" policy emits false alarms. This is the single largest gap between "annotations exist" and "labels exist".
2. **Window-end / deadline semantics.** Window *start* is well-defined for every class; window *end* is not. For order/missing it is the point-of-no-return (after which a reminder is useless), which CC4D does not annotate — currently approximated as `step end + grace`. The PNR concept (MATT, landscape B12) is the principled target but must be set by convention per cluster, declared once.
3. **Temperature power-level timing.** The 50/66 power-level events ("low instead of high") have neither a Qualcomm timestamp nor a derivable window — anchored at step start as a reactive event over the whole active interval, flagged `low_confidence_temperature` in the release. **Scored**, not suspended (2026-06-21).
4. **No preventive/anticipatory windows are GT.** Resolved 2026-06-14: the only annotation-grounded timing we have is *reactive* (when the error became visible — Qualcomm timestamps, or step boundaries). A genuinely *preventive* reminder ("warn before it goes wrong") would need either an invented tolerance constant (the dropped ±33%) or a run-start the data doesn't annotate. Preventiveness is therefore scored as **earliness within a reactive window** (STS), not as a separate anticipatory window. This is a real scope limit of CC4D+Qualcomm worth stating plainly in the paper.

Everything else (missing-step derivation, timing bounds, execution-error import, decision-point minting, silence GT) is mechanical and reproducible from existing annotation.

### 4.2 Mechanical derivation algorithms (IMPLEMENTED: `eval/gt_build_family_a.py` → `data/cc4d_family_a/`)

**Status: built and run over all 384 recordings (2026-06-15; temperature folded in 2026-06-21).** The extractor below is implemented in `eval/gt_build_family_a.py`; it emits one `data/cc4d_family_a/{recording_id}.json` per recording (schema = §4 events + FA-2 decision_points) plus `_summary.json`. Corpus output: **2,409 scored events** — precondition {order 789 (one per CC4D Order-tagged step; 52% break a real DAG edge), missing_step 237}, execution_error {technique 453, preparation 360, measurement 325, temperature 67 (54 Qualcomm-timed + 13 power-level via step-start anchor)}, parameter/timing 178; **no suspended classes**; **171 recordings with no scored event** (164 clean + 7 all-dropped/unmapped); 45 missing-step events dropped as undecidable. Validated: reproduces the 8_45 worked example and agrees with the hand-built activity-8 `TRUTH` in `eval/eval_score_activity8.py` on clean recordings (8_16/8_3/8_25 → empty) and on timing+missing for 8_26/8_31; the one divergence (8_50: emits order, not a 2nd missing) is the intended order-vs-missing reclassification. Run `python eval/gt_build_family_a.py --only 8_45` to inspect a single recording.



Notation: a CC4D recording `rid` has `steps = [{step_id, start_time, end_time, description, has_errors}]` (skipped step ⇒ `start_time = end_time = -1`), an error record `errors[step_id] = [{tag, description}]`, an executed-order list `recording_id_step_idx[rid]` (step ids in execution order, repeats kept), and a recipe DAG `(nodes={node_id: text}, edges=[[a,b]])`. Map a global `step_id` to a graph `node` by exact step-text match (`step_idx_description.json` ↔ graph `steps`); instance-suffix repeated nodes. The Qualcomm row for `rid` (join key `video_id == recording_id`, **verified exact**) has parallel arrays `output_timestamps / output_texts / output_types`.

**Class 4 — Execution error (FULLY MECHANICAL; verified 97.6%).**
For each Qualcomm event with `output_types[i]` matching `feedback_action_aligned_mistake_{cat}_error`, `cat ∈ {technique, preparation, measurement, timing, temperature}`:
1. `t = output_timestamps[i]` is the visibility timestamp (window start `s`).
2. Find the CC4D step whose `[start_time-δ, end_time+δ]` contains `t` (δ = 10 s); confirm that step carries the matching CC4D tag (`technique→Technique Error`, etc.).
3. Window `[t, step.end_time + grace]`; emit `{class: execution_error, subtype: cat, window, anchor: step}`.
Measured: **1,355 / 1,388 mistake events (97.6%)** land in a CC4D step that carries the matching tag. The 33 misses (subtle/borderline + CC4D tag noise) plus the 136 CC4D execution tags Qualcomm dropped are the only `[judg]` residue (documented keep/drop). Temperature joins here too; its 50 power-level events have no Qualcomm timestamp → anchor at `step.start_time` over the whole active interval, flagged `low_confidence_temperature` (scored, not suspended).

**Class 3 — Parameter violation / timing (REACTIVE, annotation-grounded; revised 2026-06-14).**
Fact = the CC4D `Timing Error` tag (lookup). **Window = the Qualcomm reactive timestamp**, identical machinery to the execution-error class — verified **161/177 (91%)** of Timing tags have a matching Qualcomm `timing` event in-window. So timing is scored reactively, not as a manufactured preventive window.

*Why the preventive `step.start + bound × 1.33` window was dropped:* it is not cleanly GT-derivable. (a) The ±33% tolerance is **our own invented convention** (`CONVERSION_AND_EVAL_PROTOCOL.md` C3), with no basis in CC4D or the literature. (b) The true "overtime" instant is measured from the **appliance-run start**, which CC4D does not annotate — `step.start_time` includes pre-run setup, and the only thing that knows the real run-start is a *detector*, so using it to define GT would break the circularity guard. The "fire early, before the step ends" goal therefore is **not** a separate window; it is **earliness credit inside the reactive window** (the STS diagnostic, FA-3). The ±33% constant, if used at all, lives only at replay as one of *our detector's* parameters, never as ground truth.

The 16/177 timing tags without a Qualcomm timestamp anchor at `step.end_time` (the latest the error is certainly visible), flagged low-confidence-timing. Do **not** use CC4D segment length as the fact — `end - start` is the whole-step span, not the appliance run (confirms a crossing only 69% of the time).

*Implication (open):* with timing scored reactively and temperature already re-homed (§3.0), "parameter violation" no longer has a distinct *derivation* from execution error — both are reactive Qualcomm-timestamp events. It may collapse to a **subtype label** under execution error rather than a separate scored class. Flagged for decision; kept as its own row for now so per-class P/R stays visible.

**Class 2a — Missing step (MECHANICAL; verified 88% derivable, 12% dropped).**
Fact = CC4D `Missing Step` tag (**98% (280/285) of tags are on a `t=-1` step**, confirming "skipped"). Timing:
1. `n = node(skipped step_id)`; `successors = {b : (n,b) ∈ edges}`.
2. Fire time `s = min start_time over executed successors of n` (the moment a step that needed the skipped one begins). Window `[s, s + grace]`.
3. Use **transitive** successors (a step is owed when ANY dependent runs, not just a direct child); earliest executed dependent = tightest window. **84% (237/282 mappable)** derivable as implemented; the rest (skipped step terminal, or all dependents also skipped/unmapped) are **undecidable → dropped, declared per rule** (45 corpus-wide). (Direct-successor-only gives 66%; transitive is both higher-coverage and the correct "owed" semantics.)

**Class 2b — Order violation (PARTIALLY MECHANICAL — this is why order is the `[judg]` class).**
Fact = CC4D `Order Error` tag; the out-of-order step **was executed, so it has a real `start_time`** ⇒ window start is mechanical (`s = step.start_time`).

**Granularity caveat (added 2026-06-14): order tags must be grouped into deviations, not scored per step.** The 795 figure is tag-INSTANCES (≈ 794 erroneous steps, ~1 tag/step); they cluster heavily — **mean 6.8 order-tagged steps per affected recording (117 recordings, max 22)**. A single reorder touches many steps symmetrically: in 8_45, steps 90/84/87 ("before microwaving") + step 89 ("after adding cinnamon and sugar") are the *same* adds↔microwave swap described from both sides = **one** ordering deviation, hence at most **one** reminder. So before adjudication, **cluster co-referential order-tagged steps** (same pivot step, parsed from the templated "before/after X" text) into one deviation event; adjudicate and score the *deviation*, not the tagged step. Net: the true candidate-reminder count is far below 795 (~1–2 deviations/recording). (Missing-step and execution errors do **not** collapse this way — distinct skips / distinct faults are distinct reminders.)

What is **not** mechanical:
- *Which precondition it violated* (needed for the message): only **55% (436/795)** of Order tags coincide with a DAG-edge violation. The CC4D annotators' "correct order" is a stricter canonical sequence than the graph's partial order, so 45% violate an ordering the DAG marks as free/parallel. For those, the precondition must be parsed from the tag's templated free text ("performed before X" — 304 unique strings, §quirk 5), not read off the graph.
- *Reminder-worthiness*: a tagged order error need not warrant interruption (8_50 early-sugar is harmless to remind about). This is the benign/harmful adjudication — **not mechanical**, one call per cluster.

So the honest mechanical boundary per class: **execution error = end-to-end mechanical; parameter/timing = mechanical given a bound (50% auto-parsed); missing step = mechanical (88%); order = fact+time mechanical, message 55% graph / 45% text-parse, worthiness manual.**

## 5. Expected GT volume (from measured counts)

| Class | Events (approx.) |
|---|---|
| 1. Next-step guidance | 5,263 — not scored as reminders; FA-2 negatives + IC-Acc appendix only |
| 2a. Missing step | ≤ 285 (minus undecidable: skipped step with no executed successor) |
| 2b. Order violation | 795 tag-instances cluster to ~1–2 *deviations* per recording over 117 recordings (mean 6.8 tagged steps each); harmful subset of those *deviations* (adjudication determines) |
| 3. Parameter (preventive, timing only) | ~150–177 (timed steps whose recordings carry the Timing tag + parse) |
| 4. Execution error (tech/prep/meas + temperature) | 1,170 + 66 temp (+ reviewed subset of 136 dropped) |
| ~~5. Safety~~ | **excluded — no GT (§3.0)** |
| **FA-2 positives (scored classes 2–4)** | **~2–3 k** + negatives at fixed ratio |

## 6. Execution order

1. ✅ **DONE — `eval/gt_build_family_a.py`** (Qualcomm import; missing-step via transitive DAG; order grouping; decision-point minting) run over all 384 recordings → `data/cc4d_family_a/`. Discharges the v1 promise that C5 be scripted before scaling.
2. ✅ **DONE — validated** against the hand-built activity-8 `TRUTH`: agrees on clean recordings and timing/missing; the 8_50 order-vs-missing difference is a documented design change, not a bug.
3. **Order-error adjudication pass** (critical path, §4.1): cluster the 304 strings, one benign/harmful call per cluster with one-line rationale (the 8_50 early-sugar case is the calibration example). Until this is done, precondition-violation labels for order errors do not exist.
4. **Window-end convention** (§4.1): set per-cluster PNR/grace defaults, declared once.
5. **Freeze + adopt Qualcomm splits** (train 213 / val 62 / test 109) for comparability with LiveMamba.
6. Run the three arms (detectors / periodic VLM / detectors+escalation) + LiveMamba-protocol table.

*(No safety sweep — excluded, §3.0.)*

Then the audio claim writes itself: first Family A (when-to-intervene) benchmark on CC4D, first proactive-timing benchmark anywhere with non-speech audio available to systems, with end-to-end MLLM baselines already published on the same recordings (LiveMamba) and a metric (G-Mean F1) frontier VLMs demonstrably fail.

## 6.5 Firewall calibration against prior proactive-assistance work (verified from primary sources, 2026-06-14)

We checked what each prior method is **handed at inference** vs. what it must **predict**, to confirm our GT/predictor firewall matches field norms and to see where the line is drawn. Sources read in full: PWR (2606.04970), Pro2Assist (2605.04227), ProAssist (2506.05904); CC4D baselines from the dataset paper + our annotation review.

| Work | Recipe/plan given? | Step-state (where am I) | Decision timing | Error/reminder **content** given? | Oracle ablation? |
|---|---|---|---|---|---|
| **PWR** | yes — plan with completed/current/next tags + per-step visual cues | **predicted** (self-planned) in deployable setting; GT only in a separate oracle | query points **given** (fixed 2 fps grid; label predicted) | **no** — golden utterance is train target + judge ref only; OOP deviations even held out of training | **yes, "PWR-Oracle"**, labeled non-deployable: GT plan lifts G-Mean .57→.91, OOP recall 78.7→99.6% |
| **Pro2Assist** | yes — WikiHow step graph (per *task*, not per recording) | **predicted** (step + 4-way status); completed-step history is the model's *own* past predictions, starts empty | model **chooses** every moment; GT validity `[s,e]` used for **scoring only** | **no** — response is a prediction/distillation target | **no** — GT step/status are training supervision + scoring refs only |
| **ProAssist** | optional — recipe text in system prompt (helps; recommended config) | **predicted** from the stream; must even self-generate progress summaries | model **chooses** every frame; matching window is asymmetric (R<L) to block *future*-frame leakage | **no** — synthesized dialogues are scoring refs only | **no** — "knowledge-conditioned" (recipe) is framed as realistic RAG, not an oracle |
| **CC4D B1 supervised-ER** | step taxonomy | **GT step segment GIVEN** (trimmed clip = you're told which step) | offline per given segment | **no** — error label is the target | the given GT segment *is* an unlabeled privilege (bypasses step tracking) |
| **CC4D B2 zero-shot-ER** | step text + taxonomy questions (recipe-derived) | **GT step/clip GIVEN** | offline per given clip | **no** | same — step identity handed in |
| **Ours (intended)** | yes — recipe/DAG, the one shared input | **predicted** by detectors+graph (main); optional GT-stage oracle as upper bound | FA-2: grid **given** (PWR-style); FA-1: model chooses | **NO** — the firewall | optional, PWR-style, reported separately |

**What each must proactively PRODUCE (the steps are given as the question template, never as the answer):** PWR — interrupt/silent at each query point + a free-form utterance (instruction/correction/recovery) + out-of-plan detection. Pro2Assist — current step + 4-way status + binary trigger + response text, all predicted from video. ProAssist — speak/silent every frame + the full assistant utterance. **CC4D B1/B2 — neither is proactive**: they are offline per-segment error/normal classification given the GT clip, so CC4D defines *no* when-to-intervene task at all (the gap we fill; this is why our proactive metrics come from PWR/Pro2Assist, not CC4D). Ours — track stage from audio+RGB, decide when to fire, emit a *typed* reminder (templated, so we carry grounding + timing + reminder-type selection, minus the free-form language generation PWR/ProAssist also do).

The recurring shape: the recipe is shared knowledge, so "proactive" means **grounding** (where is the user, did they deviate — predicted, not given), **timing** (when to speak), and **delivery** (the intervention). Giving the steps does not trivialize the task because the deviation and its moment live in the specific recording.

**Three findings that calibrate our design:**

1. **Content-leakage is universally forbidden — we match the field exactly.** No method hands the predictor the mistake type, the reminder text, or which error will occur; these are always train targets / scoring references. PWR is strictest (holds out-of-plan deviations out of training entirely). Our firewall (recipe shared, error tags / Qualcomm timestamps / traces never fed to the predictor) is the same norm, not an unusually harsh one.

2. **Step-state is the one contested axis, and CC4D's *own* baselines are the leakiest of all.** B1/B2 hand the model the GT step segment — i.e. they bypass "where am I in the procedure," which every other work (and the literature consensus, PWR/Pro2Assist) identifies as the actual bottleneck. The modern deployable methods all *predict* step-state. **Our predictor tracks stages itself from detectors, so on this axis we are stricter than CC4D's native baselines** — a defensible contribution to state plainly ("unlike CC4D supervised/zero-shot ER, we do not assume GT step segmentation").

3. **An oracle-stage ablation is the field-standard way to expose step-state privilege honestly.** PWR isolates it as PWR-Oracle and quantifies the gap (the .57→.91 jump is *entirely* plan-conditioning, not perception). We should mirror this: report our main number with self-tracked stages, plus an optional GT-stage oracle upper bound, clearly labeled non-deployable — turning the step-state question into a measured result instead of a hidden assumption.

**One protocol refinement this surfaced (FA-2):** PWR's decision-point *grid* is GT-independent (every 0.5 s); only the *label* at each point comes from events. We must mint FA-2 negatives the same way — on a regular grid + all step boundaries — so the mere *presence* of a query carries no signal. If positives sat only at suspicious moments, a model could infer "I'm being asked, so something's due." Added to §4.1's decision-point rules.

## 7. Risks / open questions

- **License**: Qualcomm layer is research-only DLA — fine for experiments; re-releasing GT *derived from their timestamps* (class 4 windows) needs a check before publication. Precondition + parameter classes derive from CC4D (CC-BY-NC-ND style) + our own work; the temperature subset of execution error is CC4D-derived (anchored at step start, no Qualcomm dependency).
- **Order-adjudication subjectivity**: mitigate with per-cluster written rationales + a small double-annotation sample for agreement; benign clusters stay in the release marked `benign` (others can re-score with their own policy).
- **Decision-point negative sampling** is a design choice that moves the metric; declare ratio + sampling rule in the release, and report window-membership FA-1 (sampling-free) alongside.
- **Repeated steps / overlaps**: instance suffixes + the collision-merge rule must be in the extractor's unit tests (8_26, 10_47 are known tricky recordings).
