# Task-Graph → Sensor-Graph Compiler Prompt (v0, exploration stage; 2026-06-12)

> 📦 **This is BOX 3 — Recipe → sensor mapping (the predictor).** Top-level map: `PIPELINE_THREE_BOXES.md`. Recipe-only by the firewall: it never sees Box-1 GT (error tags, Qualcomm timestamps, traces). It is scored in Box 2 (`REMINDER_EVALUATION.md`) against the Box-1 answer key (`FAMILY_A_CC4D_AUGMENTATION.md`).

Status: **exploration**, not production. We have hand-built exactly one sensor-stage map (`tasks/task_spiced_hot_chocolate_cc4d.json`, activity 8). This prompt encodes the recipe→sensor-map *planning* rules (the predictor half of the former `CONVERSION_AND_EVAL_PROTOCOL.md`, now archived under `docs/_archive/`; see `PIPELINE_THREE_BOXES.md`) so a powerful LLM agent can draft the remaining 23 recipes. The detector vocabulary it may bind is the canonical `tasks/AUDIO_RUNTIME_LIBRARY.md` (filtered runtime set; this prompt's inline A1-A10 list is superseded where they differ).

> ⛔ **GROUND-TRUTH FIREWALL (read first; revised 2026-06-14).** This is the **PREDICTOR** pipeline. It produces the system's *plan* — what stages exist, what could go wrong, and which detectors to watch with — from the **recipe alone**. It must **NEVER** see anything derived from the eval recordings: not the CC4D error tags, not which recordings erred, not the Qualcomm mistake timestamps, not per-recording execution traces / DAG-violation rates. Those build the **answer key** (`FAMILY_A_CC4D_AUGMENTATION.md`, the GT pipeline) and feeding any of them here leaks the test labels into the predictor and invalidates every score. The predictor and the GT pipeline share **exactly one thing: the recipe (DAG + step text)** — the same procedure knowledge a deployed assistant would ship with. The predictor *anticipates* the error space from recipe structure ("a timed step can run over; a precondition can be skipped; a quantity can be wrong"); it is never told which errors actually occurred.

Two things are deliberately open at this exploration stage:

1. **The detector library is fixed but partially validated** — the agent may only bind listed primitives, and must respect their validation status (a binding to an untested primitive is a hypothesis, not a capability).
2. **The reminder taxonomy is closed** (`FAMILY_A_CC4D_AUGMENTATION.md` §3.0) — the agent plans detectors/reminders for the *anticipated* error space; whether a given recording actually triggers one is decided independently on the GT side and never enters this prompt.

Every agent output is reviewed by hand before use; accepted outputs become gold references for the eventual production compiler (Family D deliverable).

## How to run

**One input slot only** (the firewall). Paste the prompt, attach the gold example. Per project policy, run on Qwen (saltyfish server) or Claude — one recipe per call.

- `{RECIPE_GRAPH_JSON}` — verbatim `data/cc4d/annotations/task_graphs/{recipe}.json` (DAG + step text). **Nothing else.** Parallelism/flexible-order is read off the DAG itself (unordered node pairs), not from any execution trace.

---

## THE PROMPT

```text
You are compiling a cooking-recipe task graph into a "sensor-stage map": an executable
JSON for SENSOR CONTROL. Sensors are RGB + Audio, plus an expensive VLM. Knowing the
procedure ahead of time, you plan, per stage: which sensors to run vs. sleep (energy),
what cheap detectors + thresholds give a fast verdict without the VLM (latency), and
where a cheap event should TRIGGER one VLM call instead of running the VLM continuously.
The aim is NOT for cheap detectors to judge every error - many cooking errors need
visual reasoning - it is to spend the VLM only at the few moments worth it, and turn
sensors off otherwise. Output a single JSON object matching the OUTPUT SCHEMA below.

You will reason step by step, but your final answer must be a single JSON object
matching the OUTPUT SCHEMA at the end.

========================= INPUT (recipe only) =========================
RECIPE TASK GRAPH (nodes = steps with natural-language text; edges [a,b] mean step a
must precede step b; START/END are virtual):
{RECIPE_GRAPH_JSON}

This is your ONLY input about this recipe. You are NOT given how anyone actually
cooked it, nor which mistakes occurred - you must ANTICIPATE the error space from the
recipe's structure alone (a timed step can run over/under; a precondition can be
skipped or done out of order; a quantity/identity/setting can be wrong). Derive
allowed parallelism and flexible ordering from the DAG itself (node pairs with no
path between them are unordered).

========================= SENSOR VOCABULARY (CLOSED) =========================
You may ONLY bind detectors from this list. Status matters: "validated" = probed on
real CC4D recordings; "untested" = catalog-plausible, binding it is a hypothesis.

AUDIO (always-on, ~1% of one CPU core, shared 16 kHz STFT):
  A1 sustained_band(profile)  - appliance motor/fan/hum: microwave, blender, kettle,
                                grinder. VALIDATED (microwave 11/12 runs, 0 false).
  A2 tonal_burst              - end-of-cycle beeps, timer dings, igniter clicks.
                                VALIDATED (offsets to ~1 s).
  A3 transient_train(rate,len)- rhythmic impacts: chop/whisk/stir clinks.
                                DROPPED 2026-06-15 - NOT armable as a completion event,
                                trigger, or step recognizer (0.64 recall, 17 false
                                strong-clinks over 16 recordings; material-blind).
                                Allowed ONLY as optional corroboration inside a stage the
                                graph ALREADY believes is active.
  A4 texture_dynamics(band)   - sizzle/boil sustained level (rolling median). TESTED
                                over 9 stovetop recipes / 137 recs: A-solve on quiet-
                                prep/loud-fry (Pan Fried Tofu 1.0/0.04), B-trigger on
                                loud-fry/noisy-prep, none on gentle/dry cooking. DSP on
                                CPU; ~22 s lookahead (stage anchor, NOT reactive).
  AL learned_cook_tagger      - CNN14/YAMNet on log-mel -> AudioSet Frying/Sizzle/
                                Boiling/Steam. ALWAYS-ON ON-DEVICE (int8 on NPU, ~10 ms/
                                win, mW). COMPLEMENT to A4 on loud-fry stages: ~equal
                                cook recall at ~half the prep false-alarm (validated
                                all 9 stovetop recipes). A4 stays the default coverage
                                anchor (wins on simmer, e.g. chutney); fuse them. AL
                                does NOT rescue gentle/dry cooking (pancakes, toast).
  A5 water_running            - tap on/off edges, rinse, fill. UNTESTED, expected easy.
  A6 pour                     - DROPPED 2026-06-15 - NOT armable (0.5-4 false alarms/min,
                                ~15.5 s lookahead). Logged secondary signal only; never
                                bind it as a completion event or trigger.
  A7 rustle                   - packet/wrapper tear, bag crinkle. UNTESTED, promising.
  A8 scrub_scrape             - spatula/pan scrape, scrubbing friction. UNTESTED.
  A9 timer(t_start, t_end)    - duration measurement between any two detected events.
                                VALIDATED. Bounds come from step text.
  A10 context cues            - footstep (user walked away), door/cupboard transients.
                                Use only to gate reminders (e.g. unattended), never as
                                step completion.

RGB-MEDIUM (triggered, never per-frame on CPU):
  R1 roi_grounding(text)      - open-vocabulary object localization (OWLv2 class),
                                once per stage on a sharp frame. Use for: tool/vessel
                                identity ("knife", "ramekin"), object presence,
                                discrete counting of large items.
  R2 roi_transfer             - pre/post hand-occlusion appearance change of a tracked
                                container ROI (HSV-hist step + liquid-level edge).
                                The workhorse for silent add/place/pour-into steps.
  R3 roi_motion_periodic      - rhythmic motion in a tracked ROI (stirring, whisking).
                                ROI-RESTRICTED ONLY: the global-frame variant was
                                probed and FAILED (head sway wins). Needs >=8 fps
                                bursts.
  R4 display_read             - OCR on appliance display (microwave clock/power),
                                sharpest frame when user faces it. Use for power-level
                                / temperature settings.
  R5 hand_near_roi            - skin+motion gate near a tracked ROI; cheap context for
                                "user is acting on X".

EXPENSIVE (escalation only - one targeted call, never periodic):
  V1 vlm_query(question, evidence_window) - send buffered frames (+audio clip) with ONE
                                specific question. Used when a cheap trigger says THIS
                                moment matters but cheap sensors cannot verify WHAT
                                happened (amounts, fine technique, ingredient identity).

========================= COMPILATION RULES =========================
STEP 1 - instance disambiguation. If two graph nodes share identical/near-identical
text (e.g. two "cook for 1 minute" nodes) or a step repeats, give them instance ids
(cook_1, cook_2). Never merge them; never rely on text matching alone.

STEP 2 - completion event class, one per step. Choose the CHEAPEST signal class that
confirms the step HAPPENED (not that it was done correctly):
  audio_appliance | audio_transient | audio_texture | rgb_transfer |
  rgb_motion_periodic | rgb_state_change | logic_timer | vlm_only
Empirical priors for CC4D recipes: ~50% of steps are silent transfers (add/place/
spread/sprinkle/coat) -> rgb_transfer; appliance and heat steps -> audio; chop/whisk/
stir -> audio_transient (conditioned); pure waits ("allow to cool") -> logic_timer
anchored on the previous completion event.

STEP 3 - split off riders. Any attribute the completion event cannot verify - quantity
("2 tbsp", "1/5 tsp", "5 meatballs"), identity ("skimmed milk", "marinara not
ketchup"), settings ("on high"), fine technique ("until no lumps") - becomes an
expensive_criteria entry with a vlm_query action and, where it matches the CC4D error
taxonomy, a violation tag (Measurement/Preparation/Technique/Temperature Error).
Exception: discrete counts of large objects and tool/vessel identity may bind
R1 roi_grounding (medium) instead of VLM.

STEP 4 - detector binding with parameters. Bind each completion event to primitives:
  - timed steps: parse the duration from text; bounds = stated value +/- 33%
    (fixed convention). Emit timer bounds even when the duration is vague
    ("until golden") - then bound = [typical*0.5, typical*2] and mark "soft".
  - appliance steps: sustained_band onset/offset + tonal_burst end-beep fusion (the
    beep is the authoritative offset - hum alone truncates at low SNR).
  - every audio_transient binding must state its conditioning stage (what the graph
    must believe is happening for the train to mean this step).
  - runs shorter than 20 s are below the validated hum floor - mark duration checks
    on them "unscorable_cheap".

STEP 5 - plan reminders/detectors for the ANTICIPATED error space. For every place the
recipe structure ADMITS an error, plan how the system would catch it - from the recipe
alone. You are NOT told which errors occur; plan for all structurally-possible ones.
Use exactly these reminder families:

  F1 precondition_violation: for each DAG edge a->b, if step b's START is detected
     while precondition a has no completion event, the system should react. Plan one
     reminder per (precondition, trigger-stage) pair. If a is cheaply verifiable
     (audio/RGB completion event), emit a direct reminder; if not, emit an ESCALATION
     request ("ask the VLM: did a happen?"). Do NOT try to pre-judge which violations
     are benign vs harmful - that is a GT-side decision you have no information for and
     must not guess. Plan the detector for all of them; surfacing policy is set later.
  F2 parameter_violation: for each step with a duration/quantity/setting bound from
     STEP 4, plan an overtime_<step> trigger (bound crossed, still running) and an
     undertime check at run end. The ±33% tolerance is YOUR detector's firing
     threshold (a predictor parameter), not a claim about ground truth.
  F3 execution_error escalation hooks: for each expensive_criteria, a trigger that
     fires the vlm_query at the moment the evidence first exists (on the add event, on
     sizzle onset, ...). The surfaced reminder is conditional on the VLM answer.
  F4 safety (closed template list, descriptive tier, explicitly UNSCORED - no GT
     exists for it, FAMILY_A §3.0): appliance_done_unattended, heat_on_at_end,
     hot_handling. Plan them for deployment realism; they are never scored.

  Anticipated-error typing: tag each planned reminder with the error TYPE it targets
  from the general CC4D taxonomy {timing, temperature, measurement, preparation,
  technique, order, missing_step} - this is recipe-level anticipation (a vocabulary of
  what CAN go wrong), NOT a lookup of what did. No per-recording references exist here.

STEP 6 - per-stage SENSOR-CONTROL SCHEDULE (the project's core output). For each step/
stage, decide which sensors run and at what cost, driven by the anticipated errors from
STEP 5. The goal is ENERGY + LATENCY: keep the expensive sensor (VLM) off unless a
cheap event says spend it, and sleep whole sensor modalities when nothing checkable can
happen. Assign each (stage, anticipated_error) one SENSING ROLE:
  A SOLVE  - a cheap detector confirms OR refutes the error alone; no VLM ever. Use for:
             durations (audio timer), appliance state (hum/beep), precondition/order
             (graph completion-state), missing LOUD steps (absence of expected audio).
  B TRIGGER- a cheap detector catches the MOMENT but cannot judge the error; it fires
             ONE targeted vlm_query at that moment. Use for: identity/amount/technique
             that ride on a detectable event (add-event, sizzle-onset, step boundary).
  C NONE   - no cheap detector can solve OR reliably time the error (silent fine
             manipulation, continuous amounts, "stir improperly"). Declare it: either
             fall back to periodic VLM for that stage or accept the miss. DO NOT pretend
             a threshold covers it - listing role C honestly is required.
Then per stage output a sensor schedule: which of {audio, rgb_roi, vlm} are
active/idle/triggered, and WHY a modality can sleep (e.g. "RGB idle during microwave run
- audio hum tracks it"; "audio idle during silent plating - nothing audible, RGB ROI
watches"). This schedule is what the energy claim is measured against.

STEP 7 - stage collapse for scoring. Steps that are (a) mutually unordered in the DAG
and (b) indistinguishable to the bound cheap detectors (e.g. three silent adds into
the same mug) collapse into one coarse stage. Output the stage_map. Everything else
1:1; unlabeled time = "other".

KNOWN FAILURE MODES - do not repeat them:
  - Do not bind global-frame motion periodicity (probed: AUC ~ chance).
  - Do not use pour or any single transient as a standalone completion event.
  - Do not claim material/ingredient identity from audio (humans manage 48.8%).
  - Do not write open-ended VLM prompts as detectors ("is the user pouring milk?" as
    a CLAP/VLM text prompt scored at chance); escalation questions must be specific
    and evidence-windowed.
  - Do not invent reminder types outside F1-F4.

========================= OUTPUT SCHEMA =========================
{
  "task_id": str, "title": str, "source": str,
  "cc4d_step_ids": {readable_step_id: graph_node_id, ...},   // instance-suffixed
  "steps": [
    {"step_id": str, "order": int, "instruction": str,
     "preconditions": [step_id, ...],                        // graph edges verbatim
     "expected_duration_s": {"min": n, "typical": n, "max": n},
     "completion_events": [
       {"name": str, "tier": "cheap|medium|expensive",
        "detector": "<primitive ids + params + AND/OR composition>",
        "success": str,
        "conditioning_stage": str|null,                      // required for A3/A6
        "status": "validated|hypothesis"}],                  // from primitive status
     "expensive_criteria": [
       {"name": str, "why": str, "action": "vlm_query: <specific question>",
        "violation": "<CC4D error tag>|null"}]}],
  "reminders": [
    {"reminder_id": str, "family": "F1|F2|F3|F4", "step_id": str,
     "trigger": str, "message": str, "type": "reminder|warning|escalation",
     "anticipated_error_type": "timing|temperature|measurement|preparation|technique|order|missing_step",
     "detector_tier": "cheap|medium|expensive",            // what the trigger costs
     "scored": bool}],                                      // false for F4 safety (no GT)
  "sensor_schedule": [                                      // STEP 6 - the core output
    {"stage": str,
     "sensors": {"audio": "active|idle", "rgb_roi": "active|idle|triggered",
                 "vlm": "off|triggered"},
     "sleep_rationale": str,                                // why a modality can sleep
     "anticipated_errors": [
       {"type": "timing|temperature|measurement|preparation|technique|order|missing_step",
        "sensing_role": "A_solve|B_trigger|C_none",
        "cheap_signal": str,                                // detector + threshold, or null
        "vlm_trigger": str|null,                            // when to escalate + question
        "fallback": str|null}]}],                           // for role C: periodic VLM / accept miss
  "stage_map": {stage_name: [step_id, ...]},
  "open_questions": [str]   // anything you were unsure about - this list is read by a human
}
// NO ground-truth fields. This file is a PLAN derived from the recipe; it must contain
// nothing that could only be known by reading the eval recordings or their labels.
// Role C entries are REQUIRED where honest - they bound the energy claim, not weaken it.

Before answering, self-check: every step has exactly one completion-event class and
>=1 binding; every precondition is a graph edge; every reminder is F1-F4; every A3/A6
binding has a conditioning_stage; duplicate-text nodes got instance ids; every stage has
a sensor_schedule entry with a sleep_rationale; every anticipated error has a sensing
role (A/B/C) and role-C errors are stated honestly with a fallback; every "hypothesis"
binding is also in open_questions; the output contains NO field that could only be known
from an eval recording or its labels.
```

---

## Gold few-shot example

Attach `tasks/task_spiced_hot_chocolate_cc4d.json` (activity 8) as the worked example. Note for the agent: that file predates this schema — match its *judgment* (which it derived from the recipe), not its exact field set. It contains no per-recording GT, so it is firewall-safe to show.

## Appendix A — per-recipe DATASET ANALYSIS (for us, NOT a compiler input)

> ⛔ **This table is GT-derived (it counts how the eval recordings were executed and how often they erred). It is for OUR understanding of the corpus only and must NOT be pasted into the compiler** — doing so is exactly the leak the firewall forbids. The compiler computes the one column it legitimately needs (free% = DAG parallelism) from the recipe graph itself; the rest (dagViol%, skip%, errRec%, uniqOrd) are eval-trace statistics the predictor may never see.

Columns: recordings; graph nodes (excl. START/END); free% = step pairs unordered by the DAG (parallelism, *DAG-derivable*); uniqOrd = distinct executed orders; dagViol% = recordings whose executed order violates ≥1 edge (upper bound — includes annotation noise); skip% = step instances with t=-1; errRec% = recordings with ≥1 error. (uniqOrd/dagViol/skip/errRec are GT-side trace stats — never a compiler input.)

| recipe | rec | nodes | free% | uniqOrd | dagViol% | skip% | errRec% |
|---|---|---|---|---|---|---|---|
| blenderbananapancakes | 19 | 14 | 19 | 18 | 47 | 3.0 | 63 |
| breakfastburritos | 16 | 11 | — | — | ~25 | — | — |
| broccolistirfry | 16 | 25 | 32 | 16 | 31 | 3.2 | 38 |
| buttercorncup | 14 | 12 | 17 | 11 | 50 | 10.1 | 64 |
| capresebruschetta | 18 | 11 | 64 | 17 | 56 | 5.1 | 67 |
| cheesepimiento | 15 | 11 | 24 | 15 | 53 | 3.6 | 60 |
| coffee | 15 | 16 | 31 | 15 | 40 | 2.9 | 47 |
| cucumberraita | 20 | 11 | 60 | 19 | 40 | 5.0 | 40 |
| dressedupmeatballs | 16 | 16 | 26 | 16 | 44 | 3.9 | 62 |
| herbomeletwithfriedtomatoes | 17 | 15 | 14 | 16 | 53 | 5.9 | 65 |
| microwaveeggsandwich | 18 | 12 | 24 | 17 | 67 | 7.4 | 72 |
| microwavefrenchtoast | 14 | 11 | 13 | 10 | 21 | 2.6 | 36 |
| microwavemugpizza | 13 | 14 | 8 | 13 | 38 | 2.2 | 62 |
| mugcake | 17 | 20 | 14 | 17 | 29 | 5.9 | 59 |
| panfriedtofu | 15 | 19 | 5 | 8 | 33 | 2.8 | 53 |
| pinwheels | 12 | 19 | 5 | 7 | 42 | 8.7 | 67 |
| ramen | 17 | 15 | 11 | 16 | 35 | 2.7 | 41 |
| sautedmushrooms | 14 | 18 | 10 | 13 | ~50* | 4.0 | 57 |
| scrambledeggs | 16 | 23 | 17 | 16 | 50 | 6.0 | 62 |
| spicedhotchocolate | 16 | 7 | 14 | 13 | 50 | 8.9 | 62 |
| spicytunaavocadowraps | 18 | 17 | 44 | 18 | 44 | 7.8 | 61 |
| tomatochutney | 15 | 19 | 5 | 15 | 67 | 6.7 | 67 |
| tomatomozzarellasalad | 18 | 9 | 17 | 15 | 6 | 3.1 | 39 |
| zoodles | 15 | 13 | 9 | 13 | 60 | 6.2 | 73 |

\* sautedmushrooms' raw 100% is an artifact: its graph has TWO nodes with identical text "cook-cook the pan, often stirring, for 1 minute", and recordings repeat global step id 207 — text-based node matching collapses the instances and fabricates violations. After disambiguation the genuine top violations remain real (e.g. mince-garlic-before-heat in 7/14 recordings). This is exactly why STEP 1 of the prompt exists. breakfastburritos was computed separately (name missing from the first sweep); its top violations: mix-before-pour 4/16, whisk-before-microwave 3/16.

Reading of the table: recordings genuinely diverge — almost every recording of every recipe is a **unique topological order** (uniqOrd ≈ rec), and 20–67% of recordings violate at least one DAG edge, so precondition tracking is live on every recipe, not just error recordings. Parallelism varies 5–64% free pairs: high-free recipes (capresebruschetta 64, cucumberraita 60, spicytunaavocadowraps 44) are where stage collapse and precondition ambiguity bite; low-free stovetop recipes (panfriedtofu, tomatochutney 5%) are nearly linear and timer/sizzle-dominated.

## Validation plan for the prompt itself (exploration exit criteria)

1. Run on **spicedhotchocolate** with the gold example REMOVED → compare against the hand-built JSON (the only available gold). Agreement on event classes + reminder set = pass.
2. Run on **microwaveeggsandwich** (same detector family, validated primitives only) → human-review cost should be minutes, not hours.
3. Run on **panfriedtofu** (forces the untested sizzle primitive → tests the hypothesis-flagging path) and **pinwheels** (worst audio fit, 0% cheap — tests that the agent escalates instead of inventing detectors).
4. **Firewall check**: confirm no output field encodes per-recording GT (no error counts, no "this recording erred", no Qualcomm timestamps). The plan must be identical whether or not the eval recordings exist.

After the plan is accepted, it is scored by running its detectors over the recordings and comparing the *fired* reminders to the independently-built GT truth table (`FAMILY_A_CC4D_AUGMENTATION.md`). The two pipelines meet only at scoring time, never at authoring time.
