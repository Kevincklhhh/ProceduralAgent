# Procedure Monitor Compiler Guide

This guide tells an LLM how to turn a recipe task graph into an **executable**
`procedure_monitor` JSON — a sensor-conditioned state machine for online step
recognition. Its job is to detect step starts and completions throughout a video,
maintain procedure context, and emit a transition trace plus stage intervals.

- Template: `tasks/procedure_monitor_template.json` (schema 0.6).
- Detector catalog (authority): `tasks/AUDIO_RUNTIME_LIBRARY.md`.
- Worked instance: `tasks/cc4d/spicedhotchocolate.monitor.json`.
- The compiled JSON is consumed by `eval/plan_loader.py` (compiles conditions to
  predicates) and run by `eval/monitor_runtime.py`.

This monitor is not a VLM replica and not a proactive assistant. Do not add error
checks, reminders, user messages, or intervention policies. It uses the recipe
graph plus selected sensor events to update procedure state over time.

## Input Boundary (firewall)

Use recipe-only information: step ids and instruction text; step order and
preconditions; durations/quantities stated in the recipe; and the detector
vocabulary in `tasks/AUDIO_RUNTIME_LIBRARY.md`. Do NOT use recording labels,
mistake annotations, Qualcomm timestamps, per-video execution traces, or which
recordings contain mistakes. The monitor is a plan compiled from the recipe, not
an answer key.

## Detector vocabulary (authority: AUDIO_RUNTIME_LIBRARY.md)

Bind only these primitives. Each emits the named events; bind a detector to a step
only if the step's transition matches the event AND the catalog gate holds.

| Primitive | Emits | Use |
|---|---|---|
| `D1` microwave_cycle | `cycle_start`, `cycle_end` | microwave run start + authoritative (beep-fused) end |
| `D2` appliance_motor | `motor_on`, `motor_off` | blender/grinder (recipe-gated) |
| `D3` cook_end | `cook_end` | end of fry/saute/boil (bounded hangover) |
| `D4` cook_start | `cook_start_candidate` | onset of frying — B-trigger only (fires one VLM call) |
| `D5` water_flow | `water_on`, `water_off` | tap rinse/wash/fill (bind to one step) |
| `D6` timer | `overtime`, `undertime`, `precondition_violation` | duration/precondition logic (no audio) |
| `VLM` | step verdict | the expensive sensor — B-trigger target / C-none periodic |

Do NOT bind anything in the catalog's *Tested and excluded* list (kettle boil,
chop/stir clink-train, pour, etc.). In particular there is **no clink/stir
detector**: a silent stir is C-none (VLM), never an audio completion event.

## Sensing roles (derived, but record them)

- **A-solve:** a step's rules are settled by a D1–D6 sensor event (no VLM).
- **B-trigger:** a `D4` (or similar) candidate event is gated by a `vlm_verdict`
  leaf — one VLM call confirms it.
- **C-none:** no audio detector applies (silent add/place/spread, fine technique,
  ingredient identity). Resolved by periodic VLM. ~48% of CC4D steps are C-none —
  report this honestly; do not invent an audio detector for them.

Set each step/block's `sensing_role` accordingly; the loader also derives it from
the leaf types present.

## Condition grammar (executable — no prose conditions)

Each `start_when` / `complete_when` rule has `rule_id`, an optional human-readable
`when` (documentation only), a structured `cond`, and a `state_update`. The list
of rules is an implicit OR (first to fire wins, each with its `rule_id`).

**Leaf conditions:**

- `{"type":"eligible"}` — all `requires` are complete (and the step isn't already
  active/complete). Use for steps whose start is just "the predecessor finished".
- `{"type":"sensor_event","primitive":"D1","event":"cycle_end","after":"<step>.start","min_field":{"duration_s":8},"confidence_min":0.0}`
  — earliest unconsumed detector event matching, with `t_s >= after` time. CONSUMING
  (consume-once, so a later rule can't reuse it).
- `{"type":"next_anchor","primitive":"D1","event":"cycle_start","after":"<step>.start"}`
  — same match but NON-consuming: the event belongs to the next stage; use it in a
  `complete_when` to close the current stage at the successor's onset.
- `{"type":"step_state","step":"<id>","is":"complete"}` — graph-state reference.
- `{"type":"elapsed","since":"step_active","step":"<id>","max_s":90,"fires_at":"max"}`
  — timer fallback (or `min_s`/`fires_at":"min"`).
- `{"type":"vlm_verdict","for_step":"<id>","expect_status":["Step transition"]}`
  — satisfied when a VLM verdict for that step has landed with a matching status.

**Combinators:** `{"type":"any","of":[...]}` and `{"type":"all","of":[...]}`.

**`after` anchors** are `"<step_id>.start"` or `"<step_id>.complete"`, resolved to
the recorded transition time at runtime. Use them to bind, e.g., the first
microwave to `microwave_initial` and the second to `heat_serve` (consume-once +
`after` keep them distinct).

**State-update ops:** `set_state{step,to}`, `set_foreground{step}`, `open{steps}`,
`mark_members{block,confirmed,rest}`, `add_background{monitor}`,
`remove_background{monitor}`. `set_state` to `complete` records the completion time,
moves the step to completed, and re-evaluates successor eligibility.

## Compilation steps

1. Copy the recipe graph: every step in `graph.steps` (or inside a `step_blocks`
   entry), preconditions as `graph.edges`, readable step ids.
2. Fill `requires`/`produces` with short state names (`milk_in_mug`,
   `milk_heated_once`), not the instruction text.
3. Choose `monitor.mode` (`foreground` while the user manipulates objects,
   `background` for a process they can walk away from, `block` for a tracked
   region) and `primitives` from the catalog; `sleep` the expensive ones.
4. Write `start_when` (entry) and `complete_when` (exit) rules with structured
   `cond` + `state_update`. A completion may use `next_anchor` (close at the
   successor onset) or `elapsed` (timer fallback). Completion rules must be causal:
   they use evidence available up to now plus the detector's declared latency
   (all ≤ 10 s — see the catalog).
5. Use a `step_block` for unordered/overlapping silent steps (e.g. several adds
   into one mug). If cheap sensors can only detect the coarse block, close the
   block and `mark_members` (confirmed complete, rest `unknown`) rather than
   inventing fine labels; let the C-none VLM poll label members.
6. Set the `vlm_policy` (poll cadence + cost caps) and per-block `vlm.poll`.

## Minimal example (a real compiled rule)

```json
{
  "step_id": "microwave_initial",
  "order": 2,
  "instruction": "Microwave the contents of the mug for 1 minute",
  "duration_constraint_s": 60,
  "requires": ["fill_milk"],
  "produces": ["milk_heated_once"],
  "sensing_role": "A-solve",
  "monitor": { "mode": "foreground", "primitives": ["D1", "D6"],
               "watch_for": ["D1.cycle_start", "D1.cycle_end"], "sleep": ["VLM"] },
  "start_when": [
    { "rule_id": "mw1_start", "when": "first microwave hum onset after fill",
      "cond": { "type": "sensor_event", "primitive": "D1", "event": "cycle_start", "after": "fill_milk.start" },
      "state_update": [ { "op": "set_state", "step": "microwave_initial", "to": "active" },
                        { "op": "set_foreground", "step": "microwave_initial" } ] }
  ],
  "complete_when": [
    { "rule_id": "mw1_done", "when": "first microwave cycle ends (beep-fused offset)",
      "cond": { "type": "any", "of": [
        { "type": "sensor_event", "primitive": "D1", "event": "cycle_end", "after": "microwave_initial.start" },
        { "type": "elapsed", "since": "step_active", "step": "microwave_initial", "max_s": 90, "fires_at": "max" } ] },
      "state_update": [ { "op": "set_state", "step": "microwave_initial", "to": "complete" },
                        { "op": "open", "steps": ["quiet_middle"] } ] }
  ]
}
```

## Output checklist

Before returning JSON, check:

- Every recipe step appears in `graph.steps` or inside `step_blocks`.
- Every recipe precondition appears in `graph.edges`.
- Every step/block has `requires`, `produces`, `sensing_role`, `monitor`,
  `start_when`, `complete_when`.
- Every rule has `rule_id`, a structured `cond`, and a `state_update`.
- Every `monitor.primitives` item is one of `D1`–`D6` or `VLM` (catalog vocabulary);
  nothing from *Tested and excluded* is bound; no clink/stir audio completion.
- Every `cond` leaf uses a real event from the catalog and a valid `after` anchor.
- There are no fields named `error_checks`, `errors`, `reminders`, `actions`,
  `messages`, or `proactive`.
- No recording-specific labels, GT timings, or ground truth anywhere.
