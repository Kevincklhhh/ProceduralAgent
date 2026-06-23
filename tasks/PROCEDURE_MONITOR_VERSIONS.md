# Procedure Monitor — Pipeline Stages & Assumptions

Human-facing notes for the procedure-monitor artifacts. The JSON itself is machine-facing
and carries no version-narrative fields; record those here.

**These are two pipeline STAGES, not two competing versions.** A recipe is compiled in two
hops; the stage-2 artifact is the *output of compiling* the stage-1 artifact.

```
recipe  (tasks/cc4d/<name>.json)
   │  STAGE 1   recipe → DAG with criteria
   ▼
<name>.criteria.json        schema 2.0   ← nodes form: partial-order DAG + plain claims
   │  STAGE 2   criteria → sensor control plan
   ▼
<name>.sensorplan.json      schema 2.0   ← nodes form: + {detector, detection_criteria} bindings
   │  runtime
   ▼
eval/proposed_runtime.py    (via eval/proposed_plan_loader.py)
```

| Stage | Template | Worked instance | Schema |
|---|---|---|---|
| 1 — recipe → criteria | `tasks/recipe_to_criteria_template.json` | `tasks/cc4d/spicedhotchocolate.criteria.json` | 2.0 (nodes form) |
| 2 — criteria → sensor plan | `tasks/criteria_to_sensorplan_template.json` | `tasks/cc4d/spicedhotchocolate.sensorplan.json` | 2.0 (nodes form) |

The convention is `<name>.criteria.json` (stage 1) and `<name>.sensorplan.json` (stage 2).
Both are the **nodes form**: a flat `nodes[]` list, each node carrying `preconditions` +
`recognition`/`completion` (plain-text observers in stage 1; `{detector, detection_criteria}`
bindings in stage 2). There is no authored `cond`/`state_update`/`graph` grammar.

RUNNABLE (closed 2026-06-17): `eval/proposed_runtime.py` / `proposed_plan_loader.py` now run
the `.sensorplan.json` directly. `load_plan()` detects `artifact_type=="sensor_control_plan"`
and `sensorplan_to_graph()` compiles the nodes form into the executable graph at load time
(D1/`none` nodes → steps; the contiguous VLM-recognized nodes → one C-none `step_block`).
Default `--plan` is now `spicedhotchocolate.sensorplan.json`. The old executable grammar
(`graph.steps`/`step_blocks`/`cond`/`state_update`) still exists, but only as that internal
compiled form — it is generated, never authored.

## Stage 1 — DAG with criteria (schema 2.0, nodes form)

Template `tasks/recipe_to_criteria_template.json`; worked instance
`tasks/cc4d/spicedhotchocolate.criteria.json`. The recipe becomes a **flat list of nodes
under a partial order** — *not* a linear chain. Each node carries:

- `preconditions` — the step_ids that must be COMPLETE before it can start (`[]` for a root).
  Nodes with the same preconditions are an **emergent unordered group** (e.g. the three
  hot-chocolate adds, each requiring only the microwave); no order is imposed among them.
- `recognition` / `completion` — each just a plain-language **`claim`** of what proves the
  step is happening now / is finished. **Channel-blind:** stage 1 holds *no* observers, *no*
  detectors, *no* `cond`/`state_update` grammar — how to SENSE each claim is decided in
  stage 2. (`completion.inferred_from` names a successor when the step has no own end signal.)
- `checks` (optional, T2) — one entry per recipe-stated error the step can have, tagged
  `reminder ∈ {measurement|technique|preparation|temperature|timing}` with a `claim`. These
  are the proactive-reminder hooks. **Do not** author order/missing_step here — those are
  emitted from DAG state by the runtime (see `docs/REMINDER_RUNTIME.md`), not per-node.

### Authoring the checks (anticipate the error space from recipe structure)

The predictor never sees which errors actually occurred (the firewall). It *anticipates* the
error space from the recipe alone: a timed step can run over (→ `timing`); a quantity/identity
can be wrong (→ `measurement`/`preparation`); a technique can be skimped (→ `technique`). Write
one check per such recipe-stated expectation; the runtime decides cost (free DAG state, the D6
clock, or a VLM call) in stage 2.

### Assumptions (stage 1)

- **Partial order, not linear** — concurrently-eligible steps are left unordered; the runtime
  tracks an active *set* (size ≥ 1), not a single cursor.
- **Order errors / skips are runtime concerns**, not stage-1 structure: the DAG states the
  legal partial order; deviations from it are detected at runtime as reminders.

## Stage 2 — sensor control plan (schema 2.0)

Same node list as stage 1, with each node's `recognition`/`completion`/`checks` given a
`{detector, detection_criteria}` binding plus a `sensing_role`. Template
`tasks/criteria_to_sensorplan_template.json`; authoring guide
`tasks/PROCEDURE_MONITOR_COMPILER.md`. The executable graph (`cond`/`state_update`/
`step_block`) is **generated** by `eval/proposed_plan_loader.sensorplan_to_graph()` at load
time — never authored — and run by `eval/proposed_runtime.py`. The loader **collapses** the
contiguous run of VLM-recognized (silent, C-none) nodes into one `step_block` resolved by the
periodic VLM.

### How a node's claim maps to a sensor

The author picks the detector per claim from `tasks/AUDIO_RUNTIME_LIBRARY.md`:

- **a real catalog event** (microwave hum → `D1`, motor → `D2`, cook → `D3`/`D4`, water →
  `D5`) → bind that detector; the loader emits a consuming `sensor_event` (or a B-trigger
  `vlm_verdict` for a `D4` candidate).
- **no own end signal** (`completion.inferred_from` set) → the loader uses a non-consuming
  `next_anchor` on the successor's event + an `elapsed` timer fallback (the inference made
  explicit by `inferred_from`, kept honest by consume-once).
- **silent on both sides** → `detector: "VLM"`; the node joins the C-none `step_block`, the
  VLM labels members, and the next real anchor closes the block.
- **timing check** → `detector: "D6"` against `duration_constraint_s`.

### What stage 2 adds beyond the criteria

Detector bindings + per-event latency (from the catalog), `duration_constraint_s` + `elapsed`
fallbacks (durations parsed from the recipe instruction text), `vlm_policy` (poll cadence +
cost caps), `runtime_config.tick_s`, and `cc4d_step_id` provenance.

### Cleanup vs the stale template

The stage-2 template drops fields the runtime never read: `monitor.watch_for`, `monitor.sleep`,
`output_schema` (output is documented separately), `graph.edges` (the DAG lives in stage 1; the
runtime uses `requires` + `open` ops), `produces`, and the block `reason` enum. It keeps the
load-bearing grammar (`graph.steps` / `graph.step_blocks`, `step_id`/`block_id`, `requires`,
`monitor.primitives`, `cond`, `state_update`, `vlm.poll`) and adds provenance fields
(`_compiled_from`, `from_criterion`) linking each node/rule to its stage-1 source.

## Stale — original combined template (schema 0.6)

`tasks/_archive/procedure_monitor_template_0.6_STALE.json` (archived 2026-06-17; see
`tasks/_archive/README.md`): the original single executable template that did stages 1 and 2
in one hop. **Superseded** by the two-stage split. Known defect that motivated the split: it
conflated current-step completion with next-step start (the same `D1.cycle_start` served as
both `fill_milk.complete` via `next_anchor` and `microwave_initial.start`), with no field
marking the next-anchor completion as an inference (the nodes form makes this explicit via
`completion.inferred_from`). Do not author new monitors from it.

The worked 0.6 instance `tasks/cc4d/spicedhotchocolate.monitor.json` is **kept in place** (not
archived): it is the visualizer's fallback plan and the byte-identical golden reference for
`proposed_plan_loader.sensorplan_to_graph()` (the sensorplan compiles to exactly it). It is no
longer the running default — the runtime now drives `.sensorplan.json` directly (see above).
