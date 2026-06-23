# Criteria Generation — Probing Mode (firewall RELAXED)

How to generate a recipe's Stage-1 `<recipe>.criteria.json` (the per-step checks) **in
probing mode**, where we deliberately let the observed error space inform what to check.
This is a *process spec* an LLM (or a person) follows with the listed context in hand — not a
Python script. The mechanical aggregation it consumes is produced by
`scripts/probe_recipe_reminders.py`.



---

## 1. Context fed to the generator

| # | Input | Path | Required? | What it supplies |
|---|---|---|---|---|
| 1 | **Recipe task json** | `tasks/cc4d/<recipe>.json` | **yes** | DAG (`preconditions`), step text, `duration_constraint_s` — the firewall-clean substrate. |
| 2 | **Probe reminder inventory** (slim) | `tasks/cc4d_probe/<recipe>.reminders.json` | **yes** | Per step: the observed reminder subtypes, `n_occurrences`/`n_recordings`, and unique `descriptions` (e.g. "added 4 pieces", "used whole milk", "Spill"). This is the agent-facing view — no per-occurrence provenance. |
| — | *(record, not an input)* | `tasks/cc4d_probe/<recipe>.instances.json` | no | Full per-occurrence record behind the counts — each error's `recording_id`, `cc4d_step_id`, `video_window`. Kept for traceability; the agent does **not** read it. |

The generator works **per node**, walking the recipe DAG in `order`.

---

## 2. Per-node procedure

Emit **one node per distinct `step_id`**, in the recipe's `order`. For each, fill the fields below.

> **Recurring step_ids (loops).** Some recipes repeat a step_id in `order` (e.g. a
> microwave→stir→microwave cycle, or a "cook 1 min" step run several times). `step_id` is the
> join key and must stay **unique**, so emit **one node** for it — do not emit a second node
> with the same id. Use the **first occurrence's** `preconditions` and `duration_constraint_s`.
> (This collapses the loop into a single node; representing repeated executions as distinct
> node *instances* is a known open limitation — see `docs/REMINDER_RUNTIME.md` "node re-entry /
> loops". Note such recipes in `_provenance`.) The slim probe is already collapsed per step_id,
> so its counts line up with the single node.

### 2.1 Copy these verbatim from the recipe (no judgement)

- **`step_id`** — copy the recipe's `step_id` exactly, **same value and type** (the CC4D recipe
  uses integers, e.g. `88`). Do **not** invent readable slugs: they aren't reproducible and
  would break the `preconditions` and cross-artifact (criteria→sensorplan→runtime) joins, which
  all key on this id. *(The hand-authored `spicedhotchocolate.criteria.json` predates this rule
  and uses aliases like `"fill_milk"`; that is cosmetic legacy — use verbatim ids for new recipes.)*
- **`instruction`** — copy the recipe step text.
- **`preconditions`** — copy the recipe's `preconditions` array verbatim (same ids as above).
- **`duration_constraint_s`** — copy if the recipe step has one, else `null`.

### 2.2 Write `recognition.claim` (recipe text only)

Plain-language statement of what, seen in the frames, proves this step is **happening now** —
usually the step's main verb + object ("the user is pouring milk into the mug"). Two cases:

- **Root step** (`preconditions: []`): note it is active from the start.
- **Unordered fork** (this step shares its `preconditions` with one or more siblings — same
  prerequisite set): say it is *one of N concurrently-eligible* steps and that recognition must
  say *which* is active now (no order is imposed among them).

### 2.3 Write `completion.claim` + `inferred_from` (recipe text only)

Plain-language end-state of the step. Then decide `inferred_from`:

- If the step ends with its **own observable event** (an appliance stops, the item is plated),
  describe it and set `"inferred_from": null`.
- If the step **just stops** with no distinct end signal of its own (pour / add / stir / mix),
  set `inferred_from` to a **successor `step_id`** whose onset proves this step is over, and say
  so in the claim (e.g. "no end sound; inferred from the microwave starting"). **Tiebreak when
  the step has several successors:** pick the **recipe-order-earliest direct successor** (the
  one listing this step in its `preconditions` that comes first in recipe `order`). For an
  unordered fork that reconverges, that earliest successor is typically the shared
  reconvergence node — use it.

### 2.4 Write `checks` (T2) — probe-derived only

> **Retired:** the former "recipe-anticipated" source (parse the step text for pinned
> parameters) and the union/merge rule were **removed** on 2026-06-22 — text-parsing for which
> subtype a step "pins" is not deterministic. The archived method and corpus-impact numbers are
> in `tasks/cc4d_probe/_archive/ARCHIVE_NOTE_recipe_anticipated_checks.md`. Recognition and
> completion (§2.2/§2.3) are unchanged and remain recipe-only.

Emit **one check per distinct `reminder` subtype that the probe observed under this step** —
every `execution_error` / `parameter_violation(timing)` / `temperature` subtype appearing in
input #2 with any count ≥ 1. The **set** of checks is thus fully determined by the annotations
(no judgement about what the text implies). Subtypes `order` / `missing_step` are excluded
(§2.5, runtime-emitted); `other` is excluded (no check subtype).

Phrase each `claim` as a **recipe expectation sharpened by the observed `descriptions`** — e.g.
"the recipe calls for 2 pieces; flag a different count", not "the user added 4". Use the
descriptions to make the deviation concrete (a spill, a 10% power level), but state it as the
norm being checked. One check per subtype — no duplication is possible (the slim probe is
pre-collapsed per `(step_id, subtype)`, so there is **no merge step**). Each `claim` is
plain-language and **channel-blind** — no detector here; Stage 2 binds it.

Every authored check is now probe-derived (firewall-relaxed by construction); the old
`[probe-added]` prefix is therefore dropped — the whole `checks` block is the relaxed slice,
recorded in `_provenance`.

### 2.5 Do NOT author `order` or `missing_step`

These (`precondition_violation`) are emitted by the **runtime from DAG state** (a node
recognized active while its preconditions are unmet, or a skipped node whose successor ran),
never as per-node checks. The probe lists them for visibility; skip them. (Restated in §4.)

---

## 3. Output contract

Write `tasks/cc4d_probe/<recipe>.criteria.json` — schema 2.0, **nodes form, claim-only**. Use
exactly this skeleton (constants are literal; `<...>` are filled per §2):

```json
{
  "schema_version": "2.0",
  "artifact_type": "procedure_criteria",
  "_mode": "PROBING_FIREWALL_RELAXED",
  "_provenance": "Stage-1 probing mode (firewall relaxed). recognition/completion are recipe-only (firewall-clean); ALL checks are probe-derived from the observed error space (<probe file consumed>) -- one per observed subtype -- and are NOT firewall-clean, so do not score these criteria as a clean predictor result. order/missing_step are runtime-emitted (not authored). <one line on the recipe DAG shape>.",
  "task": {
    "task_id": "<recipe.task_id>",
    "title": "<recipe.title>",
    "source": "tasks/cc4d/<recipe>.json"
  },
  "nodes": [
    {
      "step_id": "<recipe step_id, verbatim>",
      "instruction": "<recipe step text>",
      "preconditions": ["<recipe preconditions, verbatim>"],
      "duration_constraint_s": "<recipe value or null>",
      "recognition": { "claim": "<§2.2>" },
      "completion": { "claim": "<§2.3>", "inferred_from": "<successor step_id or null>" },
      "checks": [
        { "reminder": "measurement|technique|preparation|temperature|timing",
          "claim": "<§2.4; one per probe-observed subtype, recipe-expectation wording>" }
      ]
    }
  ]
}
```

A node with no checks carries `"checks": []`. Stage 2
(`tasks/criteria_to_sensorplan_template.json` + `tasks/PROCEDURE_MONITOR_COMPILER.md`) later
binds each claim to a `{detector, detection_criteria}` — `timing`→`D6`,
execution/temperature→`VLM` — unchanged by mode; the generator does not do that here.

### Self-check before finishing

- One node per **distinct** `step_id` (no duplicate ids, even if the recipe repeats one);
  `step_id`/`preconditions` byte-match the recipe (first occurrence for a repeated id).
- Every `inferred_from` names a real successor `step_id` (or is `null`).
- No `order`/`missing_step`/`other` checks anywhere.
- Exactly one check per `(step_id, subtype)` the probe observed; no recipe-anticipated checks
  added beyond what the probe shows; output is valid JSON.

---

## 4. Order / missing_step are out of scope here (restated)

The two largest GT classes (`order` 795, `missing_step` 285 corpus-wide) are **structural**.
They need no per-node criterion and no VLM: the active-set FSM already knows the legal partial
order, so a step running with unmet preconditions (order) or a skipped step whose successor
runs (missing) is read straight off state. Authoring them as checks would double-count and
add cost. See `docs/REMINDER_RUNTIME.md` §1 (Precondition, `detector: none`).

---

## 5. Worked delta — does probing add criteria for spiced hot chocolate?

> **Legacy framing.** This section compares probe-observed subtypes against the *retired*
> recipe-anticipated checks, to show what text-only authoring missed. Under the current rule
> (§2.4, probe-only) **every** row's "probe adds" column simply *is* the check set — there is no
> "authored" column anymore. Kept because the spill/technique insight still motivates the design.

Comparing the existing hand-authored criteria (`tasks/cc4d_probe/spicedhotchocolate.criteria.json`)
against the probe (execution + timing + temperature subtypes only; order/missing excluded):

| Step | Authored | Probe adds | Note |
|---|---|---|---|
| fill_milk | preparation | **technique** (spill ×6), **measurement** (underfill ×1) | spill is the headline miss |
| microwave_initial | timing | **temperature** (power 10% ×1) | scored; low VLM recall (dial off-screen) |
| add_cinnamon | measurement, preparation | **technique** (spill ×1) | |
| add_sugar | measurement, preparation | — | full match |
| add_chocolate | measurement | **technique** (spill ×1) | |
| mix | technique | — | full match |
| heat_serve | timing | **temperature** (power ×4), **preparation** (short-heat ×2) | temp scored (low recall); "heat 40 s" overlaps timing |

**Yes — probing adds real criteria, headlined by a `technique`/spill check on the pour/add
steps** (fill_milk ×6, plus the adds). This is exactly the "the recipe wouldn't pin precisely"
gap: no recipe phrase states "don't spill," so firewall-clean authoring misses it, yet it is
the single most frequent execution error on `fill_milk`. The clean way to cover it is a
**generic `technique` (spill/mess) check on any pour/transfer step**, since spilling is
step-shape-driven, not parameter-driven.

The rest are marginal: underfill (×1), the power-level `temperature` checks (scored, but expect
low VLM recall — dial off-screen), and the heat_serve `preparation` example
("heat 40 s") which is really a timing deviation under a different tag. `add_sugar` and `mix`
already match the probe exactly — recipe-text anticipation was complete there.

**Takeaway:** for this recipe, probing's net contribution is one genuinely-missed pattern
(spill/technique on pour steps) plus confirmation that the parameter-pinned checks
(measurement/preparation/timing) were already complete from recipe text alone.
