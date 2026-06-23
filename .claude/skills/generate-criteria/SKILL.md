---
name: generate-criteria
description: Generate a Stage-1 procedure criteria.json for a CC4D recipe in PROBING MODE (firewall relaxed) — recipe DAG + step text + observed-error probe → per-step recognition/completion claims and proactive-reminder checks. Use when asked to author/generate/regenerate a recipe's criteria.json, fill in checks for a recipe, or run the probing-mode criteria generator. Stage-1 only (the predictor; not GT, not the sensorplan).
---

# Generate criteria.json (probing mode)

Authors `<recipe>.criteria.json` for one CC4D recipe by following the process spec
**`tasks/CRITERIA_GENERATION_PROBING.md`** — that doc is the single source of truth for the
field rules (step_id reuse, recognition/completion + `inferred_from`, the recipe∪probe check
union, the merge/`[probe-added]` rule, the output skeleton, the self-check). This skill is the
operational wrapper around it; **do not** restate or re-derive its rules — read it and apply it.

> ⚠️ **PROBING MODE — firewall relaxed.** This generator reads the observed error space (a
> Box-1 input). The output is for authoring/analysis only and is stamped
> `_mode: PROBING_FIREWALL_RELAXED`. It must **not** be scored as a firewall-clean predictor
> result. (See `docs/PIPELINE_THREE_BOXES.md` for the firewall.)

## Procedure

1. **Resolve the recipe stem** (e.g. `spicedhotchocolate`). The recipe must exist at
   `tasks/cc4d/<stem>.json`.

2. **Ensure the slim probe exists.** If `tasks/cc4d_probe/<stem>.reminders.json` is missing or
   stale, regenerate it:
   ```
   python3 scripts/probe_recipe_reminders.py --recipe <stem>
   ```
   (Writes the slim agent view `<stem>.reminders.json` and the full record
   `<stem>.instances.json`.)

3. **Read exactly these three files — and nothing else of the kind:**
   - `tasks/CRITERIA_GENERATION_PROBING.md` (the process spec)
   - `tasks/cc4d/<stem>.json` (recipe — input #1)
   - `tasks/cc4d_probe/<stem>.reminders.json` (slim probe — input #2)

   Do **NOT** read `*.instances.json`, any existing `*.criteria.json`/`*.sensorplan.json`,
   `recipe_to_criteria_template.json`, or `PROCEDURE_MONITOR_*`. The spec is self-contained;
   reading those defeats the point and risks copying instead of generating.

4. **Generate** per the spec's §2 per-node procedure and §3 output contract, then run its
   "Self-check before finishing" checklist. Write to:
   ```
   tasks/cc4d_probe/<stem>.generated.criteria.json
   ```
   (Use the `.generated.` infix so a freshly generated artifact never silently overwrites a
   hand-authored `<stem>.criteria.json`.)

5. **Report** concisely: node count; per-node checks (subtype + which are `[probe-added]`);
   every `inferred_from` value; self-check pass/fail; and any point where the spec was
   ambiguous (so it can be tightened).

## Scope

- **Stage 1 only.** Binding claims to detectors/VLM (`<stem>.sensorplan.json`) is Stage 2 —
  a separate step (`tasks/criteria_to_sensorplan_template.json` +
  `tasks/PROCEDURE_MONITOR_COMPILER.md`), not this skill.
- **One recipe per run.** For a batch, loop the procedure per stem.
- `order` / `missing_step` are never authored as checks — the runtime emits them from DAG
  state (spec §2.5 / §4).
