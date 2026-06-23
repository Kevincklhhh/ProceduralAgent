# Archived docs

These are kept for history only — **not authoritative**. Most predate the **2026-06-15
three-box restructure** (`../PIPELINE_THREE_BOXES.md`); the 2026-06-20 batch predates the
two-stage criteria→sensorplan templates. Do not cite them as current design.

| Archived | Superseded by |
|---|---|
| `CONVERSION_AND_EVAL_PROTOCOL.md` | Split into the three boxes: predictor → the two-stage templates + `tasks/PROCEDURE_MONITOR_COMPILER.md`; GT (step C5) → `../FAMILY_A_CC4D_AUGMENTATION.md` + `eval/gt_build_family_a.py`; evaluation (Part 2) → `../REMINDER_EVALUATION.md`. Kept for the activity-8 worked example. |
| `SENSOR_GRAPH_COMPILER_PROMPT.md` (2026-06-20) | Box-3 authoring is now the **two-stage templates** (`tasks/recipe_to_criteria_template.json` → `tasks/criteria_to_sensorplan_template.json`) + the Stage-2 guide `tasks/PROCEDURE_MONITOR_COMPILER.md`; runtime is `../REMINDER_RUNTIME.md`. This was a single-stage "exploration" prompt referencing the pre-split `task_spiced_hot_chocolate_cc4d.json` and inline A1–A10 detectors. |
| `SENSOR_SCHEDULE_PILOT.md` (2026-06-20) | First hand-authored, GT-aware pilot policy for hot chocolate (reminders deferred). Superseded by the firewall-clean `tasks/cc4d/spicedhotchocolate.sensorplan.json` + the runtime (`eval/proposed_runtime.py`). |
| `TASK_DEFINITION.md` (2026-06-20) | Class-typed reminder + GT schema → `../FAMILY_A_CC4D_AUGMENTATION.md` §4 / `../REMINDER_EVALUATION.md`; the durable sensor-control I/O & cost contract was folded into `../PROJECT_BACKGROUND.md` §2a. |
| `multi-sensor/` (7 files, 2026-06-10) | Early brainstorming from before the project narrowed to RGB+audio **sensor control**. The graph-gated-VLM framing, the multi-sensor/thermal routing idea, and the B0–B4 baseline plans are reorganized into the three boxes. `multi-sensor/PROJECT_MEMORY.md` is a narrower duplicate of `../PROJECT_MEMORY.md`. |

What changed: thesis moved from error detection / graph-gated stage-tracking to **sensor control (energy/latency at equal coverage)**; GT became **mechanical-only** (order, temperature-power-level suspended; safety, next-step guidance, preventive windows excluded); the predictor and the GT answer-key were firewalled into separate pipelines.
