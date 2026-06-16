# Archived docs

These predate the **2026-06-15 three-box restructure** (`../PIPELINE_THREE_BOXES.md`) and are
kept for history only — **not authoritative**. Do not cite them as current design.

| Archived | Superseded by |
|---|---|
| `CONVERSION_AND_EVAL_PROTOCOL.md` | Split into the three boxes: predictor → `../SENSOR_GRAPH_COMPILER_PROMPT.md`; GT (step C5) → `../FAMILY_A_CC4D_AUGMENTATION.md` + `eval/build_family_a_gt.py`; evaluation (Part 2) → `../REMINDER_EVALUATION.md`. Kept for the activity-8 worked example. |
| `multi-sensor/` (7 files, 2026-06-10) | Early brainstorming from before the project narrowed to RGB+audio **sensor control**. The graph-gated-VLM framing, the multi-sensor/thermal routing idea, and the B0–B4 baseline plans are reorganized into the three boxes. `multi-sensor/PROJECT_MEMORY.md` is a narrower duplicate of `../PROJECT_MEMORY.md`. |

What changed: thesis moved from error detection / graph-gated stage-tracking to **sensor control (energy/latency at equal coverage)**; GT became **mechanical-only** (order, temperature-power-level suspended; safety, next-step guidance, preventive windows excluded); the predictor and the GT answer-key were firewalled into separate pipelines.
