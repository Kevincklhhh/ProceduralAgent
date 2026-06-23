# tasks/_archive — superseded templates

Kept for reference only; do not author new artifacts from these.

## `procedure_monitor_template_0.6_STALE.json` (archived 2026-06-17)

The original **single-hop** procedure-monitor template (schema 0.6): it did stages 1 and 2
(recipe→criteria→sensor plan) in one combined executable artifact.

**Replaced by** the two-stage, nodes-form templates (both schema 2.0):
- `tasks/recipe_to_criteria_template.json` — Stage 1 (recipe → criteria)
- `tasks/criteria_to_sensorplan_template.json` — Stage 2 (criteria → sensor control plan)

Known defect that motivated the split: it conflated a step's *completion* with the next
step's *start* (one `D1.cycle_start` served as both `fill_milk.complete` and
`microwave_initial.start`) with no field marking the borrowed-anchor completion as an
inference. The nodes form makes that explicit via `completion.inferred_from`.

See `tasks/PROCEDURE_MONITOR_VERSIONS.md` for the full pipeline-stage narrative. The
worked 0.6 instance `tasks/cc4d/spicedhotchocolate.monitor.json` is left in place as the
visualizer fallback plan and as the byte-identical golden reference for
`proposed_plan_loader.sensorplan_to_graph()`.
