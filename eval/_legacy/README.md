# eval/_legacy — archived runners

Superseded code, kept for reference only. Not on any live path; do not import.

## `sensorplan_runtime.py` (archived 2026-06-17)

Stand-alone causal runner for the **old transitions-form** sensor plan
(`procedure_start` + `transitions[]`). The `spicedhotchocolate.sensorplan.json` it
targeted was later rewritten into the **nodes form** (schema 2.0), so this script's
`load_plan` now crashes (`KeyError: 'procedure_start'`).

**Replaced by:** the `eval/proposed_*` pipeline. `proposed_plan_loader.load_plan()`
detects `artifact_type == "sensor_control_plan"` and compiles the nodes-form sensorplan
into the executable graph (`sensorplan_to_graph()`), then `proposed_runtime.py` drives it.
Same sensor-control behavior (D1 anchors settle fill/microwave/heat for free; the VLM is
duty-cycled to the silent middle).

**One bit not carried over:** this runner's order-enforced VLM used a *high-water clamp*
(a prediction that regresses below the high-water step is clamped forward). The
`proposed_` VLM arm does **not** clamp — fine for the hot-chocolate adds (an unordered
group), but if monotonic step labeling is wanted elsewhere, port the clamp from
`run_vlm_window()` here.
