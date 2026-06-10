# Related-Work Baselines We Can Test

## Bottom Line

Related work gives useful baselines for the first RGB-only cooking prototype. None directly tests our proposed cheap-RGB-criteria graph, but several define strong VLM/proactive-assistant baselines that we can replicate.

Recommended first baseline suite:

1. `VLM-Procedure`: periodic VLM observer with recipe/procedure context.
2. `PWR-3P`: prompt-only interrupt/silent assistant with optional plan state.
3. `Oracle-Plan PWR`: same assistant, but with ground-truth current plan/cues to isolate the value of plan quality.

These are enough to answer the first empirical question:

> Does a graph-gated RGB system reduce VLM calls while preserving stage tracking and proactive reminder quality compared with strong VLM-watching baselines?

## Candidate Baselines

| Source | Baseline to replicate | Inputs | Outputs | Why useful | Cost to test |
| --- | --- | --- | --- | --- | --- |
| Pro2Assist | `VLM-Procedure` | sampled RGB frames, recipe steps, task knowledge | step, execution status, proactive trigger, response | Closest to our current `VLM_BASELINE_SYSTEM.md`; simple and directly comparable | Low |
| Pro2Assist | `Vanilla ICL`, `ICL-EN`, `CoT` | sampled RGB frames, few-shot examples, optional expert knowledge | step/status/trigger/response | Tests whether prompting alone solves procedural assistance | Medium |
| Plan-Watch-Recover | `PWR-3P No Plan` | video clip, user goal, chat history | `$silent` or `$interrupt` + guidance | Strong prompt-only baseline for when-to-speak decisions | Low |
| Plan-Watch-Recover | `PWR-3P ZeroShot Planner` | video clip plus model-generated plan/cues | `$silent` or `$interrupt` + guidance | Tests whether generated procedural plans help a VLM decide interventions | Medium |
| Plan-Watch-Recover | `PWR-3P Oracle Planner` | video clip plus ground-truth plan/cues | `$silent` or `$interrupt` + guidance | Upper-bound baseline: if even oracle cues fail, the issue is VLM use, not plan quality | Medium |
| ProAssist | streaming video-to-dialogue model | video prefix, dialogue history, recipe/task knowledge | assistant utterance or silence over time | Released code/model; useful external system baseline | High |
| Wearable AI / EgoProactive | starter-kit proactive baseline | egocentric video intervals, query, dialog | `$silent` or `$interrupt` answers | Public benchmark format matches interrupt/silent evaluation | Medium to high |

## Most Useful Baseline: Pro2Assist VLM-Procedure

Pro2Assist defines a baseline adapted from a reactive procedural assistant. It extends the prompt to support visual inputs and proactive reasoning. In their real-world evaluation, this baseline uses periodic sampling to enable continuous perception.

For our cooking prototype:

- sample RGB every `N` seconds,
- provide the pan-fried egg recipe,
- ask the VLM for current step, execution status, completion, mistake/hazard, and assistant action,
- evaluate against annotated stage segments and valid reminder windows.

This is exactly the baseline already specified in [VLM_BASELINE_SYSTEM.md](VLM_BASELINE_SYSTEM.md).

Recommended initial settings:

- `vlm_interval_s`: 5 seconds, then 2 seconds for a denser version,
- `frames_per_call`: 3 to 8 frames,
- `temperature`: 0,
- output: strict JSON.

Metrics to borrow:

- `Step-Acc`: predicted step matches ground truth,
- `Status-Acc`: not started, in progress, about to finish, transition, complete,
- `Acc-P`: proactive trigger correctness,
- `MD`: missed detection,
- `FD`: false detection,
- `STS`: step-aware timeliness score,
- VLM call count and latency.

## Best Decision Baseline: PWR-3P

Plan-Watch-Recover defines a clean decision task:

```text
Given a video clip and task context, emit either:
$silent
or
$interrupt <short guidance>
```

This maps well to our proactive reminder problem. We can replicate three planning conditions:

1. `No Plan`: give only task goal and recent frames.
2. `ZeroShot Planner`: ask a VLM to generate the current plan and visual cues, then pass them to the assistant prompt.
3. `Oracle Planner`: give our manually annotated current step, next step, and visual completion/incompletion cues.

The oracle version is especially useful. It separates two questions:

- Is the plan/cue representation good enough?
- Can the VLM use those cues to make calibrated interrupt/silent decisions?

PWR's useful metrics:

- `IF1`: F1 for interrupt,
- `SF1`: F1 for silent,
- `G-Mean F1`: geometric mean of interrupt and silent F1, penalizing always-interrupt and always-silent collapse,
- `PQS`: proactive quality score combining correct silence and judged interrupt quality,
- recovery quality, if testing deviations.

## External Dataset Option

The Wearable AI / EgoProactive dataset is useful if we want a public interrupt/silent benchmark before collecting our own videos. Its Hugging Face card lists an `egoproactive` configuration with 700 validation examples, candidate decision intervals, user query, domain/task tags, dialog history, and reference answers in `$silent` or `$interrupt <utterance>` format.

Tradeoff:

- Pros: public schema, starter-kit code, direct interrupt/silent labels.
- Cons: access requires accepting dataset conditions; videos are large; it may not isolate cooking or our cheap-RGB criteria.

Use it later for external validation. Do not block the first cooking prototype on it.

## What Not To Use First

Avoid starting with:

- training VideoLLM-online or ProAssist models,
- downloading all released datasets,
- reproducing full Pro2Assist hardware/motion pipeline,
- evaluating broad egocentric procedural learning datasets.

Those are useful later, but they would delay the first clean experiment.

## Proposed First Experiment

Use our own pan-fried egg videos.

Annotate:

- stage segments,
- valid reminder windows,
- visible mistake/hazard windows,
- optional oracle visual cues for each stage.

Run three baselines:

1. `Periodic VLM-Procedure`: JSON stage/status/reminder output every 5 seconds.
2. `PWR No Plan`: `$silent`/`$interrupt` prompt with only task context.
3. `PWR Oracle Plan`: `$silent`/`$interrupt` prompt with current step and visual cues.

Then compare our future graph-gated RGB system against them on:

- VLM calls,
- stage accuracy,
- transition timing,
- reminder precision/recall,
- G-Mean F1 for interrupt/silent,
- latency and model cost.

## Takeaway For Novelty

These baselines help sharpen the claim:

> Prior systems ask a VLM to continuously infer procedural state and intervention timing. Our system uses a compiled graph to decide when cheap RGB criteria are sufficient and when VLM reasoning is worth the cost.

The baseline comparison should show whether the graph reduces VLM usage without losing the intervention quality that VLM-watching systems provide.

## Source Notes

Local papers reviewed:

- `procedural/pro2assist_proactive_procedural_assistance_2605.04227.txt`
- `procedural/plan_watch_recover_2606.04970.txt`
- `procedural/proactive_assistant_dialogue_generation_2506.05904.txt`
- `procedural/building_egocentric_procedural_ai_assistant_2511.13261.txt`
- `procedural/PROJECT_MEMORY.md`

External availability checked:

- Wearable AI / EgoProactive dataset and starter kit: https://huggingface.co/datasets/facebook/wearable-ai
- ProAssist project page: https://pro-assist.github.io/
- ProAssist code and model/data instructions: https://github.com/pro-assist/ProAssist
