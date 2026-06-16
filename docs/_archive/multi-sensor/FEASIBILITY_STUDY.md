# Feasibility Study Plan

## Study Question

Can an executable sensing-and-assistance graph reduce VLM usage while preserving task-stage tracking and reminder quality? Start with RGB-only sensing, then extend to multisensor scheduling if the representation works.

## Minimal Prototype

Start with one RGB-only task:

- pan-fried egg, using [COOKING_CASE_STUDY.md](COOKING_CASE_STUDY.md) as the first manual graph, or
- [RGB_ONLY_START.md](RGB_ONLY_START.md) as the first simplified prototype path.

Implement:

- a manually authored executable graph with 8-12 stages,
- 3-5 lightweight RGB criteria,
- a stage-belief tracker,
- RGB sampling-rate or burst simulation,
- sparse VLM escalation,
- replay-based evaluation before live smartglass interaction.

## Required Graph Fields

Each stage should include:

- `goal`: what the user is trying to accomplish,
- `criteria`: simple observable progress/completion checks,
- `cheap_checks`: sensors or lightweight RGB methods that can test those criteria,
- `full_sensing_needed_when`: uncertainty, semantic ambiguity, conflict, or safety risk,
- `reminder_condition`: when to intervene,
- `safety_condition`: when to warn immediately.

Example:

```json
{
  "stage": "cook_white",
  "goal": "cook until the egg white is mostly set",
  "criteria": ["egg white becomes more opaque", "elapsed time reaches expected window"],
  "cheap_checks": ["low-rate RGB color/texture tracking", "timer"],
  "full_sensing_needed_when": ["doneness is ambiguous", "cheap visual criteria conflict"],
  "reminder_condition": "expected doneness window starts",
  "safety_condition": "darkening suggests possible burning"
}
```

## Example Cheap RGB Policies

| Stage | Simple criteria | Cheap RGB checks | Escalate when |
| --- | --- | --- | --- |
| Setup pan | pan visible in stove ROI | object/region localization | pan/burner not localized |
| Add oil | hand/container motion over pan | motion + surface appearance change | oil/butter identity matters |
| Add egg | egg-like region appears | region/color change in pan | egg/shell identification uncertain |
| Cook egg | white becomes more opaque | color/texture trend + timer | doneness preference matters |
| Plate egg | egg moves to plate ROI | object tracking between regions | tracking lost |
| Turn off stove | hand reaches control area | motion near control ROI | control state is occluded |

## Baselines

- Continuous or fixed-interval VLM, specified in [VLM_BASELINE_SYSTEM.md](VLM_BASELINE_SYSTEM.md).
- Related-work baselines from [RELATED_WORK_BASELINES.md](RELATED_WORK_BASELINES.md), especially Pro2Assist-style `VLM-Procedure` and PWR-style interrupt/silent prompting.
- VLM at every procedure step.
- Cheap RGB criteria without graph-guided scheduling.
- Human-authored policy versus MLLM-generated policy, if studying graph generation.

## Metrics

- Stage classification accuracy.
- Stage transition timing error.
- Safety-event precision, recall, and warning latency.
- Reminder precision and missed-reminder rate.
- VLM calls per task and per minute.
- RGB frame rate or sampled-frame count.
- Estimated model cost or latency.
- End-to-end latency from event to assistant action.
- User interruption burden, if a user study is included.

## Data Collection

Minimum viable dataset:

- 2 tasks,
- 8-12 stages per task,
- 5-10 repeated runs per task,
- RGB video,
- ground-truth stage boundaries,
- labels for safety events, reminders, deviations, pauses, and repeated steps.

## Expected Feasibility Evidence

A promising result would show:

- comparable stage tracking to fixed-interval VLM,
- fewer VLM calls,
- lower estimated model cost or latency,
- useful reminders based on cheap RGB criteria,
- qualitative examples where the graph chooses cheap RGB checks before VLM.

## Main Risks

- Stage criteria may be too vague to check with cheap RGB methods.
- Users may skip, repeat, pause, or interleave steps.
- RGB criteria may fail under occlusion, lighting changes, camera motion, or transparent ingredients such as oil.
- Reminder timing may be technically correct but annoying.
- The graph generator may produce plausible but unsafe policies without review.

## Next Steps

1. Pick one task domain.
2. Write one executable graph by hand.
3. Record 5-10 RGB task videos.
4. Build offline detectors and a replay tracker.
5. Compare graph-guided sensing against periodic VLM.
6. Use failures to refine the stage criteria and escalation rules.
