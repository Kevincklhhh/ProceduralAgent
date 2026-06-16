# Procedure-Guided Multisensor Smartglasses

## Core Idea

Build a smartglass assistant that compiles a tutorial, recipe, protocol, or educational video into an executable sensing-and-assistance graph.

The graph is not only a task dependency graph. Each stage also defines:

- the user's current task goal,
- lightweight criteria for checking progress or completion,
- sensors needed for those criteria,
- when to escalate to RGB/VLM or full sensing,
- proactive reminder and safety-warning conditions.

Theme:

> Procedure structure can reduce wearable sensing cost by telling the system which simple evidence is enough at each stage, and when expensive multimodal reasoning is actually needed.

## Key Feasibility Assumption

Sensor or model gating is feasible only if the executable graph decomposes the procedure into stages with simple local goals and checkable criteria. Some stages should be verifiable without full sensing or VLM reasoning:

- "pan temperature is rising" can use thermal only,
- "water is boiling" can use audio or thermal,
- "soldering iron is hot" can use thermal,
- "microwave is running" can use audio,
- "user may be doing the wrong object/action" may need RGB/VLM escalation.

If a stage requires open-ended scene understanding, the system should not pretend a cheap sensor is enough. It should activate richer sensing or ask for confirmation.

The first prototype can be RGB-only: use cheap RGB checks, such as motion, region tracking, color change, and timers, and call the VLM only when those checks are ambiguous. Multisensor scheduling can come after this representation works.

## System Shape

Compiler input:

- procedure source: text, manual, recipe, protocol, or tutorial video,
- device profile: sensors, power, latency, sampling rate, startup cost,
- user/task context: skill level, environment, safety goals,
- optimization goal: energy, latency, accuracy, privacy, or interruption burden.

Compiler output:

- `ExecutableGraph`: stages, dependencies, goals, criteria, and transitions,
- `SensorPolicy`: which sensors to run for each stage and at what rate,
- `EscalationPolicy`: when to call VLM or enable full sensing,
- `FeedbackPolicy`: reminders, warnings, confirmations, and teaching hints.

Runtime output:

- current stage belief,
- sensor on/off or sampling-rate commands,
- sparse VLM requests,
- proactive reminders or safety warnings,
- execution log for latency, energy, and errors.

## Runtime Loop

1. Track a belief over possible current stages.
2. Use the executable graph to choose the cheapest sensor or model that can test the current stage criteria.
3. Escalate to RGB/VLM when cheap criteria are uncertain, conflicting, safety-critical, or semantically underspecified.
4. Trigger reminders or warnings only when the graph says the stage goal, timing, or safety condition justifies intervention.
5. Log decisions to evaluate stage accuracy, energy, latency, VLM usage, and interruption burden.

## Novelty Claim

The novelty is not simply turning tutorials into task graphs. Related work already studies procedural step graphs, instructional video graphs, and task progress tracking.

The proposed novelty is compiling a tutorial into an executable sensing-and-assistance graph: a representation that jointly defines task dependencies, sensor usage, VLM escalation, and proactive reminder criteria.

## Initial Scope

Best first target: RGB-only cooking assistance for one simple dish. This tests whether procedure-derived criteria can reduce VLM calls before adding thermal/audio sensors.

Avoid starting with "all educational procedures"; that scope is too broad and makes the sensor criteria unclear.

## Open Decisions

- First domain: cooking, soldering, lab protocol, repair, or another hands-on task?
- Main contribution: adaptive sensing, procedural assistance, or graph generation?
- Hardware: real smartglasses, external synchronized sensors, or replay-based prototype?
- Compute: on-device, phone, edge, or cloud VLM?
- Feedback channel: audio, visual overlay, haptic cue, or phone notification?
- Evaluation priority: energy, latency, stage accuracy, safety, or user interruption burden?

## Supporting Notes

- [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md): related-work positioning, professor discussion points, and idea boundaries.
- [FEASIBILITY_STUDY.md](FEASIBILITY_STUDY.md): first prototype, metrics, baselines, data collection, and risks.
- [COOKING_CASE_STUDY.md](COOKING_CASE_STUDY.md): manual pan-fried egg example mapping recipe steps to sensors, criteria, escalation, and reminders.
- [RGB_ONLY_START.md](RGB_ONLY_START.md): first prototype path using cheap RGB criteria instead of non-RGB sensors.
- [VLM_BASELINE_SYSTEM.md](VLM_BASELINE_SYSTEM.md): baseline where a VLM periodically watches cooking and outputs stage, completion, mistakes, reminders, and cost logs.
- [RELATED_WORK_BASELINES.md](RELATED_WORK_BASELINES.md): related-work baselines to replicate, including Pro2Assist-style VLM-Procedure and PWR interrupt/silent prompting.
