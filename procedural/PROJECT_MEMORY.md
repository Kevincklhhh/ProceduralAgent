# Procedural Wearable Agent Project Memory

This file is a living memory for the next-project direction. It records what we are trying to do, what we are intentionally not trying to do, and the reasoning behind those choices.

## Current North Star

We are exploring a wearable / smart-glasses / egocentric AI agent for procedural task assistance.

The interesting agentic behavior is not merely recognizing what the user is doing. It is deciding when to stay silent, when to intervene, what to say, how to recover from deviations, and how to maintain useful task memory over time.

The strongest current framing:

> A wearable procedural agent that supports human-performed tasks by maintaining procedural state, detecting meaningful deviations, calibrating intervention timing, and producing recovery guidance.

## Vision

Imagine a smart-glasses agent that helps a person carry out unfamiliar or high-attention procedures in the physical world.

Example scenarios:

- Learning to cook an unseen dish: the user has never made the recipe before. The agent watches their hands, ingredients, cookware, and progress; stays quiet during normal execution; intervenes when the user is about to miss a consequential step; and helps recover when the user improvises, substitutes, or gets distracted.
- Engineering / research lab: the user follows a protocol involving tools, samples, reagents, instruments, or safety checks. The agent does not replace the protocol or the human; it tracks where the user is, verifies that required preconditions are satisfied, catches recoverable deviations, and helps produce an auditable record.
- Field repair or assembly: the user performs a physical procedure with parts, tools, and manuals. The agent keeps task state, retrieves relevant instructions, notices when the user is blocked or off-plan, and documents what was completed.
- Clinical self-care or medication routines: the agent helps users follow recurring procedures with high cost for omissions or mix-ups, while preserving human confirmation and caregiver escalation when appropriate.

The broader vision is not "a VLM that recognizes actions better." It is a personal procedural agent that turns egocentric perception into situated assistance: when to speak, when to stay silent, when to ask, when to recover, what to remember, and what external record or tool state to update.

## Why This Seems Worth Studying

Coding agents work because they operate in a workspace with artifacts, tools, state, history, and tests. A wearable procedural agent needs an equivalent structure in the physical world.

Procedural tasks provide that structure:

- There is a goal.
- There are expected steps.
- There is observable progress.
- There are possible mistakes and deviations.
- There are moments where intervention helps.
- There are moments where interruption is harmful.
- There may be logs, checklists, documentation, or external tools to update.

This is more compelling than a generic memory reminder because the value can be risk reduction, error recovery, documentation, compliance, accessibility, or training.

## ThermalKitchens Pivot

We have an existing smart-glasses cooking dataset at `/Users/kailaicui/Desktop/thermal_kitchen`, with paired RGB and thermal evidence, raw thermal bins, quiz/evaluation code, and prior experiments around RGB-vs-thermal modality routing.

This may provide a stronger "nail" than object-centric memory. Thermal sensing exposes physical cooking states that RGB/action history often cannot determine reliably: heat readiness, residual heat, cooling, uneven heating, touch safety, doneness proxies, and active heating. This creates a real reason for a wearable assistant to manage multimodal evidence instead of simply asking a VLM to infer everything from RGB video.

Important caveat from existing notes: this should not become "better thermal VQA by prompt tuning." Prior experiments in `Kitchen_Method` suggest that current VLMs struggle with thermal/RGB conflict, object-region grounding, tool-readout interpretation, and cooking physics. That means the interesting procedural-agent angle is likely not raw accuracy alone, but deciding when thermal evidence is needed, how to ground it to the right moment/object, and when that evidence should change an intervention decision.

Candidate framing:

> A thermal-aware wearable cooking assistant that decides when RGB is enough, when thermal sensing is necessary, and when heat-state evidence justifies intervention.

Candidate evaluation target:

> Holding the VLM mostly fixed, does modality-aware context/invocation/intervention policy improve task-relevant decisions such as "wait", "continue", "warn", "ready for next step", or "unsafe to touch" under bounded latency and limited sensing budget?

## What We Want To Do

- Work above the perception layer: the project should be about agent behavior around human-performed procedural tasks, not about making the base VLM better at recognizing task state.
- Keep the human as the physical actor. The wearable agent acts through guidance, memory, verification, documentation, and digital tools.

## What We Do Not Want To Do

- Do not make improved task-state perception accuracy the main contribution. Leave that to foundation-model researchers with VLM-scale training resources.
- Do not drift into robotics / VLA. If the main action is physical motor control, it is probably outside our intended scope.

## Boundary With Robotics / VLA

VLA: the robot is the actor.

Wearable procedural agent: the human is the actor. The AI acts through:

- guidance
- silence
- recovery suggestions
- memory updates
- checklist updates
- documentation
- reminders
- digital tool use
- escalation to a human caregiver / supervisor when appropriate

This boundary matters because otherwise the project becomes a robotics control paper.

## Related Work Snapshot

Direct-hit papers currently in this folder:

- `plan_watch_recover_2606.04970.pdf`
- `pro2assist_proactive_procedural_assistance_2605.04227.pdf`
- `proactive_assistant_dialogue_generation_2506.05904.pdf`
- `building_egocentric_procedural_ai_assistant_2511.13261.pdf`

Working read:

- `Plan, Watch, Recover` is the closest current work. It introduces Pro2 Bench and EgoProactive, evaluates interrupt vs silent decisions, guidance quality, and out-of-plan recovery. Its benchmark structure is useful, but much of the evaluation is offline causal replay over preprocessed decision windows. EgoProactive also uses scripted deviations.
- `Pro2Assist` is close on system framing: continuous step-aware proactive assistance with AR glasses, curated procedural data, and a small real-world user study. It evaluates step/status accuracy, proactive trigger accuracy, timing, response quality, latency, power, and subjective usefulness.
- `Proactive Assistant Dialogue Generation` focuses on synthetic proactive assistant dialogues from existing egocentric/procedural datasets. It is useful for dialogue/timing evaluation, but it is still mostly dataset-driven generation rather than grounded deployment.
- `Building Egocentric Procedural AI Assistant` surveys the space and stress-tests VLMs on procedural error detection and procedural learning. It reinforces that generic VLMs are weak at procedural assistance, but we should not make base perception accuracy our main battleground.

Important observation: much of this area evaluates "streaming" through offline causal replay over preprocessed decision windows. That leaves room for work on real online constraints, persistent memory, latency, uncertainty, and interaction cost.

## Bottlenecks We Care About

- Intervention policy: deciding when to speak, stay silent, ask a clarification, or escalate. False-positive interruptions are costly; false-negative silence can also be costly.
- Recovery, not just error detection: after a deviation, the agent must know whether the task is still recoverable and what the user should do next.
- Interaction-burden evaluation: metrics should separately measure unnecessary interruptions, missed useful interventions, late interventions, and harmful advice.
- System-resource evaluation: latency, power, bandwidth, model-call count, and sensing duty cycle should be measured separately from interaction burden.
- Real online behavior: the agent should work under bounded latency, incremental context, and partial evidence rather than only offline replay over selected windows.
- Robustness to imperfect plans: the task plan may be incomplete, wrong, user-adapted, or underspecified.
- Human workflow fit: the agent should support task completion without over-narrating, over-controlling, or increasing cognitive load.
- Memory and documentation: the agent may need to remember completed steps, evidence, exceptions, user preferences, and what should be logged externally.

## Bottlenecks We Want To Avoid

These are real bottlenecks, but they are not where we want the core contribution to live.

- Base VLM perception accuracy: improving low-level task-state recognition, object recognition, hand-object tracking, or fine-grained action classification should not be our main claim.
- Foundation model scaling: training a better general video-language model is out of scope.
- More frames / bigger context as the solution: simply feeding more video into a larger VLM is not an agent contribution.
- Generic video QA or captioning: answering questions about video after the fact is different from assisting a human during a procedure.
- Pure robotics / VLA control: if the central output is a motor command for a robot body, the project has crossed into robotics.
- Prompt-only procedural assistant: a stronger prompt around an existing VLM is unlikely to be enough as a research contribution.

## Potential Openings

- Non-scripted deviations rather than scripted out-of-plan clips.
- Cost-sensitive intervention policies instead of balanced F1 alone.
- Human-cost evaluation: interruption burden, trust, workload, and recovery success.
- Domains where errors matter: medication routines, lab procedures, clinical self-care, sterile caregiving, field-service documentation, safety/compliance checks.
- Real online behavior: bounded latency, incremental state updates, recovery from earlier mistaken beliefs.
- Robustness to imperfect plans and imperfect perception.
- Learning when to ask a clarification instead of giving an instruction.

## Current Skepticisms

- If the application is only convenience, people will not tolerate latency and errors.
- If the contribution is just perception accuracy, we are competing with foundation-model scale.
- If the task is too broad, the project becomes a demo rather than research.
- If evaluation ignores interruption cost, the assistant may look good in metrics but feel bad to users.
- If deviations are scripted too cleanly, the benchmark may not capture real human messiness.

## Next Questions

- What domain gives us a strong motivation beyond convenience?
- What action space should the agent have beyond speaking?
- Can we define a better evaluation target than step accuracy or interrupt F1?
- How can we model "expected cost of silence" vs "expected cost of interruption"?
- What should the agent remember across minutes, days, or repeated attempts?
- Can we build a benchmark where recovery, uncertainty, and interruption cost are unavoidable?
