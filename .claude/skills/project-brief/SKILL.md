---
name: project-brief
description: Keeps the optimization goal straight and enforces a plain-language wrap-up. Load at the START of a task to recall what we're actually optimizing (sensor control — energy/latency at equal coverage, NOT error-detection accuracy), and at the END to report implementation/analysis results in plain words with jargon defined. Use when the user asks "what are we optimizing", or to summarize/explain/report results.
---

# Project Brief — goal + how to report

Apply the relevant half: **§1 at the start of a task** (keep the thesis straight), **§2 when a task is done** (how to report). Don't restate this file to the user — apply it. (Project setup/structure lives in project memory and `docs/`, not here.)

## 1. The optimization goal — keep the thesis straight

The contribution is **sensor CONTROL**, measured as **energy/latency saved at equal coverage** — *not* "% of errors detected." Given the procedure ahead of time, plan per stage: which cheap RGB+audio detectors run vs. sleep (energy), where a cheap detector gives a verdict without the VLM (latency), and where a cheap event should **trigger one** VLM call instead of running it continuously.

> **Current framing (2026-06-28):** the live arm sharpens this to **reminder latency** — the win is *low latency from evidence → spoken reminder* (constant ~133 ms trigger via pre-encoded prefill + bounded claim checks), with *reminder accuracy* as the open risk. Sensor control is the substrate; latency is the headline; accuracy is unsolved. "Not error-detection accuracy" still means *don't headline a detection-% number* — it does **not** mean accuracy is irrelevant. Full framing + open problems: `docs/LATENCY_STORY.md`.

Classify each (stage, anticipated error) into a sensing role:
- **A — solve:** a cheap sensor settles it alone (timing via audio, appliance state, precondition/order via graph state). The real win slice.
- **B — trigger:** a cheap event fires **one** targeted VLM call (duty-cycled, not continuous).
- **C — none:** no cheap solve/trigger (silent fine technique, continuous amounts). **Report this fraction honestly — don't hide it.**

Headline must read "sensor scheduling at X% of always-on-VLM energy/latency at equal coverage," never "audio solves cooking." Report **per-class, never pooled** (most errors are visual-leaning; pooling flatters the audio arm).

## 2. When the task is done — report in plain language

Before you call a task finished, deliver a wrap-up that a non-specialist on this project could follow:

1. **Plain language first.** Lead with what you did / found in ordinary words and why it matters to the goal in §1. No wall of identifiers.
2. **Define every piece of jargon on first use** — inline, one clause. This includes our own terms: *window, reactive vs. preventive, the firewall, A-solve/B-trigger/C-none, mechanical-only, excluded-by-design, decision point, G-Mean F1, STS*. If a term isn't worth defining, it isn't worth using.
3. **Ground claims in what actually ran.** State what you executed and verified vs. what you only reasoned about. Quote real numbers/outputs, not expected ones. If a step was skipped or failed, say so plainly.
4. **Be honest about limits.** Name the fraction that doesn't work (e.g. the C-none slice, low-recall classes like temperature power-level, un-run steps) rather than rounding up to "done."
5. **Use a concrete example** when explaining a derivation or result (a real recording id, a real event), the way `data/cc4d_family_a/8_45.json` makes the GT tangible.
6. **Offer the next decision, not a survey.** End with a recommendation or a single focused question, not an exhaustive menu.

Litmus test before sending: *would the user understand this without already knowing the codebase, and is every claim something I actually verified this session?*
