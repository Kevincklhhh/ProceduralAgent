# Probing prior proactive-reminder work by implementation (2026-06-12)

Script: `eval/probe_prior_work.py` → `results.json`. Re-implements the LiveMamba/Qualcomm
protocol (IC-Acc = completion detected within a 30 s window centered on GT; mistake P/R/F1,
same window, one-to-one greedy matching, untyped), probes it with GT-replay baselines,
re-scores our replay_v1 arms under it, and replays PREGO-style graph-logic mistake
detection with oracle perception over all 384 CC4D recordings.

## E1 — What the LiveMamba protocol rewards (test split, n=109)

| baseline | IC-Acc | mistake P | R | F1 |
|---|---|---|---|---|
| (paper) Gemini-2.5-Flash zero-shot | .231 | | | .02 |
| (paper) LiveMamba (trained) | .315 | | | .13 |
| always silent | .000 | 0 | 0 | .00 |
| **chatty (alert every 10 s)** | **1.000** | .05 | 1.00 | **.09** |
| **open-loop timer (train-split duration priors, zero perception)** | **.387** | – | – | – |
| oracle | 1.000 | 1.00 | 1.00 | 1.00 |
| oracle + ±15 s jitter | .986 | | | .99 |
| oracle + ±30 s jitter | .604 | | | .54 |
| oracle + ±60 s jitter | .464 | | | .39 |

Findings:
1. **A zero-perception timer beats the published SOTA**: per-recipe mean step durations
   (train split) predict completions at IC-Acc .387 vs LiveMamba's .315 and Gemini's .231.
   The timer emits exactly one completion per step index in order, so this stands even
   under stricter typed matching.
2. **LiveMamba's headline mistake F1 (.13) is barely above always-alert (.09)** under the
   stated protocol. The mistake-detection problem is effectively unsolved by end-to-end
   streaming models.
3. **The protocol's effective bar is ±15 s completion timing** (scores collapse between
   ±15 and ±30 s jitter) — comfortably within audio-detector tolerance (beep/hum offsets
   are ~1 s precise).
4. Caveat: our IC-Acc matching is untyped-by-time (chatty's 1.000 shows it is gameable);
   the paper's variant may require naming the instruction. The timer and arm numbers are
   robust to this; the chatty row is the upper bound of gaming.

## E2 — Our existing arms under their protocol (activity 8, 6 recordings — subset, not directly comparable)

| arm | IC-Acc | mistake P/R/F1 |
|---|---|---|
| detector_replay (audio detectors + graph) | **.514** | **.44 / .31 / .36** |
| detector_plus_escalation | .514 | .18 / .31 / .23 (escalation events count as FPs untyped) |
| periodic_vlm_qwen (10 s interval) | .243 | .00 / .00 / .00 |

- Our periodic Qwen arm lands at IC-Acc .243 ≈ the paper's zero-shot Gemini (.231) —
  external calibration that the re-implementation is in the right regime.
- The frozen audio-detector arm, designed for a different truth table, already clears the
  published bar on both axes (on this subset). Its cost: detector CPU-seconds vs the Qwen
  arm's 53.7 s mean latency per call at a 10 s decision interval (not live-runnable).

## E3 — PREGO with oracle perception + task-graph anticipator (all 384 recordings)

| target | P | R | F1 |
|---|---|---|---|
| Missing steps (at recording end) | .99 | 1.00 | 1.00 |
| Order-error tags (step-level) | .43 | .60 | .50 |
| Any error tag | .66 | .44 | .53 |

- **Missing-step detection is solved mechanically** (286/286 recovered, 2 FPs).
- **Order-error GT misaligns with DAG semantics in both directions** even with oracle
  perception: 315 order-tagged steps are graph-feasible (tags issued relative to the
  sampled linear order, not the DAG) and 637 graph violations carry no order tag
  (annotator granularity/cascades). F1 .50 is the *ceiling* for pure logic against raw
  tags → the benign-vs-harmful adjudication pass over the 795 order tags is not optional
  polish; it is the annotation contribution.
- Execution errors are invisible to logic, as expected (measurement 97/331 "caught" only
  via coincidental co-occurring violations) → escalation arm required for class 4.

## Bottleneck synthesis (what implementation taught us that reading didn't)

| prior-work bottleneck | evidence here |
|---|---|
| End-to-end models below trivial baselines on their own protocol | timer .387 > LiveMamba .315; chatty F1 .09 ≈ LiveMamba .13 |
| Decision discipline, not perception, drives mistake-F1 | chatty has R=1.00 but P=.05; periodic Qwen emits ~nothing matching |
| Latency makes VLM arms non-live | 53.7 s/call at 10 s interval (measured, our hardware) |
| GT itself is the obstacle for order errors | E3 both-direction misalignment, F1 ceiling .50 |
| The real sensing target | completion events within ±15 s — exactly the regime of cheap audio/timer detectors + duration priors |
