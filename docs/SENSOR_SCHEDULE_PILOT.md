# Sensor Schedule — Pilot (Spiced Hot Chocolate, manual GT-aware) — 2026-06-15

First hand-authored sensor-control policy: **per-stage sensor usage + step-completion
criteria** for the activity-8 pilot recipe. Scope for now is (1) stage tracking and (2)
marking steps complete; **proactive reminders are deferred.** Machine-readable form:
`data/sensor_schedule_spicedhotchocolate.json`. Primitive ids: `tasks/AUDIO_RUNTIME_LIBRARY.md`.

> **Firewall note.** This is authored *with* ground-truth awareness (verified against the
> 8_16 clean pilot + detector probes), so it is a **reference / human-authored arm**, not a
> Box-3 predictor output. The rules are recipe-structure-driven (see below), so an automated
> Box-3 should be able to reproduce an equivalent from recipe text + DAG alone — but until it
> does, do not score this policy as the predictor on the recordings whose GT informed it.

## Control space

Per stage we set two knobs: **audio** ∈ {on, off} and **RGB fps** ∈ {0, 1, 2} (0 = sensor
off). Audio DSP shares one always-on 16 kHz front-end (<0.1% of a CPU core) so "audio on" is
nearly free; RGB capture/decode is the real energy lever; the VLM (V1) is trigger-only and
never duty-cycled. Detector libraries already exist: **audio A0–A10**, **RGB R1–R5** in
`tasks/AUDIO_RUNTIME_LIBRARY.md`.

## The structural fact this recipe exploits

Stages **alternate quiet ↔ loud**: `fill (quiet) → microwave (loud) → adds (quiet) → mix
(loud) → microwave/serve (loud)`. The two microwave stages give clean 59 s hum runs + end
beeps (probed on 8_16: 0 false), and mix gives a validated stir clink-train. So every *quiet*
step's completion can be pinned by the *next loud anchor* ("next-anchor rule") instead of
positively detecting the quiet action — e.g. *fill is done when the microwave hum starts*.
Every step also has a timer fallback (A10).

## Per-stage schedule

| Stage (step) | Audio | RGB fps | Step-complete criteria (first to fire) |
|---|---|---|---|
| fill_milk (88) | on (A0,A6) | **0** | hum onset A1 (next-anchor) · timer 120 s |
| microwave_initial (89) | on (A1,A2) | **0** | hum offset A1 (run≥20 s) · end beep A2 · timer 90 s |
| add_chocolate (90) | on (A3) | **1** (R3,R4) | hand→mug episode + ROI state-change · timer 90 s |
| add_cinnamon (84) | on (A3) | **1** (R3,R4) | hand→mug episode + ROI state-change · timer 90 s |
| add_sugar (87) | on (A3) | **1** (R3,R4) | hand→mug episode · stir onset A3 (next-anchor closes block) · timer 90 s |
| mix (85) | on (A3) | **0** | clink-train offset A3 · next hum onset A1 · timer 120 s |
| heat_serve (83) | on (A1,A2,A6,A9) | **0** | hum offset / beep · serve-pour + idle A9 (recipe end) · timer 90 s |

## Why each RGB decision

- **RGB off for fill / microwave / mix / heat_serve.** Fill completion = next anchor.
  Microwaves are pure audio A-solve (user walks away — off-view). Stir-vision *failed*
  (global motion AUC ≈ chance); ROI stir would need expensive ≥10 fps bursts for nothing the
  clink-train doesn't already give.
- **RGB on (1 fps) only for the adds block.** The three adds are structurally quiet (audio
  can't mark them) and visually similar. 1 fps + hand-gate (R3) + ROI state-change (R4)
  segments the add *episodes* to mark each add step complete. Ingredient *identity* is not
  decided here — that escalates to one V1 (VLM) call only if a later reminder needs it.

## Energy result (8_16)

RGB capture runs for only the adds block — **~162 s of the 452 s recipe ≈ 36% duty cycle at
1 fps**; the other ~64% is audio-DSP-only. **Zero VLM calls** in the base policy. This is the
"sensor scheduling at X% of always-on-VLM energy at equal coverage" headline in concrete
form for one recipe.

## Probe result — the adds RGB criterion, tested on all 16 SHC recordings (2026-06-15)

`detectors/probes/probe_rgb_adds.py` (tuned on 8_16, frozen, eval on the other 15;
`results_rgb_adds.json`) tested the per-add completion criterion at 3 fps, tight mug ROI:

- **R3 hand-gate (absolute YCrCb skin %) FAILED.** Skin-colored surfaces (wood counter,
  milk, beige mug) hold skin% ≈ 0.04–0.46 baseline, so an absolute threshold fires the whole
  window — no clean hand-episode bracketing. R3-as-skin is dead on this footage.
- **R4 (mug-ROI HSV state-change) WORKS for detection, over-segments for counting.**
  Pooled **recall 0.953 (41/43 adds** have a state-change peak in window); **precision 0.418**
  (98 peaks for 43 adds — ~2.3 peaks/add from hand occlusion + re-positioning). 15/16
  recordings hit 100% recall; only **8_11** missed (1/3 — weak ROI signal).
- **The count prior doesn't rescue precision.** "Expect 3, take 3 strongest peaks" fails
  even on 8_16 (the strongest peak, chi²=5.7 @299 s, is spurious/post-window; top-3-by-chi²
  would miss the sugar add). Reliable *fine per-ingredient* completion needs a real
  hand-episode gate (R3 beyond skin color — medium cost) or a VLM escalation.

**Implication for the schedule:** R4 reliably tells us *the adds block is happening / an add
occurred* (coarse, high recall) — good enough to keep the adds block tracked and to close it
(peaks stop → stir starts). But the chosen *fine per-ingredient* completion is **not cleanly
achievable with the cheap primitives as-is**; it should fall back to one V1 (VLM) escalation
when per-ingredient identity/order is actually needed (i.e. for later reminders), not run
continuously. RGB stays on (1 fps) during adds for the coarse signal; fine = trigger V1.

## Probe result — the audio backbone, tested on all 16 SHC recordings (2026-06-15)

`detectors/probes/probe_audio_anchors.py` ran the frozen detectors (no retuning) over all 16
(`results_audio_anchors.json`), scoring each anchor against its GT stage window (±15 s):

- **Microwave hum (A1) generalizes: recall 0.931 (27/29 microwave+heat steps).** Two misses
  are explainable: **8_33** (no hum detected at all) and **8_50** (different microwave, no
  detectable hum — already a documented disclosure). 6 "false" hum runs on **8_19/8_20** are
  *fragmentation* (one cycle split into several short runs) — benign for stage tracking, but
  it breaks clean duration measurement (matters for later timing reminders).
- **End beep (A2): recall 0.931 (27/29)** at microwave/heat ends — a strong independent
  corroborator. Hum-OR-beep coverage of the microwave stages is near-total.
- **Stir clink-train (A3) is WEAK at scale: recall 0.643 (9/14 mix steps), 17 false strong
  clinks.** Stir is the one cheap anchor that does *not* hold up across recordings.

**Why the schedule still survives the weak stir:** mix completion has a **next-anchor
backstop** — `clink offset OR next microwave-hum onset OR timer`. For 4 of the 5 mix-clink
misses (8_19/8_25/8_35/8_44) the following heat-step hum *was* detected, so mix still closes;
only **8_33** (no hum and no clink anywhere) fully fails. The multi-criteria completion design
is vindicated: the reliable hum/beep anchors carry the recipe and backstop the unreliable stir.

**Net for the backbone:** the cheap-audio spine (hum + beep) that covers 4 of 5 stages is
**~93% reliable across all 16**, no VLM, no RGB. The two weak spots are stir (backstopped) and
the 8_33/8_50 hum dropouts (need a fallback — beep, or a short VLM/RGB confirmation).

## Honest limits / open items

- **One recipe, GT-aware.** Generalization to the other 23 recipes (esp. stovetop ones with
  no microwave anchor — `texture_dynamics` A4 is still *untested*) is unproven.
- **Hum fragmentation (8_19/8_20)** inflates run count — fine for tracking, not for timing GT.
- **8_33 is the worst case**: no hum, no stir clink detected — the cheap backbone is blind
  there; it would need the beep-only fallback or an escalation.
- **Several primitives are untested/weak** (A4 sizzle, A5 water, A7 rustle; A6 pour weak).
- **Completion latency.** Next-anchor completion means a quiet step is only confirmed done
  when the following loud step starts (acceptable for completion; matters for reminders).
- **Reminders deferred.** The schedule emits no reminders yet; the timing/identity hooks
  (hum-duration, ingredient escalation) are noted but not wired.
