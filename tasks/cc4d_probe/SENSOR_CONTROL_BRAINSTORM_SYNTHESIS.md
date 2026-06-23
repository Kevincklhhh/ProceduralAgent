# Sensor-Control Brainstorm — Corpus Synthesis (21 CC4D recipes)

**What this is.** A corpus-level read of the per-recipe detector/sensor-control brainstorm. For
every *claim* in the generated criteria — each step's **recognition** (is this step happening
now?), **completion** (is it done?), and each **check** (a proactive thing to flag: timing,
measurement, ingredient/prep, technique, temperature) — a subagent tagged the cheapest sensor
*modality* and a *sensing role*. This doc aggregates 1,232 claims across the 21 non-loop recipes.

**Provenance / firewall.** These are *feasibility hypotheses* reasoned from the criteria + recipe
DAG + the proven audio catalog — **not** measured detector performance. Firewall-neutral (no error
tags / Qualcomm timestamps / traces). The criteria include `[probe-added]` checks drawn from the
observed-error space, so the technique/temperature counts are inflated vs pure recipe text.

**Artifacts**
- Per-recipe: `tasks/cc4d_probe/<recipe>.brainstorm.json` (21 files, per-claim detail + schedule)
- Machine-readable roll-up: `tasks/cc4d_probe/_corpus_brainstorm_aggregate.json`
- Reproducible aggregator: `scripts/aggregate_brainstorm.py`

Definitions: **A-solve** = a cheap sensor or pure recipe-graph logic settles it, the vision-language
model (VLM, the expensive sensor) never fires. **B-trigger** = a cheap event wakes the VLM for
exactly one call. **C-none** = no cheap handle; camera on + VLM continuous/periodic.

---

## 1. The headline: per-claim potential vs per-step reality

The per-claim count answers "what could each claim use *alone*." But at runtime a step's
recognition + completion + every check are **live together**, so the sensor config is pinned to the
**most expensive** claim active. Counting at the unit that actually gets scheduled collapses the
opportunity:

| View (unit) | A-solve (camera can sleep) | B-trigger (one VLM call) | C-none (VLM continuous) |
|---|---|---|---|
| Per-claim (isolated) | 350 — **28.4%** | 529 — 42.9% | 353 — **28.7%** |
| **Per-step (max role)** | 9 — **3.0%** | 86 — 28.2% | 210 — **68.9%** |
| Per fork-window (concurrent steps merged) | 8 — 4.0% | 60 — 30.3% | 130 — 65.7% |

**Deployable read: only ~3% of steps (9/305) let the camera fully sleep; ~69% must run the VLM
continuously.** The 28% per-claim A-solve was never schedulable — runtime concurrency erases it.

---

## 2. What pins each step to the expensive role

Subtype that forces a step to **C-none** (counted on the binding step):

| Subtype | Steps pinned to C-none |
|---|---|
| **technique** | **125** |
| recognition | 74 |
| measurement | 61 |
| preparation | 56 |
| temperature | 21 |
| completion | 15 |
| timing | 1 |

The `technique` check is the dominant cost driver: a "don't spill / stir thoroughly / flip cleanly"
check is unpredictable-in-time, so it is inherently continuous and drags its step's cheap sibling
claims (clean A-solve timing/completion) up to C-none with it.

### Sensitivity — how much sleep comes back if continuous checks are relaxed

| Policy | A-solve | B-trigger | C-none |
|---|---|---|---|
| As-is | 9 (3%) | 86 (28%) | 210 (69%) |
| Drop `technique` | 14 (5%) | 131 (43%) | 160 (52%) |
| Drop `technique`+`temperature` | 17 (6%) | 142 (47%) | 146 (48%) |

Removing technique+temperature only moves full-sleep 3%→6%; just **4 steps** have a
technique/temperature check as the *sole* blocker. The real effect is a **C→B shift**
(continuous VLM → duty-cycled VLM): C-none 69%→48%, B-trigger 28%→47%.

**Conclusion: camera-OFF is essentially dead at the scheduling unit. The achievable lever is
duty-cycling — turning C-none into B-trigger — not sleep.** And dropping technique checks is *not*
equal coverage; it trades an error class for energy.

---

## 3. Per-claim modality & role mix (the "what could each claim use" view)

Modality totals (best cheap fit per claim): `vlm` 589 (48%), `cv` 283 (23%),
`dag`/graph-logic-no-sensor 252 (20%), **`audio` only 108 (9%)**.

Role × modality (top): C-none·vlm 349 · B-trigger·cv 264 · **A-solve·dag 249** · B-trigger·vlm 240 ·
A-solve·audio 86. **Of 350 A-solves, 249 are free graph logic and only 86 are audio** — the cheap
win is mostly recipe-graph inference, not the microphone.

Per-class role mix (never pooled):

| Claim kind | A-solve | B-trigger | C-none | Read |
|---|---|---|---|---|
| completion (305) | **267** | 23 | 15 | ~88% free — inferred from successor / appliance end-beep |
| timing check (43) | **38** | 4 | 1 | ~88% A-solve — timer logic (D6) + audio anchor |
| recognition (305) | 36 | **195** | 74 | a trigger problem — cheap cue, VLM confirms which step |
| preparation (202) | 6 | **140** | 56 | ingredient/container identity → one VLM frame |
| measurement (143) | **0** | 82 | 61 | zero A-solve; cv can't count |
| technique (205) | 1 | 79 | **125** | 61% C-none — biggest irreducible block |
| temperature (29) | 2 | 6 | **21** | 72% C-none; audio can't read power-level |

---

## 4. Sensor parameters the plan can pre-schedule

- **Power OFF** (biggest lever, but rare): only ~3% of steps qualify — pure microwave/blender/timed
  windows where every live claim is A-solve.
- **Resolution / ROI**: ~43% of claims need only a small crop (mug/pan/board/appliance), at
  thumbnail res — a real compute lever *even while the camera is on for a C-none/B step*.
- **Color → monochrome**: ~11% of claims (motion/presence/state-change) need no color.
- **FPS**: B-trigger steps want low standby fps spiking to one frame on a cheap cue; C-none want
  periodic (~0.2–1 fps), not 30.
- **Audio detector arming** (recipes gating each, of 21): D6 timer 21 (free logic), D1 microwave 9,
  D4 cook-start 8, D5 water 8, D3 cook-end 7, D2 motor 3.
- **Thermal pixel (NOT in today's catalog)**: ~7.5% of claims flagged thermal-cheapenable, clustered
  in the temperature + cook-doneness checks that are otherwise C-none — the strongest case for
  *adding* a sensor, since it attacks the worst class.
- **IMU** (stir/whisk/shake): ~4% — speculative, weakest evidence.

---

## 5. Honest limits

- **~69% of steps need continuous VLM** at equal coverage. Camera-off is ~3% of steps.
- The A-solve win is carried by **free graph logic (249) far more than audio (86)**; audio's
  standalone contribution is ~7% of claims.
- B-trigger (the achievable win) still spends the VLM, once — a duty-cycle/latency win, not
  elimination, and it assumes a cheap trigger exists (RGB probes show counting fails, skin-gate
  failed → triggers for recognition/measurement are often weak).
- Thermal / IMU / monochrome are reasoning hypotheses, **not validated on CC4D**.
- The `[probe-added]` technique/temperature checks inflate the C-none block; if anything the
  continuous-VLM fraction against pure recipe-text criteria is slightly *under*stated here.
- Loop recipes (sautedmushrooms, dressedupmeatballs, pinwheels) excluded — runtime can't yet handle
  repeated step_ids.

---

## 5b. Base vs probe-added split (the firewall line)

Claims come from two sources. **Base** = rule-generated from the recipe step text + DAG
(recognition, completion, recipe-anticipated checks) — **firewall-clean**. **`[probe-added]`** =
added by inspecting the annotated CC4D mistakes (observed-error space) — **not** firewall-clean.
Corpus: 1,024 base claims (610 recognition/completion + 414 recipe-text checks) + 208 probe-added.
The (step_id, subtype) join to the authoritative `criteria.json` is unambiguous (0 mixed buckets).

Claim-level role mix by source:

| Source | A-solve | B-trigger | C-none |
|---|---|---|---|
| Base (1024) | 346 (34%) | 433 (42%) | 245 (**24%**) |
| Probe-added (208) | 4 (**2%**) | 96 (46%) | 108 (**52%**) |

Probe-added checks are overwhelmingly expensive (52% C-none, 2% A-solve) — they derive from
observed mistakes, which concentrate in the subtle technique/temperature failures no cheap sensor
catches.

Per-step max-role floor, base-only vs all:

| | A-solve (sleep) | B-trigger | C-none (continuous VLM) |
|---|---|---|---|
| All claims | 9 (3%) | 86 (28%) | 210 (**69%**) |
| Base-only (firewall-clean) | 19 (6%) | 105 (34%) | 181 (**59%**) |

Going firewall-clean lifts camera-off 3%→6% and drops continuous-VLM 69%→59%. **29 steps are
pushed to C-none *only* by a probe-added check.** But the floor is structural: even on pure
recipe-text criteria, **~59% of steps still need continuous VLM** (driven by recognition +
measurement + base technique), not an artifact of the probe relaxation.

## 6. Next decision

Role fractions are exhausted. The number the thesis needs is an **energy/latency** one: simulate a
window-level schedule (camera fps/res/off + VLM call cadence per window) and report "% of frames
captured & VLM calls fired vs an always-on 1-fps + every-frame-VLM baseline, at equal coverage."
That converts these role counts into the kWh/latency claim. Open prior question: should `technique`
checks stay in scope, given they are the dominant cost driver and many are `[probe-added]`?
