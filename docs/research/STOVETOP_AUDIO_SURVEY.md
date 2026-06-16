# Stovetop audio survey — can cheap audio anchor cooking? (2026-06-15)

**Question.** The microwave hum (A1) anchors microwave recipes at ~0.93 recall, near-zero
false. Does the analogous **sustained-cook** detector (A4, rolling-median band-energy gate)
anchor *stovetop* recipes — and on which ones is it an **A-solve** (cheap sensor settles it
alone, no VLM) vs only a **B-trigger** (cheap cue wakes one VLM call) vs **none**?

**Method.** One detector (`detectors_lib.detect_sizzle_runs`, params FROZEN on Broccoli
Stir Fry 23_5, *no per-recipe tuning*) run over 9 stovetop recipes (137 recordings).
Per recipe: cook-phase recall (a sustained run covers ≥40% of the cook window),
false-prep rate (sizzle seconds in prep ÷ cook seconds), and the **acoustic contrast**
(median 1.5–7 kHz level in cook steps minus prep steps). `probe_sizzle.py --activity N`;
results in `detectors/probes/results_sizzle_act*.json`. Cook-step sets are hand-mapped per
recipe (approximate windows — treat the *ordering* as the signal, not 2nd-decimal values).

## Result — detectability tracks the prep/cook acoustic contrast

| Recipe | Recall | False-prep | Contrast (cook−prep dB) | Verdict |
|---|---|---|---|---|
| **Pan Fried Tofu** (25) | **1.00** | **0.04** | +16.9 | **A-solve (hum-grade)** |
| Sautéed Mushrooms (20) | 0.86 | 0.29 | +20.1 | B-trigger |
| Broccoli Stir Fry (23) | 0.88 | 0.19 | +20.4 | B-trigger |
| Scrambled Eggs (16) | 0.81 | 0.18 | +11.5 | B-trigger |
| Zoodles (18) | 0.80 | 0.28 | +2.5 | B-trigger (noisy) |
| Herb Omelet (22) | 0.71 | 0.10 | +9.9 | weak B-trigger |
| Tomato Chutney (15) | 0.60 | 0.17 | +8.7 | weak B-trigger |
| Caprese Bruschetta (29) | 0.38 | 0.32 | +5.6 | none (quiet toast) |
| Banana Pancakes (21) | **0.00** | 0.15 | **−2.2** | none (fry < prep) |

**The governing law.** Cheap-audio cook detection works when the cooking is acoustically
*louder and more sustained than the prep*. The two failures prove it from the bottom:
- **Banana Pancakes: negative contrast.** The gentle low-heat butter-fry of small batter
  puddles is *quieter* than the prep — whose loudest event is the **blender**. The fry never
  clears the floor → recall 0.0. The agent's correct target here is the *blender hum*, not
  the fry (recipe-gated detector selection working as intended).
- **Caprese Bruschetta: dry toast, +5.6 dB.** Barely above prep → 0.38.

**A-solve vs B-trigger is NOT microwave-vs-stovetop.** Pan Fried Tofu is hum-grade
(recall 1.0 / false 0.04) purely because its prep is near-silent (cut + pat-dry tofu) while
its fry is loud and long. The loud stir-fries (Mushrooms, Stir Fry) have *higher* cook
loudness but *noisy* prep (whisking sauce, handling produce, spiralizing), so false-prep
rises to 0.19–0.29 → B-trigger. So the agent should choose A-solve vs B-trigger from the
**prep/cook contrast prior**, per recipe.

## Two recipe-gated caveats (the agent must handle these)

1. **A1 (microwave hum) must be recipe-gated AND re-parameterized per appliance.** Run blind
   on these no-microwave recipes it fires 21–35 false runs each. And the microwave-tuned
   params catch the *blender* only 5/19 on Banana Pancakes — a blender's motor hum sits at a
   different fundamental than the 60/120 Hz mains line. "Pick A1" is not enough; the agent
   must bind the band/mains-line to the specific appliance.
2. **Coarse anchor, not a fine timer.** "Recall" = the sustained run covers the cook phase;
   it does not resolve in-cook pauses (e.g. tofu's "remove from heat to reduce spitting").

## Implication for the sensor-control plan (Box 3)

Given a recipe, the planner predicts a cheap-audio **cook anchor role** from the contrast:
- **A-solve**: quiet-prep loud-fry recipes (Pan Fried Tofu-like) → audio alone tracks the
  cook stage, zero VLM.
- **B-trigger**: loud-fry noisy-prep recipes (stir fry, sautés, scrambles) → audio fires a
  VLM call to disambiguate cook-vs-prep.
- **none / use a different target**: gentle/dry cooking (pancakes, toast) → the fry is not a
  usable anchor; pick another loud event in the recipe (blender, kettle) or fall back to RGB/VLM.

Grounding: EPIC-SOUNDS (sizzling = long-form, but spectrally inseparable from water/whisk;
human audio-only recognition 20.8%) and EPIC-Fusion (audio is a *prime discriminator* for
'fry'/'wash' but works best fused with vision). See `DETECTOR_CATALOG.md` A4/A1.

## A4 (DSP) vs AL (CNN14) head-to-head — all 9 recipes (2026-06-15)

Same metric (cook coverage ≥0.40 = recall; prep-positive sec ÷ cook sec = false-prep),
AL given 20 s gap-bridging to match A4's persistence (`stress_stovetop_a4_al.py`,
`stress_stovetop_al_bridged.py`; `results_stress_stovetop*.json`).

| Recipe | A4 rec / false | AL-bridged rec / false |
|---|---|---|
| Pan Fried Tofu | **1.00** / 0.04 | 0.80 / 0.03 |
| Broccoli Stir Fry | 0.88 / 0.19 | 0.81 / **0.07** |
| Sauteed Mushrooms | 0.86 / 0.29 | 0.79 / **0.13** |
| Scrambled Eggs | 0.81 / 0.18 | **0.88 / 0.12** |
| Herb Omelet | **0.71** / 0.10 | 0.59 / 0.10 |
| Tomato Chutney | **0.60** / 0.17 | 0.27 / 0.05 |
| Zoodles | **0.80** / 0.28 | 0.67 / **0.14** |
| Banana Pancakes | 0.00 / 0.14 | 0.10 / 0.03 |
| Caprese Bruschetta | 0.38 / 0.32 | 0.06 / 0.03 |

**Verdict: COMPLEMENTARY, not replacement** (refutes the earlier "bind AL not A4").
A4 = the recall/coverage anchor (wins raw recall + SIMMER, e.g. chutney 0.60 vs 0.27 —
the CNN doesn't score a low simmer as "frying"). AL = the precision arm — on loud fries
it matches A4 recall at ~½ the prep false-alarm, and on Scrambled Eggs it beats A4 on
both. Neither rescues gentle/dry cooking (pancakes, caprese ~0 for both — the contrast
law again). Best use: **fuse** — A4 for coverage, AL to suppress prep false-alarms on
noisy-prep B-trigger recipes. AL needs gap-bridging (raw per-frame coverage is sparse).

### Fusion measured (`fusion_stovetop.py`, `results_fusion_stovetop.json`)

Mean over 9 recipes (recall / false-prep): A4 0.67/0.19 · AL 0.55/0.08 · Union 0.74/0.21 ·
Intersection 0.47/0.06 · **A4-gated-by-AL 0.64/0.12**. The winner is **`A4_gated_by_AL`**
(keep each A4 run only where an AL run overlaps it): recall ≈ A4 at **−37% false-prep**, and
it **preserves the simmer** (Chutney 0.60) that plain intersection destroys (0.07, AL doesn't
fire on a low simmer). Per noisy-prep recipe the false-prep drop is the win: Stir Fry
0.19→0.11, Mushrooms 0.29→0.22, Eggs 0.18→0.14. Union gives the highest recall (0.74) at the
cost of precision; intersection is too aggressive (omelet 0.47, simmer 0.07). In sensor-control
terms each suppressed prep false-alarm is a wasted VLM trigger avoided → ~37% fewer wasted cook-
confirm calls at equal recall. Fusion does NOT expand coverage to gentle/dry cooking (pancakes
~0, caprese marginal) — that slice stays C-none.

### Is ~0.7 the A4 recall ceiling? (on-heat-window re-score, `rescore_onheat.py`)

The pooled ~0.67 mean is misleading two ways. (1) It POOLS the C-none recipes (Pancakes 0.00,
Caprese 0.38) that audio physically cannot do; per-class the **audible-cook** recipes already
average **0.84** at ≥0.40 coverage and **0.92** at "detected at all" (≥0.10). (2) The cook
*window* was [min,max] over ALL cook sub-steps — including silent adds, empty-pan "heat oil",
and off-heat "set" — which dilutes coverage. Re-scoring against the UNION of only the
**on-heat-with-food** sub-steps **confirms the dilution** (mean coverage jumped, e.g. Stir Fry
0.65→0.87, Chutney 0.49→0.70, Omelet 0.45→0.59) — but **recall barely moved** (audible-cook mean
0.84→0.86; all-9 0.67→0.71), because recall is a binary at 0.40 and the remaining misses are
genuinely quiet cooks, not window artifacts. Biggest single gain: **Tomato Chutney 0.57→0.79**
(its simmer was badly diluted by tempering/add steps). So: **~0.7 is a pooled-with-inaudible
number; the real per-class ceiling is ~0.86 (audible cooks), and the residual gap to 1.0 is
quiet/low-heat cooks + the irreducible C-none recipes — NOT a fixable window artifact.**

### Boundary metric — the one that matches our setting (`boundary_eval.py`)

Coverage is the wrong objective: our task is **detect frying start/end + fire the VLM**, not
cover a window. Re-scored as signed median onset/offset latency vs the on-heat phase
[cs,ce] (+ = late; A4 = sizzle runs; AL = cook-prob>0.1 frames merged ≤5 s, NO 20 s bridge):

| Recipe | A4 onset / offset | AL onset / offset | A4 / AL false-trig per rec |
|---|---|---|---|
| Pan Fried Tofu | +20.9 / **+37.9** | +67.1 / **+8.0** | 0.33 / 0.60 |
| Broccoli Stir Fry | +1.9 / +8.9 | +9.3 / **+1.7** | **2.00** / 0.56 |
| Sauteed Mushrooms | −6.3 / **+50.8** | +10.1 / **+8.9** | 1.07 / 1.86 |
| Scrambled Eggs | +9.6 / +33.8 | +7.8 / +21.3 | 1.44 / 1.31 |
| Herb Omelet | +43.5 / +12.5 | +79.5 / +6.2 | 0.35 / 0.18 |
| Tomato Chutney | +1.3 / +42.2 | +13.1 / −90.4 | 1.43 / 1.43 |
| Zoodles | −25.3 / +15.2 | −8.8 / −2.2 | 0.67 / 0.40 |

**Takeaways.** (1) **A4's OFFSET lags badly (+34 to +51 s)** on tofu/mushrooms/eggs/chutney —
its 45 s median + 20 s merge keep the run alive long after cooking stops. Coverage hid this;
it confirms the long window is wrong for end-of-step. (2) **AL nails OFFSET (~2–21 s)** on
audible cooks — it is the cook-END detector (matches the earlier sizzle-end median 11 s).
(3) **ONSET is hard for BOTH** (tens of seconds, sign confounded): A4's "early" onsets are
mostly prep broadband firing before the cook (false-early), and AL is late because frying
ramps gradually. Neither is a clean few-second onset detector. (4) A4 throws 1–2 prep
**false-triggers/rec** on busy recipes (Stir Fry 2.0) → wasted VLM calls; AL is cleaner on
the worst cases. Design consequence: **use AL for step-END; do NOT trust the sustained
detector for step-START — use a transient onset cue (igniter-click A2 = stove-on, add-to-pan
sizzle burst) and/or let the VLM confirm onset.** C-none (pancakes, caprese) stays VLM.

### Onset-cue test (`onset_eval.py`) — can step-START be caught cheaply?

Built a **causal rising-edge sizzle-onset** detector (1.5–7 kHz vs a *trailing* 30 s median,
+8 dB, hold ≥5 s — the food-in-pan burst, low lookahead) and tested the pre-cook transient
cues. Median signed onset latency vs frying start (audible recipes): **sizzle-onset +2 to +11 s**
— vs A4 +20…+44 s and AL +67…+80 s, i.e. **3–7× tighter and causal/online-able.** BUT it only
fires within ±15 s of the true start in **~60 %** of recordings (det±15 0.53–0.67) → low-latency
when it fires, not reliable alone. Pre-cook cues: **A2 tonal-beep as igniter = FAILS** (0.07–0.53;
igniters are broadband clicks, many stoves electric — not tonal); **A3 transient-train fires
0.88–1.00 but is useless** (generic prep manipulation, no specificity — it'd fire on any prep).
**Verdict: step-START is a B-TRIGGER, not an A-solve** — the cheap rising-edge gives a ~6–11 s
low-latency trigger ~60 % of the time; the missed ~40 % + confirmation go to the VLM. Net
stovetop stack: **START = sizzle-onset rising-edge → VLM-confirm; END = AL (~8 s); sustained
A4 only for "cooking ongoing" gating.** C-none (pancakes/caprese) onset unreliable for all cues.
