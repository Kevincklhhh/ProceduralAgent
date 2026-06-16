# Replay Experiment Report - Activity 8 (mug hot chocolate)

Scored by `replay/score.py` against `detectors/gt_activity8.json`. All numbers in `replay/results/scores.json`.

## Setup

Three arms replayed over the same six recordings (8_16, 8_3, 8_25 clean; 8_26, 8_31 error runs; 8_50 an order-error run: sugar added before milk, mix and cinnamon skipped - deliberately NOT special-cased in any engine):

1. **detector_replay** - frozen audio detectors (microwave hum/beep, stir, pour) driving a procedure graph; zero VLM calls. Ingredient identity is unknowable to it; it emits an `escalation_request` instead.
2. **periodic_vlm_qwen** - Qwen3.6-27B on a local vLLM server, called every 10 s with 3 frames (480p).
3. **detector_plus_escalation** - arm 1 plus one targeted VLM call per recording at the graph's mix boundary to verify ingredients.

Detectors were frozen before replay; numeric thresholds tuned on 8_16 only (see Limitations for the structural-choice disclosure).

Reminder truth table (scored ids only):

| recording | should fire |
|---|---|
| 8_16 | - |
| 8_3 | - |
| 8_25 | - |
| 8_26 | missing_ingredient_before_mix, overtime_microwave |
| 8_31 | missing_ingredient_before_mix, undertime_microwave, undertime_microwave_2 |
| 8_50 | missing_ingredient_before_mix, missing_mix_before_heat |

Scored ids: overtime_microwave, undertime_microwave, overtime_microwave_2, undertime_microwave_2, missing_mix_before_heat, missing_ingredient_before_mix. `missing_ingredient_before_mix` is binary per recording; ingredient-level detail reported separately. For **detector_replay** that id is N/A-by-design: we score escalation-request coverage instead (it should request on ALL recordings).

## 1. Stage accuracy (per-second, coarse 5-stage + other)

Per-second labels over `[0, ceil(last GT step end)]`; GT coarse label from step segments (88=fill_milk, 89=microwave_initial, 90/84/87=adds, 85=mix, 83=heat_serve; unlabeled seconds=other; skipped steps absent). Where GT step segments overlap (e.g. 8_25: cinnamon prep during the microwave run), the most recently started step wins. Predicted gaps=other; fine `add_*` maps to `adds`.

| arm | 8_16 | 8_3 | 8_25 | 8_26 | 8_31 | 8_50 | mean |
|---|---|---|---|---|---|---|---|
| detector_replay | 89.4% | 59.8% | 64.1% | 81.2% | 66.7% | 42.1% | **67.2%** |
| periodic_vlm_qwen | 60.8% | 48.0% | 48.2% | 24.7% | 38.5% | 6.7% | **37.8%** |
| detector_plus_escalation | 89.4% | 59.8% | 64.1% | 81.2% | 66.7% | 42.1% | **67.2%** |

Coarse accuracy excluding GT-`other` seconds:

| arm | 8_16 | 8_3 | 8_25 | 8_26 | 8_31 | 8_50 | mean |
|---|---|---|---|---|---|---|---|
| detector_replay | 92.0% | 61.3% | 67.1% | 84.4% | 73.9% | 43.4% | **70.4%** |
| periodic_vlm_qwen | 62.6% | 48.5% | 48.6% | 22.8% | 37.4% | 3.7% | **37.3%** |
| detector_plus_escalation | 92.0% | 61.3% | 67.1% | 84.4% | 73.9% | 43.4% | **70.4%** |

Fine 7-way accuracy (periodic arm only - the audio/graph detector cannot distinguish the three adds by design, so fine scoring would be vacuous for it):

| arm | 8_16 | 8_3 | 8_25 | 8_26 | 8_31 | 8_50 | mean |
|---|---|---|---|---|---|---|---|
| periodic_vlm_qwen (fine) | 50.7% | 38.3% | 42.3% | 22.4% | 38.5% | 6.7% | **33.1%** |

### Microwave-anchor boundary deltas (descriptive)

Predicted `microwave_initial` / `heat_serve` interval start/end minus GT step start/end, seconds (positive = predicted later). Blank = stage not predicted or GT step absent.

| arm | rec | mw start | mw end | heat start | heat end |
|---|---|---|---|---|---|
| detector_replay | 8_16 | +2.6 | -0.1 | +10.0 | +10.7 |
| detector_replay | 8_3 | +10.7 | -16.9 | +11.3 | +12.7 |
| detector_replay | 8_25 | +51.1 | +0.9 | +15.5 | +3.1 |
| detector_replay | 8_26 | +7.3 | +0.7 | +55.8 | +5.1 |
| detector_replay | 8_31 | +28.4 | +2.9 | +8.0 | +34.0 |
| detector_replay | 8_50 | +4.1 | -0.2 |  |  |
| periodic_vlm_qwen | 8_16 | +9.6 | +25.5 |  |  |
| periodic_vlm_qwen | 8_3 |  |  | +17.9 | +12.7 |
| periodic_vlm_qwen | 8_25 | +14.8 | -39.6 | +23.6 | +3.1 |
| periodic_vlm_qwen | 8_26 |  |  | +18.0 | -45.3 |
| periodic_vlm_qwen | 8_31 | +22.0 | +32.2 |  |  |
| periodic_vlm_qwen | 8_50 |  |  |  |  |
| detector_plus_escalation | 8_16 | +2.6 | -0.1 | +10.0 | +10.7 |
| detector_plus_escalation | 8_3 | +10.7 | -16.9 | +11.3 | +12.7 |
| detector_plus_escalation | 8_25 | +51.1 | +0.9 | +15.5 | +3.1 |
| detector_plus_escalation | 8_26 | +7.3 | +0.7 | +55.8 | +5.1 |
| detector_plus_escalation | 8_31 | +28.4 | +2.9 | +8.0 | +34.0 |
| detector_plus_escalation | 8_50 | +4.1 | -0.2 |  |  |

The detector arms' microwave boundaries track GT to within a few seconds when a hum run is detected (the hum starts after walking to the microwave, so small positive start deltas are expected - GT segments include walking). The periodic arm's boundaries are quantized to its 10 s call grid and drift much further.

## 2. Reminder decisions

| arm | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| detector_replay | 2 | 2 | 2 | 50.0% | 50.0% |
| periodic_vlm_qwen | 0 | 1 | 7 | 0.0% | 0.0% |
| detector_plus_escalation | 5 | 5 | 2 | 50.0% | 71.4% |

(detector_replay is scored over 5 ids - `missing_ingredient_before_mix` is N/A-by-design; its escalation-request coverage was **6/6** recordings, as required. The two VLM-bearing arms are scored over all 6 ids.)

Per-id outcome (recordings listed):

| arm | id | TP | FP | FN |
|---|---|---|---|---|
| detector_replay | overtime_microwave | 8_26 | - | - |
| detector_replay | undertime_microwave | 8_31 | 8_25 | - |
| detector_replay | overtime_microwave_2 | - | - | - |
| detector_replay | undertime_microwave_2 | - | - | 8_31 |
| detector_replay | missing_mix_before_heat | - | 8_25 | 8_50 |
| detector_replay | missing_ingredient_before_mix | N/A by design | | |
| periodic_vlm_qwen | overtime_microwave | - | 8_16 | 8_26 |
| periodic_vlm_qwen | undertime_microwave | - | - | 8_31 |
| periodic_vlm_qwen | overtime_microwave_2 | - | - | - |
| periodic_vlm_qwen | undertime_microwave_2 | - | - | 8_31 |
| periodic_vlm_qwen | missing_mix_before_heat | - | - | 8_50 |
| periodic_vlm_qwen | missing_ingredient_before_mix | - | - | 8_26, 8_31, 8_50 |
| detector_plus_escalation | overtime_microwave | 8_26 | - | - |
| detector_plus_escalation | undertime_microwave | 8_31 | 8_25 | - |
| detector_plus_escalation | overtime_microwave_2 | - | - | - |
| detector_plus_escalation | undertime_microwave_2 | - | - | 8_31 |
| detector_plus_escalation | missing_mix_before_heat | - | 8_25 | 8_50 |
| detector_plus_escalation | missing_ingredient_before_mix | 8_26, 8_31, 8_50 | 8_16, 8_3, 8_25 | - |

Fire timestamps vs relevant GT step window (+/-15 s):

| arm | rec | id | t (s) | GT window | inside +/-15s | verdict |
|---|---|---|---|---|---|---|
| detector_replay | 8_25 | undertime_microwave | 150.5 | 75.2-149.6 | True | FP |
| detector_replay | 8_25 | missing_mix_before_heat | 371.8 | 356.4-439.8 | True | FP |
| detector_replay | 8_26 | overtime_microwave | 134.4 | 47.1-177.8 | True | TP |
| detector_replay | 8_31 | undertime_microwave | 100.7 | 48.0-97.8 | True | TP |
| periodic_vlm_qwen | 8_16 | overtime_microwave | 100.0 | 70.4-134.5 | True | FP |
| periodic_vlm_qwen | 8_16 | overtime_microwave | 130.0 | 70.4-134.5 | True | FP |
| detector_plus_escalation | 8_16 | missing_ingredient_before_mix | 319.5 | 296.6-376.9 | True | FP |
| detector_plus_escalation | 8_3 | missing_ingredient_before_mix | 166.5 | 287.1-331.3 | False | FP |
| detector_plus_escalation | 8_3 | missing_ingredient_before_mix | 166.5 | 287.1-331.3 | False | FP |
| detector_plus_escalation | 8_3 | missing_ingredient_before_mix | 166.5 | 287.1-331.3 | False | FP |
| detector_plus_escalation | 8_25 | undertime_microwave | 150.5 | 75.2-149.6 | True | FP |
| detector_plus_escalation | 8_25 | missing_mix_before_heat | 371.8 | 356.4-439.8 | True | FP |
| detector_plus_escalation | 8_25 | missing_ingredient_before_mix | 371.8 | 305.0-351.2 | False | FP |
| detector_plus_escalation | 8_26 | overtime_microwave | 134.4 | 47.1-177.8 | True | TP |
| detector_plus_escalation | 8_26 | missing_ingredient_before_mix | 280.5 | 285.6-314.9 | True | TP |
| detector_plus_escalation | 8_26 | missing_ingredient_before_mix | 280.5 | 285.6-314.9 | True | TP |
| detector_plus_escalation | 8_26 | missing_ingredient_before_mix | 280.5 | 285.6-314.9 | True | TP |
| detector_plus_escalation | 8_31 | undertime_microwave | 100.7 | 48.0-97.8 | True | TP |
| detector_plus_escalation | 8_31 | missing_ingredient_before_mix | 183.9 | 152.8-170.7 | True | TP |
| detector_plus_escalation | 8_31 | missing_ingredient_before_mix | 183.9 | 152.8-170.7 | True | TP |
| detector_plus_escalation | 8_50 | missing_ingredient_before_mix | 245.5 | 186.4-251.7 | True | TP |
| detector_plus_escalation | 8_50 | missing_ingredient_before_mix | 245.5 | 186.4-251.7 | True | TP |
| detector_plus_escalation | 8_50 | missing_ingredient_before_mix | 245.5 | 186.4-251.7 | True | TP |

Note: periodic_vlm_qwen fired `overtime_microwave` 2x on 8_16 (t=[100.0, 130.0]); deduplicated to one decision for scoring.

Note: detector_plus_escalation fired `missing_ingredient_before_mix` 3x on 8_3 (t=[166.5, 166.5, 166.5]); deduplicated to one decision for scoring.

Note: detector_plus_escalation fired `missing_ingredient_before_mix` 3x on 8_26 (t=[280.5, 280.5, 280.5]); deduplicated to one decision for scoring.

Note: detector_plus_escalation fired `missing_ingredient_before_mix` 2x on 8_31 (t=[183.9, 183.9]); deduplicated to one decision for scoring.

Note: detector_plus_escalation fired `missing_ingredient_before_mix` 3x on 8_50 (t=[245.5, 245.5, 245.5]); deduplicated to one decision for scoring.

### Ingredient-level detail (missing_ingredient_before_mix)

| arm | rec | flagged | truly missing | ing TP | ing FP | ing FN |
|---|---|---|---|---|---|---|
| detector_plus_escalation | 8_16 | chocolate | - | - | chocolate | - |
| detector_plus_escalation | 8_3 | chocolate, cinnamon, sugar | - | - | chocolate, cinnamon, sugar | - |
| detector_plus_escalation | 8_25 | chocolate | - | - | chocolate | - |
| detector_plus_escalation | 8_26 | chocolate, cinnamon, sugar | cinnamon | cinnamon | chocolate, sugar | - |
| detector_plus_escalation | 8_31 | chocolate, sugar | chocolate, sugar | chocolate, sugar | - | - |
| detector_plus_escalation | 8_50 | chocolate, cinnamon, sugar | cinnamon | cinnamon | chocolate, sugar | - |
| periodic_vlm_qwen | 8_26 | - | cinnamon | - | - | cinnamon |
| periodic_vlm_qwen | 8_31 | - | chocolate, sugar | - | - | chocolate, sugar |
| periodic_vlm_qwen | 8_50 | - | cinnamon | - | - | cinnamon |

detector_plus_escalation ingredient-level totals: TP=4, FP=9, FN=0 (precision 30.8%, recall 100.0%). The escalation VLM never misses a truly-missing ingredient but over-reports "missing" (notably chocolate, falsely flagged on 5/6 recordings): it asserts absence too readily from 10 frames sampled around the mix boundary.
periodic_vlm_qwen ingredient-level totals: TP=0, FP=0, FN=4 - it never emitted a missing-ingredient event at all.

### Descriptive events (not in P/R)

- **detector_replay** hot_mug_caution: 5/6 completed heats covered (8_16@448.5s(hit), 8_3@409.2s(hit), 8_25@431.7s(hit), 8_26@441.9s(hit), 8_31@235.9s(hit)).
  - microwave_done_prompt: no fires.
- **periodic_vlm_qwen** hot_mug_caution: 1/6 completed heats covered (8_3@410.0s(hit)).
  - microwave_done_prompt: no fires.
  - `other` events from the VLM: none (no over-talk; if anything the periodic arm under-talks).
- **detector_plus_escalation** hot_mug_caution: 5/6 completed heats covered (8_16@448.5s(hit), 8_3@409.2s(hit), 8_25@431.7s(hit), 8_26@441.9s(hit), 8_31@235.9s(hit)).
  - microwave_done_prompt: no fires.

**Known limitation (excluded from the truth table):** 8_26 step 83 carries a GT Timing Error from a split run - 8 s on high power plus ~1 min on low power. An 8 s run is below the 20 s hum floor of the audio probe, so duration-rule arms cannot catch it; no `*_microwave_2` id was expected on 8_26. The periodic VLM arm also said nothing about it.

## 3. Cost ledger

Totals across six recordings (2234 s = 37.23 min of GT-scored video); per-minute normalization in parentheses.

| arm | vlm_calls | frames_sent | vlm_latency_total_s | detector compute_s |
|---|---|---|---|---|
| detector_replay | 0 (0.0/min) | 0 (0.0/min) | 0.0 (0.0 s/min) | 17.08 (0.459 s/min) |
| periodic_vlm_qwen | 229 (6.15/min) | 687 (18.451/min) | 11469.2 (308.04 s/min) | 11500.8 (308.885 s/min) |
| detector_plus_escalation | 6 (0.161/min) | 60 (1.611/min) | 267.31 (7.18 s/min) | 19.82 (0.532 s/min) |

**The periodic baseline cannot keep up with real time on this hardware.** It is called every 10 s but a call takes ~44-54 s (measured mean across recordings ~50 s; sequential smoke test 43.6 s/call). Total VLM latency 11469.2 s to cover 2234 s of video = **5.13x real time** - i.e. run sequentially it falls ~5x behind; the replay only finished by running 2-way concurrency offline. A live assistant on this hardware would answer about stage N-25 while the user is on stage N.
The escalation arm used exactly 1 call/recording (6 calls, 60 frames, 267.31 s latency total = 2.3% of the periodic arm's latency budget); the detector arm used 0.

## 4. Per-recording walkthroughs (error runs)

### 8_26 (overtime first microwave; cinnamon skipped; GT also: whole milk + spill, 4 chocolate pieces, split second heat)

- **detector_replay**: caught `overtime_microwave` at 134.4 s, 80 s into a hum run GT says lasted ~2 min - correct and timely. Escalated for ingredients (cannot know them itself). Missed nothing it could physically see. The split second-heat (8 s high) is below the hum floor (excluded, see above).
- **periodic_vlm_qwen**: emitted NO events - missed the 2-minute overtime despite sampling the microwave 12+ times, and never questioned ingredients. Its stage track labeled most of 70-210 s as `other`.
- **detector_plus_escalation**: `overtime_microwave` TP (134.4 s) and `missing_ingredient_before_mix` TP at 280.5 s - but the VLM verdict flagged all three ingredients missing when only cinnamon was (chocolate and sugar were visibly added); recording-level TP, ingredient-level 1 TP + 2 FP.

### 8_31 (35 s first microwave, 40 s second; sugar and chocolate skipped)

- **detector_replay**: `undertime_microwave` TP at 100.7 s (measured 24.3 s hum vs GT 35 s actual - the hum probe undershoots short runs but the decision is right). MISSED `undertime_microwave_2`: the second run (40 s actual, GT heat step 175.9-233.2 s) produced a hum the engine treated as the final heat and closed with `hot_mug_caution` at 235.9 s instead of checking duration - FN.
- **periodic_vlm_qwen**: NO events; both undertime runs and both missing ingredients missed. After 160 s it labeled everything `other`.
- **detector_plus_escalation**: `undertime_microwave` TP; `missing_ingredient_before_mix` TP at 183.9 s with a PERFECT ingredient verdict (chocolate=missing, sugar=missing, cinnamon=added) - the one recording where the escalation VLM was exactly right. Same `undertime_microwave_2` FN as arm 1 (shared engine).

### 8_50 (order error: sugar before milk; mix and cinnamon skipped)

- GT order: sugar 0.7-69.3, milk 69.3-92.3, microwave 96.1-159.8, chocolate 163.3-185.9, heat 186.4-251.7. Mis-tracking here is expected data; no engine was special-cased.
- **detector_replay**: stage track is wrong by construction (labels 0-100 s `fill_milk` while the user added sugar first). It falsely detected `mix` at 245.5 s (debug shows a strong clink train 245.5-267.5 s - spoon/mug contact during serving), which suppressed `missing_mix_before_heat` - FN. The second microwave run produced NO detectable hum (debug: mw2=null; only its 4 kHz done-beep at 251.0 s registered - this is the different-microwave recording from the disclosure), so no `hot_mug_caution` and no `heat_serve` interval. Escalation was still requested (coverage held).
- **periodic_vlm_qwen**: detected `add_sugar` at 60-80 s - the ONLY arm whose stage track reflects the order error at all (RGB sees the sugar jar; audio cannot). But it emitted no events: order error never called out, missing mix and cinnamon never questioned. Fine 7-way accuracy still only 6.7% because nearly everything else was labeled `other`.
- **detector_plus_escalation**: `missing_ingredient_before_mix` recording-level TP at 245.5 s (cinnamon correctly among the flagged), but the verdict flagged all three - sugar (added early) and chocolate (added) are ingredient-level FPs. Same `missing_mix_before_heat` FN as arm 1.

## 5. Takeaways

1. **Procedure structure + cheap audio beats the periodic VLM on this task, decisively.** Reminders: detector+escalation 5 TP / recall 71.4% vs periodic VLM 0 TP / recall 0.0% (the periodic arm caught zero true reminders and its only fires were two false `overtime_microwave` on clean 8_16). Stage tracking: mean coarse 67.2% (detector arms) vs 37.8% (periodic). Cost: 0-6 VLM calls vs 229, and the periodic arm is 5.13x slower than real time, so its already-poor numbers are an OFFLINE upper bound on its live usefulness.
2. **The single targeted escalation call is where the value-per-call is.** Six calls bought ingredient awareness the graph cannot have (recall 71.4% overall, 100% recording-level recall on missing-ingredient runs and 100% ingredient-level recall), at ~2% of the periodic arm's latency. But precision is poor (50.0%): the escalation VLM asserts "missing" too readily from 10 frames - every clean recording got at least one spurious "did you add X?" prompt. Better escalation prompting/frame selection (frames AT each add window, not around the mix boundary) is the obvious next lever.
3. **Where it still needs RGB:** (a) ingredient identity - by design (the detector arm escalated on 6/6 recordings because it cannot know); (b) order errors - only the periodic VLM's stage track showed sugar-before-milk on 8_50; audio is sequence-blind between anchors; (c) second-microwave duration semantics (undertime_microwave_2 FN on 8_31) and any sub-hum-floor event; (d) quantity errors (4 chocolate pieces, whole-vs-skimmed milk on 8_26) - invisible to every arm tested.
4. The detector arms' two reminder FPs both came from one clean recording (8_25) where the hum probe fused a 24 s run and missed the stir - audio-probe errors propagate directly into reminder errors; the graph amplifies neither nor filters them.

## 6. Limitations

- **One task type.** Six recordings of one microwave-centric recipe (activity 8); nothing here measures generalization to other procedures.
- **Microwave-centric anchors.** The graph advances mainly on microwave hum/beep anchors; tasks without such a strong audio anchor would lose most of the detector arms' structure.
- **GT segment boundaries include walking** (fetching ingredients, moving to the microwave), so per-second stage accuracy and boundary deltas penalize/credit transition seconds somewhat arbitrarily; GT step segments even overlap in places (8_25, 8_16) - we resolved overlaps with a most-recently-started-step rule.
- **8_26 step 83** Timing Error (8 s high + ~1 min low) is below the 20 s hum floor and was excluded from the truth table (see Section 2).
- **Design-leakage disclosure (inherited from the hum probe, restated from `detectors/probes/results_hum_beep.json` tuning_on_8_16.note):** thresholds were set midway between hum and background medians on 8_16, where the grid shows a wide plateau (most combos give exactly 2 runs of ~60 s inside GT, 0 false). However, three STRUCTURAL choices were made after inspecting eval recordings (numeric thresholds still from 8_16 only): (1) the beep band was widened to 800-5000 Hz because 8_50 has a 4 kHz beeper (a different microwave); (2) features are median-smoothed over 5.4 s because wearers make noise next to a running microwave; (3) a broadband gate F1 > 2 dB was added because 8_31 contains a fridge-like pure-120 Hz source. The detector arms' results are therefore not fully blind to the eval set at the structural level, and clean-set FP rates (8_25) suggest the probe is still fragile.
