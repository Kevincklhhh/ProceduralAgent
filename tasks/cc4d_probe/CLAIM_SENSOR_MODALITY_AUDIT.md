# Claim Sensor Modality Audit

Scope: all `tasks/cc4d_probe/*.generated.criteria.json` files present in this
workspace. This is a hypothesis layer over the claim-only criteria, not an
executable sensor plan. It maps each recognition, completion, and reminder-check
claim to the cheapest plausible evidence path and the sensor-control knobs that
path would need.

## Corpus Counts

| item | count |
|---|---:|
| recipes | 24 |
| nodes / distinct step ids | 353 |
| total claims | 1418 |
| recognition claims | 353 |
| completion claims | 353 |
| check claims | 712 |

Check claims by subtype:

| subtype | count |
|---|---:|
| technique | 238 |
| preparation | 231 |
| measurement | 162 |
| timing | 48 |
| temperature | 33 |

## Role Taxonomy

- `A-solve`: cheap state, timer, or sensor event can settle the claim without a
  VLM call. Examples: graph/timer, microwave hum/beep, blender motor, inferred
  completion from a successor anchor.
- `B-trigger`: cheap evidence can wake one targeted RGB/VLM check. Examples:
  hand/object motion, sizzle onset, water flow, thermal anomaly, appliance display
  crop.
- `C-none`: no credible cheap trigger/solve; requires semantic RGB/VLM on a
  periodic or already-awake basis. Examples: ingredient identity, quantity, final
  visual quality.

Claim-level role mix. This counts every recognition, completion, and check claim
independently. It is useful for explaining which subclaims are cheap, but it is
not the right denominator for runtime sensor provisioning because multiple claims
inside one step run under the same active-step sensor budget.

| role | count | fraction |
|---|---:|---:|
| B-trigger | 591 | 41.7% |
| C-none | 463 | 32.7% |
| A-solve | 364 | 25.7% |

Role mix by claim kind:

| claim kind | total | A-solve | B-trigger | C-none |
|---|---:|---:|---:|---:|
| recognition | 353 | 21 | 320 | 12 |
| completion | 353 | 295 | 8 | 50 |
| check:timing | 48 | 48 | 0 | 0 |
| check:temperature | 33 | 0 | 33 | 0 |
| check:measurement | 162 | 0 | 0 | 162 |
| check:preparation | 231 | 0 | 6 | 225 |
| check:technique | 238 | 0 | 224 | 14 |

Interpretation: completion looks cheap because 277 completion claims are
explicitly inferred from the successor step's recognition event. Checks are the
hard part: measurement and preparation are mostly semantic, while technique is
often triggerable by motion/spill cues but still wants VLM confirmation.

## Runtime Max-Role Counts

At runtime, the step's sensor configuration has to cover all active recognition,
completion, and check claims. A stricter provisioning count therefore assigns
each node the highest-cost role among its claims:

`A-solve < B-trigger < C-none`.

Per-node max role:

| max role required by node | nodes | fraction |
|---|---:|---:|
| C-none | 291 | 82.4% |
| B-trigger | 56 | 15.9% |
| A-solve | 6 | 1.7% |

This is the more honest denominator for "what does this step need while it is
active?" The claim-level count says many subclaims are cheap; the node-level max
count says most recipe steps still contain at least one semantic check that can
force RGB/VLM unless the system chooses to skip, defer, or selectively sample
that check.

Breakdown by whether the node has reminder checks:

| node class | nodes | A-solve | B-trigger | C-none |
|---|---:|---:|---:|---:|
| with checks | 339 | 6 | 47 | 286 |
| no checks | 14 | 0 | 9 | 5 |

Subtypes that most often drive the node's highest-cost role:

| driver subtype/family | count |
|---|---:|
| preparation | 226 |
| measurement | 162 |
| technique | 50 |
| temperature | 15 |
| timing | 6 |

The dominant runtime bottleneck is therefore not step recognition; it is
fine-grained proactive checking. Ingredient/container identity and quantities
upgrade many otherwise-cheap steps to `C-none`.

## How Evidence Families Were Assigned

These families were assigned by a deterministic heuristic classifier, not by a
trained model and not by reading per-recording labels. The classifier consumed
only each generated criteria node's `instruction`, `recognition.claim`,
`completion.claim`, `completion.inferred_from`, and `checks[].reminder/claim`,
plus the detector vocabulary in `tasks/AUDIO_RUNTIME_LIBRARY.md`.

The priority order was:

1. Structural fields first: `completion.inferred_from != null` becomes
   `next_anchor_inference`; `check:timing` becomes `duration_logic`; passive
   wait steps become `passive_wait_timer`.
2. Check subtype next: `measurement` becomes `quantity_count_amount`; most
   `preparation` becomes `ingredient_identity_state`; `temperature` becomes
   either `thermal_heat_state` or `appliance_setting_visual`; `technique` is
   split into `spill_mess`, `manual_motion_technique`, or
   `visual_technique_quality`.
3. Recognition/completion action cues next: microwave verbs map to
   `microwave_event`; blender/puree/blitz operation maps to `motor_event`;
   rinse/running-water verbs map to `water_event`; cook/heat/fry/saute/simmer
   verbs map to `cook_heat_event`; add/pour/chop/slice/transfer/etc. map to
   `manual_action_recognition`.
4. Remaining visual semantic claims fall back to `silent_semantic_step` for
   recognition or `visual_end_state` for completion.

The key design choice is that the family names describe the cheapest plausible
evidence path for that claim, not the sensor that must always be on. For
example, `manual_action_recognition` means a cheap mono/RGB motion burst can
wake one VLM step-label query; it does not mean motion alone reliably recognizes
the step. Likewise, `thermal_heat_state` is a future-hardware hypothesis because
the current CC4D replay has RGB+audio, not thermal.

Known limitations: this pass is lexical and conservative but still imperfect. It
can misread container phrases, optional checks, or claims whose text mentions a
previous appliance step. The per-node max-role table is therefore best treated as
a planning estimate; a committed Stage-2 sensor plan should keep the per-claim
family, detector choice, and uncertainty note explicitly reviewable.

## Evidence Families

| family | count | cheapest evidence path |
|---|---:|---|
| next_anchor_inference | 277 | graph state plus successor recognition; no extra sensor |
| manual_action_recognition | 254 | mono/RGB motion burst, then one VLM step label |
| ingredient_identity_state | 225 | high-resolution RGB/VLM |
| manual_motion_technique | 180 | 5-15 fps mono/RGB ROI burst, VLM if quality matters |
| quantity_count_amount | 162 | high-resolution RGB/VLM; cheap CV only for easy coarse levels |
| cook_heat_event | 70 | D4/D3 audio plus thermal/RGB confirmation where needed |
| visual_end_state | 50 | RGB/VLM end-state check |
| duration_logic | 48 | D6 timer over active interval |
| spill_mess | 44 | mono/RGB ROI change trigger plus VLM confirmation |
| microwave_event | 32 | D1 hum/beep, camera off |
| thermal_heat_state | 25 | thermal low-fps pan/food ROI plus RGB/VLM fallback |
| visual_technique_quality | 14 | RGB/VLM |
| silent_semantic_step | 12 | periodic low-fps RGB/VLM |
| appliance_setting_visual | 8 | high-resolution RGB crop, OCR or VLM |
| water_prep_state | 6 | D5 water trigger plus RGB/VLM confirmation |
| water_event | 4 | D5 water on/off plus local context |
| passive_wait_timer | 4 | graph/timer from predecessor end |
| motor_event | 3 | D2 appliance motor, camera off |

## Modality Sets

| modality set | count | notes |
|---|---:|---|
| mono + RGB + VLM | 478 | manual action and technique claims where motion can wake semantic vision |
| RGB + VLM | 463 | ingredient, quantity, and visual end-state claims |
| graph next-anchor | 277 | inferred completions |
| audio + RGB + thermal + VLM | 70 | cook/heat claims; thermal is a proposed option, not in CC4D replay |
| graph timer | 52 | timing checks plus passive wait steps |
| audio only | 35 | microwave and motor events |
| RGB + thermal + VLM | 25 | heat/temperature checks without a reliable audio edge |
| audio + RGB + VLM | 10 | water-flow recognition/preparation checks |
| RGB OCR + RGB + VLM | 8 | appliance power/display/dial setting checks |

The current dataset is RGB+audio replay. Thermal is included as a research
hypothesis for future hardware profiles, mainly for pan heat, burner-off, water
temperature, burn risk, and preheat readiness.

## Sensor-Control Knobs

| sensor/control surface | useful parameters |
|---|---|
| graph/timer | tick rate, duration tolerance, predecessor/successor state, timeout fallback |
| microphone/DSP | power, detector arming by recipe stage, 16 kHz vs 48 kHz path, window size, hop, threshold/profile, max latency buffer |
| learned audio | class pool, sample rate, gap bridging, confidence threshold, NPU/GPU placement |
| RGB camera | off/on, fps, resolution, full frame vs ROI crop, burst length, exposure/HDR, focus, white balance, compression, RGB vs grayscale |
| monochrome camera | off/on, fps, resolution, ROI, exposure; good for motion/hand/action triggers, poor for ingredient identity/color doneness |
| thermal camera | off/on, fps, thermal resolution, pan/food ROI, absolute vs relative temperature, emissivity/calibration, threshold/trend rules |
| cheap CV compute | model/detector choice, ROI tracker, frame cadence, color vs grayscale, confidence trigger |
| VLM | never/single-trigger/periodic, frames per call, temporal window, max image dimension, ROI/full frame, prompt, confidence threshold |

Recommended minimal control vocabulary for Stage 2:

```json
{
  "camera_rgb": {
    "power": "off|on",
    "fps": 0.2,
    "resolution": "thumbnail|360p|720p|full",
    "color": "rgb|mono",
    "roi": "full|named_region",
    "burst_s": 0
  },
  "microphone": {
    "power": "off|on",
    "sample_rate_hz": 16000,
    "armed_detectors": ["D1", "D2", "D3", "D4", "D5"],
    "window_s": 10
  },
  "thermal": {
    "power": "off|on",
    "fps": 1,
    "roi": "pan|burner|food|full",
    "mode": "relative_trend|absolute_temp"
  },
  "vlm": {
    "policy": "never|single_trigger|periodic",
    "period_s": 10,
    "frames": 1,
    "max_dim": 768,
    "input": "roi|full_frame"
  }
}
```

## Per-Recipe Role Mix

Claim-level role mix by recipe:

| recipe | nodes | claims | A-solve | B-trigger | C-none |
|---|---:|---:|---:|---:|---:|
| blenderbananapancakes | 14 | 57 | 13 | 25 | 19 |
| breakfastburritos | 11 | 44 | 11 | 16 | 17 |
| broccolistirfry | 25 | 101 | 27 | 42 | 32 |
| buttercorncup | 12 | 51 | 15 | 22 | 14 |
| capresebruschetta | 11 | 52 | 10 | 22 | 20 |
| cheesepimiento | 11 | 47 | 14 | 17 | 16 |
| coffee | 16 | 58 | 17 | 24 | 17 |
| cucumberraita | 11 | 51 | 10 | 21 | 20 |
| dressedupmeatballs | 14 | 56 | 18 | 23 | 15 |
| herbomeletwithfriedtomatoes | 15 | 64 | 13 | 25 | 26 |
| microwaveeggsandwich | 12 | 51 | 17 | 16 | 18 |
| microwavefrenchtoast | 11 | 42 | 13 | 16 | 13 |
| microwavemugpizza | 14 | 53 | 16 | 18 | 19 |
| mugcake | 20 | 80 | 22 | 33 | 25 |
| panfriedtofu | 19 | 67 | 17 | 32 | 18 |
| pinwheels | 17 | 59 | 10 | 21 | 28 |
| ramen | 15 | 55 | 18 | 22 | 15 |
| sautedmushrooms | 17 | 71 | 19 | 33 | 19 |
| scrambledeggs | 23 | 96 | 23 | 44 | 29 |
| spicedhotchocolate | 7 | 30 | 9 | 13 | 8 |
| spicytunaavocadowraps | 17 | 68 | 12 | 28 | 28 |
| tomatochutney | 19 | 71 | 21 | 36 | 14 |
| tomatomozzarellasalad | 9 | 39 | 6 | 19 | 14 |
| zoodles | 13 | 55 | 13 | 23 | 19 |

Per-node max role by recipe:

| recipe | nodes | A-solve | B-trigger | C-none |
|---|---:|---:|---:|---:|
| blenderbananapancakes | 14 | 1 | 2 | 11 |
| breakfastburritos | 11 | 0 | 0 | 11 |
| broccolistirfry | 25 | 0 | 5 | 20 |
| buttercorncup | 12 | 0 | 2 | 10 |
| capresebruschetta | 11 | 0 | 1 | 10 |
| cheesepimiento | 11 | 0 | 0 | 11 |
| coffee | 16 | 1 | 2 | 13 |
| cucumberraita | 11 | 0 | 0 | 11 |
| dressedupmeatballs | 14 | 1 | 1 | 12 |
| herbomeletwithfriedtomatoes | 15 | 0 | 1 | 14 |
| microwaveeggsandwich | 12 | 0 | 0 | 12 |
| microwavefrenchtoast | 11 | 0 | 3 | 8 |
| microwavemugpizza | 14 | 1 | 2 | 11 |
| mugcake | 20 | 1 | 5 | 14 |
| panfriedtofu | 19 | 0 | 7 | 12 |
| pinwheels | 17 | 0 | 2 | 15 |
| ramen | 15 | 1 | 3 | 11 |
| sautedmushrooms | 17 | 0 | 4 | 13 |
| scrambledeggs | 23 | 0 | 5 | 18 |
| spicedhotchocolate | 7 | 0 | 2 | 5 |
| spicytunaavocadowraps | 17 | 0 | 0 | 17 |
| tomatochutney | 19 | 0 | 8 | 11 |
| tomatomozzarellasalad | 9 | 0 | 0 | 9 |
| zoodles | 13 | 0 | 1 | 12 |

## Takeaways

1. Claim-level counts and runtime provisioning counts answer different
   questions. Claim-level: 25.7% of claims are `A-solve`. Runtime max-role:
   only 1.7% of nodes are fully `A-solve` once checks are considered.
2. The cheapest large subclaim win is not a new sensor; it is exploiting the
   graph. More than three quarters of completion claims are next-anchor
   inferences, but many of those nodes still need RGB/VLM for checks.
3. Audio gives strong A-solve islands for microwave and motor steps and useful
   B-trigger islands for water/cooking, but most recipe claims are not acoustic.
4. Measurement and preparation checks are the semantic bottleneck: they mostly
   need RGB/VLM, often high-resolution ROI frames, and they dominate the
   per-node max-role upgrade to `C-none`.
5. Monochrome/RGB motion is valuable as a wake-up layer for manual actions and
   technique, but it should be treated as a trigger, not a final verifier.
6. Thermal is most defensible as a future hardware option for heat readiness,
   burn risk, water/pan temperature, and burner-off state. It does not replace
   RGB/VLM for ingredient identity, amount, or presentation quality.

## Suggested Workflow For Per-Recipe Brainstorms

For each recipe, generate a `*.brainstorm.json` before committing a Stage-2
sensor plan:

1. Enumerate every recognition, completion, and check claim.
2. Attach detector candidates in cheapest-first order: graph/timer, audio,
   cheap CV/thermal, VLM.
3. Assign `A-solve`, `B-trigger`, or `C-none`.
4. Emit a per-stage sensor schedule with camera/mic/thermal/VLM parameters.
5. Report cost by claim subtype, not only pooled across all claims.

This can be parallelized by recipe if the corpus grows; at the current 24-recipe
scale, the full pass fits comfortably in one audit.
