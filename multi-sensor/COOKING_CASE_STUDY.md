# Cooking Case Study: Pan-Fried Egg

## Purpose

This is a manual sketch of what the executable sensing-and-assistance graph could look like for one simple dish. Later, an LLM/MLLM prompt could generate the same structure from a recipe or tutorial video.

The important point is that each procedure step must become more than text. It should define:

- what the user is trying to accomplish,
- what simple evidence can verify progress,
- which sensors should be active,
- what mistakes or hazards to watch for,
- when to remind, warn, or escalate to RGB/VLM.

## Sensor Cost Assumption

Approximate sensing tiers:

- First prototype: RGB only, where low cost means simple RGB criteria and high cost means VLM.
- Low cost: timer, microphone, thermal at low rate.
- Medium cost: monochrome motion, RGB burst, thermal at higher rate.
- High cost: continuous RGB, VLM call, full multimodal sensing.

The system should use cheap sensors only when the step has simple checkable criteria. Otherwise it should escalate or ask the user to confirm.

## Dish Procedure

Task: cook one pan-fried egg.

User-facing recipe:

1. Put a pan on the stove.
2. Preheat the pan.
3. Add oil or butter.
4. Crack the egg into the pan.
5. Cook until the white is mostly set.
6. Optionally flip or cover the egg.
7. Plate the egg.
8. Turn off the stove.

## Executable Sensing-and-Assistance Graph

| Step | User goal | Completion criteria | Mistake/hazard criteria | Sensor policy | Proactive reminder |
| --- | --- | --- | --- | --- | --- |
| `setup_pan` | Pan is on burner and workspace is clear. | Pan-like object visible on burner; burner region localized. | No pan detected after task starts; flammable item near burner. | RGB burst or VLM once for scene setup; then RGB off. | If no pan is detected, ask user to place pan on burner. |
| `preheat_pan` | Bring pan to cooking temperature. | Pan thermal region rises above baseline and reaches target range. | Pan overheats; burner on but pan absent; heating lasts too long. | Thermal low rate; audio off; RGB off after pan is localized. | When target range is reached, prompt user to add oil. Warn if temperature exceeds safe range. |
| `add_oil` | Add oil or butter to pan. | Short hand motion near pan plus small thermal/visual surface change; optional sizzling if pan is hot. | No oil added after preheat; excessive smoke or temperature spike. | Thermal low rate; mono/RGB burst on motion; VLM only if oil/butter presence is unclear. | If pan remains hot without oil for too long, remind user to add oil or lower heat. |
| `crack_egg` | Put egg into pan. | Egg-like object appears in pan; sizzling onset; pan temperature drops slightly. | Egg shell may fall in pan; egg added before oil; egg added when pan too hot. | Audio on for sizzle; thermal low rate; RGB burst after hand motion; VLM if object identity or shell concern is unclear. | If oil is ready and no egg is added, prompt user to crack egg. Warn if pan is too hot before egg is added. |
| `cook_white` | Cook until egg white sets. | Time since egg added; white region becomes more opaque; sizzle continues then stabilizes. | Burning risk from high temperature; egg sticking; white still translucent after expected time. | Timer; audio low rate; thermal low rate; sparse RGB bursts every N seconds; VLM only for doneness uncertainty. | At expected doneness window, suggest checking the white. Warn if pan temperature stays too high. |
| `flip_or_cover` | Finish top side if desired. | User flips egg, covers pan, or explicitly skips this step. | Flip attempted too early; pan covered too long; no action after prompt. | Mono/RGB burst for hand/tool motion; thermal low rate; VLM if flip/cover state is ambiguous. | Ask whether user wants sunny-side-up or flipped. If flipped, start short finish timer. |
| `plate_egg` | Move cooked egg to plate. | Egg leaves pan and plate-like region receives food. | Egg remains in hot pan too long; utensil/hand close to hot pan edge. | RGB burst or mono motion; thermal low rate for hot-pan warning. | Prompt to plate when doneness criteria are met. Warn if hand is near hot region. |
| `turn_off_stove` | End task safely. | Burner heat source decreases or user confirms stove is off. | Pan/burner remains hot after plating; user walks away while burner appears on. | Thermal low rate; RGB/VLM only if burner control state is visually ambiguous. | After plating, remind user to turn off stove. Warn if heat persists. |

## Example Stage Object

```json
{
  "stage_id": "preheat_pan",
  "goal": "bring the pan to cooking temperature before adding oil",
  "completion_criteria": [
    {
      "name": "pan_region_heating",
      "sensor": "thermal",
      "detector": "localized_temperature_rise",
      "success": "pan temperature rises above baseline for a stable interval"
    },
    {
      "name": "target_temperature_reached",
      "sensor": "thermal",
      "detector": "temperature_range_check",
      "success": "pan is within task-specific target range"
    }
  ],
  "mistake_criteria": [
    {
      "name": "overheat",
      "sensor": "thermal",
      "condition": "pan temperature exceeds safe range",
      "action": "warn immediately and suggest lowering heat"
    },
    {
      "name": "preheat_timeout",
      "sensor": "timer",
      "condition": "expected preheat duration exceeded without temperature rise",
      "action": "ask whether burner is on"
    }
  ],
  "sensor_policy": {
    "on": ["thermal_low_rate", "timer"],
    "burst": [],
    "off": ["rgb", "microphone", "microphone_array"],
    "escalate_to_vlm_when": [
      "pan region cannot be localized",
      "thermal reading conflicts with expected burner/pan state"
    ]
  },
  "proactive_reminders": [
    {
      "condition": "target_temperature_reached",
      "message": "Pan is ready. Add oil or butter."
    }
  ]
}
```

## Where Cheap Sensing Works

Good cheap-sensing stages:

- `preheat_pan`: thermal checks the main goal.
- `cook_white`: timer, thermal, and sparse RGB can cover most progress.
- `turn_off_stove`: thermal can detect residual heat or cooling trend.

Mixed stages:

- `add_oil`: cheap motion and thermal cues help, but oil presence may need RGB/VLM.
- `crack_egg`: audio and thermal detect the event, but shell mistakes need vision.
- `flip_or_cover`: motion helps, but semantic state may need RGB/VLM.

Poor cheap-sensing stages:

- "Egg is perfectly cooked to user preference."
- "Season to taste."
- "The food looks appetizing."

These should use VLM, user preference, or explicit confirmation rather than pretending simple sensors are enough.

## Prompt-Generation Target

A future compiler prompt could ask the model to produce the table above from a recipe:

```text
Given a cooking procedure and a smartglass sensor profile, convert each step into an executable sensing-and-assistance stage. For each stage, output: goal, completion criteria, mistake/hazard criteria, required sensors, cheap-sensing conditions, VLM escalation conditions, proactive reminders, and expected duration.
```

The prompt output should be reviewed or constrained by safety rules before use, because a plausible generated policy can still be unsafe.
