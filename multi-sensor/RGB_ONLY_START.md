# RGB-Only Starting Point

## Why Start Here

An RGB-only prototype is a good first step. It removes the uncertainty about thermal/audio hardware and tests the central idea:

> Can a procedure graph define simple visual criteria that avoid unnecessary VLM calls?

In this version, "cheap sensing" does not mean non-RGB sensors. It means cheap computation on RGB:

- low-rate frame sampling,
- frame differencing,
- object/region tracking,
- color/texture thresholds,
- hand/object motion near a known region,
- small local classifiers,
- simple temporal rules.

The expensive operation is VLM reasoning. The graph decides when simple RGB criteria are enough and when the system should call the VLM.

## Revised Claim

First-stage claim:

> Procedure-derived criteria can reduce VLM usage by turning each cooking step into cheap RGB checks plus explicit VLM escalation rules.

Later multisensor claim:

> The same graph can also choose among RGB, thermal, audio, and other sensors when those sensors are available.

This keeps the first prototype feasible while preserving the long-term multisensor direction.

## RGB-Only Graph Fields

Each stage should define:

- `goal`: what the user should accomplish,
- `roi`: expected visual region, such as pan, stove, plate, counter,
- `cheap_rgb_criteria`: visual checks that do not need VLM,
- `mistake_criteria`: visible mistakes or hazards,
- `vlm_needed_when`: when simple RGB checks are insufficient,
- `reminder_condition`: when to prompt the user,
- `sampling_policy`: low-rate, burst, or continuous RGB.

## Pan-Fried Egg RGB-Only Example

| Step | Cheap RGB criteria | Mistake criteria | VLM needed when | Reminder |
| --- | --- | --- | --- | --- |
| `setup_pan` | Pan-shaped object appears in stove ROI. | No pan after task starts; clutter near burner. | Object/region localization fails. | Ask user to place pan on burner. |
| `add_oil` | Hand/container motion over pan; reflective surface change. | No visible add action after preheat timer. | Need to distinguish oil, butter, or another ingredient. | Remind user to add oil/butter. |
| `crack_egg` | Hand motion over pan; egg-like white/yellow region appears. | Egg appears outside pan; possible shell-like fragment. | Egg/shell identification uncertain. | Prompt user to crack egg after oil step. |
| `cook_white` | White region grows more opaque; edges change shape; elapsed time in range. | Darkening/burning-like color; no opacity change after expected time. | Doneness is ambiguous or user preference matters. | Suggest checking egg white. |
| `flip_or_cover` | Spatula/hand motion near pan; egg shape/orientation changes or lid appears. | Flip attempted before white set; cover remains too long. | Need to determine whether egg actually flipped. | Ask user whether to flip or keep sunny-side-up. |
| `plate_egg` | Egg region moves from pan ROI to plate ROI. | Egg remains in pan after done; hand near hot pan area. | Plate/egg tracking fails. | Prompt user to plate the egg. |
| `turn_off_stove` | User hand reaches control area; knob/display visual state changes if visible. | User leaves scene while stove area still appears active. | Burner/control state is occluded or not visually reliable. | Remind user to turn off stove. |

## Example Stage Object

```json
{
  "stage_id": "cook_white",
  "goal": "cook until the egg white is mostly set",
  "roi": ["pan_region", "egg_region"],
  "cheap_rgb_criteria": [
    {
      "name": "white_opacity_increase",
      "method": "track color/texture change in egg-white region over time",
      "success": "white region becomes more opaque for a stable interval"
    },
    {
      "name": "elapsed_time_window",
      "method": "timer since egg appeared in pan",
      "success": "elapsed time enters expected doneness window"
    }
  ],
  "mistake_criteria": [
    {
      "name": "possible_burning",
      "method": "dark region grows near egg edge or pan surface",
      "action": "warn or suggest lowering heat"
    },
    {
      "name": "not_setting",
      "method": "opacity does not change after expected time",
      "action": "suggest checking heat level"
    }
  ],
  "sampling_policy": {
    "default": "low_rate_rgb",
    "burst_when": ["hand enters pan ROI", "expected transition window starts"],
    "continuous_when": ["possible hazard", "tracking lost"]
  },
  "vlm_needed_when": [
    "egg-white state is visually ambiguous",
    "user preference determines doneness",
    "cheap criteria conflict"
  ],
  "proactive_reminders": [
    {
      "condition": "elapsed time reaches expected doneness window",
      "message": "Check whether the egg white is set."
    }
  ]
}
```

## What This Tests

This prototype tests whether the graph can reduce expensive semantic reasoning even when the only sensor is RGB.

Metrics:

- VLM calls per task.
- Stage tracking accuracy.
- Transition timing error.
- Reminder precision and missed-reminder rate.
- False escalation rate: cheap criteria could have handled it but VLM was called.
- Missed escalation rate: cheap criteria were trusted but VLM was needed.

Baselines:

- VLM every N seconds.
- VLM at every procedure step.
- Continuous VLM on all frames.
- Cheap RGB criteria without procedure graph.

## Limitations

RGB-only cannot reliably measure heat, audio events, or hidden stove state. It also struggles with occlusion, lighting, camera angle, transparent oil, and subjective doneness.

That is acceptable for the first prototype. The point is to validate the representation and VLM-gating policy before adding other sensors.
