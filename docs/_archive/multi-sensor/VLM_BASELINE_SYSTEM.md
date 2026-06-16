# Baseline System: Periodic VLM Cooking Observer

## Goal

Build a baseline where a VLM watches a cooking video and periodically decides:

- what procedure stage the user is in,
- whether the current stage is complete,
- whether a mistake or hazard is visible,
- whether the assistant should remind, warn, ask for confirmation, or stay silent.

This baseline does not use cheap RGB criteria or graph-gated escalation. It calls the VLM on a fixed schedule or dense frame schedule.

## Baseline Variants

Use at least two variants:

1. `periodic_vlm`: call the VLM every `N` seconds.
2. `dense_vlm`: call the VLM on every sampled frame window.

Optional variants:

3. `step_prompted_vlm`: provide the full recipe and ask the VLM to infer the current stage.
4. `oracle_step_vlm`: provide the ground-truth current step and ask only for completion, mistake, and reminder judgment.

The oracle variant isolates visual judgment from stage tracking.

## System Inputs

### Static Task Input

```json
{
  "task_id": "pan_fried_egg",
  "recipe_title": "Pan-fried egg",
  "steps": [
    {
      "step_id": "setup_pan",
      "order": 1,
      "instruction": "Put a pan on the stove."
    },
    {
      "step_id": "add_oil",
      "order": 2,
      "instruction": "Add oil or butter."
    },
    {
      "step_id": "crack_egg",
      "order": 3,
      "instruction": "Crack the egg into the pan."
    },
    {
      "step_id": "cook_white",
      "order": 4,
      "instruction": "Cook until the white is mostly set."
    },
    {
      "step_id": "plate_egg",
      "order": 5,
      "instruction": "Move the cooked egg to a plate."
    },
    {
      "step_id": "turn_off_stove",
      "order": 6,
      "instruction": "Turn off the stove."
    }
  ],
  "allowed_assistant_actions": [
    "none",
    "reminder",
    "warning",
    "ask_confirmation"
  ]
}
```

### Runtime Video Input

```json
{
  "video_id": "run_001",
  "camera_view": "egocentric_or_tripod",
  "fps": 30,
  "resolution": "1920x1080",
  "duration_s": 420,
  "sampling_policy": {
    "mode": "periodic",
    "vlm_interval_s": 5,
    "frames_per_call": 3,
    "window_s": 2
  }
}
```

### Per-Call VLM Input

```json
{
  "call_id": "run_001_t0030",
  "video_id": "run_001",
  "timestamp_s": 30.0,
  "frame_window": {
    "start_s": 29.0,
    "end_s": 31.0,
    "frames": [
      "frame_000870.jpg",
      "frame_000900.jpg",
      "frame_000930.jpg"
    ]
  },
  "task_context": {
    "recipe_title": "Pan-fried egg",
    "ordered_steps": [
      "setup_pan",
      "add_oil",
      "crack_egg",
      "cook_white",
      "plate_egg",
      "turn_off_stove"
    ],
    "recent_stage_history": [
      {
        "timestamp_s": 25.0,
        "stage_id": "crack_egg",
        "confidence": 0.72
      }
    ],
    "last_assistant_action": {
      "timestamp_s": 20.0,
      "type": "none"
    }
  },
  "output_schema_version": "vlm_baseline_v1"
}
```

## VLM Prompt Contract

The prompt should force structured output and discourage invented observations.

```text
You are observing a cooking task from a camera. Use only visible evidence in the provided frames and the recipe context.

Return JSON only. Do not include free-form prose.

Decide:
1. Which recipe stage is most likely happening now.
2. Whether the current stage appears not started, in progress, complete, skipped, or uncertain.
3. What visible evidence supports your judgment.
4. Whether any visible mistake or safety issue appears.
5. Whether the assistant should say nothing, remind, warn, or ask for confirmation.

If the frames do not contain enough evidence, use "uncertain" and explain what is missing.
```

## System Outputs

### Per-Call VLM Output

```json
{
  "call_id": "run_001_t0030",
  "timestamp_s": 30.0,
  "stage": {
    "stage_id": "cook_white",
    "status": "in_progress",
    "confidence": 0.68
  },
  "evidence": [
    {
      "type": "visual",
      "description": "An egg appears in the pan."
    },
    {
      "type": "visual",
      "description": "The egg white looks partly opaque."
    }
  ],
  "completion": {
    "is_complete": false,
    "confidence": 0.62,
    "reason": "The white is not fully set."
  },
  "mistakes_or_hazards": [
    {
      "type": "possible_burning",
      "severity": "low",
      "confidence": 0.31,
      "evidence": "No clear burning is visible, but the pan area is partly occluded."
    }
  ],
  "assistant_action": {
    "type": "none",
    "message": "",
    "urgency": "none",
    "reason": "The task appears to be progressing normally."
  },
  "uncertainty": {
    "needs_more_visual_context": true,
    "missing_evidence": ["clear view of egg surface", "stove heat setting"]
  }
}
```

### Run-Level Output

```json
{
  "video_id": "run_001",
  "task_id": "pan_fried_egg",
  "baseline_mode": "periodic_vlm",
  "vlm_interval_s": 5,
  "summary": {
    "num_vlm_calls": 84,
    "num_reminders": 3,
    "num_warnings": 1,
    "num_confirmation_requests": 2
  },
  "stage_timeline": [
    {
      "stage_id": "setup_pan",
      "start_s": 0,
      "end_s": 18,
      "confidence_mean": 0.76
    },
    {
      "stage_id": "add_oil",
      "start_s": 19,
      "end_s": 35,
      "confidence_mean": 0.61
    }
  ],
  "events": [
    {
      "timestamp_s": 35,
      "event_type": "assistant_action",
      "action_type": "reminder",
      "message": "Add the egg when the pan is ready."
    }
  ],
  "cost_log": {
    "total_frames_sent": 252,
    "estimated_input_tokens": 0,
    "estimated_output_tokens": 0,
    "mean_latency_ms": 0,
    "p95_latency_ms": 0,
    "parse_failure_rate": 0.0
  }
}
```

## Runtime Pipeline

1. Load recipe and ordered steps.
2. Sample RGB video using the baseline policy.
3. For each timestamp, send a short frame window plus recipe context to the VLM.
4. Parse structured JSON.
5. Smooth stage predictions over time:
   - keep the previous stage unless the new stage confidence exceeds a threshold, or
   - transition when the same new stage appears in `K` consecutive VLM calls.
6. Emit an assistant action if the VLM requests one and the cooldown period has passed.
7. Log every frame window, VLM output, action, latency, and parse failure.

## Baseline Runtime State

```json
{
  "current_stage_id": "cook_white",
  "stage_confidence": 0.68,
  "recent_predictions": [
    {"timestamp_s": 20, "stage_id": "crack_egg", "confidence": 0.72},
    {"timestamp_s": 25, "stage_id": "cook_white", "confidence": 0.64},
    {"timestamp_s": 30, "stage_id": "cook_white", "confidence": 0.68}
  ],
  "last_assistant_action_s": 20,
  "cooldown_s": 15,
  "parse_failures": 0
}
```

## Evaluation Inputs

Ground-truth annotations:

```json
{
  "video_id": "run_001",
  "stage_segments": [
    {
      "stage_id": "setup_pan",
      "start_s": 0,
      "end_s": 17.4
    },
    {
      "stage_id": "add_oil",
      "start_s": 17.5,
      "end_s": 33.2
    }
  ],
  "mistake_events": [
    {
      "event_id": "missed_turn_off_stove",
      "type": "safety",
      "start_s": 210.0,
      "end_s": 230.0,
      "expected_action": "warning"
    }
  ],
  "valid_reminder_windows": [
    {
      "stage_id": "cook_white",
      "start_s": 90.0,
      "end_s": 110.0,
      "expected_action": "reminder"
    }
  ]
}
```

## Evaluation Metrics

Stage tracking:

- frame-level stage accuracy,
- segment-level F1,
- transition timing error,
- time spent in `uncertain`.

Assistance:

- reminder precision and recall,
- warning precision and recall,
- action timing error,
- unnecessary-interruption count.

Cost:

- VLM calls per task,
- frames sent to VLM,
- mean and p95 VLM latency,
- output parse failure rate,
- estimated model cost.

## Why This Baseline Matters

This is the simplest strong baseline for the RGB-only project. It asks: if we let the VLM watch periodically, how well does it track cooking and issue reminders?

The graph-guided system should beat this baseline on VLM calls, latency, and interruption burden while keeping comparable stage tracking and reminder quality.
