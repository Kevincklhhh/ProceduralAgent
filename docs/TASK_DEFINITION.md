# Procedure Task Definition

> 🧭 **Status (2026-06-15): generic task/I-O schema, partially superseded.** This is the shared *task contract* (steps + reminder shape + I/O), not a per-box spec. Reconcile with the three-box restructure (`PIPELINE_THREE_BOXES.md`): (1) reminders are now **class-typed** — every reminder/event carries `class ∈ {precondition_violation, parameter_violation, execution_error}` so Box 2 can score per-class (`REMINDER_EVALUATION.md`); the authoritative GT event schema is `FAMILY_A_CC4D_AUGMENTATION.md` §4. (2) **Safety is EXCLUDED** (no GT) — the `"type":"safety"` examples below are illustrative only, not a scored class. (3) The `recent_stage_history` / periodic-VLM framing in §4 is **one Box-2 arm**, not the task definition; Box 3 predicts stage state rather than being handed it.

rgb fps
audio sampling rate, bitrate per second
throughput and latency



Four things are defined here: the procedure task (steps + proactive reminders), the ground-truth annotation structure, and the system input/output contract. One concrete task instance lives in `task_pan_fried_egg.json`; the annotation template lives in `annotation_template.json`.

## 1. Procedure Task (steps)

A procedure task is an ordered list of steps the user is expected to perform.

```json
{
  "task_id": "pan_fried_egg",
  "title": "Pan-fried egg",
  "steps": [
    {
      "step_id": "preheat_pan",
      "order": 2,
      "instruction": "Preheat the pan on the stove.",
      "expected_duration_s": {"min": 30, "typical": 90, "max": 240}
    }
  ],
  "allowed_assistant_actions": ["none", "reminder", "warning", "ask_confirmation"]
}
```

- `step_id`: stable identifier, used everywhere (predictions, annotations, reminders).
- `order`: nominal position; execution may deviate (skip, repeat, pause).
- `expected_duration_s`: optional prior, used later for timers; ignored by the VLM baseline.

## 2. Proactive Reminders

Reminders are part of the task definition. Each one names the step it belongs to, a plain-language trigger condition, and the message to deliver.

```json
{
  "reminder_id": "add_oil_when_hot",
  "step_id": "preheat_pan",
  "class": "precondition_violation",
  "trigger": "pan looks preheated and no oil has been added yet",
  "message": "The pan looks ready. Add oil or butter.",
  "type": "reminder"
}
```

- `class` (required for scoring): `precondition_violation | parameter_violation | execution_error` — the three GT-backed scored classes (`PIPELINE_THREE_BOXES.md`). Box 2 scores per-class.
- `type`: `reminder` (helpful nudge) or `warning`. Note: safety warnings (stove left on, etc.) are a deployment behavior but **not a scored class** — excluded from GT (no annotation source).
- For the VLM baseline these are prompt context only — the VLM decides when to act. Later systems will execute the triggers directly.

## 3. Annotation Structure (ground truth, one file per video)

```json
{
  "video_id": "run_001",
  "task_id": "pan_fried_egg",
  "stage_segments": [
    {"step_id": "preheat_pan", "start_s": 12.0, "end_s": 95.5}
  ],
  "reminder_windows": [
    {"reminder_id": "add_oil_when_hot", "start_s": 80.0, "end_s": 110.0, "expected_action": "reminder"}
  ],
  "mistake_events": [
    {"event_id": "stove_left_on", "type": "safety", "start_s": 300.0, "end_s": 330.0, "expected_action": "warning", "_note": "safety is EXCLUDED from scored GT — illustrative only; see PIPELINE_THREE_BOXES.md"}
  ]
}
```

- `stage_segments`: contiguous partition of the video into step_ids (plus `"other"` for off-task time). Gives frame-level stage accuracy and transition timing error.
- `reminder_windows`: the interval in which issuing the reminder is correct. An emitted reminder inside the window with matching id = true positive; outside any window = unnecessary interruption; window with no reminder = miss.
- `mistake_events`: visible mistakes/hazards with the action the assistant should have taken.

## 4. System Input / Output

### Per-call input (what the VLM sees each call)

```json
{
  "call_id": "run_001_t0030",
  "timestamp_s": 30.0,
  "frames": ["<N jpeg frames from the last window_s seconds>"],
  "task_context": {
    "title": "Pan-fried egg",
    "steps": ["<ordered step_id: instruction list>"],
    "reminders": ["<reminder trigger + message list>"],
    "recent_stage_history": [{"timestamp_s": 25.0, "step_id": "crack_egg", "confidence": 0.72}],
    "last_assistant_action": {"timestamp_s": 20.0, "type": "none"}
  }
}
```

### Per-call output (strict JSON, enforced)

```json
{
  "step_id": "cook_white",
  "status": "in_progress",
  "confidence": 0.7,
  "evidence": ["egg visible in pan, white partly opaque"],
  "hazard": null,
  "action": {"type": "none", "message": "", "reason": "progressing normally"}
}
```

- `status` ∈ `not_started | in_progress | complete | uncertain`.
- `action.type` ∈ `none | reminder | warning | ask_confirmation`.
- `hazard`: short string if a visible mistake/safety issue, else null.

### Run-level output (one per video)

```json
{
  "video_id": "run_001",
  "task_id": "pan_fried_egg",
  "baseline": {"mode": "periodic_vlm", "interval_s": 10, "frames_per_call": 3, "model": "gemini-2.5-flash"},
  "stage_timeline": [{"step_id": "preheat_pan", "start_s": 10, "end_s": 90}],
  "events": [{"timestamp_s": 95, "action_type": "reminder", "message": "Add oil or butter."}],
  "cost_log": {
    "num_vlm_calls": 42, "frames_sent": 126,
    "mean_latency_s": 1.8, "p95_latency_s": 3.1, "parse_failure_rate": 0.0
  }
}
```

`stage_timeline` is the smoothed stage belief over time (a transition requires the same new step_id in K=2 consecutive calls). `events` are all non-`none` assistant actions after the cooldown filter. `cost_log` is what graph-gated systems must beat at equal quality.
