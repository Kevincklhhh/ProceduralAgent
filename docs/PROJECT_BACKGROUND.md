# Project Background: Vision, Dataset, and Constraints

> 🧭 Merges the former `PROJECT_MEMORY.md` (vision/North Star) and `STARTING_POINT.md` (2026-06-10 design study), keeping the durable "why" and the still-valid dataset/infra facts. **Current execution is the three boxes** (`PIPELINE_THREE_BOXES.md`); the baseline suite, metrics, annotation protocol, and stage schema from the old design study now live in `REMINDER_EVALUATION.md` (Box 2), `PROACTIVE_REMINDER_GT.md` (Box 1), and the Box-3 predictor docs (two-stage templates + `tasks/PROCEDURE_MONITOR_COMPILER.md` + `REMINDER_RUNTIME.md`). Superseded ideas (ThermalKitchens pivot, B0–B4 framing, hand-annotated windows) are dropped here and preserved in git history.

## 1. Vision / North Star

A wearable / smart-glasses procedural agent that supports human-performed tasks by **maintaining procedural state, deciding when to intervene vs. stay silent, and producing timely guidance** — the human stays the physical actor; the AI acts through guidance, memory, verification, and reminders. The interesting behavior is *not* better action recognition; it is the agentic loop around it (when to speak, when to stay quiet, when to ask, when to recover, what to remember).

Why procedural tasks: they give the same structure that makes coding agents work — a goal, expected steps, observable progress, possible mistakes, moments where intervention helps and moments where interruption harms. Value lands as risk reduction, error recovery, documentation, accessibility, or training.

**Scope boundary (vs robotics/VLA):** the human is the actor; the agent never emits motor control. Out of scope: improving base VLM perception accuracy, foundation-model scaling, generic video QA, prompt-only assistants.

**The narrowed thesis (2026-06-14):** the concrete contribution is **sensor control** — given the procedure ahead of time, schedule cheap RGB+audio detectors to gate/trigger an expensive VLM, reported as *energy/latency saved at equal coverage*, not as error-detection accuracy. See `sensor-control` framing in `PIPELINE_THREE_BOXES.md`.

**Current headline (2026-06-28) — the latency arm:** the live work has sharpened this into a
**reminder-latency** story: a reminder is useful only if it arrives shortly after the evidence
appears, so the deployable win is *low latency from evidence → spoken intervention* (achievable at a
constant ~133 ms trigger via pre-encoded prefill + bounded claim checks), with *reminder accuracy*
as the open risk (zero-shot detection still over-fires badly). Sensor control is the substrate that
makes the fast-trigger path real; latency is the headline and accuracy the unsolved half. See
`docs/LATENCY_STORY.md` for the current framing, metrics, and open problems.

## 2. Bottlenecks we care about (the motivation behind Box 2's metrics)

- Intervention policy: when to speak / stay silent / ask / escalate — false interruptions and missed interventions are *both* costly.
- Recovery, not just detection: after a deviation, is the task still recoverable and what next.
- Interaction-burden evaluation: measure unnecessary interruptions, missed/late interventions separately.
- System-resource evaluation: latency, power, model-call count, sensing duty cycle — measured separately from interaction burden.
- Real online behavior under bounded latency and partial evidence; robustness to imperfect plans.

## 2a. I/O & cost contract (the sensor-control ledger; salvaged from TASK_DEFINITION 2026-06-20)

The thesis is measured in cost at equal coverage, so every arm reports the same ledger. These
are the durable knobs and the cost fields; the per-call VLM request/response schema now lives
in code (`eval/proposed_vlm_arm.py`, `eval/baseline_t1_step.py`), not in a doc.

- **Sensing knobs:** RGB fps, audio sampling rate / bitrate, VLM `window_s` + `fps` (frames per
  call), tick_s. Set per arm in the plan's `vlm_policy` / `runtime_config`.
- **Cost log (per recording, every arm):** `vlm_calls`, `frames_sent`, `vlm_latency_total_s`,
  `compute_s`, `audio_on_s` — emitted by `eval/proposed_runtime.py` under `cost`. This is what a
  cheap-detector plan must beat at equal Box-2 quality.
- **Quality is scored separately** (Box 2): the cost ledger is never traded against coverage in
  a single number — Pareto only (see risk 2 below).

## 3. Dataset decision (verified 2026-06-10, still in force)

**Primary: CaptainCook4D** — open download (no access form), 24 recipe DAGs + step timestamps + 8 error-type tags, 384 recordings, 6 official splits, loud embedded GoPro audio (AAC 48 kHz; 360p variant verified sample-identical to 4K for detectors, ~45 GB total). It is the only released set combining audio + step structure + error labels, and **no published work uses its audio** — the unclaimed combination.

| Dataset | Role | Status |
|---|---|---|
| **CaptainCook4D** | primary replay corpus (audio + step + error GT) | chosen; on disk |
| **Qualcomm Interactive Cooking** (LiveMamba, 2511.21998) | adds timestamped instruction/feedback/mistake events over all 384 CC4D recordings → the Box-1 GT timestamp source + the heavyweight end-to-end baseline | on disk; research-only DLA |
| **HD-EPIC** | naturalistic transfer (quiet audio, recipe-step GT, no errors) | on disk, 116 GB, cycle-2 |
| **EPIC-SOUNDS** | audio-detector calibration corpus (78.4k labeled segments) | grounds the audio primitives (`research/DETECTOR_CATALOG.md`) |
| EgoPER | frame-exact error onsets | email-gated; promote if access lands |
| WTaG | real human when-to-talk anchor (defuses circularity) | license form; 3 recipes |
| EgoProactive/Pro2Bench | public when-to-speak challenge | released but **audio-stripped**, 0.7–3.9 s micro-action granularity — appendix only |

Rejected as primary: self-recording (N too small, blinding impossible in a small team, IRB-heavy); EgoProactive as anchor (granularity mismatch, no audio).

## 4. Infra (2026-06-10 snapshot)

GPU 0 (RTX 6000 Ada 48 GB) free; GPUs 1–7 contended. Qwen3-VL-4B-FP8 serves on vLLM (Pro2Assist's reasoner scale → parity backend). Per project policy, run VLMs on Qwen (saltyfish), not Gemini. SAM2 + GroundingDINO checkpoints on disk.

## 5. Standing risks (carry into every experiment)

1. **Cheap detectors brittle on 360p egocentric video** (head motion swamps frame-diff; ROI loss). Mitigation: ROI-locked checks, lightweight trackers, the re-tier-don't-tune rule (a failing cheap criterion is demoted to VLM-escalation, honestly reported — degrades cost savings, not correctness).
2. **Attribution attack** — savings explained by duty-cycling + duration priors alone. Mitigation: timer-only and activity-gated-VLM controls are non-negotiable; claim is Pareto dominance over all control planes.
3. **Annotation circularity** — the GT must not be derived from our detectors. Mitigation: the firewall (`gt-predictor-firewall`) + mechanical-only GT + WTaG human anchor.
4. **Compute/serving** — one contended GPU; freeze prompts after a small pilot; 4B FP8 primary.
5. **Hollow parity** — matching a weak baseline cheaply isn't assistance evidence. Mitigation: pre-registered absolute quality floors; per-class recall so uniform failure on hard classes can't hide; oracle-stage ablation.

## 6. Durable deliverable

The open instantiation of the Pro2Assist/PWR proactive-assistance evaluation that neither released: CC4D reminder-timing GT (Box 1), compiled sensor-stage maps (Box 3), frozen decision-point sets + replay/cost harness (Box 2) — a citable benchmark, the first proactive-timing benchmark anywhere with non-speech audio available to systems.
