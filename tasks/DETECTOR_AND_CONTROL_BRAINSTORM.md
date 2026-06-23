# Detector & Sensor-Control Brainstorm (research ideation, pre–Stage 2)

A **brainstorming** pass that turns a recipe's generated criteria into a research plan for
*how* to sense each claim cheaply and *what to schedule ahead of time*. It is the ideation
that precedes Stage 2 (criteria → sensorplan): Stage 2 commits one binding per claim; this
step lays out the option space and the sensor-control opportunities so that commitment is
informed.

This is a *process spec* an agent (or a person) follows. It does **not** write a sensorplan —
it produces a brainstorm artifact for review.

> **What we are optimizing (keep the thesis straight).** The contribution is **sensor
> CONTROL** — energy/latency saved *at equal coverage*, not "% of errors detected." For every
> claim we ask: can a cheap sensor settle it so the camera/VLM can sleep or fire just once?
> Report the fraction that *can't* be cheapened honestly — don't hide it.

---

## 1. Input / output

| | |
|---|---|
| **Input** | one `tasks/cc4d_probe/<recipe>.generated.criteria.json` (the 21 non-loop recipes). Read the recipe DAG (`tasks/cc4d/<recipe>.json`) too for stage adjacency / timing. |
| **Reference** (read, don't invent blindly) | `tasks/AUDIO_RUNTIME_LIBRARY.md` (D1–D6 audio catalog), `docs/research/DETECTOR_CATALOG.md`, the RGB probes under `detectors/probes/` (what cheap CV has been shown to do / fail at). |
| **Output** | `tasks/cc4d_probe/<recipe>.brainstorm.json` — per-claim detector options + sensor-control plan, plus a recipe-level sensor schedule and honest cost estimate. |

A *claim* = every `recognition.claim`, every `completion.claim`, and every `checks[].claim`
in the criteria. Brainstorm **all** of them.

---

## 2. The three knobs the brainstorm reasons over

### 2a. Detector modality — how could this claim be sensed?
For each claim, propose candidates across the modalities, cheapest-first:

- **audio** — a catalog detector (microwave hum `D1`, motor `D2`, cook start/end `D3`/`D4`,
  water `D5`, timer `D6`) or a new acoustic cue. Cheap, always-on-able.
- **cv** — cheap on-device computer vision on RGB frames: motion/optical-flow, HSV/color
  **state-change**, appliance LED/screen state, template/QR, hand-in-ROI. Mid cost.
  (Honor the known limits: HSV state-change detects *that* an add happened but **can't count**;
  a skin hand-gate was a dead end — see the RGB probe results.)
- **vlm** — the expensive sensor. Use only where audio/CV can't settle it (fine technique,
  ingredient identity, counts, free-form amounts).

Tag each candidate with **feasibility** (high / med / low) and one line of *why*.

### 2b. Sensing role — how cheaply can it be settled?
Classify the claim by the best available candidate:
- **A-solve** — a cheap sensor settles it alone (timing via `D6`, appliance state via `D1`,
  order/precondition via DAG state). Camera/VLM can sleep. *The win slice.*
- **B-trigger** — a cheap event fires **one** targeted VLM call (duty-cycled, not continuous).
- **C-none** — no cheap solve/trigger; needs the camera on + VLM (continuous/periodic).
  **Report this fraction; don't round it away.**

### 2c. Sensor-control parameters — what can we plan ahead?
Given the procedure is known in advance, schedule the controllable knobs per stage. Enumerate,
per claim, the cheapest *sufficient* configuration:

- **RGB camera** — `power` OFF/ON (sleep is the biggest energy lever); `fps` (e.g. 30 / 1 /
  0.2); `resolution` (full / 360p / thumbnail); `color` RGB vs **monochrome** (cheaper capture
  + compute); `roi` full-frame vs crop (e.g. just the pan/mug region).
- **Microphone** — which detectors are *armed* vs asleep; sampling rate. (Cheap enough to leave
  always-on; the lever is which detectors compute.)
- **VLM** — `when`: never / single trigger / periodic(cadence); `frames` per call; `resolution`
  per call.
- **CV compute** — which cheap detectors run this stage (each costs something).

The plan-ahead insight: a claim's **DAG position** tells you what to schedule. E.g. a claim
that lives *during a microwave step* → camera OFF, audio `D1` only; a `measurement` claim
(count pieces) → wake camera to full-res RGB for one B-trigger VLM frame; a silent
`completion` inferred from a successor → camera can stay low/off until the successor's anchor.

---

## 3. Per-claim procedure

For each claim:
1. State the claim and its kind (`recognition` / `completion` / `check:<subtype>`) and its host
   step + DAG neighbours.
2. List detector candidates (§2a) across modalities, feasibility-tagged.
3. Assign the **sensing role** (§2b) from the best candidate.
4. Give the cheapest sufficient **sensor-control config** (§2c) and what to schedule ahead
   given the step's position in the recipe.
5. Note open questions / what a probe would need to verify feasibility.

Lean on priors already established: timing/appliance claims are usually **A-solve** (audio +
`D6`); order/missing claims are **A-solve** off DAG state (no sensor); counts and fine
technique are usually **C-none** or at best **B-trigger**.

---

## 4. Output schema

`tasks/cc4d_probe/<recipe>.brainstorm.json`:

```json
{
  "recipe": "<stem>",
  "_kind": "detector_and_control_brainstorm (research ideation, not a sensorplan)",
  "claims": [
    {
      "step_id": 88,
      "kind": "check:measurement",
      "claim": "the recipe calls for 2 pieces; flag a different count",
      "detector_candidates": [
        {"modality": "cv", "detector": "count blobs in mug ROI", "feasibility": "low",
         "why": "HSV state-change sees an add but cannot count reliably"},
        {"modality": "vlm", "detector": "one frame, 'how many pieces?'", "feasibility": "high",
         "why": "counting discrete items is a VLM strength"}
      ],
      "sensing_role": "B-trigger",
      "control_plan": {
        "camera": {"power": "on", "fps": 1, "resolution": "full", "color": "rgb", "roi": "mug"},
        "audio": {"armed": []},
        "vlm": {"when": "trigger", "frames": 1, "resolution": "full"},
        "plan_ahead": "camera off until the post-microwave add window; wake to 1 full-res frame on hand-in-mug-ROI motion"
      },
      "open_questions": "can a cheap motion/ROI cue reliably trigger the single frame?"
    }
  ],
  "recipe_schedule": {
    "by_stage": [
      {"step_id": 89, "stage": "microwave", "camera": "off", "audio": ["D1","D6"], "vlm": "none", "role": "A-solve"}
    ],
    "role_mix": {"A-solve": 0, "B-trigger": 0, "C-none": 0},
    "cost_estimate": "<X>% of always-on-VLM (frames captured / VLM calls vs a 1 fps always-on baseline); state the assumptions",
    "honest_limits": "the C-none claims and any low-feasibility detector bets, named"
  }
}
```

The `recipe_schedule` is the payoff: a per-stage sensor plan and a **headline cost** phrased as
"sensing at X% of always-on-VLM energy/latency at equal coverage," never "audio solves cooking."
Report the **role mix per claim subtype**, never pooled (most checks are visual-leaning;
pooling flatters the audio/CV arm).

---

## 5. Firewall note

This step is **firewall-neutral**: it reasons about *detector feasibility*, not ground truth.
It reads only the criteria + recipe + the public detector catalogs — never CC4D error tags,
Qualcomm timestamps, or per-recording traces. (The criteria it consumes may carry
`[probe-added]` checks; that relaxation already happened upstream and is out of scope here.)

---

## 6. Scope

- **Brainstorm, not commitment.** Output is research options + a proposed schedule, for human
  review. Stage 2 later picks one binding per claim and writes the executable sensorplan.
- **21 recipes** (loop recipes sautedmushrooms / dressedupmeatballs / pinwheels skipped until
  the runtime handles repeated step_ids).
- One recipe per run; loop for the corpus.
