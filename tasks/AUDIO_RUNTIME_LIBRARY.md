# Audio Detector Library for Online Procedural Step Recognition (v1)

A closed, self-contained set of cheap audio detectors that a procedure monitor binds, per
recipe, to recognize cooking-step transitions online. Every detector listed is validated on real
CaptainCook4D (CC4D) recordings and commits within a decision window ≤ 10 s — the requirement for
online step recognition, where a long smoothing window cannot localize a step boundary. Detectors
that fail either test are listed under *Tested and excluded* with the reason, so the boundary of
the library is explicit.

**Sensing roles.** Each detector binds to a step in one of two roles. **A-solve:** the cheap
detector settles the step alone, no vision-language model (VLM) call. **B-trigger:** the cheap
detector flags the moment and fires exactly one targeted VLM call. Steps with no audio detector
(**C-none**) fall to vision / VLM. The library's value is energy and latency: it spends the VLM
only where audio cannot decide, and keeps the expensive sensor off otherwise.

---

## Streaming model — one fixed window

At runtime the monitor ingests a **continuous RGB + audio stream**. There is a single fixed
processing window for the whole system; the per-detector "latency" values below are *not*
separate ingestion clips — they are how long each detector waits before it commits, on the one
shared stream.

- **Fixed decision tick:** the monitor updates all detector states and may emit events once per
  **1 s**.
- **Bounded look-ahead buffer ≤ 10 s.** A decision attributed to time *t* may use audio up to
  *t* + *L*, where *L* is the detector's latency below and **L ≤ 10 s for every detector** — that
  bound *is* the inclusion rule, and 10 s is the maximum audio the wearable must hold.
- A detector may also reference a long *past* history as a cheap rolling statistic; that is
  history, not look-ahead, so it does not enlarge the buffer or the latency.

So a `complete_when` rule may use evidence up to the current time plus the bound detector's
latency, and no further.

---

## Bindable detectors

### D1 · `microwave_cycle` — appliance run start + authoritative end
- **Detects:** a microwave turning on, running, and ending (a sustained hum onset plus an
  authoritative end-beep; the beep gives the offset).
- **Role:** A-solve. **Gate:** recipe contains a microwave step.
- **Emits:** `cycle_start`, `cycle_end`, `duration`.
- **Latency:** `cycle_start` ~8 s (a run is only confirmed once it has lasted the
  `min_run_s`=8 s minimum; the smoothing filters add only ~4 s), `cycle_end` ~8 s (the
  end-beep association window). Both within the ≤10 s buffer.

### D2 · `appliance_motor` — blender / grinder on/off
- **Detects:** a loud sustained motor (blender, grinder) turning on and off.
- **Role:** A-solve, **gated** — ungated it false-fires on other loud events.
- **Gate:** recipe contains a blender/grinder step; expect one motor event near it.
- **Emits:** `motor_on`, `motor_off`.
- **Latency:** ~5–7 s.

### D3 · `cook_end` — end of frying / sauté
- **Detects:** the moment active frying / sauté / boil stops.
- **Role:** A-solve, **END only** — not start, not coverage.
- **Gate:** recipe contains a fry/sauté/boil stage.
- **Emits:** `cook_end`.
- **Latency:** ~2 s.

### D4 · `cook_start` — onset of frying (trigger)
- **Detects:** the food-hits-hot-pan sizzle that begins frying.
- **Role:** B-trigger — frying ramps in with no sharp acoustic onset, so this flags a candidate
  and fires one VLM call to confirm cook start; missed onsets fall to the VLM. Never an A-solve.
- **Gate:** recipe contains a fry stage.
- **Emits:** `cook_start_candidate`.
- **Latency:** ~5–8 s.

### D5 · `water_flow` — tap on/off
- **Detects:** running tap (rinse / wash / fill) on and off. Shares D3's learned tagger, so it is
  effectively free wherever D3 already runs.
- **Role:** B-trigger / gated A-solve — recipes have several water uses, so bind to one specific
  step rather than treating every tap event as that step.
- **Gate:** bind to the specific rinse/wash/fill step.
- **Emits:** `water_on`, `water_off`.
- **Latency:** ~2 s.

### D6 · `timer` — duration & precondition logic
- **Detects:** over/under-time on timed steps and precondition (recipe-graph edge) violations,
  using procedure-graph state and stated durations, anchored on a completion event from D1–D5.
- **Role:** A-solve (pure logic, no latency).
- **Gate:** timed steps / precondition edges.
- **Emits:** `overtime`, `undertime`, `precondition_violation`.

### VLM · `VLM` — the expensive sensor (B-trigger target / C-none fallback)
- **Detects:** anything audio cannot — current step, step done/not-done, fine
  technique, ingredient identity, quantity (see *What audio cannot do*).
- **Role:** invoked by the monitor, **not** a detector. A **B-trigger** step fires
  exactly one VLM call at the flagged audio event; a **C-none** step calls it on a
  periodic schedule. It is the only expensive sensor — the whole schedule exists to
  keep it asleep otherwise.
- **Gate:** a step with no A-solve detector, or a B-trigger candidate that needs
  confirmation.
- **Emits:** a step verdict consumed by the monitor.

---

## Binding rules

1. Bind a detector to a step **only if** the step's transition matches the detector's event
   **and** the gate holds (the recipe actually contains that appliance / stage / timed value).
2. **A-solve** detectors settle the step with no VLM. **B-trigger** detectors fire one VLM call
   at the flagged moment. Steps with no matching detector are **C-none** → vision (pre/post
   appearance change for silent add/place) or periodic VLM.
3. Every detector emits within the fixed ≤ 10 s buffer on the 1 s tick, so any schedule built
   from this library is valid for online step recognition. Do not bind anything from *Tested and
   excluded*.

## Tested and excluded (with the reason)

| Detector | Why excluded |
|---|---|
| Sustained-cook coverage gate (45 s window) | window ≫ 10 s; tells you "cooking is ongoing" but cannot localize a start or end. Use D3 (end) and D4 (start) instead. |
| Kettle boil | gentle broadband ramp with no edge to localize → VLM. |
| Transient train (chop / stir-clink) | material-blind and false-fires; usable only as weak corroboration inside an already-believed stage, never as a step event. |
| Pour | too many false alarms; never reliable standalone. |
| DSP water level gate | superseded by D5 (the learned water classes are cleaner). |
| Packet rustle, spatula scrape | not validated → not bindable. |
| General-purpose audio taggers as a continuous classifier | require a GPU; not always-on within a wearable power budget. Offline cross-check only. |

## What audio cannot do (C-none → vision / VLM)

- **Silent transfers** — add / sprinkle / place / spread / measure / coat: **~48% of CC4D steps**
  are structurally silent.
- **Gentle or dry cooking** — e.g. low-heat pancakes, toast: inaudible to both DSP and the
  learned tagger.
- **Fine technique, ingredient identity, quantity** — "stir until no lumps", "skimmed milk",
  "2 tbsp": no acoustic correlate.

## Code index

Each detector is one module under `detectors/runtime/` (registry in
`detectors/runtime/__init__.py`). Detectors emit role-agnostic events; the monitor
decides A-solve / B-trigger / C-none. Shared front-ends: `audio_io.py` (16/48 kHz
loading), `panns.py` (CNN14 for D3/D5), `vlm.py` (the stubbed expensive sensor).

| ID | Module | Class | Input | Emits |
|---|---|---|---|---|
| D1 | `detectors/runtime/d1_microwave_cycle.py` | `MicrowaveCycleDetector` | 16 kHz | `cycle_start`, `cycle_end` |
| D2 | `detectors/runtime/d2_appliance_motor.py` | `ApplianceMotorDetector` | 16 kHz | `motor_on`, `motor_off` |
| D3 | `detectors/runtime/d3_cook_end.py` | `CookEndDetector` | 48 kHz → PANNs | `cook_end` |
| D4 | `detectors/runtime/d4_cook_start.py` | `CookStartDetector` | 16 kHz | `cook_start_candidate` |
| D5 | `detectors/runtime/d5_water_flow.py` | `WaterFlowDetector` | 48 kHz → PANNs | `water_on`, `water_off` |
| D6 | `detectors/runtime/d6_timer.py` | `TimerChecker` | graph state | `overtime`, `undertime`, `precondition_violation` |
| VLM | `detectors/runtime/vlm.py` | `VLMClient` | frames (stub) | step verdict |
