# Detector Catalog — full annotated primitive catalog (RESEARCH / evidence)

> 🧭 **RESEARCH DOC — not the runtime spec.** The runtime/compiler binds the filtered,
> ≤10 s-window, reliability-gated set in **`../../tasks/AUDIO_RUNTIME_LIBRARY.md`** (the single source
> of truth for what the system executes). This catalog keeps the *full* primitive history,
> including detectors that were tested and **excluded** from runtime (A3, A6, A4-as-coverage,
> kettle boil) and dated findings (v1–v12). For deciding what the system runs, use the runtime
> library; use this only for provenance.

This table is the **one authoritative list** of detector primitives, their tier, validation
status, and measured cost. It absorbed the probe-evidence and EPIC-SOUNDS design-rationale
docs (formerly `DETECTOR_FEASIBILITY.md` / `AUDIO_LIBRARY.md`, now removed) and the Box-3
predictor docs (`../../tasks/PROCEDURE_MONITOR_COMPILER.md`). **If anything disagrees with this
table, this table wins.**

Status vocabulary: **validated** = probed on real CC4D recordings; **weak** = works but
only as gated/corroborating evidence; **untested** = catalog-plausible, binding it is a
hypothesis; **failed** = probed and does not work as-is; **required-new** = needed before
scaling, not yet built. **Tier = energy budget, NOT implementation (re-cut 2026-06-15):**
**always-on** = fits the glasses mW envelope and runs continuously on-device — classical DSP
on one CPU core **plus** small NPU-quantized audio taggers (CNN14/YAMNet int8); **triggered**
= needs a GPU/accelerator, so duty-cycled or server-side, never in the always-on bank (AST,
CLAP, all RGB); **escalation** = VLM, one targeted call. This supersedes the old
cheap/medium/expensive labels, which conflated DSP-vs-neural implementation with energy
budget: AST is cheap on a server GPU yet cannot run always-on on the glasses, while a
quantized CNN14/YAMNet can. All always-on audio DSP shares one 16 kHz STFT front-end (≪0.1% of
one CPU core).

## Audio primitives (always-on on-device; A-ids = compiler vocab, #=legacy v1 id)

> The AST/CLAP rows at the bottom are **triggered** (GPU-required), not always-on — they live
> here only because they are audio. A3 and A6 are **DROPPED** (not armable standalone).

| id | # | Primitive | Tier | Status | CC4D events | Cost / evidence |
|---|---|---|---|---|---|---|
| A0 | 0 | `background_profiler` — per-recording noise floor + persistent-source profile | always-on | required-new | cross-room transfer (8_25 failure) | DSP; gates every other threshold |
| A1 | 1 | `sustained_band(profile)` | always-on | **validated, generalizes** (microwave **27/29 = 0.93 recall over all 16 SHC**; 2 misses 8_33/8_50; 6 fragmentation false-runs on 8_19/8_20). **MUST be recipe-gated**: fires 47 false hum runs over the 16 act-23 stir-fry recordings (no microwave) on stove/fan broadband — trust only when the recipe has the appliance. **Short-cycle RETUNE (2026-06-15): min_run_s 20→8, smoothing 5.4 s→1.8 s.** Originally hum-alone collapsed on short runs (Microwave Egg Sandwich act 1: 0.53 / 0.53 / **0.31** on 30 s / 15-30 s / ~10 s cycles). After retune hum-alone = **0.80 / 0.94 / 0.88**, with **NO SHC regression (still 0.93/0.93)** and false hum runs flat (5/18 act1, 5/16 SHC; SHC even improved 6→5). Bonus: lookahead 7.2 s→3.9 s (more reactive). **VALIDATED across ALL 10 microwave recipes (156 recs, 250 cycles): fused hum∨beep recall 0.99** (baseline 0.97). False runs rose 0.67→1.01/rec, BUT **truly-spurious runs (no microwave in the recording) stayed at exactly 5/156 ≈ 0.03/rec — the optimization added 0 new spurious detections**; all +54 false runs are FRAGMENTS of long runs / secondary sustained sounds *within correct recordings* (Meatballs 2.9, Mug Cake 2.0, Ramen 1.5 false/rec). Harmless for step recall; **consolidate hum runs at the stage level before any DURATION/timing check.** Raising merge_gap backfires (resurrects sub-floor noise). Evidence: `results_all_microwave.json`, `compare_microwave_baseline_opt.py`. **Other appliances tested (`appliance_water_eval.py`): blender/grinder on/off recall 0.93–1.00 = A-solve, recipe-gated (ungated 3–7 runs/rec); kettle BOIL weak (0.80, +53 s onset → B-trigger, gentle broadband ramp).** | microwave hum, blender, kettle, extractor | DSP ~490× RT |
| A2 | 2 | `tonal_burst` | always-on | **validated, generalizes** (end-beep **27/29 = 0.93** at microwave/heat ends over 16 SHC). **On a 2nd recipe (Microwave Egg Sandwich act 1) the beep covers short cycles**: beep 0.93 / 0.76 / 0.69. After the A1 short-cycle retune, **hum∨beep FUSION = 1.00 / 1.00 / 1.00 (perfect over all 48 cycles)**, and 1.00 on SHC too — improved hum + beep catch complementary cycles. Confirms A1+A2 must be fused. | end/keypad beeps, igniter clicks, timer dings | DSP |
| A3 | 3 | `transient_train` — material-blind | **DROPPED** | **not armable standalone** (2026-06-15): 0.64 recall, 17 false strong-clinks over 16 SHC (`probe_audio_anchors.py`); material-blind. Keep ONLY as conditioned corroboration inside an already-believed stage — never a step recognizer or completion event. | ~~chop/stir/whisk~~ | DSP |
| A4 | 4 | `texture_dynamics(band)` — sustained-level (rolling-median) sizzle/fry | always-on | **tested over 9 stovetop recipes / 137 recs (frozen 23_5 params); reliability tracks acoustic prep/cook CONTRAST** — see `STOVETOP_AUDIO_SURVEY.md`. **A-solve (hum-grade)**: Pan Fried Tofu recall 1.0 / false 0.04 (quiet prep, loud fry). **B-trigger**: Stir Fry 0.88/0.19, Sautéed Mushrooms 0.86/0.29, Scrambled Eggs 0.81/0.18, Zoodles 0.80/0.28 (loud fry but noisy prep). **none**: Caprese Bruschetta 0.38 (dry toast), Banana Pancakes 0.00 (NEGATIVE contrast — gentle fry quieter than prep). Governing law: cook must be louder+more-sustained than prep; A-solve vs B-trigger is NOT microwave-vs-stovetop. (Persistence gate 3-14× better false-prep than naive spectral v1.) | sizzle/fry/deglaze, boil rumble | DSP (CPU); on-device prefer **AL** (NPU tagger), which front-runs this at far lower prep false-alarm; gates all stovetop recipes |
| A5 | 5 | `water_running` | always-on | **tested** (`appliance_water_eval.py`): tap rinse DETECTABLE (recall 0.76–1.00 over 4 recipes) BUT **low precision — 9–14 runs/rec** (level fires on all broadband: sizzle, handling). → needs the **edge-pair on/off** (sharp >6 dB edges) or recipe-gating; level-only is NOT a clean A-solve. **CNN beats DSP here (`water_cnn_eval.py`, AudioSet 288/370/371/444 water classes): ~HALVES false fires at equal/better recall** (Mushrooms 12.9→4.7 runs/rec, Raita 13.6→5.7, recall 1.00) — and it's **FREE if AL already runs** (same CNN14 forward, different class indices). Still 2–7 runs/rec (recipes have several water uses) → recipe-gating still needed for clean localization, but CNN+gating = usable. | tap rinse/wash, faucet fill, drain | DSP or **AL water-classes** |
| A6 | 6 | `pour` | **DROPPED** | **not armable** (2026-06-15): 0.5–4 FA/min and ~15.5 s lookahead — can never gate a decision; logged secondary signal only, never a completion event or trigger. | ~~liquid pour~~ | DSP |
| A7 | 7 | `rustle` | always-on | untested (promising) | packet/wrapper tear & crinkle | DSP |
| A8 | 8 | `scrub_scrape` | always-on | untested | spatula/tong scrape, scrub | DSP |
| A9 | 9 | context cues: `footstep` / `open-close` / `human` | always-on | optional layer | walk-away (reminder gating), idle, speech | DSP |
| A10 | 10 | `timer + graph logic` | always-on | **validated** | all duration checks, precondition violations | logic |
| AL | — | **`learned_cook_tagger`** — PANNs CNN14 / YAMNet (AudioSet Frying+Sizzle+Boiling+Steam, pooled) | **always-on (on-device)** | **stress-tested vs A4 over ALL 9 stovetop recipes / ~135 recs** (`stress_stovetop_a4_al.py`, coverage≥0.40 metric, AL with 20 s gap-bridging to match A4's persistence): **COMPLEMENTARY, not a drop-in replacement** (refutes the earlier "bind AL not A4"). On loud fries AL ≈ A4 recall at ~½ the prep false-alarm (Stir Fry A4 0.88/0.19 vs AL 0.81/**0.07**; Mushrooms 0.86/0.29 vs 0.79/**0.13**; **Scrambled Eggs AL 0.88/0.12 BEATS A4 0.81/0.18 on both**). A4 wins raw recall (Tofu 1.00 vs 0.80) and SIMMER (Tomato Chutney 0.60 vs 0.27 — CNN doesn't score simmer as frying). Neither rescues gentle/dry cooking (Pancakes ~0, Caprese ~0). AL also needs gap-bridging (raw per-frame coverage is sparse) + detects sizzle-END ~40-46 s. **FUSION measured (`fusion_stovetop.py`, `results_fusion_stovetop.json`): the deployable point is `A4_gated_by_AL` (keep each A4 run only where an AL run supports it) — recall ≈ A4 (mean 0.64 vs 0.67) at false-prep −37% (0.19→0.12), and PRESERVES simmer (Chutney 0.60) which plain intersection destroys (0.07). Per noisy-prep recipe: Stir Fry false 0.19→0.11, Mushrooms 0.29→0.22, Eggs 0.18→0.14. Union maxes recall (0.74) but raises false; intersection mins false (0.06) but kills omelet/simmer recall.** | **A4_gated_by_AL is the recommended stovetop cook detector** (A4 coverage, AL vetoes prep false-alarms); pure complement otherwise. sizzle-end / add-surge transitions | CNN14 80 M → **~80 MB int8 on Hexagon NPU, ~10 ms/win, mW-class, always-on-capable** (8 Elite Gen 5); YAMNet 3.7 M lighter. NOT GPU-only. |
| — | — | `AST` (AudioSet-527 tagger) | **triggered** (GPU) | **validated** for microwave (pooled AUC 0.868); NO for pour/add | server/offline cross-check + pseudo-label oracle — DSP A1 already beats it on microwave (0.93 recall vs AUC 0.868); not always-on on-device (86.6M, GPU) | 16–18 ms/win GPU, 745 ms CPU |
| — | — | `CLAP` (open-vocab text prompts) | **triggered** (GPU) | **validated** as zero-shot segment labeler; weak on quiet transients | disambiguation, per-env calibration; not always-on on-device | ~9 ms/win GPU |

Dropped from v1 (per EPIC-SOUNDS material-ambiguity rule): material-specific collision
discrimination; `impulse_single` standalone; `friction_flatness` (folded into A7/A8).

## RGB primitives (TRIGGERED tier — GPU-required, never always-on; measured on RTX 6000 Ada)

| id | Primitive | Status | Cost | Role |
|---|---|---|---|---|
| R1 | OWLv2 / GroundingDINO open-vocab grounding | validated (cost) | 31.5 ms/frame GPU | ROI acquisition ONCE per stage on sharpest frame |
| R2 | MOSSE / template / CSRT tracker ladder | validated (cost) | 0.3 / 0.4 / 15 ms | per-frame ROI track + reacquire (needs ego-shift compensation) |
| R3 | skin+motion hand gate (YCrCb + diff) | **efficacy FAILED** on egocentric SHC (skin-colored surfaces → no discrimination); cost ok | 0.9 ms @480p | hand-presence gate — needs a real hand detector, not skin color |
| R4 | HSV-hist shift / liquid-level / steam variance | **efficacy: detection OK, counting weak** — SHC adds recall 0.95/precision 0.42 over 16 recs (`probe_rgb_adds.py`); over-segments w/o a gate | 0.05–0.25 ms/ROI | pre/post hand-episode state-change = rgb_transfer; coarse "add happened", not fine count |
| R5 | EasyOCR display reader | validated (py3.13) | 11 ms/crop GPU | microwave/timer digit read |
| — | global-frame motion periodicity (stir) | **failed** (AUC ≈ chance) | — | dead — any stir vision must be ROI-restricted ≥10 fps bursts |

## VLM tier (expensive; trigger-driven, never duty-cycled)

| id | Primitive | Cost | Role |
|---|---|---|---|
| V1 | Qwen2.5-Omni-7B (audio+RGB+text) / Qwen2.5-VL | 1–3 s/query GPU | on-demand fusion confirmation at escalation only |

## Notes that bind on Box 3 (the predictor)

- The predictor may bind only primitives in this table and must respect status (binding an
  `untested` primitive is a hypothesis, not a capability).
- **A3 and A6 are DROPPED (2026-06-15) — not armable standalone.** A3 may appear only as
  conditioned corroboration inside an already-believed stage; A6 only as a logged secondary
  signal. Neither is ever a completion event, trigger, or step recognizer.
- **Only one learned model belongs in the always-on bank: AL** (CNN14/YAMNet int8 on NPU),
  the cook detector — bind it instead of DSP A4 on-device. **AST/CLAP and all RGB are
  `triggered`** (GPU-required), never always-on; the energy claim is measured against the
  always-on bank only.
- The ±33% timing tolerance is a **detector parameter** here, never ground truth (firewall;
  see `PIPELINE_THREE_BOXES.md`).
