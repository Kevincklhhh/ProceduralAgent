# Audio Detector Library v2 (EPIC-SOUNDS-grounded)

> 🧭 **Role (2026-06-15): this is the audio-primitive DESIGN-RATIONALE doc** (why each primitive exists, grounded in EPIC-SOUNDS). The canonical status/tier table is `DETECTOR_CATALOG.md`; the primitive table below is the rationale behind it. If they disagree, the catalog wins.

Revision of the v1 library (DETECTOR_FEASIBILITY.md §6) after reading EPIC-SOUNDS (`related_work/epic_sounds_2302.00646.pdf`): 78.4k human-annotated audio segments over the 100 h of EPIC-KITCHENS-100, 44 classes derived purely from what annotators could hear. It is the only empirical census of *which kitchen actions actually sound*, and it both validates and corrects our v1 design.

## Design rules imported from EPIC-SOUNDS findings

1. **Material discrimination is abandoned.** 24 of the 44 ES classes are material-pair collisions (metal/wood, plastic-only, ...), but humans verify materials from audio in only 48.8% of cases (cloth 8%, glass→metal confusion 42%), and models over-predict metal-only. → Our transient detectors stay **material-blind** (at most ringy vs dull); the *graph stage assigns meaning* (transient train during `mix` = stirring; during prep = chopping). This is the conditioning argument, now with citable evidence.
2. **Trains, not single hits.** ES measures repeated audio events per visual action: cut/chop 2.28-to-1, beep 1.47-to-1, metal/wood 1.24-to-1 ("stop-start" pattern). → train statistics (rate, length) are the right feature, confirming `transient_train`.
3. **Completion events are moments, not spans.** ES documents systematic audio/visual boundary misalignment ('close bin': visual onset = grasp, audio onset = lid slam). → we keep using audio onsets/offsets as *completion instants*, scored with ±15 s tolerance.
4. **Audio hears off-view events.** Top classes with NO visual overlap: footstep 28.5%, click 21.1% — sounds from actions outside the camera view. (Our microwave-hum-while-user-walks-away exploit is an instance.)
5. **Quiet steps are structurally quiet.** Most-silent visual actions: insert mixture 84%, fold dough 50%, hang cloth 41% — confirms our 48% quiet residue (add/sprinkle/place/spread) is a property of the actions, not of our detectors.
6. **Per-room background calibration is mandatory.** ES annotators tag persistent `background` (fridge/fan/radio/washing machine) as a first-class phenomenon; our 8_25/CLAP cross-room threshold failures were exactly uncalibrated background. → new primitive 0.
7. **Generic audio tagging is hard — conditioned detection is the easier problem we actually have.** Audio-only SOTA on ES recognition is ~56% top-1 / ~26 mCA over 44 classes (humans: 20.8%!). We never need 44-way open classification: the graph asks *binary, stage-conditioned* questions ("is a transient train present, given we expect mixing?") with strong priors.

## The primitives (v2)

The canonical primitive list — id, tier, status, cost, **and** the EPIC-SOUNDS class each grounds to — is **`DETECTOR_CATALOG.md`** (audio rows A0–A10). This doc owns only the *why*: the EPIC-SOUNDS design rules above, the recognizability-tier analysis below, and the dropped-from-v1 rationale here.

Dropped from v1 (per design rule 1): material-specific collision discrimination; `impulse_single` as a standalone detector (single hits are unreliable per ES — they count only as weak evidence inside a stage that expects them); `friction_flatness` folded into `rustle` + `scrub_scrape`.

## Refined CC4D audio-feasible events

The CC4D step-mention taxonomy (358 steps), now expressed in ES vocabulary with recognizability tiers:

| Tier | Meaning | CC4D event classes (step mentions) |
|---|---|---|
| **A — reliably detectable, even context-free** | ES top-recognizable classes + our validated probes | microwave/appliance cycles (39), water/tap (7), sizzle+boil (19), beep/click, rustle/packet (≥1, undercounted in step text), kettle (3) |
| **B — detectable as events, semantics need stage context** | material-blind transients; ES shows class identity is unrecoverable from audio alone | chop trains (57), stir/whisk (41), pour (8), scrape, egg crack (2) |
| **C — weak/rare one-offs** | low-frequency ES tail | spray hiss (1), can opener, jar twist, snip |
| **silent — structurally inaudible** | ES silent-action evidence | add/sprinkle/place/measure/spread (~171 steps, 48%) → RGB/VLM tiers |

Net effect of the refinement: the *count* of audio-feasible steps doesn't change (~52% mention-level / ~35% judged-primary), but Tier B — the majority of audio-relevant steps — is now explicitly **conditional**: those detections are only meaningful through graph-state conditioning, which is the system's thesis, and ES gives the citation for why (material/semantic ambiguity is irreducible in audio alone).

## New capability: large-scale primitive calibration *before* CC4D replay

ES (and HD-EPIC-Sounds, locally at `~/NeuroTrace/kitchen/HDEPIC/.../HD_EPIC_Sounds.csv`) provide thousands of labeled real-kitchen segments per class. Each primitive can now be tuned/validated against them instead of a single CC4D recording (the v1 tune-on-8_16 weakness):

- `water_running`, `tonal_burst(beep)`, `texture_dynamics(sizzle/boil)`, `rustle`, `scrub_scrape`, `transient_train(cut/chop, stir)` → direct ES class segments (sound recognition split) for threshold setting + per-class ROC.
- `background_profiler` → ES `background` segments across 45 kitchens = a background diversity corpus.
- Microwave hum has no dedicated ES class (subsumed under kettle/appliance/background) — our CC4D probe remains its validation.

Protocol amendment for v2 scaling: primitive thresholds are calibrated on ES/HD-EPIC-Sounds (foreign kitchens), then applied to CC4D **without CC4D tuning** — a strictly stronger generalization claim than tune-on-one-recording, and it retires the design-leakage disclosures.
