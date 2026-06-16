# Detector Feasibility: Tasks, Detectors, Probe Results

> 🧭 **Role (2026-06-15): this is the detector PROBE-EVIDENCE doc.** The canonical primitive list (id/tier/status/cost) now lives in `DETECTOR_CATALOG.md`; §3 below is retained as measured-cost detail and is the *evidence* behind the catalog's status flags. The recipe-selection logic in §1 ("spiced hot chocolate stays primary", "Top 5 candidates") is **historical** — Box 3 (`SENSOR_GRAPH_COMPILER_PROMPT.md`) now compiles all 24 recipes; §1 is kept as the detector-friendliness census. Design-leakage disclosure in §4.1 is superseded by the EPIC-SOUNDS calibration plan in `AUDIO_LIBRARY.md` §3 (tune on foreign kitchens, not on eval recordings).

Single reference for "what tasks exist, what RGB/audio detectors can we use, do they actually work".
All MEASURED numbers were produced on this machine (py3.13.5, torch 2.7.1+cu118, transformers 4.57.1, opencv 4.13, RTX 6000 Ada, AMD EPYC 9354). Probe numbers come from the six local CaptainCook4D spiced-hot-chocolate recordings (activity 8: 8_16 tuning, 8_26 / 8_3 / 8_25 / 8_31 / 8_50 frozen eval) scored against `gt_activity8.json`.

**Bottom line:** a pure-DSP hum+beep detector (~490x real-time on one CPU core) found 11/12 GT microwave runs with ZERO false runs and flagged both headline timing errors; AST and CLAP confirm microwave state at AUC 0.87-0.98 for ~3 s of GPU per recording; the one clear failure is RGB global-motion periodicity for stirring (AUC ~= chance). Audio is the cheap tier that works today; vision needs ROI-restricted rework.

---

## 1. Task inventory: 24 CaptainCook4D recipes ranked by detector-friendliness

Tier % = fraction of steps whose primary detector is cheap (always-on DSP/classical), medium (RGB ROI / tagger), or VLM-mandatory. All 24 recipe DAGs have some parallelism, so the Par column is uniformly yes; differences are in anchor density, not graph shape.

| # | Recipe | Steps | Cheap% | Med% | VLM% | Par | Audio anchor events | Verdict |
|---|--------|-------|--------|------|------|-----|---------------------|---------|
| 1 | **spicedhotchocolate** | 7 | 57 | 43 | 0 | y | microwave hum, end beep, door clunk, milk pour, spoon-in-mug stir clinks | Gold-standard cheap testbed: two microwave cycles, a pour, and spoon clinks confirm 4/7 steps from audio alone; zero VLM-mandatory steps. |
| 2 | **panfriedtofu** | 19 | 53 | 47 | 0 | y | igniter click, sizzle bursts (soy/sesame), sustained frying sizzle, sizzle offset/ramp, tong scrape (weak) | Best stovetop testbed: sizzle band-energy dynamics track every heat-state transition (on/off/ramp), timers cover the cooks, zero VLM. |
| 3 | **sautedmushrooms** | 18 | 56 | 39 | 6 | y | tap rinse, knife chop/mince, igniter click, sizzle onset, saute sizzle, deglaze burst | Highest cheap-tier coverage (56%); board chops + faucet + sizzle staging; only garlic peeling escalates to VLM. |
| 4 | **scrambledeggs** | 23 | 52 | 43 | 4 | y | chop, whisk, egg crack, sizzle onset/surge, lid clink, igniter clicks | Chop/crack/whisk/sizzle anchor >half the steps cheaply, timers cover cook phases; only garlic peeling needs VLM. |
| 5 | **microwaveeggsandwich** | 12 | 42 | 58 | 0 | y | egg crack, spray hiss, microwave hum x3, end beep, fork-on-ramekin clinks | Three microwave cycles + spray hiss + egg crack at 42% always-on, zero VLM-primary steps; assembly = clean RGB ROI events. Best pure RGB+audio fit. |
| 6 | ramen | 15 | 47 | 47 | 7 | y | microwave hum/beep, packet crinkle-tear, dry noodle crack, chop, water pour/fill | Dense, varied cheap anchors interleaved with easy RGB transfers - near-ideal demo of the tiered sensing graph. |
| 7 | coffee | 16 | 50 | 38 | 13 | y | kettle boil, grinder, pour (bloom+main), bean rattle, tap | Grinder and kettle are unmistakable cheap anchors, highly parallel DAG; weigh + thermometer steps force VLM. |
| 8 | tomatochutney | 19 | 42 | 53 | 5 | y | blender, mustard-seed crackle, sizzle surge (puree), saute sizzle, chop, igniter | Blender + iconic mustard-seed crackle gate + sizzle surges + two timers; only garlic peeling needs VLM. |
| 9 | blenderbananapancakes | 14 | 36 | 64 | 0 | y | blender roar, egg crack, batter-pour sizzle onset, cooking sizzle | Blender/egg-crack/sizzle anchor every phase; rest are easy pan/plate ROI transfers, zero VLM. |
| 10 | buttercorncup | 12 | 33 | 67 | 0 | y | microwave hum/beep/door, tap, frozen-kernel rattle pour | Two microwave runs + faucet thaw + kernel rattle = 1/3 of steps cheap; rest single-bowl ROI with VLM amount riders. |
| 11 | broccolistirfry | 25 | 36 | 60 | 4 | y | stir-fry sizzle, sauce-pour surge, whisk clink, mincing tap train, pour/faucet | Best stress test: 25-step heavily parallel DAG, sizzle-gated timers, 9 cheap anchors; only garlic peel truly VLM. |
| 12 | dressedupmeatballs | 16 | 38 | 56 | 6 | y | microwave hum x3, end beep, chop train (onion/carrot/garlic), stir clinks | Three microwave cycles + chop trains anchor the timeline; prep/topping exercises medium RGB; only garlic-peel VLM. |
| 13 | zoodles | 13 | 38 | 54 | 8 | y | sizzle surge (zoodles), garlic-butter sizzle onset, igniter | Sizzle onsets + three timers cover the cook phase; spiralizing = clean periodic-motion target; garlic peel VLM. |
| 14 | microwavefrenchtoast | 11 | 36 | 55 | 9 | y | microwave hum x2, end beep, fork-in-mug whisk clatter, cutlery scrape | Two microwave cycles + in-mug whisk clatter; deep-mug occlusion stresses RGB transfer/stir; vanilla-add VLM. |
| 15 | herbomeletwithfriedtomatoes | 15 | 33 | 60 | 7 | y | egg crack, chop train, sizzle (fry + egg pour), fork beating, igniter | Crisp cheap cues, but doneness steps ride on subjective RGB state-change judgment. |
| 16 | breakfastburritos | 11 | 27 | 73 | 0 | y | microwave hum/beep/door, egg crack, whisk clink | Cheap anchors at key stage gates but mostly quiet bowl/tortilla transfers; roll step RGB-weak. |
| 17 | cheesepimiento | 11 | 27 | 73 | 0 | y | microwave hum/beep/door, chop tap train | Two microwave runs + audible fine chopping; bowl transfers simple ROI but lean on VLM for spice/amount identity. |
| 18 | cucumberraita | 11 | 27 | 64 | 9 | y | tap rinse, chop train, whisk clatter, stir clinks | Only 3 audio anchors; four near-identical spice adds create identity confusion that stresses VLM escalation. |
| 19 | mugcake | 20 | 20 | 75 | 5 | y | microwave hum/beep, whisking scrape, scissor snip (weak) | Microwave anchor + two whisk events + cool-down timer punctuate a long RGB measure/add chain; one honest VLM (bag seal). |
| 20 | tomatomozzarellasalad | 9 | 22 | 78 | 0 | y | tap rinse, knife slice | No-cook: 7/9 steps are near-identical silent sprinkle/drizzle gestures over one platter ROI. Hard RGB-discrimination benchmark, poor audio testbed. |
| 21 | spicytunaavocadowraps | 17 | 18 | 71 | 12 | y | can-opener ratchet/tab pop, scallion chop, drain trickle | Mostly silent cold assembly; good vision-ladder stress test, weak audio; roll/toothpick force VLM. |
| 22 | capresebruschetta | 11 | 18 | 82 | 0 | y | bread-knife sawing, chop taps | Weakest audio of the practical set: no appliances; 82% medium-tier RGB - stresses escalation budget. |
| 23 | microwavemugpizza | 14 | 7 | 93 | 0 | y | microwave hum/beep/door (end only), stir clinks (weak) | Single late microwave anchor; otherwise a pure mug-ROI RGB-transfer gauntlet. Medium-tier stress test, zero VLM. |
| 24 | pinwheels | 19 | 0 | 84 | 16 | y | jar-lid twist (weak), knife scrape in jar (weak) | Worst fit: near-silent fine manipulation, zero cheap anchors, three floss steps force VLM. Adversarial RGB-only stress test only. |

**Top 5 candidates:** spicedhotchocolate, panfriedtofu, sautedmushrooms, scrambledeggs, microwaveeggsandwich.
- Rows 1+5 (microwave recipes) reuse the already-validated hum+beep DSP stack verbatim; rows 2-4 (stovetop) extend it with one new detector family (sizzle band-energy dynamics) that the catalog says is cheap and robust near the pan.
- **Spiced hot chocolate stays primary** because: highest cheap% (57) at the smallest step count (7); 0% VLM-mandatory; its two microwave cycles are exactly what the only fully-validated detector (hum+beep, Sec. 4.1) measures; six GT recordings incl. scripted Timing Errors are already on disk and probed end-to-end; and every probe artifact (`gt_activity8.json`, results_*.json, tuned thresholds frozen on 8_16) targets it. Switching recipes now would discard the calibration without buying detector coverage we have not yet proven.

---

## 2. Cross-dataset audio availability (who can host our RGB+audio replay eval)

| Dataset | Audio | Stage GT | Reminder GT | Fit |
|---------|-------|----------|-------------|-----|
| **CaptainCook4D** | YES. GoPro mp4s ship embedded AAC 48kHz stereo; ffprobe-verified on two local 4K downloads (`downloader/data/captain_cook_4d/gopro/resolution_4k/*.mp4`), volumedetect mean -36.0 dB / max -4.7 dB (loud, live). HoloLens mic was captured per paper but is NOT in the public download manifest - GoPro track only. 360p variant VERIFIED (2026-06-12, 8_16_360p from Box): same AAC 48kHz stereo track as 4K (16k-mono extracts sample-identical, Pearson r=1.0000, zero lag), hum+beep detector outputs identical to 4K-derived wavs; ~1.05 Mbps total (~45 GB for all 94.5 h). | Per-recipe task-graph DAG (steps+edges JSON, any topo order valid) + per-video step segments with start/end/has_errors (skipped = -1.0) + 8 error-type tags incl. Timing Error. | None direct; error segments give mistake-time labels, ProMQA repackages into timestamped QA. Reminder windows must be derived (already planned). | **high** - chosen dataset. Open download, scripted errors, 24 recipes, 384 recordings. |
| **HD-EPIC** | YES, verified locally: AAC 48kHz stereo in `kitchen/HDEPIC/data/HD-EPIC/Videos/P08/*.mp4`; volumedetect mean -48.3 dB / max -12.7 dB - real but QUIET ambient Aria audio (tune detector gains). HD-EPIC-Sounds timestamped audio-event CSV also on disk. | Real recipe structure verified locally: `complete_recipes.json` (named recipes, ordered step IDs + text, per-capture temporal anchors) + per-video recipe_timestamps.csv + recipe_step_localization VQA. | None (unscripted correct executions; no error labels). Windows derivable from step boundaries. | **high** - best naturalistic complement to CC4D; ~41h, CC BY-NC, partially downloaded already. |
| EgoPER | yes - HoloLens2 mic, shipped in `{task}_other_modalities.zip` (not verified locally). | Frame-wise step labels + task-graph edge lists + error/normal per segment + active-object boxes. | Mistake-time labels only (trigger half of corrective reminder). | high, but gated: email institutional request, links expire in 2 weeks; verify audio not silent on receipt. |
| WTaG | yes (gated) - user AND instructor mic. A/V requires extra license form. | StepDetection.txt start/end/step per video; 3 recipes only. | Best real-human when-to-speak GT: y/n at 5,921 query points + intent + Mistake Correction type + helpful/annoying ratings. | high, with leakage caveat: GT reminders are AUDIBLE in the track and speech dominates environmental sound. 56 recs / ~10h. |
| HoloAssist | yes but PITCH-SHIFTED (voice privacy) - embedded in Video_pitchshift.mp4. Not verified locally. | Coarse+fine action ranges with per-action correctness (4-way incl. corrected-by-instructor). No canonical task graph. | Real timestamped instructor interventions with Conversation Purpose labels; official mistake-detection benchmarks (YETI). | high* - 166h, permissive CDLAv2, direct download; but pitch-shift may distort spectral features for beep/boil detectors, plus same dialogue-leakage problem. Verify on a sample first. |
| Ego-Exo4D | yes - Aria 7-mic spatial audio in frame-aligned MP4s (not verified locally; check per-take redaction flags). | Keystep segments + relational labels (pre_conditions, missing_steps, procedural_mistake) evaluated ONLINE (cAP) = real mistake/omission timing GT. | None direct; expert commentary is post-hoc, not live intervention. | medium (high if mistake-timing GT acceptable). Signed license + very large download. |
| Ego4D Goal-Step | partial - roughly half of Ego4D base videos have audio; must filter per-video via ego4d.json flags. | Hierarchical goal->step->substep (48k segments, 430h cooking) with essential/optional flags. | None (PARSE-Ego4D / ProAssist add synthetic layers). | medium. License + AWS; variable in-the-wild audio. |
| ProAssist release | no in release - HF repo ships pre-extracted frames + dialogue JSON only (verified via HF API tree). Audio recoverable only by re-joining timestamps to source videos (WTaG gated-yes, HoloAssist pitch-shifted, EPIC yes, Ego4D partial, Assembly101/EgoExoLearn no). | Inherits step annotations from 6 source datasets. | LLM-synthesized assistant turns w/ timestamps + intents, 30k dialogs/479h - synthetic, not human. | medium; nontrivial re-alignment engineering, 152GB. |
| EPIC-KITCHENS-100 | yes - GoPro audio; EPIC-SOUNDS (78.4k labeled audible events) is built on it. | None procedural (verb/noun segments only, no recipes/graphs/mistakes). | None. | low for replay, but uniquely valuable as a **detector calibration set** (timestamped sizzle/water/appliance labels). |
| EgoProactive / Pro2Bench | NO - audio-stripped, verified two ways (starter-kit README line 233; ffprobe on all 3 local val mp4s: single h264 stream). | Plan P=(s1..sN;c) with completed/current/next + visual cues. | Best released when-to-speak GT (9,935 decision points, silent/interrupt + golden utterance, IF1/SF1/PQS). | low for us (no audio); remains the RGB-only public challenge anchor. 0.7-3.9s micro-action granularity. |
| IndustReal | no - HL2SS modality list has no mic; paper has zero audio mentions (pdftotext-grepped). | Strongest completion-timing GT (PSR exact completion frames, state machine, errors as steps; F1+POS+delay tau). | None. | low; borrow its delay-aware PSR metrics. |
| Assembly101 | no - 8 fixed RGB + 4 ego monochrome cams, no mic anywhere; participants work silently. | 100k coarse / ~1M fine segments; no canonical graph. | Mistake annotations per segment (correct/mistake/correction + remark). | low; monochrome ego also hurts RGB. |
| EgoExoLearn | no in release (mp4+gaze+features only; audio not documented anywhere in repo/paper). | Temporal action segmentation over 8 procedural tasks. | None. | low. |

**Replay-eval shortlist:** CaptainCook4D (primary, loud audio + timing-error GT) -> HD-EPIC (naturalistic transfer, quiet audio, recipe-step GT, zero new download friction) -> EgoPER (if email-gated access lands). EPIC-SOUNDS as the audio-detector calibration corpus. WTaG/HoloAssist only with dialogue-leakage handling.

---

## 3. Detector implementation notes (method + confounds)

> The **canonical primitive list** (id / tier / status / cost) is `DETECTOR_CATALOG.md`. This section keeps only what the compact catalog omits: the DSP **method core + confounds** (§3a), the full pretrained-model **comparison** beyond the catalog's AST/CLAP rows (§3b), and the per-variant **vision ladder** the catalog summarizes as R1–R5 (§3c).

### 3a. Audio - classical DSP method + confounds (all: <<0.1% of one core on a shared 16 kHz STFT front-end, ~32 ms window / 16 ms hop; all run today on scipy/numpy)

| Primitive (catalog id) | Method core | Robustness / confounds |
|----------|-------------|------------------------|
| Microwave hum (A1) | 120/180/240 Hz mains comb score (peak/neighbor-median) AND sustained 200 Hz-2 kHz fan band; median filt ~1 s; run-length >= 8 s; ratio features only (glasses AGC). | Good (long stationary signature). Confounds: fridge compressor, range hood - confirm with door transient or end-beep. **Probed: works (Sec. 4.1).** |
| Appliance end-beep (A2) | 0.8-4 kHz peak, prominence >= 15-20 dB, low flatness, burst 0.1-1 s, then inter-burst regularity (3-5 beeps ~0.5-1 s apart). | Highest-value always-on trigger; piezo cuts through noise. Repetition test rejects phone/smoke-alarm chirps. **Probed: marks true offsets to ~1 s (Sec. 4.1).** |
| Kettle/pot boil (A1/A4) | 1-8 kHz energy vs 30-60 s rolling baseline; electric-kettle end = click transient AND >6 dB step-down within ~1 s (far more reliable than the slow ramp). | Moderate (slow drift fights AGC). Stovetop whistle: NOT an AudioSet-527 class - needs the classical tonal-burst machinery. |
| Blender/grinder (A1) | Full-band RMS step >= 15-20 dB + motor harmonic comb + >= 1 s. | Very high; also acts as a MASK flag suppressing all other audio detectors while active. |
| Sizzle/frying (A4) | E(4-8k)/E(0.5-2k) ratio + HF flatness + sustained; HF-envelope kurtosis separates crackly sizzle from stationary tap; sizzle onsets ramp, taps have hard edges. | Good near pan. Confounds: running water, rain. Key detector for the stovetop top-5 recipes. |
| Pouring liquid (A6) | 0.5-4 kHz burst 1-10 s + AM-spectrum peak 5-20 Hz (gurgle) + optional rising Helmholtz pitch glide. | Moderate - quiet; medium-confidence cue, confirm with RGB. **Probed: 6/6 recall but 0.5-4.0 FA/min ungated (Sec. 4.2).** |
| Water tap (A5) | Sustained 1-8 kHz, low frame variance (stationarity) + abrupt >6 dB edges in 100-200 ms. | Good; edge pair = clean washed-hands/filled-pot signal. |
| Chopping (A3) | Spectral-flux onsets + autocorrelation of onset train (>= 3 onsets, IOI 0.2-1 s, low jitter). | Good for rhythmic chopping; single cuts missed (VLM covers). |
| Spoon clink/stir (A3) | Onset + narrowband damped ringing 2-8 kHz (30-100 ms decay) + quasi-periodic repetition 0.5-3 Hz; bonus: rising ring pitch over ~10 s = "hot chocolate effect" (cocoa dissolving). | Moderate. **Probed: strong tier (>=8 s, >=1.5 ringy onsets/s) 4/5 hits, 0-3 FA/rec (Sec. 4.2).** |
| Container opening (A7) | Generic transient gate (50-500 ms) with duration/band/centroid features. | Weakest classical detector - kitchen transients are a zoo. Use only as unlabeled low-confidence trigger to wake AST/CLAP + grab an RGB frame. |

### 3b. Audio - pretrained and LALM tiers (AST/CLAP canonical in `DETECTOR_CATALOG.md`; full comparison here)

| Model | Events | Cost | Local-ready | Verdict |
|-------|--------|------|-------------|---------|
| **AST** (MIT/ast-finetuned-audioset-10-10-0.4593) | all 527 AudioSet classes; every kitchen class verified present (Microwave oven, Beep bleep, Boiling, Blender, Sizzle, Pour, Stir, Water tap faucet, Chopping, Frying, Cutlery, Dishes, Cupboard) | 86.6M params, mAP 0.459. MEASURED: 11 ms/window GPU; 492-1236 ms CPU (16/4 threads); input always padded to 10.24 s | yes, cleanest path - verified importable in transformers 4.57.1, torchaudio fbank, no TF/librosa | **Medium-tier primary.** Free at 1 Hz on GPU; trigger-only on CPU. Calibrate per-class sigmoid thresholds; absolute scores tiny. **Probed (Sec. 4.3).** |
| **CLAP** (laion/clap-htsat-unfused) | open vocabulary via text prompts; covers AudioSet gaps (e.g. kettle whistle) | 153.5M total but only 27.5M audio tower per window. MEASURED: 169 ms CPU(4t), 9.4 ms GPU, text embeddings precomputed once | yes - verified importable, no librosa/TF | **Open-vocab disambiguator + zero-shot segment labeler.** Below supervised taggers on AudioSet-style classes. **Probed (Sec. 4.4).** |
| PANNs CNN14 | all 527 classes | 80.8M, mAP 0.431; est. 150-400 ms CPU(4t), 5-15 ms GPU | yes w/ minor glue (torchlibrosa + torchaudio loading) | CPU-only fallback if a no-GPU deployment emerges (~3-5x faster than AST on CPU). |
| BEATs (iter3+ AS2M) | all 527 classes | ~90M, mAP 0.486 (best); AST-class latency | yes w/ vendored BEATs.py (not in transformers core) | v2 upgrade only if AST accuracy falls short (+2.7 mAP for vendoring pain). |
| YAMNet | 521 classes | 3.7M, mAP ~0.31; 10-50 ms CPU | NO - TF-native; stack has no TF; unofficial torch ports unmaintained | Skip on this stack (CLAP is nearly as cheap and better). Only relevant for future on-glasses TFLite. |
| Whisper (tiny/base/small) | speech/commands only - not a sound tagger | 39M-242M; tiny ~0.3-0.8 s CPU per 5 s utterance; GPU 50-150 ms | yes - verified importable | Command channel only, gated behind cheap VAD (silero ~1 MB). Never for appliance sounds. |
| Qwen2-Audio-7B-Instruct | free-form audio QA | 8.40B, ~17 GB bf16, est. 0.5-2 s/clip GPU | yes - importable, fits one 48 GB GPU | Superseded by Qwen2.5-Omni unless audio-only wanted. |
| **Qwen2.5-Omni-7B** | joint audio+RGB+text QA - matches the glasses sensor suite | 10.73B (disable Talker head: -2 GB), ~22+ GB bf16, est. 1-3 s/query GPU | yes - importable, fits one of four 48 GB GPUs | **LALM tier.** On-demand fusion confirmations; slots next to Qwen2.5-VL in main.py. Trigger-driven, never duty-cycled. |
| Gemini audio (2.5-flash/pro) | free-form, cloud | API tokens (~32 tok/s audio), 1-3 s/request | no - cloud only; run_gemini.py plumbing exists | Offline eval / pseudo-labeling oracle only; cost + privacy disqualify always-on use. |

### 3c. Vision (all MEASURED on this machine unless noted)

| Detector | Kind | Cost (measured) | Ready | Verdict |
|----------|------|-----------------|-------|---------|
| OWLv2-base (transformers) | grounding | 31.5 ms/frame GPU fp16; 3898 ms CPU - never per-frame on CPU | yes, weights cached | Primary ROI acquisition, ONCE per stage on sharpest-of-3-5 frames (Laplacian pick); center-crop fisheye periphery. Best zero-shot vocab ("cinnamon stick"). |
| GroundingDINO-tiny | grounding | 87.5 ms GPU, 3350 ms CPU | yes | Slightly better boxes on clutter, slower; same once-per-stage role. |
| YOLO-World-S (ultralytics) | grounding | 91.7 ms CPU @640px, 5.5 ms GPU; 40.7 ms CPU @320px | yes (ultralytics 8.4.64 verified) | Only CPU-viable open-vocab grounder; also doubles as cheap "hand" prompt. Run @640 for small objects. |
| MOSSE (cv2.legacy) | tracking | 0.18-0.32 ms/frame, 3 ROIs < 1 ms | yes | Default per-frame tracker; refresh ~10 s (drifts under appearance change - melting chocolate). PSR self-confidence triggers escalation. |
| Template match (TM_CCOEFF_NORMED) | tracking/reacquire | 0.36 ms in 160px window; 5.8 ms full frame @480p | yes | Re-acquisition after tracker loss; pair with global phase-correlate (~1 ms) shift compensation. Store 2-3 templates. |
| CSRT | tracking | 13.7-19.4 ms/frame | yes | Escalation tier for a few seconds after occlusion. At 1 fps sampling ALL classical trackers fail without global shift compensation. |
| KCF | tracking | 27.7-33.5 ms/frame - SLOWER than CSRT on this opencv 4.13 build | yes | Skip entirely: slower AND less robust here. |
| Skin+motion heuristic (YCrCb + morphology + frame-diff gate) | hands | 0.82 ms @480p, 0.16 ms @320p | yes | **Recommended hand gate**: (skin AND dilated-ROI AND motion). Blur-immune. Calibrate Cr/Cb at session start; browning chocolate can enter skin gamut - always AND with motion. |
| MediaPipe Hands | hands | ~17 ms CPU (docs, not measured) | **NO** - no cp313 wheels (issue #6159) | Not worth a py3.12 venv; skin heuristic + YOLO-World "hand" covers presence/near-ROI. |
| 100DOH / EgoHOS / EgoHands-YOLO | hands/contact | 100-200 ms GPU (100DOH); YOLOv8n-EgoHands est. 2-3 ms GPU | partial (legacy deps); YOLO-World path runs today | Only family trained on egocentric viewpoints; revisit if contact-vs-hover discrimination becomes necessary. |
| Frame differencing + ROI energy | motion | 0.42 ms @480p; ROI-only ~0.02 ms | yes | Universal cheap gate. Must shift-compensate ego-motion first (phase correlate ~1 ms) or gate on head-still windows. |
| Farneback ROI flow + annulus-median ego subtraction | motion/periodic | 5.3 ms on 160px ROI; 67.4 ms full @480p | yes | Use ROI-only. **Global-frame variant probed and FAILED (Sec. 4.5)** - ROI restriction is mandatory, not optional. |
| FFT periodicity (0.5-3 Hz) on ROI energy | periodic | 0.011 ms/rFFT; real cost = 8 fps burst sampling of the ROI | yes | Needs >= 6-8 fps bursts (5 fps Nyquist truncates the band - confirmed in probe). Require ROI peak != global head-motion peak. |
| HSV hist shift (Bhattacharyya) | state/transfer | 0.05 ms/ROI | yes | Pre/post hand-episode hist step IS the rgb_transfer signal; freeze refs during occlusion; drop V channel vs AWB drift. |
| Liquid-level edge (Canny + row projection) | state/transfer | 0.25 ms on 80x160 ROI | yes | Rising level across hand episode corroborates "milk poured". Median over 5 frames vs speculars. |
| Steam shimmer (8-frame temporal variance) | state | 0.07 ms/ROI; needs 4-8 fps ROI buffer | yes | Milk-near-simmer cue; only on head-steady windows; require > 3 s sustained. |
| EasyOCR | display read | 216 ms CPU / 10.9 ms GPU per display crop; read "12:30" off synthetic noisy 7-seg after 3x upscale | yes (verified py3.13) | Microwave display reader: sharpest-frame, rectify, 3x upscale, digit allowlist, majority-vote 3 reads. Fallback ladder: per-digit segment thresholding (free, deterministic) > ssocr > tesseract+letsgodigital (needs apt) > PaddleOCR (avoid 2nd framework). |

Cross-cutting egocentric rules baked into all of the above: (1) sharpness-gate every one-shot model; (2) compensate or gate on global ego-motion before trusting any motion signal; (3) treat hand occlusion as a state - freeze references, compare pre/post; (4) distrust fisheye periphery; (5) ratio/normalized features everywhere (AGC).

---

## 4. Probe results (measured on the six activity-8 recordings vs GT)

Recordings: 8_16 (tuning, all thresholds frozen here), 8_26/8_3/8_25/8_31/8_50 (frozen eval). GT timing errors: 8_26 steps 89+83 (microwaved ~2 min instead of 1), 8_31 steps 89+83 (short runs). Step key: 88 fill milk, 89 microwave 1 min, 90 add chocolate, 87 add sugar, 85 mix, 83 heat 1 min + serve, 84 add cinnamon.
Full data: `/home/kailaic/NeuroTrace/ProceduralAgent/detectors/probes/results_{hum_beep,pour_clink,ast_tagger,clap_zeroshot,vision_motion}.json`.

### 4.1 hum_beep (classical DSP) - WORKS (partial: fuse with beeps)

Method: 100-1000 Hz band level (+6 dB over 20th-pct baseline) AND 120 Hz mains-line contrast (+5 dB) AND rolling-std stationarity (<= 0.25 dB), 2-of-3 vote, 5.4 s median smoothing, runs >= 20 s; beeps = 800-5000 Hz tonal bursts; timer flag if hum duration outside [40, 80] s.

| Rec | Runs det. | Hum durations (s) | Onset err (s) | False runs | Timer flags vs GT |
|-----|-----------|-------------------|---------------|------------|-------------------|
| 8_16 (tune) | 2/2 | 59.1, 59.9 | 2.6, 10.0 | 0 | agree 2/2 (clean) |
| 8_26 | 2/2 | **113.7** (step89), 50.7 | 7.3, 55.8 | 0 | step89 flagged CORRECT (GT 2-min error); step83 missed (hum truncated to 50.7 s; end-beeps at 174.5-178.5 s and 436-442 s mark the true offsets) |
| 8_3 | 2/2 | 59.9, 60.9 | 10.7, 11.3 | 0 | agree 2/2 (clean) |
| 8_25 | 2/2 | 20.2 (truncated), 43.0 | 51.1, 15.5 | 0 | FALSE timer alarm on step89 (low-SNR truncation) |
| 8_31 | 2/2 | **20.2** (step89), 46.3 | 28.4, 8.0 | 0 | step89 flagged CORRECT (GT short-run error); step83 missed (46.3 s inside band) |
| 8_50 | 1/2 | 58.4; step83 MISSED (low-power heat) | 4.1, - | 0 | step89 agree; step83 no read |

Totals: **11/12 GT microwave runs detected, 0 false runs across all six recordings**; both headline timing errors flagged, no flags on clean 8_16/8_3; frozen timer agreement 6/10 (+2/2 tuning). Beep validation (frozen): hum offsets with beep within 15 s = 8/9, onsets 7/9. Latency: 0.53-0.95 s wall per recording, **476-503x real-time, one CPU core, numpy/scipy STFT only**.
Caveats: at very low SNR hum duration truncates (8_25 false alarm, 8_26 step83 under-read) and one low-power heat was missed (8_50 step83) -> **drive the microwave stage from hum fused with end-of-cycle beep times** (beeps were within ~1 s of truth in exactly the failure cases). Disclosed design leakage: beep band widened, 5.4 s smoothing, broadband gate were chosen after seeing eval recordings (numeric thresholds still 8_16-only).

### 4.2 pour_clink (classical DSP) - WEAK EVIDENCE ONLY (cheap, gated use)

Pour = 0.5-4 kHz (rel +4 dB, flatness >= 0.12, 0.8-8 s); clink = spectral-flux onsets with 2-8 kHz ringy decay grouped into trains; strong tier = >= 8 s AND >= 1.5 ringy onsets/s.

| Rec | Pour hit (step88) | Pour FA/min outside | Clink hit (step85) | Clink FA events | Strong-tier |
|-----|--------------------|---------------------|---------------------|------------------|-------------|
| 8_16 (tune) | yes | 2.55 | yes (train 319.5-392 s, dens 1.99) | 14 | hit |
| 8_26 | yes | ~similar (31 det., 26 FA) | yes (280.5-314.5 s vs GT 285.6-314.9 - perfect) | 15 | **hit, 0 FA** |
| 8_3 | yes | 2.81 | yes (dens 1.83) | 10 | hit |
| 8_25 | yes | 3.50 | yes (dens 1.32) | 12 | **MISS** (1.32 < 1.5 threshold) |
| 8_31 | yes | 0.53 | yes (dens 2.06) | 3 | hit (2 FA) |
| 8_50 | yes | 3.99 | n/a (step85 skipped in GT) | 9 | n/a (1 FA) |

Pour: perfect 6/6 step-level recall but 0.5-4.0 FA/min (sugar scooping, wrapper rustle, serving pours all look pour-like) - usable ONLY gated by step prior or RGB ROI check. Clink: 5/5 raw recall but fires in nearly every manipulation step; strong tier is the usable signal (4/5 hits, 0-3 FA/rec). Latency 1.3-2.7 s/recording CPU (~0.5% real-time). Verdict: do not emit standalone completion events; use clink-train density as continuous stir evidence + pour as corroboration inside an expected window, RGB-confirmed.

### 4.3 ast_tagger (AST, AudioSet 527) - WORKS for microwave; NO for pour/add steps

"Microwave oven" score vs GT steps 89+83, 5 s windows / 2.5 s hop:

| Rec | Microwave AUC | Frozen P/R/F1 (thr 0.0107 tuned on 8_16) |
|-----|---------------|------------------------------------------|
| 8_16 (tune) | 0.98 | F1 0.964 (tuned) |
| 8_3 | 0.97 | .982/.821/.894 |
| 8_31 | 0.95 | .683/1.0/.811 |
| 8_50 | 0.86 | .878/.706/.783 |
| 8_26 | 0.79 | .835/.689/.755 (the ~2-min error run is visible: span ~40-182 s) |
| 8_25 | 0.63 | .430/.645/.516 (strong activations OUTSIDE annotated steps) |
| pooled | 0.868 | .725/.752/.738 |

Event alignment ~3-5 s on clean recordings (8_16 det. 67.5-137.5 vs GT 70.4-134.5; 8_3 332.5-412.5 vs 332.1-409.3); durations over-extend at frozen threshold (8_31 first run est. 67.5 s vs ~35 s actual). Other classes: Cutlery AUC 0.82 / Dishes 0.78 vs stir step; Pour 0.63 / Liquid 0.62 vs fill step (weak); dry-ingredient adds have NO class signature (top classes Crumpling/Scissors/Zipper). Absolute sigmoids are tiny (in-step mean 0.16-0.23) -> F1 0.96 -> 0.74 pooled under a frozen threshold; use rank/relative or hysteresis thresholding in production.
Latency: end-to-end 16-18 ms/window GPU (12.5 ms forward batch1; FE 3.9 ms warm), 745 ms CPU; full-recording sweep 2.9-7.4 s GPU batched (est. 3.2 s per 450 s; 133 s CPU). One GPU window = 0.018 s vs 2 s Gemini call (~110x cheaper) vs 45 s local Qwen (~2500x).

### 4.4 clap_zeroshot (CLAP text prompts) - PARTIAL: loud stationary yes, quiet transients no

| Event (prompt) | AUC tune 8_16 | AUC pooled eval-5 | Frozen-threshold notes |
|----------------|----------------|--------------------|------------------------|
| microwave running | 0.982 (F1 0.917) | 0.819 (all-6 0.863) | worst room 8_25: AUC 0.599 / F1 0.54 - constant kitchen hum aliases to the prompt |
| stirring (spoon + glass clink) | 0.825 | 0.844 | usable weak signal; pooled frozen F1 0.435 |
| pouring into a cup | 0.448 | 0.536 (all-6 0.515) | **chance** - CLAP cannot hear a quiet pour in 5 s windows; a task-JSON compiler can NOT just write "pouring liquid into a cup" |

Segment-level prompt RANKING is semantically excellent: on 8_16 and 8_26 every microwave GT segment ranks "a microwave oven running" first (mean prob 0.79-0.98), add-ingredient segments rank "opening a wrapper or container" first, mix ranks stirring first on the clean run -> reliable zero-shot segment LABELER even where window thresholds are brittle. 8_26's ~95 s contiguous detected run inside step 89 supports timing-error checks in quiet rooms. Latency: 1.89 ms/window GPU forward; per-recording total 2.4-4.1 s (CPU feature extraction ~20 ms/window dominates; ~0.9% of real-time); CPU single-window forward 3.0 s (GPU-only for sweeps).

### 4.5 vision_motion (global Farneback periodicity for stirring) - FAILED

Method: 320p @ 5 fps grayscale Farneback, median-vector ego compensation, 0.5-2.5 Hz FFT peak/median score, 10 s windows (note: 5 fps Nyquist already truncates the requested 0.5-3 Hz band).
- Tuning 8_16: ROC-AUC **0.486** (below chance); mean score INSIDE stir step 85 (8.38) lower than outside (9.30). Frozen eval 8_26: AUC 0.555, F1 0.135 at precision 0.072.
- All top-5 periodicity windows in BOTH recordings lie outside step 85 with 0.5-0.65 Hz peaks = head sway/walking. Median-vector compensation removes translation only; residual rotation/parallax yields a 1/f spectrum whose LF peak wins every window.
- Band variant (1.0-2.5 Hz) selected on 8_16 (AUC 0.597) fell to 0.398 on 8_26 - the signal is noise.
- Activity-level sanity partially holds: on 8_16 the microwave-wait step 89 is clearly lowest residual motion (1.90 px vs 3.1-5.2 px elsewhere), but mix is not highest, and 8_26 shows no per-step contrast at all.
- Cost is fine (12.7-15.0 ms/frame CPU, ~12x real-time) - the SIGNAL, not compute, is the blocker.
- Path to fix (next probe, not assumed): (1) ROI-restricted flow after one-shot mug/hand grounding - the only transferring signals were spatial-concentration features (post-hoc AUC 0.733/0.86 on 8_26, but not selectable from one tuning recording); (2) homography/RANSAC ego compensation; (3) 10-15 fps burst sampling.

---

## 5. Cost ledger sketch (per 450 s recording; measured latencies)

VLM reference points: Gemini 2.5 Flash ~2 s/call (API), Qwen3.6-27B ~45 s/call (local server); baseline = one call per 10 s, 3 frames/call -> 45 calls per recording.

| Tier | What runs | Wall per 450 s recording | Real-time factor | Hardware |
|------|-----------|--------------------------|------------------|----------|
| Always-on DSP | hum+beep (0.9 s) + pour+clink (2.5 s), shared STFT | ~3.4 s | ~130x faster than RT (hum_beep alone ~490x) | 1 CPU core, <1% when streaming |
| Duty-cycled AST sweep | 5 s win / 2.5 s hop, ~180 windows | 3.2 s GPU batched / 133 s CPU | 140x GPU / 3.4x CPU | RTX 6000 Ada or 4 CPU threads |
| Duty-cycled CLAP sweep | same windowing, 7 cached prompts | 3.1-4.1 s (FE-dominated) | ~110x | GPU forward + CPU FE |
| Vision motion pass (probe config) | 320p @ 5 fps Farneback | ~38 s | ~12x | CPU |
| Catalog vision base loop (est., Sec. 3c) | 1 fps decode + shift + 3x MOSSE + skin + diff/hist/level | ~6.5 ms/frame -> ~3 s | <1% of one core | CPU |
| Periodic VLM baseline - Gemini 2.5 Flash | 45 calls x 2 s | 90 s wall (sequential) + API cost | 5x RT only if parallelized | cloud |
| Periodic VLM baseline - Qwen3.6-27B | 45 calls x 45 s | 2025 s | **4.5x SLOWER than real-time - cannot run live** | local GPU server |

Exchange rates: one AST GPU window (0.018 s) ~= 1/110 Gemini call ~= 1/2500 Qwen call. A full-recording AST GPU sweep (3.2 s) costs the wall time of ~1.6 Gemini calls; the ENTIRE always-on DSP bank for a recording (3.4 s) costs less than 2 Gemini calls or 0.08 Qwen calls. Replacing even 90% of the 45 baseline VLM calls with DSP/AST gates saves ~81 s wall (Gemini) or ~1823 s (Qwen) per recording while adding ~7 s of compute. The binding constraint is detector logic quality, not compute.

---

## 6. Recommended detector library v1 + next experiment

### Library v1 (spiced hot chocolate)

Always-on cheap tier (one shared 16 kHz STFT, <<1% core):
1. **hum+beep fusion** - microwave state machine: hum run-length for "running", end-beep repetition train for the authoritative offset (fixes all three hum failure modes observed: 8_25 truncation, 8_26 step83 under-read, 8_50 step83 miss); timer flag from fused duration vs [40, 80] s.
2. **clink-train density (strong tier: >= 8 s, >= 1.5 ringy onsets/s)** - stir-progress evidence; consider lowering density threshold toward ~1.3 with an RGB confirm (8_25 missed at 1.32).
3. **pour band-energy** - corroboration ONLY inside the expected fill/serve windows, gated by step prior or RGB ROI (milk carton + mug visible); never standalone (0.5-4.0 FA/min).
4. Water-tap edges + container-opening trigger compiled in, low priority; blender detector as mask flag (other recipes).

Medium tier (triggered or 0.2-1 Hz on GPU):
5. **AST** "Microwave oven" with rank/hysteresis thresholding as cross-check on the DSP state (pooled AUC 0.868, 16-18 ms/window GPU); Cutlery/Dishes as secondary stir evidence. Do not use AST for pour/add steps.
6. **CLAP** as zero-shot segment labeler + open-vocab disambiguator (microwave segments rank correctly with mean prob 0.79-0.98); per-environment threshold calibration mandatory (8_25 hum aliasing).

Vision tier (catalog-validated for cost, NOT yet probed against GT except global motion - treat as candidates):
7. OWLv2 once-per-stage grounding (31.5 ms GPU) -> MOSSE (0.3 ms) / template (0.4 ms) / CSRT (15 ms) ladder with phase-correlate shift compensation; skin+motion hand gate (0.9 ms); HSV-hist + liquid-level pre/post hand-episode comparison as the rgb_transfer confirmation; EasyOCR display reads when facing the microwave. **Global-frame periodicity is dead** - any stirring vision must be ROI-restricted flow at >= 10 fps bursts.

Escalation: classical events below confidence -> AST+CLAP on buffered 10 s window -> persistent disagreement or user query -> Qwen2.5-Omni-7B with audio + keyframes (or Gemini offline for pseudo-labeling only).

### Next experiment

**Replay the task JSON's completion events end-to-end on all six recordings and score stage tracking + Timing-Error reminders vs the periodic-VLM baseline.**
- Inputs: six recordings, `gt_activity8.json`, frozen v1 detectors (hum+beep fusion, clink strong tier, gated pour, AST microwave cross-check).
- Run the task-graph tracker on detector events alone; emit (a) stage-advance events, (b) Timing-Error reminders when fused microwave duration leaves [40, 80] s.
- Score: stage-advance accuracy and median advance latency vs GT step boundaries; reminder precision/recall against the 4 GT timing-error steps (8_26 89+83, 8_31 89+83) and 8 clean microwave steps; reminder latency vs GT error window end.
- Baseline: periodic VLM at 10 s / 3 frames (Gemini 2.5 Flash, and Qwen3.6-27B as the local reference) on the same recordings, same scoring; record the actual cost ledger per Sec. 5.
- Known headroom to beat: detectors already get 11/12 microwave runs, 0 false runs, 2/2 headline error flags at ~1/26th the wall cost of the Gemini baseline; open risks are the 8_25 false timer alarm (beep fusion should clear it) and the 8_50 step83 miss (AST cross-check has signal there: F1 0.783).
- Success = stage tracking within a few seconds of GT on >= 5/6 recordings and reminder F1 >= the VLM baseline at < 5% of its cost; then extend the same harness to microwaveeggsandwich (same detectors) and panfriedtofu (adds sizzle detector).
