# Procedural Assistance Task Landscape (survey 2026-06-12)

> 🧭 **Reconcile with the three-box restructure (2026-06-15).** This survey predates the mechanical-only cut; read these four corrections alongside it: (1) **Preventive/anticipatory windows are not GT** — earliness is scored as STS *within a reactive window*, not as a separate output space (`FAMILY_A_CC4D_AUGMENTATION.md` §4.2). (2) **Order-error adjudication is SUSPENDED**, not active work (the 8_50 narrative below describes a suspended class). (3) **Safety is EXCLUDED** (no annotation), not a deferred deployment target. (4) **Family C / next-step guidance is excluded by design**, not "paused." The §5–5.1 Qualcomm review here is a summary; the authoritative GT derivation is `FAMILY_A_CC4D_AUGMENTATION.md`. Survey value (lit map, metric sources, the "VLMs fail at when-to-intervene" evidence) is unchanged.

Purpose: enumerate every formally defined "proactive reminder / procedure assistance" task in the literature — input (causal or not), output, baselines, metrics, and adaptability to our sensor-graph (cheap audio/timer/RGB detectors + task-graph logic, CC4D data). Sources: 24 papers on disk (read this session) + Semantic Scholar sweep of all 39 CC4D-citing works. Incorporates the schema-level dataset survey (formerly `RELATED_WORK_TASK_STRUCTURES.md`, merged in 2026-06-15 as §6–§7 below).

## 0. Strategic picture (read this first)

1. **The user-strategy is confirmed by the survey**: the 2025–26 proactive-task papers define the tasks but do not give us usable data — Pro2Assist (2605.04227) releases nothing; PWR's EgoProactive (2606.04970) IS released (gated HF, 23 GB) but is **video-only (no audio)** and its decision points are 0.7–3.9 s micro-action coaching, finer than recipe stages. ProAct-75 data unreleased; ProAssist is LLM-synthetic references. So: adopt the *task definitions and metrics* from the new papers, instantiate them on an *older released dataset with audio + step + error annotations* = CaptainCook4D. No published work uses CC4D's audio at all (confirmed via citation sweep) — that combination is unclaimed.
2. **One urgent check**: arXiv **2511.21998 (Qualcomm "LiveMamba", live task guidance)** already annotated CaptainCook4D with timestamped instructions / feedback / mistake alerts for streaming guidance. Before building our reminder truth tables further, examine their annotation layer — partly reusable as GT, and LiveMamba becomes the heavyweight end-to-end baseline our sensor-graph undercuts.
3. **Consistent cross-paper finding** supporting our architecture: VLMs are bad at exactly the part detectors+graph can do. GPT-5.2 collapses to ≤.08 G-Mean F1 on interrupt/silent (PWR); frontier MLLMs <13 mF1 on step detection (ProAct); streaming models 7.9 F1 on step instruction and collapse past 180 s (OmniPro); GPT baselines ≈ chance on when-to-talk and over-talk vs humans (WTaG); non-speech audio is every omni-model's weakest modality (OmniPro). Meanwhile YETI shows SSIM + object-count deltas *beat* VLM-feature baselines for when-to-intervene, and ProAssist finds more visual tokens don't help — "bottlenecked by procedural reasoning, not perception."

## 1. Task catalog

### 1.0 Terminology

**Evaluation settings** — how much of the recording the system may use at the moment it decides:
- **Offline** — the full video is available at inference; the method may exploit frames *after* the moment it labels (whole-video alignment, classifying a segment after it ends). Most error-detection work is offline.
- **Causal** — at decision time t the system sees only data from [0, t]; future frames forbidden. Umbrella term for the next three.
- **Causal at decision points** — input is causal, but the *benchmark* supplies the timestamps at which the model must answer (the model does not choose when to act). PWR and WTaG score this way.
- **Online** — causal AND processed incrementally as the stream arrives; a small, reported processing delay is tolerated (TI-PREGO explicitly distinguishes "online" from hard "real-time").
- **Streaming (proactive)** — online AND the model itself decides at every timestep whether to emit output; silence is an action and there is no external query schedule. Hardest setting; what ProAssist/OmniPro-online/ESTP require.
- **Truncated-video QA** — one-shot offline evaluation whose video input is cut at the question moment, so the *information* is causal even though nothing is processed as a stream (ProMQA style).
- **One-class (OCC)** — training uses only correct executions; errors are unseen until test time. Matches deployment, where error examples are scarce.

**Input source vocabulary**: ego RGB (head-mounted) vs exo RGB (third-person); audio — speech (dialogue/ASR) vs non-speech (appliance hum, beeps, sizzle); IMU; gaze; hand pose; depth. Text side-inputs: recipe text / task graph / structured plan, dialogue history, standing instruction ("alert me when…"), persona-memory.

**Output format vocabulary**: binary decision token (speak/silent, error/normal); step label from a closed set; temporal segment (start, end, label); typed timestamped event (t, type-id, message); ordered completed-step list; free-form NL utterance; multiple choice; structured tuple.

**SG-fit** (our column): adaptability to our sensor-graph — ✚✚ directly servable by cheap detectors + task-graph logic; ✚ structural subcases only; ○ needs full VLM/LLM (escalation target or metric template only).

### Family A — When-to-intervene (proactive timing)

---
**A1. Intervention Decision — PWR / "Plan, Watch, Recover" (arXiv 2606.04970, Meta, 2026)**
- *Definition*: at each annotated decision point t, decide whether the assistant should speak now or stay silent; companion sub-tasks score the utterance (LLM judge) and out-of-plan mistake detection + recovery.
- *Setting*: **causal at decision points** (video/dialogue up to t only; query times given — 9,935 points on EgoProactive, 42,275 across Pro2Bench).
- *Input*: ego RGB @2 fps (8 frames per 8 s clip, plan-anchored clip selection), dialogue history (text), task goal (text), optionally a structured plan: step list with completed/current/next tags + per-step "visual cues" for step-complete vs step-ongoing evidence. **No audio.**
- *Output*: binary token `$interrupt$`/`$silent$`; on interrupt, a free-form utterance (golden references avg 9–17 words).
- *Evaluated on*: **EgoProactive** (new; 700 smart-glasses videos, scripted omission/reorder/substitution mistakes with recovery) + Pro2Bench re-annotations of **Ego4D, Ego-Exo4D, EPIC-KITCHENS, HoloAssist, HowTo100M**. CC4D explicitly rejected ("single domain, no paired recovery guidance").
- *Baselines*: zero-shot Gemini 3.1 Pro, Claude Opus 4.6, GPT-5.2, Qwen3-VL-235B, Qwen3.6-VL-27B, Llama 4 Maverick; fine-tuned monolithic vs their duplex+planner.
- *Metrics*: **G-Mean F1** = √(Interrupt-F1 × Silent-F1) (degenerate policies → 0); PQS unified score; OOP detection recall; judge 1–5 on recovery.
- *SG-fit*: ✚✚ — the decision is step-transition awareness against a plan; their per-step "visual cues" are our detectors expressed in text.

---
**A2. Proactive Trigger Prediction — Pro2Assist (arXiv 2605.04227, Columbia/CUHK, 2026)**
- *Definition*: at each inference moment decide whether to initiate assistance now; jointly identify current step + execution status; timing scored against per-step GT validity periods.
- *Setting*: **online** (deployed live on RayNeo X3 AR glasses; per-moment decisions, past context only).
- *Input*: ego RGB (current + 9 previous frames from a 10 FPS stream; IMU-gyro gates sampling 1→2 Hz), WikiHow-derived step graph (text), completed-step history (text), hand-motion cues (text). Audio only for the initial spoken command.
- *Output*: binary trigger; step label (closed set) + 4-way status {just start, in progress, about to finish, transition}; response text if triggered.
- *Evaluated on*: curated set built from **GTEA** (3 cooking tasks), **EgoPER** (5 recipes), **EgoProceL** (tent/PC assembly) — 1,089 samples, 10 tasks; + 20 self-recorded real-world sessions. **Not CC4D; dataset unreleased.**
- *Baselines*: VideoLLM-online-8B, ProAgent, vanilla/ICL/CoT prompting on Qwen3-VL 2B–30B and InternVL3 2B–14B.
- *Metrics*: Acc-P, Missed/False Detection rates; **STS = exp(−(t̂−s)/(e−s))** if step matches and t̂ ∈ [s,e], else 0.
- *SG-fit*: ✚✚ — closest architectural precedent (cheap sensing gates expensive VLM); their [s,e] periods ≡ our `reminder_windows`.

---
**A3. Proactive Dialogue Generation — ProAssist (arXiv 2506.05904, Meta/UMich, 2025)**
- *Definition*: generate the full assistant side of a task-guidance dialogue from a live video stream: at every frame either speak (instruction / mistake correction / feedback) or stay silent.
- *Setting*: **streaming** (decision every frame @2 FPS; matching window favors early predictions to prevent future-frame exploitation).
- *Input*: ego RGB stream + dialogue history + task goal; optional recipe text (recommended config). User turns are text; **no audio**.
- *Output*: per frame, free-form NL utterance or silence token; inference threshold θ tunes speak probability.
- *Evaluated on*: ProAssist dataset — 30,135 **LLM-synthesized** dialogues (LLaMA-3.1-70B pipeline) over 479 h from **Ego4D-GoalStep, EPIC-Kitchens, HoloAssist, Assembly101, EgoExoLearn, WTaG**. Not CC4D.
- *Baselines*: VideoLLM-Online (LLaMA-3.1-8B + SigLIP) variants; frontier MLLMs deemed infeasible (~6,000 calls per 30-min video).
- *Metrics*: bipartite-matched P/R/F1 with semantic + asymmetric temporal cost; LLM-judge incl. "promptness". Human-validated (Pearson ≈ .45).
- *SG-fit*: ✚ — the speak/silent decision fits us; free-form content needs an LLM (our templated reminders are an easier subset). Key finding for us: recipe conditioning helps more than extra visual tokens.

---
**A4. When-to-Talk — WTaG (arXiv 2311.00738, UMich, EMNLP 2023)**
- *Definition*: at each query point, decide if the instructor should speak in the next x seconds; sub-tasks classify intent, instruction type (incl. Mistake Correction / Next Step), and generate the utterance.
- *Setting*: **causal at query points** (a point after either party speaks or after 10 s silence; 5,921 points).
- *Input*: latest ego frame converted to text (BLIP-2 caption or EgoHOS+CLIP object/state list) + dialogue history + recipe text + elapsed time. Audio used as ASR dialogue only.
- *Output*: binary speak/not; 5-way intent; 4-way instruction type; free-form utterance.
- *Evaluated on*: **WTaG** (own: 56 recordings, ~10 h, 3 recipes, real human instructor-user pairs). Not CC4D.
- *Baselines*: zero-shot GPT-3.5 with the three perception front-ends.
- *Metrics*: micro-F1 (all ≈ chance on when-to-talk; LLMs over-talk); human helpful/annoying ratings on generations.
- *SG-fit*: ✚✚ — real-human when-to-talk GT; the demonstrated LLM failure is the gap detectors+graph target.

---
**A5. Intervention Frame Detection — YETI (arXiv 2501.09355, 2025)**
- *Definition*: identify the frames where an AI agent should proactively intervene, using only lightweight per-frame signals (frame-to-frame SSIM + PaliGemma object-count deltas @1 fps).
- *Setting*: **online** (signals computable on the fly; designed for on-device AR).
- *Input*: ego RGB only; rate-limit hyperparameters (min conversation interval, one intervention per episode). No audio.
- *Output*: set of intervention timestamps (binary per frame).
- *Evaluated on*: **HoloAssist** (482 videos with instructor interventions). Not CC4D.
- *Baselines*: HoloAssist's TimeSformer intervention baselines (RGB/Hands/Eye combos).
- *Metrics*: P/R/F1 with TP = within **±5 s** of GT intervention start. YETI 56.2 F1 vs 42.3 best TimeSformer.
- *SG-fit*: ✚✚ — existence proof that cheap signals beat VLM features at *when*; cannot decide *what type* (Correct-Mistake precision 11.7) — that's where graph state adds value.

---
**A6. Trigger Detection + Proactive Action Selection — ProAct-75 (arXiv 2602.03430, 2026)**
- *Definition*: five sub-tasks at each timestep: trigger needed now? which task? which step (graph node)? next n human actions? which robot action (graph-feasible, ideally on a parallel thread) or WAIT?
- *Setting*: quasi-**online** (sliding 5-keyframe window, stride 3, over the stream).
- *Input*: **exocentric** RGB window + task list + AND/OR task graph. No audio.
- *Output*: 5-tuple (binary trigger, task label, step label, action sequence, robot action ∈ steps ∪ WAIT).
- *Evaluated on*: ProAct-75 (75 tasks, 5,383 videos from **Ego-Exo4D, COIN, UCF-Crime** + 495 self-collected). Not CC4D; **data unreleased**.
- *Baselines*: GPT-4o, Gemini-2.5-Flash/Pro, Qwen3-VL/Omni-30B, Qwen2.5-VL; ProAct-Helper (LoRA + entropy-driven graph search).
- *Metrics*: mAcc/mF1 per sub-task; Saved Steps, thread Entropy Ratio, Parallel-Action %; trigger-hallucination analysis. Frontier MLLMs < 13 mF1 on step detection.
- *SG-fit*: ✚✚ conceptually (action selection is pure graph logic) but exocentric, robot-oriented, unreleased.

---
**A7. Standing-Instruction Streaming Alerts — OmniPro (arXiv 2605.18577, 2026)**
- *Definition*: 9 sub-tasks under one shape — a standing instruction is issued at stream start ("alert me when X / track my progress and give next-step guidance"); the model must respond at the right trigger moments. Most relevant sub-task: **Sequential Step Instruction** (track procedure progress, give the next step at the right time); also instant event alert, state-transition monitoring, counting, grounding.
- *Setting*: two protocols — **Probe** (causal at decision points: queried at −5..−2 s expecting "no" and 0..+3 s expecting the answer) and **Online** (true streaming: model decides when to emit).
- *Input*: continuous **audio-visual** stream (speech + non-speech sound + visual; 84% of samples audio-dependent) + standing instruction text.
- *Output*: structured exact-match answers (YES/NO, integer, state name) or short text at trigger times.
- *Evaluated on*: 2,700 human-verified samples over 1,262 videos from **LongVALE + COIN** (COIN supplies the procedural step-instruction subset). Not CC4D.
- *Baselines*: Gemini-3-Flash (best probe, 40.4), Qwen3-Omni-30B, Qwen2.5-Omni, MiniCPM-o 4.5, video-SALMONN, Phi-4-mm; online: LiveStar, MMDuet2, MiniCPM-o (20.9 F1; step-instruction only 7.9).
- *Metrics*: Probe accuracy (pre/post contrast implicitly scores timing); **Online F1 with ±3 s tolerance**, content must also be correct; over-triggering penalized.
- *SG-fit*: ✚✚ — only benchmark with non-speech audio as a first-class trigger; their findings (non-speech sound = weakest VLM modality; streaming collapse past 180 s) are direct evidence for cheap audio detectors.

---
**A8. Event/Intent-Driven Proactive Alerting — EgoPro-Bench (arXiv 2605.07299, SenseTime, 2026)**
- *Definition*: given a standing instruction (Object/Action subsets: "notify me when the mushroom appears") or a persona-memory (Intent subset: intervene once, helpfully, at the right moment), emit `<Attention>`/`<Silence>` per timestep.
- *Setting*: **streaming** (frame-by-frame @1 FPS with dialogue history).
- *Input*: ego RGB stream + instruction text or user-memory text. No audio.
- *Output*: per-timestep attention/silence; Intent subset additionally a free-form personalized utterance (one per GT interval).
- *Evaluated on*: 2,400 eval videos assembled from **EgoBlind, StreamGaze, EgoExoLearn, EgoTextVQA, EgoSchema, LLaVA-Video, Ego4D, EgoQA**. Not CC4D.
- *Baselines*: Qwen2.5-VL 3B–72B, Qwen3-VL 4B–30B, TimeChatOnline, VideoChat-R1.5, their ProAct-Stream (SFT+RL).
- *Metrics*: P/R/F1 + **mIoU** + **GHA** (≥1 correct response per GT interval); Hungarian matching + LLM-judge (memory consistency, quality) for Intent.
- *SG-fit*: ✚ — event subsets ≈ open-vocabulary detector + temporal logic; intent subset needs memory reasoning (in our setting the "intent" is precompiled in the task JSON).

---
**A9. Streaming Proactive QA — ESTP / Eyes Wide Open (arXiv 2510.14560, NeurIPS 2025)**
- *Definition*: questions arrive at timestamps; answers depend on *future* visual evidence, so the model must hold the question, watch, and answer inside an annotated valid interval — neither early nor late, silent when uncertain.
- *Setting*: **streaming** (per-timestep action ∈ {silence, respond, request-hi-res-frame}).
- *Input*: ego RGB stream + timestamped text queries (multi-turn, 46% contextually linked). No audio.
- *Output*: per-timestep action + free-form answer with emission timestamp.
- *Evaluated on*: ESTP-Bench — 2,264 QA over 890 **Ego4D** validation videos; generalization on OVO-Bench, QAEGO4D. Not CC4D.
- *Baselines*: VideoLLM-Online, MMDuet, offline MLLMs under polling; their VideoLLM-EyeWO.
- *Metrics*: **ESTP-F1** — TP credit = LLM-judged answer quality × graded timeliness inside the valid interval; out-of-interval and unnecessary responses are FP terms in the denominator.
- *SG-fit*: ○ — open QA needs a VLM; but ESTP-F1 is the best-engineered when-to-respond metric in the literature (template for our scoring).

### Family B — Mistake / error detection

---
**B1. Supervised Error Recognition — CaptainCook4D (arXiv 2312.14556, NeurIPS 2024 D&B)**
- *Definition*: classify each step of a recording as error vs normal; 1-second sub-segments are classified then majority-voted per step.
- *Setting*: **offline** (ground-truth step segments are given; whole segment available).
- *Input*: frozen-backbone features over the trimmed step segment; multimodal variant uses **video + audio + depth + text via ImageBind** — the only published audio-on-CC4D numbers anywhere.
- *Output*: binary error/normal per step.
- *Evaluated on*: **CC4D** (384 recordings, 94.5 h, 24 recipes; step split + recordings split).
- *Baselines*: MLP / step-transformer / multimodal-transformer heads over 3D-ResNet, SlowFast, X3D, VideoMAE, Omnivore, ImageBind features.
- *Metrics*: Acc/P/R/F1/AUC. No timing.
- *SG-fit*: ✚ — order/missing reduce to graph; Technique/Measurement/Preparation need perception; **Timing/Temperature errors suit audio + clock detectors**.

---
**B2. Zero-Shot Error Recognition — CaptainCook4D (same paper)**
- *Definition*: prompt a VLM with step + error-taxonomy questions about a trimmed clip; recognize errors with no training.
- *Setting*: **offline** per segment. *Input*: trimmed clip + text prompts (Llama3-generated from task graphs/taxonomy). *Output*: binary per step.
- *Evaluated on*: **CC4D**. *Baselines*: Video-LLaVA, TimeChat — both near-useless (F1 ≤ ~7, recall-skewed).
- *Metrics*: Acc/P/R/F1. *SG-fit*: ○.

---
**B3. Online Mistake Detection (open-set, one-class) — PREGO (CVPR 2024) / TI-PREGO (arXiv 2411.02570)**
- *Definition*: first formalization of online procedural mistake detection: at each time τ, (1) recognize the current step from frames up to τ, (2) anticipate the expected step from the recognized history via an LLM; recognized ≠ anticipated ⇒ mistake. Trained only on correct executions.
- *Setting*: **online, one-class** (strictly causal; test videos trimmed at the first mistake).
- *Input*: ego RGB only (the LLM anticipator consumes the recognized **symbol stream**, not pixels).
- *Output*: binary correct/mistake per recognized step.
- *Evaluated on*: **Assembly101-O** and **Epic-tent-O** (their re-splits: correct procedures in train, mistake videos in test). CC4D cited but never benchmarked — **no online-MD benchmark on CC4D exists**.
- *Baselines*: transition matrix, BERT-NSP, OadTR; recognizers {Oracle, OadTR, MiniROAD} × anticipators {GPT-3.5, Llama-2/3, Gemma, Mistral}, zero/few-shot, CoT, LoRA.
- *Metrics*: mistake-class P/R/F1; TI-PREGO adds balanced precision for class imbalance. Online by construction; no latency score.
- *SG-fit*: ✚✚ — the symbolic anticipation layer IS an implicit task graph; Oracle-recognition rows prove the logic layer does the work.

---
**B4. One-Class Error Detection + Action Segmentation — EgoPER (CVPR 2024)**
- *Definition*: train on error-free videos only; at test, segment the video into steps and flag erroneous frames/segments; separately output the set of omitted steps.
- *Setting*: **offline** (paper explicitly assumes the entire video at inference), one-class.
- *Input*: ego RGB (I3D features); dataset also ships **audio**, depth, gaze, hand — unused in experiments.
- *Output*: per-frame step label (+background) + per-frame error flag → segment-level labels; omitted-step set via edit distance to training transcripts.
- *Evaluated on*: **EgoPER** (own: 386 cooking videos, 28 h, 5 recipes, HoloLens2) + HoloAssist + ATA.
- *Baselines*: video-anomaly methods (HF2-VAD, S3R) vs their EgoPED (TAS backbone + active-object GCN + contrastive step prototypes).
- *Metrics*: EDA (segment error-detection accuracy), micro-AUC, Omission-Acc/IoU, TAS metrics. No timing.
- *SG-fit*: ✚ — omissions reduce to sequence/graph logic; slips/modifications need prototype-level perception.

---
**B5. Ordering Mistake Detection on Symbols — "Every Mistake Counts in Assembly" (arXiv 2307.16453)**
- *Definition*: given a stream of symbolic assembly steps (verb, this-part, that-part), label each step correct/mistake/correction using learned spatial (part topology) + temporal (ordering) belief rules built online across episodes.
- *Setting*: **online over symbols** (vision assumed solved upstream; they contribute new part-pair GT labels).
- *Input*: ground-truth symbolic action sequence — no pixels at all.
- *Output*: 3-way {correct, mistake, correction} (+6-class fine variant).
- *Evaluated on*: **Assembly101** (328 sequences, 101 toys, leave-one-out splits). Not CC4D.
- *Baselines*: TempAgg, LSTM vs their BeliefBuilder + Inferencer (Acc 86.0, mF1 71.8).
- *Metrics*: per-class P/R, Acc, mean F1. No latency score.
- *SG-fit*: ✚✚ maximal — direct proof ordering-mistake detection reduces to transition tracking + graph constraints.

---
**B6. Procedure Step Recognition (PSR) — IndustReal (WACV 2024)**
- *Definition*: at any time t output the ordered list of procedure steps **correctly completed** so far (completion ≠ execution: a step attempted wrongly must not be listed). Flexible execution order allowed.
- *Setting*: **online** ("does not require a full recording as input"; 178 fps pipeline).
- *Input*: ego RGB up to t (HoloLens2; stereo/depth/gaze available) + the procedure step set. No audio recorded.
- *Output*: ordered completed-step list with timestamps + confidences.
- *Evaluated on*: **IndustReal** (own: 84 recordings, 5.8 h, toy-car assembly/maintenance, 27 participants; 14 of 38 error types only in val/test). Not CC4D.
- *Baselines*: YOLOv8 assembly-state detection + 3 logic layers (state-change → confidence accumulation → procedure-order filter; the order filter gives the biggest gain).
- *Metrics*: POS (Damerau-Levenshtein order similarity); F1 where firing **before** actual completion counts as FP; **average delay τ** over TPs (≈15–22 s; 60+ s on error recordings) — the only pre-2025 task scoring latency.
- *SG-fit*: ✚✚ — completion events ≡ our detector firings; failure mode (near-identical error states → 65% FP) shows where VLM escalation is needed.

---
**B7. Mistake Detection + Intervention-Type Prediction — HoloAssist (ICCV 2023)**
- *Definition*: (a) label each fine-grained action correct/mistake using features from the enclosing coarse action's start up to the current fine action's end; (b) given a 1–5 s window *before* a known intervention, predict its type {correct-mistake, follow-up instruction, confirm-action}.
- *Setting*: **causal within given segments** (GT action segmentation assumed; intervention time given for (b)).
- *Input*: ego RGB + hand pose + eye gaze (+IMU ablations); audio recorded but unused in benchmarks.
- *Output*: (a) binary per fine action (~6% positives); (b) 3-way type label.
- *Evaluated on*: **HoloAssist** (own: 166 h, 350 instructor-performer pairs, 20 object-centric tasks — no cooking). Not CC4D.
- *Baselines*: multimodal TimeSformer combos; hands-only is strongest for mistakes (F1 40.2).
- *Metrics*: F-score, per-class P/R. When-to-intervene itself not scored (YETI later filled that on this data).
- *SG-fit*: ✚ — their analysis shows graph structure predicts intervention urgency (order-constrained steps get immediate corrections).

---
**B8. Text-Referenced Mistake Detection — EgoOops (arXiv 2410.05343)**
- *Definition*: align an untrimmed video to a written procedure (step-level video-text alignment), then classify each step {correct, mistake, correction}; thesis: some mistakes (tape where text says glue) are invisible without the text.
- *Setting*: **offline** (Drop-DTW global alignment over the whole video).
- *Input*: ego RGB + procedural text. No audio.
- *Output*: per-step temporal segment aligned to text + 3-way label.
- *Evaluated on*: **EgoOops** (own: 50 videos, 6.8 h, 5 text-following domains beyond cooking). Transfer baselines trained on **CC4D** and Assembly101 (the CC4D-trained classifier transfers better).
- *Baselines*: ActionFormer; StepFormer++ + MLP; zero-shot InternVL2.5-8B / Qwen2-VL-7B.
- *Metrics*: mAP@tIoU (mistake+correction classes); per-class P/R/F1. No timing.
- *SG-fit*: ○ — wrong-tool/wrong-way detection inherently needs text-conditioned perception (our `expensive_criteria` escalation case).

---
**B9. Live Guidance + Timestamped Mistake Alerts — LiveMamba / Qualcomm Interactive Cooking (arXiv 2511.21998, NeurIPS 2025)**
- *Definition*: watch a cooking stream causally and produce the reference coaching script: next-step instruction at each step completion, success confirmation, typed mistake correction at the moment the error is apparent.
- *Setting*: **streaming**.
- *Input*: ego RGB stream (+ recipe/plan text). Audio present in CC4D videos but unused.
- *Output*: typed timestamped messages — types: instruction / success / mistake_{technique, preparation, measurement, timing, temperature} / finish (+ unaligned variants in the planning config).
- *Evaluated on*: **CC4D — all 384 recordings**, via their hand-annotated GT script layer (verified in §5; local copy `data/qualcomm_interactive_cooking/`).
- *Baselines*: SOTA MLLMs vs their LiveMamba streaming architecture.
- *Metrics*: alert-to-timestamp alignment + content match.
- *SG-fit*: ✚✚ — the heavyweight end-to-end baseline on our exact dataset; their GT timestamps partly reusable (execution errors only — no order adjudication, see §5).

---
**B10. Streaming Mistake Intervention — Ego-MC-Bench (arXiv 2606.09547, 2026)**
- *Definition*: provide step-by-step cooking guidance and intervene with a textual correction as soon as a mistake is apparent.
- *Setting*: **streaming** (models prompted at 5 s intervals).
- *Input*: ego RGB stream + recipe. Voice in source recordings transcribed; models consume video+text.
- *Output*: timestamped free-form interventions + instructions.
- *Evaluated on*: **Ego-MC-Bench** (own, new: ~10 h, 40 sessions, 559 recipe steps, 954 verified feedback messages) + Ego-CoMist synthetic counterfactual training set. Not CC4D.
- *Baselines*: turn-based Qwen, Gemini, InternVL; streaming VideoLLM-online, LiveCC, ProAssist — all poor.
- *Metrics*: instruction-completion accuracy; **intervention P/R/F1 with a 30 s temporal window**; BERTScore/ROUGE-L for text.
- *SG-fit*: ✚✚ — the 30 s-window intervention-F1 protocol is our cross-paper comparable (M1).

---
**B11. Early Mistake Detection (learned exit) — MistExit (arXiv 2603.14252, 2026)**
- *Definition*: streaming binary keystep-correctness with a learned exit policy: commit a verdict while having observed as little of the step as possible (RL exit + future-feature anticipation).
- *Setting*: **streaming**. *Input*: video stream. *Output*: binary verdict + exit time.
- *Evaluated on*: "diverse real-world procedural video datasets" (CC4D likely included given citations — **unverified**, abstract doesn't enumerate).
- *Metrics*: accuracy vs fraction-of-video-observed curve.
- *SG-fit*: ✚✚ — formalizes the earliness/accuracy trade-off our trigger thresholds face.

---
**B12. Mistake Attribution — MATT (arXiv 2511.20525, 2025)**
- *Definition*: given a mistake, output the violated semantic role, the **Point-of-No-Return timestamp** (after which the mistake can't be undone), and the spatial location.
- *Setting*: **offline**. *Input*: video. *Output*: role + PNR timestamp + location.
- *Evaluated on*: EPIC-KITCHENS-M and Ego4D-M (built by their automated engine). Not CC4D (cited).
- *Baselines*: MisFormer. *Metrics*: role accuracy, PNR error.
- *SG-fit*: ✚ — PNR is the principled "latest useful reminder time", a concept for our deadline semantics.

### Family C — Procedural QA (content GT, fixed timestamp) — PAUSED as an eval target (2026-06-12)

Decision: not pursued for now — scoring needs an LLM judge and serving needs a VLM. Retained role: sensors act as a **context-window terminator** for escalation — a detector event (e.g. beep) closes an n-second window which is sent, with current graph state, as a targeted VLM query (= our "detectors + targeted escalation" arm; precedents: Pro2Assist IMU gating, ESTP's request-hi-res action, ProAssist's reasoning-not-perception bottleneck). ProMQA stays relevant as cheap future validation: a graph-state tracker answers its next/missing/order questions nearly for free.

| # | Task | Setting & input | Output | Evaluated on / built from | Baselines | Metrics | SG-fit | Source |
|---|---|---|---|---|---|---|---|---|
| C1 | Multimodal procedural QA | truncated-video QA: recipe DAG + ego video cut at query point + question text; no audio | free-form answer | **built on CC4D** (401 QAs, 24 recipes, 231 recordings; reuses its videos, recipes, error annotations) | GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet (44.1; human 74.5), Qwen2-VL-72B, Llama 3.1 | GPT-4o judge, 3-point ×50 | ✚✚ next/missing/order ≈ pure graph queries; question taxonomy maps 1:1 to our reminder families | ProMQA 2410.22211 |
| C2 | Procedural QA, assembly | truncated-video QA: task graph (DOT text) + parts image + multi-view video + question; no audio | free-form answer | built on **Assembly101** (646 QAs); releases **81 hand-made task graphs** | GPT-5 w/think (58.0), Gemini 2.5 Pro, Claude 3.7; human 70.7 | GPT-4o judge | ✚ graphs = graph-construction eval resource | ProMQA-Assembly 2509.02949 |
| C3 | Expert-dialogue VQA | offline MC-QA: 5 s ego clip before a *real* trainee question + 5 options; audio used only to author QAs | MC choice | **Ego-EXTRA** (own: 50 h Wizard-of-Oz, real experts coaching; bike/kitchen/bakery/assembly; ~15K QAs). Not CC4D | LLaVA-OneVision (33%), Qwen2.5-VL, text LLMs ≈ random; human 89.7 | accuracy | ○ but the **Pro-Active protocol's timestamped unprompted expert turns** are the most ecologically valid reminder GT found | Ego-EXTRA 2512.13238 |
| C4 | Industrial MD + reasoning QA + collaboration + summaries | offline per segment: ego(+exo) RGB with **audio**+gaze ablations + MC questions | binary mistake; MC answer; text summary | **IndEgo** (own: 3,460 ego recordings, 197 h, Aria glasses incl. audio; industrial tasks). Not CC4D | VideoLLaMA3, InternVL2.5, Qwen2.5-VL, Gemini 2.0 FT (best MD F1 ≈ .41) | P/R/F1 + severity F1; accuracy | ✚ audio helps in their ablations; keystep summary ≈ free byproduct of our step tracker | IndEgo 2511.19684 |

### Family D — Task-graph compilation (our "LLM compiler" competitors/seeds)

Not a runtime assistance task: this is the offline **authoring** problem — generating the machine-readable procedure (graph, steps, completion criteria, triggers) that Families A–C presuppose. In our pipeline: automating conversion steps C2–C4 of `CONVERSION_AND_EVAL_PROTOCOL.md`; the deferred compiler deliverable, with our hand conversions as gold references.

| # | Task | Input → output | Evaluated on | Notes | Source |
|---|---|---|---|---|---|
| D1 | Task graph learning (direct MLE on edge weights) | observed step sequences → DAG (partial order of key-steps) | graph quality on **CC4D** (+14.5% F1 in journal version), EgoPER, EgoProceL; downstream online-MD on Assembly101-O (+19.8%), Epic-tent-O, Ego-Exo4D | released machine-readable CC4D graphs — seed for our C1 step | TGML 2406.01486 (NeurIPS'24 spotlight) / 2502.17753 |
| D2 | How-to video → task assistant | YouTube how-to video → step list with demonstration details + **completion criteria**, then live smart-glasses monitoring | user study, 12 BLV participants cooking (−58.5% errors vs audio-only recipe) | closest published analog of our detector/trigger fields in task JSON | Vid2Coach 2506.00717 (UIST'25) |
| D3 | SOP text → structured plan | unstructured SOP text → decision-tree / logic graph; soundness checked **deterministically via PDDL** + LLM-judged completeness | SOP text corpora (no video) | citable precedent for recipe→task-JSON compilation + automatic precondition validation | SOPStruct 2504.00029 |
| D4 | Multiple-valid-next-action error detection | video stream → all valid next actions predicted + reconstruction-based comparison | **CC4D** and EgoPER (per surrounding literature; unverified from abstract) | really a Family-B method; listed here because it handles the branching ambiguity our precondition lists encode symbolically | AMNAR 2503.22405 (CVPR'25) |

### Family E — Step/stage tracking (the foundational substrate; added 2026-06-12)

Listed last to preserve cross-references, but logically first: every Family A/B task presupposes knowing where the user is in the procedure, and it is empirically the bottleneck (PREGO's oracle-recognizer ablation; ProAct's <13 mF1 step detection; PWR's oracle-plan condition lifting OOP recall 78.7→99.6%).

**Two formulations** — which coincide on clean runs and diverge exactly on error recordings:
- **Frame-centric**: a function t → current step label (incl. background/"other"). Tracks *execution*: what is happening now. Metric: per-second accuracy / MoF (duration-weighted — a 5 s-late boundary on a 60 s step costs only 8%).
- **Completion-centric**: a function t → ordered list of steps *completed* so far. Tracks *achievement*. Metrics: order similarity + per-step delay τ (each step weighted equally; latency explicit).
- **Relation**: on a clean recording, completion-centric ≈ frame-centric sampled at segment ends (a step completes when its frames stop). They diverge when execution ≠ completion: a step attempted but abandoned or done wrong produces frames but no completion (IndustReal's defining point — a step executed incorrectly must NOT be listed as completed; their baselines' FP rate on near-identical error states is 65%), and repeated attempts produce multiple frame segments but one completion. Our graph consumes completion predicates (preconditions); our sustained detectors (hum) are frame-shaped, our transient detectors (beep) are completion-shaped — we need both views, reporting per-second stage accuracy + delay τ.

---
**E1. Multi-Step Localization (MSL + RobustMSL) — CaptainCook4D (arXiv 2312.14556)**
- *Definition*: localize start/end of every step in an untrimmed recording and label it; **RobustMSL** trains on normal recordings only and reports normal-test vs error-test separately (robustness of step tracking to errors).
- *Setting*: **offline** (temporal action localization). *Input*: untrimmed video features. *Output*: temporal segments + step labels.
- *Evaluated on*: **CC4D** (environment/person/recording splits). *Baselines*: ActionFormer heads on 3D-ResNet, SlowFast, VideoMAE, Omnivore.
- *Metrics*: mAP and R@{1,5} at tIoU {0.1, 0.3, 0.5}. Paper itself notes task-graph priors + probabilistic filtering should help on error recordings.
- *SG-fit*: this IS our substrate task, offline flavor; our causal per-second variant is stricter.

---
**E2. Online step / keystep recognition — OadTR (ICCV'21) / MiniROAD (ICCV'23) / Ego-Exo4D procedure understanding (CVPR'24)**
- *Definition*: per-frame current-action/keystep label from past frames only; Ego-Exo4D's variant additionally predicts per-segment relational labels (preconditions met, optional, repeatable, procedural_mistake) **online**.
- *Setting*: **online**, frame-centric. *Input*: ego RGB stream. *Output*: per-frame step label (+ relational flags in Ego-Exo4D).
- *Evaluated on*: THUMOS/TVSeries (OAD tradition); **Ego-Exo4D** keysteps. MiniROAD is PREGO/TI-PREGO's recognizer — the de-facto standard online recognizer in this literature.
- *Metrics*: per-frame mAP / **calibrated AP (cAP)**; accuracy.
- *SG-fit*: the component our detector bank + graph state replaces; published numbers show it's the weakest link of VLM pipelines.

---
**E3. Procedure Step Recognition — IndustReal (cross-ref B6)**: the completion-centric formulation, with the only latency metric (delay τ) — see B6 for full entry.

**E4. Step + execution status — Pro2Assist (cross-ref A2)**: frame-centric step label + 4-way within-step status {just start, in progress, about to finish, transition}; Step-Acc 93.6 / Status-Acc 77.2 on their curated set — status is the harder half, and is what preventive (during-step) reminders need.

**E5. Procedure learning / key-step discovery — CC4D task + EgoProceL line**: self-supervised frame→key-step assignment across videos of the same recipe; CnC-style baselines; framewise P/R/IoU (low on CC4D, ~15 P, due to long steps). **Offline**, no annotations — relevant only as the zero-annotation lower bound.

Sub-task cross-references: WTaG step detection (inside A4), ProAct step detection (inside A6), EgoOops video-text alignment (inside B8).

## 2. What "standard procedure assistance tasks" means, operationally

The literature decomposes into five recurring task types; our system should be evaluated on the first four, with the fifth as the deferred compiler deliverable:

1. **Step/stage tracking** (**Family E**; also A1 substrate, B6, ProAct/WTaG step detection) — causal per-second step label (frame-centric) or completed-step list (completion-centric; see E preamble for when the two diverge). Our stage-accuracy metric already matches the frame-centric view; IndustReal's delay τ adds the completion-latency view.
2. **Online mistake/deviation detection** (B3, B5, B9–B11) — causal binary/typed deviation flag. Order + Missing-Step errors reduce to graph logic (PREGO line proves it); Timing/Temperature errors are *uniquely* audio+timer-detectable (our edge); Technique/Measurement/Preparation need escalation to VLM (matches our `expensive_criteria`).
3. **Proactive reminder/intervention timing** (A1–A8) — causal interrupt/silent (or typed reminder) with window-based scoring. Adopt: Pro2Assist STS over `reminder_windows` + PWR G-Mean F1 at decision points (anti-degenerate) + Ego-MC-Bench's 30 s-window intervention F1 as the cross-paper comparable.
4. **Guidance/reminder content** (A3, C1) — what to say. Ours is templated (closed reminder families), so LLM-judge content scoring is secondary; ProMQA (on CC4D!) doubles as content GT for next/missing/order reminders.
5. **Recipe → task-JSON compilation** (D1–D3) — deferred deliverable; hand conversions are gold references; TGML graphs + SOPStruct's PDDL check are the baselines/validators.

## 3. Gaps we occupy (defensible claims)

- **No work uses CC4D audio** — not even CC4D's citing papers (verified via Semantic Scholar sweep of all 39). CC4D ships GoPro audio + HoloLens mic; its own paper shows audio helps SupervisedER.
- **No online mistake-detection or reminder-timing benchmark exists on CC4D** (PREGO line went to Assembly101-O/Epic-tent-O; LiveMamba is closest — check its annotations).
- **No paper combines** explicit task-graph/precondition state + streaming when-to-intervene scoring + typed (parameter/precondition/safety) reminder taxonomy. PWR recovery is free-form; error-detection outputs are binary labels.
- **Every 2025–26 streaming entrant is an end-to-end MLLM** and they all report weak numbers (IA-QTF1 .368; step-inst 7.9 F1; G-Mean ≤ .51 zero-shot). None tries cheap detectors + symbolic graph; YETI is the only cheap-signal precedent and it has no task structure.

## 4. Metric kit (narrowed 2026-06-12; selection criteria: universal acceptance / newest online-proactive capability / instantiable on CC4D-with-audio)

Caveat declared once: no proactive-timing metric has ever been evaluated on CC4D or with audio — that IS our gap. Audio precedents to cite: OmniPro (audio-visual online F1 ±3 s) and CC4D's own offline ER with ImageBind audio variants.

### Core three (headline numbers)

| # | Metric | Why this one | From |
|---|---|---|---|
| M1 | **Windowed intervention P/R/F1** — TP iff fired within tolerance of GT event/window; silence on clean recordings scored. Report ±15 s (ours) + 30 s (cross-paper) | the only metric shape the field converged on: HoloAssist/YETI ±5 s, OmniPro ±3 s (audio-visual), Ego-MC-Bench 30 s, EgoPro-Bench intervals, our v1 ±15 s | convergent / our v1 |
| M2 | **G-Mean F1** = √(IF1·SF1) at fixed causal decision points (+ PQS-style unified score) | newest proactive-capability yardstick (2026, 6-dataset benchmark); degenerate policies → 0; the metric frontier VLMs publicly fail (GPT-5.2 ≤ .08) — clearest stage for detectors+graph to win | PWR/Pro2Bench |
| M3 | **Online mistake-detection F1, PREGO protocol** — one-class, strictly causal, mistake-class P/R/F1 (balanced precision per TI-PREGO) | the acknowledged online-MD standard (PREGO, TI-PREGO, TGML numbers exist); first instantiation on CC4D + audio arm = our claim | PREGO 2404.01933 |

### Secondary (diagnostics / supplementary columns, not headlines)

STS earliness score (Pro2Assist — free once windows exist); step-completion delay τ (IndustReal); earliness-vs-accuracy curve (MistExit); Point-of-No-Return deadline semantics (MATT); cost per video-minute incl. Inference Ratio / Proactive Hit Rate (ours + Pro2Assist) — cost is always reported but is our axis, not an acknowledged metric. Legacy anchor: one comparison row on CC4D's offline ER F1/AUC (the only existing audio-on-CC4D numbers).

## 5. Qualcomm Interactive Cooking Dataset — verified contents (inspected 2026-06-12)

`huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset` — the annotation layer behind LiveMamba (Bhattacharyya et al., "Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?", **NeurIPS 2025**, arXiv 2511.21998). Ungated; license = research-only "Data License Agreement"; **manually annotated** texts + timestamps; text-only parquet (~2.5 MB) referencing CC4D videos by `video_id` (e.g. `8_16`). Local copy: `data/qualcomm_interactive_cooking/` (gitignored).

### Two configs

**`main`** — **all 384 CC4D recordings** (train 213 / val 62 / test 109), all 24 recipes; 201 recordings have ≥1 mistake feedback, 183 clean. Per video: a message timeline, mean 28.8 messages. Columns: `video_id, activity_name, remaining_plan` (linear remaining step list before each message), `output_timestamps` (sub-second precision), `output_texts`, `output_types`, `output_actions` (mostly null). Type vocabulary:
- `instruction` — next-step instruction, issued at the moment the previous step completes;
- `feedback_action_aligned_success` — confirmation at step completion;
- `feedback_action_aligned_mistake_{technique|preparation|measurement|timing|temperature}_error` — timestamped mistake feedback, subtypes ≡ CC4D's execution-error taxonomy (no missing-step/order iisubtypes in `main`);
- `feedback_finish_all` — end of recipe.

**`advanced_planning`** — 302 recordings (209/57/36), adds `recipe_graph` (full DAG: `nodes_str_to_int/int_to_str`, `edges`, `recipe_text`, `action_description_cnt`; repeated actions disambiguated with `_1/_2` suffixes — i.e. machine-readable CC4D graphs, an alternative seed to TGML's) and `remaining_plan_w_suffix`. Crucially adds **`feedback_action_unaligned_{success|mistake_*}`** types: fired when the user performs a different step than instructed ("You are not following the instruction. It looks like you are trying to…") — i.e. **plan-deviation / order-deviation feedback**, the analogue of our graph-precondition reminders.

### What this gives us
1. **Truth-table extension for the error families our v1 declared out of scope**: human-annotated timestamps for Measurement / Preparation / Technique / Temperature (+ more Timing) mistake feedback — the "when did the error become visible" GT we lacked. C5 stays mechanical: map their typed, timestamped feedback events per recording.
2. **Step-completion cross-check**: `instruction` timestamps mark step boundaries independently of CC4D's own step annotations.
3. **Plan-state GT**: `remaining_plan` per message.
4. **The heavyweight baseline**: LiveMamba's protocol = generate these messages live; our sensor-graph arm can be scored on the same timeline at a fraction of the cost.
5. **Full activity-8 coverage** incl. our tuning recording 8_16 (their train) and 8_26/8_50 (train/test) — adopt their splits where possible to stay comparable.

### What is genuinely new vs CC4D's own annotations (worked example: 10_47, Pinwheels — verified by DAG replay)

CC4D already has step segments + error tags, e.g. step 111 `[31.6–59.2] scoop jelly` tagged `Preparation Error: use spoon instead of knife; Order Error: performed before spreading nut butter`. We replayed 10_47's GT step order against the pinwheels DAG and compared with the Qualcomm labels. Decomposition of their label structure (empirically verified, 12/12 unaligned events consistent):

- `aligned/unaligned` prefix = does the executed step match the assistant's **last instruction** (their adaptive instruction-policy state — derivable mechanically from GT step segments + a policy choice);
- `success/mistake_*` suffix = does the executed step carry a CC4D **execution-error tag** (Prep/Tech/Measure/Timing/Temp) — a direct re-attachment of CC4D tags;
- **order/graph validity plays no role**: `[212.3]` slide-floss violates the DAG (requires toothpicks+trim, not done) yet is labeled `unaligned_success`; there is no order or missing-step mistake type anywhere in their vocabulary.

So order-violation truth IS fully implied by CC4D's DAG + step segments (our C5 [mech] precondition check reproduces all of CC4D's order tags and more), and the Qualcomm layer adds **no order adjudication** — in fact its `unaligned_success` labels contradict our precondition-reminder semantics (slicing without toothpicks gets "success"). The v1 decision to keep Order-Error reminder adjudication as our own judgment step stands, unsolved by this dataset.

What survives as genuinely new:

1. **Sub-step mistake-visibility timestamps.** Mistake feedback lands at a specific moment *inside* the step segment — e.g. the spread-nut-butter technique error at **126.4 s** inside step `[118.4–153.2]`, ~27 s before step end. CC4D's granularity is the whole step window; this is "earliest moment a detector could reasonably fire" GT for execution errors. (Many other events land within seconds of step end, so the gain varies.)
2. **NL instruction/feedback texts + an adaptive reference policy timeline** — usable as content-scoring references and for protocol comparability with LiveMamba (same GT timeline, their model as the end-to-end baseline).
3. **Standardized splits over all 384 recordings.**

### Caveats
- **Interaction model differs**: theirs is a chatty coach (instruct before + confirm after every step); ours is silence-by-default with typed reminders. Use their *timestamps/types as GT events*, not their messaging policy; silence-scoring remains our protocol's.
- Feedback is **reactive** (after the mistake is visible) — our preventive triggers (overtime-during-run, precondition-before-action) remain a distinct, complementary output space.
- Annotations are text+timestamps only; **no audio used** — CC4D audio remains unclaimed.
- Research-only license: fine for experiments; check before redistributing derived truth tables.

### §5.1 Deep review (2026-06-12): coverage of the five reminder classes + annotation plan

Paper protocol (from full text): **IC-Acc** = fraction of instructions whose completion the model detects within a **30 s window** of GT completion; **mistake P/R/F1** same window; ROUGE-L/BERTScore on TP feedback only. Main set *explicitly excludes order errors and missing steps*; advanced_planning adds plan-divergence with a graph re-planner. **Headline numbers**: best zero-shot (Gemini-2.5-Flash) IC-Acc 23.1% / mistake F1 **0.02**; LiveMamba 31.5% / **0.13** — enormous headroom. No audio anywhere.

Event statistics (all 384 recordings, main config): 5,263 instructions, 4,067 success confirmations, **1,388 mistake events**: technique 33.1%, preparation 26.7%, measurement 24.5%, timing 11.8%, temperature 3.9% → **audio-leaning (timing+temp) = 15.7%; visual-leaning = 84.3%**. Mistake timestamps are genuinely sub-step: median 8 s before step end, 62% fire >5 s, 43% >10 s before the step closes. Timing/temperature events spread over ~20 of 24 recipes (absent only in the no-cook recipes).

Coverage of the five reminder classes:

| Class | Status on CC4D+Qualcomm | Manual work needed |
|---|---|---|
| 1. Next-step guidance | **Fully covered** — 5,263 instruction timestamps ≡ step-completion GT; IC-Acc protocol + published numbers directly adoptable; modality-neutral | none (windows trivially derived) |
| 2. Precondition violation | **Not covered** (main set excludes order/missing by design; unaligned labels ≠ adjudication, §5) ; CC4D order/missing tags + DAG give candidates mechanically | **adjudicate each DAG violation** benign vs reminder-worthy + window (the 8_50 judgment) |
| 3. Parameter violation | **Partial** — 218 timing/temperature events w/ visibility timestamps (reactive); preventive bound-crossing times derivable from step text | validate derived bounds; set window ends (PNR-style deadline where ambiguous) |
| 4. Execution error | **Richly covered** — 1,170 timestamped tech/prep/meas events; NOT audio-servable → exercises the escalation arm | none for events; windows = [visibility timestamp, step end + grace] |
| 5. Safety/unattended | **Absent everywhere** (not in CC4D taxonomy nor Qualcomm) | **sweep recordings** against the small fixed template list (appliance-done-unattended, stove-on-at-end, hot-handling), modality-neutrally defined |

**What Qualcomm missed (cross-match audit, CC4D error tags vs Qualcomm mistake events, type-matched within step ±10 s):**
- **Structurally absent (by design): 1,088 of 2,574 CC4D error instances (42%)** — 795 Order Errors + 285 Missing Steps + 8 Other have no Qualcomm counterpart of any kind. These are exactly our precondition-violation reminder class.
- **Dropped execution errors: 136 of 1,486 (9%)** — Preparation 13%, Temperature 18%, Technique 10%, Timing 8%, Measurement 2% missed; inspection shows borderline/subtle cases (uneven muffin cut, oversized ramekin) and some CC4D tag noise (a "2 tbsp instead of 1" salsa error tagged Timing). Their declared "remove noisy annotations" step.
- **Reverse direction: 97% of Qualcomm's 1,388 events match a CC4D tag** — they added almost no events beyond CC4D's own tags; their contribution is timestamps + phrasing, confirming §5.
- → **Our annotation layer therefore targets**: (a) the 1,080 structural errors — missing-step reminders are mechanical from the DAG (fire when a dependent step starts), order errors need the benign/harmful adjudication; (b) review of the 136 dropped execution errors with a documented keep/drop rule (some are a useful "hard/subtle" subset, some are CC4D tag noise to fix); (c) safety templates (absent from both); (d) preventive parameter windows (bound-crossing during the run; Qualcomm's timestamps are reactive-only).

Anti-audio-bias design (reviewer-proofing):
1. **Suite composition follows the data, not our detectors**: CC4D's natural error distribution puts only 15.7% of mistake events in audio-leaning classes — quote this number; the benchmark cannot be accused of being built around beeps.
2. **All five classes stay in scope** (v1's exclusion of execution errors is lifted now that GT exists); report **per-class P/R/F1, never pooled only**.
3. **Every arm runs on every class** — the audio-only arm's failure on execution errors is a reported result, not an omission.
4. **Adopt LiveMamba's own protocol** (IC-Acc + 30 s mistake F1) as one results table: evaluating on *their* task design removes suite-selection bias entirely; our added classes (2, 5) only extend it.
5. Audio event inventories (beeps/hums) are detector calibration data, never ground truth — keeps the truth table modality-blind.
6. Claim shape: "cheap-sensing-first **with escalation** matches/beats periodic VLM at equal class coverage and a fraction of the cost" — not "audio solves assistance."

---

## 6. Dataset task-structure formalisms (schema survey, merged from RELATED_WORK_TASK_STRUCTURES 2026-06-10)

How prior datasets *encode* a procedural task (orthogonal to the Family taxonomy above — that organizes by task type, this by annotation structure). Field names are verbatim from primary sources.

**A. Flat ordered step list.** CrossTask (ID, name, WikiHow URL, N, ordered step names; per-video `step#, start_s, end_s`); WTaG (recipe text + `StepDetection.txt` = `start, end, step` + Start/Done rows); Pro2Bench/EgoProactive (the most formal: plan **P = (s₁…s_N; c)** with `completed|current|next` status + free-text visual cues per step split into in-progress→Silent vs completed→Interrupt — the text form of our completion-evidence idea).

**B. DAG / task graph.** **CaptainCook4D** (per-recipe `{"steps":{node_id:instruction}, "edges":[[from,to]]}`; any topological order valid; step annos `{step_id, start_time, end_time, has_errors}`, skipped→`-1.0`; 8 error tags); EgoPER (per-task edge lists, scores predicted sequence by min edit distance over all topological orders; frame-wise step+error); Ego-Exo4D (graph as per-segment relational labels `pre_conditions/future_steps/missing_steps/is_optional/procedural_mistake`, online cAP); ProAct (AND/OR DAGs, `trigger∈{0,1}` decision points; unreleased); IndustReal (state machine, errors as first-class steps, PSR pins completion frame; F1+POS+τ); **Differentiable Task Graph Learning** (ships machine-readable CC4D graphs + unmet-precondition mistake detection — directly reusable).

**C. Hierarchy / taxonomy.** Ego4D Goal-Step (goal→step→substep; `is_relevant∈{essential,optional,irrelevant}` = cleanest prior for "which omission deserves a reminder"); COIN (domain→task→step, no order).

**D. Implicit (annotation-only, no recipe artifact).** HoloAssist (coarse/fine action ranges + `Action Correctness` 4-way + instructor `Conversation Purpose`); Assembly101 (per-segment `label∈{correct,mistake,correction}`, `remark∈{wrong order,wrong position,…}`).

**Pattern:** every structured dataset attaches **error semantics to the structure** (CC4D error tags, Ego-Exo4D `procedural_mistake`, IndustReal error-twin steps). None attaches *sensing* semantics — criteria tiers / sensor policies / escalation rules on the same graph are our addition (Box 3).

## 7. Paper corpus (`../related_work/`)

Core proactive-assistance: `plan_watch_recover_2606.04970`, `pro2assist_2605.04227`, `proactive_assistant_dialogue_generation_2506.05904` (ProAssist), `building_egocentric_procedural_ai_assistant_2511.13261`. Structure/datasets (added 2026-06-10): `captaincook4d_2312.14556`, `egoper_cvpr2024`, `holoassist_2309.17024`, `wtag_2311.00738`, `ego4d_goalstep_neurips2023`, `ego_exo4d_2311.18259`, `egoexolearn_2403.16182`, `crosstask_1903.08225`, `coin_1903.02874`, `assembly101_2307.16453`, `industreal_2310.17323`, `contextagent_2505.14668`, `yeti_2501.09355`, `promqa_2410.22211`, `promqa_assembly_2509.02949`, `egooops_2410.05343`, `indego_2511.19684`, `prego_2404.01933`, `ti_prego_2411.02570`, `differentiable_task_graph_learning_2406.01486`, `ego_extra_2512.13238`, `proact_2602.03430`, `egopro_bench_2605.07299`, `estp_2510.14560`, `omnipro_2605.18577`, `parse_ego4d_2407.09503`. Audio (2026-06-11): `epic_sounds_2302.00646` (grounds `research/AUDIO_LIBRARY.md`).
