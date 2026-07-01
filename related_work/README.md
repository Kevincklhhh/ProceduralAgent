# Related Work Corpus

This directory groups papers by the role they play in the project, not by venue or chronology.

## 01_core_methods

Directly relevant systems and methods for proactive procedural assistance, online guidance, streaming intervention, or selective VLM use. These are the papers most likely to appear as baselines, close comparisons, or direct contrasts in the main related-work section.

Representative papers: PWR, Pro2Assist, ProAssist, InvAgent, ContextAgent, ProAct, YETI, WTaG, Eyes Wide Open.

## 02_benchmarks_datasets

Datasets and benchmarks that define evaluation settings, annotations, user-assistance traces, or egocentric procedural corpora. These papers are mostly cited to justify dataset choice, benchmark coverage, or ground-truth construction.

Representative papers: CaptainCook4D, HoloAssist, EgoPro Bench, Ego-EXTRA, EgoPER, Ego-Exo4D, Assembly101, IndEgo, OmniPro, HD-EPIC.

## 03_task_structure_mistake_detection

Papers that define task state, step completion, mistake semantics, online mistake detection, or procedure-consistency checks. These are useful for explaining what counts as a potential mistake and how prior systems compare observations against expected procedure progress.

Representative papers: PREGO, TI-PREGO, IndustReal, EgoOops, PARSE-Ego4D.

## 04_sensing_modalities

Audio, visual, and audio-visual sensing papers that justify cheap detector design, egocentric action cues, and multimodal fusion. These support the sensing side of the system rather than the assistant policy itself.

Representative papers: EPIC-SOUNDS, EPIC-Fusion, OWL, Seeing and Hearing Egocentric Actions.

## 05_procedural_qa_background

Broader procedural-video understanding, step localization, and multimodal QA papers. These are background for procedural representation and evaluation tasks, but are not the closest proactive-assistance baselines.

Representative papers: COIN, CrossTask, Ego4D GoalStep, EgoExoLearn, ProMQA, ProMQA-Assembly.

## 06_surveys_overviews

Survey and landscape papers used as maps of the broader egocentric procedural-assistant area.

Representative papers: Building Egocentric Procedural AI Assistant.

## 07_vlm_latency

VLM latency, streaming-video, visual-token caching, token pruning, and serving/runtime papers. These papers support the MLSys angle: reducing event-to-response latency by caching or pre-encoding visual evidence, maintaining streaming memory, pruning visual tokens, or reusing serving-state rather than calling a full VLM over raw video at every decision point.

Representative papers: VLCache, STC-Cacher, VideoLLM-online, VideoStreaming, Flash-VStream, StreamingVLM, Dispider, MMInference, FastVLM, LLaVA-Mini, EVS, StreamingAssistant, SpecVLM, vLLM, SGLang.

## Missing But Important

`2511.21998` / LiveMamba / Qualcomm Interactive Cooking is already discussed in project docs and eval notes, but the paper file is not currently stored here. Once added, it should likely live in `01_core_methods/` if treated as an end-to-end model baseline, with a note in `02_benchmarks_datasets/` if the Qualcomm Interactive Cooking annotation layer is cited as a dataset.
