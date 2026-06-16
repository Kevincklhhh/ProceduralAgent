# ProceduralAgent

Smartglass procedural assistant research: compile a recipe/procedure into an **executable sensing-and-assistance graph** — each stage carries completion-event detectors (cheap audio/DSP → medium RGB → expensive VLM), graph preconditions, and proactive-reminder triggers — then show that this graph tracks the user and issues reminders **better and far cheaper** than periodically calling a VLM. Current scope: RGB + audio only, cooking, CaptainCook4D as primary benchmark.

Project concept and rationale: [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md) (this project) and [docs/multi-sensor/PROJECT_MEMORY.md](docs/multi-sensor/PROJECT_MEMORY.md) (parent multi-sensor idea).

## Directory map

| Directory | What lives there |
| --- | --- |
| `docs/` | Research docs: task definition, detector feasibility study, related-work survey, project memory. `docs/multi-sensor/` = parent-project notes. |
| `related_work/` | Paper PDFs (30+) + extracted text for key ones. Index: [docs/RELATED_WORK_TASK_STRUCTURES.md](docs/RELATED_WORK_TASK_STRUCTURES.md) |
| `tasks/` | Task-graph JSONs (the compiled "sensing graph" per recipe) + ground-truth annotation templates. Schema: [docs/TASK_DEFINITION.md](docs/TASK_DEFINITION.md) |
| `detectors/` | Detector library (`detectors_lib.py` — frozen, validated audio detectors) and `probes/` (the feasibility experiments that validated them, scripts + results JSONs + plots) |
| `eval/` | Main evaluation scripts (see "How to run") |
| `experiments/` | Run artifacts and reports. `replay_v1/` = the 3-arm detector-vs-VLM experiment ([REPORT.md](experiments/replay_v1/REPORT.md)); `hdepic_eggs_v0/` = first baseline sanity runs on an HD-EPIC clip |
| `data/` | **gitignored, 97 GB** — datasets and derived media: `cc4d/annotations` (cloned CC4D annotations repo), `cc4d/downloader` (cloned downloader + six 4K videos), `videos_360p/` (all 383 GoPro 360p recordings + `12_6_hololens_pv.mp4`, HoloLens-only/no-audio), `audio/` (16k/48k wavs for all 383 GoPro recordings), `videos_480p/`, `clips/`, `gt_activity8.json` |

## What we have run

| # | Experiment | Where | Outcome |
| --- | --- | --- | --- |
| 0 | Periodic-VLM baseline sanity check on an HD-EPIC scrambled-eggs clip (Gemini 9/18, Qwen 11/18 stage acc) | `experiments/hdepic_eggs_v0/` | Baseline works end-to-end; established Qwen needs guided JSON |
| 1 | Detector feasibility probes — 5 probes on six CC4D Spiced-Hot-Chocolate 4K recordings vs GT, thresholds tuned on clean run 8_16 only, frozen, evaluated on the other five | `detectors/probes/`, report: [docs/DETECTOR_FEASIBILITY.md](docs/DETECTOR_FEASIBILITY.md) | DSP hum+beep works (11/12 runs, 0 false); AST/CLAP localize microwave; RGB global-motion stirring FAILED |
| 2 | Task inventory: all 24 CC4D recipes classified by detector-coverable steps; cross-dataset audio availability survey | [docs/DETECTOR_FEASIBILITY.md](docs/DETECTOR_FEASIBILITY.md) §1–2 | Spiced hot chocolate ranks #1; most proactive-assistance datasets ship NO audio (EgoProactive, IndustReal, Assembly101 ...) |
| 3 | **Replay experiment v1** — 3 arms × 6 recordings: `detector_replay` (audio DSP + graph, 0 VLM), `periodic_vlm_qwen` (call every 10 s, 229 calls), `detector_plus_escalation` (DSP + exactly 1 VLM call/recording) | [experiments/replay_v1/REPORT.md](experiments/replay_v1/REPORT.md) | Detector graph: 67.2% coarse stage acc, reminder R=50% (71.4% with escalation) at ~0 cost; periodic Qwen: 37.8%, R=0%, 5.13× real time |

## Audio models / detectors: what we used, I/O, what they did well

| Detector / model | Input | Output | Did well | Failed / weak |
| --- | --- | --- | --- | --- |
| **Microwave hum + beep** (classical DSP: 100–1000 Hz band level + 120 Hz mains-line contrast + spectral stationarity, 2-of-3 vote; beep = narrowband tonal bursts 0.8–5 kHz) | 16 kHz mono wav | hum runs `[onset, offset, duration]`, beep times | 11/12 GT microwave runs, **0 false runs**, ~490× real-time on one CPU core; drives timing-error warnings (2/2 headline flags) | low-SNR onset truncation (8_25 false undertime); missed one low-power run (8_50 heat); beep fusion repairs offsets only |
| **Pour + clink** (classical DSP: 1–8 kHz band-energy bursts; ringy-onset trains via high-band spectral flux) | 48 kHz mono wav | pour bursts; clink trains (weak + strong tier) | recall (pour 6/6, clink 5/5); strong-tier clink = usable stir/mix cue; ~0.5% real-time | precision — pour 0.5–4 FA/min; weak evidence only, must be gated by graph state |
| **AST** `MIT/ast-finetuned-audioset-10-10-0.4593` (86 M, supervised AudioSet tagger) | 16 kHz, 5 s windows, 2.5 s hop | 527 AudioSet class sigmoids per window | 'Microwave oven' AUC 0.95–0.98 clean / 0.87 pooled, zero training; **~18 ms/window on GPU** (3–7 s per 450 s recording) | tiny absolute scores → brittle fixed thresholds; no signal for dry-ingredient adds; pour weak (AUC 0.63) |
| **CLAP** `laion/clap-htsat-unfused` (zero-shot audio-text) | 48 kHz, 5 s windows + text prompts ("a microwave oven running", ...) | per-window prompt similarities | microwave AUC 0.982 on tuning run; **excellent zero-shot segment labeler** (right prompt ranks first on every microwave/add/mix GT segment) → a task-JSON compiler can emit text prompts for loud sustained events | thresholds don't transfer across rooms (pooled 0.819); quiet transients at chance (pour 0.515) |
| **RGB global motion** (Farneback flow, median ego-compensation, 0.5–3 Hz periodicity; no model) | 320p @ 5 fps video | residual-motion + periodicity timeline | activity-level sanity on the clean run; cheap (~13 ms/frame CPU) | **FAILED for stirring** (AUC ≈ chance; head sway wins). Fix path: ROI-restricted flow after one-shot grounding, 10–15 fps bursts |
| **Qwen3.6-27B** (VLM, vLLM at `saltyfish.eecs.umich.edu:8000`) | (a) periodic: 3 frames over last 2 s every 10 s + task context → (b) escalation: 10 frames spanning `[0, t_esc]` | strict JSON — (a) `{step_id, status, confidence, evidence, hazard, action}` (b) `{chocolate_added, cinnamon_added, sugar_added, evidence, confidence}` | escalation recall 4/4 truly-missing ingredients with 1 call/recording | ~44–54 s/call (periodic policy = 5.13× real time, unrunnable live); periodic reminders 0/7; escalation precision 30.8% ("not visible" → "missing"; chocolate falsely flagged 5/6) |

Always use Qwen on saltyfish (`QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000`, model `Qwen/Qwen3.6-27B`, `response_format json_object`, `max_tokens 2000`), not Gemini.

## How to run

```bash
# Arm 1: detector-only replay (audio DSP + task graph) -> experiments/replay_v1/results/detector_replay/
python3 eval/engine.py

# Arm 2: periodic VLM baseline (one recording)
export QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000
python3 eval/periodic_vlm.py --video data/videos_480p/8_16.mp4 \
    --task tasks/task_spiced_hot_chocolate_cc4d.json --backend qwen \
    --interval 10 --frames-per-call 3 --out experiments/replay_v1/runs/8_16
python3 eval/convert_periodic_vlm.py        # -> unified results format

# Arm 3: escalation (needs arm-1 results + 480p videos)
python3 eval/run_escalation.py

# Score all arms -> experiments/replay_v1/results/scores.json + REPORT.md
python3 eval/score.py
```

## Data (rebuilding `data/`, which is gitignored)

- 4K videos: `data/cc4d/downloader/metadata/download_links.json` has the Box URLs (`gopro_4k` key per `{activity}_{recording}`); the six activity-8 recordings used are 8_16, 8_3, 8_25, 8_26, 8_31, 8_50.
- 360p + wavs for ALL recordings: `python3 scripts/download_cc4d_360p.py` (idempotent, ~39 GB video + ~41 GB wav). 360p carries the identical AAC track as 4K (verified on 8_16, detector outputs byte-identical). `12_6` is HoloLens-only: no GoPro video at any resolution, no audio in the public release — `videos_360p/12_6_hololens_pv.mp4` is the video-only PV stream.
- Wavs: `ffmpeg -i {rec}_4K.mp4 -vn -ac 1 -ar 16000 data/audio/{rec}_16k.wav` (and `-ar 48000` for `_48k.wav`).
- 480p copies: `ffmpeg -i {rec}_4K.mp4 -vf "scale=854:480" -c:v libx264 -preset veryfast -crf 26 -an data/videos_480p/{rec}.mp4`.
- `data/gt_activity8.json`: extracted from `data/cc4d/annotations/annotation_json/{complete_step_annotations,error_annotations}.json` for the six recordings (steps + per-step error tags).
