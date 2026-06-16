# ProceduralAgent Task Visualizer

Two-server setup mirroring `kitchen/self_data/annotator`:

- **`video-server.js` (port 4002)** — serves videos from
  `../data/{videos_480p,clips,videos_360p}` with HTTP range support, plus JSON
  APIs over the full CC4D annotation set (`../data/cc4d/annotations/`) and the
  hand-written `../tasks/*.json`. Dependency-free (plain `node:http`).
- **`frontend-server.js` (port 3010)** — static server for `public/`
  (vanilla JS, no build step).

## Run

```bash
./START.sh
# then open http://127.0.0.1:3010
```

## What it shows

Covers **all 384 CC4D recordings across 24 recipes** (plus the HD-EPIC clip).
The six activity-8 recordings are served from `videos_480p`; everything else
from `videos_360p`; duplicate ids across sources are de-duped (480p wins). The
HoloLens-only `12_6_hololens_pv.mp4` is mapped to recording `12_6`'s GT.

Header controls:
- **Recipe filter** — narrow the list to one recipe (with per-recipe counts).
- **errors only** — show only the 220 recordings with annotated errors.
- **Video dropdown** — grouped into `<optgroup>`s by recipe; each entry flags
  `⚠N` error-step count, `480p`, hand-written `ann`, and `qc`/`qc⚠N` for the
  Qualcomm guidance layer (with its typed-mistake count) where present.

For the selected video:
- Video player with click-to-seek timeline and playhead.
- Timeline layer toggles for **GT**, **Baseline VLM**, **Proposed monitor**, and
  **Reference** views. GT / Baseline / Proposed are enabled by default; the
  Qualcomm reference layer is opt-in to keep the view readable.
- Timeline tracks:
  - **Stages (ann)** / **Reminder windows** / **Mistakes** — from a
    hand-written `tasks/annotation_*.json` (activity-8 + HD-EPIC clip only)
  - **GT steps** — CC4D step segments (joined from `complete_step_annotations`),
    labeled with a short description; dashed outline = step has annotated
    errors; steps with `start_time == -1` were not performed and are skipped
  - **GT errors** — per-step error tags (Technique/Preparation/Timing/...)
  - **Qualcomm steps** — the Qualcomm Interactive Cooking guidance layer (the
    live-assistant annotation behind LiveMamba): one span per `instruction`
    message, running to the next instruction (or `finish_all`)
  - **Qualcomm feedback** — point markers on that layer: ✓ success
    confirmations (thin green) and ⚠ typed mistakes (red — technique /
    preparation / measurement / timing / temperature). A mistake's timestamp
    is when it first becomes *visible* (median ~8 s before the step ends), so
    it sits earlier than the corresponding CC4D error span
  - **Baseline VLM** — periodic-VLM stage predictions and call/action markers,
    loaded from either `experiments/t1_baseline/` or periodic-VLM arms in the
    newer `experiments/replay_v1/` results.
  - **Proposed monitor** — sensor schedule, sparse VLM calls, monitor stages,
    sensor events, and state transitions from `experiments/proposed_system/`.
- Side panel:
  - recording metadata chips (activity id, person, environment,
    normal/error, recipe-graph size)
  - **Steps** — the performed GT segments with cleaned descriptions, time
    ranges, inline error tags, and `actual:` modified-description; click to
    seek; the active step is highlighted at the playhead
  - **Reminders** — proactive triggers from the hand-written task (hidden when
    none)
  - **At playhead** — error tags / reminder windows / mistakes active now

### Task JSON view

The **Task JSON** tab in the header browses the raw `tasks/*.json` files —
every task definition, annotation, and the annotation template — rendered as
a collapsible JSON tree (expand/collapse-all buttons included). The side
panel in the video view links directly to the task / annotation file backing
the current video.

## API

- `GET /api/videos` — all videos with duration (from `video_information.csv`),
  recipe/person/environment, and error flags
- `GET /api/tasks` — parsed `tasks/task_*.json`, keyed by `task_id`
- `GET /api/taskfiles` — every JSON file in `tasks/` with kind/task_id/video_id
- `GET /api/taskfiles/:filename` — raw content of one `tasks/` file
- `GET /api/timeline/:videoId` — bundled
  `{meta, task, task_graph, annotation, gt, qualcomm}` for one video
- `GET /videos/:source/:filename` — video stream (range requests supported)

GT comes from `data/cc4d/annotations/annotation_json/` —
`complete_step_annotations.json` (step segments + recipe/person/env) joined
with `error_annotations.json` (per-step error tags + modified descriptions).
The recipe DAG (`task_graphs/<recipe>.json`) is matched by
`activity_name.toLowerCase().replace(/[^a-z0-9]/g,'')`. Where a hand-written
`tasks/task_*.json` exists (activity 8), it's preferred for the richer
reminders/expected-durations and its GT steps are linked via the
`cc4d_step_ids` block. `annotation_template.json` (unfilled `video_id`) is
ignored.

The Qualcomm layer ships as parquet (`data/qualcomm_interactive_cooking/main/`),
which the dependency-free server can't read. `scripts/export_qualcomm_timeline.py`
pre-flattens all three splits into
`data/qualcomm_interactive_cooking/qualcomm_timeline.json` (keyed by `video_id`
== `recording_id`), and the server reads that. The per-recording derivation
mirrors `eval/probe_prior_work.py:load_qualcomm` (instructions, derived
completions, success confirmations, typed mistakes). Re-run the script after
refreshing the dataset.
