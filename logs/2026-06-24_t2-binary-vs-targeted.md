# T2 reminder detection — framing head-to-head: "any error?" vs "did THIS mistake happen?"

- **Date:** 2026-06-24
- **Author:** claude-opus-4-8 (Claude Code session)
- **Scope:** The binary-vs-targeted head-to-head on **spiced hot chocolate** only — does naming the
  specific mistake beat a generic "is there an error in this step?", with the setup otherwise held
  fixed. Excludes: the earlier survey-mode smoke test, the 8-frame/1-fps sampling smoke tests, and
  the standalone targeted upper-bound run (referenced for context, not the subject here).
- **Code:** `eval/baseline_t2_reminder.py` (`--mode binary`, with `run_binary` / `run_targeted`)
  @ `6562aea` — **working tree dirty**: `eval/baseline_t2_reminder.py` has uncommitted changes
  (the T2 harness was authored this session and is not in that commit).

## Question
We observed that telling the VLM the exact mistake to look for ("did THIS happen?") gave ~0.44
recall, far above the ~0.14 of open-ended flagging. Is that gain real and broad, or an artifact of
one comparison? Hold the data, model, and time windows fixed; change **only the question**, on the
same step units, and measure both.

## Setup
- **Data:** recipe `spicedhotchocolate` (CC4D activity 8), all 16 recordings. GT = per-step
  execution-error events from `data/cc4d_family_a/<rec>.json` (classes `execution_error` +
  `parameter_violation/timing`; `order`/`missing_step` excluded). Per-step time windows from
  `data/cc4d/annotations/annotation_json/step_annotations.json`. Videos `data/videos_360p/8_*.mp4`.
- **Model:** `Qwen/Qwen3.6-27B`, served at `http://saltyfish.eecs.umich.edu:8000` — verified live
  via `/v1/models` (server reports exactly this one id).
- **Evaluation unit:** `(recording, step)` — **56 units, 42 positive** (step has ≥1 execution error),
  14 negative. Units are the exact set the targeted run touched, so both arms score identical units.
- **Windowing:** `oracle` — frames sampled over the GT step span (start→end). This is a detection
  *ceiling*, not deployable (a real system lacks GT step boundaries).
- **Parameters:** `--sample-fps 1.0 --max-frames 32`; 360p resized to max-dim 768, JPEG q85.
- **Two framings compared at step level:**
  - **binary_generic** — one call/step: "Did the person make ANY mistake while performing this
    step?" No specific mistake named, **no observed-error leak**.
  - **targeted_named** — the per-subtype "did THIS specific mistake happen?" calls (~k per step,
    one per authored check subtype), OR-aggregated to a step verdict. Reused from the targeted run.
- **Command(s):**
  ```bash
  # targeted run (produced the per-subtype calls + the (rid,step) unit set), 82 calls
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python3 eval/baseline_t2_reminder.py --recipe spicedhotchocolate --vlm qwen --mode targeted \
    --sample-fps 1.0 --max-frames 32 --neg-per-cell 2

  # binary head-to-head (this run): 56 generic calls on the same units, reuses targeted calls.jsonl
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
  python3 eval/baseline_t2_reminder.py --recipe spicedhotchocolate --vlm qwen --mode binary \
    --sample-fps 1.0 --max-frames 32
  ```
- **Inputs consumed:** `experiments/baseline_t2/spicedhotchocolate_qwen_targeted/calls.jsonl`
  (unit set + targeted per-subtype verdicts), `step_annotations.json`, `data/cc4d_family_a/8_*.json`.

## Results
Step-level, same 56 units (42 positive). Numbers read from
`experiments/baseline_t2/spicedhotchocolate_qwen_binary/summary.json`.

| Framing | TP | FN | FP | TN | Recall | False-alarm | Precision | F1 | Calls/step |
|---|---|---|---|---|---|---|---|---|---|
| **Binary generic** ("any error?") | 9 | 33 | 1 | 13 | **0.214** | 0.071 | 0.900 | **0.346** | 1 |
| **Targeted named** (per-subtype, OR'd) | 19 | 23 | 3 | 11 | **0.452** | 0.214 | 0.864 | **0.594** | ~k |

Overhead (binary run): 56 calls, 1666 frames, 989 s wall (~18 s/call). Targeted run for reference:
82 calls, 2405 s. Targeted per-subtype recall (from its summary): timing 0.60, technique 0.50,
temperature 0.50, measurement 0.40, preparation 0.22.

## Interpretation
Holding everything fixed but the question, **naming the specific mistake roughly doubles recall
(0.21 → 0.45) and lifts F1 (0.35 → 0.59)** — the generic "is there an error here?" catches only ~1
in 5 errors even with a perfect step window; the pointed per-subtype questions catch ~1 in 2. This
supports the design claim that a duty-cycled VLM call should ask a *specific* "did X happen?",
driven by a cheap detector that picks X — not an open-ended "what's wrong?". It is a trade, not free:
recall rises but so do false alarms (0.07 → 0.21) and cost (~k× more calls); precision stays similar
(~0.9), so the extra detections are mostly real.

## Caveats & limits
- **Leak confound (important):** the targeted prompts still contain the `(observed: …, N recordings)`
  clause from the criteria; the binary prompt does **not**. So this compares *(naming + leak)* vs
  *(generic, no leak)* — part of the 2× may be the leak, not the naming. A leak-stripped targeted
  re-run is needed to attribute the gain to naming alone.
- **Detection ceiling, not deployable:** oracle step windows give the VLM the right step identity and
  boundaries for free; a real system must infer these (T1 recognition / detector firing) with noise.
- **In-sample:** the criteria checks were probe-derived from these same recordings.
- **Small / single-recipe:** n = 56 units (42 positive), one recipe (microwave-heavy, so timing/
  temperature are over-represented relative to the corpus).
- **Asymmetric cost:** targeted spends ~k calls/step vs 1; the OR over k calls also gives it more
  chances to false-fire (part of the higher false-alarm rate).
## Artifacts
- `experiments/baseline_t2/spicedhotchocolate_qwen_binary/summary.json` — this head-to-head's scores.
- `experiments/baseline_t2/spicedhotchocolate_qwen_binary/calls.jsonl` — per-(rid,step): gt,
  binary_pred, targeted_pred, frames, latency.
- `experiments/baseline_t2/spicedhotchocolate_qwen_targeted/{summary,calls}.json` — the targeted run
  (per-subtype verdicts + unit source).
