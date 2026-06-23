# eval/ — evaluation scripts, grouped by role

Filenames are prefixed by their role: `proposed_*`, `baseline_*`, `legacy_*`, `eval_*`,
`gt_*`, `probe_*`. Two generations live here: the older activity-8 `replay_v1` pilot
(`legacy_*` + `eval_score_activity8.py`) and the current Family-A `proposed_system`
(`proposed_*` + `eval_score_corpus.py`).

## Group 1 — Proposed approach (`proposed_system` runtime)
Plan-driven graph state machine over cheap detectors, with a cost-capped VLM arm.

| # | File | Role |
| --- | --- | --- |
| 1.1 | `proposed_runtime.py` | **Entry point.** Loads a compiled plan, runs detectors once, releases events causally, drives the graph state machine on a fixed tick, optionally polls the VLM arm in the C-none block. Writes `experiments/proposed_system/results/<rec>.json` (stage_intervals + transition_trace + sensor_events + cost). |
| 1.2 | `proposed_plan_loader.py` | lib — compiles a plan JSON (`tasks/cc4d/*.monitor.json`, schema 0.6) into predicate closures + typed `state_update` ops; binds `D*` primitives to `detectors/runtime.REGISTRY`. |
| 1.3 | `proposed_vlm_arm.py` | lib — the VLM arm. Polls a C-none block (cadence + cost capped by `vlm_policy`) to label silent members. Modes: `qwen` (real, saltyfish), `mock` (offline). Reuses `baseline_t1_step.py` for the Qwen client + frame sampling. |
| 1.4 | `proposed_verify.py` | self-check — per-second agreement of the runtime stage track vs the `legacy_detector_replay.py` oracle, plus GT block-coarse accuracy. |

```bash
QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
python eval/proposed_runtime.py \
  --plan tasks/cc4d/spicedhotchocolate.monitor.json \
  --recs 8_16,8_3,8_25,8_26,8_31,8_50 --vlm qwen --trace
```

## Group 2 — Baselines

| # | File | Role |
| --- | --- | --- |
| 2.1 | `baseline_t1_step.py` | **T1 baseline** — online current-step recognition (T1 only), RGB-only, Pro2Assist-style. Every N s: 1-fps frames + recipe + completed steps + own recent responses → current step. Writes the unified per-recording JSON. Also imported by `proposed_vlm_arm.py` (1.3). |
| 2.2 | `baseline_periodic_vlm.py` | periodic-VLM baseline (full T1+T2): every N s send frames + task context, strict-JSON step/action output, K-consecutive smoothing, action cooldown. Writes `calls.jsonl` + `summary.json`. |
| 2.3 | `baseline_periodic_vlm_convert.py` | plumbing — converts 2.2's `summary.json` into the unified result format (`periodic_vlm_qwen` arm). |

```bash
# T1 baseline over the corpus
python eval/baseline_t1_step.py --corpus --interval 10 --arm qwen36_i10
```

## Group 3 — Legacy proposed-system (`replay_v1`, superseded by Group 1)

| # | File | Role |
| --- | --- | --- |
| 3.1 | `legacy_detector_replay.py` | detector-only replay of a **hardcoded** activity-8 chain. `proposed_runtime.py` generalizes this into the predicate-driven loop. Still serves as the oracle for `proposed_verify.py`. |
| 3.2 | `legacy_escalation.py` | detector replay + the sparse VLM calls the detector arm requested (one Qwen call per escalation). Produces the `detector_plus_escalation` arm. |

## Group 4 — Evaluation / scoring

| # | File | Role |
| --- | --- | --- |
| 4.1 | `eval_score_corpus.py` | **Main referee (Box 2).** Scores any arm dir over the full CC4D Family-A corpus: FA-1 P/R/F1, FA-2 G-Mean, T1 active-step accuracy, cost. Reads `<results-dir>/<arm>/<rid>.json`. |
| 4.2 | `eval_score_activity8.py` | activity-8 pilot referee for the 3-arm `replay_v1` experiment. Welded to `experiments/replay_v1`. |

```bash
python eval/eval_score_corpus.py --results-dir experiments/proposed_system/results --arms proposed_system
```

## Group 5 — Ground-truth generation (Box 1)

| # | File | Role |
| --- | --- | --- |
| 5.1 | `gt_build_family_a.py` | mechanical proactive-reminder GT / answer-key builder from CC4D + Qualcomm annotations. Firewalled from the predictor (Box 3). Writes `data/cc4d_family_a/`. |

## Group 6 — Prior-work probes

| # | File | Role |
| --- | --- | --- |
| 6.1 | `probe_prior_work.py` | re-implements LiveMamba (IC-Acc + windowed P/R/F1) and PREGO-style protocols to situate our arms against published numbers. Writes `experiments/probe_prior_work/`. |

---

### `replay_v1` pilot run order (legacy)
`legacy_detector_replay.py` → `baseline_periodic_vlm.py` (×6 recordings) +
`baseline_periodic_vlm_convert.py` → `legacy_escalation.py` → `eval_score_activity8.py`.

### `baseline_periodic_vlm.py` knobs
`--interval` (s between calls), `--frames-per-call`, `--window` (s the frames span),
`--k-consecutive` (stage-switch smoothing), `--cooldown` (min s between assistant actions),
`--max-seconds`. Task definitions live in `../tasks/`; the unified result format and the
reminder truth table are documented in `eval_score_corpus.py` and `../docs/REMINDER_EVALUATION.md`.
