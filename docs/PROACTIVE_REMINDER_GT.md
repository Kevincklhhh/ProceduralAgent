# Proactive-Reminder GT (`cc4d_proactive`)

Date: 2026-06-28. Supersedes the `cc4d_family_a` / Family-A scheme (its design doc has been
removed; see git history).

## What this is

Per-recording ground truth for **T2 — proactive reminders**: for each CC4D video, the
**time** and **content** of every reminder a silent-by-default assistant *should* have
spoken. One JSON per recording in `data/cc4d_proactive/`, built mechanically by
`eval/gt_build_proactive.py`. This is **Box 1** (the answer key) and is firewalled from the
recipe→sensor predictor (Box 3): never feed it to the predictor — see
`PIPELINE_THREE_BOXES.md`.

## Scheme: event detection, not decision points

The old `cc4d_family_a` carried two parallel framings — a flat event list **and** a
per-step-boundary `interrupt`/`silent` decision-point classification (plus an `adjudication`
field). All of that is gone. This GT is **pure event detection**:

- only **positive** reminder events are annotated;
- **silence is implicit** — any predicted reminder that matches no GT reminder is a false
  alarm; no `silent` labels, no decision points, no `adjudication`.

This matches `REMINDER_EVALUATION.md` ("Do not densely annotate silence").

## Scope (this version)

**Execution mistakes** (technique, preparation, measurement, timing, temperature) live in
`data/cc4d_proactive/` (this doc, `eval/gt_build_proactive.py`).

**Order and missing-step** reminders are now built — **deterministically, no LLM** — by
`eval/gt_build_om.py` into a SEPARATE directory `data/cc4d_proactive_om/{rid}.json` (189
recordings: 151 order + 183 missing). Policy: order = DAG precondition-edge violations only
(CC4D's text sequence ignored; DAG-legal reorderings dropped); missing = first executed
transitive DAG-dependent's start; content is templated. See `ORDER_MISSING_CONVERSION.md`.
Schema is **flat + group metadata**: each reminder has `t`/`subtype`/`content`/`source` plus
`members`/`opportunities`/`episode_span`/`one_shot`. Kept separate from the execution GT until
a merge is approved.

**Advisory usefulness audit** (`data/cc4d_proactive_om_audit/`, `eval/gt_write_om_audit.py`):
an LLM judges each mechanically-built reminder against the original annotations + DAG
(`makes_sense`/`useful`/`actionable_at_t`/`severity`/`suggestion`). It is **advisory only** —
never mutates the GT (preserves reproducibility + firewall). Corpus audit (334 reminders):
188 ok / 123 minor / 23 major; missing-step reminders 89% useful, order reminders 64% useful
(many DAG violations are functionally-harmless technicalities or fire too early/late). The
audit also surfaced a correctness bug — 8 steps tagged CC4D "Missing" yet actually executed —
now excluded by `gt_build_om.py` (added to `dropped`).

Corpus: **1,383 reminders over 200 recordings; 184 clean** (no reminder). By subtype:
technique 453, preparation 360, measurement 325, timing 178, temperature 67. By source:
`qualcomm` 1,355, `cc4d_only` 28.

## Schema

```json
{
  "recording_id": "8_50",
  "activity_name": "Spiced Hot Chocolate",
  "recipe": "spicedhotchocolate",
  "duration_s": 251.7,
  "is_error": true,
  "reminders": [
    {
      "id": "8_50_r1",
      "t": 184.3,                                  // reminder time — a SINGLE timestamp
      "subtype": "technique",
      "content": "You spilled one piece of chocolate.",
      "anchor_step": 90,
      "source": "qualcomm"
    }
  ]
}
```

- **`t`** — the reminder time, a single timestamp (the Qualcomm mistake-visibility moment,
  or a mechanical CC4D anchor in the fallback case). **No window / tolerance is stored** —
  matching tolerance belongs in the evaluation script, not the answer key.
- **`content`** — the reminder text.
- **`subtype`** — the scored class.
- **`anchor_step`** — the CC4D step the reminder is attached to.
- **`source`** — `qualcomm` or `cc4d_only` (provenance, below).
- **`flag`** — `low_confidence_timing` / `low_confidence_temperature` on `cc4d_only` events.

## Provenance — what comes from where

Each reminder fuses CC4D and Qualcomm:

| field | CC4D | Qualcomm |
|---|---|---|
| which step erred + error **type** (`subtype`) | ✅ `error_annotations.json` tag | — |
| step segment `[start, end]` | ✅ `complete_step_annotations.json` | — |
| reminder **time** `t` | (fallback anchor only) | ✅ `output_timestamps` (visibility ts) |
| reminder **content** | (fallback: error description) | ✅ `output_texts` (the feedback message) |

So Qualcomm's contribution is the **single mistake-visibility timestamp** and the **message
text**; CC4D says *which* step erred and *what type*. CC4D error tags are step-level and
untimestamped — Qualcomm is what gives the sub-step time.

**`source: "qualcomm"` (1,355 events)** — a Qualcomm `mistake` event matched (±10 s, same
type) to a CC4D-tagged step. `t` = Qualcomm timestamp; `content` = Qualcomm message.

**`source: "cc4d_only"` (28 events)** — a timing/temperature CC4D tag with *no* matching
Qualcomm event. Mechanical time fallback: timing → `step.end` (latest the error is certainly
visible); temperature → `step.start` (a wrong power level is in effect from step start).
`content` = the CC4D error description. Flagged low-confidence.

## Timing is a single point; tolerance lives in evaluation

Qualcomm provides a **single timestamp**, not a window, so the GT stores exactly that: one
`t` per reminder. There is deliberately **no `window` / grace in the answer key** — matching
tolerance (e.g. ±15 s, Qualcomm's 30 s-window convention) is the evaluation script's job and
is applied there, not baked into the GT. (TODO: wire the tolerance into the scorer.)

## Build / score

```bash
python eval/gt_build_proactive.py            # writes data/cc4d_proactive/ + _summary.json
python eval/gt_build_proactive.py --only 8_50 --no-write   # inspect one recording
python eval/eval_score_corpus.py --arms oracle,silent --no-write   # Box-2 referee sanity
```

The Box-2 scorer (`eval/eval_score_corpus.py`) reads **both** dirs and merges them per
recording: `cc4d_proactive` (execution → `execution_error`/`parameter_violation`) and
`cc4d_proactive_om` (order/missing → `precondition_violation`). Reminders are points, so each
gets a zero-width window `[t,t]`; matching is exact at `tol=0` and ±tol for the `fa1_pm15s`/
`fa1_pm30s` variants. One-shot is automatic — each reminder is one GT point, matched
one-to-one. Oracle self-scores P/R/F1 = 1.0 across all 7 classes (1712 events: 1383 execution
+ 151 order + 178 missing). The retired FA-2 (decision-point) G-Mean is skipped when no
`decision_points` are present.

The **visualizer** T2 track (`visualizer/video-server.js` → `/api/timeline`) likewise merges
both dirs into one GT-reminders lane (execution + order + missing), rendered as colored point
markers with `@t`; the predicted lane comes from `qualcomm_run` arms.

## Migration notes (2026-06-28)

- Retired: `eval/gt_build_family_a.py` → `eval/_legacy/`; `data/cc4d_family_a/` deleted.
- Repointed to `data/cc4d_proactive/` + new schema: `eval/eval_score_corpus.py`,
  `eval/baseline_t2_reminder.py`, the visualizer (`visualizer/video-server.js`),
  `scripts/probe_recipe_reminders.py` (imports `ACT2FILE` from `gt_build_proactive`).
- Older docs referencing `cc4d_family_a` describe the
  retired scheme; this doc is the current reference.
