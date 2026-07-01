# The Latency Story — Reminder Latency as the Headline (2026-06-28)

> 🧭 **What this doc is.** The current North Star for the active arm of the project: making a
> proactive procedural reminder *useful in real time*, which is a **latency** problem first and an
> **accuracy** problem second. It sits on top of — does not replace — the three-box architecture
> (`PIPELINE_THREE_BOXES.md`), the task definitions (`TASK_T1_STEP_LOCALIZATION.md`,
> `BENCHMARK_INDEX.md`), and the evaluation referee (`REMINDER_EVALUATION.md`). Where the older
> framing said "sensor control = energy/latency saved at equal coverage," this doc keeps that
> substrate but reframes the headline: **the deployable win is low reminder latency; the open risk
> is reminder accuracy.** See `PROJECT_BACKGROUND.md §1` for the original thesis statement.
>
> This is a framing doc — definition, task, open agenda. The measured latency and accuracy results
> live in `logs/` (pointers in §4), kept out of here on purpose.

---

## 1. Why latency is the headline

A reminder is only useful if it arrives **shortly after the evidence that justifies it appears.** A
correct reminder delivered 30 s late, after the user has already moved on (poured the next
ingredient, turned off the heat), is worthless or harmful. So the quantity that decides product
viability is not "did the model eventually notice" but "how long from *evidence visible* to
*intervention spoken*."

This is also where the project's sensor-control thesis cashes out concretely: "gate/trigger an
expensive VLM with cheap detectors" only matters if the triggered VLM call is *fast* — which is what
makes the B-trigger role deployable rather than a paper abstraction.

---

## 2. Definition — reminder latency

> **Reminder latency** = time from the annotated **evidence timestamp** (the moment a mistake
> becomes *evident* in the stream) to the model's **generated intervention**.

It decomposes into two additive parts:

```
reminder_latency  =  t_wait        +  t_inference
                     (time before a   (VLM forward:
                      request is sent   encode/prefill
                      to the VLM)       + decode)
```

- **`t_wait` — pre-request delay.** How long after the evidence appears before *anything* asks the
  VLM. In a continuous-VLM system this is ~0 but you pay full inference every tick. In a
  duty-cycled system this is the cheap-detector trigger delay + tick quantization. This is the term
  the *sensing policy* (A-solve / B-trigger / C-none) controls.
- **`t_inference` — VLM forward.** Preprocess + vision encode (ViT) + LM **prefill** (building the
  KV cache over procedure + evidence frames) + **decode** (generating the answer tokens). Prefill is
  the window-scaling term; decode is the sequential, latency-dominating-but-boundable term.

The whole latency story is: **move `t_inference` work off the critical path** (pre-encode prefill
ahead of the trigger) and **bound the decode** (ask a pointed yes/no instead of an open-ended
"what's wrong?"), so that when the trigger fires, only a small constant residual remains.

---

## 3. The proactive-reminder task (what we are actually scoring)

**Task.** Watch a streaming egocentric cooking video. At each moment, decide **interrupt vs.
stay silent**, and when interrupting, emit a **typed, timestamped reminder** about a procedural
mistake. Silence is the default; clean recordings must stay silent.

**Substrate:** CaptainCook4D (24 recipes, 384 recordings) with step DAGs, step timestamps, and
error tags. Ground truth is the firewalled Box-1 truth table (`data/cc4d_family_a/`); the predictor
never sees it (see `gt-predictor-firewall`).

**Scored taxonomy (mechanical-only, from `REMINDER_EVALUATION.md`):**
- `execution_error` → technique / preparation / measurement / **temperature**
- `parameter/timing`
- `precondition/missing_step`
- `precondition/order`

**Metrics (full spec in `REMINDER_EVALUATION.md`):**

| Axis | Metric | Note |
|---|---|---|
| Did we fire the right reminders? | **per-class windowed Precision / Recall / F1** | one TP per expected `(recording, event)`; any un-owed fire = FP; silence scored. ±15 s (ours) / ±30 s (LiveMamba-comparable) |
| When to speak vs. stay silent | **G-Mean F1** = √(Interrupt-F1 × Silent-F1) | hard negatives = step completions where nothing is owed; degenerate always-silent/always-speak → 0 |
| Was it timely? | **STS** = exp(−(t̂−s)/(e−s)), step-completion delay τ | earliness within window; this is where reminder *latency* surfaces in the offline metric |
| What did it cost? | vlm_calls, frames, **vlm_latency_total_s**, compute_s | per video-minute; real-time feasibility stated |

**Precision vs. recall, and what "usable" means.** Precision and recall trade off and we report
both per class — never pooled (most mistake events are visual-leaning, so a pooled number hides the
audio arm). A **usable** proactive reminder system is **precision-dominated**: a false reminder
interrupts the user for no reason and trains them to ignore the assistant, so a deployable bar is
roughly **precision ≳ 0.8 at whatever recall that allows**, *not* high recall with frequent false
alarms. Current zero-shot baselines are nowhere near this — they over-fire badly (false alarms ≫
true detections; see the logs in §4), which is the precise failure a usable system cannot have.

---

## 4. Open problems (the live agenda)

These are unsolved and drive the next experiments. Supporting results live in the logs named below.

1. **"Trash performance even before reminder quality."** Before we even judge *how good* a reminder
   is, raw detection is weak — and **timing and temperature errors are especially bad**. Temperature
   is the audio/off-screen-leaning slice (`REMINDER_EVALUATION.md` expects low detector recall
   there); timing errors require localizing *when* against an oracle-free clock. Both are precisely
   the subtypes where the VLM has no reliable visual handle. **First job is to lift the floor on
   these, separate from reminder phrasing.** (See `logs/2026-06-28_qualcomm-turnbased-baseline.md`,
   `logs/2026-06-24_t2-binary-vs-targeted.md`; memory `vlm-t2-detection-ceiling`.)

2. **How to "train" / condition the model to know the error *types*.** Naming the specific mistake
   beats an open-ended "what's wrong?", so the model can verify a *specific* hypothesis far better
   than it can generate one. Open question: where does the per-step candidate-mistake list come from
   in a *firewall-clean* system (today it is GT-derived = in-sample ceiling)? Options on the table:
   recipe-text-only anticipated checks (the Stage-1 criteria, see `CRITERIA_GENERATION_PROBING.md`),
   light fine-tuning on error taxonomy, or a cheap detector that picks the hypothesis. **The
   predictor must not see GT** (`gt-predictor-firewall`).

3. **What errors do we target? "Is spilling an error?"** Scope question. The CC4D taxonomy is
   mechanical (technique/preparation/measurement/temperature/timing/order/missing). Accidents like
   spilling, dropping, contamination are *not* cleanly in that taxonomy and have no GT window — do we
   target them, and with what evidence? **Decision needed: freeze the targeted-error set and state
   explicitly what is out of scope.** (Cf. `reminder-labels-need-existing-gt`: classes must derive
   from existing annotation; safety was cut for lacking GT.)

4. **Missing-step and out-of-order errors.** These are *structural* (DAG) errors, not
   single-moment visual events, so the "evidence timestamp" is ill-defined (the mistake is an
   *omission* or a *reordering*, evident only relative to procedure state). They are scored straight
   off the CC4D tags (order = one event per tagged step; see `REMINDER_EVALUATION.md`), and roughly
   half of order events break no real DAG edge (cheap-DAG-recoverable share). **Open: how does a
   streaming predictor even *time* a missing-step reminder, and how is latency defined when the
   evidence is an absence?**

5. **Firewall-clean candidate lists + leak-stripped re-runs.** The detection results to date are
   firewall-relaxed (GT-derived candidate lists, oracle step windows). The deployable numbers need:
   self-tracked step identity (T1), recipe-text-only checks, and targeted re-runs without the leaked
   observed-error clause.

---

## 5. What "usable" would look like (targets, not results)

- **Latency:** trigger → spoken reminder small and roughly constant in steady state — realizable if
  the prefill is precomputed off the trigger path and the decode is bounded (§2); `t_wait` set by
  the sensing policy and reported separately.
- **Accuracy:** per-class **precision ≳ 0.8** (do-no-harm; few false interruptions) at the best
  recall that allows; false alarms must drop far below today's over-firing regime.
- **Cost:** beat always-on-VLM `vlm_latency_total_s` / `vlm_calls` at **equal Box-2 quality**
  (Pareto, never a single traded number — `PROJECT_BACKGROUND.md §2a`).

The research bet is that **anticipation + cheap triggering + bounded verification** can lift
precision into the usable band without continuous VLM — the latency mechanism exists; §4 lists what
stands in the way on accuracy.

---

## 6. Pointers

- **Task / metrics:** `REMINDER_EVALUATION.md` (Box 2 referee), `TASK_T1_STEP_LOCALIZATION.md`
  (step substrate), `BENCHMARK_INDEX.md` (canonical naming).
- **GT / predictor split:** `PIPELINE_THREE_BOXES.md`, `PROACTIVE_REMINDER_GT.md` (Box 1),
  `gt-predictor-firewall` (memory).
- **Latency evidence (logs):** `logs/2026-06-25_preencode-4arm-latency.md`,
  `logs/2026-06-25_streaming-rolling-prefill-and-scaling.md`,
  `logs/2026-06-25_decode-cost-open-vs-bounded.md`; memory `preencode-kv-cache-latency`.
- **Accuracy ceilings (logs):** `logs/2026-06-28_qualcomm-turnbased-baseline.md`,
  `logs/2026-06-24_t2-binary-vs-targeted.md`, `logs/2026-06-27_minicpmo-perframe-choc-count.md`;
  memory `vlm-t2-detection-ceiling`.
- **Original thesis (substrate):** `PROJECT_BACKGROUND.md §1`, memory
  `sensor-control-thesis-and-role-split`.
</content>
