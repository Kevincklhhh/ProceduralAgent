# MiniCPM-o 4.5 per-frame error probe — can a proactive streaming VLM catch a chocolate-count mistake?

- **Date:** 2026-06-27
- **Author:** claude-opus-4-8 (Claude Code session)
- **Scope:** Two phases on **one real error step** (8_26 step 90, "Add 2 pieces of chocolate"):
  (1) the targeted per-frame probe over the GT error window at two listen-gates + a context-length
  measurement, and (2) a **2×2 prompt×fps** detection test (greedy). Excludes: the local-deployment
  smoke test, the 8_16 proactivity / binary-vs-targeted runs (different recording, separate scope).
- **Code:** `replication/MiniCPM-o-Demo/proactivity_test.py` + `prompts/shc_choc_count{,_neutral}.txt`
  — harness lives inside the cloned MiniCPM-o-Demo repo and is **untracked by the main project repo**
  (main repo @ `6562aea`). MiniCPM-o demo code is the upstream OpenBMB clone, unmodified.

## Question
Given the *oracle* error-step window and a prompt naming the exact mistake to watch for, can the
full-duplex proactive VLM (MiniCPM-o 4.5) actually detect a **measurement/count error** — the user
adding 4 chocolate pieces when the recipe wants 2 — and if not, do a stronger/neutral prompt or
denser frame sampling (the two levers floated: prompt strength, context management) fix it?

## Setup
- **Data:** CC4D recipe `spicedhotchocolate` (activity 8), recording **8_26**, video
  `data/videos_480p/8_26.mp4` (no audio stream → silent 16 kHz fed). Step 90 "Add-Add 2 pieces of
  chocolate to the mug".
- **Ground truth:** `data/cc4d/annotations/annotation_json/error_annotations.json` — 8_26 step 90,
  window **179.75–219.54 s**, error = **Measurement Error: "Added 4 pieces of chocolate to the mug"**
  (recipe requires 2). So the correct detector output is "more than 2 / four".
- **Model:** `openbmb/MiniCPM-o-4_5` (9B) loaded locally from `/data/kailaic/hf_cache/MiniCPM-o-4_5`
  via `MiniCPMO.from_pretrained(..., _attn_implementation="sdpa")`, bf16, on **ronaldo GPU 7**
  (env `minicpmo`). TTS-free init: `generate_audio=False`, `ref_audio=None` (token2wav never loaded).
  LLM is full attention (`use_sliding_window=False`, `max_position_embeddings=40960`); duplex
  `sliding_window_mode=off`.
- **Parameters:** window `--start 179.7 --end 219.6` (40 one-second ticks); decision cadence fixed
  1 Hz. "fps N" = N video frames per 1-s tick (`max_slice_nums=1`). `listen_prob_scale` (gate):
  lower = speaks more. Phase 1: `decode_mode=sampling`, gates 1.0 (default) and 0.4 (eager).
  Phase 2: `decode_mode=greedy`, gate 0.4, prompt∈{primed,neutral} × fps∈{1,5}.
- **Prompts:** `prompts/shc_choc_count.txt` ("primed": states the step + "recipe requires EXACTLY 2
  … say it is correct if exactly 2 go in"); `prompts/shc_choc_count_neutral.txt` ("neutral": "count
  how many you SEE … say 'I can't tell' … do NOT guess"; never mentions 2 or correct/mistake).
- **Command(s):**
  ```bash
  # Phase 1 — per-frame probe, two gates (sampling)
  conda run -n minicpmo python proactivity_test.py --video ../../data/videos_480p/8_26.mp4 \
    --gpu 7 --start 179.7 --end 219.6 --fps 1.0 --system-prompt-file prompts/shc_choc_count.txt \
    --listen-prob-scale 1.0 --out proactivity_8_26_choc_default.json    # gate 1.0
  # ...same with --listen-prob-scale 0.4 --out proactivity_8_26_choc_eager.json
  # ...rerun of eager after adding ctx logging -> proactivity_8_26_choc_eager_ctx.json

  # Phase 2 — 2x2 prompt x fps (greedy, gate 0.4)
  conda run -n minicpmo python proactivity_test.py --video ../../data/videos_480p/8_26.mp4 \
    --gpu 7 --start 179.7 --end 219.6 --fps {1|5} --decode-mode greedy --listen-prob-scale 0.4 \
    --system-prompt-file prompts/shc_choc_count{,_neutral}.txt \
    --out proactivity_8_26_2x2_{primed|neutral}_fps{1|5}.json
  ```
- **Inputs:** the video above; the GT json above; the two prompt files. Each run also dumps its
  exact input frames to `<out>_frames/`.

## Results

### Phase 1 — per-frame probe over the GT window (40 one-second ticks, 1 frame/tick)
| Gate | Spoke (ticks) | What it emitted | Caught the 4-vs-2 error? |
|---|---|---|---|
| 1.0 (default), sampling | 0 / 40 | nothing — silent all 40 s | No (silent) |
| 0.4 (eager), sampling, run A | 10 / 40 | vid 180.7 "…waiting for you to drop them in so I can count"; vid 189.7 "…I'm ready to count." — never a count | No (never committed) |
| 0.4 (eager), sampling, run B | 37 / 40 | vid 180.7→ repeated "I see two pieces of chocolate, which is correct." (≈every tick, incl. before bag opened) | No (false "correct"; hallucinated) |

Same gate/prompt, runs A vs B differ only by sampling seed → two opposite failure modes (silent
vs confident false-"correct"). **Context measured** (run B, `ctx_tokens` = LLM KV length, full
attention): 307 tok at the first emission (vid 180.7) → **3461 tok at end (vid 218.7) = 8.4% of the
40960 window**; grows ~83 tok/s (≈ 1 frame ~64 vision tok + 1 s audio + `<unit>`/`</unit>` + text).

### Phase 2 — 2×2 prompt × fps (greedy, deterministic; 40 ticks)
| Cell | Counts emitted | Hit GT (4)? | Notable utterance | avg ms/tick | ctx max (of 40960) |
|---|---|---|---|---|---|
| primed · 1 fps | — | No | "Looking at the chocolate pieces being added." (no number) | 378 | 3.3k (8%) |
| primed · 5 fps | 1, 2 | No | "I see one piece… not the required 2." → later "Two pieces added. That's correct." (contradicts) | 1146 | 13.9k (34%) |
| neutral · 1 fps | 1, 3 | No | "I see 3 pieces" → "1 piece" → "1 piece" | 416 | 3.3k (8%) |
| neutral · 5 fps | 1, 3 | No | **"I can't tell how many."** → "3 pieces" → "1 piece" | 1279 | 13.9k (34%) |

**No cell ever output the GT count 4** (only 1, 2, or 3). Prompt effect (real): primed → parrots
"2/correct" or vague; neutral → emits actual numbers and the honest "can't tell" abstention, never a
false "correct". fps effect: 5 frames/tick increases engagement and enabled the only "can't tell",
but costs ~3× latency and ~4× context for no accuracy gain.

## Interpretation
Even with the oracle error-step window and the mistake named in the prompt, MiniCPM-o 4.5 never
correctly counts the chocolate pieces — counting 2-vs-4 small chips in this motion-blurred
egocentric clip is a genuine perception limit, not a prompt- or context-size artifact (context
capacity peaked at 34% of 40960). A neutral, non-leading prompt fixes the model's *honesty* (real
numbers + "can't tell", no false "correct") but not its *accuracy*; denser frames only add cost.
This supports the project thesis that instantaneous measurement errors need a cheap specialized
detector firing a precise trigger at the moment of the add, not a continuous (or merely denser) VLM.

## Caveats & limits
- **Detection ceiling, not deployable:** oracle GT step window given for free; a real system must find it.
- **Single error, single recording (n=1):** 8_26 step 90 only; one error subtype (count/measurement).
- **Action ≠ window:** the add is ~instant within a 40 s step; the decisive drop frame may never be sampled at 1–5 fps (most emitted counts occur before the bag is even opened, ~vid 194 s).
- **Sampling-dependent (Phase 1):** identical gate/prompt gave opposite outputs; greedy used in Phase 2 to remove this.
- **No audio:** CC4D preview videos have no audio track, so the 1 Hz loop ran on silent input + frames only.

## Artifacts
- `replication/MiniCPM-o-Demo/proactivity_8_26_choc_default.json` — Phase 1, gate 1.0 (silent).
- `replication/MiniCPM-o-Demo/proactivity_8_26_choc_eager.json` — Phase 1, gate 0.4 run A (waiting-to-count).
- `replication/MiniCPM-o-Demo/proactivity_8_26_choc_eager_ctx.json` — Phase 1, gate 0.4 run B + per-tick `ctx_tokens`.
- `replication/MiniCPM-o-Demo/proactivity_8_26_2x2_{primed,neutral}_fps{1,5}.json` — Phase 2 cells (+ `_frames/`).
- `replication/MiniCPM-o-Demo/proactivity_2x2.log`, `proactivity_choc.log` — console traces.
- `replication/MiniCPM-o-Demo/proactivity_test.py`, `prompts/shc_choc_count{,_neutral}.txt` — harness + prompts.
- GT: `data/cc4d/annotations/annotation_json/error_annotations.json` (8_26 step 90).
