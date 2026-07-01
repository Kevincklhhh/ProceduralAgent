---
name: write-experiment-log
description: Write a detailed experiment log (.md) into logs/ for an experiment run during this conversation. FIRST asks the user which experiment(s) in the dialogue to log (the scope), then records grounded setup + results + caveats. Use when the user says "write an experiment log", "log this experiment / run / result", "record this to logs/", "document what we just ran", or similar. Captures what was actually executed this session, not a plan.
---

# Write experiment log

Turn an experiment that happened **in this conversation** into a durable, grounded log under
`logs/`. The log must be reproducible and honest: real commands, real numbers read back from
artifacts, real caveats — not remembered or expected values. Aligns with the `project-brief`
reporting rules (plain language, define jargon, ground every claim in what actually ran).

## Procedure

### 1. Establish scope — ASK FIRST (blocking)
Do not start writing until the user has chosen the scope. The scope is *which experiment(s) from
this dialogue* the log covers.

1. Scan the conversation for distinct experiment runs — each command actually executed, its
   parameters, and the result artifact it produced (e.g. a `summary.json`, an output file, a
   printed metric). Build a short candidate list.
2. Call **AskUserQuestion** with those candidates as options (one per distinct run, plus an
   "all experiments this session" option). `multiSelect: true` if several could be logged
   together. The user may pick one, several, or type a custom scope via "Other". Keep option
   labels concrete (e.g. "Binary vs targeted head-to-head (56 units)", not "experiment 2").
3. If the user already named the scope in their request, confirm it in one line and skip the
   question only if unambiguous.

### 2. Gather grounded details (for the in-scope run(s) only)
**Read the artifacts back from disk — quote actual values, never recall them.** For each run:

- **Question / motivation** — what the run was meant to answer (from the dialogue).
- **Setup**
  - data: recordings/recipe/split, GT source + path.
  - model: exact id and where it ran (verify, e.g. server `/v1/models`); don't assume.
  - parameters: every knob (fps, frame cap, windowing/oracle, mode, thresholds, neg sampling…).
  - command(s): the exact invocation(s) run, copy-paste-able (incl. env vars).
  - code: script path(s); if a git repo, capture the commit (`git rev-parse --short HEAD`) and note
    if the working tree was dirty.
  - inputs: the input artifact paths consumed.
- **Results** — the real numbers, pulled from the output files (re-open `summary.json`/outputs and
  quote). Use tables. Include overhead (calls, frames, wall-time) when measured.
- **Interpretation** — **≤ 3 sentences**, plain language, what it means for the project goal. No
  background, no restating the results table.
- **Caveats & limits** — a **short bullet list, ≤ 5 bullets, one line each**: in-sample, oracle
  windows, prompt leaks, small n, confounds, anything not verified. Name the fraction that didn't
  work; cut anything not specific to this run.
- **Artifacts** — every file the run wrote (paths), so results can be re-found.

If a number can't be verified against an on-disk artifact, mark it "(from conversation, unverified)".

### 3. Write the log
Path: `logs/YYYY-MM-DD_<short-slug>.md` (today's date; slug from the scope, e.g.
`2026-06-24_t2-binary-vs-targeted`). If a log for the same scope already exists, **update it**
(add a dated section) rather than create a duplicate. Use this skeleton:

```markdown
# <Experiment title>

- **Date:** YYYY-MM-DD
- **Author:** <model id> (Claude Code session)
- **Scope:** <what this log covers — and what it deliberately excludes>
- **Code:** <script paths> @ <git short-sha>[ (dirty)]

## Question
<what the run was meant to answer>

## Setup
- **Data:** <recordings / recipe / split / GT source + path>
- **Model:** <exact id + where it ran, verified>
- **Parameters:** <fps, frames, windowing, mode, …>
- **Command(s):**
  ```bash
  <exact invocation incl. env>
  ```
- **Inputs:** <consumed artifact paths>

## Results
<tables with REAL numbers read from the output artifacts; include overhead>

## Interpretation
<≤ 3 sentences, plain language; define any jargon on first use>

## Caveats & limits
<short bullets, ≤ 5, one line each: in-sample / oracle / leaks / n / confounds>

## Artifacts
- <path> — <what it is>
```

### 4. Report
Print the log path and a one-line summary. Do not restate the whole log to the user.

## Scope notes
- **This session only.** Log experiments that actually ran in the current conversation. If the user
  asks to log something from a past session, say so and ask them to point at the artifacts.
- **One scope per run.** For several unrelated experiments, either log the user's chosen subset or
  (if they pick "all") write one log per experiment, clearly separated.
- **Grounding over completeness.** A short log with verified numbers beats a long one with
  remembered ones.
