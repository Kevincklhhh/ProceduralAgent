# Order / Missing-Step Conversion for Proactive Reminders

Date: 2026-06-28.

> ✅ **IMPLEMENTED (2026-06-28) — fully deterministic, no LLM.** `eval/gt_build_om.py` writes
> `data/cc4d_proactive_om/{rid}.json`. Two policy decisions finalize the design below:
> 1. **ORDER = DAG violations only.** CC4D's canonical *sequence* order is IGNORED. An order
>    error exists only when execution breaks a DAG precondition edge (dependent before
>    prerequisite, or prerequisite after dependent). CC4D Order tags that break no DAG edge
>    are dropped — a DAG-legal reordering is not reminder-worthy. This deliberately OVERRIDES
>    CC4D's order annotation (reversing old family_a's "tag is GT") on the principle that a
>    proactive assistant should speak only on real precondition violations. Violating steps
>    cluster by DAG adjacency into one one-shot reminder per deviation.
> 2. **`content` is templated**, not generated (swap an LLM in later for richer phrasing).
>    Timing/grouping — the scored part — is 100% mechanical.
>
> Corpus (189 recordings): 151 order + 183 missing reminders; 372 order tags dropped
> (DAG-legal), 52 missing dropped (no executed dependent); 9 recordings net zero. The earlier
> LLM-end-to-end workflow (`convert-order-missing`) is superseded for generation — kept only
> as an optional audit pass. Implemented schema is "flat + group metadata"
> (see `PROACTIVE_REMINDER_GT.md`); the grouped-record discussion below is design rationale.

This note describes how to extend `cc4d_proactive` with order-error and missing-step
reminders. The conceptual rules below are now realized mechanically in `eval/gt_build_om.py`.

## Goal

The target task is a silent-by-default assistant watching a user perform a procedure. The
assistant should speak when there is actionable evidence that the user has made a mistake,
then stop unless a new distinct mistake becomes actionable.

For order and missing-step errors, the hard part is that CC4D is an offline annotation
dataset, not an interactive recovery dataset. It tells us that a step was skipped or done
out of order, but it does not tell us what the user would have done after a reminder. So the
benchmark should not require repeated reminders until the end of the passive video.

The intended behavior is:

- one reminder per distinct actionable deviation;
- short trigger opportunities when the deviation becomes evident;
- no extra credit for repeating the same reminder;
- no penalty for failing to warn about a skipped step that never becomes relevant to any
  later executed action.

## Core Rule

Do not convert one CC4D tag into one proactive reminder.

Instead, convert CC4D order/missing annotations into grouped deviations:

```text
raw CC4D tags -> grouped deviation -> one-shot reminder opportunities
```

A grouped deviation is the user-facing mistake that a live assistant would reasonably
summarize in one reminder. A group may contain several CC4D tags, because a single wrong
choice can make many later steps look out of order.

## Missing Steps

A skipped step becomes proactively remindable when the user starts a later executed step
that depends on it.

Example rule in plain language:

> If a required step was skipped, remind when the user begins the first executed downstream
> step that needed the skipped step.

If several skipped prerequisites become relevant at the same time, group them into one
reminder. If a skipped step has no later executed dependent step, do not score it as a
proactive reminder. It is a valid offline error, but there is no grounded live moment in the
passive video where the assistant is supposed to speak.

The reminder window should be short and tied to that dependent-step start, for example
`[successor_start, successor_start + 15s]`.

## Order Errors

Order errors should be grouped by the underlying procedural deviation, not by the number of
steps marked with `Order Error`.

Important caveat: a CC4D `Order Error` tag is not the same thing as a violated DAG
precondition. CC4D often marks violations of the recipe's linear canonical order even when
the recipe DAG allows the steps to be done in parallel. Those canonical-only tags are useful
evidence, but they should not automatically become primary proactive reminders.

The strongest mechanical grouping signal is the recipe DAG frontier: at a given moment, what
prerequisites are still unmet for the step the user has started? Steps that violate the same
unmet frontier are one deviation, even if CC4D tags several of them.

Use CC4D order tags as candidate evidence, not as final reminder units. A grouped order
deviation should be part of the primary proactive-reminder GT when it is DAG-backed, or when
it fuses with a missing-step deviation. If an order tag is canonical-only and never connects
to a DAG-backed group, keep it as a secondary `canonical_order` diagnostic or leave it out
until manually adjudicated.

Order groups can also use the CC4D free-text pivot as supporting evidence. Descriptions such
as "before heating the pan", "after adding sugar", or "before trimming the edges" often
identify the same deviation from opposite sides.

The scoring window should not be the entire long episode. Instead, the group should carry
short reminder opportunities:

- the first time the deviation becomes evident;
- later starts of major dependent steps, if the first opportunity was missed;
- at most one true positive for the whole group.

The long episode span is useful for duplicate suppression and diagnostics, but it should not
be the scoring window. Otherwise a model could speak hundreds of seconds late and still get
credit.

## One-Shot Scoring

Each grouped deviation is one-shot.

A prediction that matches any valid opportunity for the group receives one true positive.
Additional reminders for the same unresolved group are duplicates. They should not increase
recall, and depending on the scorer they may be ignored as duplicates or counted as
over-talk false positives.

This matches the user need better than persistent repetition: a real assistant should give a
clear correction once, then avoid repeating it unless the task enters a new procedural failure
state.

## Example: `8_50` Spiced Hot Chocolate

Original recipe:

1. Fill a mug with milk.
2. Microwave for 1 minute.
3. Add cinnamon.
4. Add sugar.
5. Add chocolate.
6. Mix.
7. Heat and serve.

Actual execution:

| Time | Action |
|---|---|
| `0.7-69.3s` | added sugar |
| `69.3-92.3s` | filled mug with milk |
| `96.1-159.8s` | microwaved |
| `163.3-185.9s` | added chocolate and spilled one piece |
| `186.4-251.7s` | heated and served |
| never | added cinnamon |
| never | mixed |

The current raw order/missing annotations contain three order events and two missing-step
events. A live assistant should not be expected to say five separate procedural reminders.

Better conversion:

| Group | Opportunity | Reminder intent |
|---|---|---|
| early sugar before milk/microwave | `0.7s`; later opportunities at `69.3s` and `96.1s` if missed | "Sugar should be added after the milk is microwaved." |
| chocolate spill | Qualcomm timestamp `184.3s` | "One chocolate piece spilled; replace it or put it back." |
| missing cinnamon + mix before final heat | `186.4s` | "Before heating, add cinnamon and mix the mug." |

The early-sugar group may contain the raw order tags on sugar, milk, and microwave, but it is
one user-facing deviation. Some members, such as "milk after sugar", are canonical-order
consequences rather than standalone DAG violations; they stay inside the DAG-backed group and
do not create separate reminders. The missing cinnamon and missing mix tags fire together
because the user starts final heating without both prerequisites.

## Example: `16_44` Scrambled Eggs

This is the stress case. The user starts by heating the pan before most preparation is done,
then performs many preparation steps late. Raw conversion produces 16 order events and 4
missing-step events.

The recipe structure is roughly:

```text
prep ingredients + egg mixture
  -> heat pan
  -> add salt/onions
  -> saute onions
  -> add garlic/chilli
  -> cook
  -> add turmeric + tomatoes
  -> cook covered
  -> pour eggs
  -> mix eggs
  -> garnish
```

Actual high-level failure pattern:

| Time | Action |
|---|---|
| `3.3-87.6s` | heats pan immediately |
| `93.2-628.1s` | performs many prep steps late |
| `526.7s` | adds garlic before sauteing onions |
| `634.8s` | adds chilli before sauteing onions |
| `646.4s` | cooks before the onion/tomato branch is in the right state |
| `744.5s` | whisks egg mixture very late |
| `772.8s` | adds onions late |
| `789.4s` | sautees onions late |
| `903.1s` | pours eggs |
| never | adds turmeric, adds tomatoes, cooks covered, adds salt to bowl |

Better conversion:

| Group | Opportunity | Reminder intent |
|---|---|---|
| heat before prep | starts at `3.3s`; short late opportunities when late prep reveals the problem | "Pause: finish preparing the ingredients and egg mixture before heating the pan." |
| garlic/chilli/cook before sauteed onions | starts around `526.7s`, with later opportunities around `634.8s` and `646.4s` | "Add and saute the onions before adding garlic/chilli and cooking everything." |
| pour eggs before tomato/turmeric branch | `903.1s` | "Before pouring eggs, you still need turmeric, tomatoes, and the covered-cook step." |

The first group should not span the whole video as a scoring window. Later prep steps after
the early pan heat are evidence or partial recovery, not new reminders. The second and third
groups are new frontiers: they are new moments where the user starts a downstream step while
important prerequisites are still unmet.

## Example: `10_26` Pinwheels

This recording has no order events but several skipped steps. A direct conversion would make
six missing-step reminders.

Better conversion:

| Group | Opportunity | Reminder intent |
|---|---|---|
| missing nut-butter steps before moving on | around `34.6s` | "Add and spread the nut butter before the jelly/filling sequence continues." |
| missing floss setup before slicing | around `235.4s` | "Set up the floss under the roll before slicing." |
| discarded ends skipped | dropped | no later executed dependent step gives a proactive timestamp |

The dropped terminal skip is still an offline error, but it should not be scored as a live
proactive reminder without an independent annotation of when a user-facing reminder would
have helped.

## Edge Cases

**Giant cascades.** If a single early mistake makes many later steps out of order, do not
score every later tag. Group by the unmet DAG frontier and use short opportunities.

**Parallel steps.** If the recipe DAG permits several steps in any order, do not create an
order reminder just because the textual recipe list has an order. Canonical-only CC4D order
tags should be reported separately or manually adjudicated; they are not mechanically valid
primary proactive reminders.

**Terminal missing steps.** If a skipped step has no executed successor, drop it from the
proactive reminder task. There is no grounded moment to speak in the passive video.

**Multiple skipped prerequisites at one successor.** Group them into one reminder. For
example, "add cinnamon and mix before heating" is one useful reminder, not two simultaneous
interruptions.

**Late recovery.** If the user later performs a prerequisite after doing a downstream step,
do not create a fresh reminder just because the late step is tagged as out of order. Treat it
as recovery evidence for the earlier deviation unless it opens a new unmet frontier.

**Duplicate predictions.** A model that repeats the same reminder should not receive extra
credit. The benchmark measures whether it catches the deviation, not how many times it says
the same thing.

## Recommended Release Shape

The extension should store grouped deviations, not only flat reminder timestamps. A compact
record can include:

```json
{
  "id": "16_44_order_heat_before_prep",
  "subtype": "order",
  "content": "Pause: finish preparing the ingredients and egg mixture before heating the pan.",
  "members": ["16_44_e16", "16_44_e19", "..."],
  "opportunities": [[3.3, 18.3], [93.2, 108.2]],
  "episode_span": [3.3, 102.6],
  "one_shot": true,
  "source": "cc4d_order+dag"
}
```

For the current point-based `cc4d_proactive` schema, each grouped deviation can still expose
a primary timestamp `t = opportunities[0][0]`, while keeping the opportunity list as an
extension field for the order/missing scorer.

## Summary

Order/missing conversion is feasible, but only if the benchmark changes the unit of
annotation and treats raw CC4D order tags as candidates rather than final labels:

```text
raw tag -> DAG/frontier check -> grouped deviation -> one-shot reminder
```

That preserves the CC4D error facts while matching the end goal: a proactive assistant that
speaks once, at the moment a mistake becomes actionable, instead of producing an offline
audit of every consequence of a wrong script.
