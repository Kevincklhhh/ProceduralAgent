"""proposed_runtime -- execute a compiled procedure-monitor plan (proposed system, entry point).

Ingests a recording's audio (+ video for the VLM arm), runs the bound detectors
once, releases their events causally (visible only at t >= event_time + latency,
every latency <= 10 s so this equals a true per-tick streaming detector), and
drives the plan's graph state machine on a fixed tick. Emits the plan output:
stage_intervals + transition_trace + sensor_events + cost.

Generalizes the hardcoded chain in eval/legacy_detector_replay.py into a predicate-driven loop:
completions are checked before starts (so one anchor can close stage N and open
N+1 at the same instant), boundaries are attributed to the event time (t_fire),
and a C-none foreground block is polled by the VLM arm (cadence-gated,
cost-capped) to label its silent members.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eval/
sys.path.insert(0, os.path.join(BASE, 'detectors'))              # detectors/
import proposed_plan_loader as plan_loader
import runtime as detlib
from runtime import audio_io

RECS = ['8_16', '8_3', '8_25', '8_26', '8_31', '8_50']
OUT_DIR = os.path.join(BASE, 'experiments/proposed_system/results')


# --------------------------------------------------------------------------
# runtime state + evaluation context
# --------------------------------------------------------------------------
class RuntimeState:
    def __init__(self, units):
        self.step_states = {u.uid: ("eligible" if not u.requires else "not_started")
                            for u in units}
        self.start_time = {}
        self.complete_time = {}


class Ctx:
    def __init__(self, state, events):
        self.state = state
        self._events = events
        self.t = 0.0
        self.consumed = set()      # id(event) -> consume-once across the whole run
        self.vlm_verdicts = {}     # for_step -> {status, t, confidence}

    def events_up_to(self, t):
        return [e for e in self._events if e["release_t"] <= t]


# --------------------------------------------------------------------------
# event source: compute detectors once, tag each event with its release time
# --------------------------------------------------------------------------
def build_event_source(plan, rec):
    prims = plan.audio_primitives()
    need16 = any(p in ("D1", "D2", "D4") for p in prims)
    need48 = any(p in ("D3", "D5") for p in prims)
    x16 = fs16 = x48 = None
    if need16 or not prims:
        fs16, x16 = audio_io.load_audio_16k(rec)
    if need48:
        fs48, x48 = audio_io.load_audio_48k(rec)
    duration = len(x16) / fs16 if x16 is not None else len(x48) / fs48

    events = []
    t0 = time.perf_counter()
    for p in prims:
        det = plan.detector_bindings[p]()
        if p in ("D1", "D2", "D4"):
            evs = det.detect(x16, fs16)
        else:
            evs = det.detect(x48)
        events += evs
    compute_s = time.perf_counter() - t0
    for e in events:
        e["release_t"] = e["t_s"] + plan.latency(e["primitive"], e["event"])
    events.sort(key=lambda e: e["release_t"])
    return events, duration, compute_s


# --------------------------------------------------------------------------
# applying a rule's structured state_update ops
# --------------------------------------------------------------------------
def apply_update(plan, state, ctx, uid, update, t_fire, fg):
    for op in update:
        kind = op["op"]
        if kind == "set_state":
            step, to = op["step"], op["to"]
            state.step_states[step] = to
            if to == "active" and step not in state.start_time:
                state.start_time[step] = t_fire
            if to == "complete":
                state.complete_time[step] = t_fire
        elif kind == "set_foreground":
            fg.append((t_fire, op["step"]))
        elif kind == "open":
            for s in op.get("steps", []):
                u = plan.by_id.get(s)
                if u and all(state.complete_time.get(r) is not None for r in u.requires):
                    if state.step_states.get(s) in (None, "not_started", "eligible"):
                        state.step_states[s] = "eligible"
        elif kind == "mark_members":
            blk = plan.by_id.get(op["block"])
            for m in (blk.members if blk else []):
                mid = m["step_id"]
                if state.step_states.get(mid) != "complete":
                    state.step_states[mid] = op.get("rest", "unknown")
        elif kind in ("add_background", "remove_background"):
            pass  # background-monitor bookkeeping is a no-op in this milestone


# --------------------------------------------------------------------------
# main run
# --------------------------------------------------------------------------
def run_recording(plan, rec, vlm=None, video=None, trace_dir=None):
    events, duration, compute_s = build_event_source(plan, rec)
    state = RuntimeState(plan.units)
    ctx = Ctx(state, events)
    fg = []                       # (t_fire, uid) foreground timeline
    trace = []                    # transition_trace
    fired = set()                 # (uid, phase, rule_id) consume-once for rules
    last_poll = {}                # block uid -> last VLM poll time
    poll_counts = {}              # block uid -> polls so far (cost cap)
    vlm_calls = frames_sent = 0
    vlm_latency = 0.0

    # --- T2 reminder state ---
    reminders = []                # emitted proactive reminders (execution / timing / missing)
    emitted_checks = set()        # (node, subtype) dedup so a persistent deviation fires once
    member_recog_t = {}           # block member step_id -> first recognized time

    def fire(uid, phase, rule, m):
        nonlocal trace
        apply_update(plan, state, ctx, uid, rule.state_update, m.t_fire, fg)
        for e in m.consumes:
            ctx.consumed.add(id(e))
        trace.append(dict(t_s=round(m.t_fire, 2), transition=phase, step_or_block=uid,
                          rule_id=rule.rule_id,
                          evidence=[f"{e.get('primitive','')}.{e.get('event','')}"
                                    for e in m.evidence if isinstance(e, dict)],
                          confidence=round(m.confidence, 3)))
        fired.add((uid, phase, rule.rule_id))

    tick = plan.tick_s
    n_ticks = int(math.ceil(duration / tick)) + 1
    for k in range(n_ticks):
        ctx.t = k * tick

        # 1. completions (before starts)
        for u in plan.units:
            if state.step_states.get(u.uid) != "active":
                continue
            for r in u.complete_rules:
                if (u.uid, "complete", r.rule_id) in fired:
                    continue
                m = r.pred(ctx, u.uid)
                if m is not None:
                    fire(u.uid, "complete", r, m)
                    break

        # 2. starts of eligible units
        for u in plan.units:
            if state.step_states.get(u.uid) != "eligible":
                continue
            for r in u.start_rules:
                if (u.uid, "start", r.rule_id) in fired:
                    continue
                m = r.pred(ctx, u.uid)
                if m is not None:
                    fire(u.uid, "start", r, m)
                    break

        # 3. VLM arm: ONE merged periodic call on the active foreground unit that needs the
        #    VLM -- the C-none block (recognize which member + check it) or any active step
        #    carrying a VLM check. Recognition and all of the step's checks ride one call;
        #    it emits a reminder only when the VLM judges the user off-track.
        if vlm is not None:
            unit = next((u for u in plan.units
                         if state.step_states.get(u.uid) == "active"
                         and ((u.is_block and u.sensing_role == "C-none" and u.vlm.get("poll"))
                              or (not u.is_block
                                  and any(c.get("detector") == "VLM" for c in u.checks)))),
                        None)
            if unit is not None:
                pol = plan.vlm_policy
                uid = unit.uid
                since = ctx.t - last_poll.get(uid, -1e9)
                npolled = poll_counts.get(uid, 0)
                budget_ok = vlm_calls < pol.get("vlm_budget", 1e9)
                cap_ok = npolled < pol.get("max_polls_per_step", 1e9)
                if since >= pol.get("period_s", 10.0) and budget_ok and cap_ok:
                    done = ([m["step_id"] for m in unit.members
                             if state.step_states.get(m["step_id"]) == "complete"]
                            if unit.is_block else [])
                    verdict, lat, nf = vlm.poll_and_check(video, ctx.t, plan, unit, completed=done)
                    vlm_calls += 1
                    frames_sent += nf
                    vlm_latency += lat
                    last_poll[uid] = ctx.t
                    poll_counts[uid] = npolled + 1
                    cand_ids = ({m["step_id"] for m in unit.members} if unit.is_block
                                else {uid})
                    if verdict and verdict.get("step_id") in cand_ids:
                        sid = verdict["step_id"]
                        # recognition update (foreground label) -- only blocks need this;
                        # a single step is already the foreground unit.
                        if unit.is_block:
                            ctx.vlm_verdicts[sid] = {"status": verdict.get("status"),
                                                     "t": ctx.t,
                                                     "confidence": verdict.get("confidence", 1.0)}
                            if not fg or fg[-1][1] != sid:
                                fg.append((ctx.t, sid))
                            if sid not in member_recog_t:
                                member_recog_t[sid] = ctx.t
                        # emit any reminders the VLM raised for the step it named (deduped)
                        for rem in verdict.get("reminders", []):
                            tag = rem.get("reminder")
                            if not tag or (sid, tag) in emitted_checks:
                                continue
                            emitted_checks.add((sid, tag))
                            cls, sub = _reminder_class(tag)
                            reminders.append({"t": round(ctx.t, 2), "node": sid,
                                              "class": cls, "subtype": sub, "detector": "VLM",
                                              "evidence": rem.get("observed"),
                                              "message": rem.get("message", "")})

    # ---- derive stage_intervals from the foreground timeline ----
    fg = _dedup_fg(fg)
    stage_intervals = []
    for i, (t, uid) in enumerate(fg):
        end = fg[i + 1][0] if i + 1 < len(fg) else duration
        if end > t + 1e-6:
            stage_intervals.append(dict(stage=uid, start_s=round(t, 2), end_s=round(end, 2)))

    # ---- sensor_events = the released raw detector (audio) event log ----
    sensor_events = _released_sensor_events(events, duration)

    # ---- T2 reminders (see docs/REMINDER_RUNTIME.md): M3 (VLM execution checks) were
    #      emitted inline above; add M2 (timing, D6) and M1 (missing_step, FSM state) ----
    reminders.extend(_timing_reminders(plan, state))
    reminders.extend(_missing_step_reminders(plan, state, member_recog_t,
                                             members_observable=vlm is not None))
    reminders.sort(key=lambda r: r["t"])

    cost = dict(audio_on_s=round(duration, 2), vlm_calls=vlm_calls,
                frames_sent=frames_sent, vlm_latency_total_s=round(vlm_latency, 2),
                compute_s=round(compute_s, 2))
    return dict(recording=rec, task_id=plan.task["task_id"], arm="proposed_system",
                stage_intervals=stage_intervals, transition_trace=trace,
                sensor_events=sensor_events, reminders=reminders, cost=cost), duration


def _dedup_fg(fg):
    """Sort by time; if several foreground changes share a time, keep the last."""
    fg = sorted(fg, key=lambda x: x[0])
    out = []
    for t, uid in fg:
        if out and abs(out[-1][0] - t) < 1e-6:
            out[-1] = (t, uid)
        else:
            out.append((t, uid))
    return out


def _released_sensor_events(events, duration):
    return [dict(t_s=round(e["t_s"], 2), primitive=e["primitive"], event=e["event"],
                 confidence=round(e.get("confidence", 1.0), 3)) for e in events]


def _reminder_class(reminder):
    """CC4D check tag -> (scored class, subtype). See docs/REMINDER_EVALUATION.md."""
    if reminder == "timing":
        return ("parameter_violation", "timing")
    # measurement | technique | preparation | temperature
    return ("execution_error", reminder)


def _timing_reminders(plan, state):
    """M2: for each node carrying a `timing` check bound to D6, compare its active
    interval against duration_constraint_s and emit over/undertime. Template-driven:
    only nodes that authored a timing check are checked."""
    out = []
    timer = detlib.TimerChecker()
    for u in plan.units:
        for chk in getattr(u, "checks", []):
            if chk.get("detector") != "D6" or chk.get("reminder") != "timing":
                continue
            if not u.duration_constraint_s:
                continue
            s, c = state.start_time.get(u.uid), state.complete_time.get(u.uid)
            if s is None or c is None:
                continue
            ev = timer.check_duration(u.uid, s, c, u.duration_constraint_s)
            if ev:
                out.append({"t": round(ev["t_s"], 2), "node": u.uid,
                            "class": "parameter_violation", "subtype": "timing",
                            "detector": "D6",
                            "evidence": f"{ev['event']}: {ev['duration_s']}s vs "
                                        f"{ev['expected_s']}s expected",
                            "message": chk.get("detection_criteria", "")})
    return out


def _missing_step_reminders(plan, state, member_recog_t, members_observable=True):
    """M1 (missing_step only; order ignored for now): over the ORIGINAL node DAG, any
    node never executed whose transitive successor WAS executed is a missing step,
    anchored at the earliest executed successor's start (matches Box-1 derivation).
    Generic over any sensorplan; needs the nodes form (skipped for pre-compiled graphs).

    A node is only claimable as missing if it was OBSERVABLE but unobserved. Silent
    block members are observable only when the VLM arm ran (`members_observable`); with
    the VLM off they are merely unsensed, not skipped, so we do not flag them."""
    nodes = plan.raw.get("nodes")
    if not nodes:
        return []
    pre = {n["step_id"]: list(n.get("preconditions", [])) for n in nodes}
    succ = defaultdict(list)
    for sid, ps in pre.items():
        for p in ps:
            succ[p].append(sid)

    member_ids = {m["step_id"] for u in plan.units if u.is_block for m in u.members}

    def start_of(sid):
        if sid in member_ids:
            return member_recog_t.get(sid)
        return state.start_time.get(sid)

    def executed(sid):
        return start_of(sid) is not None

    out = []
    for sid in pre:
        if executed(sid):
            continue
        if sid in member_ids and not members_observable:
            continue  # silent member with no VLM running -> unobserved, not missing
        # earliest executed transitive successor
        seen, stack, anchor = set(), list(succ[sid]), None
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            t = start_of(x)
            if t is not None:
                anchor = t if anchor is None else min(anchor, t)
            stack.extend(succ[x])
        if anchor is not None:
            out.append({"t": round(anchor, 2), "node": sid,
                        "class": "precondition_violation", "subtype": "missing_step",
                        "detector": "none",
                        "evidence": "node never recognized; a downstream step ran",
                        "message": f"missing step: {sid}"})
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(BASE, "tasks/cc4d/spicedhotchocolate.sensorplan.json"))
    ap.add_argument("--recs", default=",".join(RECS))
    ap.add_argument("--vlm", choices=["off", "mock", "qwen"], default="off")
    ap.add_argument("--video-dir", default=os.path.join(BASE, "data/videos_360p"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    plan = plan_loader.load_plan(a.plan)
    if a.vlm != "off":
        import proposed_vlm_arm as vlm_step
    os.makedirs(a.out_dir, exist_ok=True)
    # sidecar: the COMPILED plan (graph form) the runtime actually drove. The visualizer
    # renders its plan panel from this, so the units line up with the run's stages even
    # though the authored artifact is the nodes-form sensorplan (which has no graph).
    with open(os.path.join(a.out_dir, "_compiled_plan.json"), "w") as fh:
        json.dump({"task": plan.task, "vlm_policy": plan.vlm_policy,
                   "graph": plan.raw["graph"]}, fh, indent=1)

    for rec in a.recs.split(","):
        # fresh arm per recording (independent poll state / trace files)
        vlm = (vlm_step.VLMArm(mode=a.vlm, trace_dir=a.out_dir if a.trace else None,
                               recording=rec)
               if a.vlm != "off" else None)
        video = os.path.join(a.video_dir, f"{rec}.mp4")
        res, dur = run_recording(plan, rec, vlm=vlm,
                                 video=video if a.vlm != "off" else None,
                                 trace_dir=a.out_dir if a.trace else None)
        with open(os.path.join(a.out_dir, f"{rec}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        stages = " | ".join(f"{s['stage']} {s['start_s']}-{s['end_s']}"
                            for s in res["stage_intervals"])
        print(f"[{rec}] dur={dur:.0f}s vlm_calls={res['cost']['vlm_calls']}")
        print(f"   stages: {stages}")
    print("\nwrote results to", a.out_dir)


if __name__ == "__main__":
    main()
