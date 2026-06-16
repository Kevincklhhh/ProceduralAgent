"""monitor_runtime -- execute a compiled procedure-monitor plan.

Ingests a recording's audio (+ video for the VLM arm), runs the bound detectors
once, releases their events causally (visible only at t >= event_time + latency,
every latency <= 10 s so this equals a true per-tick streaming detector), and
drives the plan's graph state machine on a fixed tick. Emits the plan output:
stage_intervals + transition_trace + sensor_events + cost.

Generalizes the hardcoded chain in eval/engine.py into a predicate-driven loop:
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eval/
sys.path.insert(0, os.path.join(BASE, 'detectors'))              # detectors/
import plan_loader
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

        # 3. VLM arm: while a C-none block is active, poll it (cadence + cost capped)
        #    to label its silent members. Keyed on the BLOCK, not the foreground
        #    label (which becomes a member after the first poll).
        if vlm is not None:
            blk = next((u for u in plan.units if u.is_block
                        and u.sensing_role == "C-none" and u.vlm.get("poll")
                        and state.step_states.get(u.uid) == "active"), None)
            if blk is not None:
                pol = plan.vlm_policy
                bid = blk.uid
                since = ctx.t - last_poll.get(bid, -1e9)
                npolled = poll_counts.get(bid, 0)
                budget_ok = vlm_calls < pol.get("vlm_budget", 1e9)
                cap_ok = npolled < pol.get("max_polls_per_step", 1e9)
                if since >= pol.get("period_s", 10.0) and budget_ok and cap_ok:
                    done = [m["step_id"] for m in blk.members
                            if state.step_states.get(m["step_id"]) == "complete"]
                    verdict, lat, nf = vlm.poll(video, ctx.t, plan, blk, completed=done)
                    vlm_calls += 1
                    frames_sent += nf
                    vlm_latency += lat
                    last_poll[bid] = ctx.t
                    poll_counts[bid] = npolled + 1
                    if verdict and verdict.get("step_id") in {m["step_id"] for m in blk.members}:
                        mid = verdict["step_id"]
                        ctx.vlm_verdicts[mid] = {"status": verdict.get("status"),
                                                 "t": ctx.t,
                                                 "confidence": verdict.get("confidence", 1.0)}
                        if not fg or fg[-1][1] != mid:
                            fg.append((ctx.t, mid))

    # ---- derive stage_intervals from the foreground timeline ----
    fg = _dedup_fg(fg)
    stage_intervals = []
    for i, (t, uid) in enumerate(fg):
        end = fg[i + 1][0] if i + 1 < len(fg) else duration
        if end > t + 1e-6:
            stage_intervals.append(dict(stage=uid, start_s=round(t, 2), end_s=round(end, 2)))

    # ---- D6 duration checks on completed timed steps -> sensor_events ----
    sensor_events = _released_sensor_events(events, duration)
    sensor_events += _duration_checks(plan, state)

    cost = dict(audio_on_s=round(duration, 2), vlm_calls=vlm_calls,
                frames_sent=frames_sent, vlm_latency_total_s=round(vlm_latency, 2),
                compute_s=round(compute_s, 2))
    return dict(recording=rec, task_id=plan.task["task_id"], arm="proposed_system",
                stage_intervals=stage_intervals, transition_trace=trace,
                sensor_events=sensor_events, cost=cost), duration


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


def _duration_checks(plan, state):
    out = []
    timer = detlib.TimerChecker()
    for u in plan.units:
        if not u.duration_constraint_s:
            continue
        s, c = state.start_time.get(u.uid), state.complete_time.get(u.uid)
        if s is None or c is None:
            continue
        ev = (timer.check_duration(u.uid, s, c, u.duration_constraint_s))
        if ev:
            out.append(dict(t_s=ev["t_s"], primitive="D6", event=ev["event"],
                            confidence=ev["confidence"], step_id=u.uid,
                            duration_s=ev["duration_s"], expected_s=ev["expected_s"]))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(BASE, "tasks/cc4d/spicedhotchocolate.monitor.json"))
    ap.add_argument("--recs", default=",".join(RECS))
    ap.add_argument("--vlm", choices=["off", "mock", "qwen"], default="off")
    ap.add_argument("--video-dir", default=os.path.join(BASE, "data/videos_360p"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    plan = plan_loader.load_plan(a.plan)
    if a.vlm != "off":
        import vlm_step
    os.makedirs(a.out_dir, exist_ok=True)

    for rec in a.recs.split(","):
        # fresh arm per recording (independent poll state / trace files)
        vlm = (vlm_step.VLMArm(mode=a.vlm, trace_dir=a.out_dir if a.trace else None)
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
