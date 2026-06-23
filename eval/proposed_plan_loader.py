"""plan_loader -- turn an abstract procedure-monitor plan JSON into executable objects.

Accepts either schema: the canonical Stage-2 **sensor plan** (nodes form, schema 2.0,
`tasks/cc4d/spicedhotchocolate.sensorplan.json`) -- compiled to the executable graph
by `sensorplan_to_graph()` at load time -- or, directly, an already-compiled graph plan
(schema 0.6, `tasks/cc4d/spicedhotchocolate.monitor.json`, kept as the golden reference).
Returns a `CompiledPlan` the runtime can drive: each `start_when` /
`complete_when` rule's structured `cond` becomes a predicate closure
`pred(ctx, unit_id) -> Match|None`, each `state_update` stays a list of typed ops,
every `monitor.primitives` id binds to a detector in `detectors/runtime.REGISTRY`
(or the VLM), and a per-event latency table feeds the runtime's causal gate.

Condition grammar (leaves): eligible, sensor_event, next_anchor, step_state,
elapsed, vlm_verdict; combinators: all, any. State-update ops: set_state,
set_foreground, open, mark_members, add_background, remove_background.
"""
import json
import os
import sys
from collections import namedtuple

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE, 'detectors') not in sys.path:
    sys.path.insert(0, os.path.join(BASE, 'detectors'))
import runtime as detlib  # detectors/runtime package (REGISTRY, detector classes)

# --------------------------------------------------------------------------
# per-event causal latency L (s): an event detected at audio-time e is visible
# to the state machine only at t >= e + L. Every L <= 10 s (the online buffer).
# --------------------------------------------------------------------------
LATENCY = {
    ("D1", "cycle_start"): 8.0, ("D1", "cycle_end"): 8.0,
    ("D2", "motor_on"): 6.0, ("D2", "motor_off"): 6.0,
    ("D3", "cook_end"): 10.0,
    ("D4", "cook_start_candidate"): 8.0,
    ("D5", "water_on"): 2.0, ("D5", "water_off"): 2.0,
}
DEFAULT_LATENCY = 10.0


def latency(primitive, event):
    return LATENCY.get((primitive, event), DEFAULT_LATENCY)


# Legacy DETECTOR_CATALOG ids (A*/R*/V1) -> runtime (D*/VLM). Compiled plans should
# already use D*, but this lets the loader accept older/compiler-emitted ids.
LEGACY_MAP = {
    "A0": ("D1", None), "A1": ("D1", "cycle_start"), "A2": ("D1", "cycle_end"),
    "A10": ("D6", None), "V1": ("VLM", None),
}


def normalize_primitive(pid):
    return LEGACY_MAP.get(pid, (pid, None))[0]


# --------------------------------------------------------------------------
# Match: what a satisfied predicate returns. `consumes` lists events the rule
# eats (consume-once) so a later rule can't re-fire on the same event.
# --------------------------------------------------------------------------
Match = namedtuple("Match", ["t_fire", "evidence", "confidence", "consumes"])


def _resolve_anchor(anchor, state):
    """'step.start'|'step.complete' -> the recorded time, or None if not set yet."""
    if not anchor:
        return None
    sid, kind = anchor.rsplit(".", 1)
    return (state.start_time if kind == "start" else state.complete_time).get(sid)


# --------------------------------------------------------------------------
# leaf compilers: each returns pred(ctx, unit) -> Match|None
# --------------------------------------------------------------------------
def _compile_eligible(spec, unit):
    requires = unit["requires"]

    def pred(ctx, uid):
        st = ctx.state.step_states.get(uid, "not_started")
        if st not in ("not_started", "eligible"):
            return None
        times = [ctx.state.complete_time.get(r) for r in requires]
        if any(t is None for t in times):
            return None
        return Match(max(times) if times else 0.0, [], 1.0, [])
    return pred


def _event_match(ctx, primitive, event, after, min_field, conf_min):
    after_t = _resolve_anchor(after, ctx.state)
    if after is not None and after_t is None:
        return None  # anchor not reached yet -> nothing qualifies
    best = None
    for e in ctx.events_up_to(ctx.t):
        if e["primitive"] != primitive or e["event"] != event:
            continue
        if id(e) in ctx.consumed:
            continue
        if after_t is not None and e["t_s"] < after_t:
            continue
        if conf_min and e.get("confidence", 1.0) < conf_min:
            continue
        if min_field:
            ok = all(e.get(k, float("-inf")) >= v for k, v in min_field.items())
            if not ok:
                continue
        if best is None or e["t_s"] < best["t_s"]:
            best = e
    return best


def _compile_sensor_event(spec, unit):
    primitive = normalize_primitive(spec["primitive"])
    event, after = spec["event"], spec.get("after")
    min_field, conf_min = spec.get("min_field"), spec.get("confidence_min")

    def pred(ctx, uid):
        e = _event_match(ctx, primitive, event, after, min_field, conf_min)
        if e is None:
            return None
        return Match(e["t_s"], [e], e.get("confidence", 1.0), [e])  # consuming
    return pred


def _compile_next_anchor(spec, unit):
    primitive = normalize_primitive(spec["primitive"])
    event, after = spec["event"], spec.get("after")

    def pred(ctx, uid):
        e = _event_match(ctx, primitive, event, after, None, None)
        if e is None:
            return None
        return Match(e["t_s"], [e], e.get("confidence", 1.0), [])  # non-consuming
    return pred


def _compile_step_state(spec, unit):
    step, want = spec["step"], spec["is"]

    def pred(ctx, uid):
        if ctx.state.step_states.get(step) != want:
            return None
        t = (ctx.state.complete_time.get(step) if want == "complete"
             else ctx.state.start_time.get(step))
        return Match(t if t is not None else ctx.t, [], 1.0, [])
    return pred


def _compile_elapsed(spec, unit):
    step = spec.get("step")
    min_s, max_s = spec.get("min_s"), spec.get("max_s")
    fires_at = spec.get("fires_at", "max" if max_s is not None else "min")

    def pred(ctx, uid):
        ref = ctx.state.start_time.get(step)
        if ref is None:
            return None
        if fires_at == "max" and max_s is not None:
            return Match(ref + max_s, [], 0.5, []) if ctx.t - ref >= max_s else None
        if fires_at == "min" and min_s is not None:
            return Match(ref + min_s, [], 0.5, []) if ctx.t - ref >= min_s else None
        return None
    return pred


def _compile_vlm_verdict(spec, unit):
    for_step = spec["for_step"]
    expect = set(spec.get("expect_status", []))

    def pred(ctx, uid):
        v = ctx.vlm_verdicts.get(for_step)
        if v and (not expect or v.get("status") in expect):
            return Match(v.get("t", ctx.t), [v], v.get("confidence", 1.0), [])
        return None
    return pred


def _compile_any(spec, unit):
    branches = [compile_cond(c, unit) for c in spec["of"]]

    def pred(ctx, uid):
        hits = [b(ctx, uid) for b in branches]
        hits = [h for h in hits if h is not None]
        if not hits:
            return None
        return min(hits, key=lambda m: m.t_fire)  # earliest-firing branch wins
    return pred


def _compile_all(spec, unit):
    branches = [compile_cond(c, unit) for c in spec["of"]]

    def pred(ctx, uid):
        hits = [b(ctx, uid) for b in branches]
        if any(h is None for h in hits):
            return None
        ev, cons = [], []
        for h in hits:
            ev += h.evidence
            cons += h.consumes
        return Match(max(h.t_fire for h in hits), ev,
                     min(h.confidence for h in hits), cons)
    return pred


_LEAF = {
    "eligible": _compile_eligible, "sensor_event": _compile_sensor_event,
    "next_anchor": _compile_next_anchor, "step_state": _compile_step_state,
    "elapsed": _compile_elapsed, "vlm_verdict": _compile_vlm_verdict,
    "any": _compile_any, "all": _compile_all,
}


def compile_cond(spec, unit):
    t = spec["type"]
    if t not in _LEAF:
        raise ValueError(f"unknown cond type: {t}")
    return _LEAF[t](spec, unit)


def _cond_references_vlm(spec):
    if spec["type"] == "vlm_verdict":
        return True
    if spec["type"] in ("any", "all"):
        return any(_cond_references_vlm(c) for c in spec["of"])
    return False


# --------------------------------------------------------------------------
# compiled containers
# --------------------------------------------------------------------------
class CompiledRule:
    def __init__(self, raw, unit):
        self.rule_id = raw["rule_id"]
        self.when = raw.get("when", "")
        self.pred = compile_cond(raw["cond"], unit)
        self.state_update = raw.get("state_update", [])
        self.needs_vlm = _cond_references_vlm(raw["cond"])
        self.opens = [s for op in self.state_update if op["op"] == "open"
                      for s in op.get("steps", [])]


class CompiledUnit:
    """A step or a step_block, uniformly."""
    def __init__(self, raw, is_block):
        self.is_block = is_block
        self.uid = raw["block_id"] if is_block else raw["step_id"]
        self.order = raw.get("order", 0)
        self.requires = raw.get("requires", [])
        self.produces = raw.get("produces", [])
        self.sensing_role = raw.get("sensing_role")
        self.monitor = raw.get("monitor", {})
        self.instruction = raw.get("instruction", "")
        self.duration_constraint_s = raw.get("duration_constraint_s")
        self.cc4d_step_id = raw.get("cc4d_step_id")
        self.members = raw.get("members", []) if is_block else []
        self.vlm = raw.get("vlm", {})
        # T2 reminder hooks (recipe-derived; consumed by the reminder runtime, not the FSM)
        self.checks = raw.get("checks", [])
        u = {"requires": self.requires}
        self.start_rules = [CompiledRule(r, u) for r in raw.get("start_when", [])]
        self.complete_rules = [CompiledRule(r, u) for r in raw.get("complete_when", [])]


class CompiledPlan:
    def __init__(self, plan):
        self.raw = plan
        self.task = plan["task"]
        self.vlm_policy = plan.get("vlm_policy", {})
        self.tick_s = plan.get("runtime_config", {}).get("tick_s", 1.0)
        g = plan["graph"]
        self.units = ([CompiledUnit(s, False) for s in g.get("steps", [])]
                      + [CompiledUnit(b, True) for b in g.get("step_blocks", [])])
        self.units.sort(key=lambda u: u.order)
        self.by_id = {u.uid: u for u in self.units}
        self.edges = g.get("edges", [])
        # which detector primitives must run, bound to REGISTRY classes
        self.primitives_used = set()
        for u in self.units:
            for p in u.monitor.get("primitives", []):
                self.primitives_used.add(normalize_primitive(p))
        self.detector_bindings = {p: detlib.REGISTRY[p]
                                  for p in self.primitives_used
                                  if p in detlib.REGISTRY}

    def audio_primitives(self):
        """Detector ids that produce a released event stream (exclude D6 logic / VLM)."""
        return [p for p in self.primitives_used
                if p in self.detector_bindings and p != "D6"]

    def latency(self, primitive, event):
        return latency(primitive, event)


# --------------------------------------------------------------------------
# Stage-2 sensor plan (nodes form, schema 2.0) -> executable graph (schema 0.6).
# Deterministic compile, no prose parsing: each node's recognition/completion is
# read by its `detector` + `inferred_from` + `preconditions` only.
#   * D1/none-recognized nodes  -> individual `steps`
#   * the contiguous run of VLM-recognized nodes -> one C-none `step_block`
#     (the silent middle: the unordered adds + the mix)
# A node recognized by D1 is "the appliance is running" => its start is a
# cycle_start; if its completion has no own end signal (inferred_from set) it
# closes on the NEXT cycle_start, else on its own cycle_end. Every rule also
# carries an elapsed fallback so the machine still advances if D1 is missing.
# --------------------------------------------------------------------------
BLOCK_ID = "quiet_middle"


def _recog_detector(node):
    return node.get("recognition", {}).get("detector") or "none"


def _fallback_max_s(node, is_block=False):
    if is_block:
        return 300
    d = node.get("duration_constraint_s")
    return (d + 30) if d else 120


def sensorplan_to_graph(plan):
    nodes = plan["nodes"]
    silent = [n["step_id"] for n in nodes if _recog_detector(n) == "VLM"]
    in_block = set(silent)

    def remap(reqs):
        out = []
        for r in reqs:
            r2 = BLOCK_ID if r in in_block else r
            if r2 not in out:
                out.append(r2)
        return out

    # unit list (preserving plan order); the block takes the slot of its 1st member
    unit_ids, block_emitted = [], False
    for n in nodes:
        if n["step_id"] in in_block:
            if not block_emitted:
                unit_ids.append(BLOCK_ID)
                block_emitted = True
        else:
            unit_ids.append(n["step_id"])
    order_of = {uid: i + 1 for i, uid in enumerate(unit_ids)}

    block_requires = remap([r for sid in silent for r in
                            next(n for n in nodes if n["step_id"] == sid)["preconditions"]])
    block_requires = [r for r in block_requires if r != BLOCK_ID]

    def requires_of(uid):
        if uid == BLOCK_ID:
            return block_requires
        n = next(x for x in nodes if x["step_id"] == uid)
        return remap(n["preconditions"])

    def children_of(uid):
        return [v for v in unit_ids if uid in requires_of(v)]

    steps, block = [], None
    for n in nodes:
        sid = n["step_id"]
        if sid in in_block:
            continue
        det_r = _recog_detector(n)
        requires = requires_of(sid)
        prims = ["D1"]
        if n.get("duration_constraint_s"):
            prims.append("D6")

        if det_r == "none" or not n["preconditions"]:
            start_cond = {"type": "eligible"}
        else:
            after = (BLOCK_ID if n["preconditions"][0] in in_block
                     else n["preconditions"][0]) + ".start"
            start_cond = {"type": "sensor_event", "primitive": "D1",
                          "event": "cycle_start", "after": after}

        inferred = n.get("completion", {}).get("inferred_from")
        own_end = {"type": "sensor_event", "primitive": "D1", "event": "cycle_end",
                   "after": sid + ".start"}
        next_anchor = {"type": "next_anchor", "primitive": "D1", "event": "cycle_start",
                       "after": sid + ".start"}
        fallback = {"type": "elapsed", "step": sid,
                    "max_s": _fallback_max_s(n), "fires_at": "max"}
        comp_cond = {"type": "any", "of": [next_anchor if inferred else own_end, fallback]}

        complete_update = [{"op": "set_state", "step": sid, "to": "complete"}]
        opens = children_of(sid)
        if opens:
            complete_update.append({"op": "open", "steps": opens})

        steps.append({
            "step_id": sid, "order": order_of[sid], "instruction": n.get("instruction", ""),
            "requires": requires, "sensing_role": n.get("sensing_role"),
            "duration_constraint_s": n.get("duration_constraint_s"),
            "checks": n.get("checks", []),
            "monitor": {"mode": "foreground", "primitives": prims},
            "start_when": [{"rule_id": f"{sid}_start", "cond": start_cond,
                            "state_update": [{"op": "set_state", "step": sid, "to": "active"},
                                             {"op": "set_foreground", "step": sid}]}],
            "complete_when": [{"rule_id": f"{sid}_done", "cond": comp_cond,
                               "state_update": complete_update}],
        })

    if silent:
        members = [{"step_id": sid,
                    "instruction": next(n for n in nodes if n["step_id"] == sid)
                    .get("instruction", ""),
                    "checks": next(n for n in nodes if n["step_id"] == sid)
                    .get("checks", [])}
                   for sid in silent]
        opens = children_of(BLOCK_ID)
        block_done_update = [{"op": "mark_members", "block": BLOCK_ID, "rest": "unknown"},
                             {"op": "set_state", "step": BLOCK_ID, "to": "complete"}]
        if opens:
            block_done_update.append({"op": "open", "steps": opens})
        block = {
            "block_id": BLOCK_ID, "order": order_of[BLOCK_ID], "members": members,
            "requires": block_requires, "sensing_role": "C-none",
            "monitor": {"mode": "block", "primitives": ["VLM"]},
            "vlm": {"poll": True,
                    "question": "Which add/mix step is the user doing now?"},
            "start_when": [{"rule_id": f"{BLOCK_ID}_start", "cond": {"type": "eligible"},
                            "state_update": [{"op": "set_state", "step": BLOCK_ID, "to": "active"},
                                             {"op": "set_foreground", "step": BLOCK_ID}]}],
            "complete_when": [{"rule_id": f"{BLOCK_ID}_done", "cond": {"type": "any", "of": [
                {"type": "next_anchor", "primitive": "D1", "event": "cycle_start",
                 "after": BLOCK_ID + ".start"},
                {"type": "elapsed", "step": BLOCK_ID,
                 "max_s": _fallback_max_s(None, is_block=True), "fires_at": "max"}]},
                "state_update": block_done_update}],
        }

    edges = [{"from": r, "to": uid, "relation": "precedes"}
             for uid in unit_ids for r in requires_of(uid)]
    return {"steps": steps, "step_blocks": [block] if block else [], "edges": edges}


def load_plan(path):
    with open(path) as fh:
        plan = json.load(fh)
    if plan.get("artifact_type") == "sensor_control_plan" or "nodes" in plan:
        plan["graph"] = sensorplan_to_graph(plan)
    return CompiledPlan(plan)


if __name__ == "__main__":
    p = load_plan(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(BASE, "tasks/cc4d/spicedhotchocolate.sensorplan.json"))
    print("task:", p.task["task_id"], "| tick", p.tick_s, "s")
    print("primitives used:", sorted(p.primitives_used),
          "| audio:", p.audio_primitives())
    for u in p.units:
        kind = "block" if u.is_block else "step"
        print(f"  [{u.order}] {u.uid} ({kind}, {u.sensing_role}) "
              f"requires={u.requires} start={[r.rule_id for r in u.start_rules]} "
              f"complete={[r.rule_id for r in u.complete_rules]} "
              f"needs_vlm={[r.needs_vlm for r in u.complete_rules]}")
