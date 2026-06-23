#!/usr/bin/env python3
"""sensorplan_runtime -- causal replay driven by the GOLDEN stage-2 sensor plan.

Consumes tasks/cc4d/spicedhotchocolate.sensorplan.json (transition-centric:
per-boundary {detector, detection_criteria}) and assigns a step to every time t
of a recording. Embodies the sensor-control thesis: cheap audio (D1 microwave
detector) settles the fill / microwave / heat boundaries for free, and the
expensive VLM is DUTY-CYCLED to only the silent middle (the adds + mix), where it
runs as an ORDER-ENFORCED current-step detector (the eval/baseline_t1_step.py
mechanism: a prediction that regresses below the high-water step is clamped
forward). Running the VLM over the whole video would just be that baseline; here
it touches only [cycle_end#1, cycle_start#2].

The sensor plan is GOLDEN and read-only: the silent set, the two D1-anchored steps
(microwave_initial, heat_serve) and the ordered chain are all DERIVED from it.
String<->numeric step-id mapping is done by `order` against the recipe at load
time (the plan stays clean, no cc4d ids baked in).

Output (one JSON per recording, superset of what eval/eval_score_corpus.py and
eval/proposed_verify.py read):
  {recording, task_id, arm, stage_intervals, transition_trace, sensor_events,
   cost, events:[], escalation_requests:[], _meta}

Usage:
  python eval/sensorplan_runtime.py --selftest
  QWEN_VIDEO_SERVER_URL=http://saltyfish.eecs.umich.edu:8000 QWEN_VIDEO_MODEL=Qwen/Qwen3.6-27B \
    python eval/sensorplan_runtime.py --vlm qwen --arm sensorplan_qwen --trace
(--vlm off is the audio-only ablation: D1 anchors, no VLM in the middle.)
"""
import argparse
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # eval/
sys.path.insert(0, os.path.join(BASE, 'detectors'))  # detectors/  -> `import runtime`
import runtime as detlib                            # detectors/runtime (REGISTRY, audio_io)
from runtime import audio_io

RECS = ['8_16', '8_3', '8_25', '8_26', '8_31', '8_50']
PLAN = os.path.join(BASE, 'tasks/cc4d/spicedhotchocolate.sensorplan.json')
RECIPE = os.path.join(BASE, 'tasks/cc4d/spicedhotchocolate.json')
OUT_DIR = os.path.join(BASE, 'experiments/proposed_system')   # so the visualizer monitor panel reads it

D1_LATENCY = 8.0          # proposed_plan_loader.LATENCY[("D1", cycle_start|cycle_end)]
# elapsed-timer fallbacks when D1 anchors are missing (recipe/sensorplan durations)
FALLBACK = dict(fill_s=120.0, microwave_s=60.0, middle_s=300.0, heat_s=60.0)


# --------------------------------------------------------------------------
# load the golden sensor plan + recipe, build the id map and the silent set
# --------------------------------------------------------------------------
def load_plan(plan_path, recipe_path):
    plan = json.load(open(plan_path))
    recipe = json.load(open(recipe_path))
    # ordered string-id chain: procedure_start then each transition's `to` (drop DONE)
    chain = [plan['procedure_start']['step']]
    for tr in plan['transitions']:
        if tr['to'] != 'DONE':
            chain.append(tr['to'])
    # map by order (chain idx 0 -> recipe order 1)
    by_order = {s['order']: s for s in recipe['steps']}
    if sorted(by_order) != list(range(1, len(chain) + 1)):
        raise SystemExit(f"recipe orders {sorted(by_order)} are not 1..{len(chain)}")
    num2str, str2num, str_step = {}, {}, {}
    for i, sid in enumerate(chain):
        rs = by_order[i + 1]
        num2str[rs['step_id']] = sid
        str2num[sid] = rs['step_id']
        str_step[sid] = dict(step_id=rs['step_id'], order=rs['order'],
                             instruction=rs['instruction'])
    # silent middle = steps whose next_start is resolved by the VLM (derived from the plan)
    silent_ids = [tr['to'] for tr in plan['transitions']
                  if tr['next_start'].get('detector') == 'VLM' and tr['to'] != 'DONE']
    # the two D1-anchored steps: `to` of a transition whose next_start detector is D1
    d1_started = [tr['to'] for tr in plan['transitions']
                  if tr['next_start'].get('detector') == 'D1']
    return dict(title=recipe['title'], chain=chain, num2str=num2str, str2num=str2num,
                str_step=str_step, silent_ids=silent_ids, d1_started=d1_started,
                first_id=chain[0])


# --------------------------------------------------------------------------
# D1 audio -> microwave cycles -> four anchors  (A,B = cycle 1; C,D = cycle 2)
# --------------------------------------------------------------------------
def run_d1(rec):
    fs, x16 = audio_io.load_audio_16k(rec)
    duration = len(x16) / fs
    t0 = time.perf_counter()
    events = detlib.REGISTRY['D1']().detect(x16, fs)
    compute_s = time.perf_counter() - t0
    for e in events:
        e['release_t'] = round(e['t_s'] + D1_LATENCY, 2)
    starts = sorted((e for e in events if e['event'] == 'cycle_start'), key=lambda e: e['t_s'])
    ends = sorted((e for e in events if e['event'] == 'cycle_end'), key=lambda e: e['t_s'])
    cycles = []
    for i, s in enumerate(starts):
        end = ends[i]['t_s'] if i < len(ends) and ends[i]['t_s'] >= s['t_s'] else None
        if end is None:
            after = [e['t_s'] for e in ends if e['t_s'] >= s['t_s']]
            end = after[0] if after else s['t_s'] + FALLBACK['microwave_s']
        cycles.append((float(s['t_s']), float(end)))
    return cycles, events, duration, compute_s


def anchors_from_cycles(cycles, duration):
    """Return (A,B,C,D) microwave anchors + a 'degraded' tag, with timer fallbacks."""
    degraded = None
    if len(cycles) >= 2:
        (A, B), (C, D) = cycles[0], cycles[1]
        if len(cycles) > 2:
            degraded = 'extra_cycles'
    elif len(cycles) == 1:
        A, B = cycles[0]
        C, D = B + FALLBACK['middle_s'], B + FALLBACK['middle_s'] + FALLBACK['heat_s']
        degraded = 'one_cycle'
    else:
        A = FALLBACK['fill_s']
        B = A + FALLBACK['microwave_s']
        C = B + FALLBACK['middle_s']
        D = C + FALLBACK['heat_s']
        degraded = 'no_d1_anchors'
    A, B, C, D = (max(0.0, min(v, duration)) for v in (A, B, C, D))
    A, B, C, D = sorted([A, B, C, D])            # guard monotonicity
    return (A, B, C, D), degraded


# --------------------------------------------------------------------------
# order-enforced VLM, restricted to the silent steps, bounded to [t_lo, t_hi)
# (factored from eval/baseline_t1_step.run_one; clamp logic identical)
# --------------------------------------------------------------------------
def run_vlm_window(video, backend, interval, t_lo, t_hi, silent_steps, title, trace_fh):
    from baseline_t1_step import (sample_frames_1fps, safe_json, build_user_prompt)
    silent_steps = sorted(silent_steps, key=lambda s: s['order'])
    by_order = {s['order']: s for s in silent_steps}
    # the VLM is shown the STRING step ids (add_cinnamon, ...) and returns them; also
    # accept the numeric recipe id as a fallback. Either resolves to the step's order.
    valid = {}
    for s in silent_steps:
        valid[str(s['step_id'])] = s['order']
        if s.get('num') is not None:
            valid[str(s['num'])] = s['order']
    lo_o, hi_o = silent_steps[0]['order'], silent_steps[-1]['order']
    task = {'title': title, 'steps': silent_steps}

    out, history = [], []
    vlm_calls = frames_sent = 0
    lat_total = 0.0
    high_water = 0
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    t = t_lo + interval
    while t <= t_hi + 1e-6:
        status = evidence = raw = err = None
        jpegs = sample_frames_1fps(cap, fps, t, interval)
        frames_sent += len(jpegs)
        completed = [s for s in silent_steps if s['order'] < high_water]
        prompt = build_user_prompt(task, history[-5:], completed)
        t0 = time.time()
        try:
            raw = backend.call(jpegs, prompt)
        except Exception as e:
            raw, err = '', f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        lat_total += lat
        vlm_calls += 1
        parsed = safe_json(raw) or {}
        status, evidence = parsed.get('status'), parsed.get('evidence')
        sid = str(parsed.get('step_id'))
        o = valid.get(sid, high_water or lo_o)
        # clamp into the silent range and enforce monotonic (no regression)
        o = max(lo_o, min(o, hi_o))
        o = max(o, high_water) if high_water else o
        high_water = max(high_water, o)
        step_str = by_order[o]['step_id']                # already the sensorplan string id

        s0, e0 = max(t_lo, t - interval), min(t, t_hi)
        if e0 > s0 + 1e-6:
            out.append(dict(stage=step_str, start_s=round(s0, 1), end_s=round(e0, 1), _ev='VLM'))
        history.append(dict(t=round(t, 1), step=step_str,
                            status=status or '?', evidence=evidence or ''))
        if trace_fh is not None:
            trace_fh.write(json.dumps(dict(t=round(t, 1), start_s=round(s0, 1),
                                           end_s=round(e0, 1), pred_step=step_str,
                                           status=status, evidence=evidence, raw=raw,
                                           error=err, latency_s=round(lat, 2))) + '\n')
            trace_fh.flush()
        t += interval

    if cap is not None:
        cap.release()
    if not out and t_hi > t_lo + 1e-6:           # short/empty middle -> single placeholder
        out.append(dict(stage=by_order[lo_o]['step_id'],
                        start_s=round(t_lo, 1), end_s=round(t_hi, 1), _ev='VLM'))
    return out, dict(vlm_calls=vlm_calls, frames_sent=frames_sent,
                     vlm_latency_total_s=round(lat_total, 2))


# --------------------------------------------------------------------------
def merge_runs(intervals):
    """Collapse adjacent intervals with the same stage."""
    out = []
    for iv in intervals:
        if out and out[-1]['stage'] == iv['stage'] and abs(out[-1]['end_s'] - iv['start_s']) < 1e-6:
            out[-1]['end_s'] = iv['end_s']
        else:
            out.append(dict(iv))
    return out


EVIDENCE = {'microwave_initial': 'D1.cycle_start', 'heat_serve': 'D1.cycle_start'}


def assemble(rec, P, anchors, middle, events, duration, costs, degraded, mode, interval):
    A, B, C, D = anchors
    raw = []

    def add(stage, s, e, ev):
        if e > s + 1e-6:
            raw.append(dict(stage=stage, start_s=round(s, 1), end_s=round(e, 1), _ev=ev))

    add(P['first_id'], 0.0, A, 'procedure_start (t=0)')
    mw = P['d1_started'][0] if P['d1_started'] else 'microwave_initial'
    heat = P['d1_started'][1] if len(P['d1_started']) > 1 else 'heat_serve'
    add(mw, A, B, 'D1.cycle_start')
    raw += middle
    add(heat, C, D, 'D1.cycle_start')

    merged = merge_runs(raw)
    stage_intervals = [dict(stage=iv['stage'], start_s=iv['start_s'], end_s=iv['end_s'])
                       for iv in merged]
    trace = [dict(t_s=iv['start_s'], transition='start', step_or_block=iv['stage'],
                  evidence=[iv.get('_ev', EVIDENCE.get(iv['stage'], 'VLM'))], confidence=1.0)
             for iv in merged]
    sensor_events = [dict(t_s=e['t_s'], primitive=e['primitive'], event=e['event'],
                          confidence=e.get('confidence', 1.0), release_t=e.get('release_t'))
                     for e in events]
    cost = dict(audio_on_s=round(duration, 2), vlm_calls=costs['vlm_calls'],
                frames_sent=costs['frames_sent'],
                vlm_latency_total_s=costs['vlm_latency_total_s'],
                compute_s=round(costs['compute_s'], 2))
    return dict(recording=rec, task_id='spicedhotchocolate', arm=None,
                stage_intervals=stage_intervals, transition_trace=trace,
                sensor_events=sensor_events, cost=cost, events=[], escalation_requests=[],
                _meta=dict(degraded=degraded, vlm_mode=mode, interval_s=interval,
                           anchors=dict(A=round(A, 1), B=round(B, 1),
                                        C=round(C, 1), D=round(D, 1))))


# --------------------------------------------------------------------------
def run_recording(P, rec, mode, video_dir, interval, out_dir, arm, trace):
    cycles, events, duration, compute_s = run_d1(rec)
    (A, B, C, D), degraded = anchors_from_cycles(cycles, duration)
    # silent steps carry the STRING id (shown to + returned by the VLM) + numeric fallback
    silent_steps = [dict(step_id=sid, num=P['str2num'][sid],
                         order=P['str_step'][sid]['order'],
                         instruction=P['str_step'][sid]['instruction'])
                    for sid in P['silent_ids']]

    trace_fh = None
    if trace:
        tdir = os.path.join(out_dir, arm, 'traces')
        os.makedirs(tdir, exist_ok=True)
        trace_fh = open(os.path.join(tdir, f'{rec}.jsonl'), 'w')

    backend = None
    if mode == 'qwen':
        from baseline_t1_step import Qwen
        backend = Qwen()
    video = os.path.join(video_dir, f'{rec}.mp4')
    if mode != 'off':
        middle, vcost = run_vlm_window(video, backend, interval, B, C,
                                       silent_steps, P['title'], trace_fh)
    else:
        middle, vcost = [], dict(vlm_calls=0, frames_sent=0, vlm_latency_total_s=0.0)
    if trace_fh is not None:
        trace_fh.close()

    vcost['compute_s'] = compute_s
    res = assemble(rec, P, (A, B, C, D), middle, events, duration, vcost, degraded, mode, interval)
    res['arm'] = arm
    os.makedirs(os.path.join(out_dir, arm), exist_ok=True)
    with open(os.path.join(out_dir, arm, f'{rec}.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    stages = ' | '.join(f"{s['stage']} {s['start_s']}-{s['end_s']}" for s in res['stage_intervals'])
    print(f"[{rec}] dur={duration:.0f}s cycles={len(cycles)} degraded={degraded} "
          f"anchors=({A:.0f},{B:.0f},{C:.0f},{D:.0f}) vlm_calls={vcost['vlm_calls']}")
    print(f"   {stages}")
    return res


# --------------------------------------------------------------------------
def selftest(P):
    print("chain (ordered):", P['chain'])
    print("num2str:", P['num2str'])
    print("silent middle (VLM-resolved):", P['silent_ids'])
    print("D1-anchored steps:", P['d1_started'])
    assert P['first_id'] == P['chain'][0]
    assert len(P['silent_ids']) >= 1 and len(P['d1_started']) == 2
    orders = [P['str_step'][s]['order'] for s in P['silent_ids']]
    assert orders == sorted(orders), "silent steps must be contiguous/ordered"
    print(f"silent orders {orders} OK; selftest passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default=PLAN)
    ap.add_argument('--recipe', default=RECIPE)
    ap.add_argument('--recs', default=','.join(RECS))
    ap.add_argument('--vlm', choices=['off', 'qwen'], default='qwen')
    ap.add_argument('--video-dir', default=os.path.join(BASE, 'data/videos_360p'))
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--interval', type=float, default=10.0)
    ap.add_argument('--arm', default='sensorplan')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    P = load_plan(a.plan, a.recipe)
    if a.selftest:
        selftest(P)
        return
    for rec in a.recs.split(','):
        run_recording(P, rec, a.vlm, a.video_dir, a.interval, a.out_dir, a.arm, a.trace)
    print('\nwrote results to', os.path.join(a.out_dir, a.arm))


if __name__ == '__main__':
    main()
