"""engine.py -- Arm A: DETECTOR-ONLY replay of the task-graph state machine.

Event-driven state machine over cheap frozen audio detectors (no VLM, no RGB)
for CaptainCook4D activity 8 (Spiced Hot Chocolate), six recordings.

Nominal task graph (anchors authoritative, graph inference bridges
undetectable steps):
    fill_milk -> microwave_initial -> adds -> mix -> heat_serve

Transitions:
  - start in fill_milk at t=0
  - fill_milk -> microwave_initial : hum-run-1 onset (microwave running
    implies fill done -- graph inference)
  - microwave_initial -> adds      : microwave-1 done = hum-run-1 offset
    fused with beeps (last beep in [offset-5, offset+20] if any; probe showed
    beeps mark true offsets when the hum truncates)
  - adds -> mix                    : first STRONG clink-train onset after
    microwave-1 done
  - mix (or adds) -> heat_serve    : hum-run-2 onset (whether or not mix seen)
  - heat_serve runs to end of recording.

Hum-run assignment: first kept run (>=20 s) = microwave 1; next kept run
starting after microwave-1 done = microwave 2; extra runs are logged only.

Causality: detectors carry their built-in smoothing lookahead (hum mask up to
~7.2 s worst case from cascaded centered median filters, dominant window
5.38 s; clink-train confirmation 5 s; beep grouping 1.5 s).  The done-fusion
additionally finalizes up to 20 s after the hum offset (it must wait for the
beep window to close).  No decision uses audio beyond t + these lookaheads,
and no recording is special-cased.

Semantics choices (fixed a priori, not tuned):
  - "beep cluster at the offset" = any beep(s) with center time in
    [offset-5, offset+20]; done time = last such beep.
  - missing_mix_before_heat checks whether any strong clink train OVERLAPS
    (mw1_done, hum2_onset); the mix TRANSITION itself is onset-based per the
    task graph, so a strong train that started before microwave-1 done can
    suppress the reminder without creating a mix interval (logged).
  - "user activity" for microwave_done_prompt = any clink train or pour
    detection overlapping (done, done+30], or a hum onset in that window.
"""
import json
import os
import time

import numpy as np

import sys
sys.path.insert(0, '/home/kailaic/NeuroTrace/ProceduralAgent/detectors')
import detectors_lib as dl

OUT_DIR = '/home/kailaic/NeuroTrace/ProceduralAgent/experiments/replay_v1/results/detector_replay'
RECS = ['8_16', '8_3', '8_25', '8_26', '8_31', '8_50']
ARM = 'detector_replay'

DONE_FUSE_PRE_S = 5.0     # beep window starts offset-5
DONE_FUSE_POST_S = 20.0   # ... and ends offset+20
OVERTIME_S = 80.0         # warning if hum still on at onset+80
UNDERTIME_S = 40.0        # warning if fused duration < 40
PROMPT_DELAY_S = 30.0     # microwave_done_prompt after 30 s of no activity

ESCALATION_REASON = ('verify which ingredients were added to the mug '
                     'before mixing/heating')


def fuse_done(run, beeps):
    """Fused microwave-done time: last beep in [offset-5, offset+20], else offset."""
    onset, offset = run
    cand = [b['t'] for b in beeps
            if offset - DONE_FUSE_PRE_S <= b['t'] <= offset + DONE_FUSE_POST_S]
    if cand:
        return float(max(cand)), True
    return float(offset), False


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and a1 > b0


def run_recording(rec, hb_params, pc_params):
    # ---------------- detector pass (timed -> cost.compute_s) ----------------
    t0 = time.perf_counter()
    fs16, x16 = dl.load_audio_16k(rec)
    audio_dur = len(x16) / fs16
    hum_runs = dl.detect_hum_runs(x16, fs16, hb_params)
    beeps = dl.detect_beeps(x16, fs16, hb_params)
    fs48, x48 = dl.load_audio_48k(rec)
    trains = dl.detect_clink_trains(x48, fs48, pc_params)
    pours = dl.detect_pours(x48, fs48, pc_params)   # secondary, logged only
    compute_s = time.perf_counter() - t0

    strong = [tr for tr in trains if tr['strong']]
    events = []
    escalations = []
    notes = []

    def emit(t, etype, eid, msg):
        events.append(dict(t=round(float(t), 2), type=etype, id=eid, message=msg))

    # ---------------- hum-run assignment ----------------
    mw1 = hum_runs[0] if hum_runs else None
    mw1_done, mw1_fused, mw2, mw2_done, mw2_fused = None, False, None, None, False
    extra_runs = []
    if mw1 is not None:
        mw1_done, mw1_fused = fuse_done(mw1, beeps)
        rest = [r for r in hum_runs[1:] if r[0] > mw1_done]
        extra_runs = [r for r in hum_runs[1:] if r[0] <= mw1_done]
        if rest:
            mw2 = rest[0]
            mw2_done, mw2_fused = fuse_done(mw2, beeps)
            extra_runs += rest[1:]
    if extra_runs:
        notes.append('extra hum runs (unassigned): '
                     + str([[round(a, 1), round(b, 1)] for a, b in extra_runs]))

    # ---------------- state machine / stage intervals ----------------
    bounds = []   # (t, stage entered at t)
    mix_onset = None
    if mw1 is not None:
        bounds.append((mw1[0], 'microwave_initial'))
        bounds.append((mw1_done, 'adds'))
        cand = [tr['start'] for tr in strong if tr['start'] > mw1_done]
        if mw2 is not None:
            cand = [c for c in cand if c < mw2[0]]
        if cand:
            mix_onset = float(cand[0])
            bounds.append((mix_onset, 'mix'))
        if mw2 is not None:
            bounds.append((mw2[0], 'heat_serve'))

    stage_intervals = []
    cur_stage, cur_t = 'fill_milk', 0.0
    for t, stage in bounds:
        t = float(t)
        if t > cur_t:
            stage_intervals.append(dict(stage=cur_stage,
                                        start_s=round(cur_t, 2),
                                        end_s=round(t, 2)))
        cur_stage, cur_t = stage, max(cur_t, t)
    if audio_dur > cur_t:
        stage_intervals.append(dict(stage=cur_stage,
                                    start_s=round(cur_t, 2),
                                    end_s=round(float(audio_dur), 2)))

    # ---------------- events ----------------
    if mw1 is not None:
        on1, off1 = mw1
        if off1 - on1 >= OVERTIME_S:
            emit(on1 + OVERTIME_S, 'warning', 'overtime_microwave',
                 f'Microwave run 1 still running {OVERTIME_S:.0f}s after it '
                 f'started (t={on1:.1f}s); typical recipe time is 60s.')
        dur1 = mw1_done - on1
        if dur1 < UNDERTIME_S:
            emit(mw1_done, 'warning', 'undertime_microwave',
                 f'Microwave run 1 finished after only {dur1:.1f}s '
                 f'(fused done {mw1_done:.1f}s); recipe expects ~60s.')

        # microwave_done_prompt: no detected user activity within 30 s of done
        w0, w1 = mw1_done, mw1_done + PROMPT_DELAY_S
        activity = any(overlaps(tr['start'], tr['end'], w0, w1) for tr in trains)
        activity |= any(overlaps(p['start'], p['end'], w0, w1) for p in pours)
        activity |= any(w0 < r[0] <= w1 for r in hum_runs)
        if not activity:
            emit(w1, 'reminder', 'microwave_done_prompt',
                 f'Microwave finished at {mw1_done:.1f}s and no activity was '
                 f'detected for {PROMPT_DELAY_S:.0f}s; take the mug out and '
                 'continue with the additions.')

    if mw2 is not None:
        on2, off2 = mw2
        if off2 - on2 >= OVERTIME_S:
            emit(on2 + OVERTIME_S, 'warning', 'overtime_microwave_2',
                 f'Microwave run 2 still running {OVERTIME_S:.0f}s after it '
                 f'started (t={on2:.1f}s); typical recipe time is 60s.')
        dur2 = mw2_done - on2
        if dur2 < UNDERTIME_S:
            emit(mw2_done, 'warning', 'undertime_microwave_2',
                 f'Microwave run 2 finished after only {dur2:.1f}s '
                 f'(fused done {mw2_done:.1f}s); recipe expects ~60s.')

        # missing_mix_before_heat: no strong train overlapping (done, hum2 onset)
        mixed = any(overlaps(tr['start'], tr['end'], mw1_done, on2)
                    for tr in strong) if mw1 is not None else False
        if not mixed:
            emit(on2, 'reminder', 'missing_mix_before_heat',
                 f'Second microwave run started at {on2:.1f}s but no stirring '
                 'was detected since the first microwave finished; the recipe '
                 'mixes the mug before the final heat.')
        emit(mw2_done, 'warning', 'hot_mug_caution',
             f'Second microwave run finished at {mw2_done:.1f}s; '
             'the mug will be hot - handle with care.')

    # ---------------- escalation request (arm C executes these) ----------------
    esc_t = None
    if mix_onset is not None:
        esc_t = mix_onset
    elif mw2 is not None:
        esc_t = mw2[0]
    if esc_t is not None:
        escalations.append(dict(t=round(float(esc_t), 2), reason=ESCALATION_REASON))

    events.sort(key=lambda e: e['t'])

    result = dict(
        recording=rec,
        arm=ARM,
        stage_intervals=stage_intervals,
        events=events,
        escalation_requests=escalations,
        cost=dict(vlm_calls=0, frames_sent=0, vlm_latency_total_s=0.0,
                  compute_s=round(compute_s, 2),
                  notes='; '.join(notes)),
    )
    debug = dict(
        recording=rec,
        audio_dur_s=round(float(audio_dur), 2),
        hum_runs=[[round(a, 2), round(b, 2), round(b - a, 2)] for a, b in hum_runs],
        mw1=None if mw1 is None else dict(
            onset=round(mw1[0], 2), hum_off=round(mw1[1], 2),
            fused_done=round(mw1_done, 2), beep_fused=mw1_fused,
            fused_dur_s=round(mw1_done - mw1[0], 2)),
        mw2=None if mw2 is None else dict(
            onset=round(mw2[0], 2), hum_off=round(mw2[1], 2),
            fused_done=round(mw2_done, 2), beep_fused=mw2_fused,
            fused_dur_s=round(mw2_done - mw2[0], 2)),
        extra_hum_runs=[[round(a, 2), round(b, 2)] for a, b in extra_runs],
        beeps=beeps,
        clink_trains=trains,
        strong_trains=[tr for tr in trains if tr['strong']],
        mix_onset=None if mix_onset is None else round(mix_onset, 2),
        pours_secondary=pours,
        lookahead_note=('hum mask <=~7.2s (cascaded centered medians, dominant '
                        'window 5.38s), clink train 5s, beeps 1.5s; done-fusion '
                        'finalizes up to 20s after hum offset'),
    )
    return result, debug


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hb_params, pc_params = dl.load_frozen_params()
    summary = []
    for rec in RECS:
        result, debug = run_recording(rec, hb_params, pc_params)
        with open(f'{OUT_DIR}/{rec}.json', 'w') as fh:
            json.dump(result, fh, indent=1)
        with open(f'{OUT_DIR}/{rec}.debug.json', 'w') as fh:
            json.dump(debug, fh, indent=1)
        summary.append((rec, result, debug))
        ev = ', '.join(f"{e['id']}@{e['t']}" for e in result['events']) or '-'
        esc = ', '.join(str(e['t']) for e in result['escalation_requests']) or '-'
        stages = ' | '.join(f"{s['stage']} {s['start_s']}-{s['end_s']}"
                            for s in result['stage_intervals'])
        print(f"[{rec}] hum={debug['hum_runs']}  mix={debug['mix_onset']}  "
              f"compute={result['cost']['compute_s']}s")
        print(f"   stages: {stages}")
        print(f"   events: {ev}")
        print(f"   escalation: {esc}")
    print('\nwrote results to', OUT_DIR)
    return summary


if __name__ == '__main__':
    main()
