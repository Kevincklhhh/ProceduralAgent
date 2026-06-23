#!/usr/bin/env python3
"""eval_score_activity8.py (was score.py) -- Referee scorer for the 3-arm replay
experiment (activity 8, mug hot chocolate). Activity-8 pilot; corpus version is eval_score_corpus.py.

Arms:    detector_replay | periodic_vlm_qwen | detector_plus_escalation
GT:      detectors/gt_activity8.json
Outputs: replay/results/scores.json, replay/REPORT.md
"""
import json
import math
import os
from collections import defaultdict

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
GT_PATH = os.path.join(BASE, 'data', 'gt_activity8.json')
RESULTS_DIR = os.path.join(BASE, 'experiments', 'replay_v1', 'results')
ARMS = ['detector_replay', 'periodic_vlm_qwen', 'detector_plus_escalation']
RECS = ['8_16', '8_3', '8_25', '8_26', '8_31', '8_50']

STEP2FINE = {88: 'fill_milk', 89: 'microwave_initial', 90: 'add_chocolate',
             84: 'add_cinnamon', 87: 'add_sugar', 85: 'mix', 83: 'heat_serve'}
FINE2COARSE = {'add_chocolate': 'adds', 'add_cinnamon': 'adds', 'add_sugar': 'adds'}


def coarse(lbl):
    return FINE2COARSE.get(lbl, lbl)


SCORED_IDS = ['overtime_microwave', 'undertime_microwave', 'overtime_microwave_2',
              'undertime_microwave_2', 'missing_mix_before_heat',
              'missing_ingredient_before_mix']

# Truth table (recording -> ids that SHOULD fire).
# NOTE: 8_26 step 83 carries a GT Timing Error (8 s high + ~1 min low power); the
# 8 s run is below the 20 s hum floor, so it is EXCLUDED from the truth table by
# protocol (discussed in REPORT.md as a known limitation).
TRUTH = {
    '8_26': {'overtime_microwave', 'missing_ingredient_before_mix'},
    '8_31': {'undertime_microwave', 'undertime_microwave_2',
             'missing_ingredient_before_mix'},
    '8_50': {'missing_mix_before_heat', 'missing_ingredient_before_mix'},
    '8_16': set(), '8_3': set(), '8_25': set(),
}
TRULY_MISSING = {'8_26': {'cinnamon'}, '8_31': {'sugar', 'chocolate'},
                 '8_50': {'cinnamon'},
                 '8_16': set(), '8_3': set(), '8_25': set()}

WINDOW_TOL = 15.0  # seconds


def load_gt():
    with open(GT_PATH) as f:
        return json.load(f)


def load_arm(arm, rec):
    with open(os.path.join(RESULTS_DIR, arm, rec + '.json')) as f:
        return json.load(f)


def gt_segments(gt_rec):
    """Return list of (start, end, fine_stage) for non-skipped steps."""
    segs = []
    for s in gt_rec['steps']:
        if s['start_time'] >= 0 and s['end_time'] >= 0:
            segs.append((float(s['start_time']), float(s['end_time']),
                         STEP2FINE[s['step_id']]))
    return segs


def gt_step_window(gt_rec, step_id):
    for s in gt_rec['steps']:
        if s['step_id'] == step_id and s['start_time'] >= 0:
            return (float(s['start_time']), float(s['end_time']))
    return None


def gt_label(segs, t):
    """GT label at time t. Where GT step segments overlap (they do, e.g. 8_25
    cinnamon prep during the microwave run), pick the most recently STARTED
    step ('current step' semantics)."""
    hits = [s for s in segs if s[0] <= t < s[1]]
    if not hits:
        return 'other'
    return max(hits, key=lambda s: s[0])[2]


def pred_label(intervals, t):
    for iv in intervals:
        if iv['start_s'] <= t < iv['end_s']:
            return iv['stage']
    return 'other'


def horizon(gt_rec):
    return int(math.ceil(max(s['end_time'] for s in gt_rec['steps']
                             if s['end_time'] >= 0)))


def stage_accuracy(gt_rec, arm_res, fine=False):
    segs = gt_segments(gt_rec)
    T = horizon(gt_rec)
    n = n_correct = 0
    n_excl = n_correct_excl = 0
    for i in range(T):
        t = i + 0.5
        g = gt_label(segs, t)
        p = pred_label(arm_res['stage_intervals'], t)
        if not fine:
            g, p = coarse(g), coarse(p)
        n += 1
        ok = (g == p)
        n_correct += ok
        if g != 'other':
            n_excl += 1
            n_correct_excl += ok
    return {'n_seconds': n,
            'accuracy': n_correct / n,
            'accuracy_excl_gt_other': (n_correct_excl / n_excl) if n_excl else None}


def boundary_deltas(gt_rec, arm_res):
    """Predicted microwave_initial / heat_serve interval start/end minus GT
    step 89/83 start/end. Multiple predicted intervals of a stage: first start,
    last end. Descriptive only."""
    out = {}
    for stage, step_id in (('microwave_initial', 89), ('heat_serve', 83)):
        w = gt_step_window(gt_rec, step_id)
        ivs = [iv for iv in arm_res['stage_intervals'] if iv['stage'] == stage]
        if w is None or not ivs:
            out[stage] = None
            continue
        out[stage] = {'start_delta_s': round(ivs[0]['start_s'] - w[0], 1),
                      'end_delta_s': round(ivs[-1]['end_s'] - w[1], 1)}
    return out


def relevant_window(gt_rec, rid):
    """GT window an event of this id should fall in (for the +/-15 s check)."""
    if rid in ('overtime_microwave', 'undertime_microwave'):
        return gt_step_window(gt_rec, 89)
    if rid in ('overtime_microwave_2', 'undertime_microwave_2'):
        return gt_step_window(gt_rec, 83)
    if rid == 'missing_mix_before_heat':
        return gt_step_window(gt_rec, 83)
    if rid == 'missing_ingredient_before_mix':
        w = gt_step_window(gt_rec, 85)
        return w if w is not None else gt_step_window(gt_rec, 83)
    return None


def ingredient_of(msg):
    m = msg.lower()
    for ing in ('chocolate', 'cinnamon', 'sugar'):
        if ing in m.split('?')[0]:
            return ing
    return None


def score_reminders(gt, arm, arm_results):
    scored = list(SCORED_IDS)
    na_ids = []
    if arm == 'detector_replay':
        # missing_ingredient is N/A by design for the graph-only arm.
        scored = [i for i in SCORED_IDS if i != 'missing_ingredient_before_mix']
        na_ids = ['missing_ingredient_before_mix']

    per_id = {i: {'TP': [], 'FP': [], 'FN': []} for i in scored}
    fires = []          # detailed fire log
    duplicates = []     # same id fired >1x on a recording
    for rec in RECS:
        res = arm_results[rec]
        ev_ids = defaultdict(list)
        for e in res['events']:
            if e['id'] in SCORED_IDS:
                ev_ids[e['id']].append(e)
        for rid, evs in ev_ids.items():
            if len(evs) > 1:
                duplicates.append({'recording': rec, 'id': rid, 'count': len(evs),
                                   'times': [round(e['t'], 1) for e in evs]})
            for e in evs:
                w = relevant_window(gt[rec], rid)
                inside = (w is not None and
                          w[0] - WINDOW_TOL <= e['t'] <= w[1] + WINDOW_TOL)
                fires.append({'recording': rec, 'id': rid, 't': round(e['t'], 1),
                              'gt_window': [round(w[0], 1), round(w[1], 1)] if w else None,
                              'inside_window_pm15s': inside if w else None,
                              'scored': rid in scored,
                              'verdict': None})
        for rid in scored:
            should = rid in TRUTH[rec]
            did = rid in ev_ids
            if should and did:
                per_id[rid]['TP'].append(rec)
            elif did and not should:
                per_id[rid]['FP'].append(rec)
            elif should and not did:
                per_id[rid]['FN'].append(rec)
    # attach verdicts to fire log
    for f in fires:
        if not f['scored']:
            f['verdict'] = 'N/A (id not scored for this arm)'
        elif f['recording'] in per_id.get(f['id'], {}).get('TP', []):
            f['verdict'] = 'TP'
        else:
            f['verdict'] = 'FP'

    tp = sum(len(v['TP']) for v in per_id.values())
    fp = sum(len(v['FP']) for v in per_id.values())
    fn = sum(len(v['FN']) for v in per_id.values())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    # escalation request coverage (mainly relevant for detector_replay protocol,
    # reported for every arm)
    esc = {rec: len(arm_results[rec].get('escalation_requests', []))
           for rec in RECS}
    coverage = sum(1 for v in esc.values() if v > 0)

    # ingredient-level detail (any arm that emits missing_ingredient events)
    ing_detail = {}
    for rec in RECS:
        flagged = set()
        for e in arm_results[rec]['events']:
            if e['id'] == 'missing_ingredient_before_mix':
                ing = ingredient_of(e['message'])
                if ing:
                    flagged.add(ing)
        truly = TRULY_MISSING[rec]
        ing_detail[rec] = {
            'flagged': sorted(flagged), 'truly_missing': sorted(truly),
            'ing_TP': sorted(flagged & truly),
            'ing_FP': sorted(flagged - truly),
            'ing_FN': sorted(truly - flagged)}
    ing_tp = sum(len(v['ing_TP']) for v in ing_detail.values())
    ing_fp = sum(len(v['ing_FP']) for v in ing_detail.values())
    ing_fn = sum(len(v['ing_FN']) for v in ing_detail.values())

    # descriptive events
    hot_mug = []
    done_prompt = []
    other_events = []
    for rec in RECS:
        w83 = gt_step_window(gt[rec], 83)
        for e in arm_results[rec]['events']:
            if e['id'] == 'hot_mug_caution':
                hit = (w83 is not None and
                       w83[0] - WINDOW_TOL <= e['t'] <= w83[1] + WINDOW_TOL)
                hot_mug.append({'recording': rec, 't': round(e['t'], 1),
                                'near_gt_heat_step_pm15s': hit})
            elif e['id'] == 'microwave_done_prompt':
                done_prompt.append({'recording': rec, 't': round(e['t'], 1),
                                    'message': e['message']})
            elif e['id'] == 'other':
                other_events.append({'recording': rec, 't': round(e['t'], 1),
                                     'message': e['message']})
    n_completed_heats = sum(1 for rec in RECS if gt_step_window(gt[rec], 83))

    return {
        'scored_ids': scored,
        'na_by_design': na_ids,
        'per_id': {k: {kk: vv for kk, vv in v.items()} for k, v in per_id.items()},
        'totals': {'TP': tp, 'FP': fp, 'FN': fn,
                   'precision': precision, 'recall': recall},
        'fires': fires,
        'duplicate_fires': duplicates,
        'escalation_requests_per_rec': esc,
        'escalation_request_coverage': f'{coverage}/{len(RECS)}',
        'ingredient_detail': ing_detail,
        'ingredient_totals': {'TP': ing_tp, 'FP': ing_fp, 'FN': ing_fn,
                              'precision': ing_tp / (ing_tp + ing_fp) if (ing_tp + ing_fp) else None,
                              'recall': ing_tp / (ing_tp + ing_fn) if (ing_tp + ing_fn) else None},
        'descriptive': {
            'hot_mug_caution': {'fires': hot_mug,
                                'hits': sum(1 for h in hot_mug if h['near_gt_heat_step_pm15s']),
                                'completed_heats': n_completed_heats},
            'microwave_done_prompt': done_prompt,
            'other_id_events': other_events},
    }


def score_costs(gt, arm_results):
    tot = {'vlm_calls': 0, 'frames_sent': 0, 'vlm_latency_total_s': 0.0,
           'compute_s': 0.0}
    per_rec = {}
    horizon_s = 0
    for rec in RECS:
        c = arm_results[rec]['cost']
        per_rec[rec] = c
        for k in tot:
            tot[k] += c[k]
        horizon_s += horizon(gt[rec])
    minutes = horizon_s / 60.0
    return {
        'totals': {k: round(v, 2) for k, v in tot.items()},
        'per_recording': per_rec,
        'total_recording_s': horizon_s,
        'total_recording_min': round(minutes, 2),
        'per_recording_minute': {
            'vlm_calls': round(tot['vlm_calls'] / minutes, 3),
            'frames_sent': round(tot['frames_sent'] / minutes, 3),
            'vlm_latency_s': round(tot['vlm_latency_total_s'] / minutes, 2),
            'compute_s': round(tot['compute_s'] / minutes, 3)},
        'vlm_latency_x_realtime': round(tot['vlm_latency_total_s'] / horizon_s, 2),
    }


def main():
    gt = load_gt()
    all_results = {arm: {rec: load_arm(arm, rec) for rec in RECS} for arm in ARMS}

    scores = {'protocol': {
        'recordings': RECS,
        'horizon': 'per-second labels over [0, ceil(last GT step end)] per recording',
        'gt_overlap_rule': 'where GT step segments overlap, the most recently started step wins',
        'scored_ids': SCORED_IDS,
        'truth_table': {k: sorted(v) for k, v in TRUTH.items()},
        'truly_missing_ingredients': {k: sorted(v) for k, v in TRULY_MISSING.items()},
        'window_tolerance_s': WINDOW_TOL,
        'known_limitation_8_26_step83': (
            '8_26 step 83 carries a GT Timing Error from a split run (8 s high '
            '+ ~1 min low power); an 8 s run is below the 20 s hum floor, so '
            'duration-rule arms cannot catch it - excluded from the truth table.'),
    }}

    # ---- 1. stage accuracy ----
    stage = {}
    for arm in ARMS:
        per_rec = {}
        for rec in RECS:
            res = all_results[arm][rec]
            entry = {'coarse': stage_accuracy(gt[rec], res, fine=False)}
            if arm == 'periodic_vlm_qwen':
                entry['fine_7way'] = stage_accuracy(gt[rec], res, fine=True)
            entry['boundary_deltas'] = boundary_deltas(gt[rec], res)
            per_rec[rec] = entry
        mean_coarse = sum(per_rec[r]['coarse']['accuracy'] for r in RECS) / len(RECS)
        mean_excl = sum(per_rec[r]['coarse']['accuracy_excl_gt_other'] for r in RECS) / len(RECS)
        entry = {'per_recording': per_rec,
                 'mean_coarse_accuracy': mean_coarse,
                 'mean_coarse_accuracy_excl_gt_other': mean_excl}
        if arm == 'periodic_vlm_qwen':
            entry['mean_fine_7way_accuracy'] = sum(
                per_rec[r]['fine_7way']['accuracy'] for r in RECS) / len(RECS)
        if arm != 'periodic_vlm_qwen':
            entry['note'] = ('audio/graph detector cannot distinguish the three '
                             'add_* sub-steps by design; scored at coarse level only')
        stage[arm] = entry
    scores['stage_accuracy'] = stage

    # ---- 2. reminders ----
    scores['reminders'] = {arm: score_reminders(gt, arm, all_results[arm])
                           for arm in ARMS}

    # ---- 3. costs ----
    scores['costs'] = {arm: score_costs(gt, all_results[arm]) for arm in ARMS}

    out_json = os.path.join(RESULTS_DIR, 'scores.json')
    with open(out_json, 'w') as f:
        json.dump(scores, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else o)
    print('wrote', out_json)

    write_report(gt, scores, all_results)


# ------------------------------------------------------------------ report
def pct(x):
    return f'{100*x:.1f}%' if x is not None else 'n/a'


def write_report(gt, S, all_results):
    L = []
    A = L.append
    A('# Replay Experiment Report - Activity 8 (mug hot chocolate)')
    A('')
    A('Scored by `eval/eval_score_activity8.py` against `detectors/gt_activity8.json`. '
      'All numbers in `replay/results/scores.json`.')
    A('')
    A('## Setup')
    A('')
    A('Three arms replayed over the same six recordings (8_16, 8_3, 8_25 clean; '
      '8_26, 8_31 error runs; 8_50 an order-error run: sugar added before milk, '
      'mix and cinnamon skipped - deliberately NOT special-cased in any engine):')
    A('')
    A('1. **detector_replay** - frozen audio detectors (microwave hum/beep, stir, pour) '
      'driving a procedure graph; zero VLM calls. Ingredient identity is unknowable to it; '
      'it emits an `escalation_request` instead.')
    A('2. **periodic_vlm_qwen** - Qwen3.6-27B on a local vLLM server, called every 10 s '
      'with 3 frames (480p).')
    A('3. **detector_plus_escalation** - arm 1 plus one targeted VLM call per recording at '
      'the graph\'s mix boundary to verify ingredients.')
    A('')
    A('Detectors were frozen before replay; numeric thresholds tuned on 8_16 only '
      '(see Limitations for the structural-choice disclosure).')
    A('')
    A('Reminder truth table (scored ids only):')
    A('')
    A('| recording | should fire |')
    A('|---|---|')
    for rec in RECS:
        A(f'| {rec} | {", ".join(sorted(TRUTH[rec])) or "-"} |')
    A('')
    A('Scored ids: ' + ', '.join(SCORED_IDS) + '. `missing_ingredient_before_mix` is '
      'binary per recording; ingredient-level detail reported separately. For '
      '**detector_replay** that id is N/A-by-design: we score escalation-request '
      'coverage instead (it should request on ALL recordings).')
    A('')

    # ---- stage accuracy ----
    A('## 1. Stage accuracy (per-second, coarse 5-stage + other)')
    A('')
    A('Per-second labels over `[0, ceil(last GT step end)]`; GT coarse label from step '
      'segments (88=fill_milk, 89=microwave_initial, 90/84/87=adds, 85=mix, '
      '83=heat_serve; unlabeled seconds=other; skipped steps absent). Where GT step '
      'segments overlap (e.g. 8_25: cinnamon prep during the microwave run), the most '
      'recently started step wins. Predicted gaps=other; fine `add_*` maps to `adds`.')
    A('')
    hdr = '| arm | ' + ' | '.join(RECS) + ' | mean |'
    A(hdr)
    A('|---|' + '---|' * (len(RECS) + 1))
    for arm in ARMS:
        st = S['stage_accuracy'][arm]
        row = [pct(st['per_recording'][r]['coarse']['accuracy']) for r in RECS]
        A(f'| {arm} | ' + ' | '.join(row) + f' | **{pct(st["mean_coarse_accuracy"])}** |')
    A('')
    A('Coarse accuracy excluding GT-`other` seconds:')
    A('')
    A(hdr)
    A('|---|' + '---|' * (len(RECS) + 1))
    for arm in ARMS:
        st = S['stage_accuracy'][arm]
        row = [pct(st['per_recording'][r]['coarse']['accuracy_excl_gt_other']) for r in RECS]
        A(f'| {arm} | ' + ' | '.join(row) + f' | **{pct(st["mean_coarse_accuracy_excl_gt_other"])}** |')
    A('')
    st = S['stage_accuracy']['periodic_vlm_qwen']
    row = [pct(st['per_recording'][r]['fine_7way']['accuracy']) for r in RECS]
    A('Fine 7-way accuracy (periodic arm only - the audio/graph detector cannot '
      'distinguish the three adds by design, so fine scoring would be vacuous for it):')
    A('')
    A('| arm | ' + ' | '.join(RECS) + ' | mean |')
    A('|---|' + '---|' * (len(RECS) + 1))
    A('| periodic_vlm_qwen (fine) | ' + ' | '.join(row) +
      f' | **{pct(st["mean_fine_7way_accuracy"])}** |')
    A('')
    A('### Microwave-anchor boundary deltas (descriptive)')
    A('')
    A('Predicted `microwave_initial` / `heat_serve` interval start/end minus GT step '
      'start/end, seconds (positive = predicted later). Blank = stage not predicted '
      'or GT step absent.')
    A('')
    A('| arm | rec | mw start | mw end | heat start | heat end |')
    A('|---|---|---|---|---|---|')
    for arm in ARMS:
        for rec in RECS:
            bd = S['stage_accuracy'][arm]['per_recording'][rec]['boundary_deltas']
            def fmt(stagek, key):
                d = bd.get(stagek)
                return f'{d[key]:+.1f}' if d else ''
            A(f'| {arm} | {rec} | {fmt("microwave_initial","start_delta_s")} | '
              f'{fmt("microwave_initial","end_delta_s")} | '
              f'{fmt("heat_serve","start_delta_s")} | {fmt("heat_serve","end_delta_s")} |')
    A('')
    A('The detector arms\' microwave boundaries track GT to within a few seconds when a '
      'hum run is detected (the hum starts after walking to the microwave, so small '
      'positive start deltas are expected - GT segments include walking). The periodic '
      'arm\'s boundaries are quantized to its 10 s call grid and drift much further.')
    A('')

    # ---- reminders ----
    A('## 2. Reminder decisions')
    A('')
    A('| arm | TP | FP | FN | precision | recall |')
    A('|---|---|---|---|---|---|')
    for arm in ARMS:
        t = S['reminders'][arm]['totals']
        A(f'| {arm} | {t["TP"]} | {t["FP"]} | {t["FN"]} | {pct(t["precision"])} | {pct(t["recall"])} |')
    A('')
    A('(detector_replay is scored over 5 ids - `missing_ingredient_before_mix` is '
      'N/A-by-design; its escalation-request coverage was '
      f'**{S["reminders"]["detector_replay"]["escalation_request_coverage"]}** recordings, '
      'as required. The two VLM-bearing arms are scored over all 6 ids.)')
    A('')
    A('Per-id outcome (recordings listed):')
    A('')
    A('| arm | id | TP | FP | FN |')
    A('|---|---|---|---|---|')
    for arm in ARMS:
        for rid in SCORED_IDS:
            pid = S['reminders'][arm]['per_id'].get(rid)
            if pid is None:
                A(f'| {arm} | {rid} | N/A by design | | |')
                continue
            A(f'| {arm} | {rid} | {", ".join(pid["TP"]) or "-"} | '
              f'{", ".join(pid["FP"]) or "-"} | {", ".join(pid["FN"]) or "-"} |')
    A('')
    A('Fire timestamps vs relevant GT step window (+/-15 s):')
    A('')
    A('| arm | rec | id | t (s) | GT window | inside +/-15s | verdict |')
    A('|---|---|---|---|---|---|---|')
    for arm in ARMS:
        for f in S['reminders'][arm]['fires']:
            w = f'{f["gt_window"][0]}-{f["gt_window"][1]}' if f['gt_window'] else '-'
            A(f'| {arm} | {f["recording"]} | {f["id"]} | {f["t"]} | {w} | '
              f'{f["inside_window_pm15s"]} | {f["verdict"]} |')
    for arm in ARMS:
        for d in S['reminders'][arm]['duplicate_fires']:
            A('')
            A(f'Note: {arm} fired `{d["id"]}` {d["count"]}x on {d["recording"]} '
              f'(t={d["times"]}); deduplicated to one decision for scoring.')
    A('')
    A('### Ingredient-level detail (missing_ingredient_before_mix)')
    A('')
    A('| arm | rec | flagged | truly missing | ing TP | ing FP | ing FN |')
    A('|---|---|---|---|---|---|---|')
    for arm in ('detector_plus_escalation', 'periodic_vlm_qwen'):
        for rec in RECS:
            d = S['reminders'][arm]['ingredient_detail'][rec]
            if not d['flagged'] and not d['truly_missing']:
                continue
            A(f'| {arm} | {rec} | {", ".join(d["flagged"]) or "-"} | '
              f'{", ".join(d["truly_missing"]) or "-"} | {", ".join(d["ing_TP"]) or "-"} | '
              f'{", ".join(d["ing_FP"]) or "-"} | {", ".join(d["ing_FN"]) or "-"} |')
    it = S['reminders']['detector_plus_escalation']['ingredient_totals']
    A('')
    A(f'detector_plus_escalation ingredient-level totals: TP={it["TP"]}, FP={it["FP"]}, '
      f'FN={it["FN"]} (precision {pct(it["precision"])}, recall {pct(it["recall"])}). '
      'The escalation VLM never misses a truly-missing ingredient but over-reports '
      '"missing" (notably chocolate, falsely flagged on 5/6 recordings): it asserts '
      'absence too readily from 10 frames sampled around the mix boundary.')
    it2 = S['reminders']['periodic_vlm_qwen']['ingredient_totals']
    A(f'periodic_vlm_qwen ingredient-level totals: TP={it2["TP"]}, FP={it2["FP"]}, '
      f'FN={it2["FN"]} - it never emitted a missing-ingredient event at all.')
    A('')
    A('### Descriptive events (not in P/R)')
    A('')
    for arm in ARMS:
        d = S['reminders'][arm]['descriptive']
        hm = d['hot_mug_caution']
        fires = ', '.join(f'{x["recording"]}@{x["t"]}s'
                          f'({"hit" if x["near_gt_heat_step_pm15s"] else "miss"})'
                          for x in hm['fires']) or 'none'
        A(f'- **{arm}** hot_mug_caution: {hm["hits"]}/{hm["completed_heats"]} completed '
          f'heats covered ({fires}).')
        if d['microwave_done_prompt']:
            A(f'  - microwave_done_prompt fires: ' +
              ', '.join(f'{x["recording"]}@{x["t"]}s' for x in d['microwave_done_prompt']))
        else:
            A('  - microwave_done_prompt: no fires.')
        if d['other_id_events']:
            A('  - `other` events (over-talk): ' +
              '; '.join(f'{x["recording"]}@{x["t"]}s "{x["message"][:60]}"'
                        for x in d['other_id_events']))
        elif arm == 'periodic_vlm_qwen':
            A('  - `other` events from the VLM: none (no over-talk; if anything the '
              'periodic arm under-talks).')
    A('')
    A('**Known limitation (excluded from the truth table):** 8_26 step 83 carries a GT '
      'Timing Error from a split run - 8 s on high power plus ~1 min on low power. An '
      '8 s run is below the 20 s hum floor of the audio probe, so duration-rule arms '
      'cannot catch it; no `*_microwave_2` id was expected on 8_26. The periodic VLM '
      'arm also said nothing about it.')
    A('')

    # ---- costs ----
    A('## 3. Cost ledger')
    A('')
    tot_min = S['costs']['detector_replay']['total_recording_min']
    A(f'Totals across six recordings ({S["costs"]["detector_replay"]["total_recording_s"]} s '
      f'= {tot_min} min of GT-scored video); per-minute normalization in parentheses.')
    A('')
    A('| arm | vlm_calls | frames_sent | vlm_latency_total_s | detector compute_s |')
    A('|---|---|---|---|---|')
    for arm in ARMS:
        c = S['costs'][arm]
        t, m = c['totals'], c['per_recording_minute']
        A(f'| {arm} | {t["vlm_calls"]} ({m["vlm_calls"]}/min) | '
          f'{t["frames_sent"]} ({m["frames_sent"]}/min) | '
          f'{t["vlm_latency_total_s"]} ({m["vlm_latency_s"]} s/min) | '
          f'{t["compute_s"]} ({m["compute_s"]} s/min) |')
    A('')
    cp = S['costs']['periodic_vlm_qwen']
    A(f'**The periodic baseline cannot keep up with real time on this hardware.** It is '
      f'called every 10 s but a call takes ~44-54 s (measured mean across recordings '
      f'~50 s; sequential smoke test 43.6 s/call). Total VLM latency '
      f'{cp["totals"]["vlm_latency_total_s"]} s to cover {cp["total_recording_s"]} s of '
      f'video = **{cp["vlm_latency_x_realtime"]}x real time** - i.e. run sequentially it '
      f'falls ~5x behind; the replay only finished by running 2-way concurrency offline. '
      f'A live assistant on this hardware would answer about stage N-25 while the user is '
      f'on stage N.')
    ce = S['costs']['detector_plus_escalation']
    A(f'The escalation arm used exactly 1 call/recording (6 calls, '
      f'{ce["totals"]["frames_sent"]} frames, {ce["totals"]["vlm_latency_total_s"]} s '
      f'latency total = {round(100*ce["totals"]["vlm_latency_total_s"]/cp["totals"]["vlm_latency_total_s"],1)}% '
      f'of the periodic arm\'s latency budget); the detector arm used 0.')
    A('')

    # ---- walkthroughs ----
    A('## 4. Per-recording walkthroughs (error runs)')
    A('')
    A('### 8_26 (overtime first microwave; cinnamon skipped; GT also: whole milk + spill, '
      '4 chocolate pieces, split second heat)')
    A('')
    A('- **detector_replay**: caught `overtime_microwave` at 134.4 s, 80 s into a hum run '
      'GT says lasted ~2 min - correct and timely. Escalated for ingredients (cannot know '
      'them itself). Missed nothing it could physically see. The split second-heat '
      '(8 s high) is below the hum floor (excluded, see above).')
    A('- **periodic_vlm_qwen**: emitted NO events - missed the 2-minute overtime despite '
      'sampling the microwave 12+ times, and never questioned ingredients. Its stage '
      'track labeled most of 70-210 s as `other`.')
    A('- **detector_plus_escalation**: `overtime_microwave` TP (134.4 s) and '
      '`missing_ingredient_before_mix` TP at 280.5 s - but the VLM verdict flagged all '
      'three ingredients missing when only cinnamon was (chocolate and sugar were '
      'visibly added); recording-level TP, ingredient-level 1 TP + 2 FP.')
    A('')
    A('### 8_31 (35 s first microwave, 40 s second; sugar and chocolate skipped)')
    A('')
    A('- **detector_replay**: `undertime_microwave` TP at 100.7 s (measured 24.3 s hum vs '
      'GT 35 s actual - the hum probe undershoots short runs but the decision is right). '
      'MISSED `undertime_microwave_2`: the second run (40 s actual, GT heat step '
      '175.9-233.2 s) produced a hum the engine treated as the final heat and closed '
      'with `hot_mug_caution` at 235.9 s instead of checking duration - FN.')
    A('- **periodic_vlm_qwen**: NO events; both undertime runs and both missing '
      'ingredients missed. After 160 s it labeled everything `other`.')
    A('- **detector_plus_escalation**: `undertime_microwave` TP; '
      '`missing_ingredient_before_mix` TP at 183.9 s with a PERFECT ingredient verdict '
      '(chocolate=missing, sugar=missing, cinnamon=added) - the one recording where the '
      'escalation VLM was exactly right. Same `undertime_microwave_2` FN as arm 1 '
      '(shared engine).')
    A('')
    A('### 8_50 (order error: sugar before milk; mix and cinnamon skipped)')
    A('')
    A('- GT order: sugar 0.7-69.3, milk 69.3-92.3, microwave 96.1-159.8, chocolate '
      '163.3-185.9, heat 186.4-251.7. Mis-tracking here is expected data; no engine was '
      'special-cased.')
    A('- **detector_replay**: stage track is wrong by construction (labels 0-100 s '
      '`fill_milk` while the user added sugar first). It falsely detected `mix` at '
      '245.5 s (debug shows a strong clink train 245.5-267.5 s - spoon/mug contact '
      'during serving), which suppressed `missing_mix_before_heat` - FN. The second '
      'microwave run produced NO detectable hum (debug: mw2=null; only its 4 kHz '
      'done-beep at 251.0 s registered - this is the different-microwave recording from '
      'the disclosure), so no `hot_mug_caution` and no `heat_serve` interval. '
      'Escalation was still requested (coverage held).')
    A('- **periodic_vlm_qwen**: detected `add_sugar` at 60-80 s - the ONLY arm whose '
      'stage track reflects the order error at all (RGB sees the sugar jar; audio '
      'cannot). But it emitted no events: order error never called out, missing mix and '
      'cinnamon never questioned. Fine 7-way accuracy still only '
      f'{pct(S["stage_accuracy"]["periodic_vlm_qwen"]["per_recording"]["8_50"]["fine_7way"]["accuracy"])} '
      'because nearly everything else was labeled `other`.')
    A('- **detector_plus_escalation**: `missing_ingredient_before_mix` recording-level '
      'TP at 245.5 s (cinnamon correctly among the flagged), but the verdict flagged all '
      'three - sugar (added early) and chocolate (added) are ingredient-level FPs. '
      'Same `missing_mix_before_heat` FN as arm 1.')
    A('')

    # ---- takeaways ----
    A('## 5. Takeaways')
    A('')
    t_d = S['reminders']['detector_replay']['totals']
    t_p = S['reminders']['periodic_vlm_qwen']['totals']
    t_e = S['reminders']['detector_plus_escalation']['totals']
    sd = S['stage_accuracy']
    A(f'1. **Procedure structure + cheap audio beats the periodic VLM on this task, '
      f'decisively.** Reminders: detector+escalation {t_e["TP"]} TP / recall '
      f'{pct(t_e["recall"])} vs periodic VLM {t_p["TP"]} TP / recall {pct(t_p["recall"])} '
      f'(the periodic arm caught zero true reminders and its only fires were two false '
      f'`overtime_microwave` on clean 8_16). Stage tracking: mean coarse '
      f'{pct(sd["detector_replay"]["mean_coarse_accuracy"])} (detector arms) vs '
      f'{pct(sd["periodic_vlm_qwen"]["mean_coarse_accuracy"])} (periodic). Cost: 0-6 VLM '
      f'calls vs 229, and the periodic arm is {S["costs"]["periodic_vlm_qwen"]["vlm_latency_x_realtime"]}x '
      f'slower than real time, so its already-poor numbers are an OFFLINE upper bound on '
      f'its live usefulness.')
    A(f'2. **The single targeted escalation call is where the value-per-call is.** Six '
      f'calls bought ingredient awareness the graph cannot have (recall '
      f'{pct(t_e["recall"])} overall, 100% recording-level recall on missing-ingredient '
      f'runs and 100% ingredient-level recall), at ~2% of the periodic arm\'s latency. '
      f'But precision is poor ({pct(t_e["totals"]["precision"]) if isinstance(t_e, dict) and "totals" in t_e else pct(t_e["precision"])}): '
      f'the escalation VLM asserts "missing" too readily from 10 frames - every clean '
      f'recording got at least one spurious "did you add X?" prompt. Better escalation '
      f'prompting/frame selection (frames AT each add window, not around the mix '
      f'boundary) is the obvious next lever.')
    A(f'3. **Where it still needs RGB:** (a) ingredient identity - by design (the '
      f'detector arm escalated on 6/6 recordings because it cannot know); (b) order '
      f'errors - only the periodic VLM\'s stage track showed sugar-before-milk on 8_50; '
      f'audio is sequence-blind between anchors; (c) second-microwave duration semantics '
      f'(undertime_microwave_2 FN on 8_31) and any sub-hum-floor event; (d) quantity '
      f'errors (4 chocolate pieces, whole-vs-skimmed milk on 8_26) - invisible to every '
      f'arm tested.')
    A(f'4. The detector arms\' two reminder FPs both came from one clean recording '
      f'(8_25) where the hum probe fused a 24 s run and missed the stir - audio-probe '
      f'errors propagate directly into reminder errors; the graph amplifies neither '
      f'nor filters them.')
    A('')

    # ---- limitations ----
    A('## 6. Limitations')
    A('')
    A('- **One task type.** Six recordings of one microwave-centric recipe (activity 8); '
      'nothing here measures generalization to other procedures.')
    A('- **Microwave-centric anchors.** The graph advances mainly on microwave hum/beep '
      'anchors; tasks without such a strong audio anchor would lose most of the detector '
      'arms\' structure.')
    A('- **GT segment boundaries include walking** (fetching ingredients, moving to the '
      'microwave), so per-second stage accuracy and boundary deltas penalize/credit '
      'transition seconds somewhat arbitrarily; GT step segments even overlap in places '
      '(8_25, 8_16) - we resolved overlaps with a most-recently-started-step rule.')
    A('- **8_26 step 83** Timing Error (8 s high + ~1 min low) is below the 20 s hum '
      'floor and was excluded from the truth table (see Section 2).')
    A('- **Design-leakage disclosure (inherited from the hum probe, restated from '
      '`detectors/probes/results_hum_beep.json` tuning_on_8_16.note):** thresholds were '
      'set midway between hum and background medians on 8_16, where the grid shows a '
      'wide plateau (most combos give exactly 2 runs of ~60 s inside GT, 0 false). '
      'However, three STRUCTURAL choices were made after inspecting eval recordings '
      '(numeric thresholds still from 8_16 only): (1) the beep band was widened to '
      '800-5000 Hz because 8_50 has a 4 kHz beeper (a different microwave); (2) features '
      'are median-smoothed over 5.4 s because wearers make noise next to a running '
      'microwave; (3) a broadband gate F1 > 2 dB was added because 8_31 contains a '
      'fridge-like pure-120 Hz source. The detector arms\' results are therefore not '
      'fully blind to the eval set at the structural level, and clean-set FP rates '
      '(8_25) suggest the probe is still fragile.')
    A('')

    out_md = os.path.join(BASE, 'experiments', 'replay_v1', 'REPORT.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(L))
    print('wrote', out_md)


if __name__ == '__main__':
    main()
