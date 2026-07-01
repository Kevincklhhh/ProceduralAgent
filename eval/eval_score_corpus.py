#!/usr/bin/env python3
"""BOX 2 — corpus referee. Scores predicted reminders against the Box-1 truth table
(`data/cc4d_proactive/`) over the full CC4D Family A corpus. See docs/REMINDER_EVALUATION.md.

This is the generalization of the activity-8 pilot `eval/eval_score_activity8.py` to all 384 recordings.
The pilot stays as-is (it is welded to the replay-arm files + REPORT in experiments/replay_v1).

What it scores, per arm:
  FA-1  per-class windowed P/R/F1 (membership t in [s,e]; also +/-15 s and +/-30 s radius).
        Silence on clean recordings is scored (a fire not owed = FP). Order additionally
        broken down by `dag_edge_violation` -> the cheap DAG-state detector's recoverable
        share (the ~52% upper bound for the A-solve order trigger).
  FA-2  G-Mean F1 = sqrt(Interrupt-F1 x Silent-F1) over the GT decision_points.
  T1    Active-Step Accuracy (overlap-tolerant current-step recognition): per-second,
        pred is correct if it names ANY step active at t (the GT set G_t), or 'other' when
        G_t is empty. GT segments from CC4D executed steps; coarse == step-id.
  Cost  vlm_calls / frames / vlm_latency / detector compute, normalized per video-minute.

Prediction (arm) input — one JSON per recording under <results-dir>/<arm>/<rid>.json:
  {"recording","arm",
   "stage_intervals":[{"stage","start_s","end_s"}],
   "events":[{"t", "class", "subtype", "id"?, "message"?}],   # class+subtype = scored key
   "escalation_requests":[...], "cost":{"vlm_calls","frames_sent","vlm_latency_total_s","compute_s"}}

With no --results-dir, two GT-DERIVED reference arms run (clearly non-deployable):
  oracle  - emits every GT event at window start + GT stage segments  => sanity P=R=F1=1.0
  silent  - emits nothing                                             => R=0, silence correct

Usage:  python eval/eval_score_corpus.py [--results-dir DIR --arms a,b] [--only 8_45] [--no-write]
Outputs: data/cc4d_proactive/_scores_corpus.json  (+ stdout summary)
"""
import json, os, math, glob, argparse
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
GT_DIR = os.path.join(BASE, 'data', 'cc4d_proactive')           # execution mistakes
OM_DIR = os.path.join(BASE, 'data', 'cc4d_proactive_om')        # order + missing_step
ANN = os.path.join(BASE, 'data', 'cc4d', 'annotations', 'annotation_json')
GRACE = 15.0                                   # FA-2 interrupt tolerance; default radius unit

SCORED_CLASSES = [                             # (cls, subtype) -> "cls/subtype" key
    'precondition_violation/order', 'precondition_violation/missing_step',
    'execution_error/technique', 'execution_error/preparation',
    'execution_error/measurement', 'execution_error/temperature',
    'parameter_violation/timing']


def ckey(e):
    return f"{e['cls']}/{e.get('subtype')}" if 'cls' in e else f"{e['class']}/{e.get('subtype')}"


# ---------------------------------------------------------------- loading
# GT reminders are POINT timestamps (`t`, no window) in two dirs: execution mistakes
# (cc4d_proactive) and order/missing (cc4d_proactive_om). We give each a zero-width
# window [t, t] so the tolerance-based matcher (match_class: s-tol <= pred <= e+tol) works
# unchanged -> exact-point at tol=0, +/-tol otherwise. cls is synthesized per subtype.
def _exec_events(reminders):
    out = []
    for r in reminders:
        t = r['t']
        out.append({'t': t, 'cls': 'parameter_violation' if r.get('subtype') == 'timing'
                    else 'execution_error', 'subtype': r.get('subtype'),
                    'window': [t, t], 'anchor': r.get('anchor_step'),
                    'id': r.get('id'), 'source': r.get('source')})
    return out


def _om_events(reminders):
    out = []
    for r in reminders:                              # all om order reminders are DAG violations
        t = r['t']; sub = r.get('subtype')
        out.append({'t': t, 'cls': 'precondition_violation', 'subtype': sub,
                    'window': [t, t], 'dag_edge_violation': True if sub == 'order' else None,
                    'anchor': r.get('anchor_step'),
                    'id': r.get('id'), 'source': r.get('source')})
    return out


def split_rids(split):
    """Recording ids belonging to a qualcomm-timeline split ('test'/'validation'/'train')."""
    tl = json.load(open(os.path.join(BASE, 'data', 'qualcomm_interactive_cooking',
                                     'qualcomm_timeline.json')))
    return {v for v, d in tl.items() if d.get('split') == split}


def load_gt(only=None, rids=None):
    gt = {}
    for f in sorted(glob.glob(os.path.join(GT_DIR, '*.json'))):     # base: execution + all recordings
        if os.path.basename(f).startswith('_'):
            continue
        d = json.load(open(f))
        rid = d['recording_id']
        if only and rid != only:
            continue
        if rids is not None and rid not in rids:
            continue
        gt[rid] = {'recording_id': rid, 'duration_s': d.get('duration_s', 0),
                   'is_error': d.get('is_error', False), 'events': _exec_events(d.get('reminders', []))}
    for f in sorted(glob.glob(os.path.join(OM_DIR, '*.json'))):     # overlay: order + missing
        if os.path.basename(f).startswith('_'):
            continue
        d = json.load(open(f))
        rid = d['recording_id']
        if only and rid != only:
            continue
        if rids is not None and rid not in rids:
            continue
        if rid not in gt:
            gt[rid] = {'recording_id': rid, 'duration_s': d.get('duration_s', 0),
                       'is_error': True, 'events': []}
        gt[rid]['events'].extend(_om_events(d.get('reminders', [])))
    return gt


def load_steps():
    """Per-recording executed steps (start/end) for stage-accuracy GT."""
    ann = json.load(open(os.path.join(ANN, 'complete_step_annotations.json')))
    return {v: ann[v]['steps'] for v in ann}


# ---------------------------------------------------------------- arms
def arm_oracle(gt_rec, steps):
    ev = [{'t': e['window'][0], 'class': e['cls'], 'subtype': e.get('subtype')}
          for e in gt_rec['events'] if e.get('window')]
    seg = [{'stage': s['step_id'], 'start_s': float(s['start_time']), 'end_s': float(s['end_time'])}
           for s in steps if s['start_time'] >= 0 and s['end_time'] >= 0]
    return {'stage_intervals': seg, 'events': ev, 'escalation_requests': [],
            'cost': {'vlm_calls': 0, 'frames_sent': 0, 'vlm_latency_total_s': 0.0, 'compute_s': 0.0}}


def arm_silent(gt_rec, steps):
    return {'stage_intervals': [], 'events': [], 'escalation_requests': [],
            'cost': {'vlm_calls': 0, 'frames_sent': 0, 'vlm_latency_total_s': 0.0, 'compute_s': 0.0}}


def load_arm_file(results_dir, arm, rid):
    p = os.path.join(results_dir, arm, rid + '.json')
    return json.load(open(p)) if os.path.exists(p) else \
        {'stage_intervals': [], 'events': [], 'escalation_requests': [],
         'cost': {'vlm_calls': 0, 'frames_sent': 0, 'vlm_latency_total_s': 0.0, 'compute_s': 0.0}}


# ---------------------------------------------------------------- FA-1
def match_class(gt_windows, pred_ts, tol):
    """Optimal one-to-one match (maximizes TPs). gt_windows: [(s, e, tag)]; pred_ts: [t].
    A pred matches a window if s-tol <= t <= e+tol. Classic interval-point matching: take
    windows by ascending right endpoint, assign each the smallest still-free pred inside it
    (order-independent, so a perfect oracle scores exactly 1.0 even with nested windows).
    Returns (tp_tags, n_fp, fn_tags)."""
    preds = sorted((t, j) for j, t in enumerate(pred_ts))
    used_pred = set()
    tp_tags, matched_gt = [], set()
    for gi in sorted(range(len(gt_windows)), key=lambda i: (gt_windows[i][1], gt_windows[i][0])):
        s, e, tag = gt_windows[gi]
        for t, pj in preds:                    # ascending t -> smallest free pred in window
            if pj in used_pred:
                continue
            if s - tol <= t <= e + tol:
                used_pred.add(pj); matched_gt.add(gi); tp_tags.append(tag); break
    n_fp = len(pred_ts) - len(used_pred)
    fn_tags = [w[2] for i, w in enumerate(gt_windows) if i not in matched_gt]
    return tp_tags, n_fp, fn_tags


def match_class_item(gt_windows, preds, tol):
    """Identification match (route 3): a pred is a TP only if it is the SAME menu item as a GT
    reminder -- same anchor step AND same subtype (subtype is unique per step, verified) AND
    within tolerance. gt_windows: [(s, e, anchor, tag)]; preds: [(t, step_id)].
    A pred with step_id is None (arm gave no location) can never item-match -> always FP.
    Returns (tp_tags, n_fp, fn_tags)."""
    plist = sorted((t, sid, j) for j, (t, sid) in enumerate(preds))
    used = set()
    tp_tags, matched = [], set()
    for gi in sorted(range(len(gt_windows)), key=lambda i: (gt_windows[i][1], gt_windows[i][0])):
        s, e, anchor, tag = gt_windows[gi]
        for t, sid, pj in plist:
            if pj in used:
                continue
            if sid is not None and sid == anchor and s - tol <= t <= e + tol:
                used.add(pj); matched.add(gi); tp_tags.append(tag); break
    n_fp = len(preds) - len(used)
    fn_tags = [w[3] for i, w in enumerate(gt_windows) if i not in matched]
    return tp_tags, n_fp, fn_tags


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (tp + fp + fn) else None)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r, 'f1': f}


def score_fa1_item(gt, arm_res, tol):
    """Route-3 identification scoring: TP requires same menu item (anchor step + subtype) AND
    timing. Strictly stronger than score_fa1 (which is timing + subtype, pooled over steps)."""
    per_class = {}
    for ck in SCORED_CLASSES:
        tp = fp = fn = 0
        for rid, g in gt.items():
            gw = [(e['window'][0], e['window'][1], e.get('anchor'), e.get('dag_edge_violation'))
                  for e in g['events'] if e.get('window') and ckey(e) == ck]
            preds = [(e['t'], e.get('step_id')) for e in arm_res[rid]['events'] if ckey(e) == ck]
            tpt, nfp, fnt = match_class_item(gw, preds, tol)
            tp += len(tpt); fp += nfp; fn += len(fnt)
        per_class[ck] = prf(tp, fp, fn)
    tot = prf(sum(c['TP'] for c in per_class.values()),
              sum(c['FP'] for c in per_class.values()),
              sum(c['FN'] for c in per_class.values()))
    return {'per_class': per_class, 'pooled': tot}


def score_fa1(gt, arm_res, tol):
    per_class = {}
    order_dag = {'TP_dag': 0, 'TP_nodag': 0, 'FN_dag': 0, 'FN_nodag': 0}
    for ck in SCORED_CLASSES:
        tp = fp = fn = 0
        for rid, g in gt.items():
            gw = [(e['window'][0], e['window'][1], e.get('dag_edge_violation'))
                  for e in g['events'] if e.get('window') and ckey(e) == ck]
            pt = [e['t'] for e in arm_res[rid]['events'] if ckey(e) == ck]
            tpt, nfp, fnt = match_class(gw, pt, tol)
            tp += len(tpt); fp += nfp; fn += len(fnt)
            if ck == 'precondition_violation/order':
                for tag in tpt:
                    order_dag['TP_dag' if tag else 'TP_nodag'] += 1
                for tag in fnt:
                    order_dag['FN_dag' if tag else 'FN_nodag'] += 1
        per_class[ck] = prf(tp, fp, fn)
    tot = prf(sum(c['TP'] for c in per_class.values()),
              sum(c['FP'] for c in per_class.values()),
              sum(c['FN'] for c in per_class.values()))
    dt, dn = order_dag['TP_dag'], order_dag['FN_dag']
    nt, nn = order_dag['TP_nodag'], order_dag['FN_nodag']
    order_dag['recall_dag_edge'] = dt / (dt + dn) if (dt + dn) else None
    order_dag['recall_non_dag_edge'] = nt / (nt + nn) if (nt + nn) else None
    return {'per_class': per_class, 'pooled': tot, 'order_dag_breakdown': order_dag}


# ---------------------------------------------------------------- FA-2
# RETIRED scheme (decision-point interrupt-vs-silent). The cc4d_proactive GT is pure
# event-detection (no decision_points), so this returns None unless legacy GT is present.
def score_fa2(gt, arm_res, tol):
    if not any('decision_points' in g for g in gt.values()):
        return None
    cm = Counter()                              # confusion over decision points
    for rid, g in gt.items():
        ptimes = [e['t'] for e in arm_res[rid]['events']]
        for dp in g.get('decision_points', []):
            pred_interrupt = any(abs(pt - dp['t']) <= tol for pt in ptimes)
            gt_interrupt = (dp['label'] == 'interrupt')
            cm[(gt_interrupt, pred_interrupt)] += 1
    tp_i = cm[(True, True)]; fp_i = cm[(False, True)]; fn_i = cm[(True, False)]
    tp_s = cm[(False, False)]; fp_s = cm[(True, False)]; fn_s = cm[(False, True)]
    int_f1 = prf(tp_i, fp_i, fn_i)['f1']
    sil_f1 = prf(tp_s, fp_s, fn_s)['f1']
    gmean = math.sqrt(int_f1 * sil_f1) if (int_f1 and sil_f1) else 0.0
    return {'interrupt_f1': int_f1, 'silent_f1': sil_f1, 'gmean_f1': gmean,
            'n_interrupt_points': tp_i + fn_i, 'n_silent_points': tp_s + fn_s}


# ---------------------------------------------------------------- stage acc
def stage_acc(gt, arm_res, steps_by_rec):
    n = nc = ne = nce = 0
    for rid, g in gt.items():
        steps = steps_by_rec.get(rid, [])
        segs = [(float(s['start_time']), float(s['end_time']), s['step_id'])
                for s in steps if s['start_time'] >= 0 and s['end_time'] >= 0]
        if not segs:
            continue
        T = int(math.ceil(max(e for _, e, _ in segs)))
        ivs = arm_res[rid]['stage_intervals']
        for i in range(T):
            t = i + 0.5
            G = {s[2] for s in segs if s[0] <= t < s[1]}                     # active-step SET (overlap-tolerant)
            phits = [iv for iv in ivs if iv['start_s'] <= t < iv['end_s']]   # most-recently-started wins
            pl = max(phits, key=lambda iv: iv['start_s'])['stage'] if phits else 'other'
            ok = (pl in G) if G else (pl == 'other')                          # Active-Step Accuracy
            n += 1; nc += ok
            if G:
                ne += 1; nce += ok
    return {'accuracy': nc / n if n else None,
            'accuracy_excl_other': nce / ne if ne else None, 'n_seconds': n}


# ---------------------------------------------------------------- cost
def score_cost(gt, arm_res):
    tot = Counter(); secs = 0.0
    for rid, g in gt.items():
        c = arm_res[rid].get('cost', {})
        for k in ('vlm_calls', 'frames_sent', 'vlm_latency_total_s', 'compute_s'):
            tot[k] += c.get(k, 0) or 0
        secs += g['duration_s']
    mins = secs / 60.0 or 1.0
    return {'totals': dict(tot), 'video_min': round(secs / 60.0, 1),
            'per_min': {k: round(v / mins, 3) for k, v in tot.items()},
            'vlm_latency_x_realtime': round(tot['vlm_latency_total_s'] / secs, 3) if secs else None}


# ---------------------------------------------------------------- gt stats
def gt_stats(gt):
    cls = Counter(); od = Counter(); clean = 0
    for g in gt.values():
        scored = [e for e in g['events'] if e.get('window')]
        if not scored:
            clean += 1
        for e in scored:
            cls[ckey(e)] += 1
            if ckey(e) == 'precondition_violation/order':
                od['dag_edge' if e.get('dag_edge_violation') else 'non_dag_edge'] += 1
    return {'recordings': len(gt), 'clean_recordings_no_scored_event': clean,
            'scored_events_by_class': dict(cls.most_common()),
            'order_dag_edge_split': dict(od),
            'order_dag_edge_recoverable_frac':
                round(od['dag_edge'] / (od['dag_edge'] + od['non_dag_edge']), 3)
                if (od['dag_edge'] + od['non_dag_edge']) else None}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir')
    ap.add_argument('--arms', help='comma-separated arm names under --results-dir')
    ap.add_argument('--only')
    ap.add_argument('--split', help="restrict scoring to a qualcomm-timeline split (e.g. test)")
    ap.add_argument('--rids-file', help="JSON {'rids':[...]} or [...]; restrict scoring to these")
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()

    rids = None
    if args.split:
        rids = split_rids(args.split)
    if args.rids_file:
        d = json.load(open(args.rids_file))
        fset = set(d['rids'] if isinstance(d, dict) else d)
        rids = fset if rids is None else (rids & fset)
    gt = load_gt(args.only, rids=rids)
    steps_by_rec = load_steps()

    if args.results_dir:
        arms = (args.arms or '').split(',') if args.arms else \
            [d for d in os.listdir(args.results_dir)
             if os.path.isdir(os.path.join(args.results_dir, d))]
        arm_results = {a: {rid: load_arm_file(args.results_dir, a, rid) for rid in gt} for a in arms}
    else:
        arms = ['oracle', 'silent']
        arm_results = {
            'oracle': {rid: arm_oracle(gt[rid], steps_by_rec.get(rid, [])) for rid in gt},
            'silent': {rid: arm_silent(gt[rid], steps_by_rec.get(rid, [])) for rid in gt}}

    scores = {'gt_stats': gt_stats(gt), 'arms': {}}
    for a in arms:
        ar = arm_results[a]
        scores['arms'][a] = {
            'fa1_membership': score_fa1(gt, ar, 0.0),
            'fa1_pm15s': score_fa1(gt, ar, 15.0),
            'fa1_pm30s': score_fa1(gt, ar, 30.0),
            'fa1_item_pm15s': score_fa1_item(gt, ar, 15.0),     # route 3: same-item identification
            'fa1_item_pm30s': score_fa1_item(gt, ar, 30.0),
            'fa2_gmean': score_fa2(gt, ar, GRACE),
            'stage': stage_acc(gt, ar, steps_by_rec),
            'cost': score_cost(gt, ar)}

    # ---- stdout summary ----
    gs = scores['gt_stats']
    print(f"GT: {gs['recordings']} recordings, {gs['clean_recordings_no_scored_event']} clean, "
          f"{sum(gs['scored_events_by_class'].values())} scored events")
    print(f"  order cheap-DAG-recoverable share: {gs['order_dag_edge_recoverable_frac']} "
          f"({gs['order_dag_edge_split']})")
    for a in arms:
        s = scores['arms'][a]
        p = s['fa1_membership']['pooled']
        fa2 = f"  | FA-2 G-Mean={round(s['fa2_gmean']['gmean_f1'],3)}" if s['fa2_gmean'] else ""
        print(f"\n[{a}] FA-1 membership pooled  P={p['precision']} R={p['recall']} F1={p['f1']}"
              f"{fa2}"
              f"  | stage acc={round(s['stage']['accuracy'],3) if s['stage']['accuracy'] is not None else None}")
        pi = s['fa1_item_pm15s']['pooled']
        print(f"     route3 same-item (±15s) pooled  P={_f(pi['precision'])} "
              f"R={_f(pi['recall'])} F1={_f(pi['f1'])}  (vs timing+type F1={_f(s['fa1_pm15s']['pooled']['f1'])})")
        for ck in SCORED_CLASSES:
            c = s['fa1_pm15s']['per_class'][ck]
            ci = s['fa1_item_pm15s']['per_class'][ck]
            print(f"     {ck:38s} timing+type F1={_f(c['f1'])} | same-item TP={ci['TP']:3d} "
                  f"FP={ci['FP']:4d} FN={ci['FN']:3d} P={_f(ci['precision'])} R={_f(ci['recall'])} F1={_f(ci['f1'])}")
        od = s['fa1_membership']['order_dag_breakdown']
        print(f"     order recall by detectability: dag-edge={_f(od['recall_dag_edge'])} "
              f"non-dag-edge={_f(od['recall_non_dag_edge'])}")

    if not args.no_write:
        out = os.path.join(GT_DIR, '_scores_corpus.json')
        json.dump(scores, open(out, 'w'), indent=1)
        print('\nwrote', out)


def _f(x):
    return f'{x:.3f}' if isinstance(x, float) else str(x)


if __name__ == '__main__':
    main()
