#!/usr/bin/env python3
"""BOX 2 — corpus referee. Scores predicted reminders against the Box-1 truth table
(`data/cc4d_family_a/`) over the full CC4D Family A corpus. See docs/REMINDER_EVALUATION.md.

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
Outputs: data/cc4d_family_a/_scores_corpus.json  (+ stdout summary)
"""
import json, os, math, glob, argparse
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
GT_DIR = os.path.join(BASE, 'data', 'cc4d_family_a')
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
def load_gt(only=None):
    gt = {}
    for f in sorted(glob.glob(os.path.join(GT_DIR, '*.json'))):
        if os.path.basename(f).startswith('_'):
            continue
        d = json.load(open(f))
        if only and d['recording_id'] != only:
            continue
        gt[d['recording_id']] = d
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


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (tp + fp + fn) else None)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r, 'f1': f}


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
def score_fa2(gt, arm_res, tol):
    cm = Counter()                              # confusion over decision points
    for rid, g in gt.items():
        ptimes = [e['t'] for e in arm_res[rid]['events']]
        for dp in g['decision_points']:
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
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()

    gt = load_gt(args.only)
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
        print(f"\n[{a}] FA-1 membership pooled  P={p['precision']} R={p['recall']} F1={p['f1']}"
              f"  | FA-2 G-Mean={round(s['fa2_gmean']['gmean_f1'],3)}"
              f"  | stage acc={round(s['stage']['accuracy'],3) if s['stage']['accuracy'] is not None else None}")
        for ck in SCORED_CLASSES:
            c = s['fa1_membership']['per_class'][ck]
            print(f"     {ck:38s} TP={c['TP']:4d} FP={c['FP']:4d} FN={c['FN']:4d} "
                  f"P={_f(c['precision'])} R={_f(c['recall'])} F1={_f(c['f1'])}")
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
