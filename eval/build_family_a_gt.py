#!/usr/bin/env python3
"""BOX 1 — Proactive-reminder GT generation (answer key). See docs/PIPELINE_THREE_BOXES.md.

Mechanically converts CC4D + Qualcomm annotations into a per-recording timeline of
proactive-reminder ground truth. MECHANICAL-ONLY as of 2026-06-15: every scored event's
window start comes from either a Qualcomm timestamp or a DAG-structural rule — nothing
that needs a human judgment call. Firewall: this is the GT/answer-key pipeline; it must
NOT share inputs with the recipe->sensor-map predictor (Box 3) beyond the recipe.

SCORED (two derivation mechanisms):
  execution_error      - fact: CC4D technique/preparation/measurement/temperature tag;
                         window start = Qualcomm visibility timestamp (REACTIVE).
  parameter_violation  - subtype timing; fact: CC4D Timing tag; window start = Qualcomm
                         timing timestamp, or step.end fallback when absent (the latest
                         the error is certainly visible — still mechanical). Same
                         derivation as execution_error; kept a separate class so per-class
                         P/R stays visible.
  precondition_violation/missing_step
                       - fact: CC4D Missing-Step tag; window start = first executed
                         transitive DAG-successor start (DERIVED). ~88% derivable; the
                         rest have no executed successor -> dropped by rule.
  precondition_violation/order
                       - fact: CC4D Order Error tag; window start = the out-of-order step's
                         own start_time (REACTIVE: visible when the step runs). One event
                         per tagged step. NO benign/harmful adjudication — the CC4D tag IS
                         the ground truth; we do not override it. `dag_edge_violation` is a
                         diagnostic flag (can DAG-state catch it?), it does not gate scoring.

SUSPENDED (non-mechanical; emitted into out['suspended'], never scored, no decision
points — reversible, see docs/PIPELINE_THREE_BOXES.md "Suspended GT"):
  execution_error/temperature (power-level subset) - no Qualcomm timestamp and no window
                         semantics ("low instead of high" is wrong from step start).

Excluded by design: safety (no GT), next-step guidance (= step recognition).

Outputs: data/cc4d_family_a/{recording_id}.json + data/cc4d_family_a/_summary.json
Usage:   python eval/build_family_a_gt.py [--only 8_45] [--no-write]
"""
import json, re, argparse, os
from collections import defaultdict, Counter

ANN = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc4d', 'annotations')
QC  = os.path.join(os.path.dirname(__file__), '..', 'data', 'qualcomm_interactive_cooking', 'main')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc4d_family_a')
GRACE = 15.0  # seconds appended to window ends

ACT2FILE = {
 'Microwave Egg Sandwich':'microwaveeggsandwich','Dressed Up Meatballs':'dressedupmeatballs',
 'Microwave Mug Pizza':'microwavemugpizza','Ramen':'ramen','Coffee':'coffee',
 'Pan Fried Tofu':'panfriedtofu','Mug Cake':'mugcake','Spiced Hot Chocolate':'spicedhotchocolate',
 'Microwave French Toast':'microwavefrenchtoast','Pinwheels':'pinwheels','Tomato Chutney':'tomatochutney',
 'Spicy Tuna Avocado Wraps':'spicytunaavocadowraps','Caprese Bruschetta':'capresebruschetta',
 'Sauted Mushrooms':'sautedmushrooms','Scrambled Eggs':'scrambledeggs','Blender Banana Pancakes':'blenderbananapancakes',
 'Herb Omelet with Fried Tomatoes':'herbomeletwithfriedtomatoes','Broccoli Stir Fry':'broccolistirfry',
 'Tomato Mozzarella Salad':'tomatomozzarellasalad','Butter Corn Cup':'buttercorncup',
 'Cucumber Raita':'cucumberraita','Zoodles':'zoodles','Cheese Pimiento':'cheesepimiento',
 'Breakfast Burritos':'breakfastburritos'}
TAGMAP = {'technique':'Technique Error','preparation':'Preparation Error','measurement':'Measurement Error',
          'timing':'Timing Error','temperature':'Temperature Error'}


def load():
    j = lambda f: json.load(open(os.path.join(ANN, 'annotation_json', f)))
    ann = j('complete_step_annotations.json')
    err = {r['recording_id']: r for r in j('error_annotations.json')}
    stepdesc = j('step_idx_description.json')
    import pandas as pd
    q = pd.concat([pd.read_parquet(os.path.join(QC, f'{s}-00000-of-00001.parquet'))
                   for s in ['train','validation','test']], ignore_index=True)
    qmap = {r['video_id']: r for _, r in q.iterrows()}
    graphs = {}
    for f in set(ACT2FILE.values()):
        g = json.load(open(os.path.join(ANN, 'task_graphs', f'{f}.json')))
        nodes = {int(k): v for k, v in g['steps'].items() if v not in ('START', 'END')}
        edges = [(a, b) for a, b in g['edges'] if a in nodes and b in nodes]
        g2n = {}                                  # global step_id -> graph node, by text
        for sid, d in stepdesc.items():
            for n, t in nodes.items():
                if t.strip() == d.strip():
                    g2n[int(sid)] = n
        graphs[f] = (nodes, edges, g2n)
    return ann, err, qmap, graphs


def build(vid, ann, err, qmap, graphs):
    rec = ann[vid]
    fkey = ACT2FILE.get(rec['activity_name'])
    nodes, edges, g2n = graphs[fkey]
    n2g = {n: s for s, n in g2n.items()}
    succ = defaultdict(set)                       # transitive closure: a step is "owed"
    for a, b in edges:                            # when ANY (not just direct) dependent runs
        succ[a].add(b)
    changed = True
    while changed:
        changed = False
        for x in list(succ):
            new = set().union(*(succ.get(y, set()) for y in succ[x])) if succ[x] else set()
            if not new <= succ[x]:
                succ[x] |= new; changed = True
    steps = rec['steps']
    seg = {s['step_id']: s for s in steps}
    tags = defaultdict(list)
    for sa in err.get(vid, {}).get('step_annotations', []):
        for e in sa.get('errors', []):
            tags[sa['step_id']].append(e)

    events, suspended, eid = [], [], 0
    def add(**kw):
        nonlocal eid; eid += 1
        events.append({'event_id': f'{vid}_e{eid}', **kw})

    # ---- Qualcomm-reactive errors: execution_error + parameter(timing) ----
    row = qmap.get(vid)
    matched_tagsteps = set()
    if row is not None:
        for t, typ in zip(row['output_timestamps'], row['output_types']):
            if 'mistake' not in typ:
                continue
            cat = typ.split('mistake_')[1].split('_error')[0]
            want = TAGMAP.get(cat)
            cont = [s for s in steps if s['start_time'] >= 0 and s['start_time']-10 <= t <= s['end_time']+10]
            st = next((s for s in cont if want in [e['tag'] for e in tags.get(s['step_id'], [])]), None)
            if not st:
                continue
            matched_tagsteps.add((st['step_id'], want))
            cls = 'parameter_violation' if cat == 'timing' else 'execution_error'
            add(cls=cls, subtype=cat, window=[round(float(t),1), round(st['end_time']+GRACE,1)],
                anchor_step=st['step_id'], source='qualcomm_ts+cc4d_tag', adjudication='n/a')

    # fallback for Timing tags with NO Qualcomm event -> step.end (mechanical: latest the
    # error is certainly visible). Temperature tags with no Qualcomm event are SUSPENDED
    # (power-level subset: no window semantics, anchor would be a guess).
    for sid, elist in tags.items():
        s = seg.get(sid)
        if not s or s['start_time'] < 0:
            continue
        if any(e['tag'] == 'Timing Error' for e in elist) and (sid, 'Timing Error') not in matched_tagsteps:
            add(cls='parameter_violation', subtype='timing',
                window=[round(s['end_time'],1), round(s['end_time']+GRACE,1)],
                anchor_step=sid, source='cc4d_tag_only(no_qualcomm_ts)',
                adjudication='n/a', flag='low_confidence_timing')
        if any(e['tag'] == 'Temperature Error' for e in elist) and (sid, 'Temperature Error') not in matched_tagsteps:
            suspended.append({'cls': 'execution_error', 'subtype': 'temperature', 'anchor_step': sid,
                              'source': 'cc4d_tag_only(no_qualcomm_ts)', 'reason': 'power_level_no_window'})

    # ---- precondition: missing step (CC4D tag + DAG successor start) ----
    for sid, elist in tags.items():
        if not any(e['tag'] == 'Missing Step' for e in elist):
            continue
        n = g2n.get(sid)
        if n is None:
            continue
        succ_starts = [seg[n2g[m]]['start_time'] for m in succ.get(n, [])
                       if n2g.get(m) in seg and seg[n2g[m]]['start_time'] >= 0]
        if not succ_starts:                       # undecidable: no executed successor -> drop
            add(cls='precondition_violation', subtype='missing_step', window=None,
                anchor_step=sid, source='cc4d_missing_tag+dag', adjudication='dropped',
                flag='no_executed_successor')
            continue
        ws = round(min(succ_starts), 1)
        add(cls='precondition_violation', subtype='missing_step', window=[ws, round(ws+GRACE,1)],
            reminder_id=f'missing_{nodes[n].split("-")[0].strip().lower()}_before_successor',
            anchor_step=sid, anchor='successor_start', source='cc4d_missing_tag+dag', adjudication='n/a')

    # ---- precondition: ORDER violation (SCORED, one event per CC4D Order-tagged step) ----
    # Ground truth = the CC4D Order Error tag itself. No benign/harmful adjudication:
    # overriding the human annotation with our own "this one's harmless" verdict is not a
    # legitimate move (it would make CC4D no longer GT). Every tagged step is reminder-worthy.
    # Window start = the out-of-order step's own start_time (the moment the deviation is
    # visible); end = step.end + grace. `dag_edge_violation` is a DIAGNOSTIC ONLY (can the
    # cheap DAG-state detector catch this deviation via a real edge? the ~55% recoverable
    # slice) — it does NOT gate scoring; non-DAG-edge order errors are scored exactly the same.
    for sid, elist in tags.items():
        oe = next((e for e in elist if e['tag'] == 'Order Error'), None)
        s = seg.get(sid)
        if oe is None or not s or s['start_time'] < 0:
            continue
        n = g2n.get(sid)
        dag_viol = False                           # does a real DAG edge expose this reorder?
        if n is not None:
            for a, b in edges:
                if b == n and n2g.get(a) in seg and seg[n2g[a]]['start_time'] >= 0 \
                        and seg[n2g[a]]['start_time'] > s['start_time']:
                    dag_viol = True; break         # prerequisite executed AFTER this step
                if a == n and n2g.get(b) in seg and seg[n2g[b]]['start_time'] >= 0 \
                        and seg[n2g[b]]['start_time'] < s['start_time']:
                    dag_viol = True; break          # dependent executed BEFORE this step
        m = re.search(r'(?:before|after)\s+(.*)', oe['description'].lower().strip())
        pivot = m.group(1).strip() if m else oe['description'].lower().strip()
        add(cls='precondition_violation', subtype='order',
            window=[round(s['start_time'], 1), round(s['end_time'] + GRACE, 1)],
            anchor_step=sid, pivot=pivot, dag_edge_violation=dag_viol,
            source='cc4d_order_tag', adjudication='n/a')

    # ---- decision points (FA-2): interrupt at window starts; silent at clean step ends ----
    dps = []
    fired = set()
    for ev in events:
        if ev.get('window'):
            dps.append({'t': ev['window'][0], 'label': 'interrupt', 'event_id': ev['event_id']})
            fired.add(round(ev['window'][0], 1))
    for s in sorted([s for s in steps if s['start_time'] >= 0], key=lambda s: s['end_time']):
        te = round(s['end_time'], 1)
        if all(abs(te - f) > GRACE for f in fired):
            dps.append({'t': te, 'label': 'silent', 'event_id': None, 'kind': 'step_completion'})

    last_end = max([s['end_time'] for s in steps if s['end_time'] >= 0], default=0.0)
    return {'recording_id': vid, 'activity_name': rec['activity_name'], 'recipe': fkey,
            'duration_s': round(last_end, 1), 'is_error': err.get(vid, {}).get('is_error', False),
            'events': events, 'suspended': suspended,
            'decision_points': sorted(dps, key=lambda d: d['t'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='single recording id, e.g. 8_45')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    ann, err, qmap, graphs = load()
    vids = [args.only] if args.only else list(ann)
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
    cls_count = Counter(); susp_count = Counter(); n_clean = 0; n_dropped = 0
    for vid in vids:
        if ACT2FILE.get(ann[vid]['activity_name']) is None:
            continue
        out = build(vid, ann, err, qmap, graphs)
        scored = [e for e in out['events'] if e.get('window')]
        for e in scored:
            cls_count[f"{e['cls']}/{e['subtype']}"] += 1
        for e in out['suspended']:
            susp_count[f"{e['cls']}/{e['subtype']}"] += 1
        n_dropped += sum(1 for e in out['events'] if e['adjudication'] == 'dropped')
        if not scored:
            n_clean += 1
        if not args.no_write:
            json.dump(out, open(os.path.join(OUT, f'{vid}.json'), 'w'), indent=1)
        if args.only:
            print(json.dumps(out, indent=1))
    if not args.only:
        summary = {'recordings': len(vids), 'recordings_no_scored_event': n_clean,
                   'dropped_undecidable': n_dropped, 'scored_events_by_class': dict(cls_count.most_common()),
                   'suspended_by_class': dict(susp_count.most_common())}
        print(json.dumps(summary, indent=1))
        if not args.no_write:
            json.dump(summary, open(os.path.join(OUT, '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
