#!/usr/bin/env python3
"""Assemble per-recording CONVERSION INPUTS for order/missing-step proactive reminders.

This is pure, deterministic data assembly — NO grouping/timing/content judgment (that is the
LLM's job in the convert workflow). It just projects CC4D + DAG annotations into clean
step_id space so each subagent gets the recipe graph, the executed trace, and the order/
missing tags without re-deriving the graph-node<->global-step_id text mapping itself.

Firewall: GT-side assembly; never fed to the recipe->sensor predictor.

For every recording carrying >=1 Order Error or Missing Step tag, writes:
  data/cc4d_proactive_om_input/{recording_id}.json

Usage: python eval/gt_build_om_input.py [--only 8_50] [--no-write]
"""
import json, argparse, os
from collections import defaultdict, Counter

from gt_build_proactive import ACT2FILE  # same activity_name -> graph-file stem map

BASE = os.path.join(os.path.dirname(__file__), '..')
ANN = os.path.join(BASE, 'data', 'cc4d', 'annotations')
PROACTIVE = os.path.join(BASE, 'data', 'cc4d_proactive')
OUT = os.path.join(BASE, 'data', 'cc4d_proactive_om_input')


def load():
    j = lambda f: json.load(open(os.path.join(ANN, 'annotation_json', f)))
    ann = j('complete_step_annotations.json')
    err = {r['recording_id']: r for r in j('error_annotations.json')}
    stepdesc = {int(k): v for k, v in j('step_idx_description.json').items()}
    graphs = {}
    for f in set(ACT2FILE.values()):
        g = json.load(open(os.path.join(ANN, 'task_graphs', f'{f}.json')))
        nodes = {int(k): v for k, v in g['steps'].items() if v not in ('START', 'END')}
        edges = [(a, b) for a, b in g['edges'] if a in nodes and b in nodes]
        g2n = {}                                    # global step_id -> graph node (by text)
        for sid, t in stepdesc.items():
            for n, nt in nodes.items():
                if nt.strip() == t.strip():
                    g2n[sid] = n
        graphs[f] = (nodes, edges, g2n)
    return ann, err, stepdesc, graphs


def build(vid, ann, err, stepdesc, graphs):
    rec = ann[vid]
    fkey = ACT2FILE.get(rec['activity_name'])
    nodes, edges, g2n = graphs[fkey]
    n2g = {n: s for s, n in g2n.items()}            # graph node -> global step_id

    tags = defaultdict(list)
    for sa in err.get(vid, {}).get('step_annotations', []):
        for e in sa.get('errors', []):
            tags[sa['step_id']].append(e)

    def text(sid):
        return stepdesc.get(sid, f'step {sid}')

    occ = Counter(s['step_id'] for s in rec['steps'])                 # total occurrences (loops)
    exec_occ = Counter(s['step_id'] for s in rec['steps'] if s['start_time'] >= 0)

    order_tags = [{'step_id': sid, 'text': text(sid), 'description': e['description']}
                  for sid, el in tags.items() for e in el if e['tag'] == 'Order Error']
    missing_tags = [{'step_id': sid, 'text': text(sid), 'description': e['description'],
                     'occurrences': occ[sid], 'executed_count': exec_occ[sid]}
                    for sid, el in tags.items() for e in el if e['tag'] == 'Missing Step']

    executed = sorted([{'step_id': s['step_id'], 'text': s['description'],
                        'start': round(float(s['start_time']), 1), 'end': round(float(s['end_time']), 1)}
                       for s in rec['steps'] if s['start_time'] >= 0 and s['end_time'] >= 0],
                      key=lambda s: s['start'])

    # DAG edges projected into global step_id space (prerequisite -> dependent); keep node-space
    # too for any edge whose endpoints don't map by text (so the agent still sees the structure).
    dag_edges = []
    for a, b in edges:
        ga, gb = n2g.get(a), n2g.get(b)
        dag_edges.append({'from_step_id': ga, 'to_step_id': gb,
                          'from_text': nodes[a], 'to_text': nodes[b]})

    canonical = [{'step_id': n2g.get(n), 'text': t} for n, t in nodes.items()]

    existing = []
    p = os.path.join(PROACTIVE, f'{vid}.json')
    if os.path.exists(p):
        existing = [{'t': r['t'], 'subtype': r['subtype'], 'content': r['content'],
                     'anchor_step': r.get('anchor_step')}
                    for r in json.load(open(p)).get('reminders', [])]

    last_end = max([s['end'] for s in executed], default=0.0)
    return {'recording_id': vid, 'activity_name': rec['activity_name'], 'recipe': fkey,
            'duration_s': round(last_end, 1),
            'canonical_steps': canonical, 'dag_edges': dag_edges,
            'executed_trace': executed, 'order_tags': order_tags, 'missing_tags': missing_tags,
            'existing_execution_reminders': existing}


def has_om(vid, err):
    for sa in err.get(vid, {}).get('step_annotations', []):
        for e in sa.get('errors', []):
            if e['tag'] in ('Order Error', 'Missing Step'):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    ann, err, stepdesc, graphs = load()
    vids = [args.only] if args.only else list(ann)
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
    written = 0
    ids = []
    for vid in vids:
        if ACT2FILE.get(ann[vid]['activity_name']) is None or not has_om(vid, err):
            continue
        out = build(vid, ann, err, stepdesc, graphs)
        ids.append(vid)
        if not args.no_write:
            json.dump(out, open(os.path.join(OUT, f'{vid}.json'), 'w'), indent=1)
            written += 1
        if args.only:
            print(json.dumps(out, indent=1))
    if not args.only:
        print(json.dumps({'recordings_with_order_or_missing': len(ids), 'written': written,
                          'out_dir': 'data/cc4d_proactive_om_input', 'ids': ids}, indent=1))


if __name__ == '__main__':
    main()
