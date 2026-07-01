#!/usr/bin/env python3
"""Proposed approach v1 -- order/missing-step detector from recipe-DAG + step recognition.

The A-solve arm for the precondition_violation classes (order, missing_step): a cheap graph
check, NO VLM in the detection itself. Watch the recognized step stream and compare against the
recipe DAG (prerequisite -> dependent edges):

  ORDER    a recognized step ran before a prerequisite (or a prerequisite ran after a
           dependent) -> a DAG precondition edge is violated. Violating steps joined by a DAG
           edge are clustered; one reminder per deviation, anchored at the earliest violator.
  MISSING  a recipe step that never appears in the recognized stream but has a recognized
           transitive dependent -> the skip becomes evident at that dependent's start.

This mirrors eval/gt_build_om.py's logic EXACTLY but is FIREWALL-CLEAN: it never reads CC4D's
Order/Missing answer tags. It only uses (a) the recipe DAG + canonical step list (shared recipe
knowledge, allowed) and (b) a recognized step track (its input). GT uses the annotator tags to
decide which violations are real; v1 trusts the DAG + recognition alone -- so the gap between
this and the oracle GT is exactly "DAG violations CC4D did not tag" (precision) and "skips the
recognizer missed" (recall).

Step track:
  --track oracle              use the GT step segmentation as a perfect recognizer (upper bound)
  --track <results_dir>/<arm> use a T1 arm's stage_intervals (the honest, deployable number)

Output: unified arm format -> <out-dir>/<arm>/<rid>.json, scored by eval/eval_score_corpus.py.

Usage:
  python eval/proposed_om_v1.py --track oracle --arm om_v1_oracle
  python eval/proposed_om_v1.py --track experiments/t1_baseline/qwen36_i10 --arm om_v1_t1
"""
import argparse, json, os
from collections import defaultdict, deque

from gt_build_om_input import load as load_inputs, build as build_input, has_om, ACT2FILE

BASE = os.path.join(os.path.dirname(__file__), '..')


def clean(text):
    if '-' in text:
        head, rest = text.split('-', 1)
        if ' ' not in head and rest.strip().lower().startswith(head.strip().lower() + ' '):
            return rest.strip()
    return text.strip()


def humanlist(items):
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ''
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def detect(inp, track):
    """track: list of {step_id,start,end} (the recognized step stream). Returns events list."""
    edges = [(e['from_step_id'], e['to_step_id']) for e in inp['dag_edges']
             if e['from_step_id'] is not None and e['to_step_id'] is not None]
    canon = {s['step_id'] for s in inp['canonical_steps'] if s['step_id'] is not None}
    text = {s['step_id']: s['text'] for s in inp['canonical_steps'] if s['step_id'] is not None}

    estart, eend = {}, {}
    for s in track:
        sid = s['step_id']
        text.setdefault(sid, f'step {sid}')
        if sid not in estart or s['start'] < estart[sid]:
            estart[sid] = s['start']; eend[sid] = s['end']
    executed = set(estart)

    prereqs, deps = defaultdict(set), defaultdict(set)
    for a, b in edges:
        deps[a].add(b); prereqs[b].add(a)

    events = []

    # ---------------- ORDER (ungated: every executed step checked against the DAG) ----------
    def is_viol(x):
        tx = estart[x]
        if any(p in executed and estart[p] > tx for p in prereqs.get(x, ())):
            return True
        if any(d in executed and estart[d] < tx for d in deps.get(x, ())):
            return True
        return False

    viol = {x for x in executed if is_viol(x)}
    parent = {x: x for x in viol}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in edges:
        if a in viol and b in viol:
            parent[find(a)] = find(b)
    clusters = defaultdict(list)
    for x in viol:
        clusters[find(x)].append(x)
    for members in clusters.values():
        anchor = min(members, key=lambda x: estart[x])
        events.append({'t': round(estart[anchor], 1), 'class': 'precondition_violation',
                       'subtype': 'order', 'step_id': anchor,
                       'message': f'{clean(text.get(anchor, f"step {anchor}"))} was done out of order '
                                  'relative to the recipe steps.'})

    # ---------------- MISSING (ungated: canonical steps absent from the recognized stream) --
    skipped = [s for s in canon if s not in executed]

    def triggers(s):
        seen, q, trig = {s}, deque([s]), set()
        while q:
            x = q.popleft()
            for d in deps.get(x, ()):
                if d in seen:
                    continue
                seen.add(d)
                (trig.add(d) if d in executed else q.append(d))
        return trig

    by_anchor = defaultdict(list)
    for s in skipped:
        tr = triggers(s)
        if not tr:
            continue                                  # no recognized dependent -> no live moment
        by_anchor[min(tr, key=lambda d: estart[d])].append(s)
    for anchor, members in by_anchor.items():
        mem_text = humanlist([clean(text.get(m, f'step {m}')) for m in members])
        verb = 'it is' if len(members) == 1 else 'they are'
        events.append({'t': round(estart[anchor], 1), 'class': 'precondition_violation',
                       'subtype': 'missing_step', 'step_id': anchor,
                       'message': f'You skipped {mem_text}; {verb} needed before '
                                  f'{clean(text.get(anchor, f"step {anchor}"))}.'})

    events.sort(key=lambda e: e['t'])
    return events


def track_from_arm(results_dir, arm, rid):
    p = os.path.join(results_dir, arm, rid + '.json')
    if not os.path.exists(p):
        return None
    si = json.load(open(p)).get('stage_intervals', [])
    return [{'step_id': s['stage'], 'start': float(s['start_s']), 'end': float(s['end_s'])} for s in si]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default='oracle',
                    help="'oracle' (GT step segmentation) or '<results_dir>/<arm>' for a T1 arm")
    ap.add_argument('--out-dir', default='experiments/proposed_om')
    ap.add_argument('--arm', required=True)
    ap.add_argument('--only')
    args = ap.parse_args()

    ann, err, stepdesc, graphs = load_inputs()
    vids = [args.only] if args.only else list(ann)
    out = os.path.join(BASE, args.out_dir, args.arm)
    os.makedirs(out, exist_ok=True)

    arm_dir = None
    if args.track != 'oracle':
        results_dir, arm = os.path.split(args.track.rstrip('/'))
        arm_dir = (results_dir, arm)

    n = nev = 0
    for vid in vids:
        if ACT2FILE.get(ann[vid]['activity_name']) is None:
            continue
        inp = build_input(vid, ann, err, stepdesc, graphs)
        if args.track == 'oracle':
            track = inp['executed_trace']               # perfect recognizer
        else:
            track = track_from_arm(arm_dir[0], arm_dir[1], vid)
            if track is None:
                continue                                # arm did not cover this recording
        events = detect(inp, track)
        res = {'recording': vid, 'arm': args.arm, 'stage_intervals': [
                   {'stage': s['step_id'], 'start_s': s['start'], 'end_s': s['end']} for s in track],
               'events': events, 'escalation_requests': [],
               'cost': {'vlm_calls': 0, 'frames_sent': 0, 'vlm_latency_total_s': 0.0, 'compute_s': 0.0},
               '_meta': {'detector': 'proposed_om_v1', 'track': args.track}}
        json.dump(res, open(os.path.join(out, vid + '.json'), 'w'), indent=1)
        n += 1; nev += len(events)
    print(f"wrote {n} recordings, {nev} events -> {out}")


if __name__ == '__main__':
    main()
