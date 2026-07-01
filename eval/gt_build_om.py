#!/usr/bin/env python3
"""Deterministic order/missing proactive-reminder GT builder (NO LLM).

Policy (decided 2026-06-28):
  ORDER  - CC4D's canonical *sequence* order is IGNORED. An order error exists only when the
           execution violates a recipe DAG precondition edge: a dependent ran before its
           prerequisite, or a prerequisite ran after its dependent. CC4D Order tags that break
           no DAG edge are DROPPED (a DAG-legal reordering is not reminder-worthy). Violating
           tagged steps are clustered by DAG adjacency -> one one-shot reminder per deviation,
           anchored at the earliest violating step's start.
  MISSING - CC4D Missing-tagged (skipped) steps, anchored at the start of the first executed
           TRANSITIVE dependent (follow dag_edges through skipped steps to the first executed
           step). Skips sharing that anchor are grouped; a skip with no executed transitive
           dependent is DROPPED (no grounded live moment).

content is TEMPLATED deterministically (swap an LLM in later for richer phrasing).

Reuses eval/gt_build_om_input.py for the per-recording step_id-space projection.
Writes data/cc4d_proactive_om/{recording_id}.json (one file per recording with >=1 order or
missing tag, even if it nets zero reminders, so the drops are recorded).
Firewall: GT-side answer key; never feed the recipe->sensor predictor.

Usage: python eval/gt_build_om.py [--only 8_50] [--no-write]
"""
import json, argparse, os
from collections import defaultdict, deque, Counter

from gt_build_om_input import load as load_inputs, build as build_input, has_om, ACT2FILE

BASE = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(BASE, 'data', 'cc4d_proactive_om')


def clean(text):
    """Strip CC4D's 'Verb-Verb full description' duplicate prefix -> 'full description'."""
    if '-' in text:
        head, rest = text.split('-', 1)
        if ' ' not in head and rest.strip().lower().startswith(head.strip().lower() + ' '):
            return rest.strip()
    return text.strip()


def humanlist(items):
    items = list(items)
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def convert(inp):
    rid = inp['recording_id']
    edges = [(e['from_step_id'], e['to_step_id']) for e in inp['dag_edges']
             if e['from_step_id'] is not None and e['to_step_id'] is not None]
    text = {s['step_id']: s['text'] for s in inp['canonical_steps'] if s['step_id'] is not None}
    estart, eend = {}, {}
    for s in inp['executed_trace']:
        text.setdefault(s['step_id'], s['text'])
        if s['step_id'] not in estart or s['start'] < estart[s['step_id']]:
            estart[s['step_id']] = s['start']; eend[s['step_id']] = s['end']
    executed = set(estart)
    prereqs, deps = defaultdict(set), defaultdict(set)     # x: {P: P->x} ; x: {D: x->D}
    for a, b in edges:
        deps[a].add(b); prereqs[b].add(a)

    reminders, dropped = [], []

    # ---------------- MISSING ----------------
    # CC4D occasionally tags a step "Missing Step" even though it has an executed segment
    # (annotation peculiarity). Such a step was NOT skipped — exclude it (else we emit a false
    # "you skipped X" reminder). Recorded in dropped for transparency.
    skipped = []
    for t in inp['missing_tags']:
        sid = t['step_id']
        if sid not in executed:
            skipped.append(sid)
            continue
        # tagged Missing but has an executed segment -> not a clean skip. Two sub-cases:
        if t.get('occurrences', 1) > 1:        # repeated/loop step: a later iteration skipped
            reason = (f"loop/repeated step ({t.get('executed_count', '?')} of {t['occurrences']} "
                      "iterations executed); a skipped iteration is not representable in the "
                      "step_id-keyed GT")
        else:                                   # single-occurrence contradiction
            reason = "CC4D Missing tag on a step executed once (contradictory annotation); not a skip"
        dropped.append({'step_id': sid, 'subtype': 'missing_step', 'reason': reason})
    skipset = set(skipped)

    def triggers(s):
        """Executed steps reachable from skipped step s through skipped-only intermediates
        (the first executed step on each forward path = the moment the skip becomes evident)."""
        seen, q, trig = {s}, deque([s]), set()
        while q:
            x = q.popleft()
            for d in deps.get(x, ()):
                if d in seen:
                    continue
                seen.add(d)
                if d in executed:
                    trig.add(d)               # first executed on this path; stop expanding
                else:
                    q.append(d)               # skipped / non-executed: keep going
        return trig

    by_anchor = defaultdict(list)
    mem_trig = {}
    for s in skipped:
        tr = triggers(s)
        if not tr:
            dropped.append({'step_id': s, 'subtype': 'missing_step',
                            'reason': 'no executed transitive dependent (no grounded live moment)'})
            continue
        anchor = min(tr, key=lambda d: estart[d])
        by_anchor[anchor].append(s); mem_trig[s] = tr
    for anchor, members in by_anchor.items():
        trig_all = set().union(*(mem_trig[m] for m in members))
        opp = sorted({round(estart[d], 1) for d in trig_all})
        t = round(estart[anchor], 1)
        mem_text = humanlist([clean(text.get(m, f'step {m}')) for m in members])
        verb = 'it is' if len(members) == 1 else 'they are'
        reminders.append({
            'subtype': 'missing_step', 't': t, 'anchor_step': anchor,
            'content': f'You skipped {mem_text}; {verb} needed before {clean(text.get(anchor, f"step {anchor}"))}.',
            'members': sorted(members), 'opportunities': opp,
            'episode_span': [t, round(max(eend[d] for d in trig_all), 1)],
            'one_shot': True, 'source': 'cc4d_missing+dag'})

    # ---------------- ORDER ----------------
    order_tagged = {t['step_id'] for t in inp['order_tags']}

    def is_viol(x):
        if x not in executed:
            return False
        tx = estart[x]
        if any(p in executed and estart[p] > tx for p in prereqs.get(x, ())):
            return True                        # a prerequisite ran AFTER x
        if any(d in executed and estart[d] < tx for d in deps.get(x, ())):
            return True                        # a dependent ran BEFORE x
        return False

    viol = {x for x in order_tagged if is_viol(x)}
    for x in sorted(order_tagged - viol):
        dropped.append({'step_id': x, 'subtype': 'order',
                        'reason': 'CC4D Order tag breaks no DAG edge (DAG-legal reordering); ignored'})

    parent = {x: x for x in viol}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in edges:                         # cluster violating steps joined by a DAG edge
        if a in viol and b in viol:
            parent[find(a)] = find(b)
    clusters = defaultdict(list)
    for x in viol:
        clusters[find(x)].append(x)

    def late_ancestors(anchor):
        """Transitive prerequisites of anchor that ran after it or were skipped (to name)."""
        seen, q, out = {anchor}, deque([anchor]), set()
        while q:
            x = q.popleft()
            for p in prereqs.get(x, ()):
                if p in seen:
                    continue
                seen.add(p); q.append(p)
                if p in skipset or (p in executed and estart[p] > estart[anchor]):
                    out.add(p)
        return out

    for members in clusters.values():
        anchor = min(members, key=lambda x: estart[x])
        t = round(estart[anchor], 1)
        opp = sorted({round(estart[x], 1) for x in members})
        anc = late_ancestors(anchor)
        a_text = clean(text.get(anchor, f'step {anchor}'))
        if anc:
            pr = humanlist([clean(text.get(p, f'step {p}'))
                            for p in sorted(anc, key=lambda p: text.get(p, ''))])
            content = f'{a_text} was done out of order — it should come after {pr}.'
        else:
            content = f'{a_text} was done out of order relative to the recipe steps.'
        reminders.append({
            'subtype': 'order', 't': t, 'anchor_step': anchor, 'content': content,
            'members': sorted(members), 'opportunities': opp,
            'episode_span': [t, round(max(eend[x] for x in members), 1)],
            'one_shot': True, 'source': 'cc4d_order+dag'})

    reminders.sort(key=lambda r: r['t'])
    for i, r in enumerate(reminders, 1):
        r_id = {'id': f'{rid}_om{i}'}; r_id.update(r); reminders[i-1] = r_id
    return {'recording_id': rid, 'activity_name': inp['activity_name'], 'recipe': inp['recipe'],
            'duration_s': inp['duration_s'], 'reminders': reminders, 'dropped': dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    ann, err, stepdesc, graphs = load_inputs()
    vids = [args.only] if args.only else list(ann)
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
    n_files = n_order = n_missing = n_clean = 0
    drop = Counter()
    for vid in vids:
        if ACT2FILE.get(ann[vid]['activity_name']) is None or not has_om(vid, err):
            continue
        out = convert(build_input(vid, ann, err, stepdesc, graphs))
        n_files += 1
        no = sum(1 for r in out['reminders'] if r['subtype'] == 'order')
        nm = sum(1 for r in out['reminders'] if r['subtype'] == 'missing_step')
        n_order += no; n_missing += nm
        if not out['reminders']:
            n_clean += 1
        for d in out['dropped']:
            drop[d['subtype']] += 1
        if not args.no_write:
            json.dump(out, open(os.path.join(OUT, f'{vid}.json'), 'w'), indent=1)
        if args.only:
            print(json.dumps(out, indent=1))
    if not args.only:
        print(json.dumps({'recordings': n_files, 'order_reminders': n_order,
                          'missing_reminders': n_missing,
                          'recordings_netting_zero_reminders': n_clean,
                          'dropped': dict(drop), 'out_dir': 'data/cc4d_proactive_om'}, indent=1))


if __name__ == '__main__':
    main()
