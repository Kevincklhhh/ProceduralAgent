#!/usr/bin/env python3
"""Generate one task JSON per CC4D recipe (24 total) -> tasks/cc4d/{recipe}.json.

FIREWALL-CLEAN (Box 3 predictor): inputs are recipe knowledge ONLY —
  - task_graphs/{recipe}.json        (DAG: node text + edges)
  - annotation_json/step_idx_description.json  (global step_idx -> description)
No error tags, no Qualcomm timestamps, no per-recording execution traces are read.

step_id is the CC4D GLOBAL step_idx so the output is directly scorable against
complete_step_annotations.json (which T1 / stage_acc uses as GT). Durations are NOT
imported from recordings (that would be a trace); only a `duration_constraint_s` parsed
from the step TEXT (e.g. "for 1 minute") is kept.
"""
import json, os, re, glob, collections

BASE = os.path.join(os.path.dirname(__file__), '..')
GRAPHS = os.path.join(BASE, 'data', 'cc4d', 'annotations', 'task_graphs')
DESC = os.path.join(BASE, 'data', 'cc4d', 'annotations', 'annotation_json', 'step_idx_description.json')
OUT = os.path.join(BASE, 'tasks', 'cc4d')


def norm(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def instruction_of(text):
    """'Fill-Fill a microwave-safe mug...' -> 'Fill a microwave-safe mug...'"""
    return text.split('-', 1)[1].strip() if '-' in text else text.strip()


def slug_of(instr):
    words = re.findall(r'[a-z0-9]+', instr.lower())[:4]
    return '_'.join(words) or 'step'


_UNIT_S = {'second': 1, 'seconds': 1, 'sec': 1, 'minute': 60, 'minutes': 60,
           'min': 60, 'hour': 3600, 'hours': 3600}


def duration_constraint_s(text):
    m = re.search(r'(\d+)\s*(seconds?|sec|minutes?|min|hours?)', text.lower())
    return int(m.group(1)) * _UNIT_S[m.group(2)] if m else None


def topo_order(nodes, edges):
    """Kahn over real-step nodes only. nodes: set of local ids; edges: [(a,b)] a before b."""
    adj = collections.defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        if a in nodes and b in nodes:
            adj[a].append(b); indeg[b] += 1
    # stable: pop smallest ready id for deterministic output
    ready = sorted([n for n in nodes if indeg[n] == 0], key=int)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort(key=int)
    order += [n for n in nodes if n not in order]   # cycle fallback (shouldn't happen)
    return order


def main():
    desc2idx = {norm(v): int(k) for k, v in json.load(open(DESC)).items()}
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for gf in sorted(glob.glob(os.path.join(GRAPHS, '*.json'))):
        stem = os.path.basename(gf)[:-5]
        g = json.load(open(gf))
        raw = g['steps']                                   # {local_id: text}
        real = {k: t for k, t in raw.items()
                if norm(t) not in ('start', 'end')}
        edges = [(str(a), str(b)) for a, b in g['edges']]
        order = topo_order(set(real), edges)

        local2global, unmatched = {}, []
        for lid, txt in real.items():
            gi = desc2idx.get(norm(txt))
            local2global[lid] = gi
            if gi is None:
                unmatched.append(txt)

        preds = collections.defaultdict(list)
        for a, b in edges:
            if a in real and b in real:
                preds[b].append(a)

        steps = []
        for i, lid in enumerate(order, 1):
            txt = real[lid]
            instr = instruction_of(txt)
            gi = local2global[lid]
            pre = sorted(local2global[a] for a in preds[lid]
                         if local2global.get(a) is not None)
            step = {'step_id': gi, 'order': i, 'name': slug_of(instr),
                    'instruction': instr, 'preconditions': pre}
            dc = duration_constraint_s(txt)
            if dc is not None:
                step['duration_constraint_s'] = dc
            steps.append(step)

        # provisional, text-grounded timing reminders only (Box-3 compiler will expand)
        reminders = []
        for s in steps:
            if 'duration_constraint_s' in s:
                d = s['duration_constraint_s']
                reminders.append({
                    'reminder_id': f"timing_{s['name']}",
                    'step_id': s['step_id'], 'class': 'parameter_violation',
                    'subtype': 'timing',
                    'trigger': f"this step has run noticeably beyond {d}s",
                    'message': f"The recipe says about {d}s for this step — check timing.",
                    'type': 'warning', '_provisional': True})

        task = {
            'task_id': stem,
            'title': re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', stem).title(),
            'source': f"data/cc4d/annotations/task_graphs/{stem}.json",
            '_generated_by': 'scripts/gen_task_jsons.py (firewall-clean: DAG + step text only)',
            'steps': steps,
            'reminders': reminders,
            'allowed_assistant_actions': ['none', 'reminder', 'warning', 'ask_confirmation'],
        }
        json.dump(task, open(os.path.join(OUT, stem + '.json'), 'w'), indent=2)
        summary.append((stem, len(steps), len(reminders), len(unmatched), unmatched))

    print(f"wrote {len(summary)} task JSONs -> {OUT}")
    tot_un = sum(u for *_, u, _ in summary)
    for stem, ns, nr, nu, un in summary:
        flag = f"  !! {nu} UNMATCHED: {un}" if nu else ""
        print(f"  {stem:28s} steps={ns:2d} timing_reminders={nr}{flag}")
    print(f"\nTOTAL unmatched nodes (no global step_idx): {tot_un}")


if __name__ == '__main__':
    main()
