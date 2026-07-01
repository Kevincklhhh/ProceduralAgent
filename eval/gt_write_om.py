#!/usr/bin/env python3
"""Materialize the order/missing conversion workflow output into GT files.

Reads a workflow result JSON (the {results:[{recording_id, groups, dropped, verdict}]} object,
either bare or wrapped in a task-output envelope with a top-level "result" string/object) and
writes one file per recording:

  data/cc4d_proactive_om/{recording_id}.json

Each group becomes a flat reminder (t, subtype, content, source) plus group metadata
(members, opportunities, episode_span, one_shot). Kept SEPARATE from data/cc4d_proactive/
(execution mistakes) until a merge is explicitly approved. Only writes recordings whose
consistency-check verdict has no error-severity issue (ok=true), unless --force.

Usage: python eval/gt_write_om.py <workflow_output.json> [--force] [--no-write]
"""
import json, argparse, os

BASE = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(BASE, 'data', 'cc4d_proactive_om')


def extract_results(raw_text):
    """Pull the {results:[...]} payload out of a bare result or a task-output envelope."""
    obj = json.loads(raw_text)
    if 'results' in obj:
        return obj
    if 'result' in obj:                     # task-output envelope: result may be str or obj
        r = obj['result']
        return r if isinstance(r, dict) else json.loads(r)
    raise ValueError('no results/result key found in input')


def to_file(rec):
    rid = rec['recording_id']
    reminders = []
    for i, g in enumerate(rec.get('groups', []), 1):
        reminders.append({
            'id': f'{rid}_om{i}',
            't': g['t'],
            'subtype': g['subtype'],
            'content': g['content'],
            'anchor_step': g['anchor_step'],
            'source': g['source'],
            'members': g.get('members', []),
            'opportunities': g.get('opportunities', []),
            'episode_span': g.get('episode_span'),
            'one_shot': g.get('one_shot', True),
        })
    reminders.sort(key=lambda r: r['t'])
    return {'recording_id': rid, 'reminders': reminders,
            'dropped': rec.get('dropped', []), 'verdict': rec.get('verdict')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workflow_output')
    ap.add_argument('--force', action='store_true', help='write even if verdict.ok is false')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()

    payload = extract_results(open(args.workflow_output).read())
    recs = payload['results']
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
    written = skipped = 0
    n_order = n_missing = n_dropped = 0
    for rec in recs:
        v = rec.get('verdict')
        ok = bool(v and v.get('ok'))
        if not ok and not args.force:
            skipped += 1
            print(f'  SKIP {rec.get("recording_id")}: verdict not ok (use --force)')
            continue
        out = to_file(rec)
        n_order += sum(1 for r in out['reminders'] if r['subtype'] == 'order')
        n_missing += sum(1 for r in out['reminders'] if r['subtype'] == 'missing_step')
        n_dropped += len(out['dropped'])
        if not args.no_write:
            json.dump(out, open(os.path.join(OUT, f'{out["recording_id"]}.json'), 'w'), indent=1)
        written += 1
    print(json.dumps({'written': written, 'skipped_unverified': skipped,
                      'order_reminders': n_order, 'missing_reminders': n_missing,
                      'dropped_steps': n_dropped, 'out_dir': 'data/cc4d_proactive_om'}, indent=1))


if __name__ == '__main__':
    main()
