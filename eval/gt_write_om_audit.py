#!/usr/bin/env python3
"""Materialize the order/missing USEFULNESS AUDIT workflow output into advisory sidecar files.

The audit is an LLM judgment layer (does each mechanically-built reminder make sense / help the
user?). It is ADVISORY ONLY: it never mutates the answer key in data/cc4d_proactive_om/. Keeping
it separate preserves the GT's reproducibility and the GT/predictor firewall.

Reads a workflow result JSON (bare {results:[...]} or a task-output envelope with "result")
and writes data/cc4d_proactive_om_audit/{recording_id}.json per recording, plus a corpus
_summary.json with aggregate usefulness stats.

Usage: python eval/gt_write_om_audit.py <workflow_output.json> [--no-write]
"""
import json, argparse, os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(BASE, 'data', 'cc4d_proactive_om_audit')


def extract_results(raw_text):
    obj = json.loads(raw_text)
    if 'results' in obj:
        return obj
    if 'result' in obj:
        r = obj['result']
        return r if isinstance(r, dict) else json.loads(r)
    raise ValueError('no results/result key found in input')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workflow_output')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    import glob
    payload = extract_results(open(args.workflow_output).read())
    recs = payload['results']
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
        for rec in recs:                                  # write/overwrite only the audited recs
            json.dump({'recording_id': rec['recording_id'], 'summary': rec.get('summary', ''),
                       'reminders': rec.get('reminders', [])},
                      open(os.path.join(OUT, f'{rec["recording_id"]}.json'), 'w'), indent=1)

    # recompute corpus summary from ALL sidecars on disk (so partial refreshes stay correct)
    scan = OUT if not args.no_write else None
    files = glob.glob(os.path.join(OUT, '[0-9]*.json')) if scan else []
    n_rem = 0; by_sev = Counter(); n_not_useful = n_not_actionable = n_not_sense = 0
    src = ([json.load(open(f)) for f in files] if files else recs)
    for rec in src:
        for r in rec.get('reminders', []):
            n_rem += 1
            by_sev[r.get('severity')] += 1
            n_not_useful += 0 if r.get('useful') else 1
            n_not_actionable += 0 if r.get('actionable_at_t') else 1
            n_not_sense += 0 if r.get('makes_sense') else 1
    summary = {'recordings': len(src), 'reminders_audited': n_rem,
               'by_severity': dict(by_sev), 'not_useful': n_not_useful,
               'not_actionable': n_not_actionable, 'does_not_make_sense': n_not_sense,
               'refreshed_in_last_run': [r['recording_id'] for r in recs],
               'out_dir': 'data/cc4d_proactive_om_audit'}
    print(json.dumps(summary, indent=1))
    if not args.no_write:
        json.dump(summary, open(os.path.join(OUT, '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
