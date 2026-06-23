#!/usr/bin/env python3
"""PROBING MODE (firewall RELAXED) — gather, per recipe, every proactive reminder that
actually occurred across all recordings of that recipe.

This deliberately reads the Box-1 input (CC4D error annotations) and joins it onto the
Box-3 recipe DAG (tasks/cc4d/<recipe>.json) by global step_id. The result is the EMPIRICAL
error space per step: which reminder kinds were ever observed on each step and how often.

Two companion files per recipe:
  <recipe>.instances.json  -- FULL RECORD: one `instances` entry per observed error, each a
                              pointer back to its source (recording_id, cc4d_step_id,
                              video_window). Kept for traceability.
  <recipe>.reminders.json  -- AGENT-FACING slim view: per step, observed subtypes + counts +
                              unique example descriptions, NO provenance. This is what the
                              criteria-authoring agent reads (it does not need the record).

>>> FIREWALL NOTE <<<
The clean predictor authors checks from RECIPE TEXT ALONE (see docs/PIPELINE_THREE_BOXES.md,
docs/REMINDER_RUNTIME.md). This script is the opposite: it is a *probing / training* aid that
peeks at the answer key to see what the recipe-anticipated checks SHOULD cover. Its output
(`tasks/cc4d_probe/<recipe>.reminders.json`) must NOT be fed into a firewall-clean Box-3
evaluation — it is for authoring/analysis only. Outputs are written under `tasks/cc4d_probe/`
(all probing artifacts live there) and every file is stamped `"_mode":
"PROBING_FIREWALL_RELAXED"` so they can't be confused with firewall-clean criteria.

Usage:
  python scripts/probe_recipe_reminders.py            # all recipes
  python scripts/probe_recipe_reminders.py --recipe spicedhotchocolate
"""
import json, os, sys, argparse
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
ANN = os.path.join(BASE, 'data', 'cc4d', 'annotations', 'annotation_json')
CC4D = os.path.join(BASE, 'tasks', 'cc4d')          # recipe task jsons (firewall-clean source)
OUT = os.path.join(BASE, 'tasks', 'cc4d_probe')     # all probing artifacts live here

# CC4D error tag -> (reminder class, subtype) — the same taxonomy Box 1/Box 2 score on.
TAG2REMINDER = {
    'Technique Error':    ('execution_error',        'technique'),
    'Preparation Error':  ('execution_error',        'preparation'),
    'Measurement Error':  ('execution_error',        'measurement'),
    'Temperature Error':  ('execution_error',        'temperature'),
    'Timing Error':       ('parameter_violation',    'timing'),
    'Missing Step':       ('precondition_violation',  'missing_step'),
    'Order Error':        ('precondition_violation',  'order'),
    'Other':              ('other',                   'other'),
}


def load_annotations():
    """Return (error_records, activity_id -> recipe_file_stem)."""
    from eval.gt_build_family_a import ACT2FILE
    err = json.load(open(os.path.join(ANN, 'error_annotations.json')))
    ann = json.load(open(os.path.join(ANN, 'complete_step_annotations.json')))
    id2file = {}
    for r in ann.values():
        f = ACT2FILE.get(r['activity_name'])
        if f:
            id2file[r['activity_id']] = f
    return err, id2file


def probe_recipe(stem, activity_id, err_records):
    """Aggregate every observed error onto the recipe DAG, keyed by step_id."""
    task = json.load(open(os.path.join(CC4D, f'{stem}.json')))
    steps = {s['step_id']: s for s in task['steps']}

    recs = [r for r in err_records if r['activity_id'] == activity_id]
    n_error_recs = sum(1 for r in recs if r.get('is_error'))

    # per (step_id, subtype): occurrence count, recordings touched, and one `instances` entry
    # per observed error -- each a pointer back to the source annotation (which recording, what
    # was written, and where in the video) so any probe-derived check is traceable to its origin.
    agg = defaultdict(lambda: {'class': None, 'tag': None,
                               'recordings': set(), 'instances': []})
    unmapped = Counter()  # error step_ids that don't exist in the recipe DAG

    for r in recs:
        for sa in r['step_annotations']:
            sid = sa['step_id']
            for e in sa.get('errors', []):
                tag = e.get('tag')
                cls, sub = TAG2REMINDER.get(tag, ('other', 'other'))
                if sid not in steps:
                    unmapped[(sid, sub)] += 1
                    continue
                a = agg[(sid, sub)]
                a['class'], a['tag'] = cls, tag
                a['recordings'].add(r['recording_id'])
                st, en = sa.get('start_time', -1), sa.get('end_time', -1)
                a['instances'].append({
                    'recording_id': r['recording_id'],
                    'cc4d_step_id': sid,
                    'description': (e.get('description') or '').strip(),
                    'video_window': [round(st, 1), round(en, 1)] if st is not None and st >= 0 else None,
                })

    # assemble per-step view in recipe order
    out_steps = []
    for sid, s in sorted(steps.items(), key=lambda kv: kv[1].get('order', 0)):
        rem = []
        for (sid2, sub), a in agg.items():
            if sid2 != sid:
                continue
            rem.append({'subtype': sub, 'class': a['class'], 'tag': a['tag'],
                        'n_occurrences': len(a['instances']),
                        'n_recordings': len(a['recordings']),
                        'instances': a['instances']})
        rem.sort(key=lambda x: -x['n_occurrences'])
        out_steps.append({
            'step_id': sid, 'order': s.get('order'),
            'instruction': s.get('instruction', ''),
            'duration_constraint_s': s.get('duration_constraint_s'),
            'preconditions': s.get('preconditions', []),
            'observed_reminders': rem,
        })

    return {
        '_mode': 'PROBING_FIREWALL_RELAXED',
        '_note': ('FULL RECORD (provenance). Authored from observed CC4D error annotations '
                  '(Box-1 input), NOT recipe text alone. Each observed error is one `instances` '
                  'entry pointing back to its source (recording_id, cc4d_step_id, video_window). '
                  'The criteria-authoring agent reads the slim companion <recipe>.reminders.json, '
                  'not this file. For authoring/analysis only; do NOT feed into firewall-clean '
                  'Box-3 evaluation.'),
        'recipe': stem, 'activity_id': activity_id,
        'n_recordings': len(recs), 'n_error_recordings': n_error_recs,
        'steps': out_steps,
        'unmapped_step_ids': [{'step_id': sid, 'subtype': sub, 'n': n}
                              for (sid, sub), n in unmapped.most_common()],
    }


def make_brief(rep):
    """Slim, agent-facing view: per step, the observed subtypes with counts and a few unique
    descriptions -- but NO per-instance provenance (recording_id / video_window). That detail
    lives in the companion <recipe>.instances.json; the authoring agent does not need it."""
    brief = {k: v for k, v in rep.items() if k != 'steps'}
    brief['_note'] = ('AGENT-FACING slim view for criteria authoring (see '
                      'tasks/CRITERIA_GENERATION_PROBING.md). Per step: observed reminder '
                      'subtypes, counts, and unique example descriptions. The full per-instance '
                      'record (recording_id, cc4d_step_id, video_window) is the companion '
                      '<recipe>.instances.json -- kept for traceability, not needed to author.')
    steps = []
    for s in rep['steps']:
        rems = []
        for rm in s['observed_reminders']:
            descs = []
            for ins in rm['instances']:
                d = ins['description']
                if d and d not in descs:
                    descs.append(d)
            rems.append({'subtype': rm['subtype'], 'class': rm['class'], 'tag': rm['tag'],
                         'n_occurrences': rm['n_occurrences'], 'n_recordings': rm['n_recordings'],
                         'descriptions': descs})
        s2 = {k: v for k, v in s.items() if k != 'observed_reminders'}
        s2['observed_reminders'] = rems
        steps.append(s2)
    brief['steps'] = steps
    return brief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', help='single recipe stem (default: all)')
    args = ap.parse_args()

    err_records, id2file = load_annotations()
    file2id = {f: i for i, f in id2file.items()}

    if args.recipe:
        stems = [args.recipe]
    else:
        stems = sorted(f[:-5] for f in os.listdir(CC4D)
                       if f.endswith('.json') and '.' not in f[:-5])

    os.makedirs(OUT, exist_ok=True)
    corpus = []
    for stem in stems:
        aid = file2id.get(stem)
        if aid is None:
            print(f'  skip {stem}: no activity_id (not in CC4D annotations)')
            continue
        rep = probe_recipe(stem, aid, err_records)
        # full provenance record + slim agent-facing view (companion files)
        json.dump(rep, open(os.path.join(OUT, f'{stem}.instances.json'), 'w'), indent=1)
        json.dump(make_brief(rep), open(os.path.join(OUT, f'{stem}.reminders.json'), 'w'), indent=1)
        # corpus tally
        by_sub = Counter()
        for st in rep['steps']:
            for r in st['observed_reminders']:
                by_sub[r['subtype']] += r['n_occurrences']
        total = sum(by_sub.values())
        corpus.append({'recipe': stem, 'activity_id': aid,
                       'n_recordings': rep['n_recordings'],
                       'n_error_recordings': rep['n_error_recordings'],
                       'n_reminder_occurrences': total,
                       'by_subtype': dict(by_sub.most_common()),
                       'n_unmapped': sum(u['n'] for u in rep['unmapped_step_ids'])})
        print(f'  {stem:32s} act={aid:2d}  recs={rep["n_recordings"]:3d} '
              f'err={rep["n_error_recordings"]:3d}  reminders={total:4d}'
              + (f'  UNMAPPED={corpus[-1]["n_unmapped"]}' if corpus[-1]['n_unmapped'] else ''))

    grand = Counter()
    for c in corpus:
        for k, v in c['by_subtype'].items():
            grand[k] += v
    summary = {'_mode': 'PROBING_FIREWALL_RELAXED', 'n_recipes': len(corpus),
               'corpus_by_subtype': dict(grand.most_common()), 'recipes': corpus}
    json.dump(summary, open(os.path.join(OUT, '_summary.json'), 'w'), indent=1)
    print(f'\nwrote {len(corpus)} recipes -> {OUT}/<recipe>.reminders.json  (+ _summary.json)')
    print('corpus reminder occurrences by subtype:',
          '  '.join(f'{k}={v}' for k, v in grand.most_common()))


if __name__ == '__main__':
    main()
