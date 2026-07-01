#!/usr/bin/env python3
"""Proactive-reminder GT (event-detection scheme). Supersedes gt_build_family_a.py.

Per recording, emits a flat list of proactive reminders, each = {t, content, subtype}.
`t` is a SINGLE timestamp (the Qualcomm mistake-visibility moment, or a mechanical CC4D
anchor in the fallback case). There is NO `window` / tolerance here by design — matching
tolerance belongs in the evaluation script, not the answer key.
Silence is implicit: any predicted reminder not matching a GT reminder is a false alarm;
there are NO `silent` labels and NO decision points. This is the event-detection framing
of docs/REMINDER_EVALUATION.md, nothing more.

SCOPE (this version): execution mistakes only — technique, preparation, measurement,
timing, temperature. Order and missing-step reminders are deliberately SKIPPED for now
(they have no Qualcomm timestamp and need DAG derivation; revisit later).

Sources:
  qualcomm  - CC4D tag fixes WHICH step erred + the type; Qualcomm gives the reminder
              TIME (sub-step visibility timestamp) and the reminder TEXT (output_texts).
              This is the useful thing Qualcomm adds. t = the Qualcomm timestamp.
  cc4d_only - timing/temperature tags with no matching Qualcomm event: time falls back to
              a mechanical anchor (timing -> step.end; temperature -> step.start, since a
              wrong power level is in effect from step start), content = CC4D description.

Firewall: this is the GT/answer-key pipeline; never feed it to the recipe->sensor predictor.

Outputs: data/cc4d_proactive/{recording_id}.json + data/cc4d_proactive/_summary.json
Usage:   python eval/gt_build_proactive.py [--only 8_50] [--no-write]
"""
import json, argparse, os
from collections import defaultdict, Counter

ANN = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc4d', 'annotations')
QC  = os.path.join(os.path.dirname(__file__), '..', 'data', 'qualcomm_interactive_cooking', 'main')
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc4d_proactive')

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
    import pandas as pd
    q = pd.concat([pd.read_parquet(os.path.join(QC, f'{s}-00000-of-00001.parquet'))
                   for s in ['train', 'validation', 'test']], ignore_index=True)
    qmap = {r['video_id']: r for _, r in q.iterrows()}
    return ann, err, qmap


def build(vid, ann, err, qmap):
    rec = ann[vid]
    steps = rec['steps']
    seg = {s['step_id']: s for s in steps}
    tags = defaultdict(list)
    for sa in err.get(vid, {}).get('step_annotations', []):
        for e in sa.get('errors', []):
            tags[sa['step_id']].append(e)

    reminders, rid = [], 0
    def add(**kw):
        nonlocal rid; rid += 1
        reminders.append({'id': f'{vid}_r{rid}', **kw})

    # ---- Qualcomm-timed execution mistakes (time + text from Qualcomm, tag from CC4D) ----
    row = qmap.get(vid)
    matched = set()                                       # (step_id, tag) already covered
    if row is not None:
        for t, typ, text in zip(row['output_timestamps'], row['output_types'], row['output_texts']):
            if 'mistake' not in typ:
                continue
            cat = typ.split('mistake_')[1].split('_error')[0]
            want = TAGMAP.get(cat)
            if want is None:                              # e.g. order/missing -> skipped here
                continue
            cont = [s for s in steps if s['start_time'] >= 0 and s['start_time']-10 <= t <= s['end_time']+10]
            st = next((s for s in cont if want in [e['tag'] for e in tags.get(s['step_id'], [])]), None)
            if not st:
                continue
            matched.add((st['step_id'], want))
            add(t=round(float(t), 1), subtype=cat, content=str(text).strip(),
                anchor_step=st['step_id'], source='qualcomm')

    # ---- CC4D-only fallback for timing/temperature tags with no Qualcomm event ----
    for sid, elist in tags.items():
        s = seg.get(sid)
        if not s or s['start_time'] < 0:
            continue
        for e in elist:
            if e['tag'] == 'Timing Error' and (sid, 'Timing Error') not in matched:
                add(t=round(s['end_time'], 1), subtype='timing', content=e['description'].strip(),
                    anchor_step=sid, source='cc4d_only', flag='low_confidence_timing')
            elif e['tag'] == 'Temperature Error' and (sid, 'Temperature Error') not in matched:
                add(t=round(s['start_time'], 1), subtype='temperature', content=e['description'].strip(),
                    anchor_step=sid, source='cc4d_only', flag='low_confidence_temperature')

    reminders.sort(key=lambda r: r['t'])
    last_end = max([s['end_time'] for s in steps if s['end_time'] >= 0], default=0.0)
    return {'recording_id': vid, 'activity_name': rec['activity_name'],
            'recipe': ACT2FILE.get(rec['activity_name']), 'duration_s': round(last_end, 1),
            'is_error': err.get(vid, {}).get('is_error', False), 'reminders': reminders}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='single recording id, e.g. 8_50')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    ann, err, qmap = load()
    vids = [args.only] if args.only else list(ann)
    if not args.no_write:
        os.makedirs(OUT, exist_ok=True)
    by_subtype = Counter(); by_source = Counter(); n_clean = 0
    for vid in vids:
        if ACT2FILE.get(ann[vid]['activity_name']) is None:
            continue
        out = build(vid, ann, err, qmap)
        for r in out['reminders']:
            by_subtype[r['subtype']] += 1; by_source[r['source']] += 1
        if not out['reminders']:
            n_clean += 1
        if not args.no_write:
            json.dump(out, open(os.path.join(OUT, f'{vid}.json'), 'w'), indent=1)
        if args.only:
            print(json.dumps(out, indent=1))
    if not args.only:
        summary = {'recordings': len(vids), 'recordings_no_reminder': n_clean,
                   'reminders_by_subtype': dict(by_subtype.most_common()),
                   'reminders_by_source': dict(by_source.most_common()),
                   'note': 'order + missing_step intentionally excluded in this version'}
        print(json.dumps(summary, indent=1))
        if not args.no_write:
            json.dump(summary, open(os.path.join(OUT, '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
