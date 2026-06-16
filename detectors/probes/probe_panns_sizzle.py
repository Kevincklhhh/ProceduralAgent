#!/usr/bin/env python3
"""Probe: PANNs (CNN14, AudioSet) as a learned cook-sound detector, used for the
COARSE transition task (recipe prior known): when does sizzle END (step done) and
when is food ADDED (surge) — NOT fine sizzle-vs-boiling classification.

Pools AudioSet cooking classes {Frying(367), Sizzle(490), Boiling(456), Steam(296)}
into one "cook-active" probability via sliding-window clip-level tagging (robust;
the DecisionLevelMax framewise head under-scales). Compares the PANNs signal to the
GT cook window and to the cheap DSP detector.

Usage: python probe_panns_sizzle.py --recs 25_4,23_5,21_3
"""
import sys, os, json, argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import detectors_lib as dl

BASE = '/home/kailaic/NeuroTrace/ProceduralAgent'
ANN = json.load(open(os.path.join(BASE, 'data', 'cc4d', 'annotations',
                                  'annotation_json', 'complete_step_annotations.json')))
COOK_IDX = [367, 490, 456, 296]   # Frying, Sizzle, Boiling, Steam
COOK_STEPS = {
    '25': {279, 286, 292, 285, 290, 277, 278, 289, 281, 288},
    '23': {266, 258, 265, 261, 253, 274, 256, 264},
    '21': {223, 226, 228, 230, 231},
    '16': {160, 162, 165, 166, 168, 170, 171, 172, 175, 178, 179},
}


def log(*a):
    print(*a, flush=True)


def load_model():
    """Load CNN14 directly on a single GPU (bypass panns_inference's DataParallel)."""
    import torch
    from panns_inference.models import Cnn14
    ckpt = f'{os.path.expanduser("~")}/panns_data/Cnn14_mAP=0.431.pth'
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(dev).eval()
    return model, dev


def cook_signal(model, dev, rec, win=2.0, hop=1.0):
    import torch
    fs, x = wavfile.read(f'{BASE}/data/audio/{rec}_48k.wav')
    x = x.astype(np.float64) / 32768.0
    x32 = resample_poly(x, 2, 3).astype(np.float32)
    W, H = int(win * 32000), int(hop * 32000)
    starts = list(range(0, max(1, len(x32) - W), H))
    probs = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(starts), B):
            segs = np.stack([x32[s:s + W] for s in starts[i:i + B]])
            inp = torch.from_numpy(segs).to(dev)
            out = model(inp, None)['clipwise_output'].cpu().numpy()  # (b, 527)
            probs.append(out[:, COOK_IDX].max(axis=1))
    sig = np.concatenate(probs)
    t = np.array(starts[:len(sig)]) / 32000.0 + win / 2
    return t, sig


def score_rec(model, dev, rec, act, th=0.2):
    t, sig = cook_signal(model, dev, rec)
    steps = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}
    cw = [(s['start_time'], s['end_time']) for sid, s in steps.items()
          if sid in COOK_STEPS[act]]
    if not cw:
        return None
    cs, ce = min(a for a, b in cw), max(b for a, b in cw)
    inc, pre = (t >= cs) & (t < ce), t < cs
    on = sig > th
    # sizzle-end = end of last on-frame; offset error vs GT cook end
    i = len(on) - 1
    while i >= 0 and not on[i]:
        i -= 1
    end_err = abs(float(t[i]) - ce) if i >= 0 else None
    return dict(
        prep_false=float((pre & on).sum() / max(1, pre.sum())),
        cook_cov=float((inc & on).sum() / max(1, inc.sum())),
        cook_detected=bool((inc & on).sum() > 0 and inc.sum() > 0),
        end_err=round(end_err, 1) if end_err is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', default='25_4,23_5,21_3')
    ap.add_argument('--activity', help='score ALL recs of this activity, aggregate')
    args = ap.parse_args()

    if args.activity:
        model, dev = load_model()
        act = args.activity
        recs = sorted([k for k in ANN if k.startswith(act + '_')],
                      key=lambda x: int(x.split('_')[1]))
        per = {}
        for rec in recs:
            r = score_rec(model, dev, rec, act)
            if r:
                per[rec] = r
                log(f"  {rec}: prep_false={r['prep_false']:.3f} cook_cov={r['cook_cov']:.2f} "
                    f"detected={r['cook_detected']} end_err={r['end_err']}")
        n = len(per)
        agg = dict(recipe=act, recordings=n,
                   mean_prep_false=round(np.mean([v['prep_false'] for v in per.values()]), 3),
                   mean_cook_cov=round(np.mean([v['cook_cov'] for v in per.values()]), 3),
                   cook_detect_rate=round(np.mean([v['cook_detected'] for v in per.values()]), 3),
                   median_end_err=round(float(np.median([v['end_err'] for v in per.values()
                                                         if v['end_err'] is not None])), 1))
        log('\nAGG ' + json.dumps(agg))
        json.dump({'summary': agg, 'per': per},
                  open(f'{BASE}/detectors/probes/results_panns_act{act}.json', 'w'), indent=1)
        log(f'wrote results_panns_act{act}.json')
        return
    model, dev = load_model()
    log(f'model loaded on {dev}')

    out = {}
    for rec in args.recs.split(','):
        act = rec.split('_')[0]
        t, sig = cook_signal(model, dev, rec)
        steps = {s['step_id']: s for s in ANN[rec]['steps'] if s['start_time'] >= 0}
        cw = [(s['start_time'], s['end_time']) for sid, s in steps.items()
              if sid in COOK_STEPS[act]]
        cs, ce = min(a for a, b in cw), max(b for a, b in cw)
        inc = (t >= cs) & (t < ce)
        pre = t < cs
        # coarse transition: cook offset = last time a sustained (>=10s) prob>0.2 ends
        TH = 0.2
        on = sig > TH
        # find last run of 'on' and its end
        last_off = None
        i = len(on) - 1
        while i >= 0 and not on[i]:
            i -= 1
        if i >= 0:
            last_off = t[i]
        res = dict(
            prep_mean=round(float(sig[pre].mean()), 3) if pre.any() else None,
            cook_mean=round(float(sig[inc].mean()), 3),
            cook_p90=round(float(np.percentile(sig[inc], 90)), 3),
            cook_frac_gt02=round(float((inc & on).sum() / inc.sum()), 2),
            prep_frac_gt02=round(float((pre & on).sum() / max(1, pre.sum())), 2),
            cook_window=[round(cs, 1), round(ce, 1)],
            detected_cook_offset=round(float(last_off), 1) if last_off else None,
            gt_cook_end=round(ce, 1),
        )
        out[rec] = res
        log(f"\n{rec} (act {act}) cook=[{cs:.0f},{ce:.0f}]")
        log(f"  cook-prob  PREP mean={res['prep_mean']}  COOK mean={res['cook_mean']} "
            f"p90={res['cook_p90']}")
        log(f"  frac>0.2  COOK={res['cook_frac_gt02']}  PREP={res['prep_frac_gt02']}")
        log(f"  detected cook OFFSET={res['detected_cook_offset']}s  vs GT cook end={res['gt_cook_end']}s")
    json.dump(out, open(f'{BASE}/detectors/probes/results_panns_sizzle.json', 'w'), indent=1)
    log('\nwrote results_panns_sizzle.json')


if __name__ == '__main__':
    main()
