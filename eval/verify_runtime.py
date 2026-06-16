"""verify_runtime -- check the proposed-system runtime against the engine.py oracle
and the GT, for spiced hot chocolate.

Two views:
  (1) reproduce-the-oracle: per-second agreement between the runtime's stage track
      and engine.py's, on the audio-anchored stages (the runtime must match the
      hardcoded state machine where both rely on the same audio anchors).
  (2) GT block-coarse accuracy: the runtime's `quiet_middle` is the C-none region;
      it is scored correct when the GT coarse label is `adds` OR `mix` (the block
      brackets that region without resolving which sub-step, which needs the VLM).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score

RECS = ['8_16', '8_3', '8_25', '8_26', '8_31', '8_50']
MINE_DIR = os.path.join(BASE, 'experiments/proposed_system/results')
ENGINE_DIR = os.path.join(BASE, 'experiments/replay_v1/results/detector_replay')
QUIET = 'quiet_middle'


def my_correct(g_coarse, stage):
    if stage == QUIET:
        return g_coarse in ('adds', 'mix')
    return score.coarse(stage) == g_coarse


def per_second(gt_rec, res, correct_fn):
    segs = score.gt_segments(gt_rec)
    T = score.horizon(gt_rec)
    n = ok = nx = okx = 0
    for i in range(T):
        t = i + 0.5
        g = score.coarse(score.gt_label(segs, t))
        p = score.pred_label(res['stage_intervals'], t)
        c = correct_fn(g, p)
        n += 1; ok += c
        if g != 'other':
            nx += 1; okx += c
    return ok / n, (okx / nx if nx else None)


def track_agreement(mine, eng):
    """Per-second agreement of the two stage tracks (quiet_middle == engine adds|mix)."""
    T = int(max(max(iv['end_s'] for iv in mine['stage_intervals']),
                max(iv['end_s'] for iv in eng['stage_intervals'])))
    n = ok = 0
    for i in range(T):
        t = i + 0.5
        p = score.pred_label(mine['stage_intervals'], t)
        e = score.coarse(score.pred_label(eng['stage_intervals'], t))
        pc = 'adds|mix' if p == QUIET else score.coarse(p)
        ec = 'adds|mix' if e in ('adds', 'mix') else e
        n += 1; ok += (pc == ec)
    return ok / n


def main():
    gt = json.load(open(os.path.join(BASE, 'data/gt_activity8.json')))
    print(f"{'rec':<6}{'track~engine':>13}{'mine GT-acc':>13}{'eng GT-acc':>12}"
          f"{'mw end Δ':>10}{'heat start Δ':>13}")
    mt = me = mg = eg = 0.0
    for rec in RECS:
        mine = json.load(open(os.path.join(MINE_DIR, f'{rec}.json')))
        eng = json.load(open(os.path.join(ENGINE_DIR, f'{rec}.json')))
        agree = track_agreement(mine, eng)
        macc, _ = per_second(gt[rec], mine, my_correct)
        eacc, _ = per_second(gt[rec], eng, lambda g, p: score.coarse(p) == g)
        bd = score.boundary_deltas(gt[rec], mine)
        mw = bd['microwave_initial']['end_delta_s'] if bd['microwave_initial'] else None
        ht = bd['heat_serve']['start_delta_s'] if bd['heat_serve'] else None
        mt += agree; mg += macc; eg += eacc
        print(f"{rec:<6}{agree:>12.1%}{macc:>13.1%}{eacc:>12.1%}"
              f"{str(mw):>10}{str(ht):>13}")
    n = len(RECS)
    print(f"{'mean':<6}{mt/n:>12.1%}{mg/n:>13.1%}{eg/n:>12.1%}")
    print("\ntrack~engine = per-second agreement of runtime vs engine.py stage track "
          "(quiet_middle == engine adds|mix).")
    print("mine GT-acc  = runtime coarse stage accuracy vs GT (quiet_middle correct on "
          "GT adds|mix).  eng GT-acc = engine coarse accuracy vs GT.")
    print("Δ columns = runtime predicted boundary minus GT step boundary (s).")


if __name__ == '__main__':
    main()
