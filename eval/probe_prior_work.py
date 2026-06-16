#!/usr/bin/env python3
"""Probe prior-work proactive-reminder protocols by implementation.

Three experiments, no video/VLM needed (plus re-scoring of existing arm outputs):

  E1  LiveMamba protocol (IC-Acc + windowed mistake P/R/F1) re-implemented from
      the paper, with probe baselines run over the Qualcomm GT:
        silent / chatty / open-loop timer (train-split step-duration priors,
        zero perception) / oracle / jittered oracle.
      Question: what does the protocol reward, and where do the published
      numbers (Gemini IC-Acc 23.1 / F1 .02; LiveMamba 31.5 / .13) sit relative
      to a no-perception timer?

  E2  Score our existing replay_v1 arms (detector_replay, detector_plus_
      escalation, periodic_vlm_qwen; activity-8 recordings) under the same
      protocol.

  E3  PREGO-style online mistake detection with oracle step recognition and a
      task-graph-feasibility anticipator, replayed over ALL CC4D recordings.
      Question: with perfect perception, how much of CC4D's error mass does
      pure graph logic catch (order/missing) and miss (execution errors)?

Usage: python3 eval/probe_prior_work.py   (writes experiments/probe_prior_work/results.json)
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QIC = ROOT / "data/qualcomm_interactive_cooking/main"
CC4D = ROOT / "data/cc4d/annotations"
ARMS_DIR = ROOT / "experiments/replay_v1/results"
OUT_DIR = ROOT / "experiments/probe_prior_work"

HALF_WINDOW = 15.0  # paper: "30-second window" centered on GT time


# ---------------------------------------------------------------- GT loading
def load_qualcomm():
    """Per recording: completion times, mistake (t, class), recipe, split."""
    recs = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(QIC / f"{split}-00000-of-00001.parquet")
        for _, r in df.iterrows():
            ts, tys = list(r.output_timestamps), list(r.output_types)
            # completions: every instruction after the first marks the previous
            # step's completion; finish_all marks the last one.
            instr = [t for t, ty in zip(ts, tys) if ty == "instruction"]
            fins = [t for t, ty in zip(ts, tys) if ty == "feedback_finish_all"]
            completions = sorted(instr[1:] + fins)
            mistakes = [
                (t, ty.split("mistake_")[1].replace("_error", ""))
                for t, ty in zip(ts, tys)
                if "mistake" in ty
            ]
            recs[r.video_id] = dict(
                recipe=r.activity_name,
                split=split,
                completions=completions,
                mistakes=mistakes,
                end=max(ts),
            )
    return recs


# ---------------------------------------------------------------- matching
def match_events(pred, gt, half_window=HALF_WINDOW):
    """One-to-one greedy time matching. Returns (tp, fp, fn)."""
    gt_free = list(gt)
    tp = 0
    for p in sorted(pred):
        best, best_d = None, None
        for g in gt_free:
            d = abs(p - g)
            if d <= half_window and (best_d is None or d < best_d):
                best, best_d = g, d
        if best is not None:
            gt_free.remove(best)
            tp += 1
    return tp, len(pred) - tp, len(gt_free)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- E1 baselines
def timer_priors(recs):
    """Per recipe: mean absolute time of k-th completion over TRAIN recordings."""
    acc = defaultdict(lambda: defaultdict(list))
    for v in recs.values():
        if v["split"] != "train":
            continue
        for k, t in enumerate(v["completions"]):
            acc[v["recipe"]][k].append(t)
    pri = {}
    for recipe, ks in acc.items():
        n = int(np.median([len(v["completions"]) for v in recs.values()
                           if v["split"] == "train" and v["recipe"] == recipe]))
        pri[recipe] = [float(np.mean(ks[k])) for k in range(n) if k in ks]
    return pri


def e1_baselines(recs):
    pri = timer_priors(recs)
    test = {k: v for k, v in recs.items() if v["split"] == "test"}
    rows = {}
    rng = np.random.default_rng(0)

    def evaluate(name, pred_fn):
        ic_tp = ic_gt = 0
        m_tp = m_fp = m_fn = 0
        for vid, v in test.items():
            pc, pm = pred_fn(v)
            tp, _, fn = match_events(pc, v["completions"])
            ic_tp += tp
            ic_gt += len(v["completions"])
            t2, f2, n2 = match_events(pm, [t for t, _ in v["mistakes"]])
            m_tp, m_fp, m_fn = m_tp + t2, m_fp + f2, m_fn + n2
        p, r, f = prf(m_tp, m_fp, m_fn)
        rows[name] = dict(ic_acc=ic_tp / ic_gt, mistake_p=p, mistake_r=r, mistake_f1=f)

    evaluate("always_silent", lambda v: ([], []))
    evaluate("chatty_10s", lambda v: (list(np.arange(10, v["end"], 10)),
                                      list(np.arange(10, v["end"], 10))))
    evaluate("openloop_timer", lambda v: (pri.get(v["recipe"], []), []))
    evaluate("oracle", lambda v: (v["completions"], [t for t, _ in v["mistakes"]]))
    for sig in [5, 15, 30, 60]:
        evaluate(
            f"oracle_jitter_{sig}s",
            lambda v, s=sig: (
                [t + rng.uniform(-s, s) for t in v["completions"]],
                [t + rng.uniform(-s, s) for t, _ in v["mistakes"]],
            ),
        )
    return rows


# ---------------------------------------------------------------- E2 our arms
def e2_arms(recs):
    rows = {}
    for arm in ["detector_replay", "detector_plus_escalation", "periodic_vlm_qwen"]:
        ic_tp = ic_gt = 0
        m_tp = m_fp = m_fn = 0
        used = []
        for f in sorted((ARMS_DIR / arm).glob("*.json")):
            if ".debug" in f.name:
                continue
            vid = f.stem
            if vid not in recs:
                continue
            res = json.load(open(f))
            # completion claims = times we leave a non-"other" stage
            iv = [x for x in res.get("stage_intervals", []) if x["stage"] != "other"]
            pc = [x["end_s"] for x in iv]
            pm = [e["t"] for e in res.get("events", [])]
            v = recs[vid]
            tp, _, fn = match_events(pc, v["completions"])
            ic_tp += tp
            ic_gt += len(v["completions"])
            t2, f2, n2 = match_events(pm, [t for t, _ in v["mistakes"]])
            m_tp, m_fp, m_fn = m_tp + t2, m_fp + f2, m_fn + n2
            used.append(vid)
        if not used:
            continue
        p, r, f = prf(m_tp, m_fp, m_fn)
        rows[arm] = dict(ic_acc=ic_tp / ic_gt if ic_gt else 0.0, mistake_p=p,
                         mistake_r=r, mistake_f1=f, recordings=used)
    return rows


# ---------------------------------------------------------------- E3 PREGO-graph
def slug(recipe):
    return re.sub(r"[^a-z]", "", recipe.lower())


def load_graphs():
    graphs = {}
    for f in (CC4D / "task_graphs").glob("*.json"):
        g = json.load(open(f))
        parents = defaultdict(set)
        for a, b in g["edges"]:
            parents[b].add(a)
        desc2nodes = defaultdict(list)
        for nid, d in g["steps"].items():
            if d not in ("START", "END"):
                desc2nodes[d].append(int(nid))
        anc = {}

        def ancestors(n, seen=None):
            seen = set() if seen is None else seen
            for p in parents[n]:
                if p not in seen:
                    seen.add(p)
                    ancestors(p, seen)
            return seen

        for nid in g["steps"]:
            anc[int(nid)] = ancestors(int(nid)) - {0}
        graphs[f.stem] = dict(desc2nodes=desc2nodes, anc=anc, steps=g["steps"])
    return graphs


def e3_prego_graph(recs):
    ea = json.load(open(CC4D / "annotation_json/error_annotations.json"))
    graphs = load_graphs()
    cm = Counter()           # order-error confusion
    cm_all = Counter()       # any-error confusion
    caught_by_tag = Counter()
    total_by_tag = Counter()
    miss_tp = miss_fp = miss_fn = 0
    unmatched_desc = 0

    for rec in ea:
        vid = rec["recording_id"]
        if vid not in recs:
            continue
        g = graphs.get(slug(recs[vid]["recipe"]))
        if g is None:
            continue
        steps = sorted(
            [s for s in rec["step_annotations"] if s["start_time"] >= 0],
            key=lambda s: s["start_time"],
        )
        completed = set()
        executed_nodes = set()
        for s in steps:
            cands = g["desc2nodes"].get(s["description"], [])
            node = next((n for n in cands if n not in completed),
                        cands[0] if cands else None)
            if node is None:
                unmatched_desc += 1
                continue
            violation = bool(g["anc"][node] - completed - executed_nodes)
            tags = [e["tag"] for e in s.get("errors", [])]
            for t in tags:
                total_by_tag[t] += 1
                if violation:
                    caught_by_tag[t] += 1
            gt_order = "Order Error" in tags
            gt_any = bool(tags)
            cm[(violation, gt_order)] += 1
            cm_all[(violation, gt_any)] += 1
            completed.add(node)
        # missing-step detection at recording end: graph nodes never executed
        all_nodes = {int(n) for n, d in g["steps"].items()
                     if d not in ("START", "END")}
        pred_missing = all_nodes - completed
        gt_missing_descs = {
            s["description"]
            for s in rec["step_annotations"]
            if any(e["tag"] == "Missing Step" for e in s.get("errors", []))
            or s["start_time"] < 0
        }
        gt_missing = set()
        for d in gt_missing_descs:
            gt_missing.update(g["desc2nodes"].get(d, []))
        gt_missing -= completed  # partially-executed "missing" tags don't count
        miss_tp += len(pred_missing & gt_missing)
        miss_fp += len(pred_missing - gt_missing)
        miss_fn += len(gt_missing - pred_missing)

    def prf_cm(c):
        tp, fp, fn = c[(True, True)], c[(True, False)], c[(False, True)]
        return dict(zip(["p", "r", "f1"], prf(tp, fp, fn)), tp=tp, fp=fp, fn=fn)

    return dict(
        order_errors=prf_cm(cm),
        any_error=prf_cm(cm_all),
        missing_steps=dict(zip(["p", "r", "f1"], prf(miss_tp, miss_fp, miss_fn)),
                           tp=miss_tp, fp=miss_fp, fn=miss_fn),
        caught_by_tag={t: f"{caught_by_tag[t]}/{total_by_tag[t]}"
                       for t in total_by_tag},
        unmatched_descriptions=unmatched_desc,
    )


def main():
    recs = load_qualcomm()
    out = dict(
        e1_protocol_probes_test_split=e1_baselines(recs),
        e2_our_arms_activity8=e2_arms(recs),
        e3_prego_graph_all_recordings=e3_prego_graph(recs),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_DIR / "results.json", "w"), indent=1)

    print("=== E1: LiveMamba protocol, GT-replay baselines (test split, n=109) ===")
    print(f"{'baseline':22s} {'IC-Acc':>7s} {'M-P':>6s} {'M-R':>6s} {'M-F1':>6s}")
    print(f"{'(paper) Gemini-2.5-F':22s} {0.231:7.3f} {'':>6s} {'':>6s} {0.02:6.2f}")
    print(f"{'(paper) LiveMamba':22s} {0.315:7.3f} {'':>6s} {'':>6s} {0.13:6.2f}")
    for k, v in out["e1_protocol_probes_test_split"].items():
        print(f"{k:22s} {v['ic_acc']:7.3f} {v['mistake_p']:6.2f} "
              f"{v['mistake_r']:6.2f} {v['mistake_f1']:6.2f}")
    print("\n=== E2: our replay_v1 arms under the same protocol (activity 8) ===")
    for k, v in out["e2_our_arms_activity8"].items():
        print(f"{k:26s} IC-Acc {v['ic_acc']:.3f}  mistake P/R/F1 "
              f"{v['mistake_p']:.2f}/{v['mistake_r']:.2f}/{v['mistake_f1']:.2f} "
              f"({len(v['recordings'])} recs)")
    print("\n=== E3: PREGO-style graph-logic mistake detection, oracle perception "
          "(all recordings) ===")
    e3 = out["e3_prego_graph_all_recordings"]
    for k in ["order_errors", "any_error", "missing_steps"]:
        v = e3[k]
        print(f"{k:14s} P {v['p']:.2f}  R {v['r']:.2f}  F1 {v['f1']:.2f} "
              f"(tp {v['tp']} fp {v['fp']} fn {v['fn']})")
    print("violations caught per CC4D tag:", e3["caught_by_tag"])
    print("unmatched step descriptions:", e3["unmatched_descriptions"])


if __name__ == "__main__":
    main()
