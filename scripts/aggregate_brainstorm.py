#!/usr/bin/env python3
"""Aggregate the per-recipe *.brainstorm.json artifacts into corpus-level
sensor-control statistics.

Key distinction this script makes (the one that matters for the thesis):
  - CLAIM-level role mix  = what each claim could use in isolation.
  - STEP-level role mix    = the role that actually gets SCHEDULED, taking the
    MAX (most expensive) role over every claim live during that step. At runtime
    a step's recognition + completion + all checks are active together, so the
    sensor config is pinned to the worst claim.
  - WINDOW-level role mix  = same max, but over the set of concurrently-eligible
    steps (DAG fork siblings share one decision window).

Reads:  tasks/cc4d_probe/<recipe>.brainstorm.json  (+ tasks/cc4d/<recipe>.json for the DAG)
Writes: tasks/cc4d_probe/_corpus_brainstorm_aggregate.json
Firewall-neutral: consumes only criteria-derived brainstorm files + the recipe DAG.
"""
import json, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "tasks", "cc4d_probe")
DAGDIR = os.path.join(ROOT, "tasks", "cc4d")
LOOP = {"sautedmushrooms", "dressedupmeatballs", "pinwheels"}  # repeated step_ids; runtime can't handle yet

RANK = {"A-solve": 0, "B-trigger": 1, "C-none": 2}
INV = {0: "A-solve", 1: "B-trigger", 2: "C-none"}
ROLES = ["A-solve", "B-trigger", "C-none"]


def worst(roles):
    return INV[max((RANK.get(r, 0) for r in roles), default=0)]


def subtype_of(c):
    k = (c.get("kind") or "")
    if k.startswith("check"):
        return k.split(":")[1] if ":" in k else (c.get("subtype") or "check")
    return k  # recognition / completion


def source_map(name):
    """(step_id, subtype) -> 'base' | 'probe', read from the authoritative criteria.json.
    recognition/completion are always 'base'. Join is unambiguous: no (step,subtype)
    bucket mixes base and probe checks across the corpus."""
    cd = json.load(open(os.path.join(PROBE, name + ".generated.criteria.json")))
    m = {}
    for n in cd.get("nodes", []):
        sid = n.get("step_id")
        for ck in n.get("checks", []):
            st = ck.get("reminder", "?")
            m[(sid, st)] = "probe" if "[probe-added]" in ck.get("claim", "") else "base"
    return m


def main():
    files = sorted(f for f in glob.glob(os.path.join(PROBE, "*.brainstorm.json"))
                   if os.path.basename(f).split(".")[0] not in LOOP)

    claim_role = collections.Counter()
    modality = collections.Counter()
    role_x_mod = collections.Counter()
    kind_x_role = collections.Counter()        # (subtype, role) -> n
    spec = collections.Counter()
    audio_use = collections.Counter()
    step_bind = collections.Counter()
    win_bind = collections.Counter()
    culprit_C = collections.Counter()          # what subtype pins a step to C-none
    per_recipe = []
    n_steps = n_win = 0

    for f in files:
        name = os.path.basename(f).split(".")[0]
        d = json.load(open(f))
        claims = d.get("claims", [])
        bystep = collections.defaultdict(list)
        for c in claims:
            r = (c.get("role") or "").strip()
            m = (c.get("modality") or "").lower().strip()
            st = subtype_of(c)
            claim_role[r] += 1
            modality[m] += 1
            role_x_mod[(r, m)] += 1
            kind_x_role[(st, r)] += 1
            s = (c.get("speculative_sensor") or "").lower().strip()
            # bucket the free-text speculative tags into canonical sensors
            bucket = "(none)"
            for key in ("thermal", "monochrome", "low-res", "imu"):
                if key in s:
                    bucket = {"thermal": "thermal-spot", "monochrome": "monochrome",
                              "low-res": "low-res-roi", "imu": "imu"}[key]
                    break
            else:
                if s and s not in ("none", "null", "-"):
                    bucket = "cv-other"
            spec[bucket] += 1
            bystep[c.get("step_id")].append((st, r))

        step_worst = {}
        rc = collections.Counter()
        aud = set()
        for sid, lst in bystep.items():
            n_steps += 1
            b = worst([r for _, r in lst])
            step_bind[b] += 1
            step_worst[sid] = b
            for st, r in lst:
                rc[r] += 1
                if b == "C-none" and r == "C-none":
                    culprit_C[st] += 1
        for c in claims:
            blob = (c.get("feasibility") or "") + " " + (c.get("config_oneline") or "")
            for D in ("D1", "D2", "D3", "D4", "D5", "D6"):
                if D in blob:
                    aud.add(D)
        for D in aud:
            audio_use[D] += 1

        # concurrent windows via DAG layered simulation
        windows = 0
        try:
            dag = json.load(open(os.path.join(DAGDIR, name + ".json")))
            nodes = dag.get("nodes") or dag.get("steps") or []
            pre = {nd.get("step_id", nd.get("id")): set(nd.get("preconditions", []) or []) for nd in nodes}
            done, remaining = set(), set(pre)
            while remaining:
                elig = [s for s in remaining if pre[s] <= done] or list(remaining)
                n_win += 1
                windows += 1
                win_bind[worst([step_worst.get(s, "A-solve") for s in elig])] += 1
                for s in elig:
                    remaining.discard(s)
                    done.add(s)
        except Exception:
            pass

        per_recipe.append({
            "recipe": name, "n_claims": len(claims), "n_steps": len(bystep),
            "n_windows": windows,
            "claim_role": {r: rc.get(r, 0) for r in ROLES},
            "audio_detectors": sorted(aud),
        })

    tot = sum(claim_role.values())

    def sens(drop):
        sb = collections.Counter()
        n = 0
        for f in files:
            d = json.load(open(f))
            bystep = collections.defaultdict(list)
            for c in d.get("claims", []):
                if subtype_of(c) in drop:
                    continue
                bystep[c.get("step_id")].append((c.get("role") or "").strip())
            for sid, rs in bystep.items():
                n += 1
                sb[worst(rs)] += 1
        return {"n": n, **{r: sb.get(r, 0) for r in ROLES}}

    # ---- base vs probe-added source split (firewall-clean vs error-space-derived) ----
    src_claim = {"base": collections.Counter(), "probe": collections.Counter()}
    src_step_all = collections.Counter()
    src_step_base = collections.Counter()
    n_src_steps = 0
    cnone_only_probe = 0
    for f in files:
        name = os.path.basename(f).split(".")[0]
        sm = source_map(name)
        d = json.load(open(f))
        bystep = collections.defaultdict(list)
        for c in d.get("claims", []):
            st = subtype_of(c)
            src = "base" if st in ("recognition", "completion") else sm.get((c.get("step_id"), st), "base")
            r = (c.get("role") or "").strip()
            src_claim[src][r] += 1
            bystep[c.get("step_id")].append((src, r))
        for sid, lst in bystep.items():
            n_src_steps += 1
            wa = worst([r for _, r in lst])
            wb = worst([r for s, r in lst if s == "base"])
            src_step_all[wa] += 1
            src_step_base[wb] += 1
            if wa == "C-none" and RANK[wb] < 2:
                cnone_only_probe += 1

    out = {
        "_kind": "corpus_sensor_control_brainstorm_aggregate",
        "_note": "Firewall-neutral feasibility hypotheses, not measured detector performance. "
                 "STEP/WINDOW mixes are the deployable numbers (max role over concurrent claims).",
        "n_recipes": len(files), "n_claims": tot, "n_steps": n_steps, "n_windows": n_win,
        "claim_level_role": {r: claim_role[r] for r in ROLES},
        "step_level_role": {r: step_bind[r] for r in ROLES},
        "window_level_role": {r: win_bind[r] for r in ROLES},
        "modality": dict(modality.most_common()),
        "role_x_modality": {f"{r}|{m}": n for (r, m), n in role_x_mod.most_common()},
        "subtype_x_role": {f"{k}|{r}": n for (k, r), n in sorted(kind_x_role.items())},
        "step_cnone_culprit_subtype": dict(culprit_C.most_common()),
        "speculative_sensor": dict(spec.most_common()),
        "audio_detector_recipes": {D: audio_use.get(D, 0) for D in ("D1", "D2", "D3", "D4", "D5", "D6")},
        "step_role_sensitivity": {
            "as_is": sens(set()),
            "drop_technique": sens({"technique"}),
            "drop_technique_temperature": sens({"technique", "temperature"}),
        },
        "source_split": {
            "_note": "base = rule-generated from recipe step text/DAG (firewall-clean); "
                     "probe = [probe-added] from the observed-error space (not firewall-clean).",
            "claim_level": {
                "base": {r: src_claim["base"][r] for r in ROLES},
                "probe": {r: src_claim["probe"][r] for r in ROLES},
            },
            "step_level_max_role": {
                "all_claims": {r: src_step_all[r] for r in ROLES},
                "base_only": {r: src_step_base[r] for r in ROLES},
            },
            "steps_pushed_to_cnone_only_by_probe": cnone_only_probe,
        },
        "per_recipe": per_recipe,
    }
    dst = os.path.join(PROBE, "_corpus_brainstorm_aggregate.json")
    json.dump(out, open(dst, "w"), indent=2)
    print("wrote", dst)
    print(f"claims={tot} steps={n_steps} windows={n_win} recipes={len(files)}")
    print("claim-level :", out["claim_level_role"])
    print("step-level  :", out["step_level_role"])
    print("window-level:", out["window_level_role"])


if __name__ == "__main__":
    main()
