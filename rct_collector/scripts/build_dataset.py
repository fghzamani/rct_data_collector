#!/usr/bin/env python3
"""Build the clean, derived dataset of a campaign from its append-only raw files.

    python3 -m rct_collector.scripts.build_dataset --campaign-dir <campaign>

Reads   raw/rct_results.csv (one row per attempt), supervisor/quarantine.jsonl,
        MANIFEST.json, the pose pool.
Writes  dataset/attempts.csv      every attempt + accepted / quarantined flags
        dataset/probes.csv        ONE row per probe id (the accepted attempt),
                                  pose-pool context joined in, relative paths
        dataset/probes_valid.csv  probes.csv where probe_valid == 1
        dataset/dataset_card.json counts, exclusions, event cells, and the
                                  rerun-selection check
        dataset/README.md         how to read the files

Acceptance rule, per probe id: the LAST attempt that is neither quarantined by
the supervisor nor an EXCEPTION row. If every attempt was rejected, the last one
is kept with probe_valid=0 and invalid_reasons += supervisor_quarantine, so the
probe stays visible instead of silently vanishing.

Pure standard library: runs on the host as well as in the container.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

from rct_collector.scripts.campaign import CampaignPaths, read_jsonl

csv.field_size_limit(sys.maxsize)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    return str(v).strip() in ("1", "1.0", "True", "true")


def _run_time(run_id: str):
    try:
        return datetime.strptime(str(run_id), "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _rel(path: str, root: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def build(campaign_dir: str, quiet: bool = False) -> dict:
    P = CampaignPaths(campaign_dir)
    if not os.path.exists(P.results_csv):
        raise SystemExit(f"no {P.results_csv}")
    with open(P.results_csv, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(line.replace("\x00", "") for line in f))

    man = json.load(open(P.manifest)) if os.path.exists(P.manifest) else {"epochs": []}
    epochs = [(datetime.fromisoformat(e["started_at"]), e["epoch"]) for e in man.get("epochs", [])]
    quarantine = {}
    for q in read_jsonl(P.quarantine):
        quarantine[(str(q.get("run_id")), int(q.get("trial_id", -1)))] = q.get("reason", "")

    # Per-probe data audit by the supervisor (footprint coverage by exact arm
    # geometry, MPPI footprint-check flag, blind contacts): audit_* columns.
    audits = {}
    for a in read_jsonl(os.path.join(P.sup, "audit_probes.jsonl")):
        audits[(str(a.get("run_id")), int(a.get("trial_id", -1)))] = a

    pool = []
    import glob as _glob
    frozen = (man.get("epochs") or [{}])[-1].get("frozen", "")
    cands = sorted(_glob.glob(os.path.join(frozen, "pose_pool__*"))) if frozen else []
    if cands:
        with open(cands[0]) as f:
            pool = json.load(f)
            pool = pool.get("poses", pool) if isinstance(pool, dict) else pool
    elif not quiet:
        print("WARNING: no frozen pose pool; pool_* context columns omitted")

    # Footprint polygon version per epoch, from the frozen param_space.py, for
    # rows written before the collector recorded `fp_version` itself.
    import hashlib
    import re as _re
    fpv = {}
    for e in man.get("epochs", []):
        ps = os.path.join(e.get("frozen", ""), "code", "rct_collector", "scripts", "param_space.py")
        if os.path.exists(ps):
            fpv[e["epoch"]] = {
                lbl: hashlib.sha256(poly.encode()).hexdigest()[:8]
                for lbl, poly in _re.findall(r'"(tucked|carry)":\s*\{\s*"footprint":\s*"([^"]+)"',
                                             open(ps).read())}

    # ── attempts ────────────────────────────────────────────────────────────
    for i, r in enumerate(rows):
        r["_order"] = i
        rt = _run_time(r.get("run_id", ""))
        ep = 0
        for started, e in epochs:
            if rt is not None and rt >= started.replace(microsecond=0):
                ep = e
        r["code_epoch"] = ep
        if not r.get("fp_version"):
            r["fp_version"] = fpv.get(ep, {}).get(r.get("arm_requested_label") or "tucked", "")
        key =(str(r.get("run_id")), int(float(r.get("trial_id") or -1)))
        for k2, v2 in audits.get(key, {}).items():
            if k2.startswith("audit_"):
                r[k2] = v2
        r["quarantined"] = int(key in quarantine)
        r["quarantine_reason"] = quarantine.get(key, "")
        r["accepted"] = 0

    by_trial = defaultdict(list)
    replicates = defaultdict(list)
    for r in rows:
        tid = int(float(r.get("trial_id") or -1))
        if tid <= 0:
            continue
        if (r.get("attempt_kind") or "main") == "replicate":
            replicates[tid].append(r)      # noise measurement, never accepted
        else:
            by_trial[tid].append(r)
    for tid, att in by_trial.items():
        att.sort(key=lambda r: (r.get("timestamp", ""), r["_order"]))
        for k, r in enumerate(att, 1):
            r["attempt_no"] = k
            r["n_attempts"] = len(att)

    probes = []
    for tid in sorted(by_trial):
        att = by_trial[tid]
        ok = [r for r in att if not r["quarantined"] and r.get("status") != "EXCEPTION"]
        acc = ok[-1] if ok else att[-1]
        acc["accepted"] = 1
        p = {k: v for k, v in acc.items() if not k.startswith("_")}
        if not ok:
            p["probe_valid"] = 0
            p["invalid_reasons"] = ";".join(
                x for x in [p.get("invalid_reasons", ""), "supervisor_quarantine"] if x)
        p["n_quarantined_attempts"] = sum(r["quarantined"] for r in att)
        # Portable paths.
        p["json_relpath"] = _rel(p.get("json_path", ""), P.root)
        p["telemetry_relpath"] = _rel(p.get("tel_path", ""), P.root)
        # Pose-pool context: the pre-treatment situation each probe was built for.
        idx = p.get("pose_pool_index", "")
        try:
            e = pool[int(float(idx))]
        except (ValueError, IndexError, TypeError):
            e = None
        if e:
            d = e.get("difficulty", {}) or {}
            p.update({
                "pool_share": e.get("share", ""), "pool_stratum": e.get("stratum", ""),
                "pool_start_region": "_".join(map(str, e.get("start_region", []) or [])),
                "pool_crossed": int(bool(e.get("crossed"))),
                "pool_distance_m": e.get("distance", ""),
                "pool_difficulty": d.get("class", ""),
                "pool_tags": ";".join(d.get("tags", [])),
                "pool_min_gap_m": d.get("min_gap_m", ""),
                "pool_corridor_m": d.get("corridor_m", ""),
                "pool_initial_turn_deg": d.get("initial_turn_deg", ""),
                "pool_dynamic_sites": ";".join(map(str, d.get("dynamic_sites", []) or [])),
                "pool_dist_to_dynamic_m": d.get("dist_to_dynamic_m", ""),
            })
            for t in ("corner", "doorway", "narrow", "detour", "corridor", "dense", "dynamic"):
                p[f"pool_tag_{t}"] = int(t in d.get("tags", []))
        probes.append(p)

    os.makedirs(P.dataset, exist_ok=True)

    def _write(path, recs):
        keys = []
        seen = set()
        for r in recs:
            for k in r:
                if k not in seen and not k.startswith("_"):
                    seen.add(k)
                    keys.append(k)
        lead = [k for k in ("trial_id", "run_id", "attempt_kind", "attempt_no", "accepted",
                            "quarantined", "probe_valid", "invalid_reasons", "status",
                            "collision", "probe_stalled", "arm_requested_label",
                            "pool_share") if k in seen]
        keys = lead + sorted(k for k in keys if k not in lead)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        os.replace(tmp, path)

    _write(os.path.join(P.dataset, "attempts.csv"), rows)
    _write(os.path.join(P.dataset, "probes.csv"), probes)
    valid = [p for p in probes if _truthy(p.get("probe_valid"))]
    _write(os.path.join(P.dataset, "probes_valid.csv"), valid)

    rep_rows, rep_summary = _replicates(replicates, {int(float(p["trial_id"])): p for p in probes})
    if rep_rows:
        _write(os.path.join(P.dataset, "replicates.csv"), rep_rows)
    card = _card(rows, probes, valid, quarantine, man)
    card["replicates"] = rep_summary
    with open(os.path.join(P.dataset, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    with open(os.path.join(P.dataset, "README.md"), "w") as f:
        f.write(README)
    if not quiet:
        print(json.dumps({k: card[k] for k in ("n_attempts", "n_probes", "n_valid",
                                               "events_valid", "carry_share_valid")}, indent=2))
    return card


def _replicates(replicates, accepted):
    """Test-retest of the SAME probe (same pose, config, apply time). Gazebo
    and Nav2 are not deterministic; this measures how much of the outcome is
    irreducible simulation noise."""
    out, per = [], []
    for tid, reps in sorted(replicates.items()):
        a = accepted.get(tid)
        group = ([a] if a else []) + [r for r in reps if not r.get("quarantined")]
        group = [g for g in group if g.get("status") != "EXCEPTION"]
        if len(group) < 2:
            continue
        cols = [int(_truthy(g.get("collision"))) for g in group]
        stl = [int(_truthy(g.get("probe_stalled"))) for g in group]
        per.append((tid, len(group), sum(cols), sum(stl)))
        for g in group:
            out.append({"trial_id": tid, "run_id": g.get("run_id"),
                        "attempt_kind": g.get("attempt_kind") or "main",
                        "collision": g.get("collision"), "probe_stalled": g.get("probe_stalled"),
                        "status": g.get("status"), "probe_valid": g.get("probe_valid"),
                        "arm_requested_label": g.get("arm_requested_label"),
                        "collision_time_sec": g.get("collision_time_sec"),
                        "probe_progress_m": g.get("probe_progress_m")})
    if not per:
        return out, None
    agree_c = sum(1 for _, n, c, _s in per if c in (0, n)) / len(per)
    agree_s = sum(1 for _, n, _c, s_ in per if s_ in (0, n)) / len(per)
    mixed = [t for t, n, c, _ in per if 0 < c < n]
    return out, {
        "n_probes": len(per), "runs_per_probe": sorted({n for _, n, _, _ in per}),
        "collision_all_agree": round(agree_c, 3), "stall_all_agree": round(agree_s, 3),
        "probes_with_mixed_collision": mixed,
        "note": "share of replicated probes whose runs all agree; 1 - that is the "
                "irreducible simulation noise in the outcome"}


def _card(rows, probes, valid, quarantine, man) -> dict:
    coll = lambda r: _truthy(r.get("collision"))
    stall = lambda r: _truthy(r.get("probe_stalled"))
    arm = lambda r: r.get("arm_requested_label", "") or "?"

    cells = defaultdict(lambda: {"n": 0, "collisions": 0, "stalls": 0})
    fp_cells = defaultdict(lambda: {"n": 0, "collisions": 0, "stalls": 0})
    for r in valid:
        for c in (cells[f"{arm(r)}|{r.get('pool_share', '?')}"],
                  fp_cells[f"{arm(r)}@{r.get('fp_version') or '?'}"]):
            c["n"] += 1
            c["collisions"] += coll(r)
            c["stalls"] += stall(r)

    # Rerun-selection check. If quarantined attempts collided (or stalled) far
    # more often than accepted ones, the glitch rule is correlated with the
    # outcome and re-running them biases the outcome rates downward.
    q = [r for r in rows if r["quarantined"]]
    acc = [r for r in rows if r["accepted"]]
    rate = lambda rs, fn: round(sum(fn(r) for r in rs) / len(rs), 4) if rs else None
    sel = {"n_quarantined_attempts": len(q),
           "collision_rate_quarantined": rate(q, coll),
           "collision_rate_accepted": rate(acc, coll),
           "stall_rate_quarantined": rate(q, stall),
           "stall_rate_accepted": rate(acc, stall),
           "by_reason": dict(Counter(r["quarantine_reason"].split(":")[0] for r in q))}
    cq, ca = sel["collision_rate_quarantined"], sel["collision_rate_accepted"]
    sel["warning"] = (
        "quarantined attempts collide >2x as often as accepted ones: the glitch "
        "rule may be outcome-dependent" if (cq is not None and ca and len(q) >= 10
                                           and cq > 2 * ca) else "")

    rtfs = sorted(v for v in (_f(r.get("tel_rtf_min_1s_precontact")) for r in valid)
                  if v is not None)
    return {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "epochs": [e["epoch"] for e in man.get("epochs", [])],
        "pilot": man.get("pilot"),
        "n_attempts": len(rows),
        "n_probes": len(probes),
        "n_valid": len(valid),
        "attempts_by_kind": dict(Counter(r.get("attempt_kind", "main") or "main" for r in rows)),
        "probes_by_status": dict(Counter(p.get("status", "") for p in probes)),
        "invalid_reasons": dict(Counter(x for p in probes
                                        for x in (p.get("invalid_reasons") or "").split(";")
                                        if x)),
        "events_valid": {"collisions": sum(coll(r) for r in valid),
                         "stalls": sum(stall(r) for r in valid)},
        "carry_share_valid": rate(valid, lambda r: arm(r) == "carry"),
        "cells_arm_x_share_valid": dict(sorted(cells.items())),
        # The treatment level is (arm label, footprint polygon version): one
        # label with two polygons is two different treatments.
        "cells_arm_at_fp_version_valid": dict(sorted(fp_cells.items())),
        "rows_per_epoch": dict(Counter(str(r["code_epoch"]) for r in probes)),
        "telemetry_coverage_valid": rate(valid, lambda r: (_f(r.get("tel_n")) or 0) > 0),
        "rtf_precontact_min_p05_valid": (rtfs[int(0.05 * (len(rtfs) - 1))] if rtfs else None),
        "rerun_selection_check": sel,
    }


README = """# Campaign dataset (derived — rebuild with build_dataset, never edit)

| file | one row per | use |
|---|---|---|
| probes_valid.csv | probe that passed every validity check | **analysis** |
| probes.csv | probe id (the accepted attempt) | exclusions audit |
| attempts.csv | every attempt the collector made | supervisor audit |
| dataset_card.json | — | counts, event cells, rerun-selection check |

Key columns
- `collision` / `y_h`: contact inside the watch window; `probe_stalled`: < 0.10 m progress.
- `arm_requested_label`: carry / tucked (the footprint treatment); the other
  treatment knobs are the `param__*` columns; `param_actual__*` are read-backs.
- `fp_local_end`, `fp_global_end`, `fp_sync_end`: footprint each costmap held at
  the end of the window; `arm_delivered`, `arm_openness_max`: what the arm did.
- `pool_*`: the pre-treatment situation the start/goal pair was drawn for
  (share = spread / tight / corridor, tags, route min gap, dynamic sites).
- `tel_*`: telemetry health (rate, gaps, real-time factor before first contact,
  missing streams) and summaries: links in the first contact (`tel_first_contact_arm`
  = the arm/gripper struck), footprint areas the costmaps PUBLISHED at window end
  (`tel_fp_published_sync_end`), plans, recoveries.
- `telemetry_relpath`: 20 Hz time series, relative to the campaign dir. Load with
  `rct_collector.telemetry.load_probe_telemetry(path)` -> (columns, meta, events).
- `code_epoch`: which frozen input set produced the row (see MANIFEST.json).
- `audit_*` (supervisor data audit, per probe): `audit_fp_outside_cm` = how far the
  arm's exact collision geometry got outside the footprint the costmaps held
  during the watch window (negative = inside, by that margin); `audit_mppi_fp_checked`
  = inflation radius >= the footprint's circumscribed radius (Nav2 MPPI 1.1.20
  checks the full footprint only then); `audit_contact_centre_clear_m` /
  `audit_contact_blind` = robot centre to nearest map obstacle at first contact,
  and whether that exceeded the inflation radius (footprint not checked).
- Telemetry events `plan` (global plan: `stamp`, arc length `s_m` per pose),
  `local_traj` (MPPI optimal trajectory: point k at `stamp + k*dt`) and
  `local_plan_window` (the tracked part of the global plan), each with `t_sim`.
- `fp_version`: 8-hex hash of the footprint polygon the label stood for. The
  treatment is (`arm_requested_label`, `fp_version`). Campaign 2: epoch-0 carry
  rows used a polygon that left the gripper fingers 2-4 cm outside (and the
  swing up to ~0.2 m, with `fp_policy` track_arm); from epoch 1 carry is the
  swept envelope of all arm collision geometry, applied at treatment time.
  Tucked rows are the same treatment in every epoch.

`attempt_kind` is `main` or `rerun`. A rerun repeats the same probe id with the
same pose, configuration and apply time; reruns exist only for attempts the
supervisor quarantined for an outcome-independent glitch (see attempts.csv
`quarantine_reason`).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    build(a.campaign_dir, quiet=a.quiet)


if __name__ == "__main__":
    main()
