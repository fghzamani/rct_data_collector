#!/usr/bin/env python3
"""Live health monitor for an RCT collection campaign.

Watches the collector's rct_results.csv while it fills and fails loudly the
moment the data stops being usable, instead of after 7000 trials. Every check
here exists because that exact failure already cost a campaign:

  footprint delivery   campaign A2 recorded footprint=carry for 3726 probes in
                       which no costmap ever received it. param_actual__ was
                       empty on every one of them and nothing noticed.
  target count         9 ROS targets is correct (7 knobs; inflation and
                       footprint each hit local AND global costmap). 7 means
                       the footprint pair silently vanished.
  speed delivery       probe_vx_mean / commanded cap should sit near 1.0.
  arm verification     arm_verified=0 means the costmap footprint and the
                       physical arm disagree -- those rows collided at 75%.
  randomisation        config must stay independent of the risk state, or the
                       whole identification argument is gone.
  event richness       a collision model cannot be fitted from 44 events.
  spatial spread       3 of 66 cells held 50% of campaign A2's collisions.

Usage
    python3 -m rct_collector.scripts.rct_monitor --data <out>/rct_results.csv
    python3 -m rct_collector.scripts.rct_monitor --data ... --once   # one report
"""
import os
import csv
import sys
import json
import time
import math
import argparse
from collections import Counter, defaultdict

RISK = ["risk__r_min", "risk__r_width", "risk__r_ttc",
        "risk__r_dens", "risk__r_clear", "risk__r_curve"]
CFG = ["param__controller_server__speed_limit_pct",
       "param__controller_server__FollowPath.vx_std",
       "param__controller_server__FollowPath.ConstraintCritic.cost_weight",
       "param__controller_server__FollowPath.CostCritic.cost_weight",
       "param__controller_server__FollowPath.PathAlignCritic.cost_weight",
       "param__local_costmap__inflation_layer.inflation_radius",
       "param__local_costmap__footprint"]
FP_CMD = "param__local_costmap__footprint"
FP_ACT = "param_actual__local_costmap__footprint"

# target bands; outside these the campaign is not giving the models what they need
TARGET = {
    "collision_rate": (0.04, 0.30),
    "stall_rate":     (0.08, 0.40),
    "carry_share":    (0.40, 0.60),
    "valid_rate":     (0.90, 1.01),
    "vx_ratio":       (0.70, 1.40),
    "max_cell_share": (0.00, 0.12),
}


def f(row, key):
    try:
        v = float(str(row.get(key, "")).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def truthy(v):
    return str(v).strip().lower() in ("1", "1.0", "true", "t")


def pearson(xs, ys):
    n = len(xs)
    if n < 30:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)) or 1e-12
    sy = math.sqrt(sum((y - my) ** 2 for y in ys)) or 1e-12
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


class Monitor:
    def __init__(self, path, target_n):
        self.path = path
        self.target_n = target_n
        self.seen = 0
        self.alerts = Counter()

    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="", errors="ignore") as fh:
            return list(csv.DictReader(l.replace("\x00", "") for l in fh))

    # ---------------- per-trial integrity ----------------
    def per_trial(self, rows):
        bad = defaultdict(list)
        for r in rows:
            tid = r.get("trial_id", "?")
            if not truthy(r.get("treatment_valid")):
                bad["treatment_invalid"].append(tid); continue
            if not truthy(r.get("config_verified")):
                bad["config_unverified"].append(tid)
            # THE campaign-A2 killer: a drawn footprint that was never applied
            if str(r.get(FP_CMD, "")).strip() and not str(r.get(FP_ACT, "")).strip():
                bad["footprint_not_delivered"].append(tid)
            for k in ("params_rejected_count", "params_mismatched_count",
                      "params_unverified_count"):
                v = f(r, k)
                if v:
                    bad[k].append(tid)
            if "arm_verified" in r and not truthy(r.get("arm_verified")):
                bad["arm_unverified"].append(tid)
            ratio = f(r, "probe_vx_mean_over_cap")
            if ratio is not None and not (TARGET["vx_ratio"][0] <= ratio <= TARGET["vx_ratio"][1]):
                bad["speed_not_delivered"].append(tid)
            if all(f(r, k) is None for k in RISK):
                bad["risk_state_missing"].append(tid)
        return bad

    # ---------------- campaign-level science ----------------
    def aggregate(self, rows):
        v = [r for r in rows if truthy(r.get("treatment_valid"))]
        n = len(v)
        out = {"rows": len(rows), "valid": n,
               "valid_rate": (n / len(rows)) if rows else 0.0}
        if not n:
            return out
        out["collision_rate"] = sum(truthy(r.get("y_h")) for r in v) / n
        out["collisions"] = sum(truthy(r.get("y_h")) for r in v)
        stall = [r for r in v if f(r, "probe_progress_m") is not None]
        if stall:
            out["stall_rate"] = sum(1 for r in stall if (f(r, "probe_progress_m") or 0) < 0.05) / len(stall)
        arm = Counter(str(r.get(FP_CMD, "")).strip() for r in v)
        tot_arm = arm.get("carry", 0) + arm.get("tucked", 0)
        out["carry_share"] = (arm.get("carry", 0) / tot_arm) if tot_arm else 0.0
        out["footprint_delivered_rate"] = sum(
            1 for r in v if str(r.get(FP_ACT, "")).strip()) / n

        # randomisation balance: |corr(config, risk)| must stay in the noise band
        worst, worst_pair = 0.0, ""
        for c in CFG:
            for k in RISK:
                xs, ys = [], []
                for r in v:
                    a = f(r, c) if c != FP_CMD else (
                        1.0 if str(r.get(c, "")).strip() == "carry" else
                        (0.0 if str(r.get(c, "")).strip() == "tucked" else None))
                    b = f(r, k)
                    if a is not None and b is not None:
                        xs.append(a); ys.append(b)
                if len(xs) >= 30:
                    rr = abs(pearson(xs, ys))
                    if rr > worst:
                        worst, worst_pair = rr, "%s x %s" % (c.split("__")[-1], k)
        out["max_abs_corr"] = round(worst, 4)
        out["max_abs_corr_pair"] = worst_pair
        out["corr_chance_band"] = round(1.96 / math.sqrt(max(n, 1)), 4)

        # spatial concentration of starts
        cells = Counter()
        for r in v:
            x, y = f(r, "start_x"), f(r, "start_y")
            if x is not None and y is not None:
                cells[(math.floor(x / 2) * 2, math.floor(y / 2) * 2)] += 1
        if cells:
            out["cells"] = len(cells)
            out["max_cell_share"] = max(cells.values()) / n
        vx = [f(r, "probe_vx_mean_over_cap") for r in v]
        vx = [x for x in vx if x is not None]
        if vx:
            vx.sort(); out["vx_ratio"] = vx[len(vx) // 2]
        return out

    def report(self, quiet_ok=False):
        rows = self.read()
        agg = self.aggregate(rows)
        bad = self.per_trial(rows)
        n = agg.get("valid", 0)
        pct = 100.0 * n / self.target_n if self.target_n else 0.0

        lines = []
        lines.append("=" * 72)
        lines.append("RCT MONITOR  %s" % time.strftime("%H:%M:%S"))
        lines.append("  %d valid / %d rows   target %d  (%.1f%%)"
                     % (n, agg.get("rows", 0), self.target_n, pct))
        if not n:
            lines.append("  no valid rows yet")
            print("\n".join(lines)); return agg

        def band(key, label, fmt="%.3f"):
            val = agg.get(key)
            if val is None:
                return
            lo, hi = TARGET.get(key, (None, None))
            ok = (lo is None) or (lo <= val <= hi)
            lines.append("  %-26s %s   %s" % (label, fmt % val,
                         "ok" if ok else "<-- OUT OF BAND [%s, %s]" % (lo, hi)))
            if not ok:
                self.alerts[key] += 1

        lines.append("-" * 72)
        lines.append("  SCIENCE")
        band("collision_rate", "collision rate")
        lines.append("  %-26s %d" % ("collision events", agg.get("collisions", 0)))
        band("stall_rate", "stall rate")
        band("carry_share", "carry share")
        band("max_cell_share", "busiest 2m start cell")
        lines.append("  %-26s %.4f  (chance band %.4f)%s"
                     % ("max |corr(config,risk)|", agg.get("max_abs_corr", 0),
                        agg.get("corr_chance_band", 0),
                        "" if agg.get("max_abs_corr", 0) <= agg.get("corr_chance_band", 1)
                        else "  <-- RANDOMISATION SUSPECT: " + agg.get("max_abs_corr_pair", "")))
        lines.append("-" * 72)
        lines.append("  DELIVERY")
        band("valid_rate", "treatment_valid rate")
        lines.append("  %-26s %.3f%s" % ("footprint delivered",
                     agg.get("footprint_delivered_rate", 0),
                     "" if agg.get("footprint_delivered_rate", 0) > 0.99
                     else "   <-- THE CAMPAIGN-A2 BUG IS BACK"))
        band("vx_ratio", "median vx / commanded cap")

        if bad:
            lines.append("-" * 72)
            lines.append("  PER-TRIAL FAULTS")
            for k, ids in sorted(bad.items(), key=lambda t: -len(t[1])):
                lines.append("  %-26s %5d trials   e.g. %s"
                             % (k, len(ids), ", ".join(ids[:3])))
        elif not quiet_ok:
            lines.append("  no per-trial faults")
        lines.append("=" * 72)
        print("\n".join(lines))
        return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rct_results.csv being written")
    ap.add_argument("--target-n", type=int, default=7000)
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--out", default=None, help="write the latest report as JSON here")
    a = ap.parse_args()

    m = Monitor(a.data, a.target_n)
    while True:
        agg = m.report()
        if a.out:
            with open(a.out, "w") as fh:
                json.dump({"ts": time.time(), **agg,
                           "alerts": dict(m.alerts)}, fh, indent=2)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
