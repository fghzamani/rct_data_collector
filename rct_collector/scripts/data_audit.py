"""Data audit for an RCT campaign: what a careful analyst checks, run by the
supervisor on every probe and every few dozen probes.

Per probe (audit_probe -> supervisor/audit_probes.jsonl, merged into the
dataset as audit_* columns):
  * footprint coverage: exact URDF collision geometry of the arm (forward
    kinematics on the recorded joint angles) vs the polygon the costmaps held
    at that instant (from the footprint_push marks). > 1 cm outside = Nav2 did
    not know where the robot was (campaign 2 run 1: fingers 4-24 cm outside).
  * MPPI footprint check: Nav2 MPPI 1.1.20 checks the full footprint only when
    the cell under the robot centre costs >= the inflation cost at the
    footprint's circumscribed radius, i.e. only if inflation_radius >= that
    radius. audit_mppi_fp_checked records it per probe.
  * contacts: robot centre to nearest map obstacle at first contact, and
    whether that exceeded the inflation radius (centre cell cost 0 -> the
    footprint was not checked: a 'blind' strike).
  * local/global trajectory streams present.

Campaign level (audit_campaign -> alerts, supervisor/audit_report.md):
  randomisation (treatment share, covariate balance across arms, uniform
  coverage of each knob), positivity of the inflation threshold per arm, the
  safety mechanism (collision rate when the footprint IS checked), footprint
  coverage, outcome drift over time, dead (constant) features, missingness
  drift, localisation, real-time factor, corridor stalls, new Nav2 error
  signatures in the stack logs.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

HALT, WARN, INFO = "halt", "warn", "info"


def _f(v, d=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "1.0", "true", "yes")


def _poly_radii(poly: np.ndarray):
    """(inscribed, circumscribed) radius as nav2_costmap_2d computes them."""
    circ = float(np.hypot(poly[:, 0], poly[:, 1]).max())
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    insc = float(np.hypot(*(a + t[:, None] * ab).T).min())
    return insc, circ


class DataAudit:
    # Bump when audit_probe() gains or changes a column: the supervisor then
    # re-audits stored probes whose audit_version is older.
    # v2: localisation error split at the first contact (audit_loc_err_*).
    VERSION = 2
    INFL_KEY = "param__local_costmap__inflation_layer.inflation_radius"

    def __init__(self, campaign_root: str, collector_cfg: dict, health: dict, log=None):
        from rct_collector.scripts.param_space import ARM_CONFIGS
        self.root = campaign_root
        self.H = health or {}
        self.log = log
        self.polys = {k: np.array(json.loads(v["footprint"])) for k, v in ARM_CONFIGS.items()}
        self.radii = {k: _poly_radii(p) for k, p in self.polys.items()}
        self.robot = None
        self.robot_error = ""
        self.dist_map = None
        self.map_error = ""
        self.map_yaml = collector_cfg.get("map_yaml_path")
        self.num_trials = int(collector_cfg.get("num_trials") or 0)
        self.log_signatures = {}
        self.sig_path = os.path.join(campaign_root, "supervisor", "log_signatures.json")
        if os.path.exists(self.sig_path):
            try:
                self.log_signatures = json.load(open(self.sig_path))
            except (OSError, ValueError):
                self.log_signatures = {}

    # ── lazy resources ──────────────────────────────────────────────────
    def _robot(self):
        if self.robot is None and not self.robot_error:
            try:
                from rct_collector.arm_geometry import load_robot
                self.robot = load_robot()
            except Exception as exc:   # noqa: BLE001
                self.robot_error = f"robot model unavailable: {exc}"
        return self.robot

    def _dmap(self):
        if self.dist_map is None and not self.map_error:
            try:
                import yaml
                from scipy.ndimage import distance_transform_edt
                y = yaml.safe_load(open(self.map_yaml))
                img_path = os.path.join(os.path.dirname(self.map_yaml), y["image"])
                raw = open(img_path, "rb").read()
                # binary PGM (P5): header tokens then bytes
                toks, i = [], 0
                while len(toks) < 4:
                    while raw[i:i + 1].isspace():
                        i += 1
                    if raw[i:i + 1] == b"#":
                        while raw[i:i + 1] not in (b"\n", b""):
                            i += 1
                        continue
                    j = i
                    while not raw[j:j + 1].isspace():
                        j += 1
                    toks.append(raw[i:j])
                    i = j
                w, h = int(toks[1]), int(toks[2])
                img = np.frombuffer(raw[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)
                occ = img < 100
                res = float(y["resolution"])
                self.dist_map = (distance_transform_edt(~occ) * res, res,
                                 float(y["origin"][0]), float(y["origin"][1]), h)
            except Exception as exc:   # noqa: BLE001
                self.map_error = f"map distance unavailable: {exc}"
        return self.dist_map

    def clearance(self, x: float, y: float) -> Optional[float]:
        dm = self._dmap()
        if dm is None:
            return None
        d, res, ox, oy, h = dm
        c, r = int((x - ox) / res), h - 1 - int((y - oy) / res)
        if 0 <= r < d.shape[0] and 0 <= c < d.shape[1]:
            return float(d[r, c])
        return None

    # ── per probe ───────────────────────────────────────────────────────
    def audit_probe(self, r: dict) -> dict:
        out = {"run_id": r.get("run_id"), "trial_id": int(_f(r.get("trial_id"), -1)),
               "attempt_kind": r.get("attempt_kind") or "main", "audit_version": self.VERSION}
        label = r.get("arm_requested_label") or "tucked"
        infl = _f(r.get(self.INFL_KEY))
        if label in self.radii and infl is not None:
            out["audit_fp_circ_m"] = round(self.radii[label][1], 3)
            out["audit_mppi_fp_checked"] = int(infl >= self.radii[label][1])
        out["audit_n_local_traj"] = int(_f(r.get("tel_n_local_traj"), 0))
        # raw rows carry the absolute tel_path; dataset rows a path relative
        # to the campaign root (telemetry_relpath)
        cands = [r.get("tel_path") or ""]
        rel = r.get("telemetry_relpath") or ""
        if rel:
            cands += [os.path.join(self.root, rel), os.path.join(self.root, "raw", rel)]
        path = next((c for c in cands if c and os.path.exists(c)), "")
        if not path:
            out["audit_error"] = "no telemetry"
            return out
        try:
            from rct_collector.telemetry import load_probe_telemetry
            cols, meta, ev = load_probe_telemetry(path)
        except Exception as exc:   # noqa: BLE001
            out["audit_error"] = f"telemetry unreadable: {exc}"
            return out
        t = cols["t_sim"]
        t0 = t[0]
        contact_t = _f(r.get("tel_first_contact_t"))
        n_end = len(t) if contact_t is None else int(np.searchsorted(t - t0, contact_t))
        # Only the watch window counts: after watch_end the revert puts the
        # tucked footprint back while the arm is still folding (not measured).
        we = [e["t_sim"] for e in ev if e.get("kind") == "watch_end"]
        if we:
            n_end = min(n_end, int(np.searchsorted(t, we[-1])))
        drive = [e["t_sim"] for e in ev if e.get("kind") == "drive_started"]
        i_start = int(np.searchsorted(t, drive[-1])) if drive else 0
        pushes = sorted((e["t_sim"], e.get("label")) for e in ev
                        if e.get("kind") == "footprint_push" and e.get("label"))

        def held(ts):
            lab = "tucked"
            for tp, lb in pushes:
                if tp <= ts:
                    lab = lb
            return lab

        # Localisation error (ground truth vs /amcl_pose) from 1 s after the
        # drive starts (AMCL is re-seeded at the teleport) up to the first
        # contact. After a contact the impact moves the robot without odometry
        # seeing it and AMCL loses track (up to ~5 m); that is a consequence of
        # the outcome, not a pre-outcome localisation problem, so it is kept
        # apart in audit_loc_err_post_max.
        le = cols.get("loc_err")
        if le is not None and drive:
            t_c = (t0 + contact_t) if contact_t is not None else np.inf
            pre = (t >= drive[-1] + 1.0) & (t < t_c) & np.isfinite(le)
            post = (t >= t_c) & np.isfinite(le)
            if pre.any():
                out["audit_loc_err_pre_max"] = round(float(le[pre].max()), 3)
                out["audit_loc_err_pre_p50"] = round(float(np.median(le[pre])), 3)
            if post.any():
                out["audit_loc_err_post_max"] = round(float(le[post].max()), 3)

        robot = self._robot()
        if robot is not None:
            try:
                from rct_collector.arm_geometry import poly_signed_dist
                worst, wlink, wt, cache = -9.0, "", None, {}
                for i in range(i_start, max(i_start + 1, min(n_end, len(t))), 2):
                    q = {k[3:]: float(cols[k][i]) for k in cols
                         if k.startswith("q__") and np.isfinite(cols[k][i])}
                    key = tuple(round(q.get(f"arm_{k}_joint", 0.0), 3) for k in range(1, 8))
                    lab = held(t[i])
                    if (key, lab) in cache:
                        d, link = cache[(key, lab)]
                    else:
                        pts = robot.arm_points(q)
                        dl = {ln: float(poly_signed_dist(self.polys.get(lab, self.polys["tucked"]),
                                                         P[:, :2]).max()) for ln, P in pts.items()}
                        link = max(dl, key=dl.get)
                        d = dl[link]
                        cache[(key, lab)] = (d, link)
                    if d > worst:
                        worst, wlink, wt = d, link, t[i] - t0
                out["audit_fp_outside_cm"] = round(worst * 100, 2)
                out["audit_fp_worst_link"] = wlink
                out["audit_fp_worst_t"] = None if wt is None else round(float(wt), 2)
            except Exception as exc:   # noqa: BLE001
                out["audit_error"] = f"fk failed: {exc}"
        else:
            out["audit_error"] = self.robot_error
        if _truthy(r.get("collision")) and contact_t is not None and n_end > 0:
            i = max(0, min(n_end - 2, len(t) - 1))
            other = str(r.get("tel_first_contact_with") or "")
            cl = self.clearance(float(cols["gt_x"][i]), float(cols["gt_y"][i]))
            out["audit_contact_with"] = other
            if cl is not None:
                out["audit_contact_centre_clear_m"] = round(cl, 3)
                if infl is not None and not other.startswith("obs_dynamic"):
                    out["audit_contact_blind"] = int(cl > infl)
        return out

    # ── campaign level ──────────────────────────────────────────────────
    def audit_campaign(self, valid: list, audits: dict, stack_logs: list) -> tuple:
        """valid: accepted, valid main rows in trial order. audits: {(run_id,
        trial_id): audit dict}. -> (findings [(level, key, msg)], summary)."""
        F, S = [], {}
        n = len(valid)
        S["n_valid"] = n
        A = [audits.get((str(r.get("run_id")), int(_f(r.get("trial_id"), -1))), {}) for r in valid]
        arm = [r.get("arm_requested_label") or "?" for r in valid]
        coll = np.array([_truthy(r.get("collision")) for r in valid], bool)
        stall = np.array([_truthy(r.get("probe_stalled")) for r in valid], bool)

        # 1. footprint coverage (geometry the costmap did not know about)
        fo = [(a.get("audit_fp_outside_cm"), a.get("audit_fp_worst_link"), r.get("trial_id"))
              for a, r in zip(A, valid) if a.get("audit_fp_outside_cm") is not None]
        S["fp_audited"] = len(fo)
        if fo:
            last = fo[-50:]
            bad = [x for x in last if x[0] > float(self.H.get("fp_outside_tol_cm", 1.0))]
            S["fp_outside_share_last50"] = round(len(bad) / len(last), 3)
            S["fp_outside_max_cm"] = max(x[0] for x in fo)
            if len(last) >= 10 and len(bad) / len(last) > float(self.H.get("halt_fp_outside_share", 0.05)):
                F.append((HALT, "fp_coverage",
                          f"arm collision geometry outside the costmap footprint by "
                          f"> {self.H.get('fp_outside_tol_cm', 1.0)} cm in {len(bad)}/{len(last)} recent probes "
                          f"(links {Counter(x[1] for x in bad).most_common(3)}): Nav2 does not know "
                          "where the arm is."))
            elif bad:
                F.append((WARN, "fp_coverage", f"{len(bad)} recent probe(s) with arm geometry outside the "
                                                f"footprint, e.g. probe {bad[-1][2]} by {bad[-1][0]} cm ({bad[-1][1]})"))
        missing_fk = sum(1 for a in A if a.get("audit_error"))
        if missing_fk:
            S["audit_errors"] = missing_fk
            errs = Counter(a.get("audit_error") for a in A if a.get("audit_error"))
            F.append((WARN, "audit_errors", f"{missing_fk} probes could not be audited: {dict(errs.most_common(3))}"))

        # 2. safety mechanism: collisions when MPPI DOES check the footprint
        for lab in sorted(set(arm)):
            idx = [i for i, a in enumerate(arm) if a == lab and A[i].get("audit_mppi_fp_checked") is not None]
            chk = [i for i in idx if A[i]["audit_mppi_fp_checked"] == 1]
            unc = [i for i in idx if A[i]["audit_mppi_fp_checked"] == 0]
            rate = lambda ii: round(float(coll[ii].mean()), 3) if ii else None
            S[f"{lab}_fp_checked_n"] = len(chk)
            S[f"{lab}_fp_unchecked_n"] = len(unc)
            S[f"{lab}_coll_rate_fp_checked"] = rate(chk)
            S[f"{lab}_coll_rate_fp_unchecked"] = rate(unc)
            S[f"{lab}_stall_rate_fp_checked"] = round(float(stall[chk].mean()), 3) if chk else None
            if len(chk) >= int(self.H.get("min_checked_probes", 40)) and \
                    rate(chk) > float(self.H.get("warn_coll_rate_fp_checked", 0.15)):
                strikes = Counter((valid[i].get("tel_first_contact_links") or "?").split(";")[0]
                                  for i in chk if coll[i])
                F.append((WARN, f"{lab}_checked_collisions",
                          f"{lab}: {rate(chk):.0%} of {len(chk)} probes collide even though MPPI checks "
                          f"the full footprint (inflation >= {self.radii.get(lab, (0, 0))[1]:.2f} m); first "
                          f"contacts {dict(strikes.most_common(3))}. Another blind mechanism?"))
            if n >= 200 and lab in self.radii and len(chk) < 0.05 * max(1, len(idx)):
                F.append((WARN, f"{lab}_positivity",
                          f"{lab}: only {len(chk)}/{len(idx)} probes above the footprint-check threshold "
                          f"({self.radii[lab][1]:.2f} m): the inflation range does not cover both sides."))
        blind = [a.get("audit_contact_blind") for a in A if a.get("audit_contact_blind") is not None]
        if blind:
            S["contacts_blind_share"] = round(sum(blind) / len(blind), 3)

        # 2b. learnable events per arm: the outcome models need enough VALID
        #     collisions and stalls in every arm (and on both sides of the
        #     footprint-check threshold) to estimate what each knob does.
        table = {}
        target = int(self.H.get("min_events_per_arm", 50))
        for lab in sorted(set(arm)):
            ii = [i for i, a in enumerate(arm) if a == lab]
            row = {"n": len(ii), "collisions": int(coll[ii].sum()), "stalls": int(stall[ii].sum())}
            for side, flag in (("fp_checked", 1), ("fp_unchecked", 0)):
                jj = [i for i in ii if A[i].get("audit_mppi_fp_checked") == flag]
                row[side] = {"n": len(jj), "collisions": int(coll[jj].sum()), "stalls": int(stall[jj].sum())}
            if self.num_trials and n:
                n_end = self.num_trials * len(ii) / n
                row["projected_at_end"] = {"collisions": int(round(row["collisions"] / len(ii) * n_end)),
                                           "stalls": int(round(row["stalls"] / len(ii) * n_end))}
                if len(ii) >= 100:
                    thin = [k for k, v in row["projected_at_end"].items() if v < target]
                    if thin:
                        F.append((WARN, f"{lab}_thin_events",
                                  f"{lab}: {row['collisions']} collisions and {row['stalls']} stalls in "
                                  f"{len(ii)} valid probes; at this rate only {row['projected_at_end']} by "
                                  f"probe {self.num_trials} (< {target} {', '.join(thin)}): too few events "
                                  "to learn knob effects."))
            table[lab] = row
        S["events_by_arm"] = table

        # 3. randomisation: treatment share, covariate balance, knob coverage
        if n >= 100:
            carry = sum(a == "carry" for a in arm) / n
            S["carry_share"] = round(carry, 3)
        if n >= 200:
            feats = [k for k in valid[0].keys() if k.startswith("risk__") and k != "risk__timestamp"] + \
                    ["d_remaining_at_apply_m", "t_apply_sim", "pool_min_gap_m", "start_x", "start_y"]
            smd = {}
            for k in feats:
                a_ = np.array([_f(r.get(k)) for r, g in zip(valid, arm) if g == "carry"], float)
                b_ = np.array([_f(r.get(k)) for r, g in zip(valid, arm) if g == "tucked"], float)
                a_, b_ = a_[np.isfinite(a_)], b_[np.isfinite(b_)]
                if len(a_) > 20 and len(b_) > 20:
                    sd = math.sqrt((a_.var() + b_.var()) / 2)
                    if sd > 0:
                        smd[k] = round(float((a_.mean() - b_.mean()) / sd), 3)
            S["smd_max"] = max(map(abs, smd.values())) if smd else None
            # chance-aware: |SMD| above both the practical bound and 3.5 sampling SEs
            na = sum(a == "carry" for a in arm)
            nb = sum(a == "tucked" for a in arm)
            lim = max(float(self.H.get("warn_smd", 0.1)), 3.5 * math.sqrt(1 / max(na, 1) + 1 / max(nb, 1)))
            S["smd_limit"] = round(lim, 3)
            badb = {k: v for k, v in smd.items() if abs(v) > lim}
            if badb:
                F.append((WARN, "balance", f"covariate imbalance between arms (|SMD| > 0.1): {badb}"))
            knobs = [k for k in valid[0].keys() if k.startswith("param__") and "footprint" not in k]
            cov = {}
            for k in knobs:
                x = np.array([_f(r.get(k)) for r in valid], float)
                x = x[np.isfinite(x)]
                if len(x) < 100 or x.max() <= x.min():
                    continue
                h, _ = np.histogram(x, bins=10, range=(x.min(), x.max()))
                cov[k.replace("param__", "")] = round(float(h.min() / (len(x) / 10)), 2)
            S["knob_min_decile_ratio"] = cov
            thin = {k: v for k, v in cov.items() if v < 0.5}
            if thin:
                F.append((WARN, "knob_coverage", f"a knob decile has < 50% of its expected probes: {thin}"))

            # 4. dead features, missingness drift
            last = valid[-200:]
            dead = [k for k in valid[0].keys() if k.startswith("risk__")
                    and k != "risk__timestamp"
                    and len({str(r.get(k)) for r in last}) == 1]
            if dead:
                S["constant_features"] = dead
                F.append((INFO, "constant_features", f"constant over the last 200 probes: {dead}"))
            first = valid[:100]
            drift = []
            for k in valid[0].keys():
                m0 = sum(_f(r.get(k)) is None and str(r.get(k, "")).strip() == "" for r in first) / len(first)
                m1 = sum(_f(r.get(k)) is None and str(r.get(k, "")).strip() == "" for r in last[-100:]) / 100
                if m0 < 0.05 and m1 > 0.5:
                    drift.append(k)
            if drift:
                F.append((WARN, "missingness", f"columns that became mostly empty: {drift[:10]}"))

        # 5. outcome drift (per arm, last 200 vs earlier)
        if n >= 400:
            for lab in sorted(set(arm)):
                ii = np.array([i for i, a in enumerate(arm) if a == lab])
                if len(ii) < 200:
                    continue
                cut = ii[ii >= n - 200]
                old = ii[ii < n - 200]
                for name, y in (("collision", coll), ("stall", stall)):
                    p1, p0 = y[cut].mean(), y[old].mean()
                    p = y[ii].mean()
                    se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / len(cut) + 1 / len(old)))
                    z = (p1 - p0) / se
                    S[f"drift_z_{lab}_{name}"] = round(float(z), 2)
                    if abs(z) > float(self.H.get("warn_drift_z", 3.5)):
                        F.append((WARN, f"drift_{lab}_{name}",
                                  f"{lab} {name} rate {p0:.3f} -> {p1:.3f} in the last 200 probes "
                                  f"(z = {z:.1f}): the simulator or stack is drifting."))

        # 6. localisation, real-time factor, corridor stalls, trajectory streams
        # pre-contact only: see audit_probe (post-contact error is an outcome effect)
        le = [x for x in (_f(a.get("audit_loc_err_pre_max")) for a in A[-100:]) if x is not None]
        if le:
            S["loc_err_p95_last100"] = round(float(np.percentile(le, 95)), 3)
            if S["loc_err_p95_last100"] > float(self.H.get("warn_loc_err_p95_m", 0.3)):
                F.append((WARN, "localisation", f"pre-contact localisation error p95 "
                                                f"{S['loc_err_p95_last100']} m over the last 100 probes"))
        lp = [x for x in (_f(a.get("audit_loc_err_post_max")) for a in A[-100:]) if x is not None]
        if lp:
            S["loc_err_post_contact_p50_last100"] = round(float(np.median(lp)), 3)
        rtf = [x for x in (_f(r.get("tel_rtf_mean")) for r in valid[-100:]) if x is not None]
        if rtf:
            S["rtf_median_last100"] = round(float(np.median(rtf)), 3)
            if S["rtf_median_last100"] < 0.9:
                F.append((WARN, "rtf", f"median real-time factor {S['rtf_median_last100']} over the last 100"))
        cor = [i for i, r in enumerate(valid) if r.get("pool_share") == "corridor"]
        if len(cor) >= 30:
            S["corridor_stall_rate"] = round(float(stall[cor].mean()), 3)
            if S["corridor_stall_rate"] > 0.5:
                F.append((WARN, "corridor_stalls", f"{S['corridor_stall_rate']:.0%} of corridor probes stall"))
        notraj = sum(1 for a in A[-50:] if a.get("audit_n_local_traj", 1) == 0)
        if A and notraj > 0.2 * len(A[-50:]):
            F.append((WARN, "local_traj", f"{notraj}/{len(A[-50:])} recent probes recorded no MPPI local "
                                          "trajectory (FollowPath.visualize off?)"))

        # 7. Nav2 / Gazebo error signatures in the stack logs
        new = self.scan_logs(stack_logs)
        if new:
            F.append((WARN if len(self.log_signatures) > len(new) else INFO, "new_log_errors",
                      f"{len(new)} new error signature(s) in the stack log: " + " | ".join(new[:3])))
        S["log_error_signatures"] = len(self.log_signatures)
        return F, S

    _SHUTDOWN_MARKERS = ("user interrupted with ctrl-c", "sending signal 'SIGINT'",
                         "sending signal 'SIGTERM'", "signal_handler(signum=2)",
                         "signal_handler(signum=15)")
    _SIG_STRIP = re.compile(r"\[\d+(\.\d+)?\]|\b\d+(\.\d+)?\b|0x[0-9a-f]+")

    def scan_logs(self, paths: list) -> list:
        """Normalise ERROR/FATAL lines (numbers stripped) into signatures; return
        the ones never seen before in this campaign."""
        new = []
        for p in paths:
            try:
                with open(p, errors="ignore") as f:
                    for line in f:
                        # Everything after the launch receives SIGINT/SIGTERM is
                        # the supervisor shutting the stack down on purpose
                        # ("process has died ... exit code -2"), not a fault.
                        if any(m in line for m in self._SHUTDOWN_MARKERS):
                            break
                        if "[ERROR]" not in line and "[FATAL]" not in line:
                            continue
                        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                        sig = self._SIG_STRIP.sub("#", line.strip())[:220]
                        if sig not in self.log_signatures:
                            self.log_signatures[sig] = 0
                            new.append(sig)
                        self.log_signatures[sig] += 1
            except OSError:
                continue
        if new or paths:
            try:
                with open(self.sig_path + ".tmp", "w") as f:
                    json.dump(self.log_signatures, f, indent=1)
                os.replace(self.sig_path + ".tmp", self.sig_path)
            except OSError:
                pass
        return new

    def write_report(self, path: str, findings: list, summary: dict):
        lines = [f"# Data audit — {datetime.now().isoformat(timespec='seconds')}", "",
                 "## Findings", ""]
        if not findings:
            lines.append("None.")
        for lvl, key, msg in findings:
            lines.append(f"- **{lvl.upper()}** `{key}`: {msg}")
        lines += ["", "## Summary", "", "```json", json.dumps(summary, indent=1, default=str), "```",
                  "", "## Footprint thresholds (MPPI checks the full footprint only if inflation >= circumscribed radius)", ""]
        for k, (i, c) in self.radii.items():
            lines.append(f"- {k}: inscribed {i:.3f} m, circumscribed {c:.3f} m")
        with open(path + ".tmp", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(path + ".tmp", path)
