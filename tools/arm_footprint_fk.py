#!/usr/bin/env python3
"""Exact arm footprint from the robot's COLLISION geometry, by forward kinematics.

measure_footprint.py approximates the arm by its link ORIGINS plus a buffer. On
TIAGo that misses the gripper fingers: their collision boxes reach 0.23 m past
gripper_link, 0.10 m beyond the furthest gripper frame. Campaign 2 epoch 0 found
the fingertips 2-4 cm outside the carry polygon with the arm out, and up to
~0.2 m outside while it swings out (61% of carry probes hit a wall).

This tool uses the URDF's collision meshes / boxes / cylinders, transformed by
FK at recorded joint angles, projected onto the ground plane (base_footprint):

  validate  FK link origins vs the TF poses recorded in telemetry (sanity)
  audit     how far each link's geometry was outside a polygon, per probe
  envelope  convex hull of the geometry over every recorded sample of the
            arm's swing and carry pose (+ base outline + margin) -> polygon

Run inside the ROS container (needs xacro + the tiago/pal_gripper meshes):
  python3 src/rct_data_collector/tools/arm_footprint_fk.py envelope --campaign campaigns/campaign2 --margin 0.03
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from rct_collector.arm_geometry import *  # noqa: F401,F403  (model, FK, polygon helpers)
from rct_collector.arm_geometry import ARM_LINK_RE, BASE, Robot, load_robot  # noqa: F401


# ── telemetry access ─────────────────────────────────────────────────────────
def probes(campaign, arm=None, epoch=None):
    import pandas as pd
    v = pd.read_csv(os.path.join(campaign, "dataset", "probes_valid.csv"))
    if arm:
        v = v[v.arm_requested_label == arm]
    if epoch is not None and "code_epoch" in v:
        v = v[v.code_epoch == epoch]
    return v


def tel(campaign, row):
    from rct_collector.telemetry import load_probe_telemetry
    cols, meta, ev = load_probe_telemetry(os.path.join(campaign, row.telemetry_relpath))
    return cols, meta, ev


def joint_dict(cols, i):
    return {k[3:]: float(cols[k][i]) for k in cols if k.startswith("q__") and np.isfinite(cols[k][i])}


def precontact_idx(cols, row, stride):
    t = cols["t_sim"] - cols["t_sim"][0]
    n = len(t)
    if row.collision == 1 and np.isfinite(row.tel_first_contact_t):
        n = int(np.searchsorted(t, row.tel_first_contact_t))
    return list(range(0, max(n, 1), stride)), t


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_validate(a, robot):
    v = probes(a.campaign, "carry").head(a.n)
    errs = {}
    for _, r in v.iterrows():
        cols, meta, _ = tel(a.campaign, r)
        for i in range(0, len(cols["t_sim"]), 40):
            q = joint_dict(cols, i)
            cache = {}
            for link in ("arm_3_link", "arm_5_link", "arm_7_link", "arm_tool_link", "gripper_link"):
                kx = f"link__{link}__x"
                if kx not in cols or not np.isfinite(cols[kx][i]):
                    continue
                p = robot.link_tf(link, q, cache)[:3, 3]
                m = np.array([cols[f"link__{link}__{c}"][i] for c in "xyz"])
                errs.setdefault(link, []).append(np.linalg.norm(p - m))
    for k, e in errs.items():
        print(f"  {k:16s} FK vs recorded TF: median {np.median(e)*1000:.1f} mm, max {np.max(e)*1000:.1f} mm (n={len(e)})")


def probe_samples(robot, cols, idx):
    """-> (pts (N,2), sample (N,), link (N,) object) for the arm collision
    geometry at the given samples. Samples with (near-)identical joint angles
    are computed once: the arm is static most of the window."""
    seen, P, S, L = {}, [], [], []
    for i in idx:
        q = joint_dict(cols, i)
        key = tuple(round(q.get(f"arm_{k}_joint", 0.0), 3) for k in range(1, 8)) + \
            (round(q.get("torso_lift_joint", 0.0), 3),)
        if key not in seen:
            seen[key] = robot.arm_points(q)
        for link, pts in seen[key].items():
            P.append(pts[:, :2])
            S.append(np.full(len(pts), i))
            L.append(np.full(len(pts), link, dtype=object))
    return np.vstack(P), np.concatenate(S), np.concatenate(L)


def cmd_audit(a, robot):
    poly = np.array(json.loads(a.polygon)) if a.polygon else None
    if poly is None:
        sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")
        from rct_collector.scripts.param_space import ARM_CONFIGS
        poly = np.array(json.loads(ARM_CONFIGS[a.arm]["footprint"]))
    v = probes(a.campaign, a.arm, a.epoch)
    rows = []
    for _, r in v.iterrows():
        cols, _, _ = tel(a.campaign, r)
        idx, t = precontact_idx(cols, r, a.stride)
        P, S, L = probe_samples(robot, cols, idx)
        d = poly_signed_dist(poly, P)
        k = int(np.argmax(d))
        last = S == idx[-1]
        kl = int(np.argmax(np.where(last, d, -9)))
        rows.append(dict(trial=r.trial_id, coll=r.collision, worst=d[k], worst_link=L[k],
                         worst_t=float(t[S[k]]), at_contact=d[kl], at_link=L[kl],
                         first=r.tel_first_contact_links))
    import pandas as pd
    d = pd.DataFrame(rows)
    for c in (1, 0):
        x = d[d.coll == c]
        if not len(x):
            continue
        print(f"{'collided' if c else 'no collision'} (n={len(x)}): arm geometry outside the polygon, "
              f"worst over the pre-contact window: median {x.worst.median()*100:.1f} cm, "
              f"p90 {x.worst.quantile(.9)*100:.1f} cm, share >1 cm {np.mean(x.worst > 0.01):.2f}")
        print(f"    at the last pre-contact sample: median {x.at_contact.median()*100:.1f} cm, "
              f"share >0 {np.mean(x.at_contact > 0):.2f}; worst links {x.worst_link.value_counts().head(4).to_dict()}")
    if a.out:
        d.to_csv(a.out, index=False)


def cmd_envelope(a, robot):
    sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")
    from rct_collector.scripts.param_space import ARM_CONFIGS
    old = np.array(json.loads(ARM_CONFIGS["carry"]["footprint"]))
    v = probes(a.campaign, "carry", a.epoch)
    tgt = np.array(ARM_CONFIGS["carry"]["joints"])
    final, swing = [], []
    torso = []
    for _, r in v.iterrows():
        cols, _, _ = tel(a.campaign, r)
        idx, t = precontact_idx(cols, r, a.stride)
        torso.append(np.nanmedian(cols["q__torso_lift_joint"]))
        at_tgt = [i for i in idx if np.nanmax(np.abs(np.array(
            [cols[f"q__arm_{k}_joint"][i] for k in range(1, 8)]) - tgt)) < 0.02]
        moving = [i for i in idx if i not in set(at_tgt)]
        for lst, store in ((at_tgt, final), (moving, swing)):
            if lst:
                P, _, _ = probe_samples(robot, cols, lst)
                store.append(convex_hull2d(P))
    q_cmd = {f"arm_{k}_joint": tgt[k - 1] for k in range(1, 8)}
    q_cmd["torso_lift_joint"] = float(np.nanmedian(torso))
    cmd_pts = np.vstack([P[:, :2] for P in robot.arm_points(q_cmd).values()])
    print(f"torso_lift median {q_cmd['torso_lift_joint']:.3f}; {len(final)} probes reached carry, {len(swing)} with swing samples")
    base = octagon(a.base_radius)
    area = lambda h: 0.5 * abs(np.dot(h[:, 0], np.roll(h[:, 1], 1)) - np.dot(h[:, 1], np.roll(h[:, 0], 1)))
    for name, sets in (("final carry pose only", final + [cmd_pts]),
                       ("swing + carry (envelope)", final + swing + [cmd_pts])):
        G = np.vstack(sets)
        P = G
        if a.margin > 0:
            P = np.vstack([G + a.margin * np.array([math.cos(t), math.sin(t)])
                           for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
        hull = simplify_outward(convex_hull2d(np.vstack([base, P])), a.max_vertices)
        over = poly_signed_dist(hull, G).max()
        print(f"\n{name}: old polygon misses the geometry by up to {poly_signed_dist(old, G).max()*100:.1f} cm")
        print(f"  new: {len(hull)} vertices, area {area(hull):.3f} m^2 (old {area(old):.3f}), "
              f"x [{hull[:,0].min():.3f}, {hull[:,0].max():.3f}], y [{hull[:,1].min():.3f}, {hull[:,1].max():.3f}], "
              f"geometry clearance to edge >= {-over*100:.1f} cm")
        print('  "footprint": "[' + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in hull) + ']"')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["validate", "audit", "envelope"])
    p.add_argument("--campaign", default="/home/forough/phd_projects/online_tuner/campaigns/campaign2")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--stride", type=int, default=2, help="use every k-th 20 Hz sample")
    p.add_argument("--arm", default="carry", help="audit: which arm label (its polygon and probes)")
    p.add_argument("--polygon", default="")
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--base-radius", type=float, default=0.275)
    p.add_argument("--max-vertices", type=int, default=16)
    p.add_argument("--out", default="")
    a = p.parse_args()
    robot = load_robot()
    {"validate": cmd_validate, "audit": cmd_audit, "envelope": cmd_envelope}[a.cmd](a, robot)


if __name__ == "__main__":
    main()
