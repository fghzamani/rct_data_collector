#!/usr/bin/env python3
"""Distribution of the campaign-2 pose pool: where it goes and how hard it is.

Usage: pool_analysis.py <pool.json> <map.yaml> <out_dir>
Plans every pair on the campaign-2 map (8-connected Dijkstra over cells where a
tucked base fits, costed like Nav2's inflation layer + NavFn at the baseline
inflation radius), so the difficulty is measured on a plausible ROUTE rather than on
the straight start->goal segment the pool labels use.
"""
import sys, json, os
from collections import Counter
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

pool_path, map_yaml, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
poses = json.load(open(pool_path))
N = len(poses)

NAV_R, CARRY_R, CARRY_W = 0.325, 0.847, 0.97
OBS = {"obs_corner_mid": (-2.00, 0.00), "obs_corner_west": (-7.50, -3.00)}
DYN_STANDBY, DYN_BLOCK = (-4.50, -2.80), (-4.50, -0.60)
DYN2_SEG = ((7.199, -1.110), (10.281, -0.250))   # obs_dynamic_2 board, blocking pose

info = yaml.safe_load(open(map_yaml))
res = info["resolution"]; ox, oy = info["origin"][:2]
img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), info["image"])).convert("L"))
H, W = img.shape
occ = img <= int(255 * (1 - info["occupied_thresh"]))
unknown = (~occ) & (img < int(255 * (1 - info["free_thresh"])))
free = (~occ) & (~unknown)
edt = distance_transform_edt(~occ) * res          # clearance to nearest occupied cell

def to_px(x, y):
    return int(round(H - 1 - (y - oy) / res)), int(round((x - ox) / res))

def to_w(r, c):
    return ox + c * res, oy + (H - 1 - r) * res

# ---------------- graph over navigable cells ----------------
nav = free & (edt >= NAV_R)
idx = -np.ones((H, W), dtype=np.int64)
rr, cc = np.nonzero(nav)
idx[rr, cc] = np.arange(len(rr))
# cost per metre: Nav2 inflation (pmb2 defaults: radius 0.55, scaling 3.0,
# robot_radius 0.275) over NavFn's neutral cost 50 -- same as PoseSampler
shape = 1.0 + np.where(edt < 0.55, 253.0 * np.exp(-3.0 * np.maximum(edt - 0.275, 0)), 0.0) / 50.0
rows, cols, wts = [], [], []
for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
    r2, c2 = rr + dr, cc + dc
    ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
    r1, c1, r2, c2 = rr[ok], cc[ok], r2[ok], c2[ok]
    ok2 = nav[r2, c2]
    r1, c1, r2, c2 = r1[ok2], c1[ok2], r2[ok2], c2[ok2]
    step = res * (np.sqrt(2) if dr and dc else 1.0)
    w = step * 0.5 * (shape[r1, c1] + shape[r2, c2])
    rows += [idx[r1, c1], idx[r2, c2]]; cols += [idx[r2, c2], idx[r1, c1]]; wts += [w, w]
G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
               shape=(len(rr), len(rr)))

def nearest_nav(r, c):
    if nav[r, c]:
        return idx[r, c]
    win = 12
    r0, c0 = max(0, r - win), max(0, c - win)
    sub = nav[r0:r + win + 1, c0:c + win + 1]
    pr, pc = np.nonzero(sub)
    if len(pr) == 0:
        return -1
    k = np.argmin((pr + r0 - r) ** 2 + (pc + c0 - c) ** 2)
    return idx[pr[k] + r0, pc[k] + c0]

# ---------------- plan every pair ----------------
heat = np.zeros((H, W))
rec = []
paths = []
for p in poses:
    s = nearest_nav(*to_px(p["start"]["x"], p["start"]["y"]))
    g = nearest_nav(*to_px(p["goal"]["x"], p["goal"]["y"]))
    if s < 0 or g < 0:
        rec.append(None); paths.append(None); continue
    _, pred = dijkstra(G, indices=s, return_predecessors=True, limit=np.inf)
    if pred[g] < 0 and g != s:
        rec.append(None); paths.append(None); continue
    seq = [g]
    while seq[-1] != s:
        seq.append(pred[seq[-1]])
    seq = np.array(seq[::-1])
    pr, pc = rr[seq], cc[seq]
    xy = np.array([to_w(a, b) for a, b in zip(pr, pc)])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    L = float(seg.sum())
    clr = edt[pr, pc]
    # heading change, measured on a 0.5 m-resampled polyline to ignore grid zig-zag
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, L, 0.5)
    rx, ry = np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])
    hd = np.arctan2(np.diff(ry), np.diff(rx))
    dh = np.abs((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi)
    tot_turn = float(np.degrees(dh.sum()))
    sharp = int((np.degrees(dh) > 30).sum())
    heat[pr, pc] += 1
    dobs = {k: float(np.min(np.hypot(xy[:, 0] - v[0], xy[:, 1] - v[1]))) for k, v in OBS.items()}
    dobs["dyn_block"] = float(np.min(np.hypot(xy[:, 0] - DYN_BLOCK[0], xy[:, 1] - DYN_BLOCK[1])))
    (ax_, ay_), (bx_, by_) = DYN2_SEG
    ex_, ey_ = bx_ - ax_, by_ - ay_
    tt = np.clip(((xy[:, 0] - ax_) * ex_ + (xy[:, 1] - ay_) * ey_) / (ex_ * ex_ + ey_ * ey_), 0, 1)
    dobs["dyn2_board"] = float(np.min(np.hypot(xy[:, 0] - ax_ - tt * ex_, xy[:, 1] - ay_ - tt * ey_)))
    rec.append(dict(
        id=p["id"], L=L, straight=p["distance"], detour=L / max(p["distance"], 1e-6),
        min_clr=float(clr.min()),
        frac_below_carry=float((clr < CARRY_R).mean()),   # arm envelope touches inflation band
        len_below_carry=float(seg[(clr[:-1] < CARRY_R)].sum()),
        frac_below_06=float((clr < 0.60).mean()),
        tot_turn=tot_turn, sharp=sharp, **{"d_" + k: v for k, v in dobs.items()}))
    paths.append(xy)

ok = [r for r in rec if r]
print(f"planned {len(ok)}/{N} pairs")
A = {k: np.array([r[k] for r in ok]) for k in ok[0] if k != "id"}

def q(a):
    return " ".join(f"{v:.2f}" for v in np.quantile(a, [.05, .25, .5, .75, .95]))

summ = {}
print("\n== pool labels ==")
cls = Counter(p["difficulty"]["class"] for p in poses)
tags = Counter(t for p in poses for t in p["difficulty"]["tags"])
ntag = Counter(p["difficulty"]["n_tags"] for p in poses)
strata = Counter(p.get("stratum") for p in poses)
print("class", dict(cls)); print("tags", dict(tags)); print("n_tags", dict(sorted(ntag.items())))
print("strata", dict(strata), " crossed", sum(bool(p.get("crossed")) for p in poses))
print("share x stratum", dict(Counter((p.get("share"), p.get("stratum")) for p in poses)))
combo = Counter("+".join(sorted(p["difficulty"]["tags"])) or "(none)" for p in poses)
print("top tag combos", combo.most_common(10))
d = np.array([p["distance"] for p in poses]); print("straight dist q05..q95", q(d))
sclr = np.array([edt[to_px(p["start"]["x"], p["start"]["y"])] for p in poses])
gclr = np.array([edt[to_px(p["goal"]["x"], p["goal"]["y"])] for p in poses])
print("start clearance q", q(sclr), " min", sclr.min().round(3))
print("goal  clearance q", q(gclr), " min", gclr.min().round(3))
mg = np.array([p["difficulty"]["min_gap_m"] for p in poses])
print("label min_gap (straight) q", q(mg))
tight = ((mg >= 0.65) & (mg < 1.20)).sum(); print("tight-band (0.65-1.20) straight gap", tight)

print("\n== planned routes ==")
for k in ["L", "detour", "min_clr", "frac_below_carry", "len_below_carry", "frac_below_06", "tot_turn", "sharp"]:
    print(f"{k:18s} {q(A[k])}")
print("routes whose min clearance < carry radius 0.847:", int((A["min_clr"] < CARRY_R).sum()))
print("routes with >=1 m below carry radius:", int((A["len_below_carry"] >= 1.0).sum()))
print("routes with min clearance < 0.485 (half carry width):", int((A["min_clr"] < CARRY_W / 2).sum()))
for k in ["d_obs_corner_mid", "d_obs_corner_west", "d_dyn_block", "d_dyn2_board"]:
    print(f"routes within 1.0/1.5/2.0 m of {k[2:]}: "
          f"{int((A[k] <= 1.0).sum())}/{int((A[k] <= 1.5).sum())}/{int((A[k] <= 2.0).sum())}")
print("guard check: planned L > 8.1 m (H=12):", int((A["L"] > 8.1).sum()), " > 5.7 m (H=8):", int((A["L"] > 5.7).sum()))

# spatial balance like the monitor: busiest 2 m start cell; 3 m region coverage
def cell(x, y, m):
    return (int(np.floor((x - ox) / m)), int(np.floor((y - oy) / m)))
for m in (2.0, 3.0):
    sc = Counter(cell(p["start"]["x"], p["start"]["y"], m) for p in poses)
    gc = Counter(cell(p["goal"]["x"], p["goal"]["y"], m) for p in poses)
    navcells = {cell(*to_w(a, b), m) for a, b in zip(rr[::7], cc[::7])}
    print(f"{m:.0f} m cells: nav cells {len(navcells)}, starts in {len(sc)} (busiest {max(sc.values())/N:.3f}),"
          f" goals in {len(gc)} (busiest {max(gc.values())/N:.3f})")
heat_cov = (heat > 0)
print(f"navigable area touched by >=1 planned route: {heat_cov[nav].mean():.1%}")
# coverage per 3 m navigable region by routes
step = int(round(3.0 / res))
regs = {}
for a, b in zip(rr, cc):
    regs.setdefault((a // step, b // step), [0, 0])
for a, b in zip(*np.nonzero(heat_cov & nav)):
    regs[(a // step, b // step)][1] += 1
cnt = np.array([v for (a, b), v in []])
reg_hit = sum(1 for k in regs if np.any(heat[k[0]*step:(k[0]+1)*step, k[1]*step:(k[1]+1)*step] > 0))
print(f"3 m navigable regions crossed by >=1 route: {reg_hit}/{len(regs)}")

json.dump(dict(n=N, planned=len(ok), records=ok), open(os.path.join(out, "pool_routes.json"), "w"), indent=1)

# ---------------- figures ----------------
ext = [ox, ox + W * res, oy, oy + H * res]
bg = np.where(occ, 0.15, np.where(unknown, 0.92, 1.0))

def base(ax, title):
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper", interpolation="nearest")
    for k, (x, y) in OBS.items():
        ax.plot(x, y, marker="s", ms=9, mfc="#f28e2b", mec="k", ls="")
    ax.plot(*DYN_BLOCK, marker="s", ms=8, mfc="#59a14f", mec="k", ls="")
    ax.plot(*DYN_STANDBY, marker="s", ms=8, mfc="none", mec="#59a14f", ls="")
    ax.plot([DYN2_SEG[0][0], DYN2_SEG[1][0]], [DYN2_SEG[0][1], DYN2_SEG[1][1]], color="#59a14f", lw=4, solid_capstyle="butt")
    ax.set_xlim(-13.5, 10.5); ax.set_ylim(-9.5, 8.5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11); ax.grid(alpha=.2)

fig, axs = plt.subplots(1, 2, figsize=(20, 8.5))
ax = axs[0]
base(ax, f"Planned-route traffic: {len(ok)} routes ({heat_cov[nav].mean():.0%} of navigable area touched)")
hm = np.ma.masked_where(heat == 0, heat)
from scipy.ndimage import maximum_filter
hm = np.ma.masked_where(maximum_filter(heat, 3) == 0, maximum_filter(heat, 3))
im = ax.imshow(hm, cmap="inferno_r", extent=ext, origin="upper", interpolation="nearest",
               norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(2, heat.max())))
plt.colorbar(im, ax=ax, fraction=.03, label="routes through cell")
ax = axs[1]
base(ax, "Start (o) and goal (x) by difficulty class")
col = {"easy": "#4e79a7", "difficult": "#e15759"}
for c in ("difficult", "easy"):
    P = [p for p in poses if p["difficulty"]["class"] == c]
    ax.scatter([p["start"]["x"] for p in P], [p["start"]["y"] for p in P], s=14, c=col[c], alpha=.7, label=f"{c} start ({len(P)})")
    ax.scatter([p["goal"]["x"] for p in P], [p["goal"]["y"] for p in P], s=14, c=col[c], marker="x", alpha=.5)
ax.legend(loc="lower left", fontsize=9)
fig.suptitle(f"Campaign-2 pool ({N} pairs): coverage. Orange = statics, green = obs_dynamic_1 block (filled) / standby (hollow), green bar = obs_dynamic_2 board")
fig.tight_layout(); fig.savefig(os.path.join(out, "pool500_coverage.png"), dpi=110); plt.close(fig)

# routes coloured by their tightest point
fig, ax = plt.subplots(figsize=(12, 9))
base(ax, "Planned routes coloured by tightest clearance (red < carry radius 0.847 m)")
cmap = plt.get_cmap("RdYlGn")
for r, xy in zip(rec, paths):
    if r is None:
        continue
    v = np.clip((r["min_clr"] - NAV_R) / (1.3 - NAV_R), 0, 1)
    ax.plot(xy[:, 0], xy[:, 1], color=cmap(v), lw=.7, alpha=.45)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(NAV_R, 1.3))
cb = plt.colorbar(sm, ax=ax, fraction=.03); cb.set_label("route min clearance [m]")
cb.ax.axhline(CARRY_R, color="k", lw=1.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "pool500_routes.png"), dpi=110); plt.close(fig)

# distributions
fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
axs = axs.ravel()
axs[0].bar(list(tags), list(tags.values()), color="#9c755f"); axs[0].set_title("difficulty tags (label, straight segment)")
axs[0].tick_params(axis="x", rotation=30)
nt = sorted(ntag); axs[1].bar([str(k) for k in nt], [ntag[k] for k in nt], color="#76b7b2"); axs[1].set_title("tags per pair")
axs[2].hist(d, bins=20, color="#bab0ac", alpha=.9, label="straight"); axs[2].hist(A["L"], bins=30, histtype="step", color="k", lw=1.5, label="planned")
axs[2].axvline(8.1, ls="--", c="r", label="guard+0.4 @H=12"); axs[2].legend(fontsize=8); axs[2].set_title("route length [m]")
axs[3].hist(A["detour"], bins=30, color="#edc948"); axs[3].set_title("detour ratio planned/straight")
axs[4].hist(A["min_clr"], bins=30, color="#59a14f"); axs[4].axvline(CARRY_R, c="r", ls="--", label="carry radius"); axs[4].axvline(CARRY_W/2, c="m", ls=":", label="carry half-width")
axs[4].legend(fontsize=8); axs[4].set_title("route min clearance [m]")
axs[5].hist(A["len_below_carry"], bins=30, color="#e15759"); axs[5].set_title("route metres with clearance < 0.847")
axs[6].hist(A["tot_turn"], bins=30, color="#af7aa1"); axs[6].set_title("total heading change along route [deg]")
dd = np.minimum.reduce([A["d_obs_corner_mid"], A["d_obs_corner_west"], A["d_dyn_block"]])
axs[7].hist([A["d_obs_corner_mid"], A["d_obs_corner_west"], A["d_dyn_block"], A["d_dyn2_board"]], bins=np.arange(0, 12.5, .5),
            label=["corner_mid", "corner_west", "dyn_1 (block pos)", "dyn_2 board"], color=["#f28e2b", "#ff9da7", "#59a14f", "#4e79a7"])
axs[7].axvline(1.5, c="k", ls="--"); axs[7].legend(fontsize=8); axs[7].set_title("route closest approach to new obstacles [m]")
fig.suptitle(f"Campaign-2 pool ({N} pairs): difficulty distributions (planned = approx. Nav2-like grid planner)")
fig.tight_layout(); fig.savefig(os.path.join(out, "pool500_distributions.png"), dpi=100); plt.close(fig)
print("figures written to", out)
