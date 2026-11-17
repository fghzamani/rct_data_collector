#!/usr/bin/env python3
"""Build the campaign-2 map FROM the Gazebo world, so it matches exactly.

Single source of truth: src/pal_gazebo_worlds/worlds/pal_office_campaign2.world.
Every collision box of PAL_office (70 walls) and of the static stubs
(obs_corner_mid, obs_corner_west) is rasterised onto the map grid; moving
obstacles (obs_dynamic_*) are not part of a static map and are left out.
Free = cells reachable from the robot spawn (0, -8.5) without crossing an
obstacle; everything else is unknown.

    python3 src/rct_data_collector/tools/build_campaign2_map.py          # build + verify
    python3 src/rct_data_collector/tools/build_campaign2_map.py --check  # verify only

Writes the same map.pgm to
    maps/pal_office_campaign2/map.pgm                          (collector, pose pool)
    src/gazebo_simulation/maps/pal_office_campaign2/map.pgm    (Nav2 / AMCL via the launch file)
map.yaml is rewritten in both places: 0.05 m cells, grid fitted to the
world's walls plus 1 m (the stock 520 x 469 grid cut off the south and east
of the building, so free space leaked off the edge of the map).

History
  2026-11-12  stubs stamped onto the stock PAL SLAM map, MIRRORED (165.3 / 75.2
              deg instead of 15.6 / 105.6, centres 0.2-0.3 m off).
  2026-11-13  stubs restamped correctly, then found that the stock PAL map
              itself is off from the world by (+0.14, -0.25) m and 1.76 deg,
              and even after that best rigid fit 6.5% of its occupied cells
              match no wall (SLAM noise). Replaced by this world-built map.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

# Workspace root (holds maps/pal_office_campaign2). Moved from the top-level
# analysis/ folder into rct_data_collector/tools/ on 2026-11-17, so it can no
# longer be derived from __file__.
ROOT = os.environ.get("ONLINE_TUNER_WS", "/home/forough/phd_projects/online_tuner")
WORLD = os.path.join(ROOT, "src/pal_gazebo_worlds/worlds/pal_office_campaign2.world")
BASE = os.path.join(ROOT, "maps/pal_office_campaign2/base/pal_office.pgm")   # grid shape only
SPAWN = (0.0, -8.5)
UNKNOWN = 205
OUTS = [os.path.join(ROOT, "maps/pal_office_campaign2/map.pgm"),
        os.path.join(ROOT, "src/gazebo_simulation/maps/pal_office_campaign2/map.pgm")]
STATIC = ("obs_corner_mid", "obs_corner_west")
RES = 0.05
OX, OY = -14.2, -12.2        # replaced in main() by a grid fitted to the world
MARGIN_M = 1.0               # grid extends this far beyond the outermost wall
OCCUPIED, FREE = 0, 254


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    toks, i = [], 0
    while len(toks) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        toks.append(data[i:j])
        i = j
    w, h = int(toks[1]), int(toks[2])
    return np.frombuffer(data[i + 1:i + 1 + w * h], np.uint8).reshape(h, w).copy()


def write_pgm(path, img):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        f.write(img.astype(np.uint8).tobytes())
    os.replace(tmp, path)


def world_boxes():
    s = open(WORLD).read()
    out = {}
    for name in STATIC + ("obs_dynamic_1", "obs_dynamic_2"):
        m = re.search(rf"<model name='{name}'>(.*?)</model>", s, re.S)
        if not m:
            sys.exit(f"{name} not found in {WORLD}")
        pose = [float(v) for v in re.search(r"<pose>([^<]+)</pose>", m.group(1)).group(1).split()]
        size = [float(v) for v in re.search(r"<size>([^<]+)</size>", m.group(1)).group(1).split()]
        out[name] = dict(x=pose[0], y=pose[1], yaw=pose[5], L=size[0], W=size[1])
    return out


def _pose(el):
    p = el.find("pose") if el is not None else None
    v = [float(x) for x in p.text.split()] if p is not None and p.text else [0.0] * 6
    return v[0], v[1], v[5]


def _comp(a, b):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], t + b[2]


def world_collision_boxes():
    """[(model, x, y, yaw, L, W)] of every static box collision in the world,
    composed model pose * link pose * collision pose (planar)."""
    import xml.etree.ElementTree as ET
    w = ET.parse(WORLD).getroot().find("world")
    out = []
    for m in w.findall("model"):
        name = m.get("name")
        if name == "ground_plane" or name.startswith("obs_dynamic"):
            continue
        mp = _pose(m)
        for link in m.findall("link"):
            lp = _comp(mp, _pose(link))
            for col in link.findall("collision"):
                b = col.find("geometry/box/size")
                if b is None:
                    if col.find("geometry") is not None and len(col.find("geometry")):
                        sys.exit(f"{name}/{link.get('name')}: non-box collision not supported")
                    continue
                sx, sy, _ = map(float, b.text.split())
                cp = _comp(lp, _pose(col))
                out.append((name, cp[0], cp[1], cp[2], sx, sy))
    return out


def rasterise(shape, boxes, pad):
    """Cells whose CENTRE lies within `pad` of a box (pad = half a cell makes
    every box at least one cell thick, so thin walls cannot leak)."""
    m = np.zeros(shape, bool)
    x, y = cell_centres(shape)
    for _, bx, by, yaw, L, W in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x - bx, y - by
        u, v = c * dx + s * dy, -s * dx + c * dy
        m |= (np.abs(u) <= L / 2 + pad) & (np.abs(v) <= W / 2 + pad)
    return m


def world_grid(boxes):
    """(origin_x, origin_y, width, height) covering every box corner + margin."""
    xs, ys = [], []
    for _, bx, by, yaw, L, W in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        for u in (-L / 2, L / 2):
            for v in (-W / 2, W / 2):
                xs.append(bx + c * u - s * v)
                ys.append(by + s * u + c * v)
    ox = math.floor((min(xs) - MARGIN_M) / RES) * RES
    oy = math.floor((min(ys) - MARGIN_M) / RES) * RES
    w = int(math.ceil((max(xs) + MARGIN_M - ox) / RES))
    h = int(math.ceil((max(ys) + MARGIN_M - oy) / RES))
    return round(ox, 3), round(oy, 3), w, h


def write_yaml(pgm_path):
    with open(os.path.join(os.path.dirname(pgm_path), "map.yaml"), "w") as f:
        f.write(f"image: map.pgm\nmode: trinary\nresolution: {RES}\n"
                f"origin: [{OX}, {OY}, 0]\nnegate: 0\noccupied_thresh: 0.65\n"
                f"free_thresh: 0.25\n"
                f"# built from pal_office_campaign2.world by src/rct_data_collector/tools/build_campaign2_map.py\n")


def build_map(shape):
    from scipy import ndimage
    boxes = world_collision_boxes()
    occ = rasterise(shape, boxes, RES / 2)
    lab, _ = ndimage.label(~occ)
    h = shape[0]
    r = h - 1 - int((SPAWN[1] - OY) / RES)
    c = int((SPAWN[0] - OX) / RES)
    free = lab == lab[r, c]
    if free[0, :].any() or free[-1, :].any() or free[:, 0].any() or free[:, -1].any():
        sys.exit("free space reaches the map border: the building is not closed "
                 "or the grid is too small")
    img = np.full(shape, UNKNOWN, np.uint8)
    img[free] = FREE
    img[occ] = OCCUPIED
    return img, boxes


def cell_centres(shape):
    h, w = shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    x = OX + (cols + 0.5) * RES
    y = OY + (h - 1 - rows + 0.5) * RES      # row 0 is the TOP of the image = max y
    return x, y


def box_mask(shape, b):
    x, y = cell_centres(shape)
    dx, dy = x - b["x"], y - b["y"]
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (np.abs(u) <= b["L"] / 2) & (np.abs(v) <= b["W"] / 2)


def blob_stats(mask, shape):
    x, y = cell_centres(shape)
    xs, ys = x[mask], y[mask]
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1])) % 180
    return xs.mean(), ys.mean(), ang, int(mask.sum())


def verify_stubs(img, boxes) -> bool:
    """Each static stub occupies its world box, centred and oriented as in
    the world; the moving obstacles are absent."""
    ok = True
    for name in STATIC:
        b = boxes[name]
        m = box_mask(img.shape, b)
        cx, cy, ang, n = blob_stats(m & (img == OCCUPIED), img.shape)
        want = math.degrees(b["yaw"]) % 180
        dang = min(abs(ang - want), 180 - abs(ang - want))
        good = math.hypot(cx - b["x"], cy - b["y"]) < 0.05 and dang < 3.0 \
            and (img[m] == OCCUPIED).all()
        ok &= good
        print(f"  {name}: world ({b['x']:.2f}, {b['y']:.2f}) {want:.1f} deg | "
              f"map ({cx:.2f}, {cy:.2f}) {ang:.1f} deg  {'OK' if good else 'WRONG'}")
    for name in ("obs_dynamic_1", "obs_dynamic_2"):
        b = dict(boxes[name], L=boxes[name]["L"] - 0.2, W=max(boxes[name]["W"] - 0.2, 0.05))
        core = box_mask(img.shape, b)
        if core.any() and (img[core] == OCCUPIED).all():
            ok = False
            print(f"  {name} is in the map; moving obstacles must not be  WRONG")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    global OX, OY
    boxes = world_boxes()
    OX, OY, w, h = world_grid([b for b in world_collision_boxes() if b[0] == "PAL_office"])
    shape = (h, w)
    print(f"grid: origin ({OX}, {OY}), {w} x {h} cells = {w * RES:.1f} x {h * RES:.1f} m")
    ref, wboxes = build_map(shape)
    print(f"world: {len(wboxes)} collision boxes "
          f"({sorted({b[0] for b in wboxes})}); free {int((ref == FREE).sum() * RES * RES)} m^2")
    if not a.check:
        for p in OUTS:
            write_pgm(p, ref)
            write_yaml(p)
            print(f"wrote {p} (+ map.yaml)")
    allok = True
    for p in OUTS:
        img = read_pgm(p)
        same = np.array_equal(img, ref)
        print(f"{p}\n  identical to the world-built map: {same}")
        allok &= same
        allok &= verify_stubs(img, boxes)
    print("ALL OK" if allok else "MISMATCH")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
