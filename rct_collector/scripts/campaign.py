"""Campaign folder layout, input fingerprinting and the audit trail.

One campaign = one directory, and nothing else ever writes into it:

  <campaign>/
    MANIFEST.json        identity: sha256 of every input that shapes a probe,
                         code epochs, pilot approval
    frozen/epoch<k>/     byte copies of those inputs, as they were run
    raw/                 the collector's output_dir. APPEND-ONLY:
                           rct_results.csv   one row per ATTEMPT (run_id, trial_id)
                           checkpoint.json   next probe of the main sequence
                           trials/*.json     probe summary + lean trajectory
                           telemetry/*.npz   rich 20 Hz telemetry
                           rct.log
    supervisor/          events.jsonl (every decision), quarantine.jsonl,
                         health.jsonl, status.json, stack_<n>.log,
                         pilot_report.md
    dataset/             DERIVED by build_dataset.py; delete and rebuild freely

Rows are never edited or deleted. A probe the supervisor rejects is listed in
quarantine.jsonl and re-run; the dataset builder picks one attempt per probe.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import socket
import time
from datetime import datetime
from typing import Optional

import yaml

ONLINE_TUNER = "/home/forough/phd_projects/online_tuner"


class CampaignPaths:
    def __init__(self, campaign_dir: str):
        self.root = os.path.abspath(campaign_dir)
        self.raw = os.path.join(self.root, "raw")
        self.frozen = os.path.join(self.root, "frozen")
        self.sup = os.path.join(self.root, "supervisor")
        self.dataset = os.path.join(self.root, "dataset")
        self.manifest = os.path.join(self.root, "MANIFEST.json")
        self.events = os.path.join(self.sup, "events.jsonl")
        self.quarantine = os.path.join(self.sup, "quarantine.jsonl")
        self.health = os.path.join(self.sup, "health.jsonl")
        self.status = os.path.join(self.sup, "status.json")
        self.alerts = os.path.join(self.sup, "alerts.jsonl")
        self.results_csv = os.path.join(self.raw, "rct_results.csv")
        self.checkpoint = os.path.join(self.raw, "checkpoint.json")

    def make(self):
        for d in (self.root, self.raw, self.frozen, self.sup):
            os.makedirs(d, exist_ok=True)


# ── jsonl helpers ────────────────────────────────────────────────────────────
def append_jsonl(path: str, rec: dict) -> dict:
    rec = {"t": datetime.now().isoformat(timespec="seconds"), **rec}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    from rct_collector.durable import fsync_dir
    fsync_dir(path)


def read_checkpoint(paths: CampaignPaths) -> int:
    try:
        with open(paths.checkpoint) as f:
            return int(json.load(f).get("completed_trials", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


# ── input fingerprinting ─────────────────────────────────────────────────────
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _share(pkg: str) -> Optional[str]:
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(pkg)
    except Exception:   # noqa: BLE001
        return None


def _module_dir(mod: str) -> Optional[str]:
    import importlib.util
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def campaign_inputs(collector_cfg_path: str, world_name: str) -> list:
    """[(name, installed_path or None, src_path or None)] for every file whose
    content shapes a probe. The INSTALLED copy is what runs; the src copy is
    only compared against it, to catch a forgotten colcon build."""
    with open(collector_cfg_path) as f:
        ccfg = yaml.safe_load(f) or {}
    src = os.path.join(ONLINE_TUNER, "src")
    items = [("collector_config", collector_cfg_path, None)]

    map_yaml = ccfg.get("map_yaml_path")
    if map_yaml:
        items.append(("map_yaml", map_yaml, None))
        with open(map_yaml) as f:
            img = (yaml.safe_load(f) or {}).get("image", "")
        items.append(("map_image", os.path.join(os.path.dirname(map_yaml), img), None))
    if ccfg.get("presampled_poses_path"):
        items.append(("pose_pool", ccfg["presampled_poses_path"], None))
    if ccfg.get("presampled_configs_path"):
        items.append(("config_pool", ccfg["presampled_configs_path"], None))

    gs, pgw, nav = _share("gazebo_simulation"), _share("pal_gazebo_worlds"), _share("pmb2_2dnav")
    items += [
        ("launch_file", gs and os.path.join(gs, "launch", "gazebo_with_navigation.launch.py"),
         os.path.join(src, "gazebo_simulation", "launch", "gazebo_with_navigation.launch.py")),
        ("sim_map_image", gs and os.path.join(gs, "maps", world_name, "map.pgm"),
         os.path.join(src, "gazebo_simulation", "maps", world_name, "map.pgm")),
        ("world", pgw and os.path.join(pgw, "worlds", f"{world_name}.world"),
         os.path.join(src, "pal_gazebo_worlds", "worlds", f"{world_name}.world")),
        ("nav_params", nav and os.path.join(nav, "config", "nav_campaign2.yaml"),
         os.path.join(src, "pmb2_navigation", "pmb2_2dnav", "config", "nav_campaign2.yaml")),
    ]
    # The robot itself. ros2-tiago-dev's tiago.urdf.xacro was edited (2026-09-09)
    # to attach a 25 cm payload_box to gripper_link; a container without that
    # edit simulates a different robot (different arm reach, contacts and
    # MoveIt behaviour) and must not feed the same campaign.
    td = _share("tiago_description")
    xacro = td and os.path.join(td, "robots", "tiago.urdf.xacro")
    items.append(("robot_xacro", xacro, None))
    if xacro and os.path.exists(xacro):
        import re as _re
        for inc in _re.findall(r'filename="(/[^"]+)"', open(xacro).read()):
            items.append((f"robot_include/{os.path.basename(inc)}", inc, None))
    oct_dir = _module_dir("online_causal_tuner.dynamic_obstacle_controller")
    items.append(("obstacle_controller",
                  oct_dir and os.path.join(oct_dir, "dynamic_obstacle_controller.py"),
                  os.path.join(src, "online_causal_tuner", "online_causal_tuner",
                               "dynamic_obstacle_controller.py")))
    rc_dir = _module_dir("rct_collector")
    rc_src = os.path.join(src, "rct_data_collector", "rct_collector")
    for f in sorted(glob.glob(os.path.join(rc_src, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(f, rc_src)
        items.append((f"code/rct_collector/{rel}",
                      rc_dir and os.path.join(rc_dir, rel), f))
    return items


def fingerprint(items: list) -> tuple:
    """-> (hashes {name: sha}, problems [str]). A problem is a missing input
    or an installed file that differs from src (stale build)."""
    hashes, problems = {}, []
    for name, inst, srcp in items:
        if not inst or not os.path.exists(inst):
            problems.append(f"{name}: installed file missing ({inst})")
            continue
        hashes[name] = sha256(inst)
        if srcp and os.path.exists(srcp) and sha256(srcp) != hashes[name]:
            problems.append(f"{name}: installed copy differs from src -> run colcon build "
                            f"({inst} vs {srcp})")
    # The collector's map and the map the launch file gives AMCL must agree.
    if "map_image" in hashes and "sim_map_image" in hashes \
            and hashes["map_image"] != hashes["sim_map_image"]:
        problems.append("map_image != sim_map_image: the collector's map and Nav2's map differ")
    return hashes, problems


def init_or_verify_manifest(paths: CampaignPaths, items: list, *, allow_change: bool,
                            extra: dict) -> dict:
    """Create the manifest on first use; afterwards require identical inputs.

    A change is refused unless allow_change, in which case a new code epoch is
    opened (inputs frozen again, the diff recorded) so every row stays
    traceable to the exact inputs that produced it."""
    paths.make()
    hashes, problems = fingerprint(items)
    if problems:
        raise RuntimeError("Campaign inputs are not consistent:\n  " + "\n  ".join(problems))

    def _freeze(epoch: int):
        dst = os.path.join(paths.frozen, f"epoch{epoch}")
        for name, inst, _ in items:
            if inst and os.path.exists(inst):
                out = os.path.join(dst, name if "/" in name else
                                   f"{name}__{os.path.basename(inst)}")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(inst, out)
        return dst

    if not os.path.exists(paths.manifest):
        man = {"campaign_dir": paths.root, "created_at": datetime.now().isoformat(),
               "host": socket.gethostname(), **extra,
               "epochs": [{"epoch": 0, "started_at": datetime.now().isoformat(),
                           "frozen": _freeze(0), "hashes": hashes}],
               "pilot": None}
        write_json_atomic(paths.manifest, man)
        append_jsonl(paths.events, {"event": "campaign_created", "epoch": 0})
        return man

    with open(paths.manifest) as f:
        man = json.load(f)
    cur = man["epochs"][-1]
    diff = sorted(k for k in set(hashes) | set(cur["hashes"])
                  if hashes.get(k) != cur["hashes"].get(k))
    if not diff:
        return man
    if not allow_change:
        raise RuntimeError(
            f"Inputs changed since epoch {cur['epoch']} of this campaign: {diff}.\n"
            "Rows from different inputs must not be mixed silently. Either revert, "
            "or rerun with --allow-input-change to open a new epoch (recorded per row).")
    ep = cur["epoch"] + 1
    man["epochs"].append({"epoch": ep, "started_at": datetime.now().isoformat(),
                          "frozen": _freeze(ep), "hashes": hashes, "changed": diff})
    write_json_atomic(paths.manifest, man)
    append_jsonl(paths.events, {"event": "new_epoch", "epoch": ep, "changed": diff})
    return man


def current_epoch(paths: CampaignPaths) -> int:
    try:
        with open(paths.manifest) as f:
            return int(json.load(f)["epochs"][-1]["epoch"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return 0


def set_manifest_field(paths: CampaignPaths, key: str, value) -> None:
    with open(paths.manifest) as f:
        man = json.load(f)
    man[key] = value
    write_json_atomic(paths.manifest, man)


def now_s() -> float:
    return time.time()
