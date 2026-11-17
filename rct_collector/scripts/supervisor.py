#!/usr/bin/env python3
"""Deterministic supervisor for an RCT campaign: pilot in the GUI, then headless.

    ros2 run rct_collector rct_supervise pilot      # 10 probes, Gazebo + RViz visible
    ros2 run rct_collector rct_supervise approve --note "looked right"
    ros2 run rct_collector rct_supervise run        # the rest, headless, unattended
    ros2 run rct_collector rct_supervise status     # from any shell
    ros2 run rct_collector rct_supervise stop       # graceful: finish probe, shut down

(all take --config, default: the installed config/campaign2_supervisor.yaml)

What it does, and why each piece exists
---------------------------------------
* Owns the whole stack. It starts YOUR launch file (gazebo_with_navigation) and
  the collector as child processes tagged with an env marker, and only ever
  kills processes carrying that marker -- never another Gazebo or container.
* Refuses to start on a dirty machine (a Gazebo already on port 11345, or a
  /clock already on this ROS domain) and on inconsistent inputs (installed
  files differ from src = forgotten colcon build; the collector's map differs
  from Nav2's; any input changed since the campaign started -- see campaign.py).
* Runs the collector in chunks (--stop-at), so every few hundred probes the
  stack is recycled before Gazebo drift accumulates, and a crash costs at most
  the probe in flight (the checkpoint advances per probe).
* Live checks every second: sim clock stalled, real-time factor collapsed,
  a core node died, no new probe for too long, memory running out.
  -> stop the collector (gracefully if possible), kill the stack, restart it,
     resume from the probe that was in flight.
* Per-probe checks on every new row (outcome-INDEPENDENT signals only):
  teleport failed, collision channel silent, telemetry streams missing,
  telemetry rate / sim gaps / real-time factor bad BEFORE the first contact,
  footprint out of sync, treatment not applied, short watch window,
  localisation lost before the treatment.
  -> the attempt is quarantined (never deleted) and the probe re-run with the
     same pose, configuration and apply time, at most max_reruns times.
  A collision slows the physics engine, so lag measured AFTER the first contact
  is deliberately ignored: re-running those probes would select against
  collisions. build_dataset reports the outcome rates of quarantined attempts
  so this can be checked.
* Rolling statistics every few probes: footprint delivery and treatment balance
  breaking -> HALT (data would be worthless); collision / stall / validity /
  speed-delivery outside the target bands -> warnings in status.json.
* Escalation: too many restarts per hour, repeated bring-up failures, the same
  probe crashing repeatedly, repeated integrity failures -> HALT: everything is
  shut down cleanly and an alert is written to supervisor/alerts.jsonl. A HALT
  needs a human (or the analyst agent) to look; `run` continues afterwards.

Everything it decides is appended to supervisor/events.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import yaml

from rct_collector.scripts.data_audit import DataAudit, HALT as AUDIT_HALT, WARN as AUDIT_WARN
from rct_collector.scripts.campaign import (CampaignPaths, append_jsonl, campaign_inputs,
                                            current_epoch, init_or_verify_manifest,
                                            read_checkpoint, read_jsonl, set_manifest_field,
                                            write_json_atomic)

csv.field_size_limit(sys.maxsize)
log = logging.getLogger("rct_supervisor")

TAG_ENV = "RCT_SUPERVISOR_TAG"
HALT_CODE = 10


def _f(v, default=None):
    try:
        x = float(v)
        return x if x == x else default
    except (TypeError, ValueError):
        return default


def _truthy(v) -> bool:
    return str(v).strip() in ("1", "1.0", "True", "true")


def _mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _memory_diagnosis() -> dict:
    """RAM, swap and kernel memory-pressure (PSI) figures. PSI 'full' is the
    share of time every task was stalled on memory: above a few percent the
    simulator's real-time factor collapses no matter what this code does."""
    d = {"mem_available_mb": round(_mem_available_mb())}
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: float(l.split()[1]) / 1024 for l in f}
        d["swap_free_mb"] = round(mi.get("SwapFree", 0))
        d["swap_total_mb"] = round(mi.get("SwapTotal", 0))
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                kind, *kv = line.split()
                vals = dict(x.split("=") for x in kv)
                d[f"psi_mem_{kind}_avg60"] = float(vals.get("avg60", "nan"))
    except (OSError, ValueError):
        pass
    return d


def _memory_starved(d: dict) -> bool:
    return d.get("psi_mem_full_avg60", 0.0) > 3.0 or (
        d.get("mem_available_mb", 1e9) < 400 and d.get("swap_free_mb", 1e9) < 200)


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class Halt(Exception):
    """Unrecoverable without a human: shut everything down and stop."""


class StopRequested(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
class ClockWatch:
    """Tiny rclpy node: /clock age and real-time factor over a sliding window."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("rct_supervisor_watch")
        self._hist = deque(maxlen=4000)        # (wall, sim)
        self._last_wall = 0.0
        self._lock = threading.Lock()
        self.node.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        # The base laser: without scans AMCL never publishes map->odom and Nav2
        # cannot activate. Gazebo's laser needs a rendering context (X display);
        # 2026-11-13 after a host reboot the X server no longer accepted the
        # container's root user, the laser went silent and the stack restarted
        # in a loop with no useful message.
        from sensor_msgs.msg import LaserScan
        self._last_scan_wall = 0.0
        self.node.create_subscription(LaserScan, "/scan_raw", self._on_scan, qos_profile_sensor_data)
        self._ex = SingleThreadedExecutor()
        self._ex.add_node(self.node)
        threading.Thread(target=self._ex.spin, daemon=True).start()

    def _on_clock(self, msg):
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        now = time.time()
        with self._lock:
            if not self._hist or sim > self._hist[-1][1]:
                self._hist.append((now, sim))
                self._last_wall = now

    def _on_scan(self, _msg):
        self._last_scan_wall = time.time()

    def scan_age(self) -> float:
        return (time.time() - self._last_scan_wall) if self._last_scan_wall else float("inf")

    def reset(self):
        with self._lock:
            self._hist.clear()
            self._last_wall = time.time()   # grace: age counts from the reset

    def clock_age(self) -> float:
        with self._lock:
            return (time.time() - self._last_wall) if self._last_wall else float("inf")

    def rtf(self, window_s: float) -> float:
        with self._lock:
            if len(self._hist) < 2:
                return float("nan")
            now = time.time()
            old = next(((w, s) for w, s in self._hist if now - w <= window_s), None)
            w1, s1 = self._hist[-1]
            if old is None or w1 - old[0] < 0.5 * window_s:
                return float("nan")
            return (s1 - old[1]) / (w1 - old[0])

    def has_messages(self) -> bool:
        with self._lock:
            return len(self._hist) > 0

    def clock_publishers(self) -> int:
        return self.node.count_publishers("/clock")

    def has_action(self, name: str) -> bool:
        from rclpy.action import get_action_names_and_types
        return any(n == name for n, _ in get_action_names_and_types(self.node))

    def has_topic_publisher(self, topic: str) -> bool:
        return self.node.count_publishers(topic) > 0


# ─────────────────────────────────────────────────────────────────────────────
class Supervisor:
    def __init__(self, cfg: dict, cfg_path: str, allow_input_change: bool = False):
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.allow_input_change = allow_input_change
        self.P = CampaignPaths(cfg["campaign_dir"])
        self.P.make()
        self.H = cfg.get("health", {})
        self.collector_cfg = cfg["collector_config"]
        with open(self.collector_cfg) as f:
            self.ccfg = yaml.safe_load(f) or {}
        self.num_trials = int(self.ccfg["num_trials"])
        self.tag = hashlib.sha1(self.P.root.encode()).hexdigest()[:12]
        self.stack = None              # Popen of ros2 launch
        self.stack_log = None
        self.stack_log_path = None
        self.stack_started = 0.0
        self.stack_chunks = 0
        self.stack_gui = False
        self.collector = None
        self.restarts = deque()        # wall times of unplanned restarts
        self.bringup_failures = 0
        self.integrity_exits = 0
        self.exception_probe = (None, 0)
        self.rows_seen = 0
        self.recent_verdicts = deque(maxlen=int(self.H.get("degraded_window", 5)))
        self.last_row_wall = time.time()
        self.warnings = {}
        self.phase = "init"
        self.last_decision = ""
        self.watch = None
        self.pending_restart = ""
        self.deadline = None
        self.mem_at_ready = _mem_available_mb()
        self.collector_first_row = False
        self.collector_stop_requested = False
        self._stack_log_pos = 0
        self._rows_cache = (None, [])
        # Rows already judged by an earlier supervisor process are not judged
        # again (they would be quarantined twice).
        self.rows_seen = len(self.read_rows())
        self._stop_flag = os.path.join(self.P.sup, "STOP")
        # Data audit: per-probe checks (audit_probes.jsonl, merged into the
        # dataset) and campaign-level checks every stats_every probes.
        self.audit = DataAudit(self.P.root, self.ccfg, self.H)
        self.audit_path = os.path.join(self.P.sup, "audit_probes.jsonl")
        self.audits = {(str(a.get("run_id")), int(a.get("trial_id", -1))): a
                       for a in read_jsonl(self.audit_path)}
        self._audit_alerted = {}
        self._n_stack = len([f for f in os.listdir(self.P.sup) if f.startswith("stack_")])
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        self._signalled = False

    # ── bookkeeping ──────────────────────────────────────────────────────
    def event(self, event: str, **kw):
        rec = append_jsonl(self.P.events, {"event": event, "phase": self.phase, **kw})
        self.last_decision = f"{rec['t']} {event}"
        log.info(f"[{event}] " + ", ".join(f"{k}={v}" for k, v in kw.items()))

    def alert(self, level: str, msg: str, **kw):
        append_jsonl(self.P.alerts, {"level": level, "msg": msg, "phase": self.phase, **kw})
        (log.error if level == "halt" else log.warning)(f"ALERT[{level}] {msg}")

    def _on_signal(self, signum, frame):
        self._signalled = True
        log.warning(f"signal {signum}: finishing the current probe, then shutting down")

    def stop_requested(self) -> bool:
        if self._signalled or os.path.exists(self._stop_flag):
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            if not getattr(self, "_deadline_logged", False):
                self._deadline_logged = True
                self.event("deadline_reached",
                           at=datetime.fromtimestamp(self.deadline).isoformat(timespec="minutes"))
            return True
        return False

    def user_stop(self) -> bool:
        """A stop the USER asked for (signal / STOP file), as opposed to the
        time budget running out."""
        return self._signalled or os.path.exists(self._stop_flag)

    def write_status(self, **extra):
        ck = read_checkpoint(self.P)
        st = {"updated": datetime.now().isoformat(timespec="seconds"), "phase": self.phase,
              "checkpoint": ck, "num_trials": self.num_trials,
              "progress": round(ck / max(self.num_trials, 1), 4),
              "rows_seen": self.rows_seen,
              "collector_pid": self.collector.pid if self.collector else None,
              "stack_pid": self.stack.pid if self.stack else None,
              "stack_gui": self.stack_gui,
              "rtf_30s": round(self.watch.rtf(30), 3) if self.watch else None,
              "clock_age_s": round(min(self.watch.clock_age(), 1e6), 1) if self.watch else None,
              "mem_available_mb": round(_mem_available_mb()),
              "unplanned_restarts_last_hour": self._restarts_last_hour(),
              "pending_reruns": self.pending_reruns(),
              "warnings": self.warnings, "last_decision": self.last_decision,
              "epoch": current_epoch(self.P), **extra}
        write_json_atomic(self.P.status, st)

    def _restarts_last_hour(self) -> int:
        now = time.time()
        while self.restarts and now - self.restarts[0] > 3600:
            self.restarts.popleft()
        return len(self.restarts)

    # ── preflight ────────────────────────────────────────────────────────
    def preflight(self):
        man = init_or_verify_manifest(
            self.P, campaign_inputs(self.collector_cfg, self.ccfg.get("world_name", "")),
            allow_change=self.allow_input_change,
            extra={"collector_config": self.collector_cfg, "supervisor_config": self.cfg_path,
                   "num_trials": self.num_trials, "seed": self.ccfg.get("seed")})
        out_dir = os.path.abspath(self.ccfg.get("output_dir", ""))
        if out_dir != self.P.raw:
            log.info(f"collector output_dir {out_dir} overridden by --output {self.P.raw}")
        return man

    def machine_is_clean(self):
        need = float(self.H.get("mem_preflight_min_mb", 900))
        t0 = time.time()
        mem = _mem_available_mb()
        while mem < need and time.time() - t0 < 90:   # a just-stopped stack frees RAM slowly
            time.sleep(5)
            mem = _mem_available_mb()
        if mem < need:
            raise Halt(f"only {mem:.0f} MB RAM available before starting the simulator "
                       f"(need {need:.0f}; Gazebo + Nav2 + collector take ~800 MB). Close "
                       "browsers / Teams / IDEs on the host, then retry.")
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", 11345)) == 0:
                raise Halt("A Gazebo server is already listening on port 11345 (another "
                           "simulation on this machine or in another --net=host container). "
                           "Close it first; two Gazebos would share topics and ports.")
        # Judge by /clock MESSAGES, not publishers: a gzserver killed with -9
        # stays in the DDS graph until its lease expires, which made the first
        # restart after a crash refuse to start.
        self.watch.reset()
        time.sleep(3.0)
        if self.watch.has_messages():
            raise Halt(f"/clock is being published on ROS_DOMAIN_ID="
                       f"{os.environ.get('ROS_DOMAIN_ID', '0')}: another simulation is running.")

    def memory_preflight(self):
        md = _memory_diagnosis()
        self.event("memory_preflight", **md)
        if _memory_starved(md) and not self.H.get("ignore_memory_pressure", False):
            raise Halt(f"The host is memory-starved before the simulator even starts ({md}). "
                       "Gazebo would run far below real time and every probe would be laggy. "
                       "Free RAM on the host (browser tabs, Teams, Spotify, extra IDE windows) "
                       "until swap has room and psi_mem_full_avg60 is ~0, then retry.")

    # ── processes ────────────────────────────────────────────────────────
    def _env(self):
        env = dict(os.environ)
        env[TAG_ENV] = self.tag
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def tagged_pids(self) -> list:
        """(pid, comm) of every live process carrying this campaign's tag."""
        me = os.getpid()
        out = []
        needle = f"{TAG_ENV}={self.tag}".encode()
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) == me:
                continue
            try:
                with open(f"/proc/{d}/environ", "rb") as f:
                    if needle not in f.read():
                        continue
                with open(f"/proc/{d}/comm") as f:
                    out.append((int(d), f.read().strip()))
            except OSError:
                continue
        return out

    def sweep(self, sig=signal.SIGKILL):
        for pid, comm in self.tagged_pids():
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    def start_stack(self, gui: bool):
        L = self.cfg["launch"]
        args = dict(L.get("args", {}))
        args.update(L.get("gui_args" if gui else "headless_args", {}))
        cmd = ["ros2", "launch", L["package"], L["file"]] + [f"{k}:={v}" for k, v in args.items()]
        self._n_stack += 1
        self.stack_log_path = os.path.join(self.P.sup, f"stack_{self._n_stack:03d}.log")
        self.stack_log = open(self.stack_log_path, "ab")
        self.watch.reset()
        self.machine_is_clean()
        self.event("stack_start", gui=gui, cmd=" ".join(cmd), log=self.stack_log_path)
        self.stack = subprocess.Popen(cmd, stdout=self.stack_log, stderr=subprocess.STDOUT,
                                      env=self._env(), start_new_session=True)
        self.stack_gui, self.stack_started, self.stack_chunks = gui, time.time(), 0
        self._stack_log_pos = 0
        t0, timeout = time.time(), float(L.get("ready_timeout_sec", 300))
        while True:
            if self.stack.poll() is not None:
                break
            if (self.watch.clock_age() < 2.0 and self.watch.has_action("/navigate_to_pose")
                    and self.watch.has_topic_publisher("/gazebo/model_states")):
                time.sleep(float(L.get("settle_sec", 20)))
                if self.watch.scan_age() > 5.0:
                    self.event("stack_no_laser", scan_age=self.watch.scan_age())
                    self.stop_stack()
                    raise Halt("Gazebo is running but the base laser (/scan_raw) publishes "
                               "nothing, so AMCL cannot localise and Nav2 cannot activate. "
                               "Gazebo's laser needs the X display: after a reboot run on the "
                               "host `xhost +SI:localuser:root` (the container runs as root), "
                               "then resume. A restart will not fix this.")
                cfg_err = self.stack_config_error()
                if cfg_err:
                    self.event("stack_config_error", lines=cfg_err)
                    self.stop_stack()
                    raise Halt(f"Nav2 failed to configure (a restart will not fix this): {cfg_err}")
                if self.watch.clock_age() < 2.0 and self.stack.poll() is None:
                    self.bringup_failures = 0
                    rtf10 = self.watch.rtf(10)
                    # Judge the real-time factor from here on: Gazebo's own
                    # start-up (sim clock crawling while models load) is not lag.
                    self.watch.reset()
                    self.mem_at_ready = _mem_available_mb()
                    self.event("stack_ready", after_s=round(time.time() - t0, 1),
                               rtf=round(rtf10, 3), **_memory_diagnosis())
                    return
            if time.time() - t0 > timeout or self.stop_requested():
                break
            time.sleep(1.0)
        self.bringup_failures += 1
        self.event("stack_bringup_failed", attempt=self.bringup_failures,
                   exit=self.stack.poll(), log=self.stack_log_path)
        self.stop_stack()
        if self.stop_requested():
            raise StopRequested()
        if self.bringup_failures >= int(self.H.get("max_bringup_failures", 3)):
            raise Halt(f"stack failed to come up {self.bringup_failures} times; "
                       f"see {self.stack_log_path}")
        time.sleep(10)
        self.start_stack(gui)

    def stop_stack(self):
        if self.stack is not None:
            self.event("stack_stop", pid=self.stack.pid,
                       uptime_s=round(time.time() - self.stack_started))
            try:
                os.killpg(self.stack.pid, signal.SIGINT)
            except OSError:
                pass
            try:
                self.stack.wait(timeout=25)
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(self.stack.pid, sig)
                    except OSError:
                        pass
                    try:
                        self.stack.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            self.stack = None
        time.sleep(2)
        left = self.tagged_pids()
        if left:
            self.event("sweep_leftovers", procs=[c for _, c in left][:20])
            self.sweep(signal.SIGKILL)
            time.sleep(2)
        if self.stack_log:
            self.stack_log.close()
            self.stack_log = None
        if self.watch:
            self.watch.reset()

    def restart_stack(self, why: str, planned: bool = False):
        self.event("stack_restart", why=why, planned=planned)
        if not planned:
            self.restarts.append(time.time())
            n = self._restarts_last_hour()
            if n > int(self.H.get("max_restarts_per_hour", 4)):
                md = _memory_diagnosis()
                hint = (f" The host is MEMORY-STARVED ({md}): Gazebo slows down because the "
                        "machine is swapping, and restarting cannot fix that. Free RAM on the "
                        "host (browser, Teams, Spotify, extra IDE windows), then `run` again."
                        if _memory_starved(md) else f" Memory looked fine ({md}).")
                raise Halt(f"{n} unplanned stack restarts in the last hour (last: {why})." + hint)
        gui = self.stack_gui
        self.stop_stack()
        time.sleep(5)
        self.start_stack(gui)

    CONFIG_ERROR_PATTERNS = ("Failed to bring up all requested nodes",
                             "has invalid type", "Original error:",
                             "Error loading XML file")

    def stack_config_error(self) -> str:
        """A Nav2 bring-up/configuration failure in the current launch log. A
        restart cannot fix these (2026-11-13: an integer cost_scaling_factor
        made both costmaps refuse to configure; the collector waited for Nav2,
        timed out and the stack was restarted three times for nothing)."""
        if not self.stack_log_path:
            return ""
        try:
            with open(self.stack_log_path, "rb") as f:
                text = f.read().decode("utf-8", "ignore")
        except OSError:
            return ""
        import re as _re
        hits = [_re.sub(r"\x1b\[[0-9;]*m", "", ln).strip()[:300] for ln in text.splitlines()
                if any(p in ln for p in self.CONFIG_ERROR_PATTERNS)]
        return " | ".join(hits[:3])

    def stack_log_fatal(self) -> str:
        """New 'process has died' lines for a core node in the launch log."""
        if not self.stack_log_path:
            return ""
        fatal = self.cfg["launch"].get("fatal_nodes", [])
        try:
            with open(self.stack_log_path, "rb") as f:
                f.seek(self._stack_log_pos)
                chunk = f.read().decode("utf-8", "ignore")
                self._stack_log_pos = f.tell()
        except OSError:
            return ""
        for line in chunk.splitlines():
            if "process has died" in line:
                if any(n in line for n in fatal):
                    return line.strip()[:300]
                self.event("stack_process_died_nonfatal", line=line.strip()[:300])
        return ""

    # ── collector ────────────────────────────────────────────────────────
    def start_collector(self, *, stop_at=None, rerun=None, kind="rerun"):
        cmd = [sys.executable, "-m", "rct_collector.scripts.main",
               "--config", self.collector_cfg, "--output", self.P.raw]
        if os.path.exists(self.P.checkpoint):
            cmd.append("--resume")
        if rerun:
            cmd += ["--rerun-trials", ",".join(map(str, rerun)), "--attempt-kind", kind]
        elif stop_at:
            cmd += ["--stop-at", str(stop_at)]
        out = open(os.path.join(self.P.raw, "collector_stdout.log"), "ab")
        out.write(f"\n===== {datetime.now().isoformat()} {' '.join(cmd)}\n".encode())
        out.flush()
        self.event("collector_start", stop_at=stop_at, rerun=rerun, kind=kind,
                   checkpoint=read_checkpoint(self.P))
        self.collector = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                          env=self._env(), start_new_session=True)
        self.collector_started = time.time()
        self.collector_first_row = False
        self.collector_stop_requested = False
        self.last_row_wall = time.time()

    def stop_collector(self, why: str, graceful: bool = True):
        if self.collector is None or self.collector.poll() is not None:
            return
        self.collector_stop_requested = True
        self.event("collector_stop", why=why, graceful=graceful)
        if graceful:
            try:
                os.killpg(self.collector.pid, signal.SIGINT)
                self.collector.wait(timeout=float(self.H.get("graceful_stop_sec", 150)))
                return
            except (OSError, subprocess.TimeoutExpired):
                self.event("collector_graceful_stop_timeout")
        try:
            os.killpg(self.collector.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.collector.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # ── rows ─────────────────────────────────────────────────────────────
    def read_rows(self) -> list:
        """All attempt rows; re-parsed only when the file changed, and never
        while it is being written (the collector rewrites it to add columns)."""
        try:
            st = os.stat(self.P.results_csv)
        except OSError:
            return []
        key = (st.st_size, st.st_mtime)
        if key == self._rows_cache[0]:
            return self._rows_cache[1]
        if time.time() - st.st_mtime < 1.0:
            return self._rows_cache[1]
        try:
            with open(self.P.results_csv, newline="", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.DictReader(line.replace("\x00", "") for line in f))
        except OSError:
            return self._rows_cache[1]
        self._rows_cache = (key, rows)
        return rows

    def glitch_reasons(self, r: dict) -> list:
        """Outcome-independent infrastructure failures of one attempt."""
        H, g = self.H, []
        st = r.get("status", "")
        if st == "EXCEPTION":
            return []          # handled by the collector's exit code + resume
        if str(r.get("teleport_ok", "1")).strip() == "0":
            g.append("teleport_failed")
        if str(r.get("baseline_arm_verified", "")).strip() == "0":
            g.append("baseline_arm_not_at_pose")   # before the treatment: outcome-independent
        # Moving obstacles run continuously (dwell <= 6 s), so in a 30-45 s probe
        # each one must move. A frozen one means its controller died: the
        # context is not what the campaign specifies.
        for k in (1, 2):
            if str(r.get(f"tel_obs{k}_moved", "")).strip() == "0":
                g.append(f"obstacle_{k}_not_moving")
        if _truthy(r.get("collision_channel_silent")):
            g.append("collision_channel_silent")
        if st == "LOCALIZATION_LOST":
            g.append("localization_lost_pre_treatment")
        if (r.get("tel_missing") or "").strip():
            g.append(f"telemetry_missing:{r['tel_missing']}")
        if self.ccfg.get("telemetry_enabled", True):
            rate = _f(r.get("tel_rate_hz"), 0.0)
            if rate < float(H.get("tel_rate_min_hz", 15)):
                g.append(f"telemetry_rate:{rate}")
            gap = _f(r.get("tel_max_gap_s_precontact"))
            if gap is not None and gap > float(H.get("sim_gap_max_s", 0.5)):
                g.append(f"sim_gap_precontact:{gap}")
            rtf = _f(r.get("tel_rtf_min_1s_precontact"))
            if rtf is not None and rtf < float(H.get("rtf_probe_min", 0.75)):
                g.append(f"laggy_precontact_rtf:{rtf}")
        applied = st not in ("POSE_EXHAUSTED", "LOCALIZATION_LOST", "BASELINE_COLLISION") \
            and not _truthy(r.get("no_switch"))
        if applied:
            if str(r.get("fp_sync_end", "")).strip() == "0":
                g.append("footprint_out_of_sync")
            reasons = r.get("invalid_reasons") or ""
            if "param_rejected_or_mismatched" in reasons or "baseline_not_applied" in reasons:
                g.append("treatment_not_applied")
            ws, wp = _f(r.get("watch_sec_sim")), _f(r.get("watch_sec_planned"))
            # The window ENDS at a collision, so a short window on a collision
            # probe is the outcome, not a glitch (the first pilot quarantined
            # one this way: outcome selection).
            if (not _truthy(r.get("collision")) and ws is not None and wp is not None
                    and ws < wp - 0.5):
                g.append(f"short_window:{ws}<{wp}")
        return g

    def process_new_rows(self) -> str:
        """Check rows appended since the last call. Returns 'degraded' when the
        stack itself looks bad, else ''."""
        rows = self.read_rows()
        if len(rows) <= self.rows_seen:
            return ""
        new = rows[self.rows_seen:]
        if any(not r.get("status") for r in new):
            return ""              # partial row; wait
        self.rows_seen = len(rows)
        self.last_row_wall = time.time()
        self.collector_first_row = True
        degraded = ""
        for r in new:
            tid = int(_f(r.get("trial_id"), -1))
            g = self.glitch_reasons(r)
            self.recent_verdicts.append(bool(g))
            qs = read_jsonl(self.P.quarantine)
            if g and any(str(q.get("run_id")) == str(r.get("run_id"))
                         and int(q.get("trial_id", -1)) == tid for q in qs):
                continue
            if g:
                n_q = sum(1 for q in qs if int(q.get("trial_id", -1)) == tid
                          and (q.get("attempt_kind") or "main") != "replicate")
                append_jsonl(self.P.quarantine, {
                    "run_id": r.get("run_id"), "trial_id": tid, "reason": ";".join(g),
                    "status": r.get("status"), "attempt_kind": r.get("attempt_kind", ""),
                    # outcome recorded ONLY for the selection audit, never used to decide
                    "outcome_collision": r.get("collision"), "outcome_stalled": r.get("probe_stalled"),
                    "quarantine_no": n_q + 1})
                self.event("probe_quarantined", trial_id=tid, run_id=r.get("run_id"),
                           reasons=g, rerun=(n_q + 1) <= int(self.H.get("max_reruns", 2)))
                if any(x.startswith(("collision_channel_silent", "telemetry_missing",
                                     "obstacle_")) for x in g):
                    degraded = f"stack-level glitch on probe {tid}: {g}"
            else:
                self.audit_probe(r)
                log.info(f"probe {tid} ok: {r.get('status')} arm={r.get('arm_requested_label')} "
                         f"coll={r.get('collision')} valid={r.get('probe_valid')} "
                         f"rtf={r.get('tel_rtf_min_1s_precontact')}")
        k = int(self.H.get("degraded_glitches", 2))
        if not degraded and sum(self.recent_verdicts) >= k:
            degraded = f"{sum(self.recent_verdicts)} glitched probes in the last {len(self.recent_verdicts)}"
        if self.rows_seen and self.rows_seen % int(self.H.get("stats_every", 25)) < len(new):
            self.rolling_stats(rows)
        return degraded

    def pending_reruns(self) -> list:
        """Probe ids whose LATEST attempt is quarantined and that still have
        rerun budget. Derived from files, so it survives supervisor restarts."""
        q = [x for x in read_jsonl(self.P.quarantine)
             if (x.get("attempt_kind") or "main") != "replicate"]
        if not q:
            return []
        qkeys = {(str(x.get("run_id")), int(x.get("trial_id", -1))) for x in q}
        nq = {}
        for x in q:
            nq[int(x.get("trial_id", -1))] = nq.get(int(x.get("trial_id", -1)), 0) + 1
        latest = {}
        for r in self.read_rows() or []:
            if (r.get("attempt_kind") or "main") == "replicate":
                continue
            tid = int(_f(r.get("trial_id"), -1))
            latest[tid] = (str(r.get("run_id")), r.get("status"))
        out = []
        for tid, (run, st) in latest.items():
            if (run, tid) in qkeys and nq.get(tid, 0) <= int(self.H.get("max_reruns", 2)):
                out.append(tid)
        return sorted(out)

    def audit_probe(self, r: dict):
        key = (str(r.get("run_id")), int(_f(r.get("trial_id"), -1)))
        a0 = self.audits.get(key)
        if a0 and not a0.get("audit_error") and int(a0.get("audit_version") or 1) >= DataAudit.VERSION:
            return                     # done (failed or outdated audits are redone)
        try:
            a = self.audit.audit_probe(r)
        except Exception as exc:   # noqa: BLE001
            a = {"run_id": key[0], "trial_id": key[1], "audit_error": f"audit crashed: {exc}"}
        self.audits[key] = a
        append_jsonl(self.audit_path, a)
        tol = float(self.H.get("fp_outside_tol_cm", 1.0))
        if (a.get("audit_fp_outside_cm") or -99) > tol:
            self.alert("warn", f"probe {key[1]}: arm geometry {a['audit_fp_outside_cm']} cm outside the "
                               f"costmap footprint ({a.get('audit_fp_worst_link')}, t={a.get('audit_fp_worst_t')} s)")

    def run_audit(self, acc: list):
        """Campaign-level data audit on the accepted, valid main rows."""
        valid = [r for r in acc if _truthy(r.get("probe_valid"))
                 and (r.get("attempt_kind") or "main") != "replicate"]
        if not valid:
            return
        for r in valid:
            self.audit_probe(r)
        logs = sorted(os.path.join(self.P.sup, f) for f in os.listdir(self.P.sup)
                      if f.startswith("stack_") and f.endswith(".log"))
        findings, summary = self.audit.audit_campaign(valid, self.audits, logs)
        self.audit.write_report(os.path.join(self.P.sup, "audit_report.md"), findings, summary)
        self.event("audit", findings=[f"{l}:{k}" for l, k, _ in findings], **{
            k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
        for lvl, k, msg in findings:
            if lvl == AUDIT_HALT:
                raise Halt(f"data audit: {msg}")
            # alert once per finding, then again only every 200 probes
            last = self._audit_alerted.get(k)
            if last is None or len(valid) - last >= 200:
                self._audit_alerted[k] = len(valid)
                self.alert("warn" if lvl == AUDIT_WARN else "info", f"data audit [{k}]: {msg}")

    def rolling_stats(self, rows: list):
        latest = {}
        for r in rows:
            if (r.get("attempt_kind") or "main") == "replicate":
                continue
            latest[int(_f(r.get("trial_id"), -1))] = r
        qk = {(str(x.get("run_id")), int(x.get("trial_id", -1))) for x in read_jsonl(self.P.quarantine)}
        acc = [r for t, r in sorted(latest.items())
               if (str(r.get("run_id")), t) not in qk and r.get("status") != "EXCEPTION"]
        applied = [r for r in acc if not _truthy(r.get("no_switch"))
                   and r.get("status") not in ("POSE_EXHAUSTED", "LOCALIZATION_LOST")]
        W = {}
        last = applied[-int(self.H.get("fp_window", 50)):]
        if len(last) >= 20:
            fp = sum(str(r.get("fp_sync_end")).strip() == "1" for r in last) / len(last)
            W["footprint_sync_lastN"] = round(fp, 3)
            if fp < float(self.H.get("halt_fp_sync_below", 0.95)):
                raise Halt(f"footprint sync {fp:.2f} over the last {len(last)} applied probes: "
                           "the costmaps are not following the arm (the campaign-A2 bug).")
        last20 = applied[-20:]
        if len(last20) >= 10:
            nd = sum(str(r.get("arm_delivered")).strip() == "0" for r in last20) / len(last20)
            W["arm_not_delivered_last20"] = round(nd, 3)
            if nd > float(self.H.get("halt_arm_not_delivered_above", 0.3)):
                raise Halt(f"the arm missed its treatment pose in {nd:.0%} of the last "
                           f"{len(last20)} applied probes: arm control is broken (the first "
                           "pilot had the arm stuck in carry while the costmaps said tucked).")
        if len(applied) >= 100:
            carry = sum(r.get("arm_requested_label") == "carry" for r in applied) / len(applied)
            W["carry_share"] = round(carry, 3)
            lo, hi = self.H.get("halt_carry_share_outside", [0.25, 0.75])
            if not lo <= carry <= hi:
                raise Halt(f"carry share {carry:.2f} over {len(applied)} probes: treatment "
                           "assignment is not balanced; randomisation is broken.")
        n = len(acc)
        if n >= 50:
            valid = [r for r in acc if _truthy(r.get("probe_valid"))]
            W["valid_rate"] = round(len(valid) / n, 3)
            if valid:
                W["collision_rate"] = round(sum(_truthy(r.get("collision")) for r in valid) / len(valid), 3)
                W["stall_rate"] = round(sum(_truthy(r.get("probe_stalled")) for r in valid) / len(valid), 3)
                W["collisions"] = sum(_truthy(r.get("collision")) for r in valid)
                W["stalls"] = sum(_truthy(r.get("probe_stalled")) for r in valid)
                vx = [_f(r.get("probe_vx_mean_over_cap")) for r in valid]
                vx = sorted(v for v in vx if v is not None)
                if vx:
                    W["vx_over_cap_median"] = vx[len(vx) // 2]
            for key, band in (self.H.get("warn_bands") or {}).items():
                v = W.get(key)
                if v is not None and not band[0] <= v <= band[1]:
                    W[f"WARN_{key}"] = f"{v} outside {band}"
        self.warnings = W
        warns = {k: v for k, v in W.items() if k.startswith("WARN_")}
        if warns:
            self.alert("warn", "rolling statistics outside target bands", **warns)
        self.event("rolling_stats", **W)
        self.run_audit(acc)

    # ── one collector run, watched ───────────────────────────────────────
    def supervise_collector(self) -> str:
        """Block until the collector exits; act on live health. Returns one of
        'done', 'restart:<why>', 'halt:<why>' (raises Halt), 'stop'."""
        H = self.H
        last_health = 0.0
        fatal_check = 0.0
        while True:
            time.sleep(1.0)
            now = time.time()
            code = self.collector.poll()
            degraded = self.process_new_rows()
            if code is not None:
                time.sleep(1.5)
                self.process_new_rows()
                return self.on_collector_exit(code)
            if self.stop_requested() and not self.collector_stop_requested:
                self.stop_collector("stop requested", graceful=True)
                continue
            age = self.watch.clock_age()
            rtf30 = self.watch.rtf(float(H.get("rtf_live_window_sec", 30)))
            reason = ""
            if age > float(H.get("clock_stall_sec", 20)):
                reason = f"sim clock stalled {age:.0f}s"
            elif self.stack is not None and self.stack.poll() is not None:
                reason = f"launch process exited ({self.stack.poll()})"
            elif now - fatal_check > 5:
                fatal_check = now
                line = self.stack_log_fatal()
                if line:
                    reason = f"core node died: {line}"
                elif not any(c == "gzserver" for _, c in self.tagged_pids()):
                    reason = "gzserver not running"
            if not reason:
                limit = float(H.get("probe_timeout_sec", 300))
                if not self.collector_first_row:
                    limit += float(H.get("collector_init_sec", 420))
                if now - self.last_row_wall > limit:
                    reason = f"no new probe for {now - self.last_row_wall:.0f}s (hang)"
            mem = _mem_available_mb()
            if not reason and mem < float(H.get("mem_hard_mb", 250)):
                reason = f"memory exhausted ({mem:.0f} MB available)"
            if reason:
                self.event("health_fail", reason=reason, rtf30=rtf30, mem_mb=round(mem))
                self.stop_collector(reason, graceful=False)
                return f"restart:{reason}"
            soft = ""
            if (self.collector_first_row and rtf30 == rtf30
                    and rtf30 < float(H.get("rtf_live_min", 0.5))):
                soft = f"real-time factor {rtf30:.2f} over 30 s"
            elif (mem < float(H.get("mem_soft_mb", 300))
                  and getattr(self, "mem_at_ready", mem) - mem > float(H.get("mem_leak_mb", 300))):
                # Only a DECLINE since the stack came up is something a restart
                # fixes (Gazebo / Nav2 growth); a machine that is simply short
                # of RAM would restart forever.
                soft = (f"memory leak: {mem:.0f} MB available, down "
                        f"{getattr(self, 'mem_at_ready', mem) - mem:.0f} MB since stack start")
            elif degraded:
                soft = degraded
            if soft and not self.collector_stop_requested:
                self.event("stack_degraded", reason=soft, **_memory_diagnosis())
                self.pending_restart = soft
                self.stop_collector(soft, graceful=True)
                continue
            if now - last_health > 10:
                last_health = now
                gz = [p for p, c in self.tagged_pids() if c == "gzserver"]
                append_jsonl(self.P.health, {
                    "rtf_30s": None if rtf30 != rtf30 else round(rtf30, 3),
                    "clock_age_s": round(age, 2), "mem_available_mb": round(mem),
                    "gzserver_rss_mb": round(sum(_rss_mb(p) for p in gz)),
                    "collector_rss_mb": round(_rss_mb(self.collector.pid)),
                    "load1": float(open("/proc/loadavg").read().split()[0]),
                    **{k: v for k, v in _memory_diagnosis().items() if k != "mem_available_mb"},
                    "rows": self.rows_seen, "checkpoint": read_checkpoint(self.P)})
                self.write_status()

    def on_collector_exit(self, code: int) -> str:
        names = {0: "OK/STOP_AT", 2: "INTERRUPTED", 3: "SIM_STALLED", 4: "EXCEPTION",
                 5: "INTEGRITY_FAILURES", 6: "CONSECUTIVE_FAILURES"}
        self.event("collector_exit", code=code, name=names.get(code, "CRASH"),
                   checkpoint=read_checkpoint(self.P))
        self.collector = None
        pending = getattr(self, "pending_restart", "")
        self.pending_restart = ""
        if self.collector_stop_requested and self.stop_requested():
            # WE stopped it because of the deadline / STOP file / a signal.
            # SIGINT also shuts down the collector's rclpy context, so its sim
            # clock freezes and it may report SIM_STALLED (exit 3): that is an
            # artefact of the stop, not a stalled simulator. (2026-11-13: this
            # was read as a stall, the restart was refused past the deadline and
            # the StopRequested escaped auto(), skipping the replicate block.)
            return "stop"
        if pending and code != 0:
            # We stopped it (degraded stack); its exit code says nothing new.
            return f"restart:{pending}"
        if code == 0:
            self.integrity_exits = 0
            return f"restart:{pending}" if pending else "done"
        if code == 2:
            if self.stop_requested():
                return "stop"
            return f"restart:{pending or 'collector interrupted unexpectedly'}"
        if code == 3:
            return "restart:collector reported SIM_STALLED"
        if code == 4:
            ck = read_checkpoint(self.P) + 1
            n = self.exception_probe[1] + 1 if self.exception_probe[0] == ck else 1
            self.exception_probe = (ck, n)
            if n >= int(self.H.get("max_exceptions_same_probe", 3)):
                raise Halt(f"probe {ck} raised an exception {n} times in a row; "
                           f"see {self.P.raw}/rct.log")
            return f"restart:collector EXCEPTION on probe {ck}"
        if code == 5:
            self.integrity_exits += 1
            if self.integrity_exits >= 2:
                raise Halt("treatment integrity failures twice in a row: parameters are "
                           "not reaching Nav2. Check param services / costmap names.")
            return "restart:INTEGRITY_FAILURES"
        if code == 6:
            raise Halt("collector stopped on CONSECUTIVE_FAILURES")
        if not self.collector_first_row:
            self.bringup_failures += 1
            if self.bringup_failures >= int(self.H.get("max_bringup_failures", 3)):
                raise Halt(f"collector failed during startup {self.bringup_failures} times "
                           f"(exit {code}); see {self.P.raw}/collector_stdout.log")
        return f"restart:collector exit {code}"

    # ── phases ───────────────────────────────────────────────────────────
    def _ensure_stack(self, gui: bool):
        if self.stack is not None and self.stack.poll() is None and self.stack_gui == gui:
            return
        if self.stack is not None:
            self.stop_stack()
        self.start_stack(gui)

    def _run_collector_until(self, target: int, gui: bool):
        """Drive probes up to `target` (and any reruns) on a stack in mode `gui`."""
        while not self.stop_requested():
            self._ensure_stack(gui)
            reruns = self.pending_reruns()
            ck = read_checkpoint(self.P)
            if reruns:
                self.phase_detail = f"rerun {reruns}"
                self.start_collector(rerun=reruns[: int(self.H.get("rerun_batch", 20))])
            elif ck < target:
                stop_at = min(target, ck + int(self.cfg.get("chunk_probes", 100)))
                self.start_collector(stop_at=stop_at)
            else:
                return
            self.write_status()
            res = self.supervise_collector()
            # The clean dataset is rebuilt after every collector run, so it is
            # current whenever the campaign is stopped or inspected.
            self.build_dataset()
            self.write_status()
            if res == "stop" or self.stop_requested():
                return
            if res.startswith("restart:"):
                self.restart_stack(res[8:], planned=False)
                continue
            # chunk done: planned recycle of a long-running stack
            self.stack_chunks += 1
            if (not gui and self.stack_chunks >= int(self.cfg.get("recycle_every_chunks", 3))
                    and read_checkpoint(self.P) < target):
                self.restart_stack(f"planned recycle after {self.stack_chunks} chunks",
                                   planned=True)

    def build_dataset(self):
        try:
            from rct_collector.scripts.build_dataset import build
            return build(self.P.root, quiet=True)
        except SystemExit:
            return None
        except Exception as exc:   # noqa: BLE001
            self.event("dataset_build_failed", error=str(exc))
            return None

    def pilot_gate(self, n: int) -> tuple:
        """Automatic pilot acceptance: every criterion a person would check in
        the report, as numbers. -> (passed, [failed criteria])."""
        rows = [r for r in self.read_rows() if (r.get("attempt_kind") or "main") != "replicate"]
        qk = {(str(x.get("run_id")), int(x.get("trial_id", -1)))
              for x in read_jsonl(self.P.quarantine)}
        latest = {}
        for r in rows:
            latest[int(_f(r.get("trial_id"), -1))] = r
        acc = [r for t, r in sorted(latest.items()) if 1 <= t <= n
               and (str(r.get("run_id")), t) not in qk and r.get("status") != "EXCEPTION"]
        applied = [r for r in acc if not _truthy(r.get("no_switch"))]
        fails = []

        def share(rs, fn):
            return sum(1 for r in rs if fn(r)) / len(rs) if rs else 0.0
        if len(acc) < n:
            fails.append(f"only {len(acc)}/{n} accepted pilot probes")
        if share(acc, lambda r: _truthy(r.get("probe_valid"))) < 0.9:
            fails.append("valid share < 0.9")
        if any(str(r.get("fp_sync_end")).strip() != "1" for r in applied):
            fails.append("a costmap footprint was out of sync at window end")
        if any(str(r.get("tel_fp_published_sync_end", "1")).strip() not in ("1", "") for r in applied):
            fails.append("published local/global footprints differ")
        if share(applied, lambda r: str(r.get("arm_delivered")).strip() == "1") < 0.9:
            fails.append("arm delivered in < 90% of probes")
        if any(str(r.get("baseline_arm_verified", "")).strip() == "0" for r in acc):
            fails.append("arm not at the baseline pose before a probe")
        if any((r.get("tel_missing") or "").strip() for r in acc):
            fails.append("telemetry streams missing")
        if any(_f(r.get("tel_rate_hz"), 0.0) < float(self.H.get("tel_rate_min_hz", 15)) for r in acc):
            fails.append("telemetry rate below threshold")
        for k in (1, 2):
            if any(f"tel_obs{k}_moved" in r for r in acc) and \
                    any(str(r.get(f"tel_obs{k}_moved")).strip() != "1" for r in acc):
                fails.append(f"moving obstacle {k} did not move in every probe")
        if share(acc, lambda r: _f(r.get("tel_rtf_min_1s_precontact"), 1.0)
                 >= float(self.H.get("rtf_probe_min", 0.75))) < 0.9:
            fails.append("real-time factor too low in > 10% of probes")
        if any(str(r.get("teleport_ok", "1")).strip() == "0" for r in acc):
            fails.append("a teleport failed")
        if any(_truthy(r.get("collision_channel_silent")) for r in acc):
            fails.append("collision channel silent")
        # data audit on the pilot probes (audit any that were missed / failed)
        for r in acc:
            self.audit_probe(r)
        au = [self.audits.get((str(r.get("run_id")), int(_f(r.get("trial_id"), -1)))) for r in acc]
        if any(a is None or a.get("audit_error") for a in au):
            errs = {a.get("audit_error") if a else "not audited" for a in au
                    if a is None or a.get("audit_error")}
            fails.append(f"data audit failed on some probes: {sorted(errs)}")
        tol = float(self.H.get("fp_outside_tol_cm", 1.0))
        out = [(a.get("trial_id"), a.get("audit_fp_outside_cm"), a.get("audit_fp_worst_link"))
               for a in au if a and (a.get("audit_fp_outside_cm") or -99) > tol]
        if out:
            fails.append(f"arm geometry outside the costmap footprint: {out[:5]}")
        if share(au, lambda a: bool(a) and (a.get("audit_n_local_traj") or 0) > 0) < 0.9:
            fails.append("MPPI local trajectories missing in > 10% of probes")
        return (not fails), fails

    def pilot(self, gui: bool = True, auto: bool = False):
        self.phase = "pilot"
        n = int(self.cfg.get("pilot_probes", 10))
        man = self.preflight()
        if (man.get("pilot") or {}).get("approved"):
            log.info("pilot already approved; use `run`")
            return
        self.watch = ClockWatch()
        self.memory_preflight()
        self.event("pilot_start", probes=n)
        try:
            self._run_collector_until(n, gui=gui)
        finally:
            self.stop_collector("pilot end", graceful=True)
            self.stop_stack()
        card = self.build_dataset()
        path = self.write_pilot_report(n, card)
        ok, fails = self.pilot_gate(n)
        with open(path, "a") as f:
            f.write("\n## Automatic gate\n\n" + ("**PASSED**: every criterion met.\n" if ok else
                    "**FAILED**:\n" + "".join(f"- {x}\n" for x in fails)))
        self.event("pilot_gate", passed=ok, fails=fails)
        self.write_status()
        print(open(path).read())
        if self.stop_requested():
            return
        if auto:
            if not ok:
                raise Halt("pilot failed the automatic gate: " + "; ".join(fails)
                           + f". Nothing beyond the pilot was collected; see {path}")
            self.approve("auto-approved: pilot passed every gate criterion")
            return
        if sys.stdin.isatty():
            ans = input("\nApprove the pilot and continue HEADLESS now? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                self.approve("approved interactively after the pilot")
                self.run_main()
                return
        print(f"\nWhen satisfied:  ros2 run rct_collector rct_supervise approve --config "
              f"{self.cfg_path}\n   then:         ros2 run rct_collector rct_supervise run "
              f"--config {self.cfg_path}")

    def approve(self, note: str):
        rep = os.path.join(self.P.sup, "pilot_report.md")
        if not os.path.exists(rep):
            raise SystemExit("no pilot report yet: run `pilot` first")
        sha = hashlib.sha256(open(rep, "rb").read()).hexdigest()
        set_manifest_field(self.P, "pilot", {
            "approved": True, "approved_at": datetime.now().isoformat(timespec="seconds"),
            "note": note, "report_sha256": sha, "checkpoint": read_checkpoint(self.P)})
        append_jsonl(self.P.events, {"event": "pilot_approved", "note": note})
        print("pilot approved")

    def run_main(self, until=None):
        self.phase = "main"
        man = self.preflight()
        if not (man.get("pilot") or {}).get("approved"):
            raise SystemExit("The pilot has not been approved. Run `pilot`, inspect it, "
                             "then `approve`.")
        if self.watch is None:
            self.watch = ClockWatch()
        self.memory_preflight()
        target = min(int(until), self.num_trials) if until else self.num_trials
        self.event("main_start", checkpoint=read_checkpoint(self.P), target=target)
        try:
            self._run_collector_until(target, gui=False)
            if self.stop_requested():
                self.event("stopped_by_request" if self.user_stop() else "stopped_at_deadline")
            elif target < self.num_trials and read_checkpoint(self.P) >= target:
                self.phase = "paused"
                self.event("reached_until", until=target)
            elif read_checkpoint(self.P) >= self.num_trials and not self.pending_reruns():
                self.phase = "complete"
                self.event("campaign_complete")
        finally:
            self.stop_collector("main end", graceful=True)
            self.stop_stack()
            self.build_dataset()
            self.write_status()
            # The STOP file is NOT removed here: inside `auto` that made the
            # user's stop look like a deadline and the replicate block ran
            # anyway (2026-11-13). main() clears a stale STOP at process start.

    def auto(self, hours: float, gui: bool = False, rep_n: int = 10, rep_times: int = 3):
        """Unattended: pilot (gated automatically) -> main collection until the
        time budget, leaving room for a short replicate block at the end. Every
        stop leaves the checkpoint; running `auto` or `run` again resumes."""
        end = time.time() + hours * 3600.0
        reserve = (rep_n * rep_times * 45.0 + 240.0) if rep_n and rep_times else 0.0
        self.event("auto_start", hours=hours,
                   ends=datetime.fromtimestamp(end).isoformat(timespec="minutes"),
                   replicate=f"{rep_n}x{rep_times}" if reserve else "none")
        self.deadline = end - reserve
        man = self.preflight()
        if not (man.get("pilot") or {}).get("approved"):
            self.pilot(gui=gui, auto=True)
            if self.user_stop():
                return
        if time.time() < self.deadline:
            self._deadline_logged = False
            try:
                self.run_main()
            except StopRequested:
                # Only the user's stop ends `auto`; the main-phase deadline
                # just hands over to the replicate block.
                if self.user_stop():
                    raise
        if reserve and not self.user_stop() and time.time() < end - 0.5 * reserve:
            self.deadline, self._deadline_logged = end, False
            self.replicate(rep_n, rep_times)
        self.event("auto_end", checkpoint=read_checkpoint(self.P))

    def replicate(self, n_probes: int, times: int, seed: int = 7):
        """Re-run a random sample of completed probes `times` more times each
        (same pose, config, apply time), tagged attempt_kind=replicate. They
        never replace the accepted attempt; build_dataset reports how often the
        runs of one probe agree, i.e. the simulation's outcome noise."""
        import random
        self.phase = "replicate"
        man = self.preflight()
        if not (man.get("pilot") or {}).get("approved"):
            raise SystemExit("approve the pilot first")
        ck = read_checkpoint(self.P)
        done = sorted({int(_f(r.get("trial_id"), -1)) for r in self.read_rows()
                       if (r.get("attempt_kind") or "main") != "replicate"
                       and r.get("status") != "EXCEPTION"} & set(range(1, ck + 1)))
        if not done:
            raise SystemExit("no completed probes to replicate")
        ids = sorted(random.Random(seed).sample(done, min(n_probes, len(done))))
        self.watch = self.watch or ClockWatch()
        self.memory_preflight()
        self.event("replicate_start", probes=ids, times=times)
        try:
            for k in range(times):
                if self.stop_requested():
                    break
                self._ensure_stack(gui=False)
                self.start_collector(rerun=ids, kind="replicate")
                res = self.supervise_collector()
                if res == "stop":
                    break
                if res.startswith("restart:"):
                    self.restart_stack(res[8:], planned=False)
        finally:
            self.stop_collector("replicate end", graceful=True)
            self.stop_stack()
            card = self.build_dataset()
            self.write_status()
        if card and card.get("replicates"):
            print(json.dumps(card["replicates"], indent=2))

    def write_pilot_report(self, n: int, card) -> str:
        rows = self.read_rows()
        q = {(str(x.get("run_id")), int(x.get("trial_id", -1))): x.get("reason")
             for x in read_jsonl(self.P.quarantine)}
        cols = [("trial", "trial_id"), ("kind", "attempt_kind"), ("status", "status"),
                ("arm", "arm_requested_label"), ("coll", "collision"),
                ("stall", "probe_stalled"), ("valid", "probe_valid"),
                ("fp L/G end", None), ("arm deliv", "arm_delivered"),
                ("t carry", "t_arm_carry"), ("rtf pre", "tel_rtf_min_1s_precontact"),
                ("tel Hz", "tel_rate_hz"), ("first contact", "tel_first_contact_links"),
                ("fp area L/G pub", None),
                ("pool", None), ("invalid / quarantine", None)]
        lines = [f"# Pilot report — {datetime.now().isoformat(timespec='minutes')}", "",
                 f"Campaign `{self.P.root}`, first {n} probes, GUI stack.", "",
                 "| " + " | ".join(c for c, _ in cols) + " |",
                 "|" + "---|" * len(cols)]
        for r in rows:
            tid = int(_f(r.get("trial_id"), -1))
            cells = []
            for c, k in cols:
                if c == "fp L/G end":
                    v = f"{r.get('fp_local_end', '')}/{r.get('fp_global_end', '')}"
                elif c == "fp area L/G pub":
                    v = f"{r.get('tel_fp_local_area_end', '')}/{r.get('tel_fp_global_area_end', '')}"
                elif c == "pool":
                    v = r.get("pose_pool_index", "")
                elif c.startswith("invalid"):
                    v = ";".join(x for x in [r.get("invalid_reasons", ""),
                                             q.get((str(r.get("run_id")), tid), "")] if x)
                else:
                    v = r.get(k, "")
                cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        moved = {k: sum(str(r.get(f"tel_obs{k}_moved")) == "1" for r in rows) for k in (1, 2)}
        ok_fp = all(str(r.get("fp_sync_end", "1")) in ("1", "") for r in rows
                    if not _truthy(r.get("no_switch")))
        lines += ["", "## Checks", "",
                  f"- footprint: local == global == arm on every applied probe: **{ok_fp}**",
                  f"- quarantined attempts: {len(q)} (re-run automatically; see events.jsonl)",
                  f"- telemetry files: {sum(1 for r in rows if r.get('tel_path'))}/{len(rows)}",
                  f"- moving obstacles moved in: box {moved[1]}/{len(rows)}, board {moved[2]}/{len(rows)} probes",
                  "", "## What to confirm by eye in Gazebo / RViz", "",
                  "- the arm unfolds to carry on carry probes and the green footprint grows in "
                  "BOTH costmaps at the same moment",
                  "- the green box (west) and blue board (east) keep shuttling the whole time, "
                  "pausing 1-6 s at each end, and back off instead of pushing the robot",
                  "- no robot teleports into a wall; no probe ends with the robot frozen",
                  ""]
        if card:
            lines += ["## Dataset card (pilot)", "", "```json",
                      json.dumps({k: card.get(k) for k in
                                  ("n_attempts", "n_probes", "n_valid", "probes_by_status",
                                   "invalid_reasons", "events_valid", "carry_share_valid")},
                                 indent=2), "```"]
        path = os.path.join(self.P.sup, "pilot_report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path


# ─────────────────────────────────────────────────────────────────────────────
def _default_cfg() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("rct_collector"), "config",
                            "campaign2_supervisor.yaml")
    except Exception:   # noqa: BLE001
        return "config/campaign2_supervisor.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "approve", "run", "replicate", "auto",
                                        "status", "stop", "check"])
    ap.add_argument("--hours", type=float, default=None,
                    help="auto/run: time budget; stops cleanly (checkpoint kept) when it ends")
    ap.add_argument("--gui", action="store_true", help="auto: show Gazebo/RViz for the pilot")
    ap.add_argument("--replicate", default="10x3",
                    help="auto: replicate block at the end, PROBESxTIMES ('0x0' = none)")
    ap.add_argument("--until", type=int, default=None,
                    help="run: stop after this probe id (e.g. a 200-probe calibration)")
    ap.add_argument("--n-probes", type=int, default=20, help="replicate: probes to repeat")
    ap.add_argument("--times", type=int, default=4, help="replicate: extra runs per probe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-input-change", action="store_true",
                    help="open a new code epoch instead of refusing changed inputs")
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config or _default_cfg())
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    P = CampaignPaths(cfg["campaign_dir"])

    if a.command == "status":
        if os.path.exists(P.status):
            print(open(P.status).read())
        for e in read_jsonl(P.events)[-15:]:
            print(json.dumps(e))
        for e in read_jsonl(P.alerts)[-5:]:
            print("ALERT", json.dumps(e))
        return
    if a.command == "stop":
        os.makedirs(P.sup, exist_ok=True)
        open(os.path.join(P.sup, "STOP"), "w").write(datetime.now().isoformat())
        print("stop requested: the supervisor finishes the current probe and shuts down")
        return

    os.makedirs(P.sup, exist_ok=True)
    logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(P.sup, "supervisor.log"))])
    sup = Supervisor(cfg, cfg_path, allow_input_change=a.allow_input_change)
    if a.command == "check":
        sup.preflight()
        print("inputs consistent; manifest", P.manifest)
        return
    if a.command == "approve":
        sup.approve(a.note)
        return
    if os.path.exists(sup._stop_flag):
        os.remove(sup._stop_flag)
    try:
        if a.command == "pilot":
            sup.pilot()
        elif a.command == "replicate":
            sup.replicate(a.n_probes, a.times)
        elif a.command == "auto":
            rn, rt = (int(v) for v in a.replicate.lower().split("x"))
            sup.auto(a.hours or 8.0, gui=a.gui, rep_n=rn, rep_times=rt)
        else:
            if a.hours:
                sup.deadline = time.time() + a.hours * 3600.0
            sup.run_main(until=a.until)
    except Halt as h:
        sup.phase = "HALTED"
        sup.alert("halt", str(h))
        sup.event("halt", reason=str(h))
        try:
            sup.stop_collector("halt", graceful=True)
            sup.stop_stack()
        finally:
            sup.write_status(halt_reason=str(h))
        logging.shutdown()
        os._exit(HALT_CODE)
    except StopRequested:
        sup.event("stopped_by_request")
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
