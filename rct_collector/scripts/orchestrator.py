#!/usr/bin/env python3
"""
RCT Orchestrator for PAL TIAGo Docker Environment.

Outer-loop controller:
1. Samples random Nav2 parameter configs (the causal intervention C_t)
2. Samples random initial and goal poses from map free space
3. Applies parameters via `ros2 param set` (dynamic reconfigure)
4. Teleports robot in Gazebo, sets AMCL initial pose
5. Runs navigation trial via NavigateToPose action
6. Records outcome data to CSV with checkpoint/resume

Design notes:
- PAL Module Manager is NOT available in public sim Docker images.
  Nav2 nodes are launched via pmb2_2dnav and stay alive between trials.
- All parameter changes use dynamic reconfigure (`ros2 param set`),
  confirmed working on the actual container.
- No file-based config override or module restart needed.
"""

import csv
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

from rct_collector.scripts.param_space import ParameterSpace, ARM_CONFIGS, ARM_JOINT_NAMES
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.trial_runner import TrialRunner, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the RCT orchestrator."""

    num_trials: int = 3000
    trial_timeout_sec: float = 180.0
    cooldown_sec: float = 3.0

    map_yaml_path: str = ""
    map_frame: str = "map"

    min_goal_distance: float = 3.0
    max_goal_distance: float = 15.0
    obstacle_clearance_m: float = 0.5
    sampling_bounds: Optional[dict] = None

    output_dir: str = "./rct_data"
    checkpoint_file: str = "checkpoint.json"
    results_csv: str = "rct_results.csv"

    max_consecutive_failures: int = 10
    reset_gazebo_between_trials: bool = True

    gazebo_robot_model: str = "tiago"
    scan_topic: str = "/scan_raw"
    odom_topic: str = "/mobile_base_controller/odom"
    collision_threshold: float = 0.15

    collect_risk_features: bool = False

    presampled_poses_path: Optional[str] = None
    presampled_configs_path: Optional[str] = None
    seed: Optional[int] = None

    # Arm control (per-episode physical arm move to match the footprint variable)
    move_arm: bool = False
    arm_control_mode: str = "joint_trajectory"  # default mode; per-pose "mode" overrides
    arm_action: str = "/arm_controller/follow_joint_trajectory"
    play_motion_action: str = "/play_motion2"
    arm_move_time_sec: float = 4.0
    arm_move_on_change_only: bool = True  # only re-move the arm when the label changes


class RCTOrchestrator:

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.param_space = ParameterSpace(seed=config.seed)
        self.pose_sampler: Optional[PoseSampler] = None
        self.trial_runner: Optional[TrialRunner] = None

        self.presampled_configs: Optional[list] = None
        self.presampled_poses: Optional[list] = None

        self.completed_trials: int = 0
        self.consecutive_failures: int = 0
        self.results: list[dict] = []
        self._shutdown_requested = False
        self._last_param_failures: list[str] = []
        self._current_arm_label: Optional[str] = None  # last successfully applied arm pose

        os.makedirs(self.config.output_dir, exist_ok=True)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.warning(f"Signal {signum} received, saving checkpoint...")
        self._shutdown_requested = True

    def initialize(self):
        if not self.config.map_yaml_path:
            raise ValueError("map_yaml_path must be set")

        logger.info(f"Loading map from {self.config.map_yaml_path}")
        self.pose_sampler = PoseSampler(
            map_yaml_path=self.config.map_yaml_path,
            obstacle_clearance_m=self.config.obstacle_clearance_m,
            min_goal_distance=self.config.min_goal_distance,
            max_goal_distance=self.config.max_goal_distance,
            sampling_bounds=self.config.sampling_bounds,
            seed=self.config.seed,
        )
        self.pose_sampler.load_map()

        self._load_presampled()

        self.trial_runner = TrialRunner(
            timeout_sec=self.config.trial_timeout_sec,
            collision_threshold=self.config.collision_threshold,
            collect_risk_features=self.config.collect_risk_features,
            scan_topic=self.config.scan_topic,
            odom_topic=self.config.odom_topic,
            gazebo_robot_model=self.config.gazebo_robot_model,
            output_dir=self.config.output_dir,
            map_yaml_path=self.config.map_yaml_path,
        )

        self._verify_nav2_running()

    def _verify_nav2_running(self):
        logger.info("Checking Nav2 is running...")
        result = subprocess.run(
            ["ros2", "action", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if "/navigate_to_pose" not in result.stdout:
            raise RuntimeError(
                "/navigate_to_pose action not found. Launch Nav2 first:\n"
                "  ros2 launch gazebo_simulation gazebo_with_navigation.launch.py"
            )
        logger.info("  /navigate_to_pose found ✓")

        # Smoke-test dynamic reconfig
        result = subprocess.run(
            ["ros2", "param", "get", "/controller_server", "controller_frequency"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"  Dynamic reconfig works ✓ ({result.stdout.strip()})")
        else:
            logger.warning("  Could not read /controller_server params")

    def _load_presampled(self):
        """Load pre-generated configs/poses if paths were given, and clamp
        num_trials to what's available so trial i always has a config[i]/pose[i]."""
        sources = [
            ("presampled_configs", self.config.presampled_configs_path,
             ParameterSpace.load_presampled, "pre-generated configs"),
            ("presampled_poses", self.config.presampled_poses_path,
             PoseSampler.load_presampled, "pre-sampled poses"),
        ]
        for attr, path, loader, label in sources:
            if path:
                loaded = loader(path)
                setattr(self, attr, loaded)
                logger.info(f"Loaded {len(loaded)} {label} from {path}")

        avail = [len(x) for x in (self.presampled_configs, self.presampled_poses)
                 if x is not None]
        if avail:
            limit = min(avail)
            if self.config.num_trials > limit:
                logger.warning(
                    f"num_trials ({self.config.num_trials}) exceeds available "
                    f"pre-generated entries ({limit}); clamping to {limit}."
                )
                self.config.num_trials = limit

    def run(self):
        self._load_checkpoint()
        start_trial = self.completed_trials
        logger.info(f"RCT collection: trials {start_trial+1}..{self.config.num_trials}")

        for trial_idx in range(start_trial, self.config.num_trials):
            if self._shutdown_requested:
                break

            trial_id = trial_idx + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"TRIAL {trial_id}/{self.config.num_trials}")
            logger.info(f"{'='*60}")

            try:
                result = self._run_single_trial(trial_id)
                self._record_result(trial_id, result)
                self.consecutive_failures = 0
            except Exception as e:
                logger.error(f"Trial {trial_id} EXCEPTION: {e}", exc_info=True)
                self.consecutive_failures += 1
                self._record_failure(trial_id, str(e))
                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.critical(f"{self.config.max_consecutive_failures} consecutive failures. Stopping.")
                    break

            self.completed_trials = trial_id
            self._save_checkpoint()

            if trial_idx < self.config.num_trials - 1:
                time.sleep(self.config.cooldown_sec)

        self._save_final_results()
        logger.info(f"Done. {self.completed_trials} trials recorded.")

    def _run_single_trial(self, trial_id: int) -> TrialResult:
        idx = trial_id - 1  # trials are 1-indexed; lists are 0-indexed

        # 1. Get this trial's config: pre-generated if available, else fresh
        if self.presampled_configs is not None:
            params = self.presampled_configs[idx]
            logger.info(f"  Config (presampled #{idx}): {self.param_space.flatten(params)}")
        else:
            params = self.param_space.sample()
            logger.info(f"  Config (fresh): {self.param_space.flatten(params)}")

        # 1a. Reset Gazebo. When we rely on arm persistence (move_arm +
        #     move-on-change), /reset_world is SKIPPED because it resets joint
        #     states and would snap the arm back every trial. The base is
        #     repositioned by the per-trial teleport instead.
        arm_persistence = self.config.move_arm and self.config.arm_move_on_change_only
        if self.config.reset_gazebo_between_trials and not arm_persistence:
            self._reset_gazebo()

        # 1b. Move the physical arm — only when needed (first trial, label change,
        #     or after a failed/unknown move). When not relying on persistence,
        #     the arm is re-commanded every trial.
        if self.config.move_arm:
            arm_pd = next(
                (p for p in self.param_space.params
                 if getattr(p, "apply_via", "") == "footprint"), None
            )
            if arm_pd is not None:
                label = params[arm_pd.node][arm_pd.name]
                if arm_persistence and label == self._current_arm_label:
                    logger.info(f"  Arm already in '{label}' — skipping move")
                else:
                    if self._move_arm(label):
                        self._current_arm_label = label
                    else:
                        self._current_arm_label = None  # unknown -> force re-move next trial
                        logger.warning(
                            f"  Arm did not reach '{label}' — physical geometry may "
                            f"not match the footprint for this trial."
                        )

        # 2. Apply via dynamic reconfigure, with read-back verification
        failed = self._apply_params(params)
        self._last_param_failures = failed
        if failed:
            logger.warning(
                f"  {len(failed)} param(s) did NOT verify (set rejected or read-back "
                f"mismatch): {failed}. Trial will be flagged in the CSV."
            )
        else:
            logger.info("  All params set and verified ✓")

        # 3. Get this trial's poses: pre-generated if available, else fresh
        if self.presampled_poses is not None:
            entry = self.presampled_poses[idx]
            start_pose, goal_pose = entry["start"], entry["goal"]
        else:
            start_pose, goal_pose = self.pose_sampler.sample_start_goal()
        logger.info(f"  Start: ({start_pose['x']:.2f}, {start_pose['y']:.2f})")
        logger.info(f"  Goal:  ({goal_pose['x']:.2f}, {goal_pose['y']:.2f})")

        # 4. Run navigation
        return self.trial_runner.run_trial(
            trial_id=trial_id,
            start_pose=start_pose,
            goal_pose=goal_pose,
            params=params,
        )

    def _apply_params(self, params: dict) -> list[str]:
        """Set all params via `ros2 param set` and verify each via read-back.

        A param is counted as failed if the `set` is rejected OR the
        subsequent `get` does not return the value we asked for. This is what
        protects RCT validity: a silently-rejected set would otherwise record
        a do(C=x) trial in which C never actually changed, biasing that
        parameter's causal coefficient.

        Two special cases from the finalized space:
        - extra_targets: one sampled value applied to several ROS params (e.g.
          inflation radius / footprint on BOTH local and global costmaps), so it
          stays a single causal knob. Every target must verify.
        - apply_via == "footprint": the sampled value is a label ("stowed" /
          "extended") resolved to a concrete polygon preset before setting.

        NOTE: read-back confirms the *parameter* changed. For inflation_radius
        and footprint it does NOT by itself prove the costmap *cost cache* /
        collision geometry was rebuilt — those are per-episode and verified
        behaviorally once (see README: inflation gradient).
        """
        failed = []
        for p in self.param_space.params:
            try:
                value = params[p.node][p.name]
            except KeyError:
                continue

            apply_via = getattr(p, "apply_via", "param_set")
            extra = getattr(p, "extra_targets", None) or []
            ros_node = getattr(p, "ros_node", None) or p.node

            if apply_via == "footprint":
                presets = getattr(p, "presets", None) or {}
                value = presets.get(value, value)
                param_type = "footprint"
            else:
                param_type = p.param_type

            for en, enm in [(ros_node, p.name)] + list(extra):
                if not self._set_and_verify(en, enm, value, param_type):
                    failed.append(f"{en}/{enm}")
        return failed

    def _set_and_verify(self, ros_node: str, name: str, value, param_type: str) -> bool:
        """Set one param, then read it back and confirm it took."""
        sv = ("true" if value else "false") if isinstance(value, bool) else str(value)
        try:
            r = subprocess.run(
                ["ros2", "param", "set", f"/{ros_node}", name, sv],
                capture_output=True, text=True, timeout=8,
            )
        except subprocess.TimeoutExpired:
            logger.debug(f"  SET TIMEOUT /{ros_node} {name}")
            return False
        if r.returncode != 0:
            logger.debug(f"  SET FAIL /{ros_node} {name}={sv}: {r.stderr.strip()}")
            return False

        got = self._get_param(ros_node, name)
        if got is None:
            logger.debug(f"  GET FAIL /{ros_node} {name} (could not read back)")
            return False
        if not self._values_match(got, value, param_type):
            logger.debug(f"  MISMATCH /{ros_node} {name}: asked {value!r}, got {got!r}")
            return False
        return True

    def _get_param(self, ros_node: str, name: str) -> Optional[str]:
        """Read a param value, returning the raw value string or None."""
        try:
            r = subprocess.run(
                ["ros2", "param", "get", f"/{ros_node}", name],
                capture_output=True, text=True, timeout=8,
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or "value is:" not in r.stdout:
            return None
        return r.stdout.split("value is:", 1)[1].strip()

    @staticmethod
    def _values_match(got_raw: str, target, param_type: str) -> bool:
        """Compare a read-back value string against the target we set."""
        try:
            if param_type == "continuous":
                tol = max(1e-3, 1e-2 * abs(float(target)))
                return abs(float(got_raw) - float(target)) <= tol
            if param_type == "discrete":
                return int(float(got_raw)) == int(target)
            if param_type == "footprint":
                import ast
                a = [[float(x) for x in pt] for pt in ast.literal_eval(str(got_raw))]
                b = [[float(x) for x in pt] for pt in ast.literal_eval(str(target))]
                if len(a) != len(b):
                    return False
                return all(
                    abs(ai - bi) <= 1e-3
                    for pa, pb in zip(a, b)
                    for ai, bi in zip(pa, pb)
                )
            if param_type == "categorical":
                return str(got_raw).strip() == str(target).strip()
            # boolean
            return str(got_raw).strip().lower() == str(target).strip().lower()
        except (ValueError, TypeError, SyntaxError):
            return False

    def _move_arm(self, label: str) -> bool:
        """Move TIAGo's arm to the configuration for `label` and wait for it to
        finish. Each pose picks its own control mode (cfg["mode"]): "play_motion"
        uses play_motion2 (collision-free named motion, e.g. "home"); else a
        FollowJointTrajectory on /arm_controller. If play_motion fails and joints
        are available, falls back to a joint trajectory.
        """
        cfg = ARM_CONFIGS.get(label)
        if cfg is None:
            logger.warning(f"  No arm config defined for '{label}'")
            return False

        mode = cfg.get("mode", self.config.arm_control_mode)

        if mode == "play_motion":
            if self._send_play_motion(cfg.get("motion_name", label), label):
                return True
            if cfg.get("joints"):
                logger.info(f"  play_motion for '{label}' failed; trying joint trajectory")
                return self._send_joint_trajectory(cfg["joints"], label)
            return False

        return self._send_joint_trajectory(cfg.get("joints"), label)

    def _send_play_motion(self, motion: str, label: str) -> bool:
        goal = f"{{motion_name: {motion}, skip_planning: false}}"
        cmd = ["ros2", "action", "send_goal", self.config.play_motion_action,
               "play_motion2_msgs/action/PlayMotion", goal]
        return self._send_arm_goal(cmd, label)

    def _send_joint_trajectory(self, joints, label: str) -> bool:
        if not joints:
            logger.warning(
                f"  Arm joints for '{label}' are unset — skipping arm motion."
            )
            return False
        if len(joints) != len(ARM_JOINT_NAMES):
            logger.warning(
                f"  Arm config '{label}' has {len(joints)} joints, expected "
                f"{len(ARM_JOINT_NAMES)}."
            )
            return False
        names = ", ".join(ARM_JOINT_NAMES)
        pos = ", ".join(str(float(v)) for v in joints)
        t = int(self.config.arm_move_time_sec)
        goal = (
            f"{{trajectory: {{joint_names: [{names}], "
            f"points: [{{positions: [{pos}], time_from_start: {{sec: {t}}}}}]}}}}"
        )
        cmd = ["ros2", "action", "send_goal", self.config.arm_action,
               "control_msgs/action/FollowJointTrajectory", goal]
        return self._send_arm_goal(cmd, label)

    def _send_arm_goal(self, cmd: list, label: str) -> bool:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.arm_move_time_sec + 20,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"  Arm motion to '{label}' timed out")
            return False
        if r.returncode == 0 and "SUCCEEDED" in r.stdout:
            logger.info(f"  Arm moved to '{label}' ✓")
            return True
        logger.warning(
            f"  Arm motion to '{label}' failed: {(r.stdout + r.stderr).strip()[-200:]}"
        )
        return False

    def _reset_gazebo(self):
        try:
            subprocess.run(
                ["ros2", "service", "call", "/reset_world", "std_srvs/srv/Empty", "{}"],
                capture_output=True, timeout=10,
            )
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  Gazebo reset failed: {e}")

    # ── Recording ───────────────────────────────────────────────────────

    def _record_result(self, trial_id: int, result: TrialResult):
        row = {"trial_id": trial_id, "timestamp": datetime.now().isoformat(), **result.to_dict()}
        failures = self._last_param_failures
        row["params_unverified_count"] = len(failures)
        row["params_unverified"] = ";".join(failures)
        row["config_valid"] = int(len(failures) == 0)
        self.results.append(row)
        self._append_csv(row)

    def _record_failure(self, trial_id: int, error_msg: str):
        row = {"trial_id": trial_id, "timestamp": datetime.now().isoformat(),
               "status": "EXCEPTION", "error": error_msg}
        self.results.append(row)
        self._append_csv(row)

    def _append_csv(self, row: dict):
        csv_path = os.path.join(self.config.output_dir, self.config.results_csv)
        exists = os.path.exists(csv_path)

        if not exists:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                w.writeheader()
                w.writerow(row)
        else:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                new_keys = set(row.keys()) - set(headers)
                if new_keys:
                    rows = list(reader)  # only need the full contents when re-writing

            if new_keys:
                # Extend headers (rare — only first few trials)
                all_h = sorted(set(headers) | set(row.keys()))
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=all_h)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                    w.writerow(row)
            else:
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=headers).writerow(row)

    # ── Checkpoint ──────────────────────────────────────────────────────

    def _save_checkpoint(self):
        path = os.path.join(self.config.output_dir, self.config.checkpoint_file)
        with open(path, "w") as f:
            json.dump({
                "completed_trials": self.completed_trials,
                "timestamp": datetime.now().isoformat(),
                "map": self.config.map_yaml_path,
            }, f, indent=2)

    def _load_checkpoint(self):
        path = os.path.join(self.config.output_dir, self.config.checkpoint_file)
        if os.path.exists(path):
            with open(path) as f:
                self.completed_trials = json.load(f)["completed_trials"]
            logger.info(f"Resumed from checkpoint: {self.completed_trials} done")
        else:
            logger.info("No checkpoint, starting fresh")

    def _save_final_results(self):
        s = lambda status: sum(1 for r in self.results if r.get("status") == status)
        summary = {
            "total": self.completed_trials,
            "SUCCESS": s("SUCCESS"), "COLLISION": s("COLLISION"),
            "TIMEOUT": s("TIMEOUT"), "FAILED": s("FAILED"),
            "CANCELED": s("CANCELED"), "PLANNING_FAILED": s("PLANNING_FAILED"),
            "EXCEPTION": s("EXCEPTION"),
        }
        path = os.path.join(self.config.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary: {summary}")
