#!/usr/bin/env python3
"""
Trial Runner — executes ONE navigation trial and records everything.

This is the combined runner: it keeps the orchestrator-friendly interface
(``TrialRunner.run_trial(trial_id, start_pose, goal_pose, params) -> TrialResult``)
so the outer loop in orchestrator.py can call it repeatedly, but internally it
uses the higher-level Nav2 ``BasicNavigator`` API plus rich per-timestep
recording (merged from the old custom_navigator_api.py):

Per trial it:
  1. Teleports the robot to the start pose in Gazebo (subprocess service call).
  2. Sets the AMCL initial pose and waits for localization to converge.
  3. Resolves this trial's footprint polygon from the sampled config
     (the arm/footprint categorical label -> ARM_CONFIGS preset).
  4. Fetches the *current* global costmap (it changes per trial because the
     inflation radius / footprint are part of the treatment) and builds a
     FootprintCollisionChecker with auto-computed geometry.
  5. Plans the global path and analyses it (footprint cost + obstacle distance).
  6. Follows the smoothed path, recording at ~10 Hz: pose, velocity, footprint
     cost, min scan, and the 8-D risk-state vector (if collect_risk_features).
  7. Detects collisions from footprint cost + LiDAR, classifies the outcome.
  8. Writes a full per-trial JSON (time-series) into output_dir/trials/ AND
     returns a flat TrialResult that the orchestrator appends to its CSV.

ROS lifecycle note: rclpy and the BasicNavigator are initialized ONCE (in
__init__), not per trial, because the orchestrator drives many trials in a row
against an already-running Nav2 stack.

Subscribes (recorder node):
    /amcl_pose                      robot pose
    <odom_topic>                    robot velocity (default /mobile_base_controller/odom)
    <scan_topic>                    LiDAR for collision / min-distance (default /scan_raw)
    /risk_state (Float64MultiArray) 8-D risk vector (only if collect_risk_features)
    
    
    
    
Footrpint polygons are:  "tucked": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [-0.138, -0.238], [-0.000, -0.275], [0.138, -0.238], [0.209, -0.181], [0.238, -0.138], [0.275, 0.000], [0.252, 0.182], [0.217, 0.242], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
"carry": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641], [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138], [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]"
"""

import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray, Bool, String
from sensor_msgs.msg import LaserScan, JointState
from nav_msgs.msg import Odometry, Path
from nav2_msgs.srv import GetCostmap
try:
    from nav2_msgs.msg import BehaviorTreeLog
except ImportError:                       # older nav2_msgs
    BehaviorTreeLog = None
from gazebo_msgs.msg import ModelStates
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import threading
from collections import deque
from rclpy.executors import SingleThreadedExecutor

logger = logging.getLogger(__name__)

# Footprint-cost recording depends on the checker (copied into this package) and
# nav2_simple_commander. If unavailable, the runner degrades to plain LiDAR-scan
# collision detection with a warning (no per-pose footprint cost).
try:
    from rct_collector.scripts.footprint_collision_checker import (
        FootprintCollisionChecker,
        LETHAL_OBSTACLE,
        INSCRIBED_INFLATED_OBSTACLE,
    )
    _HAVE_FOOTPRINT_CHECKER = True
except Exception as _e:  # pragma: no cover
    FootprintCollisionChecker = None
    LETHAL_OBSTACLE = 254
    INSCRIBED_INFLATED_OBSTACLE = 253
    _HAVE_FOOTPRINT_CHECKER = False
    logger.warning(f"FootprintCollisionChecker unavailable ({_e}); "
                   "falling back to LiDAR-only collision detection.")


# ── Quaternion helpers (avoid a tf_transformations dependency) ───────────────

def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw (rad) -> (x, y, z, w), flat robot (roll=pitch=0)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """(x, y, z, w) -> yaw (rad)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── Laser → base_link static transform ───────────────────────────────────────
# The LiDAR reports ranges in its own frame (`base_laser_link` for the PMB2
# sim). To measure distance from the robot's *base_link* origin every scan hit
# is first lifted into base_link. That transform is static, so we hard-code it
# here rather than pay a tf2 lookup per scan. It is (x, y, yaw) of the laser
# frame expressed in base_link — verify against your robot's URDF (`base_link`
# → laser joint) if you change platforms or mount the laser.
LASER_TO_BASE_LINK = (0.202, 0.0, 0.0)   # PMB2 / TIAGo base front laser


def _min_dist_base_link_to_obstacles(obstacles_xy: np.ndarray,
                                     base_link_xy: tuple) -> float:
    """Min Euclidean distance (m) from the robot's base_link origin to the
    nearest obstacle point. Measures to the base_link POINT — NOT to the
    footprint polygon — so it ignores the robot's shape/arm config.

    obstacles_xy : (K, 2) obstacle points, in the same frame as base_link_xy
                   (base_link for LiDAR hits; the map frame for static-map cells)
    base_link_xy : (x, y) of the base_link origin in that frame
                   ((0, 0) for LiDAR hits; the robot pose for the static map)

    Always >= 0 (0 if an obstacle coincides with base_link). +inf if no points.
    Fully vectorised: one subtract + hypot + min over the K points.
    """
    if obstacles_xy.shape[0] == 0:
        return float("inf")
    dx = obstacles_xy[:, 0] - base_link_xy[0]
    dy = obstacles_xy[:, 1] - base_link_xy[1]
    return float(np.sqrt(dx * dx + dy * dy).min())


def _static_map_obstacles(map_yaml_path: str,
                          occupied_thresh: float = 0.65) -> tuple[np.ndarray, float]:
    """Load a map_server map (yaml + image) and return its occupied cells.

    Returns (obstacles_xy, resolution) where obstacles_xy is (K, 2) world
    coordinates (cell centres, map frame) of every occupied cell. Follows the
    map_server convention: normalised occupancy is (1 - pixel/255), or pixel/255
    when `negate: 1`; a cell is an obstacle when that value >= occupied_thresh
    (the yaml's own `occupied_thresh` wins if present). Assumes the map origin
    yaw is 0 (true for standard map_server maps). Computed once — the map is
    static.
    """
    import yaml
    from pathlib import Path
    from PIL import Image

    with open(map_yaml_path) as f:
        info = yaml.safe_load(f)
    res = float(info.get("resolution", 0.05))
    origin = info.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    negate = int(info.get("negate", 0))
    occ_th = float(info.get("occupied_thresh", occupied_thresh))

    img = np.array(Image.open(
        Path(map_yaml_path).parent / info["image"]).convert("L"))
    h = img.shape[0]
    p = img.astype(np.float64) / 255.0
    occ = p if negate else (1.0 - p)              # normalised occupancy prob.
    rows, cols = np.where(occ >= occ_th)

    # Image row 0 is the TOP of the map; map y grows upward → flip the row axis.
    wx = ox + (cols + 0.5) * res
    wy = oy + (h - 0.5 - rows) * res
    return np.column_stack((wx, wy)), res


def _min_distance_to_obstacle(costmap_array: np.ndarray, robot_rc: np.ndarray,
                              resolution: float) -> float:
    """Min Euclidean distance (m) from a map cell to the nearest lethal cell."""
    obstacles = np.argwhere(costmap_array > INSCRIBED_INFLATED_OBSTACLE)
    if len(obstacles) == 0:
        return float("inf")
    dists = np.linalg.norm(obstacles - np.asarray(robot_rc), axis=1)
    return float(np.min(dists) * resolution)


# ── Recording data classes ───────────────────────────────────────────────────

@dataclass
class RiskStateRecord:
    """One timestamped 8-D risk-state observation from /risk_state."""
    timestamp: float
    r_min: float      # min obstacle distance
    r_width: float    # corridor width
    r_ttc: float      # time to collision
    r_dens: float     # obstacle density
    r_clear: float    # heading clearance
    r_curve: float    # path curvature
    r_grad: float     # costmap gradient
    r_vis: float      # visibility risk

    @classmethod
    def from_array(cls, timestamp: float, data: list) -> "RiskStateRecord":
        if len(data) != 8:
            raise ValueError(f"Expected 8 risk values, got {len(data)}")
        return cls(timestamp, *[float(v) for v in data])


@dataclass
class TrialResult:
    """Outcome of a single trial. Scalars go to the orchestrator CSV via
    to_dict(); the full time-series is written to a per-trial JSON separately."""

    # Identifiers
    trial_id: int = 0

    # Poses
    start_x: float = 0.0
    start_y: float = 0.0
    start_yaw: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_yaw: float = 0.0

    # Configuration (flattened treatment C_t)
    params: dict = field(default_factory=dict)

    # Outcome (Y_t)
    status: str = "UNKNOWN"     # SUCCESS, COLLISION, TIMEOUT, ABORTED, FAILED, ...
    collision: bool = False
    travel_time_sec: float = 0.0
    path_length_m: float = 0.0                       # controller (executed) path
    initial_global_path_length_m: float = 0.0               # planned global path
    goal_distance_remaining: float = 0.0
    final_xy_error: float = 0.0
    final_yaw_error: float = 0.0
    min_obstacle_distance: float = float("inf")     # closest LiDAR approach (base_link → obstacle)
    min_map_obstacle_distance: float = float("inf")  # closest static-map approach (base_link → obstacle)
    min_global_obstacle_distance: float = float("inf")  # along the planned path

    # Navigation timing window (wall clock). Used to trim the continuously-recorded
    # risk_state_history down to samples collected during navigation.
    t_nav_start: float = 0.0
    t_nav_end: float = 0.0

    # Bookkeeping for the rich JSON
    num_risk_samples: int = 0
    num_controller_samples: int = 0
    json_path: str = ""
    run_id: str = ""
    world_name: str = "pal_office"

    # LiDAR self-return diagnostics. If self_hit_fraction is high, the raw scan
    # is dominated by the robot's own structure and min_obstacle_distance would
    # be a constant if it were not filtered (see TrialRunnerNode._scan_callback).
    scan_beams_total: int = 0
    scan_beams_self: int = 0
    min_obstacle_distance_valid: bool = True

    # Pose provenance. Outcomes are measured from Gazebo ground truth
    # (/gazebo/model_states); /amcl_pose is retained only as the robot's BELIEF
    # so that localization error is observable rather than silently folded into
    # final_xy_error.
    pose_source: str = ""
    gt_msgs_seen: int = 0
    unique_pose_fraction: float = 0.0
    localization_error_m: float = float("nan")

    # Teleport fidelity. 0 means the start pose was never confirmed in Gazebo,
    # so start_x/start_y may not describe where the robot actually began.
    teleport_ok: int = 1

    # Collision-channel liveness. collision_channel_silent=1 means NOT ONE
    # message arrived on /gazebo/collision during the trial, so collision=0 is
    # indistinguishable from "the monitor plugin is not loaded".
    collision_msgs_seen: int = 0
    collision_channel_silent: int = 0
    
    global_planner_ticks: int = 0
    replan_history: list = field(default_factory=list)   # not written to CSV, only JSON
    collision_links: list = field(default_factory=list)  # not written to CSV, only JSON

    # True when THIS runner decided the terminal status (early in-tolerance
    # SUCCESS, or TIMEOUT) rather than Nav2's action result. When set, the
    # status resolution in run_trial() must NOT consult getResult(): after a
    # cancelTask() the BasicNavigator's cached status can be stale (it is only
    # written inside isTaskComplete(), which never returns True on these paths)
    # or can report SUCCEEDED from a goal that latched success just before the
    # cancel landed. Either way it would silently overwrite our status.
    terminated_by_runner: bool = False

    # ── Failure taxonomy ────────────────────────────────────────────────────
    # status says THAT the trial failed; failure_reason says WHY, so that
    # infrastructure failures can be dropped instead of being averaged in with
    # genuine navigation failures. Values:
    #   NONE                    - no failure
    #   PLANNING_FAILED_INITIAL - the pre-navigation getPath() found no path
    #   BT_PLANNER_FAILED       - ComputePathToPose went to FAILURE in the BT
    #   BT_CONTROLLER_FAILED    - FollowPath went to FAILURE in the BT
    #   BT_OTHER_FAILED         - some other BT node failed (see bt_failed_node)
    #   RUNNER_TIMEOUT          - hit TrialRunner.timeout_sec
    #   STUCK_NO_PROGRESS       - pose static for no_progress_timeout_sec
    #   COLLISION               - Gazebo contact
    #   UNKNOWN                 - Nav2 reported failure but the BT log was silent
    failure_reason: str = "NONE"
    bt_failed_node: str = ""            # last BT node observed entering FAILURE
    bt_failure_detail: str = ""         # "FollowPath=3;ComputePathToPose=1"
    nav2_result: str = ""               # Nav2's own verdict, kept separately
    longest_stall_sec: float = 0.0      # longest interval with no pose change

    # ── Dual outcome: what happened vs what the robot believed happened ─────
    #
    # The estimand for causal analysis is "did the robot physically arrive",
    # a fact about the world. Nav2's own SUCCEEDED is a fact about its BELIEF,
    # produced by a goal checker reading /amcl_pose. In the 30-trial smoke run
    # those two disagreed on 10 of 11 nominal successes, because AMCL drift
    # (mean 0.39 m, max 0.53 m) exceeded xy_goal_tolerance (0.35 m). Scoring on
    # the belief silently changes the estimand to "believes it arrived", and a
    # tuner fitted to that learns to prefer configs that make the robot
    # confidently wrong.
    #
    # Three outcome columns, all recorded on every trial:
    #
    #   success_true              ground truth terminal pose within tolerance.
    #                             THE primary outcome. Measured from
    #                             /gazebo/model_states.
    #   believed_within_tolerance the same test applied to the robot's OWN
    #                             terminal pose estimate (/amcl_pose). Always
    #                             observable, never censored, and it is exactly
    #                             what a real robot could compute onboard.
    #   success_believed          Nav2's action result (SUCCEEDED). Censored
    #                             when this runner ended the trial first — see
    #                             belief_censored.
    #
    # success_true vs believed_within_tolerance is the 2x2 to analyse; the gap
    # between them is a result in its own right, not just a nuisance.
    success_true: Optional[int] = None
    believed_within_tolerance: Optional[int] = None
    success_believed: Optional[int] = None
    belief_censored: int = 0

    # Continuous counterparts, so the gap is measurable and not only binary.
    believed_final_xy_error: float = float("nan")
    believed_final_yaw_error: float = float("nan")
    belief_error_gap_m: float = float("nan")   # true error - believed error

    # AGREE_SUCCESS | AGREE_FAIL | FALSE_SUCCESS | MISSED_SUCCESS | UNOBSERVED
    # FALSE_SUCCESS is the dangerous cell: the robot thinks it arrived and did
    # not. MISSED_SUCCESS is the benign one.
    outcome_agreement: str = ""

    # Did ground truth EVER satisfy both tolerances during the run, even if the
    # robot then drove away again? Distinguishes "never got there" from "got
    # there and overshot", which the terminal-pose test alone conflates.
    gt_ever_within_tolerance: int = 0
    t_first_within_tolerance: float = float("nan")   # sec from nav start

    # The thresholds this trial was scored against, recorded per row so any
    # outcome above can be re-derived under a different tolerance later.
    xy_goal_tolerance_used: float = float("nan")
    yaw_goal_tolerance_used: float = float("nan")

    def _score_outcomes(self) -> None:
        """Fill the dual-outcome fields from the errors already measured.

        Called at the end of run_trial(), after both the ground-truth and
        believed terminal poses are known. Never invents a verdict: trials in
        which no navigation happened leave the binaries as None so they are
        written as blank rather than scored as failures.
        """
        xy_tol = self.xy_goal_tolerance_used
        yaw_tol = self.yaw_goal_tolerance_used

        def _within(xy_err: float, yaw_err: float) -> Optional[int]:
            if not (math.isfinite(xy_err) and math.isfinite(yaw_err)):
                return None
            return int(abs(xy_err) <= xy_tol and abs(yaw_err) <= yaw_tol)

        navigated = (self.travel_time_sec or 0.0) > 0.0 and self.status != "PLANNING_FAILED"

        if navigated:
            self.success_true = _within(self.final_xy_error, self.final_yaw_error)
            self.believed_within_tolerance = _within(
                self.believed_final_xy_error, self.believed_final_yaw_error)
        else:
            # No navigation: final_xy_error is a 0.0 default, not a measurement.
            self.success_true = None
            self.believed_within_tolerance = None

        if math.isfinite(self.final_xy_error) and math.isfinite(self.believed_final_xy_error):
            self.belief_error_gap_m = float(
                self.final_xy_error - self.believed_final_xy_error)

        # Nav2's action verdict. Unreliable whenever this runner cancelled the
        # task first: BasicNavigator only writes its status inside
        # isTaskComplete(), so after a cancelTask() the value can be left over
        # from a previous trial or can have latched SUCCEEDED just before the
        # cancel landed. Record it as censored rather than as a 0.
        if self.terminated_by_runner or self.collision:
            self.success_believed = None
            self.belief_censored = 1
        elif self.nav2_result == "SUCCEEDED":
            self.success_believed = 1
        elif self.nav2_result in ("FAILED", "CANCELED"):
            self.success_believed = 0
        else:
            self.success_believed = None
            self.belief_censored = 1

        # Agreement is scored on the two POSE-based columns, because those are
        # both always observable. success_believed (the action result) is kept
        # as a separate column for diagnosing Nav2 itself.
        if self.success_true is None or self.believed_within_tolerance is None:
            self.outcome_agreement = "UNOBSERVED"
        elif self.success_true == 1 and self.believed_within_tolerance == 1:
            self.outcome_agreement = "AGREE_SUCCESS"
        elif self.success_true == 0 and self.believed_within_tolerance == 0:
            self.outcome_agreement = "AGREE_FAIL"
        elif self.success_true == 0 and self.believed_within_tolerance == 1:
            self.outcome_agreement = "FALSE_SUCCESS"
        else:
            self.outcome_agreement = "MISSED_SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        """Flat, single-level dict for CSV output.

        ``goal_distance_remaining`` is deliberately NOT emitted here: it was
        exactly equal to ``final_xy_error`` in every row of the smoke run (both
        are the Euclidean distance from the final pose to the goal), so it only
        widened the schema. It is still kept on the dataclass and written to the
        per-trial JSON.
        """
        self_frac = (self.scan_beams_self / self.scan_beams_total
                     if self.scan_beams_total else "")

        def _b(v):
            """None -> blank cell. Keeps 'not observed' distinct from 0."""
            return "" if v is None else int(v)

        d = {
            "start_x": self.start_x, "start_y": self.start_y, "start_yaw": self.start_yaw,
            "goal_x": self.goal_x, "goal_y": self.goal_y, "goal_yaw": self.goal_yaw,
            "status": self.status,
            "collision": int(self.collision),
            # ── dual outcome ────────────────────────────────────────────────
            "success_true": _b(self.success_true),
            "believed_within_tolerance": _b(self.believed_within_tolerance),
            "success_believed": _b(self.success_believed),
            "belief_censored": int(self.belief_censored),
            "outcome_agreement": self.outcome_agreement,
            "believed_final_xy_error": self.believed_final_xy_error,
            "believed_final_yaw_error": self.believed_final_yaw_error,
            "belief_error_gap_m": self.belief_error_gap_m,
            "gt_ever_within_tolerance": int(self.gt_ever_within_tolerance),
            "t_first_within_tolerance": self.t_first_within_tolerance,
            "xy_goal_tolerance_used": self.xy_goal_tolerance_used,
            "yaw_goal_tolerance_used": self.yaw_goal_tolerance_used,
            "travel_time_sec": self.travel_time_sec,
            "path_length_m": self.path_length_m,
            "initial_global_path_length_m": self.initial_global_path_length_m,
            "final_xy_error": self.final_xy_error,
            "final_yaw_error": self.final_yaw_error,
            "min_obstacle_distance": self.min_obstacle_distance,
            "min_obstacle_distance_valid": int(self.min_obstacle_distance_valid),
            "scan_self_hit_fraction": self_frac,
            "min_map_obstacle_distance": self.min_map_obstacle_distance,
            "min_global_obstacle_distance": self.min_global_obstacle_distance,
            "num_risk_samples": self.num_risk_samples,
            "num_controller_samples": self.num_controller_samples,
            "global_planner_ticks": self.global_planner_ticks,
            "pose_source": self.pose_source,
            "gt_msgs_seen": self.gt_msgs_seen,
            "unique_pose_fraction": self.unique_pose_fraction,
            "localization_error_m": self.localization_error_m,
            "teleport_ok": self.teleport_ok,
            "collision_msgs_seen": self.collision_msgs_seen,
            "collision_channel_silent": self.collision_channel_silent,
            "json_path": self.json_path,
            "failure_reason": self.failure_reason,
            "bt_failed_node": self.bt_failed_node,
            "bt_failure_detail": self.bt_failure_detail,
            "nav2_result": self.nav2_result,
            "longest_stall_sec": round(self.longest_stall_sec, 2),
        }
        for key, val in self.params.items():
            d[f"param__{key}"] = val
        return d


# ── Recorder node ─────────────────────────────────────────────────────────────

class TrialRunnerNode(Node):
    """Subscriber-only ROS node that records robot state + risk during a trial.

    It does NOT drive navigation (BasicNavigator does that); it only listens and
    buffers. One instance is reused across trials — call reset() between trials.
    """

    def __init__(self, scan_topic: str, odom_topic: str, risk_topic: str,
                 collect_risk_features: bool,
                 self_filter_radius_m: float = 0.30,
                 angle_mask_deg: Optional[list] = None,
                 gazebo_robot_model: str = "tiago"):
        super().__init__("rct_trial_runner")
        self.gazebo_robot_model = gazebo_robot_model
        self.collect_risk_features = collect_risk_features

        # LiDAR self-return rejection (see _scan_callback).
        self._self_filter_radius = float(self_filter_radius_m)
        self._angle_mask_rad = [
            (math.radians(float(lo)), math.radians(float(hi)))
            for lo, hi in (angle_mask_deg or [])
        ]
        self.scan_beams_total = 0
        self.scan_beams_self = 0

        # Latest joint positions, for arm-pose verification.
        self.joint_positions: dict = {}
        self.joint_states_stamp: float = 0.0

        # Live state.
        #
        # Two poses are tracked deliberately:
        #   gt_*    - Gazebo ground truth (/gazebo/model_states). Continuous,
        #             exact, physics-rate. ALL outcome variables are measured
        #             from this: path length, final error, clearance.
        #   robot_* - the robot's BELIEF (/amcl_pose). Kept only so that
        #             localization error (gt - belief) is observable, and so
        #             we can see what Nav2's goal checker was looking at.
        #
        # /amcl_pose is republished only after AMCL's update_min_d (~0.28 m),
        # so it is far too coarse to integrate a trajectory from: the smoke run
        # produced 477 samples containing 37 distinct poses.
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_yaw = 0.0
        self.gt_valid = False
        self.gt_msgs_seen = 0
        self.gt_model_missing_logged = False

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.linear_velocity = None
        self.angular_velocity = None
        self.min_scan_value: Optional[float] = None
        self.is_collided = False
        self._recording = False
        self._trial_id = -1

        # base_link → nearest-obstacle distance state (LiDAR-based).
        # Static laser pose in base_link; folded into the cached beam directions.
        self._laser_tx, self._laser_ty, self._laser_yaw = LASER_TO_BASE_LINK
        # Cached per-beam unit directions (base_link), keyed on scan geometry.
        self._scan_key: Optional[tuple] = None
        self._scan_ux: Optional[np.ndarray] = None
        self._scan_uy: Optional[np.ndarray] = None

        # base_link → nearest-obstacle distance state (static-map-based).
        self._map_obstacles: Optional[np.ndarray] = None   # (K, 2) map-frame cells
        self.min_map_value: Optional[float] = None
        self.min_map_overall = float("inf")

        # Buffers (cleared per trial)
        self.latest_risk_state: Optional[RiskStateRecord] = None
        self.risk_state_history: list[RiskStateRecord] = []
        self.controller_path: list[dict] = []
        self.min_scan_overall = float("inf")
        self.collision_links = []
        # Every message on /gazebo/collision is counted, not just the True ones,
        # so "no collisions" can be told apart from "no publisher".
        self.collision_msgs_seen = 0
        
        self.replan_events: deque = deque()
        self.create_subscription(Path, "/plan", self._plan_callback, 5)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._pose_callback, 5)
        self.create_subscription(
            Odometry, odom_topic, self._odom_callback, sensor_qos)
        self.create_subscription(
            LaserScan, scan_topic, self._scan_callback, sensor_qos)
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, sensor_qos)
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._model_states_callback,
            sensor_qos)
        # Behaviour-tree transitions. This is the only channel that says WHICH
        # node failed; BasicNavigator.getResult() collapses everything to FAILED.
        self.bt_failures: list[dict] = []
        if BehaviorTreeLog is not None:
            self.create_subscription(
                BehaviorTreeLog, "/behavior_tree_log", self._bt_log_callback, 10)
        self.create_subscription(Bool, "/gazebo/collision", self._collision_callback, 5)
        self.create_subscription(String, "/gazebo/collision_info", self._collision_info_callback, 5)
        if collect_risk_features:
            self.create_subscription(
                Float64MultiArray, risk_topic, self._risk_state_callback, 10)

    # ── callbacks ──
    def _plan_callback(self, msg: Path):
        if not self._recording:                     # gate: ignore plans between trials
            return
        n = len(msg.poses)
        arr = np.empty((n, 3), dtype=np.float32)
        for i, ps in enumerate(msg.poses):
            p, o = ps.pose.position, ps.pose.orientation
            arr[i] = (p.x, p.y, quaternion_to_yaw(o.x, o.y, o.z, o.w))
        self.replan_events.append({
            "timestamp": time.time(),
            "trial_id": self._trial_id,             # stamp it; see note below
            "poses": arr,
            "path_msg": msg,
        })
    
    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """AMCL pose — the robot's BELIEF about where it is.

        Not used for any outcome variable. Retained so that localization error
        (ground truth minus belief) is recoverable, and so we can tell what
        Nav2's goal checker was looking at when it declared SUCCESS.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)

    def _model_states_callback(self, msg: ModelStates):
        """Ground-truth pose straight from the physics engine.

        Published by the gazebo_ros_state plugin — the same plugin that provides
        /gazebo/set_entity_state, which the teleport already relies on, so no
        extra setup is required. Continuous and exact, unlike /amcl_pose, which
        is quantised to AMCL's update_min_d.
        """
        try:
            i = msg.name.index(self.gazebo_robot_model)
        except ValueError:
            if not self.gt_model_missing_logged:
                self.get_logger().error(
                    f"Model '{self.gazebo_robot_model}' not in /gazebo/model_states "
                    f"(names: {list(msg.name)[:8]}). Ground-truth pose unavailable; "
                    "outcomes would fall back to AMCL.")
                self.gt_model_missing_logged = True
            return

        p = msg.pose[i].position
        o = msg.pose[i].orientation
        self.gt_x = p.x
        self.gt_y = p.y
        self.gt_yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        self.gt_valid = True
        self.gt_msgs_seen += 1

        # Static-map distance from base_link to the nearest obstacle, evaluated
        # at the TRUE pose rather than the believed one.
        if self._map_obstacles is not None:
            d = _min_dist_base_link_to_obstacles(
                self._map_obstacles, (self.gt_x, self.gt_y))
            self.min_map_value = d
            if math.isfinite(d):
                self.min_map_overall = min(self.min_map_overall, d)

    # ── pose accessors ──
    def pose(self) -> tuple:
        """(x, y, yaw) for measurement: ground truth when available."""
        if self.gt_valid:
            return self.gt_x, self.gt_y, self.gt_yaw
        return self.robot_x, self.robot_y, self.robot_yaw

    def pose_source(self) -> str:
        return "gazebo_ground_truth" if self.gt_valid else "amcl"

    def localization_error(self) -> float:
        """Ground truth minus belief, in metres. NaN if ground truth is absent."""
        if not self.gt_valid:
            return float("nan")
        return float(math.hypot(self.gt_x - self.robot_x, self.gt_y - self.robot_y))

    def _odom_callback(self, msg: Odometry):
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular

    def set_static_obstacles(self, obstacles_xy: Optional[np.ndarray]) -> None:
        """Cache the static-map occupied cells (map frame) used to measure the
        base_link → nearest-obstacle distance. Set once per run; map is static."""
        self._map_obstacles = obstacles_xy

    def _ensure_scan_geometry(self, msg: LaserScan) -> None:
        """(Re)build the cached per-beam unit directions in base_link.

        Direction of beam i in base_link is (cos(yaw+θ_i), sin(yaw+θ_i)); folding
        the static laser yaw in here means the callback only scales by range and
        adds the laser translation. Recomputed only when the scan geometry
        (angle_min / increment / beam count) changes — i.e. essentially once.
        """
        n = len(msg.ranges)
        key = (msg.angle_min, msg.angle_increment, n)
        if key == self._scan_key:
            return
        ang = (msg.angle_min
               + np.arange(n, dtype=np.float64) * msg.angle_increment
               + self._laser_yaw)
        self._scan_ux = np.cos(ang)
        self._scan_uy = np.sin(ang)
        self._scan_key = key

    def _scan_callback(self, msg: LaserScan):
        """Nearest LiDAR return, in metres from the base_link origin.

        SELF-RETURN FILTERING
        ---------------------
        The raw scan (``/scan_raw`` on TIAGo) contains returns from the robot's
        own structure. Those beams are geometrically fixed in base_link, so the
        per-trial minimum latched onto them and never moved: across the whole
        smoke run this metric had a standard deviation of 0.0002 m (0.1898 to
        0.1908) while the map-based clearance varied with sd 0.26 m, and it was
        identical for the 'carry' and 'tucked' footprints and for the trials
        that ended in a collision. It was measuring the chassis, not the world.

        Two filters are applied to the points *after* they are lifted into
        base_link:
          1. ``self_filter_radius_m`` — drop any hit closer to the base_link
             origin than the robot's own extent. This is the robust one: it is
             a statement about geometry, not about beam indices.
          2. ``angle_mask_deg`` — optional explicit [lo, hi] sectors (in the
             laser frame, degrees) to ignore, for known blind spots.

        ``scan_beams_total`` / ``scan_beams_self`` are accumulated so the next
        run can be checked for this failure mode from the CSV instead of by
        eyeballing the variance.
        """
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)

        if self._angle_mask_rad and valid.any():
            self._ensure_scan_geometry(msg)
            n = len(ranges)
            beam_ang = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment
            for lo, hi in self._angle_mask_rad:
                valid &= ~((beam_ang >= lo) & (beam_ang <= hi))

        if not valid.any():
            self.min_scan_value = None
            return

        self._ensure_scan_geometry(msg)
        r = ranges[valid]
        # Lift valid hits into base_link, then take the distance from the
        # base_link origin (0, 0) to the nearest hit — NOT to the footprint.
        px = self._laser_tx + r * self._scan_ux[valid]
        py = self._laser_ty + r * self._scan_uy[valid]
        dists = np.sqrt(px * px + py * py)

        n_total = int(dists.size)
        keep = dists >= self._self_filter_radius
        n_self = n_total - int(keep.sum())

        self.scan_beams_total += n_total
        self.scan_beams_self += n_self

        if not keep.any():
            # Everything the sensor can see is inside our own footprint radius.
            # Report "unknown" rather than a bogus small number.
            self.min_scan_value = None
            return

        d = float(dists[keep].min())
        self.min_scan_value = d
        self.min_scan_overall = min(self.min_scan_overall, d)

    def _joint_state_callback(self, msg):
        """Latch the most recent joint positions, keyed by joint name.

        Used by the orchestrator to confirm the arm physically reached the pose
        implied by the footprint treatment. A FollowJointTrajectory action
        reporting SUCCEEDED is not on its own evidence that the arm arrived.
        """
        try:
            for name, pos in zip(msg.name, msg.position):
                self.joint_positions[name] = float(pos)
            self.joint_states_stamp = time.time()
        except (TypeError, ValueError):
            pass

    def _risk_state_callback(self, msg: Float64MultiArray):
        try:
            rec = RiskStateRecord.from_array(time.time(), list(msg.data))
        except ValueError as e:
            self.get_logger().warn(f"Invalid risk state data: {e}")
            return
        self.latest_risk_state = rec
        self.risk_state_history.append(rec)
        
    def _bt_log_callback(self, msg):
        """Record every BT node transition into FAILURE.

        navigate_w_replanning_only.xml has no recovery nodes, so any FAILURE
        aborts the whole navigation — but the log still distinguishes a planner
        failure from a controller failure, which the action result does not.
        """
        for ev in msg.event_log:
            if ev.current_status == "FAILURE":
                self.bt_failures.append({
                    "t": time.time(),
                    "node": ev.node_name,
                    "previous_status": ev.previous_status,
                })

    def _collision_callback(self, msg: Bool):
        # Count EVERY message, not just the positive ones. A trial that ends
        # with collision_msgs_seen == 0 tells us nothing about whether a
        # collision happened — it tells us the monitor was not publishing.
        self.collision_msgs_seen += 1
        if msg.data:
            self.get_logger().warn("Collision detected by Gazebo plugin!")
            self.is_collided = True

    def _collision_info_callback(self, msg):
        
        self.get_logger().warn("Collision info received!")
        self.collision_links.append({"t": time.time(), "info": msg.data})

    # ── recording ──
    def record_sample(self, footprint_cost: float):
        """Append the current robot state + latest risk state to the buffer."""
        lin = self.linear_velocity
        ang = self.angular_velocity
        gx, gy, gyaw = self.pose()
        entry = {
            "timestamp": time.time(),
            # Ground truth. Every downstream outcome uses this key.
            "pose": [gx, gy, gyaw],
            # What the robot thought at the same instant. The difference is
            # localization error, not navigation error.
            "pose_believed": [self.robot_x, self.robot_y, self.robot_yaw],
            "footprint_cost": float(footprint_cost),
            "linear_velocity": [lin.x, lin.y, lin.z] if lin else [0.0, 0.0, 0.0],
            "angular_velocity": [ang.x, ang.y, ang.z] if ang else [0.0, 0.0, 0.0],
            "min_scan_value": self.min_scan_value,
        }
        if self.latest_risk_state is not None:
            entry["risk_state"] = asdict(self.latest_risk_state)
        self.controller_path.append(entry)

    def reset(self):
        """Clear per-trial buffers for a fresh trial."""
        self.latest_risk_state = None
        self.risk_state_history = []
        self.controller_path = []
        self.min_scan_value = None
        self.min_scan_overall = float("inf")
        self.min_map_value = None
        self.min_map_overall = float("inf")
        self.replan_events = deque()
        self.is_collided = False
        self.collision_links = []
        self.collision_msgs_seen = 0
        self.gt_msgs_seen = 0
        self.scan_beams_total = 0
        self.scan_beams_self = 0
        self.bt_failures = []

    def get_joint_positions(self, names: list, max_age_sec: float = 5.0) -> Optional[dict]:
        """Latest positions for `names`, or None if /joint_states is stale/absent."""
        if not self.joint_positions:
            return None
        if max_age_sec is not None and self.joint_states_stamp > 0:
            if (time.time() - self.joint_states_stamp) > max_age_sec:
                return None
        return {n: self.joint_positions[n] for n in names if n in self.joint_positions}


# ── Trial runner ──────────────────────────────────────────────────────────────

class TrialRunner:
    """Owns the ROS context + BasicNavigator and executes trials in a loop.

    rclpy and the BasicNavigator are created once; run_trial() reuses them.
    """

    def __init__(
        self,
        timeout_sec: float = 150.0,
        collision_threshold: float = 0.15,
        collect_risk_features: bool = False,
        scan_topic: str = "/scan_raw", #scan
        odom_topic: str = "/mobile_base_controller/odom",
        gazebo_robot_model: str = "tiago",
        output_dir: str = "./rct_data",
        map_yaml_path: str = "",
        record_rate_hz: float = 10.0,
        collision_margin: float = 0.05,
        localization_settle_sec: float = 3.0,
        save_per_trial_json: bool = True,
        generate_plots: bool = True,
        risk_topic: str = "/risk_state",
        bt_xml_path: str = "",
        # Goal tolerances used for the runner's own early-stop check. These MUST
        # be kept equal to the goal_checker values in your controller_server
        # params, otherwise the runner and Nav2 can disagree about "arrived".
        xy_goal_tolerance: float = 0.35,
        yaw_goal_tolerance: float = 0.65,
        # DEFAULT CHANGED to False for the dual-outcome design.
        #
        # When True, the runner cancels the task the instant GROUND TRUTH
        # enters the goal tolerance. That is a true-success detector, and it
        # fires precisely on the trials where success_true == 1 — so Nav2 never
        # renders its own verdict on exactly those trials and success_believed
        # goes missing non-randomly. The disagreement between the two outcomes
        # would then be unmeasurable in the one cell that matters.
        #
        # With it False, the trial runs until Nav2 concludes (or timeout /
        # stall / collision), both verdicts are observed, and arrival is still
        # detected: gt_ever_within_tolerance and t_first_within_tolerance are
        # recorded either way. Cost is wall-clock — trials that arrive early no
        # longer end early.
        #
        # Set True only if you do not need success_believed.
        stop_when_within_tolerance: bool = False,
        # Ignore tolerance hits before this many seconds, so a trial whose start
        # pose already sits inside the goal tolerance is not declared an instant
        # success before the controller has done anything.
        min_trial_time_sec: float = 2.0,
        # Namespaces every artifact this process writes. Without it, re-running
        # trial N overwrites trial N's JSON/plot from a previous run, so older
        # CSV rows end up pointing at a file describing a different trial.
        run_id: str = "",
        # LiDAR self-return rejection (see TrialRunnerNode._scan_callback).
        scan_self_filter_radius_m: float = 0.30,
        scan_angle_mask_deg: Optional[list] = None,
        # Startup guard on the ground-truth publish rate. gazebo_ros_state
        # defaults to 1 Hz, which quantises every pose-derived outcome. 0.0
        # disables the check.
        gt_min_rate_hz: float = 20.0,
        # Stall detection. If the ground-truth pose does not move by at least
        # no_progress_dist_m / no_progress_yaw_rad within this many seconds, end
        # the trial as STUCK instead of waiting out timeout_sec. Set to 0.0 to
        # disable (the stall is then only recorded in longest_stall_sec).
        # NOTE: the detector is suppressed while no new ground-truth messages
        # arrive, so a slow /gazebo/model_states publisher cannot fake a stall.
        no_progress_timeout_sec: float = 20.0,
        no_progress_dist_m: float = 0.10,
        no_progress_yaw_rad: float = 0.20,
    ):
        self.timeout_sec = timeout_sec
        self.collision_threshold = collision_threshold
        self.collect_risk_features = collect_risk_features
        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.gazebo_robot_model = gazebo_robot_model
        self.output_dir = output_dir
        self.map_yaml_path = map_yaml_path
        self.record_period = 1.0 / record_rate_hz if record_rate_hz > 0 else 0.1
        self.collision_margin = collision_margin
        self.localization_settle_sec = localization_settle_sec
        self.save_per_trial_json = save_per_trial_json
        self.generate_plots = generate_plots
        self.risk_topic = risk_topic
        self.bt_xml_path = bt_xml_path
        self.xy_goal_tolerance = xy_goal_tolerance
        self.yaw_goal_tolerance = yaw_goal_tolerance
        self.stop_when_within_tolerance = stop_when_within_tolerance
        self.min_trial_time_sec = min_trial_time_sec
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S")
        self.scan_self_filter_radius_m = scan_self_filter_radius_m
        self.scan_angle_mask_deg = scan_angle_mask_deg
        self.gt_min_rate_hz = gt_min_rate_hz
        self.no_progress_timeout_sec = no_progress_timeout_sec
        self.no_progress_dist_m = no_progress_dist_m
        self.no_progress_yaw_rad = no_progress_yaw_rad

        if not rclpy.ok():
            rclpy.init()

        # Raw /plan messages buffered during navigation, analyzed after the trial
        # ends so the recording loop is never blocked by costmap math.
        self._pending_replans: list[dict] = []

        # Static-map occupied cells (map frame), loaded lazily once and reused
        # across trials for the pose-based footprint clearance.
        self._static_obstacles: Optional[np.ndarray] = None
        self._static_obstacles_loaded = False

        # Persistent recorder node + navigator + costmap client (created once).
        self._recorder = TrialRunnerNode(
            scan_topic, odom_topic, risk_topic, collect_risk_features,
            self_filter_radius_m=scan_self_filter_radius_m,
            angle_mask_deg=scan_angle_mask_deg,
            gazebo_robot_model=gazebo_robot_model)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._recorder)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._navigator = BasicNavigator()
        logger.info("Waiting for Nav2 to become active...")
        self._navigator.waitUntilNav2Active()
        logger.info("Nav2 active ✓")

        self._costmap_cli = self._navigator.create_client(
            GetCostmap, "/global_costmap/get_costmap")

        if self.save_per_trial_json:
            os.makedirs(os.path.join(self.output_dir, "trials"), exist_ok=True)
            
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

        # AMCL subscribes to /initialpose with SystemDefaultsQoS (reliable, volatile).
        # TRANSIENT_LOCAL on our side is compatible (offered >= requested) and latches,
        # so if we publish before AMCL connects, it still receives it on connect.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._initpose_pub = self._navigator.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", qos)

        self._assert_collision_channel()
        self._wait_for_ground_truth(min_rate_hz=self.gt_min_rate_hz)

    def _assert_collision_channel(self, timeout_sec: float = 10.0):
        """Refuse to start if nothing publishes /gazebo/collision.

        Without this the entire run records collision=0 for every trial and the
        result is indistinguishable from a genuinely collision-free run.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._recorder.count_publishers("/gazebo/collision") > 0:
                logger.info("  /gazebo/collision has a publisher ✓")
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"No publisher on /gazebo/collision after {timeout_sec:.0f}s. The "
            "gazebo_collision_monitor plugin is not loaded, so every trial would "
            "silently record collision=0. Load the plugin, or set "
            "require_collision_channel: false to collect without collision labels."
        )

    def _wait_for_ground_truth(self, timeout_sec: float = 10.0,
                               min_rate_hz: float = 20.0):
        """Block until /gazebo/model_states yields the robot, then check its rate.

        Two distinct failure modes, reported separately because the fixes differ:

        1. The topic never names the robot — wrong gazebo_robot_model, or the
           gazebo_ros_state plugin is not loaded. Outcomes would silently fall
           back to /amcl_pose.
        2. The topic is alive but slow. gazebo_ros_state defaults to
           <update_rate>1.0</update_rate>; at that rate the pose is only sampled
           once per second, so path_length_m, final_xy_error and
           min_obstacle_distance are quantised exactly as badly as AMCL would
           quantise them — and nothing in the CSV distinguishes the two.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._recorder.gt_valid:
                logger.info(
                    f"  Ground-truth pose from /gazebo/model_states ✓ "
                    f"(model '{self.gazebo_robot_model}')")
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(
                f"No ground-truth pose for model '{self.gazebo_robot_model}' on "
                f"/gazebo/model_states after {timeout_sec:.0f}s. Outcomes would be "
                "measured from /amcl_pose, which is quantised to AMCL's update_min_d "
                "(~0.28 m) and cannot support trajectory or final-error measurement. "
                "Check gazebo_robot_model matches the name in Gazebo."
            )

        if min_rate_hz <= 0:
            return
        n0, t0 = self._recorder.gt_msgs_seen, time.time()
        time.sleep(2.0)
        rate = (self._recorder.gt_msgs_seen - n0) / max(time.time() - t0, 1e-6)
        if rate < min_rate_hz:
            raise RuntimeError(
                f"/gazebo/model_states is publishing at {rate:.1f} Hz, below the "
                f"required {min_rate_hz:.0f} Hz. The gazebo_ros_state plugin "
                "defaults to <update_rate>1.0</update_rate> — set it to 50 in the "
                "world SDF. At this rate path_length_m, final_xy_error and "
                "min_obstacle_distance are all quantised and the run is unusable. "
                "Set gt_min_rate_hz: 0.0 to bypass this check deliberately."
            )
        logger.info(f"  Ground-truth pose rate {rate:.1f} Hz ✓")

    # ── public API (called by the orchestrator) ──

    def get_arm_joint_positions(self, names: list) -> Optional[dict]:
        """Latest /joint_states positions for `names`, or None if unavailable.

        Used by the orchestrator to confirm the physical arm pose matches the
        footprint treatment for the trial.
        """
        return self._recorder.get_joint_positions(names)
    def run_trial(self, trial_id: int, start_pose: dict, goal_pose: dict,
                  params: dict) -> TrialResult:
        """Execute one trial and return its (CSV-friendly) TrialResult."""
        from rct_collector.scripts.param_space import ParameterSpace

        rec = self._recorder
        rec.reset()

        result = TrialResult(
            trial_id=trial_id,
            start_x=start_pose["x"], start_y=start_pose["y"], start_yaw=start_pose["yaw"],
            goal_x=goal_pose["x"], goal_y=goal_pose["y"], goal_yaw=goal_pose["yaw"],
            params=ParameterSpace().flatten(params),
            # Stamped up front so they are present even on the early-return
            # paths below, and so every outcome stays re-derivable from the row.
            xy_goal_tolerance_used=self.xy_goal_tolerance,
            yaw_goal_tolerance_used=self.yaw_goal_tolerance,
        )

        # 1. Teleport in Gazebo, then localize.
        result.teleport_ok = int(self._teleport_robot(start_pose))
        time.sleep(1.0)  # let physics settle
        init_pose = self._make_pose_stamped(start_pose)
        # self._navigator.setInitialPose(init_pose)
        self._publish_initial_pose(start_pose, cov=None, timeout_sec=2.0)
        self._wait_for_localization()
        
        # Wipe obstacle marks left by the previous trial / pre-teleport pose.
        # Recovery behaviors (which normally do this) are absent from our BT.
        self._navigator.clearAllCostmaps() 

        # 2. Resolve this trial's footprint and build the collision checker.
        footprint = self._resolve_footprint(params)
        rec.set_static_obstacles(self._get_static_obstacles())  # base_link→obstacle (map)
        checker, costmap = self._build_footprint_checker(footprint)
        collision_distance = self.collision_threshold
        if checker is not None:
            collision_distance = checker.geometry.inscribed_radius + self.collision_margin
            print (f"  Footprint inscribed radius ============= {checker.geometry.inscribed_radius:.3f} m, ")
            print (f"  collision distance threshold =========== {collision_distance:.3f} m")

        # 3. Plan + analyse the global path.
        goal_stamped = self._make_pose_stamped(goal_pose)
        # One-time initial plan, purely for the baseline global-path record.
        # Does NOT drive execution — goToPose() below computes its own first plan internally.
        initial_path = self._navigator.getPath(init_pose, goal_stamped)
        if initial_path is None or not initial_path.poses:
            result.status = "PLANNING_FAILED"
            result.failure_reason = "PLANNING_FAILED_INITIAL"
            # No navigation happened, so both outcomes are genuinely
            # unobserved. Scoring here writes them as blank rather than 0.
            result._score_outcomes()
            self._finalize_json(rec, result, params, global_path=[])
            logger.error("  Initial global planning failed — no path.")
            return result

        global_path_data, result.initial_global_path_length_m, result.min_global_obstacle_distance = \
            self._analyze_global_path(initial_path, checker, costmap)

        # 4. Follow the smoothed path while recording.
        # smoothed = self._navigator.smoothPath(path) or path
        # self._navigator.followPath(smoothed)
        self._navigator.goToPose(goal_stamped, behavior_tree=self.bt_xml_path)
        self._record_navigation(rec, checker, collision_distance, result, checker, costmap)

        # 5. Classify outcome.
        #
        # Precedence: collision > runner decision > Nav2 action result.
        #
        # getResult() is consulted ONLY when the recording loop exited because
        # Nav2 finished the task on its own (isTaskComplete() went True). On the
        # runner-terminated paths (early in-tolerance SUCCESS, TIMEOUT) we called
        # cancelTask() and broke out, and getResult() is then unreliable in two
        # ways: (a) BasicNavigator only writes self.status inside
        # isTaskComplete(), which never returned True on those paths, so the
        # value can be left over from a PREVIOUS trial (the navigator is reused
        # across trials); (b) the goal may have latched SUCCEEDED just before the
        # async cancel landed, since a terminal goal cannot be un-terminated.
        # Either way it previously overwrote "TIMEOUT" with "SUCCESS".
        
        # The BT publishes its terminal transition slightly after the action
        # result latches. Without this pause the log that names the failing node
        # arrives after classification and the row reads UNKNOWN.
        time.sleep(0.25)

        nav_result = self._navigator.getResult()
        result.nav2_result = getattr(nav_result, "name", str(nav_result))
        if result.collision:
            result.status = "COLLISION"
        elif result.terminated_by_runner:
            # Authoritative — already set at the break site. Log the discrepancy
            # so a disagreement with Nav2 stays visible instead of silent.
            if nav_result == TaskResult.SUCCEEDED and result.status != "SUCCESS":
                logger.warning(
                    f"  Nav2 reported SUCCEEDED but the runner terminated this "
                    f"trial as {result.status}; keeping {result.status}. "
                    "(Stale/raced BasicNavigator status.)")
        elif nav_result == TaskResult.SUCCEEDED:
            result.status = "SUCCESS"
        elif nav_result == TaskResult.CANCELED:
            result.status = "CANCELED"
        elif nav_result == TaskResult.FAILED:
            result.status = "FAILED"
        else:
            result.status = result.status or "UNKNOWN"

        self._classify_failure(rec, result)

        # 6. Goal distance remaining, final xy error, and final yaw error based on
        # the last pose of the controller path.
        if rec.controller_path:
            last_pose = rec.controller_path[-1]["pose"]   # ground truth
            last_x, last_y, last_yaw = last_pose[0], last_pose[1], last_pose[2]
            # The robot's OWN estimate at the same instant. This is what Nav2's
            # goal checker was reading, and it is the only version a real robot
            # could compute onboard — so it is recorded as a first-class
            # outcome, not merely as a diagnostic.
            believed = rec.controller_path[-1].get("pose_believed")
        else:
            last_x, last_y, last_yaw = rec.pose()
            believed = [rec.robot_x, rec.robot_y, rec.robot_yaw]

        dx = goal_pose["x"] - last_x
        dy = goal_pose["y"] - last_y
        result.final_xy_error = float(math.hypot(dx, dy))
        result.goal_distance_remaining = result.final_xy_error

        dyaw = goal_pose["yaw"] - last_yaw
        result.final_yaw_error = float(math.atan2(math.sin(dyaw), math.cos(dyaw)))

        if believed and len(believed) >= 3:
            bdx = goal_pose["x"] - believed[0]
            bdy = goal_pose["y"] - believed[1]
            result.believed_final_xy_error = float(math.hypot(bdx, bdy))
            bdyaw = goal_pose["yaw"] - believed[2]
            result.believed_final_yaw_error = float(
                math.atan2(math.sin(bdyaw), math.cos(bdyaw)))

        result.min_obstacle_distance = rec.min_scan_overall
        result.min_map_obstacle_distance = rec.min_map_overall
        result.run_id = self.run_id

        # ── pose provenance and QA ────────────────────────────────────────────
        result.pose_source = rec.pose_source()
        result.gt_msgs_seen = rec.gt_msgs_seen
        result.localization_error_m = rec.localization_error()

        # Regression guard on pose quantisation. Under /amcl_pose this was
        # 0.078 on trial 3 (37 distinct poses in 477 samples); with ground truth
        # it should sit above ~0.95. A low value means the pose source has
        # silently reverted.
        poses = [tuple(smp["pose"]) for smp in rec.controller_path]
        result.unique_pose_fraction = (len(set(poses)) / len(poses)) if poses else 0.0
        if poses and result.unique_pose_fraction < 0.5:
            logger.warning(
                f"  Only {result.unique_pose_fraction:.0%} of recorded poses are "
                "distinct — the pose source looks quantised. Path length and "
                "final error are unreliable for this trial.")

        # ── collision-channel liveness ────────────────────────────────────────
        result.collision_msgs_seen = rec.collision_msgs_seen
        result.collision_channel_silent = int(rec.collision_msgs_seen == 0)
        if result.collision_channel_silent:
            logger.error(
                "  No messages on /gazebo/collision during this trial. "
                "collision=0 here means 'not observed', NOT 'did not happen'. "
                "Row flagged collision_channel_silent=1.")

        if not result.teleport_ok:
            logger.error(
                "  Start pose was never confirmed in Gazebo; start_x/start_y may "
                "not describe where this trial actually began (teleport_ok=0).")

        # LiDAR self-return diagnostics: surfaced per trial so a scan dominated
        # by the robot's own structure is visible in the CSV rather than only
        # detectable by noticing that min_obstacle_distance never varies.
        result.scan_beams_total = rec.scan_beams_total
        result.scan_beams_self = rec.scan_beams_self
        result.min_obstacle_distance_valid = math.isfinite(rec.min_scan_overall)
        if rec.scan_beams_total:
            self_frac = rec.scan_beams_self / rec.scan_beams_total
            if self_frac > 0.5:
                logger.warning(
                    f"  {self_frac:.0%} of LiDAR returns fell inside the "
                    f"self-filter radius ({self.scan_self_filter_radius_m} m). "
                    f"Check the scan topic and the filter radius."
                )
        if not result.min_obstacle_distance_valid:
            logger.warning(
                "  No usable LiDAR returns this trial; min_obstacle_distance "
                "recorded as invalid rather than as a placeholder value.")

        # ── dual outcome scoring ─────────────────────────────────────────────
        result._score_outcomes()

        if result.outcome_agreement == "FALSE_SUCCESS":
            logger.warning(
                f"  FALSE SUCCESS: the robot believes it arrived "
                f"(believed error {result.believed_final_xy_error:.3f} m) but "
                f"ground truth puts it {result.final_xy_error:.3f} m from the "
                f"goal (tolerance {result.xy_goal_tolerance_used:.3f} m). "
                f"Localization error {result.localization_error_m:.3f} m.")
        elif result.outcome_agreement == "MISSED_SUCCESS":
            logger.info(
                f"  Missed success: ground truth is within tolerance "
                f"({result.final_xy_error:.3f} m) but the robot believes it is "
                f"{result.believed_final_xy_error:.3f} m away.")

        # Nav2 claiming SUCCEEDED while its OWN estimate is outside tolerance is
        # a different fault from localization drift: the action result and the
        # goal checker disagree with each other, not with the world.
        if (result.success_believed == 1
                and result.believed_within_tolerance == 0):
            logger.error(
                f"  Nav2 returned SUCCEEDED but its own believed pose is "
                f"{result.believed_final_xy_error:.3f} m from the goal "
                f"(tolerance {result.xy_goal_tolerance_used:.3f} m). This is not "
                "localization error — the action result does not match the goal "
                "checker. Treat this row's success_believed as suspect.")

        # 7. Persist the full time-series JSON.
        self._finalize_json(rec, result, params, global_path=global_path_data)

        def _fmt(v):
            return "-" if v is None else str(v)

        logger.info(
            f"  Trial {trial_id} complete: status={result.status}, "
            f"success_true={_fmt(result.success_true)}, "
            f"believed={_fmt(result.believed_within_tolerance)}, "
            f"nav2={_fmt(result.success_believed)} "
            f"[{result.outcome_agreement}], "
            f"time={result.travel_time_sec:.1f}s, path={result.path_length_m:.2f}m, "
            f"risk_samples={result.num_risk_samples}"
        )
        return result

    # Leaf nodes of navigate_w_replanning_only.xml. Container nodes
    # (PipelineSequence, RateController) also transition to FAILURE, but only as
    # a consequence of a leaf failing, so they are poor explanations.
    _BT_LEAF_REASONS = {
        "ComputePathToPose": "BT_PLANNER_FAILED",
        "FollowPath": "BT_CONTROLLER_FAILED",
    }

    def _classify_failure(self, rec: TrialRunnerNode, result: TrialResult) -> None:
        """Fill failure_reason / bt_failed_node from the behaviour-tree log.

        Runner-decided reasons (STUCK_NO_PROGRESS, RUNNER_TIMEOUT) are already
        set at the break site and are authoritative — the BT log is only
        consulted for outcomes Nav2 decided on its own.
        """
        failures = list(rec.bt_failures)
        counts: dict[str, int] = {}
        for f in failures:
            counts[f["node"]] = counts.get(f["node"], 0) + 1
        result.bt_failure_detail = ";".join(
            f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

        # Prefer the last LEAF failure; container nodes only echo it.
        leaf = next((f["node"] for f in reversed(failures)
                     if f["node"] in self._BT_LEAF_REASONS), None)
        result.bt_failed_node = leaf or (failures[-1]["node"] if failures else "")

        if result.failure_reason != "NONE":
            return                                   # runner already decided
        if result.collision:
            result.failure_reason = "COLLISION"
        elif result.status == "SUCCESS":
            result.failure_reason = "NONE"
        elif leaf is not None:
            result.failure_reason = self._BT_LEAF_REASONS[leaf]
        elif failures:
            result.failure_reason = "BT_OTHER_FAILED"
        elif result.status in ("FAILED", "CANCELED", "UNKNOWN"):
            result.failure_reason = "UNKNOWN"

        if result.failure_reason == "UNKNOWN" and BehaviorTreeLog is None:
            logger.warning(
                "  nav2_msgs.msg.BehaviorTreeLog is unavailable, so no failure "
                "reason could be recovered. Rebuild against a nav2_msgs that "
                "provides it, or failure_reason will be UNKNOWN for every row.")

    def shutdown(self):
        try:
            self._executor.shutdown()
            self._recorder.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    # ── navigation recording loop ──
    # def _record_navigation(self, rec: TrialRunnerNode, checker, collision_distance: float,
    #                        result: TrialResult):
    #     t_start = time.time()
    #     last_record = t_start
    #     prev_pose = [rec.robot_x, rec.robot_y]
    #     local_len = 0.0

    #     while not self._navigator.isTaskComplete():
    #         rclpy.spin_once(rec, timeout_sec=0.05)
    #         now = time.time()
    #         elapsed = now - t_start

    #         footprint_cost = 0.0
    #         if checker is not None:
    #             footprint_cost = checker.footprintCostAtPose(
    #                 rec.robot_x, rec.robot_y, rec.robot_yaw)

    #         # Record at the target rate (not every loop iteration).
    #         if now - last_record >= self.record_period:
    #             rec.record_sample(footprint_cost)
    #             last_record = now
    #             local_len += math.hypot(rec.robot_x - prev_pose[0], rec.robot_y - prev_pose[1])
    #             prev_pose = [rec.robot_x, rec.robot_y]

    #         # Collision: footprint in lethal cell AND LiDAR confirms proximity.
    #         # (Fallback to LiDAR-only when no checker/costmap is available.)
    #         # hit = False
    #         # if checker is not None:
    #         #     hit = (footprint_cost >= INSCRIBED_INFLATED_OBSTACLE
    #         #            and rec.min_scan_value is not None
    #         #            and rec.min_scan_value < collision_distance)
    #         # elif rec.min_scan_value is not None:
    #         #     hit = rec.min_scan_value < self.collision_threshold
    #         if self._recorder.is_collided:
    #             rec.record_sample(footprint_cost)
    #             self._navigator.cancelTask()
    #             result.collision = True
    #             logger.warning(
    #                 f"  COLLISION: footprint_cost={footprint_cost:.0f}, "
    #                 f"min_scan={rec.min_scan_value:.3f}m < {collision_distance:.3f}m")
    #             break

    #         # Timeout guard.
    #         if elapsed > self.timeout_sec:
    #             self._navigator.cancelTask()
    #             result.status = "TIMEOUT"
    #             logger.warning(f"  Trial timed out after {elapsed:.1f}s")
    #             break

    #     result.travel_time_sec = time.time() - t_start
    #     result.path_length_m = local_len
    #     result.num_controller_samples = len(rec.controller_path)
    #     result.num_risk_samples = len(rec.risk_state_history)
    
    def _record_navigation(self, rec: TrialRunnerNode, checker, collision_distance: float,
                           result: TrialResult, footprint_checker=None, costmap=None):
        rec._recording = True
        t_start = time.time()
        result.t_nav_start = t_start
        last_record = t_start
        _px, _py, _pyaw = rec.pose()
        prev_pose = [_px, _py]
        local_len = 0.0
        self._pending_replans = []
        stall_ref = (_px, _py, _pyaw)      # pose the stall window is measured from
        stall_t0 = t_start
        stall_gt0 = rec.gt_msgs_seen

        while not self._navigator.isTaskComplete():
            time.sleep(0.01)
            now = time.time()
            elapsed = now - t_start

            # Drain /plan messages produced by the BT's RateController.
            # Buffer only — analysis is deferred until after navigation ends,
            # so ~270 footprintCostAtPose calls per replan never block this loop.
            while rec.replan_events:
                self._pending_replans.append(rec.replan_events.popleft())

            # Ground truth throughout: footprint cost, path length and the
            # tolerance check are all outcome measurements and must not be
            # contaminated by localization error.
            rx, ry, ryaw = rec.pose()

            footprint_cost = 0.0
            if checker is not None:
                footprint_cost = checker.footprintCostAtPose(rx, ry, ryaw)

            if now - last_record >= self.record_period:
                rec.record_sample(footprint_cost)
                last_record = now
                local_len += math.hypot(rx - prev_pose[0], ry - prev_pose[1])
                prev_pose = [rx, ry]

            # Stall detection. The window restarts whenever the robot moves in
            # translation OR rotation, so a legitimate in-place yaw correction
            # near the goal is not mistaken for being stuck.
            _moved_xy = math.hypot(rx - stall_ref[0], ry - stall_ref[1])
            _moved_yaw = abs(math.atan2(math.sin(ryaw - stall_ref[2]),
                                        math.cos(ryaw - stall_ref[2])))
            if (_moved_xy >= self.no_progress_dist_m
                    or _moved_yaw >= self.no_progress_yaw_rad):
                stall_ref = (rx, ry, ryaw)
                stall_t0 = now
                stall_gt0 = rec.gt_msgs_seen
            else:
                stall_len = now - stall_t0
                result.longest_stall_sec = max(result.longest_stall_sec, stall_len)
                # Only trust the stall if ground truth actually refreshed during
                # the window. A silent pose topic looks identical to a stuck
                # robot, and must not be scored as one.
                if (self.no_progress_timeout_sec > 0
                        and stall_len > self.no_progress_timeout_sec
                        and rec.gt_msgs_seen > stall_gt0):
                    rec.record_sample(footprint_cost)
                    self._navigator.cancelTask()
                    result.status = "STUCK"
                    result.failure_reason = "STUCK_NO_PROGRESS"
                    result.terminated_by_runner = True
                    logger.warning(
                        f"  No progress for {stall_len:.1f}s "
                        f"(moved {_moved_xy:.3f}m / {_moved_yaw:.3f}rad) — "
                        "ending trial as STUCK.")
                    break

            # Early termination: the robot is inside the goal tolerance.
            # Requires xy AND yaw to be satisfied at the SAME pose, so the
            # controller cannot loiter doing repeated heading corrections near
            # the goal (the behaviour that was tripping the timeout). Measured
            # from ground truth — the same source used for the final error
            # below — so a SUCCESS declared here always has final error within
            # tolerance. Under the old /amcl_pose source this branch could
            # essentially never fire: it tested a 0.25 m threshold against a
            # pose quantised to 0.28 m, and terminated_by_runner was False in
            # 20 of 20 smoke trials.
            # Arrival is ALWAYS observed and timestamped, whether or not it
            # ends the trial. gt_ever_within_tolerance separates "never got
            # there" from "got there and then drove away again", which the
            # terminal-pose test on its own conflates.
            if elapsed >= self.min_trial_time_sec:
                _dx = result.goal_x - rx
                _dy = result.goal_y - ry
                _xy_err = math.hypot(_dx, _dy)
                _dyaw = result.goal_yaw - ryaw
                _yaw_err = abs(math.atan2(math.sin(_dyaw), math.cos(_dyaw)))
                _arrived = (_xy_err <= self.xy_goal_tolerance
                            and _yaw_err <= self.yaw_goal_tolerance)

                if _arrived and not result.gt_ever_within_tolerance:
                    result.gt_ever_within_tolerance = 1
                    result.t_first_within_tolerance = elapsed
                    logger.info(
                        f"  Ground truth entered goal tolerance at {elapsed:.1f}s "
                        f"(xy={_xy_err:.3f}m, yaw={_yaw_err:.3f}rad).")

                # Terminating here censors Nav2's verdict on exactly the trials
                # that succeeded, so it is off by default. See the constructor.
                if _arrived and self.stop_when_within_tolerance:
                    rec.record_sample(footprint_cost)
                    self._navigator.cancelTask()
                    result.status = "SUCCESS"
                    result.terminated_by_runner = True
                    logger.info(
                        "  stop_when_within_tolerance=True — ending trial here. "
                        "success_believed will be censored for this row.")
                    break

            # Ground-truth collision from Gazebo physics contacts (gazebo_collision_monitor
            # plugin -> /gazebo/collision -> _collision_callback sets is_collided).
            # NOT costmap-based: inflation_radius is a treatment, so costmap cost would be
            # an endogenous outcome label.
            if self._recorder.is_collided:
                rec.record_sample(footprint_cost)
                self._navigator.cancelTask()
                result.collision = True
                logger.warning("  COLLISION detected by Gazebo plugin (/gazebo/collision)")
                break

            if elapsed > self.timeout_sec:
                self._navigator.cancelTask()
                result.status = "TIMEOUT"
                result.failure_reason = "RUNNER_TIMEOUT"
                result.terminated_by_runner = True
                logger.warning(f"  Trial timed out after {elapsed:.1f}s")
                break

        else:
            # Loop exited because Nav2 finished on its own (no break). Every
            # break path already recorded a final sample; this path did not, so
            # the last sample could be up to record_period old. Both outcomes
            # are computed from the LAST sample's pose and pose_believed, so a
            # stale terminal pose biases them directly — take one more now.
            rx, ry, ryaw = rec.pose()
            fc = checker.footprintCostAtPose(rx, ry, ryaw) if checker is not None else 0.0
            rec.record_sample(fc)

        rec._recording = False
        result.t_nav_end = time.time()
        result.travel_time_sec = result.t_nav_end - t_start
        result.path_length_m = local_len
        result.num_controller_samples = len(rec.controller_path)
        result.num_risk_samples = len(rec.risk_state_history)
        result.collision_links = [
            {"t": entry["t"] - t_start, "info": entry["info"]}
            for entry in rec.collision_links
        ]

        # Navigation is over; now it is safe to do the expensive costmap math.
        # Drain anything that arrived between the last loop iteration and task
        # completion, then analyze every buffered plan exactly once.
        while rec.replan_events:
            self._pending_replans.append(rec.replan_events.popleft())

        replan_analyses = [
            self._analyze_replan(ev, footprint_checker, costmap)
            for ev in self._pending_replans
        ]
        result.global_planner_ticks = len(replan_analyses)
        result.replan_history = replan_analyses

    # ── helpers ──
    def _make_pose_stamped(self, pose: dict) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self._navigator.get_clock().now().to_msg()
        ps.pose.position.x = float(pose["x"])
        ps.pose.position.y = float(pose["y"])
        qx, qy, qz, qw = yaw_to_quaternion(float(pose["yaw"]))
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps


    def _publish_initial_pose(self, pose: dict, cov=None, timeout_sec=5.0):
        node = self._navigator
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(pose["x"])
        msg.pose.pose.position.y = float(pose["y"])
        qx, qy, qz, qw = yaw_to_quaternion(float(pose["yaw"]))
        (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) = qx, qy, qz, qw

        c = [0.0] * 36
        if cov is None:                      # ground truth -> near-zero, NOT exactly 0
            c[0] = c[7] = c[35] = 1e-9       # var(x)=idx0, var(y)=idx7, var(yaw)=idx35
        else:
            c = list(cov)
        msg.pose.covariance = c

        # Wait for AMCL to connect its /initialpose subscription.
        start = time.time()
        while (self._initpose_pub.get_subscription_count() == 0
            and time.time() - start < timeout_sec):
            rclpy.spin_once(node, timeout_sec=0.1)
        if self._initpose_pub.get_subscription_count() == 0:
            logger.warning("  /initialpose has no subscriber (AMCL not up?); publishing anyway.")

        # Stamp at publish time; send a few times, spinning to flush.
        for _ in range(3):
            msg.header.stamp = node.get_clock().now().to_msg()
            self._initpose_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.1)





    def _wait_for_localization(self):
        """Give AMCL time to converge after the initial pose is set.

        The recorder node is spun by the background executor, so this must NOT
        call rclpy.spin_once(self._recorder, ...) — a node may belong to only
        one executor. Just wait; callbacks are being serviced on the other thread.
        """
        logger.info("  Waiting for localization to converge...")
        time.sleep(self.localization_settle_sec)
        logger.info(f"  Localization ready (waited {self.localization_settle_sec:.0f}s)")

    def _resolve_footprint(self, params: dict) -> list:
        """Map the sampled arm/footprint label (e.g. 'tucked') to its polygon."""
        import ast
        from rct_collector.scripts.param_space import ARM_CONFIGS

        label = None
        for kv in params.values():
            if "footprint" in kv:
                label = kv["footprint"]
                break
        if label is not None and label in ARM_CONFIGS:
            return ast.literal_eval(ARM_CONFIGS[label]["footprint"])
        if isinstance(label, str):
            try:
                return ast.literal_eval(label)  # already a polygon literal
            except (ValueError, SyntaxError):
                pass
        # Fallback: ~TIAGo base circle (octagon).
        logger.warning("  No footprint label in params; using default base footprint.")
        return [[0.27, 0.0], [0.19, 0.19], [0.0, 0.27], [-0.19, 0.19],
                [-0.27, 0.0], [-0.19, -0.19], [0.0, -0.27], [0.19, -0.19]]

    def _get_static_obstacles(self) -> Optional[np.ndarray]:
        """Occupied cells of the static map (map frame), loaded once and cached.

        Returns None if no map_yaml_path is configured or the map cannot be
        read — the pose-based footprint clearance then simply stays disabled.
        """
        if self._static_obstacles_loaded:
            return self._static_obstacles
        self._static_obstacles_loaded = True
        if not self.map_yaml_path or not os.path.exists(self.map_yaml_path):
            logger.warning("  No static map available; map-based clearance disabled.")
            return None
        try:
            obstacles, res = _static_map_obstacles(self.map_yaml_path)
            logger.info(f"  Static map: {len(obstacles)} occupied cells "
                        f"@ {res:.3f} m/cell for footprint clearance.")
            self._static_obstacles = obstacles
        except Exception as e:
            logger.warning(f"  Could not load static map obstacles: {e}")
            self._static_obstacles = None
        return self._static_obstacles

    def _build_footprint_checker(self, footprint: list):
        """Fetch the current global costmap and build a checker for `footprint`.
        Returns (checker, costmap_response) — both None if unavailable."""
        if not _HAVE_FOOTPRINT_CHECKER:
            return None, None
        # Wait for the costmap to repopulate after clearAllCostmaps(); otherwise
        # we snapshot an all-unknown (255) grid and every footprint cost is 255.
        costmap = self._get_costmap(wait_until_populated=True)
        if costmap is None:
            logger.warning("  Could not fetch global costmap; footprint cost disabled.")
            return None, None
        checker = FootprintCollisionChecker()
        checker.setCostmap(costmap)
        checker.setFootprint([tuple(pt) for pt in footprint])
        g = checker.geometry
        logger.info(
            f"  Footprint geometry: inscribed={g.inscribed_radius:.3f}m, "
            f"circumscribed={g.circumscribed_radius:.3f}m")
        return checker, costmap

    def _get_costmap(self, wait_until_populated: bool = False, timeout_sec: float = 5.0):
        """Fetch the current global costmap via the GetCostmap service.

        When wait_until_populated is True, keep re-fetching until the costmap has
        repopulated with real information. This matters right after
        clearAllCostmaps(): the clear resets every cell to NO_INFORMATION (255)
        and the static + obstacle layers only restamp on the next update cycle.
        Snapshotting during that window yields an all-255 grid, which makes every
        footprintCostAtPose read 255 (and every min-obstacle distance read 0)."""
        if not self._costmap_cli.wait_for_service(timeout_sec=5.0):
            return None
        deadline = time.time() + timeout_sec
        while True:
            future = self._costmap_cli.call_async(GetCostmap.Request())
            rclpy.spin_until_future_complete(self._navigator, future, timeout_sec=10.0)
            cm = future.result()
            if cm is None or not wait_until_populated:
                return cm
            data = np.asarray(cm.map.data, dtype=np.uint8)
            # Populated once any cell carries information other than "unknown" (255).
            if data.size and np.any(data != 255):
                return cm
            if time.time() >= deadline:
                logger.warning(
                    f"  Global costmap still all-unknown {timeout_sec:.1f}s after "
                    "clear; footprint costs may be unreliable.")
                return cm
            time.sleep(0.2)

    def _analyze_replan(self, ev: dict, checker, costmap) -> dict:
        """Analyze one buffered /plan message. Called AFTER navigation ends."""
        path_msg = ev["path_msg"]
        _, path_len, min_dist = self._analyze_global_path(path_msg, checker, costmap)
        return {
            "timestamp": ev["timestamp"],
            "path_length_m": path_len,
            "min_obstacle_distance": min_dist,
            "num_poses": len(path_msg.poses),
        }

    def _analyze_global_path(self, path, checker, costmap):
        """Return (per-pose list, total length, min obstacle distance)."""
        data = []
        total_len = 0.0
        min_dist = float("inf")

        costmap_array = None
        resolution = 0.05
        if checker is not None and costmap is not None:
            meta = costmap.map.metadata
            resolution = meta.resolution
            costmap_array = np.array(costmap.map.data).reshape(
                (meta.size_y, meta.size_x))

        prev = None
        px, py = None, None
        for ps in path.poses:
            x, y = ps.pose.position.x, ps.pose.position.y
            orientation = 0.0 if px is None else math.atan2(y - py, x - px)
            px, py = x, y

            footprint_cost = 0.0
            pose_min_dist = float("inf")
            if checker is not None:
                footprint_cost = checker.footprintCostAtPose(x, y, orientation)
                mx, my = checker.worldToMapValidated(x, y)
                if mx is not None and costmap_array is not None:
                    pose_min_dist = _min_distance_to_obstacle(
                        costmap_array, np.array([my, mx]), resolution)
                    min_dist = min(min_dist, pose_min_dist)

            if prev is not None:
                total_len += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)

            data.append({
                "pose": [x, y, orientation],
                "footprint_cost": float(footprint_cost),
                "min_dist_to_obstacle": pose_min_dist,
            })
        return data, total_len, min_dist

    def _teleport_robot(self, pose: dict, attempts: int = 3,
                        tol_xy: float = 0.10) -> bool:
        """Teleport in Gazebo and VERIFY the robot arrived. Returns success.

        The previous version was fire-and-forget: on a service TIMEOUT it logged
        a warning, skipped the fallback entirely (the fallback was gated on
        returncode != 0, which a TimeoutExpired never reaches) and returned. Two
        of twenty smoke trials hit that path and were still recorded as normal
        rows, one of them as SUCCESS. Verification now uses the ground-truth
        pose already streaming on /gazebo/model_states, so confirming costs
        nothing extra.
        """
        _, _, qz, qw = yaw_to_quaternion(float(pose["yaw"]))
        state = (
            f'{{"state": {{"name": "{self.gazebo_robot_model}", '
            f'"pose": {{"position": {{"x": {pose["x"]}, "y": {pose["y"]}, "z": 0.0}}, '
            f'"orientation": {{"z": {qz}, "w": {qw}}}}}}}}}'
        )
        services = (
            ("/gazebo/set_entity_state", "gazebo_msgs/srv/SetEntityState"),
            ("/set_model_state", "gazebo_msgs/srv/SetModelState"),
        )

        for attempt in range(1, attempts + 1):
            for srv, typ in services:
                try:
                    subprocess.run(["ros2", "service", "call", srv, typ, state],
                                   capture_output=True, text=True, timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"  {srv} timed out (attempt {attempt}/{attempts})")
                    continue
                except Exception as e:
                    logger.warning(f"  {srv} failed: {e}")
                    continue

                if self._verify_gazebo_pose(pose, tol_xy=tol_xy):
                    logger.info(
                        f"  Teleported to ({pose['x']:.2f}, {pose['y']:.2f}) ✓")
                    return True
            time.sleep(1.0)

        gx, gy, _ = self._recorder.pose()
        logger.error(
            f"  TELEPORT UNVERIFIED after {attempts} attempts. Requested "
            f"({pose['x']:.2f}, {pose['y']:.2f}), ground truth reports "
            f"({gx:.2f}, {gy:.2f}). Row flagged teleport_ok=0.")
        return False

    def _verify_gazebo_pose(self, pose: dict, tol_xy: float = 0.10,
                            settle_sec: float = 0.5) -> bool:
        """Confirm the ground-truth pose matches the requested one."""
        deadline = time.time() + settle_sec
        while time.time() < deadline:
            time.sleep(0.05)
        if not self._recorder.gt_valid:
            return False
        gx, gy, _ = self._recorder.pose()
        return math.hypot(gx - float(pose["x"]), gy - float(pose["y"])) <= tol_xy

    @staticmethod
    def _json_safe(obj):
        """Convert non-finite floats to None.

        The risk features now return NaN when a value genuinely cannot be
        computed (rather than a misleading 0.0). json.dump would write a bare
        `NaN`, which Python reads back happily but which is not valid strict
        JSON — pandas.read_json, jq and most non-Python parsers reject it.
        """
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: TrialRunner._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TrialRunner._json_safe(v) for v in obj]
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if math.isfinite(f) else None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    def _finalize_json(self, rec: TrialRunnerNode, result: TrialResult, params: dict,
                       global_path: list):
        """Write the full per-trial JSON (time-series) and stamp result.json_path."""
        if not self.save_per_trial_json:
            return
        payload = {
            "trial_id": result.trial_id,
            "status": result.status,
            "terminated_by_runner": result.terminated_by_runner,
            "is_collided": result.collision,
            "travel_time_sec": result.travel_time_sec,
            "global_path_length": result.initial_global_path_length_m,
            "local_path_length": result.path_length_m,
            "min_global_dist_to_obstacle": result.min_global_obstacle_distance,
            "min_obstacle_distance": result.min_obstacle_distance,
            "min_map_dist_to_obstacle": result.min_map_obstacle_distance,
            "goal_distance_remaining": result.goal_distance_remaining,
            "final_xy_error": result.final_xy_error,
            "final_yaw_error": result.final_yaw_error,
            # Dual outcome, mirrored from the CSV so a trial JSON stands alone.
            "success_true": result.success_true,
            "believed_within_tolerance": result.believed_within_tolerance,
            "success_believed": result.success_believed,
            "belief_censored": result.belief_censored,
            "outcome_agreement": result.outcome_agreement,
            "believed_final_xy_error": result.believed_final_xy_error,
            "believed_final_yaw_error": result.believed_final_yaw_error,
            "belief_error_gap_m": result.belief_error_gap_m,
            "gt_ever_within_tolerance": result.gt_ever_within_tolerance,
            "t_first_within_tolerance": result.t_first_within_tolerance,
            "xy_goal_tolerance_used": result.xy_goal_tolerance_used,
            "yaw_goal_tolerance_used": result.yaw_goal_tolerance_used,
            "initial_pose": {"x": result.start_x, "y": result.start_y, "yaw": result.start_yaw},
            "goal_pose": {"x": result.goal_x, "y": result.goal_y, "yaw": result.goal_yaw},
            "nav2_config": result.params,
            "path_global_planner": global_path,
            "path_with_controller": rec.controller_path,
            "risk_state_history": [asdict(r) for r in rec.risk_state_history if result.t_nav_start<=r.timestamp<=result.t_nav_end],
            "num_risk_samples": len(rec.risk_state_history),
            "num_controller_samples": len(rec.controller_path),
            "map_yaml": self.map_yaml_path,
            "num__global_replans": result.global_planner_ticks,
            "global_replan_history": result.replan_history,
            "collision_links": result.collision_links,
            # Pose provenance / QA, mirrored from the CSV so a trial JSON is
            # self-contained.
            "pose_source": result.pose_source,
            "gt_msgs_seen": result.gt_msgs_seen,
            "unique_pose_fraction": result.unique_pose_fraction,
            "localization_error_m": result.localization_error_m,
            "teleport_ok": result.teleport_ok,
            "collision_msgs_seen": result.collision_msgs_seen,
            "collision_channel_silent": result.collision_channel_silent,
        }
        # Namespaced by run_id: re-running trial N in a later session no longer
        # clobbers the JSON that an earlier CSV row points at.
        path = os.path.join(self.output_dir, "trials",
                            f"trial_{self.run_id}_{result.trial_id:05d}.json")
        with open(path, "w") as f:
            json.dump(self._json_safe(payload), f, indent=2)
        result.json_path = path

        if self.generate_plots:
            self._plot_trial(result, global_path, rec.controller_path)

    def _plot_trial(self, result: TrialResult, global_path: list, controller_path: list):
        """Optional per-trial path plot over the map image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import yaml
            from pathlib import Path
            from PIL import Image

            if not self.map_yaml_path or not os.path.exists(self.map_yaml_path):
                return
            with open(self.map_yaml_path) as f:
                info = yaml.safe_load(f)
            res = info.get("resolution", 0.05)
            origin = info.get("origin", [0, 0, 0])
            img = np.array(Image.open(
                Path(self.map_yaml_path).parent / info["image"]).convert("L"))
            h = img.shape[0]

            def to_px(x, y):
                return ((x - origin[0]) / res, h - (y - origin[1]) / res)

            plt.figure(figsize=(10, 10))
            plt.imshow(img, cmap="gray")
            if global_path:
                gx, gy = zip(*[to_px(p["pose"][0], p["pose"][1]) for p in global_path])
                plt.plot(gx, gy, "r*-", ms=3, lw=1, label="Global path")
            if controller_path:
                cx, cy = zip(*[to_px(p["pose"][0], p["pose"][1]) for p in controller_path])
                plt.plot(cx, cy, "b.-", ms=2, lw=1, label="Executed path")
            plt.legend()
            plt.title(f"Trial {result.trial_id} — {result.status}")
            out = os.path.join(self.output_dir, "trials",
                               f"plot_{self.run_id}_{result.trial_id:05d}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.debug(f"  Plot failed: {e}")