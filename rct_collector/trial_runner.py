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
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from nav2_msgs.srv import GetCostmap
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
    global_path_length_m: float = 0.0               # planned global path
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
    
    num_replans: int = 0
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

    def to_dict(self) -> dict[str, Any]:
        """Flat, single-level dict for CSV output."""
        d = {
            "start_x": self.start_x, "start_y": self.start_y, "start_yaw": self.start_yaw,
            "goal_x": self.goal_x, "goal_y": self.goal_y, "goal_yaw": self.goal_yaw,
            "status": self.status,
            "collision": int(self.collision),
            "travel_time_sec": self.travel_time_sec,
            "path_length_m": self.path_length_m,
            "global_path_length_m": self.global_path_length_m,
            "goal_distance_remaining": self.goal_distance_remaining,
            "final_xy_error": self.final_xy_error,
            "final_yaw_error": self.final_yaw_error,
            "min_obstacle_distance": self.min_obstacle_distance,
            "min_map_obstacle_distance": self.min_map_obstacle_distance,
            "min_global_obstacle_distance": self.min_global_obstacle_distance,
            "num_risk_samples": self.num_risk_samples,
            "num_controller_samples": self.num_controller_samples,
            "num_replans": self.num_replans,
            "json_path": self.json_path,
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
                 collect_risk_features: bool):
        super().__init__("rct_trial_runner")
        self.collect_risk_features = collect_risk_features

        # Live state
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
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)

        # Static-map distance from base_link to the nearest obstacle at this pose
        # (analogue of the LiDAR distance, from the ground-truth map). The
        # base_link origin in the map frame is just the robot position.
        if self._map_obstacles is not None:
            d = _min_dist_base_link_to_obstacles(
                self._map_obstacles, (self.robot_x, self.robot_y))
            self.min_map_value = d
            if math.isfinite(d):
                self.min_map_overall = min(self.min_map_overall, d)

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
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        if not valid.any():
            self.min_scan_value = None
            return

        self._ensure_scan_geometry(msg)
        r = ranges[valid]
        # Lift valid hits into base_link, then take the distance from the
        # base_link origin (0, 0) to the nearest hit — NOT to the footprint.
        px = self._laser_tx + r * self._scan_ux[valid]
        py = self._laser_ty + r * self._scan_uy[valid]
        d = float(np.sqrt(px * px + py * py).min())

        self.min_scan_value = d
        self.min_scan_overall = min(self.min_scan_overall, d)

    def _risk_state_callback(self, msg: Float64MultiArray):
        try:
            rec = RiskStateRecord.from_array(time.time(), list(msg.data))
        except ValueError as e:
            self.get_logger().warn(f"Invalid risk state data: {e}")
            return
        self.latest_risk_state = rec
        self.risk_state_history.append(rec)
        
    def _collision_callback(self, msg: Bool):
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
        entry = {
            "timestamp": time.time(),
            "pose": [self.robot_x, self.robot_y, self.robot_yaw],
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
        xy_goal_tolerance: float = 0.25,
        yaw_goal_tolerance: float = 0.25,
        stop_when_within_tolerance: bool = True,
        # Ignore tolerance hits before this many seconds, so a trial whose start
        # pose already sits inside the goal tolerance is not declared an instant
        # success before the controller has done anything.
        min_trial_time_sec: float = 2.0,
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
            scan_topic, odom_topic, risk_topic, collect_risk_features)
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

    # ── public API (called by the orchestrator) ──
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
        )

        # 1. Teleport in Gazebo, then localize.
        self._teleport_robot(start_pose)
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
            self._finalize_json(rec, result, params, global_path=[])
            logger.error("  Initial global planning failed — no path.")
            return result

        global_path_data, result.global_path_length_m, result.min_global_obstacle_distance = \
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
        
        nav_result = self._navigator.getResult()
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

        # 6. Goal distance remaining, final xy error, and final yaw error based on
        # the last pose of the controller path.
        if rec.controller_path:
            last_pose = rec.controller_path[-1]["pose"]
            last_x, last_y, last_yaw = last_pose[0], last_pose[1], last_pose[2]
        else:
            last_x, last_y, last_yaw = rec.robot_x, rec.robot_y, rec.robot_yaw

        dx = goal_pose["x"] - last_x
        dy = goal_pose["y"] - last_y
        result.final_xy_error = float(math.hypot(dx, dy))
        result.goal_distance_remaining = result.final_xy_error

        dyaw = goal_pose["yaw"] - last_yaw
        result.final_yaw_error = float(math.atan2(math.sin(dyaw), math.cos(dyaw)))

        result.min_obstacle_distance = rec.min_scan_overall
        result.min_map_obstacle_distance = rec.min_map_overall

        # 7. Persist the full time-series JSON.
        self._finalize_json(rec, result, params, global_path=global_path_data)

        logger.info(
            f"  Trial {trial_id} complete: status={result.status}, "
            f"time={result.travel_time_sec:.1f}s, path={result.path_length_m:.2f}m, "
            f"risk_samples={result.num_risk_samples}"
        )
        return result

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
        prev_pose = [rec.robot_x, rec.robot_y]
        local_len = 0.0
        self._pending_replans = []

        while not self._navigator.isTaskComplete():
            time.sleep(0.01)
            now = time.time()
            elapsed = now - t_start

            # Drain /plan messages produced by the BT's RateController.
            # Buffer only — analysis is deferred until after navigation ends,
            # so ~270 footprintCostAtPose calls per replan never block this loop.
            while rec.replan_events:
                self._pending_replans.append(rec.replan_events.popleft())

            footprint_cost = 0.0
            if checker is not None:
                footprint_cost = checker.footprintCostAtPose(
                    rec.robot_x, rec.robot_y, rec.robot_yaw)

            if now - last_record >= self.record_period:
                rec.record_sample(footprint_cost)
                last_record = now
                local_len += math.hypot(rec.robot_x - prev_pose[0], rec.robot_y - prev_pose[1])
                prev_pose = [rec.robot_x, rec.robot_y]

            # Early termination: the robot is inside the goal tolerance.
            # Requires xy AND yaw to be satisfied at the SAME pose, so the
            # controller cannot loiter doing repeated heading corrections near
            # the goal (the behaviour that was tripping the timeout). Measured
            # from rec.robot_* — the same /amcl_pose source used for the final
            # error below — so a SUCCESS declared here always has final error
            # within tolerance.
            if self.stop_when_within_tolerance and elapsed >= self.min_trial_time_sec:
                _dx = result.goal_x - rec.robot_x
                _dy = result.goal_y - rec.robot_y
                _xy_err = math.hypot(_dx, _dy)
                _dyaw = result.goal_yaw - rec.robot_yaw
                _yaw_err = abs(math.atan2(math.sin(_dyaw), math.cos(_dyaw)))
                if (_xy_err <= self.xy_goal_tolerance
                        and _yaw_err <= self.yaw_goal_tolerance):
                    rec.record_sample(footprint_cost)
                    self._navigator.cancelTask()
                    result.status = "SUCCESS"
                    result.terminated_by_runner = True
                    logger.info(
                        f"  Within goal tolerance (xy={_xy_err:.3f}m <= "
                        f"{self.xy_goal_tolerance:.3f}, yaw={_yaw_err:.3f}rad <= "
                        f"{self.yaw_goal_tolerance:.3f}) after {elapsed:.1f}s — "
                        "ending trial.")
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
                result.terminated_by_runner = True
                logger.warning(f"  Trial timed out after {elapsed:.1f}s")
                break

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
        result.num_replans = len(replan_analyses)
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

    def _teleport_robot(self, pose: dict):
        """Teleport the robot in Gazebo to the start pose (best effort)."""
        _, _, qz, qw = yaw_to_quaternion(float(pose["yaw"]))
        state = (
            f'{{"state": {{"name": "{self.gazebo_robot_model}", '
            f'"pose": {{"position": {{"x": {pose["x"]}, "y": {pose["y"]}, "z": 0.0}}, '
            f'"orientation": {{"z": {qz}, "w": {qw}}}}}}}}}'
        )
        cmd = ["ros2", "service", "call", "/gazebo/set_entity_state",
               "gazebo_msgs/srv/SetEntityState", state]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                logger.info(f"  Teleported to ({pose['x']:.2f}, {pose['y']:.2f})")
            else:
                logger.debug("  set_entity_state failed; trying set_model_state...")
                cmd[3] = "gazebo_msgs/srv/SetModelState"
                cmd[2] = "/set_model_state"
                subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception as e:
            logger.warning(f"  Teleport failed: {e}. Robot may not be at start pose.")

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
            "global_path_length": result.global_path_length_m,
            "local_path_length": result.path_length_m,
            "min_global_dist_to_obstacle": result.min_global_obstacle_distance,
            "min_obstacle_distance": result.min_obstacle_distance,
            "min_map_dist_to_obstacle": result.min_map_obstacle_distance,
            "goal_distance_remaining": result.goal_distance_remaining,
            "final_xy_error": result.final_xy_error,
            "final_yaw_error": result.final_yaw_error,
            "initial_pose": {"x": result.start_x, "y": result.start_y, "yaw": result.start_yaw},
            "goal_pose": {"x": result.goal_x, "y": result.goal_y, "yaw": result.goal_yaw},
            "nav2_config": result.params,
            "path_global_planner": global_path,
            "path_with_controller": rec.controller_path,
            "risk_state_history": [asdict(r) for r in rec.risk_state_history if result.t_nav_start<=r.timestamp<=result.t_nav_end],
            "num_risk_samples": len(rec.risk_state_history),
            "num_controller_samples": len(rec.controller_path),
            "map_yaml": self.map_yaml_path,
            "num__global_replans": result.num_replans,
            "global_replan_history": result.replan_history,
            "collision_links": result.collision_links,
        }
        path = os.path.join(self.output_dir, "trials", f"trial_{result.trial_id:05d}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
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
            out = os.path.join(self.output_dir, "trials", f"plot_{result.trial_id:05d}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.debug(f"  Plot failed: {e}")