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
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav2_msgs.srv import GetCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

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
    min_obstacle_distance: float = float("inf")     # closest LiDAR approach
    min_global_obstacle_distance: float = float("inf")  # along the planned path

    # Bookkeeping for the rich JSON
    num_risk_samples: int = 0
    num_controller_samples: int = 0
    json_path: str = ""

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
            "min_obstacle_distance": self.min_obstacle_distance,
            "min_global_obstacle_distance": self.min_global_obstacle_distance,
            "num_risk_samples": self.num_risk_samples,
            "num_controller_samples": self.num_controller_samples,
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

        # Buffers (cleared per trial)
        self.latest_risk_state: Optional[RiskStateRecord] = None
        self.risk_state_history: list[RiskStateRecord] = []
        self.controller_path: list[dict] = []
        self.min_scan_overall = float("inf")

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
        if collect_risk_features:
            self.create_subscription(
                Float64MultiArray, risk_topic, self._risk_state_callback, 10)

    # ── callbacks ──
    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)

    def _odom_callback(self, msg: Odometry):
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular

    def _scan_callback(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        self.min_scan_value = min(valid) if valid else None
        if self.min_scan_value is not None:
            self.min_scan_overall = min(self.min_scan_overall, self.min_scan_value)

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


# ── Trial runner ──────────────────────────────────────────────────────────────

class TrialRunner:
    """Owns the ROS context + BasicNavigator and executes trials in a loop.

    rclpy and the BasicNavigator are created once; run_trial() reuses them.
    """

    def __init__(
        self,
        timeout_sec: float = 180.0,
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
        generate_plots: bool = False,
        risk_topic: str = "/risk_state",
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

        if not rclpy.ok():
            rclpy.init()

        # Persistent recorder node + navigator + costmap client (created once).
        self._recorder = TrialRunnerNode(
            scan_topic, odom_topic, risk_topic, collect_risk_features)
        self._navigator = BasicNavigator()
        logger.info("Waiting for Nav2 to become active...")
        self._navigator.waitUntilNav2Active()
        logger.info("Nav2 active ✓")

        self._costmap_cli = self._navigator.create_client(
            GetCostmap, "/local_costmap/get_costmap")

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

        # 2. Resolve this trial's footprint and build the collision checker.
        footprint = self._resolve_footprint(params)
        checker, costmap = self._build_footprint_checker(footprint)
        collision_distance = self.collision_threshold
        if checker is not None:
            collision_distance = checker.geometry.inscribed_radius + self.collision_margin

        # 3. Plan + analyse the global path.
        goal_stamped = self._make_pose_stamped(goal_pose)
        path = self._navigator.getPath(init_pose, goal_stamped)
        if path is None or not path.poses:
            result.status = "PLANNING_FAILED"
            self._finalize_json(rec, result, params, global_path=[])
            logger.error("  Global planning failed — no path.")
            return result

        global_path_data, result.global_path_length_m, result.min_global_obstacle_distance = \
            self._analyze_global_path(path, checker, costmap)

        # 4. Follow the smoothed path while recording.
        # smoothed = self._navigator.smoothPath(path) or path
        # self._navigator.followPath(smoothed)
        self._record_navigation(rec, checker, collision_distance, result)

        # 5. Classify outcome (collision overrides the Nav2 status).
        nav_result = self._navigator.getResult()
        if result.collision:
            result.status = "COLLISION"
        elif nav_result == TaskResult.SUCCEEDED:
            result.status = "SUCCESS"
        elif nav_result == TaskResult.CANCELED:
            result.status = result.status if result.status == "TIMEOUT" else "CANCELED"
        elif nav_result == TaskResult.FAILED:
            result.status = "FAILED"
        else:
            result.status = result.status or "UNKNOWN"

        # 6. Goal distance remaining + executed path length.
        dx = goal_pose["x"] - rec.robot_x
        dy = goal_pose["y"] - rec.robot_y
        result.goal_distance_remaining = math.hypot(dx, dy)
        result.min_obstacle_distance = rec.min_scan_overall

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
            self._recorder.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    # ── navigation recording loop ──
    def _record_navigation(self, rec: TrialRunnerNode, checker, collision_distance: float,
                           result: TrialResult):
        t_start = time.time()
        last_record = t_start
        prev_pose = [rec.robot_x, rec.robot_y]
        local_len = 0.0

        while not self._navigator.isTaskComplete():
            rclpy.spin_once(rec, timeout_sec=0.05)
            now = time.time()
            elapsed = now - t_start

            footprint_cost = 0.0
            if checker is not None:
                footprint_cost = checker.footprintCostAtPose(
                    rec.robot_x, rec.robot_y, rec.robot_yaw)

            # Record at the target rate (not every loop iteration).
            if now - last_record >= self.record_period:
                rec.record_sample(footprint_cost)
                last_record = now
                local_len += math.hypot(rec.robot_x - prev_pose[0], rec.robot_y - prev_pose[1])
                prev_pose = [rec.robot_x, rec.robot_y]

            # Collision: footprint in lethal cell AND LiDAR confirms proximity.
            # (Fallback to LiDAR-only when no checker/costmap is available.)
            # hit = False
            # if checker is not None:
            #     hit = (footprint_cost >= INSCRIBED_INFLATED_OBSTACLE
            #            and rec.min_scan_value is not None
            #            and rec.min_scan_value < collision_distance)
            # elif rec.min_scan_value is not None:
            #     hit = rec.min_scan_value < self.collision_threshold
            if self._recorder.is_collided:
                rec.record_sample(footprint_cost)
                self._navigator.cancelTask()
                result.collision = True
                logger.warning(
                    f"  COLLISION: footprint_cost={footprint_cost:.0f}, "
                    f"min_scan={rec.min_scan_value:.3f}m < {collision_distance:.3f}m")
                break

            # Timeout guard.
            if elapsed > self.timeout_sec:
                self._navigator.cancelTask()
                result.status = "TIMEOUT"
                logger.warning(f"  Trial timed out after {elapsed:.1f}s")
                break

        result.travel_time_sec = time.time() - t_start
        result.path_length_m = local_len
        result.num_controller_samples = len(rec.controller_path)
        result.num_risk_samples = len(rec.risk_state_history)

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
        """Give AMCL time to converge after the initial pose is set."""
        logger.info("  Waiting for localization to converge...")
        deadline = time.time() + self.localization_settle_sec
        while time.time() < deadline:
            rclpy.spin_once(self._recorder, timeout_sec=0.05)
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

    def _build_footprint_checker(self, footprint: list):
        """Fetch the current global costmap and build a checker for `footprint`.
        Returns (checker, costmap_response) — both None if unavailable."""
        if not _HAVE_FOOTPRINT_CHECKER:
            return None, None
        costmap = self._get_costmap()
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

    def _get_costmap(self):
        if not self._costmap_cli.wait_for_service(timeout_sec=5.0):
            return None
        future = self._costmap_cli.call_async(GetCostmap.Request())
        rclpy.spin_until_future_complete(self._navigator, future, timeout_sec=10.0)
        return future.result()

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
            "is_collided": result.collision,
            "travel_time_sec": result.travel_time_sec,
            "global_path_length": result.global_path_length_m,
            "local_path_length": result.path_length_m,
            "min_global_dist_to_obstacle": result.min_global_obstacle_distance,
            "min_obstacle_distance": result.min_obstacle_distance,
            "goal_distance_remaining": result.goal_distance_remaining,
            "initial_pose": {"x": result.start_x, "y": result.start_y, "yaw": result.start_yaw},
            "goal_pose": {"x": result.goal_x, "y": result.goal_y, "yaw": result.goal_yaw},
            "nav2_config": result.params,
            "path_global_planner": global_path,
            "path_with_controller": rec.controller_path,
            "risk_state_history": [asdict(r) for r in rec.risk_state_history],
            "num_risk_samples": len(rec.risk_state_history),
            "num_controller_samples": len(rec.controller_path),
            "map_yaml": self.map_yaml_path,
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


