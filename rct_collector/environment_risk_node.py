#!/usr/bin/env python3
"""
Optimized ROS2 Risk State Calculator Node

Computes the 8-dimensional risk state vector R_t for online causal navigation tuning.
Uses local costmap for efficiency and relevance to immediate collision risk.

Design principles:
1. Decoupled update rates (sensors at HW rate, computation at 10 Hz)
2. Pre-allocated arrays (no GC during hot path)
3. Lazy gradient caching (recompute only on costmap change)
4. Vectorized numpy operations throughout
5. Lock-free sensor updates where possible
6. Graceful degradation under time pressure

Subscribes to:
    - /scan (sensor_msgs/LaserScan): LiDAR data
    - /odom or /mobile_base_controller/odom (nav_msgs/Odometry): Robot velocity
    - /amcl_pose (geometry_msgs/PoseWithCovarianceStamped): Robot pose
    - /global_costmap/costmap (nav2_msgs/Costmap): Global costmap # it was local costmap 
    - /plan (nav_msgs/Path): Current path for curvature computation

Publishes:
    - /risk_state (std_msgs/Float64MultiArray): 8-element risk vector
    - /risk_state_diagnostics (diagnostic_msgs/DiagnosticStatus): Timing info


"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import Costmap
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

import numpy as np
from typing import Optional, Tuple, List

# r_grad rebuilds its own inflation from the costmap's lethal cells, which needs
# a Euclidean distance transform. Import once at module scope; failing loudly at
# startup is better than raising inside every compute cycle.
try:
    from scipy import ndimage as _ndimage
except ImportError as _e:  # pragma: no cover
    _ndimage = None
    logging.getLogger(__name__).error(
        "scipy is unavailable (%s); r_grad will be NaN for the whole run. "
        "Install scipy or set collect_risk_features accordingly.", _e)
from dataclasses import dataclass, field
from enum import IntEnum
import time


class RiskIndex(IntEnum):
    """Indices for risk state vector components."""
    R_MIN = 0      # Minimum obstacle distance
    R_WIDTH = 1    # Corridor width
    R_TTC = 2      # Time to collision
    R_DENS = 3     # Obstacle density
    R_CLEAR = 4    # Heading clearance
    R_CURVE = 5    # Path curvature
    R_GRAD = 6     # Costmap gradient
    R_VIS = 7      # Visibility risk


@dataclass
class RiskStateConfig:
    """Configuration parameters for risk state computation."""
    # Update rates
    compute_rate_hz: float = 10.0
    
    # Spatial parameters (meters)
    density_window_radius: float = 2.0
    look_ahead_distance: float = 3.0
    blocked_distance: float = 3.0
    
    # Angular parameters
    num_polar_sectors: int = 36
    forward_sector_half_angle: float = np.pi / 3  # 60 degrees total
    lateral_tolerance: float = np.pi / 6  # 30 degrees for width computation
    
    # Velocity threshold
    velocity_epsilon: float = 0.01
    
    # Timing budget (ms)
    time_budget_ms: float = 10.0
    
    # Default values when data unavailable
    default_r_min: float = 10.0
    default_r_ttc: float = 100.0
    default_r_width: float = 10.0

    # LiDAR self-return rejection — must match trial_runner's
    # scan_self_filter_radius_m or the two pipelines disagree.
    scan_self_filter_radius_m: float = 0.30
    scan_angle_mask_deg: tuple = ()     # flat [lo1,hi1,lo2,hi2,...] in LASER frame
    risk_inflation_radius_m: float = 0.3   # FIXED — must not track the treatment
    risk_cost_scaling_factor: float = 10.0
    v_ref: float = 1.0                  # Reference velocity (m/s) for deconfounded TTC

@dataclass 
class SensorState:
    """Container for latest sensor readings (lock-free updates)."""
    # LiDAR
    scan_ranges: Optional[np.ndarray] = None
    scan_angles: Optional[np.ndarray] = None  # Pre-computed from scan geometry
    scan_range_min: float = 0.1
    scan_range_max: float = 10.0
    scan_timestamp: float = 0.0
    
    # Odometry
    velocity_x: float = 0.0
    velocity_timestamp: float = 0.0
    
    # Pose (from AMCL)
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_yaw: float = 0.0
    pose_timestamp: float = 0.0
    
    # Costmap
    costmap: Optional[np.ndarray] = None
    costmap_resolution: float = 0.05
    costmap_origin_x: float = 0.0
    costmap_origin_y: float = 0.0
    costmap_width: int = 0
    costmap_height: int = 0
    costmap_timestamp: float = 0.0
    costmap_frame: str = ""
    
    # Path
    path_points: Optional[np.ndarray] = None
    path_timestamp: float = 0.0


@dataclass
class ComputeCache:
    """Cached intermediate results to avoid redundant computation."""
    # Costmap gradient (recompute only when costmap changes)
    gradient_magnitude: Optional[np.ndarray] = None
    gradient_costmap_timestamp: float = 0.0
    
    # Path curvatures (recompute only when path changes)
    path_curvatures: Optional[np.ndarray] = None
    path_distances: Optional[np.ndarray] = None
    path_timestamp: float = 0.0
    
    # Scan geometry (recompute only when scan config changes)
    sector_indices: Optional[np.ndarray] = None
    forward_mask: Optional[np.ndarray] = None
    scan_num_beams: int = 0
    
    # Pre-allocated arrays for computation
    histogram: Optional[np.ndarray] = None
    valid_ranges: Optional[np.ndarray] = None
    angle_keep_mask: Optional[np.ndarray] = None


class OptimizedRiskStateNode(Node):
    """
    Optimized ROS2 node for real-time risk state computation.
    
    Key optimizations:
    1. Single timer for computation (decoupled from sensor rates)
    2. Pre-allocated arrays in cache
    3. Lazy gradient/curvature computation
    4. Vectorized numpy operations
    5. Graceful degradation under time pressure
    """
    
    # Feature names for documentation/debugging
    FEATURE_NAMES = ['r_min', 'r_width', 'r_ttc', 'r_dens', 
                     'r_clear', 'r_curve', 'r_grad', 'r_vis']
    
    def __init__(self):
        super().__init__('optimized_risk_state_node')
        
        # Declare and get parameters
        self._declare_parameters()
        self.config = self._load_config()
        
        # Pre-compute constants
        self.sector_width = 2 * np.pi / self.config.num_polar_sectors
        
        # State containers
        self.sensor_state = SensorState()
        self.cache = ComputeCache()
        
        # Pre-allocate histogram array
        self.cache.histogram = np.zeros(self.config.num_polar_sectors, dtype=np.float32)
        
        # Timing statistics
        self._compute_times: List[float] = []
        self._last_stats_time = time.time()
        self._cycles_computed = 0
        self._cycles_degraded = 0
        
        # Previous risk state (for graceful degradation)
        self._previous_risk_state = np.zeros(8, dtype=np.float32)
        
        # Setup QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Callback groups
        self._sensor_cb_group = ReentrantCallbackGroup()
        self._compute_cb_group = MutuallyExclusiveCallbackGroup()
        # TF: the costmap origin is expressed in the costmap's own frame
        # (odom for the local costmap), while robot_x/robot_y come from
        # /amcl_pose in the map frame. Without this, r_dens and r_grad index
        # the grid with map-frame coordinates and silently return 0.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_failures = 0
        self._tf_attempts = 0





        # Get topic names from parameters
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        path_topic = self.get_parameter('path_topic').value
        
        # Subscribers (all in reentrant group for parallel processing)
        self._scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_callback, sensor_qos,
            callback_group=self._sensor_cb_group
        )
        
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, sensor_qos,
            callback_group=self._sensor_cb_group
        )
        
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self._pose_callback, 10,
            callback_group=self._sensor_cb_group
        )
        
        self._costmap_sub = self.create_subscription(
            Costmap, costmap_topic, self._costmap_callback, reliable_qos,
            callback_group=self._sensor_cb_group
        )
        
        self._path_sub = self.create_subscription(
            Path, path_topic, self._path_callback, 10,
            callback_group=self._sensor_cb_group
        )
        
        # Publishers
        self._risk_pub = self.create_publisher(Float64MultiArray, '/risk_state', 10)
        self._diag_pub = self.create_publisher(DiagnosticStatus, '/risk_state_diagnostics', 10)
        
        # Main computation timer
        self._compute_timer = self.create_timer(
            1.0 / self.config.compute_rate_hz,
            self._compute_and_publish,
            callback_group=self._compute_cb_group
        )
        
        self.get_logger().info(
            f'Optimized risk state node started at {self.config.compute_rate_hz} Hz\n'
            f'  Scan topic: {scan_topic}\n'
            f'  Odom topic: {odom_topic}\n'
            f'  Pose topic: {pose_topic}\n'
            f'  Costmap topic: {costmap_topic}\n'
            f'  Path topic: {path_topic}'
        )
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        # Topics
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('odom_topic', '/mobile_base_controller/odom')
        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('costmap_topic', '/local_costmap/costmap_raw')
        self.declare_parameter('path_topic', '/plan')
        
        # Computation parameters
        self.declare_parameter('compute_rate_hz', 10.0)
        self.declare_parameter('time_budget_ms', 10.0)
        
        # Risk state parameters
        self.declare_parameter('density_window_radius', 2.0)
        self.declare_parameter('look_ahead_distance', 3.0)
        self.declare_parameter('blocked_distance', 3.0)
        self.declare_parameter('num_polar_sectors', 36)
        self.declare_parameter('forward_sector_half_angle', np.pi / 3)
        self.declare_parameter('scan_self_filter_radius_m', 0.30)
        self.declare_parameter('scan_angle_mask_deg', [])
        # Constants of the r_grad DEFINITION. These deliberately do NOT track
        # the inflation_radius / cost_scaling_factor treatment: if they did,
        # r_grad would be a function of C_t and the risk x config interaction
        # would partly be the treatment interacting with itself. Pin them in
        # the experiment config and report them in the paper.
        self.declare_parameter('risk_inflation_radius_m', 0.30)
        self.declare_parameter('risk_cost_scaling_factor', 10.0)
        self.declare_parameter('v_ref', 1.0)
    
    
    def _load_config(self) -> RiskStateConfig:
        """Load configuration from parameters."""
        return RiskStateConfig(
            compute_rate_hz=self.get_parameter('compute_rate_hz').value,
            time_budget_ms=self.get_parameter('time_budget_ms').value,
            density_window_radius=self.get_parameter('density_window_radius').value,
            look_ahead_distance=self.get_parameter('look_ahead_distance').value,
            blocked_distance=self.get_parameter('blocked_distance').value,
            num_polar_sectors=self.get_parameter('num_polar_sectors').value,
            forward_sector_half_angle=self.get_parameter('forward_sector_half_angle').value,
            scan_self_filter_radius_m=self.get_parameter('scan_self_filter_radius_m').value,
            scan_angle_mask_deg=tuple(self.get_parameter('scan_angle_mask_deg').value or ()),
            risk_inflation_radius_m=self.get_parameter('risk_inflation_radius_m').value,
            risk_cost_scaling_factor=self.get_parameter('risk_cost_scaling_factor').value,
            v_ref=self.get_parameter('v_ref').value,
        )
    
    # =========================================================================
    # SENSOR CALLBACKS (Lock-free, minimal processing)
    # =========================================================================
    
    def _scan_callback(self, msg: LaserScan) -> None:
        """Store latest scan with pre-computed angles."""
        # Convert to numpy array (single allocation, reused)
        ranges = np.array(msg.ranges, dtype=np.float32)
        
        # Check if scan geometry changed
        if self.cache.scan_num_beams != len(ranges):
            self._update_scan_geometry(msg, len(ranges))
        
        # Store latest data (lock-free assignment)
        self.sensor_state.scan_ranges = ranges
        self.sensor_state.scan_range_min = msg.range_min
        self.sensor_state.scan_range_max = msg.range_max
        self.sensor_state.scan_timestamp = time.time()
    
    def _update_scan_geometry(self, msg: LaserScan, num_beams: int) -> None:
        """Update cached scan geometry when configuration changes."""
        # Pre-compute angles (only when scan config changes)
        angles = np.linspace(msg.angle_min, msg.angle_max, num_beams, dtype=np.float32)
        self.sensor_state.scan_angles = angles
        
        # Pre-compute sector indices for heading clearance
        norm_angles = np.mod(angles, 2 * np.pi)
        self.cache.sector_indices = (norm_angles / self.sector_width).astype(np.int32) % self.config.num_polar_sectors
        
        # Pre-compute forward mask for TTC and visibility
        self.cache.forward_mask = np.abs(angles) < self.config.forward_sector_half_angle
        
        keep = np.ones(num_beams, dtype=bool)
        m = self.config.scan_angle_mask_deg
        for lo, hi in zip(m[0::2], m[1::2]):
            keep &= ~((angles >= np.deg2rad(lo)) & (angles <= np.deg2rad(hi)))
        self.cache.angle_keep_mask = keep
        self.get_logger().info(
            f'Scan geometry: {num_beams} beams, {int((~keep).sum())} masked by angle, '
            f'self-filter radius {self.config.scan_self_filter_radius_m} m')
        # Pre-allocate valid_ranges array
        self.cache.valid_ranges = np.empty(num_beams, dtype=np.float32)
        
        self.cache.scan_num_beams = num_beams
        
        self.get_logger().info(f'Scan geometry updated: {num_beams} beams')
    
    def _odom_callback(self, msg: Odometry) -> None:
        """Extract forward velocity from odometry."""
        self.sensor_state.velocity_x = msg.twist.twist.linear.x
        self.sensor_state.velocity_timestamp = time.time()
    
    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Update robot pose from AMCL."""
        self.sensor_state.robot_x = msg.pose.pose.position.x
        self.sensor_state.robot_y = msg.pose.pose.position.y
        
        # Quaternion to yaw (optimized)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.sensor_state.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.sensor_state.pose_timestamp = time.time()
    
    def _costmap_callback(self, msg: Costmap) -> None:
        """Update costmap and invalidate gradient cache."""
        # Reshape costmap data
        costmap = np.array(msg.data, dtype=np.float32).reshape(
            msg.metadata.size_y, msg.metadata.size_x
        )
        
        self.sensor_state.costmap = costmap
        self.sensor_state.costmap_resolution = msg.metadata.resolution
        self.sensor_state.costmap_origin_x = msg.metadata.origin.position.x
        self.sensor_state.costmap_origin_y = msg.metadata.origin.position.y
        self.sensor_state.costmap_width = msg.metadata.size_x
        self.sensor_state.costmap_height = msg.metadata.size_y
        self.sensor_state.costmap_timestamp = time.time()
        self.sensor_state.costmap_frame = msg.header.frame_id
        
        # Invalidate gradient cache (will be recomputed on next use)
        # Don't compute here - let the compute cycle handle it lazily
    
    def _path_callback(self, msg: Path) -> None:
        """Update path and invalidate curvature cache."""
        if len(msg.poses) < 3:
            self.sensor_state.path_points = None
            return
        
        # Extract path points as numpy array
        points = np.array([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in msg.poses
        ], dtype=np.float32)
        
        self.sensor_state.path_points = points
        self.sensor_state.path_timestamp = time.time()
        
        # Invalidate curvature cache
        # Don't compute here - let the compute cycle handle it lazily
    
    # =========================================================================
    # MAIN COMPUTATION (runs at fixed rate)
    # =========================================================================
    
    def _compute_and_publish(self) -> None:
        """Main computation cycle with timing budget enforcement."""
        t_start = time.perf_counter()
        
        # Check if we have minimum required data
        if self.sensor_state.scan_ranges is None:
            return
        
        # Initialize result with previous values (for graceful degradation)
        result = self._previous_risk_state.copy()
        components_computed = []
        
        # Compute in priority order, checking budget after each
        budget_ms = self.config.time_budget_ms
        
        # --- CRITICAL COMPONENTS (always compute) ---
        
        # 1. R_min - Most critical for immediate collision avoidance
        result[RiskIndex.R_MIN] = self._compute_r_min()
        components_computed.append('r_min')
        
        # 2. R_ttc - Time-critical for braking decisions
        result[RiskIndex.R_TTC] = self._compute_r_ttc()
        components_computed.append('r_ttc')
        
        # 3. R_vis - Important for uncertainty estimation
        result[RiskIndex.R_VIS] = self._compute_r_vis()
        components_computed.append('r_vis')
        
        # Check timing after critical components
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        # --- IMPORTANT COMPONENTS (compute if time permits) ---
        
        if elapsed_ms < budget_ms * 0.5:
            # 4. R_dens - General situational awareness
            result[RiskIndex.R_DENS] = self._compute_r_dens()
            components_computed.append('r_dens')
            
            # 5. R_width - Corridor navigation
            result[RiskIndex.R_WIDTH] = self._compute_r_width()
            components_computed.append('r_width')
            
            # 6. R_clear - Heading clearance
            result[RiskIndex.R_CLEAR] = self._compute_r_clear()
            components_computed.append('r_clear')
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        # --- CACHED COMPONENTS (compute if cache invalid and time permits) ---
        
        if elapsed_ms < budget_ms * 0.8:
            # 7. R_curve - Path curvature (cached)
            result[RiskIndex.R_CURVE] = self._compute_r_curve_cached()
            components_computed.append('r_curve')
            
            # 8. R_grad - Costmap gradient (cached)
            result[RiskIndex.R_GRAD] = self._compute_r_grad_cached()
            components_computed.append('r_grad')
        
        # Record final timing
        total_ms = (time.perf_counter() - t_start) * 1000
        self._compute_times.append(total_ms)
        self._cycles_computed += 1
        
        if len(components_computed) < 8:
            self._cycles_degraded += 1
        
        # Update previous state
        self._previous_risk_state = result
        
        # Publish risk state
        self._publish_risk_state(result)
        
        # Publish diagnostics periodically
        if time.time() - self._last_stats_time > 5.0:
            self._publish_diagnostics(components_computed, total_ms)
            self._last_stats_time = time.time()
    
    # =========================================================================
    # COMPONENT COMPUTATIONS (Vectorized and optimized)
    # =========================================================================
    def _robot_in_costmap_frame(self):
        """Robot (x, y) expressed in the costmap's own frame, or None."""
        frame = self.sensor_state.costmap_frame
        if not frame:
            return None
        if frame == 'map':
            return self.sensor_state.robot_x, self.sensor_state.robot_y
        self._tf_attempts += 1
        try:
            tf = self._tf_buffer.lookup_transform(frame, 'base_link', Time())
        except Exception:
            self._tf_failures += 1
            return None
        return tf.transform.translation.x, tf.transform.translation.y
    
    def _valid_scan_mask(self, extra: Optional[np.ndarray] = None) -> np.ndarray:
        """Beams that are real world returns.

        Rejects (a) below-range_min, (b) above-range_max, (c) NaN/inf,
        (d) returns from the robot's own structure inside the self-filter
        radius, (e) explicitly masked angular sectors.
        """
        ranges = self.sensor_state.scan_ranges
        lo = max(self.sensor_state.scan_range_min,
                 self.config.scan_self_filter_radius_m)
        mask = np.isfinite(ranges) & (ranges > lo) & (ranges < self.sensor_state.scan_range_max)
        if self.cache.angle_keep_mask is not None:
            mask &= self.cache.angle_keep_mask
        if extra is not None:
            mask &= extra
        return mask

    def self_hit_fraction(self) -> float:
        r = self.sensor_state.scan_ranges
        if r is None or r.size == 0:
            return 0.0
        finite = np.isfinite(r)
        if not finite.any():
            return 0.0
        return float((r[finite] <= self.config.scan_self_filter_radius_m).mean())

    def _compute_r_min(self) -> float:
        valid = self._valid_scan_mask()
        if not np.any(valid):
            return self.config.default_r_min
        return float(np.min(self.sensor_state.scan_ranges[valid]))
    
    def _compute_r_ttc(self) -> float:
        """R_ttc: Reference time to collision (d_front / v_ref).
        
        Uses fixed reference speed v_ref (1.0 m/s) rather than live velocity vx
        to keep R_ttc deconfounded from treatment C^v.
        """
        if self.cache.forward_mask is None:
            return self.config.default_r_ttc

        # Forward beams that are genuine world returns (self-returns rejected).
        valid = self._valid_scan_mask(self.cache.forward_mask)

        if not np.any(valid):
            return self.config.default_r_ttc

        d_front = float(np.min(self.sensor_state.scan_ranges[valid]))
        v_ref = self.config.v_ref if self.config.v_ref > 0 else 1.0

        return min(d_front / v_ref, self.config.default_r_ttc)
    
    def _compute_r_vis(self) -> float:
        fwd = self.cache.forward_mask
        ranges = self.sensor_state.scan_ranges
        considered = fwd & np.isfinite(ranges) & (ranges > self.config.scan_self_filter_radius_m)
        if self.cache.angle_keep_mask is not None:
            considered &= self.cache.angle_keep_mask
        n = int(considered.sum())
        if n == 0:
            return 0.0
        # occluded = a real return short of max range blocks the view
        occluded = considered & (ranges < 0.95 * self.sensor_state.scan_range_max)
        return float(occluded.sum()) / n
    
    def _compute_r_dens(self) -> float:
        """R_dens: Obstacle density in local window.

        Returns NaN (not 0.0) when the value cannot be computed, so "no data"
        is distinguishable from "genuinely no lethal cells nearby".
        """
        costmap = self.sensor_state.costmap
        if costmap is None:
            return float('nan')

        # Robot position in the COSTMAP's frame — the local costmap origin is
        # in odom, while robot_x/robot_y come from /amcl_pose in map.
        rp = self._robot_in_costmap_frame()
        if rp is None:
            return float('nan')
        rx, ry = rp

        resolution = self.sensor_state.costmap_resolution
        origin_x = self.sensor_state.costmap_origin_x
        origin_y = self.sensor_state.costmap_origin_y

        # Convert robot position to costmap coordinates
        mx = int((rx - origin_x) / resolution)
        my = int((ry - origin_y) / resolution)

        # Window size in cells
        window_cells = int(self.config.density_window_radius / resolution)

        # Extract local window with bounds checking
        h, w = costmap.shape
        x_min = max(0, mx - window_cells)
        x_max = min(w, mx + window_cells + 1)
        y_min = max(0, my - window_cells)
        y_max = min(h, my + window_cells + 1)

        if x_min >= x_max or y_min >= y_max:
            return float('nan')

        local_window = costmap[y_min:y_max, x_min:x_max]

        # Fraction of cells with lethal cost (254 in Nav2)
        return float(np.mean(local_window == 254))
    
    def _compute_r_width(self) -> float:
        """R_width: Corridor width perpendicular to heading."""
        ranges = self.sensor_state.scan_ranges
        angles = self.sensor_state.scan_angles

        if angles is None:
            return self.config.default_r_width

        robot_yaw = self.sensor_state.robot_yaw

        # Genuine world returns; rejected beams become +inf so they cannot win
        # a min() but the array stays full-length for positional indexing.
        valid = self._valid_scan_mask()
        valid_ranges = np.where(valid, ranges, np.inf)

        # World-frame angles
        world_angles = angles + robot_yaw

        # Left perpendicular (robot_yaw + 90°)
        left_angle = robot_yaw + np.pi / 2
        left_diff = np.abs(np.mod(world_angles - left_angle + np.pi, 2 * np.pi) - np.pi)
        left_mask = left_diff < self.config.lateral_tolerance
        left_ranges = valid_ranges[left_mask]
        d_left = (np.min(left_ranges)
                  if len(left_ranges) > 0 and np.any(np.isfinite(left_ranges))
                  else np.inf)

        # Right perpendicular (robot_yaw - 90°)
        right_angle = robot_yaw - np.pi / 2
        right_diff = np.abs(np.mod(world_angles - right_angle + np.pi, 2 * np.pi) - np.pi)
        right_mask = right_diff < self.config.lateral_tolerance
        right_ranges = valid_ranges[right_mask]
        d_right = (np.min(right_ranges)
                   if len(right_ranges) > 0 and np.any(np.isfinite(right_ranges))
                   else np.inf)

        # Corridor width is sum of left and right clearance
        if np.isinf(d_left) and np.isinf(d_right):
            return self.config.default_r_width
        if np.isinf(d_left):
            return float(2 * d_right)
        if np.isinf(d_right):
            return float(2 * d_left)
        return float(d_left + d_right)
        
    
    def _compute_r_clear(self) -> float:
        """R_clear: Maximum free angular sector (VFH-style)."""
        ranges = self.sensor_state.scan_ranges

        if self.cache.sector_indices is None:
            return 0.0

        # Reset histogram (pre-allocated)
        histogram = self.cache.histogram
        histogram.fill(0)

        # Genuine world returns only. Self-returns become +inf and therefore
        # never mark a sector blocked — previously the chassis kept the same
        # sectors permanently blocked, which is why this feature had 7 distinct
        # values across the whole smoke run.
        valid = self._valid_scan_mask()
        valid_ranges = np.where(valid, ranges, np.inf)

        # Mark blocked sectors
        blocked_mask = (valid_ranges < self.config.blocked_distance) & np.isfinite(valid_ranges)
        blocked_sectors = self.cache.sector_indices[blocked_mask]

        if len(blocked_sectors) > 0:
            np.add.at(histogram, blocked_sectors, 1)
        
        # Find largest contiguous free region
        binary_hist = (histogram == 0).astype(np.int8)
        
        # Handle wraparound
        extended = np.concatenate([binary_hist, binary_hist])
        
        # Find longest run of free sectors
        max_run = 0
        current_run = 0
        for val in extended:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        max_run = min(max_run, self.config.num_polar_sectors)
        
        return float(max_run * self.sector_width)
    
    def _compute_r_curve_cached(self) -> float:
        """R_curve: Max path curvature with caching."""
        # Check if cache is valid
        if (self.cache.path_curvatures is not None and 
            self.cache.path_timestamp == self.sensor_state.path_timestamp):
            # Use cached curvatures
            pass
        else:
            # Recompute curvatures
            self._update_path_curvature_cache()
        
        if self.cache.path_curvatures is None or self.cache.path_distances is None:
            return 0.0
        
        # Find max curvature within look-ahead distance
        mask = self.cache.path_distances <= self.config.look_ahead_distance
        
        if not np.any(mask):
            return 0.0
        
        return float(np.max(self.cache.path_curvatures[mask]))
    
    def _update_path_curvature_cache(self) -> None:
        """Recompute path curvatures (called only when path changes)."""
        points = self.sensor_state.path_points
        
        if points is None or len(points) < 3:
            self.cache.path_curvatures = None
            self.cache.path_distances = None
            return
        
        # Compute cumulative distances
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.cache.path_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Compute curvatures at interior points
        curvatures = np.zeros(len(points), dtype=np.float32)
        
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                sin_theta = abs(cross) / (len1 * len2)
                chord = (len1 + len2) / 2
                if chord > 1e-6:
                    curvatures[i] = 2 * sin_theta / chord
        
        self.cache.path_curvatures = curvatures
        self.cache.path_timestamp = self.sensor_state.path_timestamp
    
    def _compute_r_grad_cached(self) -> float:
        """R_grad: gradient magnitude of a fixed-inflation field over the LIVE
        costmap's lethal cells.

        Lethal (==254) cells come from the static and obstacle layers, so this
        still responds to dynamic obstacles. The inflation used here is a fixed
        constant of the risk definition, NOT the inflation_radius treatment, so
        the feature does not encode C_t.
        """
        if (self.cache.gradient_magnitude is None or
                self.cache.gradient_costmap_timestamp != self.sensor_state.costmap_timestamp):
            self._update_gradient_cache()

        if self.cache.gradient_magnitude is None:
            return float('nan')

        rp = self._robot_in_costmap_frame()
        if rp is None:
            return float('nan')
        rx, ry = rp

        resolution = self.sensor_state.costmap_resolution
        mx = int((rx - self.sensor_state.costmap_origin_x) / resolution)
        my = int((ry - self.sensor_state.costmap_origin_y) / resolution)

        h, w = self.cache.gradient_magnitude.shape
        if 0 <= mx < w and 0 <= my < h:
            return float(self.cache.gradient_magnitude[my, mx])
        return float('nan')
    
    def _update_gradient_cache(self) -> None:
        """Rebuild the exogenous cost field and its gradient (costmap changed)."""
        costmap = self.sensor_state.costmap
        if costmap is None or _ndimage is None:
            self.cache.gradient_magnitude = None
            return

        resolution = self.sensor_state.costmap_resolution

        # Observations only — excludes the inflation decay band (252..1) and
        # the inscribed band (253), both of which depend on the treatment.
        lethal = (costmap == 254)

        if not lethal.any():
            self.cache.gradient_magnitude = np.zeros_like(costmap, dtype=np.float32)
            self.cache.gradient_costmap_timestamp = self.sensor_state.costmap_timestamp
            return

        # Distance to the nearest lethal cell, in metres.
        d = _ndimage.distance_transform_edt(~lethal) * resolution

        # Our OWN inflation — constants, never the treatment values.
        rad = self.config.risk_inflation_radius_m      # e.g. 0.55
        k = self.config.risk_cost_scaling_factor       # e.g. 3.0
        cost = np.where(d <= 0.0, 254.0,
               np.where(d >= rad, 0.0, 252.0 * np.exp(-k * d))).astype(np.float32)

        grad_x = np.zeros_like(cost)
        grad_y = np.zeros_like(cost)
        grad_x[:, 1:-1] = (cost[:, 2:] - cost[:, :-2]) / 2.0
        grad_y[1:-1, :] = (cost[2:, :] - cost[:-2, :]) / 2.0

        self.cache.gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) / resolution
        self.cache.gradient_costmap_timestamp = self.sensor_state.costmap_timestamp
    
    # =========================================================================
    # PUBLISHING
    # =========================================================================
    
    def _publish_risk_state(self, risk_state: np.ndarray) -> None:
        """Publish risk state as Float64MultiArray."""
        msg = Float64MultiArray()
        
        # Setup layout for clarity
        msg.layout = MultiArrayLayout()
        msg.layout.dim = [MultiArrayDimension(label='risk_state', size=8, stride=8)]
        msg.layout.data_offset = 0
        
        msg.data = risk_state.tolist()
        
        self._risk_pub.publish(msg)
    
    def _publish_diagnostics(self, components: List[str], last_time_ms: float) -> None:
        """Publish timing diagnostics."""
        msg = DiagnosticStatus()
        msg.name = 'RiskStateCalculator'
        
        if self._compute_times:
            times = np.array(self._compute_times)
            msg.values = [
                KeyValue(key='mean_ms', value=f'{np.mean(times):.3f}'),
                KeyValue(key='p95_ms', value=f'{np.percentile(times, 95):.3f}'),
                KeyValue(key='p99_ms', value=f'{np.percentile(times, 99):.3f}'),
                KeyValue(key='max_ms', value=f'{np.max(times):.3f}'),
                KeyValue(key='last_ms', value=f'{last_time_ms:.3f}'),
                KeyValue(key='cycles_total', value=str(self._cycles_computed)),
                KeyValue(key='cycles_degraded', value=str(self._cycles_degraded)),
                KeyValue(key='degradation_rate', value=f'{100*self._cycles_degraded/max(1,self._cycles_computed):.1f}%'),
                KeyValue(key='components_last', value=','.join(components)),
                # Health signals for the two fixes this node depends on.
                # self_hit_fraction near 1.0 means the scan is dominated by the
                # robot's own structure and the self-filter radius is too small.
                # tf_failure_rate above ~0 means r_dens/r_grad are returning NaN
                # because the costmap frame cannot be resolved.
                KeyValue(key='self_hit_fraction',
                         value=f'{self.self_hit_fraction():.3f}'),
                KeyValue(key='tf_failure_rate',
                         value=f'{self._tf_failures / max(1, self._tf_attempts):.3f}'),
                KeyValue(key='costmap_frame',
                         value=self.sensor_state.costmap_frame or '<none>'),
            ]
            
            # Set status level
            p99 = np.percentile(times, 99)
            if p99 < self.config.time_budget_ms * 0.5:
                msg.level = DiagnosticStatus.OK
                msg.message = f'Running well: P99={p99:.2f}ms'
            elif p99 < self.config.time_budget_ms:
                msg.level = DiagnosticStatus.WARN
                msg.message = f'Approaching budget: P99={p99:.2f}ms'
            else:
                msg.level = DiagnosticStatus.ERROR
                msg.message = f'Exceeding budget: P99={p99:.2f}ms'
            
            # Reset stats
            self._compute_times = []
        else:
            msg.level = DiagnosticStatus.STALE
            msg.message = 'No data'
        
        self._diag_pub.publish(msg)
        
        # Also log to console
        self.get_logger().info(
            f'Risk state timing: mean={np.mean(times) if self._compute_times else 0:.3f}ms, '
            f'components={len(components)}/8, degraded={self._cycles_degraded}/{self._cycles_computed}'
        )


def main(args=None):
    rclpy.init(args=args)
    
    node = OptimizedRiskStateNode()
    
    # Use multi-threaded executor for parallel sensor callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()