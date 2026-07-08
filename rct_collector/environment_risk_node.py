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
    - /local_costmap/costmap (nav2_msgs/Costmap): Local costmap
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

import numpy as np
from typing import Optional, Tuple, List
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
    
    def _compute_r_min(self) -> float:
        """R_min: Minimum distance to any obstacle."""
        ranges = self.sensor_state.scan_ranges
        r_min = self.sensor_state.scan_range_min
        r_max = self.sensor_state.scan_range_max
        
        # Vectorized valid mask and minimum
        valid_mask = (ranges > r_min) & (ranges < r_max)
        
        if not np.any(valid_mask):
            return self.config.default_r_min
        
        return float(np.min(ranges[valid_mask]))
    
    def _compute_r_ttc(self) -> float:
        """R_ttc: Time to collision at current velocity."""
        vx = self.sensor_state.velocity_x
        
        if abs(vx) < self.config.velocity_epsilon:
            return self.config.default_r_ttc
        
        ranges = self.sensor_state.scan_ranges
        r_min = self.sensor_state.scan_range_min
        r_max = self.sensor_state.scan_range_max
        
        # Use pre-computed forward mask
        if self.cache.forward_mask is None:
            return self.config.default_r_ttc
        
        forward_ranges = ranges[self.cache.forward_mask]
        valid_mask = (forward_ranges > r_min) & (forward_ranges < r_max)
        
        if not np.any(valid_mask):
            return self.config.default_r_ttc
        
        d_front = np.min(forward_ranges[valid_mask])
        
        if vx > 0:
            return min(float(d_front / vx), self.config.default_r_ttc)
        else:
            # Moving backward - could compute backward TTC
            return self.config.default_r_ttc
    
    def _compute_r_vis(self) -> float:
        """R_vis: Fraction of max-range or invalid beams in forward sector."""
        if self.cache.forward_mask is None:
            return 0.0
        
        ranges = self.sensor_state.scan_ranges
        r_min = self.sensor_state.scan_range_min
        r_max = self.sensor_state.scan_range_max
        
        forward_ranges = ranges[self.cache.forward_mask]
        
        if len(forward_ranges) == 0:
            return 0.0
        
        # Count invalid beams (max range, below min, or NaN/inf)
        invalid_mask = (
            (forward_ranges >= r_max * 0.99) |
            (forward_ranges <= r_min) |
            ~np.isfinite(forward_ranges)
        )
        
        return float(np.mean(invalid_mask))
    
    def _compute_r_dens(self) -> float:
        """R_dens: Obstacle density in local window."""
        if self.sensor_state.costmap is None:
            return 0.0
        
        costmap = self.sensor_state.costmap
        resolution = self.sensor_state.costmap_resolution
        origin_x = self.sensor_state.costmap_origin_x
        origin_y = self.sensor_state.costmap_origin_y
        
        # Convert robot position to costmap coordinates
        mx = int((self.sensor_state.robot_x - origin_x) / resolution)
        my = int((self.sensor_state.robot_y - origin_y) / resolution)
        
        # Window size in cells
        window_cells = int(self.config.density_window_radius / resolution)
        
        # Extract local window with bounds checking
        h, w = costmap.shape
        x_min = max(0, mx - window_cells)
        x_max = min(w, mx + window_cells + 1)
        y_min = max(0, my - window_cells)
        y_max = min(h, my + window_cells + 1)
        
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        
        local_window = costmap[y_min:y_max, x_min:x_max]
        
        # Fraction of cells with lethal cost (254 in Nav2)
        return float(np.mean(local_window >= 253))
    
    def _compute_r_width(self) -> float:
        """R_width: Corridor width perpendicular to heading."""
        ranges = self.sensor_state.scan_ranges
        angles = self.sensor_state.scan_angles
        
        if angles is None:
            return self.config.default_r_width
        
        r_min = self.sensor_state.scan_range_min
        r_max = self.sensor_state.scan_range_max
        robot_yaw = self.sensor_state.robot_yaw
        
        # Valid ranges
        valid_mask = (ranges > r_min) & (ranges < r_max)
        valid_ranges = np.where(valid_mask, ranges, np.inf)
        
        # World-frame angles
        world_angles = angles + robot_yaw
        
        # Left perpendicular (robot_yaw + 90°)
        left_angle = robot_yaw + np.pi / 2
        left_diff = np.abs(np.mod(world_angles - left_angle + np.pi, 2 * np.pi) - np.pi)
        left_mask = left_diff < self.config.lateral_tolerance
        left_ranges = valid_ranges[left_mask]
        d_left = np.min(left_ranges) if len(left_ranges) > 0 and np.any(np.isfinite(left_ranges)) else np.inf
        
        # Right perpendicular (robot_yaw - 90°)
        right_angle = robot_yaw - np.pi / 2
        right_diff = np.abs(np.mod(world_angles - right_angle + np.pi, 2 * np.pi) - np.pi)
        right_mask = right_diff < self.config.lateral_tolerance
        right_ranges = valid_ranges[right_mask]
        d_right = np.min(right_ranges) if len(right_ranges) > 0 and np.any(np.isfinite(right_ranges)) else np.inf
        
        # Corridor width is sum of left and right clearance
        if np.isinf(d_left) and np.isinf(d_right):
            return self.config.default_r_width
        elif np.isinf(d_left):
            return float(2 * d_right)
        elif np.isinf(d_right):
            return float(2 * d_left)
        else:
            return float(d_left + d_right)
    
    def _compute_r_clear(self) -> float:
        """R_clear: Maximum free angular sector (VFH-style)."""
        ranges = self.sensor_state.scan_ranges
        
        if self.cache.sector_indices is None:
            return 0.0
        
        r_min = self.sensor_state.scan_range_min
        r_max = self.sensor_state.scan_range_max
        
        # Reset histogram (pre-allocated)
        histogram = self.cache.histogram
        histogram.fill(0)
        
        # Valid ranges
        valid_mask = (ranges > r_min) & (ranges < r_max)
        valid_ranges = np.where(valid_mask, ranges, np.inf)
        
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
        """R_grad: Costmap gradient magnitude with caching."""
        # Check if cache is valid
        if (self.cache.gradient_magnitude is not None and
            self.cache.gradient_costmap_timestamp == self.sensor_state.costmap_timestamp):
            # Use cached gradient
            pass
        else:
            # Recompute gradient
            self._update_gradient_cache()
        
        if self.cache.gradient_magnitude is None:
            return 0.0
        
        # Lookup at robot position
        costmap = self.sensor_state.costmap
        resolution = self.sensor_state.costmap_resolution
        origin_x = self.sensor_state.costmap_origin_x
        origin_y = self.sensor_state.costmap_origin_y
        
        mx = int((self.sensor_state.robot_x - origin_x) / resolution)
        my = int((self.sensor_state.robot_y - origin_y) / resolution)
        
        h, w = self.cache.gradient_magnitude.shape
        
        if 0 <= mx < w and 0 <= my < h:
            return float(self.cache.gradient_magnitude[my, mx])
        
        return 0.0
    
    def _update_gradient_cache(self) -> None:
        """Recompute costmap gradient (called only when costmap changes)."""
        costmap = self.sensor_state.costmap
        
        if costmap is None:
            self.cache.gradient_magnitude = None
            return
        
        resolution = self.sensor_state.costmap_resolution
        
        # Simple finite difference gradient (faster than Sobel for small costmaps)
        grad_x = np.zeros_like(costmap)
        grad_y = np.zeros_like(costmap)
        
        grad_x[:, 1:-1] = (costmap[:, 2:] - costmap[:, :-2]) / 2
        grad_y[1:-1, :] = (costmap[2:, :] - costmap[:-2, :]) / 2
        
        self.cache.gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2) / resolution
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