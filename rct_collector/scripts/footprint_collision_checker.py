#! /usr/bin/env python3
# Copyright 2021 Samsung Research America
# Copyright 2022 Afif Swaidan
# Copyright 2025 Forough Zamani- Enhanced with geometry computation and collision detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced Footprint Collision Checker for Causal Navigation.

This module extends the original FootprintCollisionChecker with:
1. Automatic geometry computation (inscribed/circumscribed radii) from footprint
2. Multi-source collision detection (costmap + LiDAR)
3. Temporal filtering for robust binary collision labeling
4. Direction-aware clearance computation for non-circular footprints

These enhancements support accurate collision labeling for causal inference
in navigation parameter tuning experiments.
"""

from math import cos, sin
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import time

import numpy as np
from geometry_msgs.msg import Point32, Polygon
from nav2_simple_commander.costmap_2d import PyCostmap2D
from nav2_simple_commander.line_iterator import LineIterator

# Costmap cost values
NO_INFORMATION = 255
LETHAL_OBSTACLE = 254
INSCRIBED_INFLATED_OBSTACLE = 253
MAX_NON_OBSTACLE = 252
FREE_SPACE = 0


@dataclass
class FootprintGeometry:
    """
    Geometric properties computed from a footprint polygon.
    
    All values are computed automatically from the footprint vertices,
    so changing the robot or footprint doesn't require manual updates.
    
    Attributes:
        inscribed_radius: Radius of largest circle fully inside footprint
        circumscribed_radius: Radius of smallest circle containing footprint
        centroid: Geometric center of the footprint (x, y)
        area: Area of the footprint polygon in m²
    """
    inscribed_radius: float
    circumscribed_radius: float
    centroid: Tuple[float, float]
    area: float
    
    # Cache for surface distances at various angles
    _surface_distances: np.ndarray = field(default=None, repr=False)
    _surface_angles: np.ndarray = field(default=None, repr=False)


class FootprintCollisionChecker:
    """
    Enhanced FootprintCollisionChecker with geometry computation and collision detection.
    
    This class provides:
    1. Original footprint cost computation on costmap
    2. Automatic geometry computation from footprint polygon
    3. LiDAR-based collision checking with footprint awareness
    4. Multi-source collision detection with temporal filtering
    
    Example:
        # Create checker and set footprint
        checker = FootprintCollisionChecker()
        footprint = [[0.3, 0.3], [-0.3, 0.3], [-0.3, -1.3], [0.3, -1.3]]
        checker.setFootprint(footprint)
        
        # Geometry is computed automatically
        print(f"Inscribed radius: {checker.geometry.inscribed_radius}")
        print(f"Circumscribed radius: {checker.geometry.circumscribed_radius}")
        
        # Check collision using multiple sources
        is_collision, diagnostics = checker.checkCollision(
            robot_x=1.0, robot_y=2.0, robot_yaw=0.5,
            scan_ranges=lidar_ranges,
            scan_angles=lidar_angles
        )
    """

    def __init__(self):
        """Initialize the FootprintCollisionChecker Object."""
        self.costmap_ = None
        self.map_data_ = None
        
        # Footprint and geometry
        self.footprint_list_: Optional[List[Tuple[float, float]]] = None
        self.footprint_polygon_: Optional[Polygon] = None
        self.geometry: Optional[FootprintGeometry] = None
        
        # Collision detection settings
        self.lethal_cost_threshold: int = INSCRIBED_INFLATED_OBSTACLE  # 253
        self.collision_margin: float = 0.05  # 5cm margin for LiDAR
        self.require_both_sources: bool = True
        self.min_confirmation_frames: int = 2
        
        # Temporal filtering state
        self._consecutive_detections: int = 0
        self._detection_history: List[Tuple[float, bool, bool, float, float]] = []
        
        # Surface distance cache settings
        self._cache_resolution: int = 360

    # =========================================================================
    # Footprint and Geometry Setup
    # =========================================================================
    
    def footprint_polyon_generator(self, footprint: List[Tuple[float, float]]) -> Polygon:
        """Generate a ROS Polygon message from footprint vertices."""
        polygon = Polygon()
        for x, y in footprint:
            pt = Point32()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.0
            polygon.points.append(pt)
        return polygon
    
    
    
    def setFootprint(self, footprint: List[Tuple[float, float]]) -> None:
        """
        Set the robot footprint and compute its geometry.
        
        Args:
            footprint: List of (x, y) vertices defining the footprint polygon
                       in robot-centric coordinates. Should be in counter-clockwise order.
        
        Example:
            # TIAGo with arm extended side
            checker.setFootprint([
                (0.3, 0.3), (-0.3, 0.3), (-0.3, -1.3), (0.3, -1.3)
            ])
        """
        if len(footprint) < 3:
            raise ValueError("Footprint must have at least 3 vertices")
        
        self.footprint_list_ = footprint
        
        # # Create ROS Polygon message
        # self.footprint_polygon_ = Polygon()
        # for x, y in footprint:
        #     pt = Point32()
        #     pt.x = float(x)
        #     pt.y = float(y)
        #     pt.z = 0.0
        #     self.footprint_polygon_.points.append(pt)
        
        self. footprint_polygon_ = self.footprint_polyon_generator(footprint)
        
        # Compute geometry
        self.geometry = self._computeGeometry(footprint)
        
        # Precompute surface distances for fast LiDAR checking
        self._precomputeSurfaceDistances(footprint)
    
    def _computeGeometry(self, footprint: List[Tuple[float, float]]) -> FootprintGeometry:
        """Compute geometric properties from footprint vertices."""
        vertices = np.array(footprint)
        
        # Compute centroid
        centroid = self._computeCentroid(vertices)
        
        # Compute area
        area = self._computePolygonArea(vertices)
        
        # Compute circumscribed radius (max distance from centroid to vertex)
        circumscribed = self._computeCircumscribedRadius(vertices, centroid)
        
        # Compute inscribed radius (min distance from centroid to edge)
        inscribed = self._computeInscribedRadius(vertices, centroid)
        
        return FootprintGeometry(
            inscribed_radius=inscribed,
            circumscribed_radius=circumscribed,
            centroid=centroid,
            area=area
        )
    
    def _computeCentroid(self, vertices: np.ndarray) -> Tuple[float, float]:
        """Compute polygon centroid using the shoelace formula."""
        n = len(vertices)
        
        x = np.append(vertices[:, 0], vertices[0, 0])
        y = np.append(vertices[:, 1], vertices[0, 1])
        
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        
        if abs(signed_area) < 1e-10:
            return (float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1])))
        
        cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        
        return (float(cx), float(cy))
    
    def _computePolygonArea(self, vertices: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    
    def _computeCircumscribedRadius(self, vertices: np.ndarray, 
                                     centroid: Tuple[float, float]) -> float:
        """Compute circumscribed radius (max distance from centroid to vertex)."""
        cx, cy = centroid
        distances = np.sqrt((vertices[:, 0] - cx)**2 + (vertices[:, 1] - cy)**2)
        return float(np.max(distances))
    
    def _computeInscribedRadius(self, vertices: np.ndarray,
                                 centroid: Tuple[float, float]) -> float:
        """Compute inscribed radius (min distance from centroid to any edge)."""
        cx, cy = centroid
        n = len(vertices)
        min_distance = float('inf')
        
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            dist = self._pointToSegmentDistance(cx, cy, p1[0], p1[1], p2[0], p2[1])
            min_distance = min(min_distance, dist)
        
        return float(min_distance)
    
    def _pointToSegmentDistance(self, px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """Compute shortest distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        
        if seg_len_sq < 1e-10:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def _precomputeSurfaceDistances(self, footprint: List[Tuple[float, float]]) -> None:
        """Precompute surface distances at regular angles for fast lookup."""
        angles = np.linspace(0, 2 * np.pi, self._cache_resolution, endpoint=False)
        distances = np.array([
            self._computeSurfaceDistanceAtAngle(footprint, angle)
            for angle in angles
        ])
        
        self.geometry._surface_angles = angles
        self.geometry._surface_distances = distances
    
    def _computeSurfaceDistanceAtAngle(self, footprint: List[Tuple[float, float]], 
                                        angle: float) -> float:
        """Compute distance from centroid to footprint boundary at given angle."""
        cx, cy = self.geometry.centroid
        ray_dx = np.cos(angle)
        ray_dy = np.sin(angle)
        
        vertices = np.array(footprint)
        n = len(vertices)
        min_dist = float('inf')
        
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            
            intersection = self._raySegmentIntersection(
                cx, cy, ray_dx, ray_dy,
                p1[0], p1[1], p2[0], p2[1]
            )
            
            if intersection is not None:
                dist = np.sqrt((intersection[0] - cx)**2 + (intersection[1] - cy)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else self.geometry.inscribed_radius
    
    def _raySegmentIntersection(self, rx: float, ry: float, rdx: float, rdy: float,
                                 x1: float, y1: float, x2: float, y2: float
                                 ) -> Optional[Tuple[float, float]]:
        """Compute intersection of ray with line segment."""
        sdx = x2 - x1
        sdy = y2 - y1
        denom = rdx * sdy - rdy * sdx
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - rx) * sdy - (y1 - ry) * sdx) / denom
        u = ((x1 - rx) * rdy - (y1 - ry) * rdx) / denom
        
        if t >= 0 and 0 <= u <= 1:
            return (rx + t * rdx, ry + t * rdy)
        
        return None
    
    def getSurfaceDistance(self, angle: float) -> float:
        """
        Get distance from centroid to footprint surface at given angle.
        
        Uses precomputed cache for efficiency.
        
        Args:
            angle: Angle in radians (0 = +x direction)
        
        Returns:
            Distance to footprint surface at that angle
        """
        if self.geometry is None or self.geometry._surface_distances is None:
            raise ValueError("Footprint not set. Call setFootprint() first.")
        
        angle = np.mod(angle, 2 * np.pi)
        idx = int(angle / (2 * np.pi) * self._cache_resolution) % self._cache_resolution
        return float(self.geometry._surface_distances[idx])
    
    def getSurfaceDistancesBatch(self, angles: np.ndarray) -> np.ndarray:
        """Get surface distances for multiple angles efficiently."""
        if self.geometry is None or self.geometry._surface_distances is None:
            raise ValueError("Footprint not set. Call setFootprint() first.")
        
        normalized = np.mod(angles, 2 * np.pi)
        indices = (normalized / (2 * np.pi) * self._cache_resolution).astype(int)
        indices = indices % self._cache_resolution
        return self.geometry._surface_distances[indices]

    # =========================================================================
    # Original Costmap-based Methods (preserved)
    # =========================================================================
    
    def set_mapData(self, map_data):
        """Set map metadata."""
        self.map_data_ = map_data
        
    def footprintCost(self, footprint: Polygon):
        """
        Iterate over all the points in a footprint and check for collision.

        Args:
            footprint (Polygon): The footprint to calculate the collision cost for

        Returns:
            LETHAL_OBSTACLE (int): If collision was found, 254 will be returned
            footprint_cost (float): The maximum cost found in the footprint points
        """
        footprint_cost = 0.0

        x0, y0 = self.worldToMapValidated(footprint.points[0].x, footprint.points[0].y)

        if x0 is None or y0 is None:
            return LETHAL_OBSTACLE

        xstart = x0
        ystart = y0

        for i in range(len(footprint.points) - 1):
            x1, y1 = self.worldToMapValidated(
                footprint.points[i + 1].x, footprint.points[i + 1].y
            )

            if x1 is None or y1 is None:
                return LETHAL_OBSTACLE

            footprint_cost = max(float(self.lineCost(x0, x1, y0, y1, step_size=0.1)), footprint_cost)
            x0 = x1
            y0 = y1

            if footprint_cost == LETHAL_OBSTACLE:
                return footprint_cost

        return max(float(self.lineCost(xstart, x1, ystart, y1, step_size=0.1)), footprint_cost)

    def lineCost(self, x0, x1, y0, y1, step_size=0.5):
        """
        Iterate over all the points along a line and check for collision.

        Args:
            x0 (float): Abscissa of the initial point in map coordinates
            y0 (float): Ordinate of the initial point in map coordinates
            x1 (float): Abscissa of the final point in map coordinates
            y1 (float): Ordinate of the final point in map coordinates
            step_size (float): Optional, Increments' resolution, defaults to 0.5

        Returns:
            LETHAL_OBSTACLE (int): If collision was found, 254 will be returned
            line_cost (float): The maximum cost found in the line points
        """
        line_cost = 0.0
        point_cost = -1.0

        # Degenerate segment: adjacent footprint vertices can truncate into the
        # same integer costmap cell when their separation is <= ~1 cell
        # (resolution). LineIterator rejects zero-length lines, so evaluate the
        # single cell directly instead.
        if int(x0) == int(x1) and int(y0) == int(y1):
            return float(self.pointCost(int(x0), int(y0)))

        line_iterator = LineIterator(x0, y0, x1, y1, step_size)

        while line_iterator.isValid():
            point_cost = self.pointCost(
                int(line_iterator.getX()), int(line_iterator.getY())
            )

            if point_cost == LETHAL_OBSTACLE:
                return point_cost

            if line_cost < point_cost:
                line_cost = point_cost

            line_iterator.advance()

        return line_cost

    def worldToMapValidated(self, wx: float, wy: float):
        """
        Get the map coordinate XY using world coordinate XY.

        Args:
            wx (float): world coordinate X
            wy (float): world coordinate Y

        Returns:
            None: if coordinates are invalid
            tuple of int: mx, my (if coordinates are valid)
        """
        if self.costmap_ is None:
            raise ValueError(
                'Costmap not specified, use setCostmap to specify the costmap first'
            )
        
        map_data = self.map_data_
        
        origin_x = map_data['origin_x']
        origin_y = map_data['origin_y']
        resolution = map_data['resolution']
        size_in_cell_x = map_data['size_in_cell_x']
        size_in_cell_y = map_data['size_in_cell_y']
        
        if wx < origin_x or wy < origin_y:
            return None, None

        mx = int((wx - origin_x) / resolution)
        my = int((wy - origin_y) / resolution)
            
        if (0 <= mx < size_in_cell_x) and (0 <= my < size_in_cell_y):
            return (mx, my)
        
        return None, None

    def pointCost(self, x: int, y: int):
        """
        Get the cost of a point in the costmap using map coordinates XY.

        Args:
            x (int): map coordinate X
            y (int): map coordinate Y

        Returns:
            np.uint8: cost of a point
        """
        if self.costmap_ is None:
            raise ValueError(
                'Costmap not specified, use setCostmap to specify the costmap first'
            )
        return self.costmap_.map.data[self.map_data_['size_in_cell_x'] * y + x]

    def setCostmap(self, costmap: PyCostmap2D):
        """
        Specify which costmap to use.

        Args:
            costmap (PyCostmap2D): costmap to use in the object's methods

        Returns:
            None
        """
        self.costmap_ = costmap
        map_data = {}
        map_data['origin_x'] = costmap.map.metadata.origin.position.x
        map_data['origin_y'] = costmap.map.metadata.origin.position.y
        map_data['resolution'] = costmap.map.metadata.resolution
        map_data['size_in_cell_x'] = costmap.map.metadata.size_x
        map_data['size_in_cell_y'] = costmap.map.metadata.size_y
        self.set_mapData(map_data)
        
        return None

    def footprintCostAtPose(self, x: float, y: float, theta: float, 
                            footprint: Optional[Polygon] = None):
        """
        Get the cost of a footprint at a specific Pose in world coordinates.

        Args:
            x (float): world coordinate X
            y (float): world coordinate Y
            theta (float): absolute rotation angle of the footprint
            footprint (Polygon): Optional, the footprint polygon. If None, uses
                                 the footprint set via setFootprint()

        Returns:
            LETHAL_OBSTACLE (int): If collision was found, 254 will be returned
            footprint_cost (float): The maximum cost found in the footprint points
        """
        if footprint is None:
            if self.footprint_polygon_ is None:
                raise ValueError("No footprint specified. Either pass footprint argument "
                               "or call setFootprint() first.")
            footprint = self.footprint_polygon_
        
        cos_th = cos(theta)
        sin_th = sin(theta)
        oriented_footprint = Polygon()

        for i in range(len(footprint.points)):
            new_pt = Point32()
            new_pt.x = x + (
                footprint.points[i].x * cos_th - footprint.points[i].y * sin_th
            )
            new_pt.y = y + (
                footprint.points[i].x * sin_th + footprint.points[i].y * cos_th
            )
            oriented_footprint.points.append(new_pt)

        return self.footprintCost(oriented_footprint)

    # =========================================================================
    # Enhanced Collision Detection (new methods)
    # =========================================================================
    
    def checkCollision(
        self,
        robot_x: float,
        robot_y: float, 
        robot_yaw: float,
        scan_ranges: np.ndarray,
        scan_angles: np.ndarray,
        footprint: Optional[Polygon] = None
    ) -> Tuple[bool, dict]:
        """
        Check for collision using both costmap and LiDAR sources.
        
        This method provides robust binary collision detection for RCT experiments
        by combining multiple sources with temporal filtering.
        
        Args:
            robot_x: Robot x position in world frame
            robot_y: Robot y position in world frame
            robot_yaw: Robot yaw angle in world frame
            scan_ranges: LiDAR range readings (meters)
            scan_angles: Angles for each range reading (radians, in robot frame)
            footprint: Optional footprint polygon. If None, uses setFootprint() value.
        
        Returns:
            Tuple of (is_collision, diagnostics_dict)
            
            diagnostics_dict contains:
                - costmap_collision: bool
                - lidar_collision: bool
                - footprint_cost: float
                - lidar_min_clearance: float
                - lidar_min_clearance_angle: float
                - consecutive_detections: int
                - is_collision: bool (final decision)
        """
        timestamp = time.time()
        diagnostics = {
            'timestamp': timestamp,
        }
        
        # Add geometry info if available
        if self.geometry is not None:
            diagnostics['inscribed_radius'] = self.geometry.inscribed_radius
            diagnostics['circumscribed_radius'] = self.geometry.circumscribed_radius
        
        # Source 1: Costmap footprint check
        costmap_collision = False
        footprint_cost = 0.0
        
        if self.costmap_ is not None:
            footprint_cost = self.footprintCostAtPose(robot_x, robot_y, robot_yaw, footprint)
            costmap_collision = footprint_cost >= self.lethal_cost_threshold
        
        diagnostics['costmap_collision'] = costmap_collision
        diagnostics['footprint_cost'] = footprint_cost
        
        # Source 2: LiDAR proximity check
        lidar_collision, lidar_min_clearance, lidar_min_angle = self._checkLidarCollision(
            scan_ranges, scan_angles
        )
        diagnostics['lidar_collision'] = lidar_collision
        diagnostics['lidar_min_clearance'] = lidar_min_clearance
        diagnostics['lidar_min_clearance_angle'] = lidar_min_angle
        
        # Combine sources
        if self.require_both_sources:
            frame_collision = costmap_collision and lidar_collision
        else:
            frame_collision = costmap_collision or lidar_collision
        
        diagnostics['frame_collision'] = frame_collision
        
        # Temporal filtering
        self._detection_history.append((
            timestamp, costmap_collision, lidar_collision, footprint_cost, lidar_min_clearance
        ))
        
        # Keep limited history
        max_history = max(10, self.min_confirmation_frames * 2)
        if len(self._detection_history) > max_history:
            self._detection_history.pop(0)
        
        if frame_collision:
            self._consecutive_detections += 1
        else:
            self._consecutive_detections = 0
        
        diagnostics['consecutive_detections'] = self._consecutive_detections
        
        # Final decision
        is_collision = self._consecutive_detections >= self.min_confirmation_frames
        diagnostics['is_collision'] = is_collision
        
        return is_collision, diagnostics
    
    def _checkLidarCollision(
        self,
        ranges: np.ndarray,
        angles: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Check LiDAR for collision, accounting for footprint geometry.
        
        Returns:
            (collision_detected, minimum_clearance, angle_of_minimum)
        """
        # Filter valid readings
        valid_mask = np.isfinite(ranges) & (ranges > 0.01) & (ranges < 30.0)
        
        if not np.any(valid_mask):
            return False, float('inf'), 0.0
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Get surface distances for all angles
        if self.geometry is not None and self.geometry._surface_distances is not None:
            surface_distances = self.getSurfaceDistancesBatch(valid_angles)
        else:
            # Fallback to inscribed radius if geometry not set
            radius = self.geometry.inscribed_radius if self.geometry else 0.3
            surface_distances = np.full_like(valid_ranges, radius)
        
        # Compute clearances (distance from robot surface to obstacle)
        clearances = valid_ranges - surface_distances
        
        # Find minimum clearance
        min_idx = np.argmin(clearances)
        min_clearance = float(clearances[min_idx])
        min_angle = float(valid_angles[min_idx])
        
        # Collision if clearance is less than margin
        collision = min_clearance < self.collision_margin
        
        return collision, min_clearance, min_angle
    
    def resetCollisionDetection(self) -> None:
        """Reset collision detection state for a new trial."""
        self._consecutive_detections = 0
        self._detection_history.clear()
    
    def getCollisionStatistics(self) -> dict:
        """Get statistics from collision detection history."""
        if not self._detection_history:
            return {}
        
        history = np.array([
            (costmap, lidar, cost, clearance)
            for _, costmap, lidar, cost, clearance in self._detection_history
        ])
        
        return {
            'num_frames': len(self._detection_history),
            'costmap_collision_rate': float(np.mean(history[:, 0])),
            'lidar_collision_rate': float(np.mean(history[:, 1])),
            'mean_footprint_cost': float(np.mean(history[:, 2])),
            'mean_lidar_clearance': float(np.mean(history[:, 3])),
            'min_lidar_clearance': float(np.min(history[:, 3])),
        }
    
    def getMinDistanceToObstacle(
        self,
        scan_ranges: np.ndarray,
        scan_angles: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get minimum distance from robot surface to nearest obstacle.
        
        This accounts for non-circular footprint geometry.
        
        Args:
            scan_ranges: LiDAR range readings
            scan_angles: Corresponding angles
        
        Returns:
            (min_clearance, angle_of_minimum)
        """
        valid_mask = np.isfinite(scan_ranges) & (scan_ranges > 0.01) & (scan_ranges < 30.0)
        
        if not np.any(valid_mask):
            return float('inf'), 0.0
        
        valid_ranges = scan_ranges[valid_mask]
        valid_angles = scan_angles[valid_mask]
        
        if self.geometry is not None and self.geometry._surface_distances is not None:
            surface_distances = self.getSurfaceDistancesBatch(valid_angles)
        else:
            surface_distances = np.zeros_like(valid_ranges)
        
        clearances = valid_ranges - surface_distances
        min_idx = np.argmin(clearances)
        
        return float(clearances[min_idx]), float(valid_angles[min_idx])


# =============================================================================
# Convenience functions for common footprints
# =============================================================================

def create_circular_footprint(radius: float, num_points: int = 16) -> List[Tuple[float, float]]:
    """Create a circular footprint approximation."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]


def create_rectangular_footprint(
    length: float, 
    width: float, 
    center_offset: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    """
    Create a rectangular footprint.
    
    Args:
        length: Dimension in x direction (front-back)
        width: Dimension in y direction (left-right)
        center_offset: Offset of rectangle center from robot origin
    """
    cx, cy = center_offset
    half_l = length / 2
    half_w = width / 2
    
    return [
        (cx + half_l, cy + half_w),
        (cx - half_l, cy + half_w),
        (cx - half_l, cy - half_w),
        (cx + half_l, cy - half_w),
    ]


def create_tiago_footprint(arm_extended: bool = True) -> List[Tuple[float, float]]:
    """Create footprint for TIAGo robot."""
    if arm_extended:
        return [
            (0.3, 0.3), (-0.3, 0.3), (-0.3, -0.1),
            (-0.15, -0.1), (-0.15, -1.3), (0.15, -1.3),
            (0.15, -0.1), (0.3, -0.1), (0.3, -0.3),
        ]
    else:
        return create_circular_footprint(radius=0.27, num_points=8)


def create_pmb2_footprint() -> List[Tuple[float, float]]:
    """Create footprint for PMB2 robot."""
    return create_circular_footprint(radius=0.27, num_points=16)


# =============================================================================
# Test
# =============================================================================

# if __name__ == '__main__':
#     print("Testing FootprintCollisionChecker geometry computation\n")
    
#     # Test with TIAGo footprint
#     checker = FootprintCollisionChecker()
    
#     print("=" * 60)
#     print("TIAGo with arm extended:")
#     print("=" * 60)
    
#     tiago_footprint = create_tiago_footprint(arm_extended=True)
#     checker.setFootprint(tiago_footprint)
    
#     print(f"Footprint vertices: {len(tiago_footprint)}")
#     print(f"Centroid: ({checker.geometry.centroid[0]:.3f}, {checker.geometry.centroid[1]:.3f})")
#     print(f"Inscribed radius: {checker.geometry.inscribed_radius:.3f}m")
#     print(f"Circumscribed radius: {checker.geometry.circumscribed_radius:.3f}m")
#     print(f"Area: {checker.geometry.area:.3f}m²")
    
#     print("\nSurface distances at various angles:")
#     for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
#         angle_rad = np.radians(angle_deg)
#         dist = checker.getSurfaceDistance(angle_rad)
#         print(f"  {angle_deg:3d}°: {dist:.3f}m")
    
#     print("\n" + "=" * 60)
#     print("PMB2 (circular):")
#     print("=" * 60)
    
#     pmb2_footprint = create_pmb2_footprint()
#     checker.setFootprint(pmb2_footprint)
    
#     print(f"Inscribed radius: {checker.geometry.inscribed_radius:.3f}m")
#     print(f"Circumscribed radius: {checker.geometry.circumscribed_radius:.3f}m")
    
#     print("\n" + "=" * 60)
#     print("Simulated LiDAR collision check:")
#     print("=" * 60)
    
#     # Reset to TIAGo
#     checker.setFootprint(tiago_footprint)
    
#     # Simulate LiDAR data with obstacle in front
#     num_beams = 360
#     angles = np.linspace(-np.pi, np.pi, num_beams)
#     ranges = np.full(num_beams, 5.0)
    
#     # Add obstacle at 0.4m in front
#     front_mask = np.abs(angles) < np.radians(30)
#     ranges[front_mask] = 0.4
    
#     min_clearance, min_angle = checker.getMinDistanceToObstacle(ranges, angles)
#     print(f"Min clearance: {min_clearance:.3f}m at angle {np.degrees(min_angle):.1f}°")
    
#     # Note: Full checkCollision() requires costmap, so we test LiDAR part only
#     collision, clearance, angle = checker._checkLidarCollision(ranges, angles)
#     print(f"LiDAR collision detected: {collision}")
#     print(f"Clearance at detection: {clearance:.3f}m")