#!/usr/bin/env python3
"""
Random Pose Sampler from OccupancyGrid Maps.

Samples random start and goal poses that are:
1. In free space (not inside obstacles or unknown regions)
2. Sufficiently far from obstacles (configurable clearance)
3. At a reasonable distance from each other (min/max distance)
4. Optionally within user-defined map bounds

The map is loaded from a standard ROS map_server YAML + PGM/PNG pair.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


class PoseSampler:
    """
    Sample random robot poses from free space in an occupancy grid map.

    The sampling process:
    1. Load the map (PGM/PNG) and its metadata (YAML)
    2. Build a distance-from-obstacles field via EDT
    3. Create a mask of "valid" cells (free + sufficient clearance)
    4. Sample start/goal from valid cells with distance constraints

    Usage:
        sampler = PoseSampler("path/to/map.yaml", obstacle_clearance_m=0.5)
        sampler.load_map()
        start, goal = sampler.sample_start_goal()
    """

    # Standard ROS map thresholds
    FREE_THRESHOLD = 230  # Pixel values above this are free space (white = 254)
    OCCUPIED_THRESHOLD = 50  # Below this are obstacles (black = 0)

    def __init__(
        self,
        map_yaml_path: str,
        obstacle_clearance_m: float = 0.6,
        min_goal_distance: float = 10.0,
        max_goal_distance: float = 15.0,
        sampling_bounds: Optional[dict] = None,
        seed: int | None = None,
    ):
        self.map_yaml_path = Path(map_yaml_path)
        self.obstacle_clearance_m = obstacle_clearance_m
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance
        self.sampling_bounds = sampling_bounds
        self.rng = np.random.default_rng(seed)

        # Loaded at load_map() time
        self.map_image: Optional[np.ndarray] = None
        self.resolution: float = 0.05
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.valid_mask: Optional[np.ndarray] = None
        self.valid_indices: Optional[np.ndarray] = None

    def load_map(self):
        """Load map from YAML + image file and precompute valid sampling mask."""
        # Load YAML metadata
        with open(self.map_yaml_path) as f:
            map_info = yaml.safe_load(f)

        self.resolution = map_info["resolution"]
        origin = map_info["origin"]
        self.origin_x = origin[0]
        self.origin_y = origin[1]

        # Load image
        image_path = self.map_yaml_path.parent / map_info["image"]
        img = Image.open(image_path).convert("L")  # Grayscale
        self.map_image = np.array(img)

        logger.info(
            f"Map loaded: {self.map_image.shape}, resolution={self.resolution}m/px, "
            f"origin=({self.origin_x:.2f}, {self.origin_y:.2f})"
        )

        # Build valid sampling mask
        self._build_valid_mask()

        n_valid = np.sum(self.valid_mask)
        total = self.valid_mask.size
        logger.info(
            f"Valid sampling area: {n_valid} cells ({100*n_valid/total:.1f}% of map)"
        )

        if n_valid < 10:
            raise ValueError(
                f"Only {n_valid} valid cells found! Check map, clearance ({self.obstacle_clearance_m}m), "
                "and sampling bounds."
            )

    def _build_valid_mask(self):
        """
        Build a boolean mask of cells where the robot can be placed.

        Steps:
        1. Threshold map into free/occupied
        2. Compute Euclidean Distance Transform from occupied cells
        3. Mask cells with distance >= obstacle_clearance
        4. Optionally apply spatial bounds
        """
        # Free space mask (high pixel values = free in ROS maps)
        free = self.map_image >= self.FREE_THRESHOLD
        occupied = self.map_image <= self.OCCUPIED_THRESHOLD

        # Distance transform: distance of each cell to nearest occupied cell
        # Note: EDT operates on binary image where True = "background" (non-obstacle)
        dist_from_obstacles = distance_transform_edt(~occupied) * self.resolution

        # Valid = free AND far enough from obstacles
        clearance_cells = self.obstacle_clearance_m
        self.valid_mask = free & (dist_from_obstacles >= clearance_cells)

        # Apply optional spatial bounds
        if self.sampling_bounds:
            bounds_mask = self._make_bounds_mask()
            self.valid_mask &= bounds_mask

        # Cache valid cell indices for fast sampling
        self.valid_indices = np.argwhere(self.valid_mask)  # (N, 2) array of [row, col]

    def _make_bounds_mask(self) -> np.ndarray:
        """Create a mask from user-specified world-coordinate bounds."""
        b = self.sampling_bounds
        rows, cols = self.map_image.shape
        mask = np.zeros((rows, cols), dtype=bool)

        for r in range(rows):
            for c in range(cols):
                wx, wy = self._pixel_to_world(r, c)
                if (
                    b.get("x_min", -np.inf) <= wx <= b.get("x_max", np.inf)
                    and b.get("y_min", -np.inf) <= wy <= b.get("y_max", np.inf)
                ):
                    mask[r, c] = True

        return mask

    def _pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel (row, col) to world coordinates (x, y)."""
        # In ROS maps: origin is bottom-left, image row 0 is top
        height = self.map_image.shape[0]
        x = col * self.resolution + self.origin_x
        y = (height - 1 - row) * self.resolution + self.origin_y
        return x, y

    def _world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to pixel (row, col)."""
        height = self.map_image.shape[0]
        col = int((x - self.origin_x) / self.resolution)
        row = int(height - 1 - (y - self.origin_y) / self.resolution)
        return row, col

    def sample_start_goal(
        self, max_attempts: int = 1000
    ) -> tuple[dict, dict]:
        """
        Sample a valid (start, goal) pose pair.

        Returns:
            Tuple of dicts, each with keys: x, y, yaw (in world frame)

        Raises:
            RuntimeError if no valid pair found within max_attempts
        """
        for attempt in range(max_attempts):
            # Sample start pose
            start_idx = self.rng.integers(len(self.valid_indices))
            start_row, start_col = self.valid_indices[start_idx]
            start_x, start_y = self._pixel_to_world(start_row, start_col)

            # Sample goal pose with distance constraint
            goal_idx = self.rng.integers(len(self.valid_indices))
            goal_row, goal_col = self.valid_indices[goal_idx]
            goal_x, goal_y = self._pixel_to_world(goal_row, goal_col)

            dist = np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)

            if self.min_goal_distance <= dist <= self.max_goal_distance:
                start_yaw = float(self.rng.uniform(-np.pi, np.pi))
                goal_yaw = float(self.rng.uniform(-np.pi, np.pi))

                return (
                    {"x": float(start_x), "y": float(start_y), "yaw": start_yaw},
                    {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                )

        raise RuntimeError(
            f"Could not find valid start/goal pair in {max_attempts} attempts. "
            f"Check distance constraints (min={self.min_goal_distance}, "
            f"max={self.max_goal_distance}) vs map size."
        )

    def generate_and_save(
        self,
        n_poses: int,
        output_path: str = "presampled_poses.json",
    ) -> list[dict]:
        """
        Pre-generate n pose pairs and save them to a JSON file.

        Each entry has: {"start": {"x", "y", "yaw"}, "goal": {"x", "y", "yaw"}, "distance": float}

        Args:
            n_poses: Number of (start, goal) pairs to generate
            output_path: Where to save the JSON file

        Returns:
            The list of generated pose pairs
        """
        import json

        poses = []
        for i in range(n_poses):
            start, goal = self.sample_start_goal()
            dist = np.sqrt((goal["x"] - start["x"]) ** 2 + (goal["y"] - start["y"]) ** 2)
            poses.append({
                "id": i,
                "start": start,
                "goal": goal,
                "distance": round(dist, 3),
            })

        with open(output_path, "w") as f:
            json.dump(poses, f, indent=2)

        logger.info(f"Saved {len(poses)} pose pairs to {output_path}")

        # Print summary stats
        distances = [p["distance"] for p in poses]
        logger.info(
            f"Distance stats: min={min(distances):.2f}m, max={max(distances):.2f}m, "
            f"mean={np.mean(distances):.2f}m, std={np.std(distances):.2f}m"
        )

        return poses

    @staticmethod
    def load_presampled(path: str) -> list[dict]:
        """Load pre-generated pose pairs from a JSON file."""
        import json

        with open(path) as f:
            poses = json.load(f)

        logger.info(f"Loaded {len(poses)} pre-sampled pose pairs from {path}")
        return poses

    def visualize_sampling_area(
        self,
        output_path: str = "sampling_area.png",
        n_samples: int = 5,
        presampled_poses: list[dict] | None = None,
    ):
        """
        Save a visualization of the valid sampling area.

        Args:
            output_path: Where to save the image
            n_samples: Number of sample pairs to draw (ignored if presampled_poses is given)
            presampled_poses: If provided, plot these instead of sampling new ones
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(self.map_image, cmap="gray")
        axes[0].set_title("Raw Map")

        axes[1].imshow(self.valid_mask, cmap="Greens")
        axes[1].set_title(f"Valid Sampling Area (clearance={self.obstacle_clearance_m}m)")

        # Overlay valid area on map
        overlay = np.stack([self.map_image] * 3, axis=-1)  # Grayscale to RGB
        overlay[self.valid_mask, 1] = 200  # Green tint on valid areas
        axes[2].imshow(overlay)

        # Use presampled poses if provided, otherwise sample fresh
        if presampled_poses:
            pose_pairs = [(p["start"], p["goal"]) for p in presampled_poses]
            axes[2].set_title(f"Pre-sampled Poses ({len(pose_pairs)} pairs)")
        else:
            pose_pairs = []
            for _ in range(n_samples):
                try:
                    start, goal = self.sample_start_goal()
                    pose_pairs.append((start, goal))
                except RuntimeError:
                    pass
            axes[2].set_title(f"Sampled Poses ({len(pose_pairs)} pairs)")

        # Plot all pose pairs
        for start, goal in pose_pairs:
            sr, sc = self._world_to_pixel(start["x"], start["y"])
            gr, gc = self._world_to_pixel(goal["x"], goal["y"])
            axes[2].plot(sc, sr, "go", markersize=6, alpha=0.7)  # Start = green
            axes[2].plot(gc, gr, "ro", markersize=6, alpha=0.7)  # Goal = red
            axes[2].plot([sc, gc], [sr, gr], "b--", alpha=0.15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Sampling visualization saved to {output_path}")