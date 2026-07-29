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


def _parse_polygon(poly):
    """Accept a footprint as a list of [x, y] or the string form used in the
    footprint config (e.g. '[[-0.275, 0.0], [0.238, 0.138], ...]'). Returns an
    (N, 2) float array, or None if not provided."""
    if poly is None:
        return None
    if isinstance(poly, str):
        import json
        poly = json.loads(poly)
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("footprint_polygon must be >=3 [x, y] vertices")
    return arr


def _polygon_radii(poly: np.ndarray) -> tuple[float, float]:
    """Inscribed and circumscribed radii about the robot origin (0, 0), which
    is the frame the footprint is defined in and the point Nav2 rotates about.

    inscribed  = min distance from origin to any edge (circle guaranteed inside)
    circumscribed = max distance from origin to any vertex (circle guaranteed
                    to contain the footprint at every yaw)
    """
    origin = np.zeros(2)
    n = len(poly)
    ins = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        denom = float(ab @ ab) or 1e-12
        t = np.clip((origin - a) @ ab / denom, 0.0, 1.0)
        ins = min(ins, float(np.linalg.norm(origin - (a + t * ab))))
    circ = float(np.max(np.linalg.norm(poly, axis=1)))
    return ins, circ


def _points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon. pts (M,2), poly (N,2) -> (M,)
    bool. Loops over the (few) polygon edges, vectorised over the many points,
    so it needs no matplotlib and stays fast at pose-generation time."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


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
        obstacle_clearance_m: float = 0.35,
        min_goal_distance: float = 10.0,
        max_goal_distance: float = 15.0,
        sampling_bounds: Optional[dict] = None,
        seed: int | None = None,
        footprint_clearance_m: float = 0.0,
        footprint_polygon=None,
        footprint_yaw_bins: int = 72,
        footprint_safety_margin_m: float = 0.0,
    ):
        self.map_yaml_path = Path(map_yaml_path)
        self.obstacle_clearance_m = obstacle_clearance_m
        # Minimum clearance so that the LARGEST footprint the experiment will
        # use (the extended "carry" arm) can physically exist at every sampled
        # pose. Set this to the carry footprint's inscribed radius.
        #
        # This is a PRE-TREATMENT, purely geometric filter: it depends only on
        # the map and the robot's physical extent, not on which arm state a
        # trial is later assigned or on whether any planner succeeds. The SAME
        # pool is therefore used for both arm states, which keeps arm
        # independent of start geometry (no selection on a treatment-affected
        # variable). Residual planning failures under carry — the arm genuinely
        # cannot traverse a corridor — are left in on purpose: that is a real
        # causal effect of the arm, not a sampling artefact, and must not be
        # filtered out.
        self.footprint_clearance_m = footprint_clearance_m
        # Orientation-aware feasibility (Option 2). When footprint_polygon is
        # given (the robot-frame vertices of the LARGEST / carry footprint), a
        # candidate pose is feasible iff that polygon, placed at the pose's
        # (x, y, yaw), does not overlap an occupied cell. This keeps poses where
        # the arm would hit a wall at one heading but fits at another — which a
        # single clearance circle would wrongly discard. As with the circular
        # option, the SAME feasible pool is used for both arm states, so arm
        # stays independent of start geometry.
        self.footprint_polygon = _parse_polygon(footprint_polygon)
        self.footprint_yaw_bins = int(footprint_yaw_bins)
        self.footprint_safety_margin_m = float(footprint_safety_margin_m)
        self._orientation_aware = self.footprint_polygon is not None
        # Filled by load_map(): EDT field, inscribed/circumscribed radii, and
        # one pixel-offset mask per yaw bin.
        self._dist_field: Optional[np.ndarray] = None
        self._r_in: float = 0.0
        self._r_circ: float = 0.0
        self._fp_masks: list = []
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

    def _precompute_footprint_masks(self):
        """One pixel-offset mask per yaw bin: the (drow, dcol) cells the
        footprint covers when placed at the origin cell at that yaw. Checking a
        candidate then reduces to occupancy lookups at these offsets."""
        poly = self.footprint_polygon
        res = self.resolution
        margin_px = self.footprint_safety_margin_m / res
        self._fp_masks = []
        for k in range(self.footprint_yaw_bins):
            yaw = 2.0 * np.pi * k / self.footprint_yaw_bins
            c, s = np.cos(yaw), np.sin(yaw)
            wx = c * poly[:, 0] - s * poly[:, 1]
            wy = s * poly[:, 0] + c * poly[:, 1]
            # World metres -> pixel offsets. +x -> +col, +y -> -row.
            pc = wx / res
            pr = -wy / res
            verts = np.column_stack([pc, pr])  # (col, row) space
            lo_c, hi_c = np.floor(pc.min()), np.ceil(pc.max())
            lo_r, hi_r = np.floor(pr.min()), np.ceil(pr.max())
            cols, rows = np.meshgrid(np.arange(lo_c, hi_c + 1),
                                     np.arange(lo_r, hi_r + 1))
            pts = np.column_stack([cols.ravel(), rows.ravel()])
            inside = _points_in_poly(pts, verts)
            drow = rows.ravel()[inside].astype(int)
            dcol = cols.ravel()[inside].astype(int)
            if margin_px > 0 and drow.size:
                # Dilate the mask by the safety margin (in pixels) so the
                # footprint keeps a buffer from walls.
                rad = int(np.ceil(margin_px))
                offs = [(dr, dc) for dr in range(-rad, rad + 1)
                        for dc in range(-rad, rad + 1)
                        if dr * dr + dc * dc <= margin_px * margin_px]
                pairs = {(int(r + dr), int(c + dc))
                         for r, c in zip(drow, dcol) for dr, dc in offs}
                drow = np.array([p[0] for p in pairs], int)
                dcol = np.array([p[1] for p in pairs], int)
            self._fp_masks.append((drow, dcol))

    def _footprint_feasible(self, row: int, col: int, yaw_bin: int) -> bool:
        """True iff the footprint at (row, col) rotated to yaw_bin does not
        overlap an occupied cell. Uses the EDT prefilter to skip the polygon
        test wherever the answer is already decided by clearance."""
        dist = self._dist_field[row, col]
        if dist >= self._r_circ:
            return True                    # fits at every yaw
        if dist < self._r_in:
            return False                   # base overlaps at every yaw
        drow, dcol = self._fp_masks[yaw_bin % self.footprint_yaw_bins]
        rr = row + drow
        cc = col + dcol
        H, W = self.map_image.shape
        if rr.min() < 0 or rr.max() >= H or cc.min() < 0 or cc.max() >= W:
            return False                   # footprint would leave the map
        return not bool(np.any(self.map_image[rr, cc] <= self.OCCUPIED_THRESHOLD))

    def _sample_feasible_pose(self, max_tries: int = 400):
        """Draw a (row, col, yaw) whose footprint fits. Rejection is over the
        JOINT (cell, yaw), so the result is uniform over feasible poses — cells
        with few feasible headings contribute proportionally less, which is the
        correct pre-treatment pose distribution."""
        for _ in range(max_tries):
            idx = self.rng.integers(len(self.valid_indices))
            row, col = self.valid_indices[idx]
            if not self._orientation_aware:
                yaw = float(self.rng.uniform(-np.pi, np.pi))
                return int(row), int(col), yaw
            yaw_bin = int(self.rng.integers(self.footprint_yaw_bins))
            if self._footprint_feasible(int(row), int(col), yaw_bin):
                # Store the exact yaw that was checked (bin centre).
                yaw = 2.0 * np.pi * yaw_bin / self.footprint_yaw_bins
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
                return int(row), int(col), float(yaw)
        raise RuntimeError(
            "Could not sample a footprint-feasible pose. The carry footprint "
            "may be too large for this map — check the visualisation, widen the "
            "map, or reduce footprint_safety_margin_m.")

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
        self._dist_field = dist_from_obstacles  # cached for footprint checks

        if self._orientation_aware:
            # Feasibility is decided per (cell, yaw) by the footprint polygon,
            # not by a single clearance circle. The valid mask here is the set
            # of cells that COULD host the footprint at some yaw: those whose
            # clearance is at least the inscribed radius (below it, even the
            # base overlaps, so no yaw works). Cells at or above the
            # circumscribed radius fit at every yaw. The band in between is
            # resolved by the per-yaw mask at sampling time.
            self._r_in, self._r_circ = _polygon_radii(self.footprint_polygon)
            self._r_in += self.footprint_safety_margin_m
            self._r_circ += self.footprint_safety_margin_m
            self._precompute_footprint_masks()
            self.valid_mask = free & (dist_from_obstacles >= self._r_in)
        else:
            # Valid = free AND far enough from obstacles. The required clearance
            # is the larger of the sampling clearance and the carry-footprint
            # radius, so every sampled pose admits the largest footprint the
            # experiment uses. Using one pool for both arm states keeps
            # arm ⟂ start geometry.
            clearance_cells = max(self.obstacle_clearance_m,
                                  self.footprint_clearance_m)
            self.valid_mask = free & (dist_from_obstacles >= clearance_cells)

        # Apply optional spatial bounds
        if self.sampling_bounds:
            bounds_mask = self._make_bounds_mask()
            self.valid_mask &= bounds_mask

        # Cache valid cell indices for fast sampling
        self.valid_indices = np.argwhere(self.valid_mask)  # (N, 2) array of [row, col]

        if self._orientation_aware:
            band = (self._dist_field >= self._r_in) & (self._dist_field < self._r_circ) & free
            auto = (self._dist_field >= self._r_circ) & free
            logger.info(
                f"Orientation-aware pose pool: footprint inscribed={self._r_in:.3f} m, "
                f"circumscribed={self._r_circ:.3f} m. "
                f"{int(auto.sum())} cells fit at every yaw, "
                f"{int(band.sum())} cells need the per-yaw test, "
                f"rest excluded. Same pool is used for BOTH arm states.")
        elif self.footprint_clearance_m > self.obstacle_clearance_m:
            eff = max(self.obstacle_clearance_m, self.footprint_clearance_m)
            logger.info(
                f"Pose pool built with footprint clearance {eff:.2f} m "
                f"(carry-feasible); this same pool is used for BOTH arm states.")

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
            # Feasible start (footprint fits at its yaw, when orientation-aware).
            start_row, start_col, start_yaw = self._sample_feasible_pose()
            start_x, start_y = self._pixel_to_world(start_row, start_col)

            # Feasible goal with the distance constraint.
            goal_row, goal_col, goal_yaw = self._sample_feasible_pose()
            goal_x, goal_y = self._pixel_to_world(goal_row, goal_col)

            dist = np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)

            if self.min_goal_distance <= dist <= self.max_goal_distance:
                return (
                    {"x": float(start_x), "y": float(start_y), "yaw": float(start_yaw)},
                    {"x": float(goal_x), "y": float(goal_y), "yaw": float(goal_yaw)},
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