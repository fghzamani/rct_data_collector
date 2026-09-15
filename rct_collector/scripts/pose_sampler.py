#!/usr/bin/env python3
"""
Random Pose Sampler from OccupancyGrid Maps.

Samples random start and goal poses that are:
1. In free space (not inside obstacles or unknown regions)
2. Sufficiently far from obstacles (configurable clearance)
3. At a reasonable distance from each other (min/max distance)
4. Optionally within user-defined map bounds

The map is loaded from a standard ROS map_server YAML + PGM/PNG pair.

Campaign A probe sampling
-------------------------
The v2 sampler (``sample_stratified_probe_pose``, ``constriction="rooms"``)
replaces the straight-line constriction test used by the v1 samplers. Three
things changed and each one is a separate defect that was suppressing the
carry-envelope effect:

1. The v1 window was measured FROM THE START, assuming a time-based apply:
   ``_horizon_event_window`` put the tight point at 3.4-3.8 m from the start
   pose. The orchestrator actually fires on REMAINING distance to goal, drawn
   uniformly in ``[guard, d_total - t_min_baseline]`` by
   ``RCTOrchestrator._draw_trigger_distance``. On an 11 m plan the treatment
   lands anywhere from 0.4 m to 8 m from the start while the constriction sits
   at a fixed 3.6 m, so the robot usually passed the tight point under the
   baseline configuration. ``_apply_window_remaining()`` below expresses the
   window in the same coordinate the trigger uses: remaining distance to goal.

2. v1 tested clearance along the STRAIGHT start->goal line, which the global
   planner is free to route around, and on the collected data it did: the
   median planned path was 1.28x the straight line and the "wall" stratum
   detoured as much as "straight". The "doorway" stratum here uses map
   topology instead. Start and goal must sit in different connected components
   of the eroded free space, so any feasible path has to pass a throat. A
   planner cannot detour around a topological requirement.

3. v1 capped the start yaw at ``max_angle_offset_deg`` from the goal bearing
   (60 degrees in the shipped pool), so no probe ever required the robot to
   turn around. Rotation is where an extended arm sweeps into geometry a
   tucked base never touches. ``initial_turn_deg`` now defaults to the full
   (0, 180) range.

Identification is unaffected. Pose selection reads only the static map, and
the configuration C is drawn independently downstream, so
``P(Y | do(C), R) = P(Y | C, R)`` still holds by randomization. What changes is
the marginal distribution of R: this is a deliberate covariate shift toward
geometry where the carry envelope binds. Pools generated with different
strata weights should be tagged so they are not silently pooled in analysis.

The v1 methods (``sample_probe_pose``, ``sample_constriction_probe_pose``,
``sample_probe_pose_enriched``) are unchanged so the existing pool remains
reproducible for provenance.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, grey_dilation, label

logger = logging.getLogger(__name__)


class PoseSampler:
    """
    Sample random robot poses from free space in an occupancy grid map.

    The sampling process:
    1. Load the map (PGM/PNG) and its metadata (YAML)
    2. Build a distance-from-obstacles field via EDT
    3. Create a mask of "valid" cells (free + sufficient clearance)
    4. Segment the map into rooms and doorway throats (v2 only)
    5. Sample start/goal from valid cells with distance constraints

    Usage:
        sampler = PoseSampler("path/to/map.yaml", obstacle_clearance_m=0.4)
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
        goal_clearance_m: Optional[float] = 0.70,
        min_goal_distance: float = 10.0,
        max_goal_distance: float = 15.0,
        sampling_bounds: Optional[dict] = None,
        seed: int | None = None,
        # --- v2 geometry -------------------------------------------------
        # Circumscribed radii of the two footprint polygons, NOT inscribed.
        # Measured from param_actual__local_costmap__footprint in the
        # Campaign A CSV: worst tucked vertex (0.217, 0.242) -> 0.325 m;
        # worst carry vertex (0.480, -0.698) -> 0.847 m.
        nav_radius_m: float = 0.325,
        carry_radius_m: float = 0.847,
        # Erosion depth used to cut the free space into rooms. Must exceed
        # half the width of the doorways you want to treat as separators and
        # stay below half the width of corridors you want to keep connected.
        # A 0.75 m doorway has 0.375 m centre clearance, so 0.55 separates it.
        room_split_clearance_m: float = 0.55,
        # Components smaller than this are noise, not rooms.
        min_room_area_m2: float = 1.5,
        # How far a throat cell may sit from the two rooms it links. Must span
        # the wall thickness plus the erosion depth on both sides.
        gateway_link_radius_m: float = 0.8,
    ):
        self.map_yaml_path = Path(map_yaml_path)
        self.obstacle_clearance_m = obstacle_clearance_m
        self.goal_clearance_m = goal_clearance_m if goal_clearance_m is not None else 0.70
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance
        self.sampling_bounds = sampling_bounds
        self.rng = np.random.default_rng(seed)

        self.nav_radius_m = nav_radius_m
        self.carry_radius_m = carry_radius_m
        self.room_split_clearance_m = room_split_clearance_m
        self.min_room_area_m2 = min_room_area_m2
        self.gateway_link_radius_m = gateway_link_radius_m

        # Loaded at load_map() time
        self.map_image: Optional[np.ndarray] = None
        self.resolution: float = 0.05
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.valid_mask: Optional[np.ndarray] = None
        self.valid_indices: Optional[np.ndarray] = None
        self.dist_field: Optional[np.ndarray] = None   # per-cell clearance (m)

        # v2 topology, built alongside the valid mask
        self.nav_mask: Optional[np.ndarray] = None      # tucked robot can stand here
        self.nav_labels: Optional[np.ndarray] = None    # reachability components
        self.room_labels: Optional[np.ndarray] = None   # 0 = throat/unassigned
        self.n_rooms: int = 0
        self.gateway_rc: Optional[np.ndarray] = None    # (N,2) throat cells

    # ------------------------------------------------------------------ #
    # Map loading
    # ------------------------------------------------------------------ #

    def load_map(self):
        """Load map from YAML + image file and precompute valid sampling mask."""
        with open(self.map_yaml_path) as f:
            map_info = yaml.safe_load(f)

        self.resolution = map_info["resolution"]
        origin = map_info["origin"]
        self.origin_x = origin[0]
        self.origin_y = origin[1]

        image_path = self.map_yaml_path.parent / map_info["image"]
        img = Image.open(image_path).convert("L")  # Grayscale
        self.map_image = np.array(img)

        logger.info(
            f"Map loaded: {self.map_image.shape}, resolution={self.resolution}m/px, "
            f"origin=({self.origin_x:.2f}, {self.origin_y:.2f})"
        )

        self._build_valid_mask()
        self._build_topology()

        n_valid = np.sum(self.valid_mask)
        total = self.valid_mask.size
        logger.info(
            f"Valid sampling area: {n_valid} cells ({100*n_valid/total:.1f}% of map)"
        )

        if n_valid < 10:
            raise ValueError(
                f"Only {n_valid} valid cells found! Check map, clearance "
                f"({self.obstacle_clearance_m}m), and sampling bounds."
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
        free = self.map_image >= self.FREE_THRESHOLD
        occupied = self.map_image <= self.OCCUPIED_THRESHOLD

        dist_from_obstacles = distance_transform_edt(~occupied) * self.resolution
        self.dist_field = dist_from_obstacles      # per-cell clearance, metres

        self.free_mask = free
        self.valid_mask = free & (dist_from_obstacles >= self.obstacle_clearance_m)
        self.goal_mask = free & (dist_from_obstacles >= self.goal_clearance_m)

        if self.sampling_bounds:
            bounds_mask = self._make_bounds_mask()
            self.valid_mask &= bounds_mask
            self.goal_mask &= bounds_mask

        self.valid_indices = np.argwhere(self.valid_mask)  # (N, 2) [row, col]
        self.goal_indices = np.argwhere(self.goal_mask)    # (N, 2) [row, col]

    def _build_topology(self):
        """Segment free space into rooms and doorway throats.

        Three derived masks, all from the distance field already computed:

        ``nav_mask``    cells where a tucked robot (nav_radius_m) fits. Its
                        connected components answer "can the robot get there
                        at all", which is the reachability precondition for
                        any sampled pair.
        ``room_labels`` connected components of the free space eroded by
                        ``room_split_clearance_m``. Doorways narrower than
                        twice that clearance pinch off, so each component is
                        approximately a room. Label 0 means throat or wall.
        ``gateway_rc``  cells a tucked robot fits through but that belong to
                        no room. These are the doorway throats, and they are
                        the anchor the doorway stratum samples around.
        """
        free = self.free_mask

        self.nav_mask = free & (self.dist_field >= self.nav_radius_m)
        self.nav_labels, n_nav = label(self.nav_mask)

        room_mask = free & (self.dist_field >= self.room_split_clearance_m)
        room_labels, n_raw = label(room_mask)

        # Drop components too small to be a room; they are corners and noise.
        min_cells = int(self.min_room_area_m2 / (self.resolution ** 2))
        if n_raw > 0:
            counts = np.bincount(room_labels.ravel(), minlength=n_raw + 1)
            keep = np.zeros(n_raw + 1, dtype=np.int32)
            next_id = 0
            for lab in range(1, n_raw + 1):
                if counts[lab] >= min_cells:
                    next_id += 1
                    keep[lab] = next_id
            room_labels = keep[room_labels]
            self.n_rooms = next_id
        else:
            self.n_rooms = 0
        self.room_labels = room_labels

        # A throat is a navigable cell belonging to no room that LINKS two
        # different rooms. Without the link test every cell in the
        # nav_radius..room_split band qualifies, which is the whole strip
        # hugging every wall — thousands of cells that are not doorways.
        k = 2 * int(round(self.gateway_link_radius_m / self.resolution)) + 1
        big = grey_dilation(room_labels, size=(k, k))
        far = np.where(room_labels > 0, room_labels, np.iinfo(np.int32).max)
        small = -grey_dilation(-far, size=(k, k))
        links_two_rooms = (big > 0) & (small < np.iinfo(np.int32).max) & (big != small)

        gateway = self.nav_mask & (room_labels == 0) & links_two_rooms
        self.gateway_rc = np.argwhere(gateway)

        logger.info(
            f"Topology: {self.n_rooms} rooms at {self.room_split_clearance_m} m "
            f"erosion, {n_nav} navigable components at {self.nav_radius_m} m, "
            f"{len(self.gateway_rc)} throat cells."
        )
        if self.n_rooms < 2:
            logger.warning(
                "Fewer than 2 rooms found — the doorway stratum will always "
                "fall back. Raise room_split_clearance_m (doorways are not "
                "pinching off) or lower it (corridors are over-splitting)."
            )

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

    def _in_grid(self, row: int, col: int) -> bool:
        return (0 <= row < self.dist_field.shape[0]
                and 0 <= col < self.dist_field.shape[1])

    # ------------------------------------------------------------------ #
    # v1 sampling — unchanged, kept so the shipped pool stays reproducible
    # ------------------------------------------------------------------ #

    def sample_start_goal(self, max_attempts: int = 1000) -> tuple[dict, dict]:
        """
        Sample a valid (start, goal) pose pair.

        Returns:
            Tuple of dicts, each with keys: x, y, yaw (in world frame)

        Raises:
            RuntimeError if no valid pair found within max_attempts
        """
        for attempt in range(max_attempts):
            start_idx = self.rng.integers(len(self.valid_indices))
            start_row, start_col = self.valid_indices[start_idx]
            start_x, start_y = self._pixel_to_world(start_row, start_col)

            goal_idx = self.rng.integers(len(self.goal_indices))
            goal_row, goal_col = self.goal_indices[goal_idx]
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

    def sample_probe_pose(
        self,
        forward_distance_m: float = 4.0,
        max_angle_offset_deg: float = 15.0,
        max_yaw_offset_deg: float = 15.0,
        max_attempts: int = 1000,
    ) -> tuple[dict, dict]:
        """
        Sample a feasible start pose and a local goal pose for Campaign A.

        Adds translational and rotational variance:
        - Distance is sampled in [forward_distance_m, forward_distance_m + 2.0].
        - Direction angle has a random offset of up to +/- max_angle_offset_deg.
        - Goal orientation yaw has a random offset of up to +/- max_yaw_offset_deg.
        """
        d_min = float(forward_distance_m)
        d_max = float(forward_distance_m + 2.0)
        dir_offset_rad = float(np.radians(max_angle_offset_deg))
        yaw_offset_rad = float(np.radians(max_yaw_offset_deg))

        for _ in range(max_attempts):
            idx = self.rng.integers(len(self.valid_indices))
            row, col = self.valid_indices[idx]
            x, y = self._pixel_to_world(row, col)
            start_yaw = float(self.rng.uniform(-np.pi, np.pi))

            d = float(self.rng.uniform(d_min, d_max))
            dir_angle = float(self.rng.uniform(-dir_offset_rad, dir_offset_rad))
            yaw_angle = float(self.rng.uniform(-yaw_offset_rad, yaw_offset_rad))

            move_heading = start_yaw + dir_angle
            goal_x = x + d * np.cos(move_heading)
            goal_y = y + d * np.sin(move_heading)
            goal_yaw = float(np.mod(move_heading + yaw_angle + np.pi, 2 * np.pi) - np.pi)

            g_row, g_col = self._world_to_pixel(goal_x, goal_y)

            in_bounds = (0 <= g_row < self.valid_mask.shape[0]
                         and 0 <= g_col < self.valid_mask.shape[1])
            if in_bounds and self.valid_mask[g_row, g_col]:
                return (
                    {"x": float(x), "y": float(y), "yaw": start_yaw},
                    {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                )

        raise RuntimeError(
            f"Could not find a feasible probe pose (forward_distance_m="
            f"{forward_distance_m}) in {max_attempts} attempts. Check clearance/map."
        )

    def _horizon_event_window(self, forward_distance_m, v_base_mps,
                              baseline_settle_sec, t_apply_max_sec, v_max_mps,
                              horizon_sec, arm_settle_sec, observe_sec):
        """DEPRECATED — measures from the start pose, which does not match the
        orchestrator's remaining-distance trigger. Kept only so the v1 samplers
        below reproduce the shipped pool. Use ``_apply_window_remaining``."""
        apply_max = v_base_mps * (baseline_settle_sec + t_apply_max_sec)
        lo = apply_max + v_base_mps * arm_settle_sec
        hi = apply_max + v_max_mps * max(0.0, horizon_sec - observe_sec)
        return lo, hi

    def _path_clear(self, sr, sc, gr, gc, n):
        rs = np.clip(np.linspace(sr, gr, n).astype(int), 0, self.dist_field.shape[0]-1)
        cs = np.clip(np.linspace(sc, gc, n).astype(int), 0, self.dist_field.shape[1]-1)
        return rs, cs, self.dist_field[rs, cs]

    def sample_probe_pose_enriched(
        self,
        forward_distance_m: float = 4.5,
        v_base_mps: float = 0.40,
        baseline_settle_sec: float = 2.0,
        t_apply_max_sec: float = 2.5,
        v_max_mps: float = 0.50,
        horizon_sec: float = 8.0,
        arm_settle_sec: float = 4.0,
        observe_sec: float = 1.0,
        nav_radius_m: float = 0.33,
        carry_radius_m: float = 0.85,
        tight_clearance_m: float = 0.55,
        max_wall_clip_frac: float = 0.10,
        wall_follow_min_frac: float = 0.15,
        min_window_tight_frac: float = 0.40,  # >=this fraction of the horizon
                                              # window must be below carry radius,
                                              # so the robot is genuinely IN tight
                                              # space, not grazing one corner

        turn_min_deg: float = 40.0,
        strata_weights=(0.15, 0.425, 0.425),   # (straight, turn, wall)
        max_angle_offset_deg: float = 60.0,
        max_yaw_offset_deg: float = 15.0,
        n_path_samples: int = 60,
        max_attempts: int = 6000,
    ):
        """v1 three-stratum sampler. Superseded by ``sample_stratified_probe_pose``
        — see the module docstring for why. Retained for reproducibility."""
        d_min = float(forward_distance_m)
        d_max = float(forward_distance_m + 2.0)
        dir_off = float(np.radians(max_angle_offset_deg))
        yaw_off = float(np.radians(max_yaw_offset_deg))
        lo, hi = self._horizon_event_window(
            forward_distance_m, v_base_mps, baseline_settle_sec, t_apply_max_sec,
            v_max_mps, horizon_sec, arm_settle_sec, observe_sec)

        if isinstance(strata_weights, dict):
            p = [strata_weights.get(m, 0.33) for m in ["straight", "turn", "wall"]]
            p_sum = sum(p)
            p = [v / p_sum for v in p] if p_sum > 0 else [0.34, 0.33, 0.33]
        else:
            p = list(strata_weights)
        mode = self.rng.choice(["straight", "turn", "wall"], p=p)


        for _ in range(max_attempts):
            idx = self.rng.integers(len(self.valid_indices))
            row, col = self.valid_indices[idx]
            x, y = self._pixel_to_world(row, col)
            start_yaw = float(self.rng.uniform(-np.pi, np.pi))

            d = float(self.rng.uniform(d_min, d_max))
            dir_angle = float(self.rng.uniform(-dir_off, dir_off))
            yaw_angle = float(self.rng.uniform(-yaw_off, yaw_off))
            move_heading = start_yaw + dir_angle
            goal_x = x + d * np.cos(move_heading)
            goal_y = y + d * np.sin(move_heading)
            goal_yaw = float(np.mod(move_heading + yaw_angle + np.pi, 2*np.pi) - np.pi)

            g_row, g_col = self._world_to_pixel(goal_x, goal_y)
            if not (0 <= g_row < self.valid_mask.shape[0]
                    and 0 <= g_col < self.valid_mask.shape[1]):
                continue
            if not self.valid_mask[g_row, g_col]:
                continue

            rs, cs, clear = self._path_clear(row, col, g_row, g_col, n_path_samples)
            if clear.min() <= 0.01:
                continue
            frac_below_nav = float((clear < nav_radius_m).mean())

            # Only path samples inside the horizon window can produce a
            # collision under the applied config. Searching the whole line makes
            # argmin land far past the window on long paths.
            dists = np.linspace(0.0, d, n_path_samples)
            win = (dists >= lo) & (dists <= hi)
            if not win.any():
                continue

            # GLOBAL GATE: the window must be sustained-tight for the carry
            # envelope. Without this, a path can cross open space and merely
            # touch one tight sample, which produces no collision.
            if float((clear[win] < carry_radius_m).mean()) < min_window_tight_frac:
                continue

            ok = False
            if mode == "straight":
                widx = np.where(win)[0]
                ti = int(widx[np.argmin(clear[widx])])
                tv = clear[ti]
                td = (ti / (n_path_samples - 1)) * d
                ok = (nav_radius_m <= tv < tight_clearance_m
                      and frac_below_nav <= max_wall_clip_frac
                      and lo <= td <= hi)

            elif mode == "turn":
                if abs(np.degrees(dir_angle)) < turn_min_deg:
                    continue
                band = (clear >= nav_radius_m) & (clear < carry_radius_m)
                if not band.any():
                    continue
                bw = band & win
                if not bw.any():
                    continue
                ti = int(np.where(bw)[0][0])
                td = (ti / (n_path_samples - 1)) * d
                ok = (nav_radius_m <= clear[ti] < carry_radius_m
                      and clear.min() >= nav_radius_m - 0.02
                      and lo <= td <= hi)

            elif mode == "wall":
                band = (clear >= nav_radius_m) & (clear < carry_radius_m)
                bw = band & win
                if bw.sum() < max(2, int(wall_follow_min_frac * win.sum())):
                    continue
                run_idx = np.where(bw)[0]
                td = (run_idx.mean() / (n_path_samples - 1)) * d
                ok = (clear.min() >= nav_radius_m - 0.02
                      and lo <= td <= hi)

            if ok:
                return (
                    {"x": float(x), "y": float(y), "yaw": start_yaw},
                    {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                    True, mode,
                )

        s2, g2 = self.sample_probe_pose(
            forward_distance_m=forward_distance_m,
            max_angle_offset_deg=15.0, max_yaw_offset_deg=max_yaw_offset_deg)
        return s2, g2, False, "fallback"

    def sample_constriction_probe_pose(
        self,
        forward_distance_m: float = 4.5,
        v_base_mps: float = 0.40,
        baseline_settle_sec: float = 2.0,
        t_apply_max_sec: float = 2.5,
        v_max_mps: float = 0.50,
        horizon_sec: float = 8.0,
        tight_clearance_m: float = 0.55,
        nav_radius_m: float = 0.33,
        max_wall_clip_frac: float = 0.10,
        max_angle_offset_deg: float = 15.0,
        max_yaw_offset_deg: float = 15.0,
        n_path_samples: int = 60,
        max_attempts: int = 4000,
    ) -> tuple[dict, dict, bool]:
        """v1 single-constriction sampler. Superseded — retained for
        reproducibility of the shipped pool."""
        d_min = float(forward_distance_m)
        d_max = float(forward_distance_m + 2.0)
        dir_off = float(np.radians(max_angle_offset_deg))
        yaw_off = float(np.radians(max_yaw_offset_deg))

        window_lo = v_base_mps * baseline_settle_sec
        window_hi = (v_base_mps * (baseline_settle_sec + t_apply_max_sec)
                     + v_max_mps * horizon_sec)

        for _ in range(max_attempts):
            idx = self.rng.integers(len(self.valid_indices))
            row, col = self.valid_indices[idx]
            x, y = self._pixel_to_world(row, col)
            start_yaw = float(self.rng.uniform(-np.pi, np.pi))

            d = float(self.rng.uniform(d_min, d_max))
            dir_angle = float(self.rng.uniform(-dir_off, dir_off))
            yaw_angle = float(self.rng.uniform(-yaw_off, yaw_off))
            move_heading = start_yaw + dir_angle
            goal_x = x + d * np.cos(move_heading)
            goal_y = y + d * np.sin(move_heading)
            goal_yaw = float(np.mod(move_heading + yaw_angle + np.pi, 2 * np.pi) - np.pi)

            g_row, g_col = self._world_to_pixel(goal_x, goal_y)
            if not (0 <= g_row < self.valid_mask.shape[0]
                    and 0 <= g_col < self.valid_mask.shape[1]):
                continue
            if not self.valid_mask[g_row, g_col]:
                continue

            rs = np.linspace(row, g_row, n_path_samples).astype(int)
            cs = np.linspace(col, g_col, n_path_samples).astype(int)
            rs = np.clip(rs, 0, self.dist_field.shape[0] - 1)
            cs = np.clip(cs, 0, self.dist_field.shape[1] - 1)
            clear = self.dist_field[rs, cs]

            tight_i = int(np.argmin(clear))
            tight_val = clear[tight_i]
            tight_dist = (tight_i / (n_path_samples - 1)) * d
            frac_below_nav = float((clear < nav_radius_m).mean())
            crossed = (nav_radius_m <= tight_val < tight_clearance_m
                       and frac_below_nav <= max_wall_clip_frac
                       and window_lo <= tight_dist <= window_hi)
            if crossed:
                return (
                    {"x": float(x), "y": float(y), "yaw": start_yaw},
                    {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                    True,
                )

        s, g = self.sample_probe_pose(
            forward_distance_m=forward_distance_m,
            max_angle_offset_deg=max_angle_offset_deg,
            max_yaw_offset_deg=max_yaw_offset_deg,
        )
        return s, g, False

    # ------------------------------------------------------------------ #
    # v2 sampling
    # ------------------------------------------------------------------ #

    def _apply_window_remaining(
        self,
        v_base_mps: float = 0.40,
        v_max_mps: float = 0.50,
        horizon_sec: float = 8.0,
        goal_tolerance_m: float = 0.25,
        trigger_margin_m: float = 0.25,
        arm_move_time_sec: float = 4.0,
        arm_settle_sec: float = 4.0,
    ) -> tuple[float, float]:
        """Remaining-distance band in which the collision-causing geometry must
        sit, in the SAME coordinate the orchestrator's trigger uses.

        ``RCTOrchestrator._draw_trigger_distance`` fires when remaining path
        length falls below a threshold drawn uniformly in
        ``[guard, d_total - t_min_baseline]``, where
        ``guard = v_max * horizon + goal_tolerance + trigger_margin``. The
        earliest the treatment can land is therefore at ``guard`` metres
        remaining.

        After the trigger, ``_set_arm_for_config`` blocks on the arm action and
        then sleeps ``arm_settle_sec`` while the base keeps driving, so the
        robot covers roughly ``v_base * (arm_move + arm_settle)`` metres before
        the collision window even opens. The window then runs for
        ``horizon_sec``.

        Returns (lo, hi) as remaining distance to goal:
          hi = guard - v_base * (arm_move + arm_settle)   window opens
          lo = guard - v_max  * (arm_move + arm_settle + horizon)  window closes

        Anything the robot reaches at a remaining distance inside [lo, hi] is
        observed with the treatment physically in place. lo is clipped at 0.
        """
        guard = v_max_mps * horizon_sec + goal_tolerance_m + trigger_margin_m
        t_arm = arm_move_time_sec + arm_settle_sec
        hi = guard - v_base_mps * t_arm
        lo = guard - v_max_mps * (t_arm + horizon_sec)
        return max(0.0, lo), max(0.0, hi)

    def _segment_clearance(self, ax, ay, bx, by, n=None):
        """Min clearance along the straight world segment a->b, in metres."""
        ar, ac = self._world_to_pixel(ax, ay)
        br, bc = self._world_to_pixel(bx, by)
        if n is None:
            n = max(8, int(np.hypot(br - ar, bc - ac)) + 1)
        rs = np.clip(np.linspace(ar, br, n).astype(int), 0, self.dist_field.shape[0] - 1)
        cs = np.clip(np.linspace(ac, bc, n).astype(int), 0, self.dist_field.shape[1] - 1)
        return float(self.dist_field[rs, cs].min())

    def _label_at(self, labels, x, y) -> int:
        r, c = self._world_to_pixel(x, y)
        if not self._in_grid(r, c):
            return 0
        return int(labels[r, c])

    def _draw_turn(self, turn_deg_range) -> float:
        """Signed initial-turn magnitude in radians, drawn from a degree range."""
        lo, hi = turn_deg_range
        mag = float(self.rng.uniform(np.radians(lo), np.radians(hi)))
        return mag * float(self.rng.choice([-1.0, 1.0]))

    def sample_doorway_probe_pose(
        self,
        d_min_m: float = 3.6,
        d_max_m: float = 5.5,
        goal_past_gateway_m: tuple = None,   # default: the apply window
        initial_turn_deg: tuple = (0.0, 180.0),
        bend_at_gateway_deg: tuple = (0.0, 60.0),
        goal_yaw_offset_deg: float = 180.0,
        gateway_clearance_band: tuple = None,  # default: [nav_radius, carry_radius)
        max_attempts: int = 4000,
        **window_kwargs,
    ):
        """Sample a pair whose every feasible path crosses a doorway throat,
        with the throat inside the post-treatment observation window.

        Construction, anchored on a throat cell g rather than on the start:

          1. Draw a throat g whose clearance admits a tucked base but not a
             carry envelope, so passing it is a genuine envelope test.
          2. Place the goal ``r_goal`` metres past g, where ``r_goal`` lies in
             the remaining-distance window from ``_apply_window_remaining``.
             The robot therefore reaches g with the treatment applied and the
             arm physically settled.
          3. Place the start ``d_total - r_goal`` metres back from g, bent by
             ``bend_at_gateway_deg`` so the throat is not always approached
             head-on.
          4. Require start and goal in DIFFERENT room components, so no
             planner route avoids the throat, and in the SAME navigable
             component, so a tucked base can actually get through.
          5. Require line-of-sight clearance of at least ``nav_radius_m`` on
             both legs, which makes start -> g -> goal a feasible route and
             makes the two-segment length a sound estimate of path length.

        Start yaw is drawn independently of the bearing over
        ``initial_turn_deg``, defaulting to the full (0, 180) range so roughly
        half the probes require a turn of more than 90 degrees before the robot
        makes any forward progress.

        Returns (start, goal, crossed, stratum, meta) where meta records the
        throat position and clearance, the distance from start to the throat,
        and the initial turn — so the analysis can condition on where the
        constriction was rather than infer it.
        """
        if self.gateway_rc is None or len(self.gateway_rc) == 0:
            return None

        lo, hi = self._apply_window_remaining(**window_kwargs)
        if goal_past_gateway_m is None:
            goal_past_gateway_m = (max(0.3, lo), max(0.6, hi))
        if gateway_clearance_band is None:
            gateway_clearance_band = (self.nav_radius_m, self.carry_radius_m)

        gb_lo, gb_hi = gateway_clearance_band
        gp_lo, gp_hi = goal_past_gateway_m

        for _ in range(max_attempts):
            gi = int(self.rng.integers(len(self.gateway_rc)))
            g_row, g_col = self.gateway_rc[gi]
            g_clear = float(self.dist_field[g_row, g_col])
            if not (gb_lo <= g_clear < gb_hi):
                continue
            gx, gy = self._pixel_to_world(int(g_row), int(g_col))

            theta = float(self.rng.uniform(-np.pi, np.pi))
            r_goal = float(self.rng.uniform(gp_lo, gp_hi))
            d_total = float(self.rng.uniform(d_min_m, d_max_m))
            r_start = d_total - r_goal
            if r_start < 1.0:
                continue

            goal_x = gx + r_goal * np.cos(theta)
            goal_y = gy + r_goal * np.sin(theta)

            bend = self._draw_turn(bend_at_gateway_deg)
            back = theta + np.pi + bend
            start_x = gx + r_start * np.cos(back)
            start_y = gy + r_start * np.sin(back)

            s_row, s_col = self._world_to_pixel(start_x, start_y)
            go_row, go_col = self._world_to_pixel(goal_x, goal_y)
            if not (self._in_grid(s_row, s_col) and self._in_grid(go_row, go_col)):
                continue
            if not (self.valid_mask[s_row, s_col] and self.valid_mask[go_row, go_col]):
                continue

            # Different rooms: any feasible path must use a throat.
            room_s = int(self.room_labels[s_row, s_col])
            room_g = int(self.room_labels[go_row, go_col])
            if room_s == 0 or room_g == 0 or room_s == room_g:
                continue

            # Same navigable component: a tucked base can actually get there.
            if self.nav_labels[s_row, s_col] != self.nav_labels[go_row, go_col]:
                continue

            # Both legs traversable, so start -> g -> goal is a real route and
            # d_total is a sound estimate of the planned path length.
            if self._segment_clearance(start_x, start_y, gx, gy) < self.nav_radius_m:
                continue
            if self._segment_clearance(gx, gy, goal_x, goal_y) < self.nav_radius_m:
                continue

            bearing = np.arctan2(gy - start_y, gx - start_x)
            turn = self._draw_turn(initial_turn_deg)
            start_yaw = float(np.mod(bearing + turn + np.pi, 2 * np.pi) - np.pi)
            gy_off = np.radians(goal_yaw_offset_deg)
            goal_yaw = float(np.mod(
                theta + self.rng.uniform(-gy_off, gy_off) + np.pi, 2 * np.pi) - np.pi)

            meta = {
                "gateway_x": round(float(gx), 3),
                "gateway_y": round(float(gy), 3),
                "gateway_clearance_m": round(g_clear, 3),
                "d_start_to_gateway_m": round(float(r_start), 3),
                "d_gateway_to_goal_m": round(float(r_goal), 3),
                "initial_turn_deg": round(float(np.degrees(turn)), 1),
                "bend_at_gateway_deg": round(float(np.degrees(bend)), 1),
                "room_start": room_s,
                "room_goal": room_g,
                "apply_window_remaining_m": [round(lo, 2), round(hi, 2)],
            }
            return (
                {"x": float(start_x), "y": float(start_y), "yaw": start_yaw},
                {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                True, "doorway", meta,
            )
        return None

    def sample_wall_probe_pose(
        self,
        d_min_m: float = 3.6,
        d_max_m: float = 5.5,
        initial_turn_deg: tuple = (0.0, 180.0),
        goal_yaw_offset_deg: float = 180.0,
        min_band_frac: float = 0.25,
        n_path_samples: int = 60,
        max_attempts: int = 4000,
        **window_kwargs,
    ):
        """Sample a pair whose approach to the goal hugs a wall closely enough
        that the carry envelope overlaps it but the tucked base does not.

        Unlike the v1 wall stratum, the sustained close-wall run is required in
        the LAST part of the path — the segment the robot covers inside the
        remaining-distance window — rather than anywhere along it.
        """
        lo, hi = self._apply_window_remaining(**window_kwargs)

        for _ in range(max_attempts):
            idx = int(self.rng.integers(len(self.valid_indices)))
            row, col = self.valid_indices[idx]
            x, y = self._pixel_to_world(int(row), int(col))

            d = float(self.rng.uniform(d_min_m, d_max_m))

            # Test 36 candidate headings to find one aligned along the corridor length
            candidate_headings = self.rng.permutation(np.linspace(-np.pi, np.pi, 36, endpoint=False))
            heading = None
            clear = None
            in_window = None
            band = None

            for h_cand in candidate_headings:
                goal_x = x + d * np.cos(h_cand)
                goal_y = y + d * np.sin(h_cand)

                g_row, g_col = self._world_to_pixel(goal_x, goal_y)
                if not self._in_grid(g_row, g_col) or not self.valid_mask[g_row, g_col]:
                    continue

                _, _, c_tmp = self._path_clear(row, col, g_row, g_col, n_path_samples)
                if c_tmp.min() < self.nav_radius_m:
                    continue

                rem = d * (1.0 - np.linspace(0.0, 1.0, n_path_samples))
                win_tmp = (rem >= lo) & (rem <= hi)
                if not win_tmp.any():
                    continue
                b_tmp = (c_tmp >= self.nav_radius_m) & (c_tmp < self.carry_radius_m)
                if b_tmp[win_tmp].mean() >= min_band_frac:
                    heading = float(h_cand)
                    clear = c_tmp
                    in_window = win_tmp
                    band = b_tmp
                    break

            if heading is None:
                continue


            bearing = heading
            turn = self._draw_turn(initial_turn_deg)
            start_yaw = float(np.mod(bearing + turn + np.pi, 2 * np.pi) - np.pi)
            gy_off = np.radians(goal_yaw_offset_deg)
            goal_yaw = float(np.mod(
                heading + self.rng.uniform(-gy_off, gy_off) + np.pi, 2 * np.pi) - np.pi)

            meta = {
                "min_clearance_in_window_m": round(float(clear[in_window].min()), 3),
                "band_frac_in_window": round(float(band[in_window].mean()), 3),
                "initial_turn_deg": round(float(np.degrees(turn)), 1),
                "apply_window_remaining_m": [round(lo, 2), round(hi, 2)],
            }
            return (
                {"x": float(x), "y": float(y), "yaw": start_yaw},
                {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                True, "wall", meta,
            )
        return None

    def sample_open_probe_pose(
        self,
        d_min_m: float = 3.6,
        d_max_m: float = 5.5,
        initial_turn_deg: tuple = (0.0, 180.0),
        goal_yaw_offset_deg: float = 180.0,
        margin_m: float = 0.15,
        n_path_samples: int = 60,
        max_attempts: int = 2000,
        **window_kwargs,
    ):
        """Control stratum: the whole path clears the carry envelope.

        This is the comparison group. Without it the pool measures a doorway
        robot rather than an office robot, and the interaction between context
        and arm state — which is the effect-modification result — has nothing
        to contrast against.
        """
        need = self.carry_radius_m + margin_m
        for _ in range(max_attempts):
            idx = int(self.rng.integers(len(self.valid_indices)))
            row, col = self.valid_indices[idx]
            x, y = self._pixel_to_world(int(row), int(col))

            d = float(self.rng.uniform(d_min_m, d_max_m))
            heading = float(self.rng.uniform(-np.pi, np.pi))
            goal_x = x + d * np.cos(heading)
            goal_y = y + d * np.sin(heading)

            g_row, g_col = self._world_to_pixel(goal_x, goal_y)
            if not self._in_grid(g_row, g_col) or not self.valid_mask[g_row, g_col]:
                continue

            _, _, clear = self._path_clear(row, col, g_row, g_col, n_path_samples)
            if clear.min() < need:
                continue

            turn = self._draw_turn(initial_turn_deg)
            start_yaw = float(np.mod(heading + turn + np.pi, 2 * np.pi) - np.pi)
            gy_off = np.radians(goal_yaw_offset_deg)
            goal_yaw = float(np.mod(
                heading + self.rng.uniform(-gy_off, gy_off) + np.pi, 2 * np.pi) - np.pi)

            meta = {
                "min_clearance_m": round(float(clear.min()), 3),
                "initial_turn_deg": round(float(np.degrees(turn)), 1),
            }
            return (
                {"x": float(x), "y": float(y), "yaw": start_yaw},
                {"x": float(goal_x), "y": float(goal_y), "yaw": goal_yaw},
                False, "open", meta,
            )
        return None

    def sample_stratified_probe_pose(
        self,
        d_min_m: float = 3.6,
        d_max_m: float = 5.5,
        strata_weights: dict = None,
        initial_turn_deg: tuple = (0.0, 180.0),
        **kwargs,
    ):
        """Draw one probe pair from the v2 strata.

        Default mix is 40% doorway, 25% wall, 35% open. The open stratum is
        deliberately large: it is the control the doorway result is measured
        against, and shrinking it strengthens the headline contrast at the cost
        of the generalization claim.

        Falls back through doorway -> wall -> open -> plain probe pose so a map
        without room structure never stalls collection. The returned stratum
        label always says which one was used, so fallbacks are visible in the
        pool rather than silent.
        """
        weights = strata_weights or {"doorway": 0.40, "wall": 0.25, "open": 0.35}
        names = list(weights.keys())
        probs = np.array([weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        choice = str(self.rng.choice(names, p=probs))

        order = [choice] + [n for n in ("doorway", "wall", "open") if n != choice]
        fns = {
            "doorway": self.sample_doorway_probe_pose,
            "wall": self.sample_wall_probe_pose,
            "open": self.sample_open_probe_pose,
        }
        for name in order:
            out = fns[name](d_min_m=d_min_m, d_max_m=d_max_m,
                            initial_turn_deg=initial_turn_deg, **kwargs)
            if out is not None:
                return out

        s, g = self.sample_probe_pose(
            forward_distance_m=d_min_m, max_angle_offset_deg=180.0,
            max_yaw_offset_deg=180.0)
        return s, g, False, "fallback", {}

    # ------------------------------------------------------------------ #
    # Pool generation
    # ------------------------------------------------------------------ #

    def generate_and_save(
        self,
        n_poses: int,
        output_path: str = "presampled_poses.json",
        campaign: str = "B",
        forward_distance_m: float = 3.5,
        constriction: bool = False,
        constriction_kwargs: Optional[dict] = None,
    ) -> list[dict]:
        """
        Pre-generate n pose pairs and save them to a JSON file.

        Each entry has: {"start": {"x","y","yaw"}, "goal": {"x","y","yaw"},
        "distance": float, "crossed": bool, "stratum": str, "meta": {...}}.

        ``constriction`` selects the Campaign A sampler:
          "rooms"     v2 room-topology strata (recommended)
          "enriched"  v1 three-stratum sampler
          True        v1 single-constriction sampler
          False       plain forward probe

        For "rooms", ``forward_distance_m`` sets the lower end of the pair
        separation and the upper end is +1.9 m. Keep it near 3.6: the
        orchestrator rejects any plan shorter than
        ``guard + t_min_baseline`` (3.4 m with the shipped defaults) as
        POSE_EXHAUSTED, and lengthening it past ~6 m re-widens the uniform
        trigger draw and lets the treatment land far from the constriction.
        """
        import json

        ck = dict(constriction_kwargs or {})
        poses = []
        n_crossed = 0
        for i in range(n_poses):
            crossed = None
            stratum = None
            meta = None
            if campaign == "A" and constriction == "rooms":
                start, goal, crossed, stratum, meta = \
                    self.sample_stratified_probe_pose(
                        d_min_m=forward_distance_m,
                        d_max_m=forward_distance_m + 1.9, **ck)
                n_crossed += int(crossed)
            elif campaign == "A" and constriction == "enriched":
                start, goal, crossed, stratum = self.sample_probe_pose_enriched(
                    forward_distance_m=forward_distance_m, **ck)
                n_crossed += int(crossed)
            elif campaign == "A" and constriction:
                start, goal, crossed = self.sample_constriction_probe_pose(
                    forward_distance_m=forward_distance_m, **ck)
                n_crossed += int(crossed)
            elif campaign == "A":
                start, goal = self.sample_probe_pose(forward_distance_m=forward_distance_m)
            else:
                start, goal = self.sample_start_goal()

            dist = float(np.sqrt((goal["x"] - start["x"]) ** 2
                                 + (goal["y"] - start["y"]) ** 2))
            entry = {
                "id": i,
                "start": start,
                "goal": goal,
                "distance": round(dist, 3),
            }
            if crossed is not None:
                entry["crossed"] = bool(crossed)
            if stratum is not None:
                entry["stratum"] = stratum
            if meta:
                entry["meta"] = meta
            poses.append(entry)

        with open(output_path, "w") as f:
            json.dump(poses, f, indent=2)

        logger.info(f"Saved {len(poses)} pose pairs to {output_path}")

        from collections import Counter
        if campaign == "A" and constriction in ("rooms", "enriched"):
            strata = Counter(p.get("stratum", "?") for p in poses)
            logger.info(f"Crossing: {n_crossed}/{n_poses} "
                        f"({100*n_crossed/max(n_poses,1):.0f}%). Strata: {dict(strata)}")
            n_fb = strata.get("fallback", 0)
            if n_fb > 0.05 * n_poses:
                logger.warning(
                    f"{n_fb} pairs fell back to a plain probe pose. Check "
                    "room_split_clearance_m and the distance band.")
        elif campaign == "A" and constriction:
            logger.info(f"Constriction-crossing: {n_crossed}/{n_poses} "
                        f"({100*n_crossed/max(n_poses,1):.0f}%); the rest fell back "
                        "to ordinary probe poses.")

        distances = [p["distance"] for p in poses]
        logger.info(
            f"Distance stats: min={min(distances):.2f}m, max={max(distances):.2f}m, "
            f"mean={np.mean(distances):.2f}m, std={np.std(distances):.2f}m"
        )
        turns = [abs(p.get("meta", {}).get("initial_turn_deg", np.nan)) for p in poses]
        turns = [t for t in turns if not np.isnan(t)]
        if turns:
            logger.info(
                f"Initial turn: median={np.median(turns):.0f} deg, "
                f"{100*np.mean(np.array(turns) > 90):.0f}% require >90 deg")

        return poses

    @staticmethod
    def load_presampled(path: str) -> list[dict]:
        """Load pre-generated pose pairs from a JSON file."""
        import json

        with open(path) as f:
            poses = json.load(f)

        logger.info(f"Loaded {len(poses)} pre-sampled pose pairs from {path}")
        return poses

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

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

        overlay = np.stack([self.map_image] * 3, axis=-1)
        overlay[self.valid_mask, 1] = 200
        axes[2].imshow(overlay)

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

        for start, goal in pose_pairs:
            sr, sc = self._world_to_pixel(start["x"], start["y"])
            gr, gc = self._world_to_pixel(goal["x"], goal["y"])
            axes[2].plot(sc, sr, "go", markersize=6, alpha=0.7)
            axes[2].plot(gc, gr, "ro", markersize=6, alpha=0.7)
            axes[2].plot([sc, gc], [sr, gr], "b--", alpha=0.15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def visualize_rooms(
        self,
        output_path: str = "rooms.png",
        presampled_poses: list[dict] | None = None,
    ):
        """Save a room/throat visualization. Check this BEFORE generating a pool.

        Panel 1 shows the room segmentation: distinct colours mean the erosion
        split the map where you expect. If the whole floor is one colour,
        ``room_split_clearance_m`` is too low and no doorway pinches off. If
        every alcove is its own room, it is too high.

        Panel 2 marks the throat cells the doorway stratum anchors on, and
        panel 3 draws the pool coloured by stratum with each doorway pair's
        throat marked, so you can confirm the pairs really straddle doorways.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.where(self.room_labels > 0, self.room_labels, np.nan),
                       cmap="tab20", interpolation="nearest")
        axes[0].imshow(self.map_image, cmap="gray", alpha=0.25)
        axes[0].set_title(
            f"Rooms: {self.n_rooms} at {self.room_split_clearance_m} m erosion")

        axes[1].imshow(self.map_image, cmap="gray")
        if self.gateway_rc is not None and len(self.gateway_rc):
            axes[1].plot(self.gateway_rc[:, 1], self.gateway_rc[:, 0],
                         ".", color="magenta", markersize=1)
        axes[1].set_title(
            f"Doorway throats ({0 if self.gateway_rc is None else len(self.gateway_rc)} cells)")

        axes[2].imshow(self.map_image, cmap="gray")
        colours = {"doorway": "tab:red", "wall": "tab:orange",
                   "open": "tab:green", "fallback": "tab:gray"}
        for p in (presampled_poses or []):
            sr, sc = self._world_to_pixel(p["start"]["x"], p["start"]["y"])
            gr, gc = self._world_to_pixel(p["goal"]["x"], p["goal"]["y"])
            col = colours.get(p.get("stratum", "fallback"), "tab:blue")
            axes[2].plot([sc, gc], [sr, gr], "-", color=col, alpha=0.35, linewidth=0.8)
            axes[2].plot(sc, sr, "o", color=col, markersize=3, alpha=0.8)
            m = p.get("meta") or {}
            if "gateway_x" in m:
                tr, tc = self._world_to_pixel(m["gateway_x"], m["gateway_y"])
                axes[2].plot(tc, tr, "x", color="k", markersize=4, alpha=0.7)
        axes[2].set_title("Pool by stratum (x = throat)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()