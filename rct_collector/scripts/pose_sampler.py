#!/usr/bin/env python3
"""
Random Pose Sampler from OccupancyGrid Maps.

Samples random start and goal poses that are:
1. In free space (not inside obstacles or unknown regions)
2. Sufficiently far from obstacles (configurable clearance)
3. At a reasonable distance from each other (min/max distance)
4. Optionally within user-defined map bounds

The map is loaded from a standard ROS map_server YAML + PGM/PNG pair.

Decision-point probe sampling
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

import math
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
        # Probe CSV: worst tucked vertex (0.217, 0.242) -> 0.325 m;
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

    def apply_keepout_sweeps(self, sweeps, margin_m: float) -> None:
        """Forbid START and GOAL poses within margin_m of the area a moving
        obstacle sweeps. sweeps: [(sx, sy, bx, by, yaw, L, W)] = a box L x W at
        heading yaw sliding from (sx, sy) to (bx, by). Routes may still cross
        the sweep -- that is the point of the obstacle -- but the robot must
        never be spawned, or asked to stop, where the box can drive into it.
        Call after load_map(), before generating."""
        if not sweeps:
            return
        h, w = self.map_image.shape
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        X = cols * self.resolution + self.origin_x
        Y = (h - 1 - rows) * self.resolution + self.origin_y
        swept = np.zeros((h, w), dtype=bool)
        for sx, sy, bx, by, yaw, L, W in sweeps:
            n = max(2, int(math.hypot(bx - sx, by - sy) / (self.resolution / 2)) + 1)
            c, sn = math.cos(yaw), math.sin(yaw)
            for t in np.linspace(0.0, 1.0, n):
                cx, cy = sx + t * (bx - sx), sy + t * (by - sy)
                dx, dy = X - cx, Y - cy
                u, v = c * dx + sn * dy, -sn * dx + c * dy
                swept |= (np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)
        near = distance_transform_edt(~swept) * self.resolution <= margin_m
        before = int(self.valid_mask.sum())
        self.keepout_mask = near
        self.valid_mask &= ~near
        self.goal_mask &= ~near
        self.valid_indices = np.argwhere(self.valid_mask)
        self.goal_indices = np.argwhere(self.goal_mask)
        logger.info(f"Keep-out around {len(sweeps)} moving-obstacle sweep(s) "
                    f"(margin {margin_m:.2f} m): {before - int(self.valid_mask.sum())} "
                    f"start cells removed")

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
        Sample a feasible start pose and a local goal pose for decision probes.

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
            # Region-targeted pool generation restricts where the start may be.
            _sm = getattr(self, "_start_mask", None)
            if _sm is not None and not _sm[s_row, s_col]:
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
    # Route stratum: pairs separated along the ROUTE, not a straight line
    # ------------------------------------------------------------------ #

    # Edge cost per metre follows Nav2's inflation layer + NavFn: a cell at
    # clearance d < INFLATION costs 253*exp(-SCALING*(d - INSCRIBED)) on top
    # of NavFn's neutral cost 50, so the multiplier is 1 + cost/50; beyond the
    # inflation radius it is 1. Values are the pmb2 nav_public_sim defaults
    # (inflation 0.55, scaling 3.0, robot_radius 0.275); the treatment moves
    # inflation over 0.15-0.60, so routes are the BASELINE planner's. Being no
    # more wall-averse than Nav2 matters: an earlier proxy charged up to
    # 1.05 m and routed around every 1.2 m corridor Nav2 would happily take.
    ROUTE_INFLATION_M = 0.55
    ROUTE_COST_SCALING = 3.0
    ROUTE_INSCRIBED_M = 0.275
    NAVFN_NEUTRAL_COST = 50.0

    # A corridor is a passage too narrow to hold a pose with the default
    # carry clearance (0.90 m), i.e. local free width < 2 x 0.90 m, but wide
    # enough for a tucked base. Width is read off the ridge of the clearance
    # field: twice the largest clearance within half the max width.
    CORRIDOR_MAX_WIDTH_M = 1.80
    CORRIDOR_TAG_M = 1.5          # route metres in a corridor for the tag

    def _build_corridor_mask(self):
        from scipy.ndimage import maximum_filter
        k = max(3, int(round(self.CORRIDOR_MAX_WIDTH_M / self.resolution)) | 1)
        ridge = maximum_filter(np.where(self.nav_mask, self.dist_field, 0.0), size=k)
        self.passage_width = 2.0 * ridge
        self.corridor_mask = self.nav_mask & (self.passage_width < self.CORRIDOR_MAX_WIDTH_M)
        logger.info(
            f"Corridors: {self.corridor_mask.sum() * self.resolution ** 2:.1f} m^2 of "
            f"passage narrower than {self.CORRIDOR_MAX_WIDTH_M} m")

    def _build_route_graph(self):
        """8-connected graph over cells where a tucked base fits."""
        from scipy.sparse import csr_matrix
        if getattr(self, "corridor_mask", None) is None:
            self._build_corridor_mask()
        nav = self.nav_mask
        H, W = nav.shape
        rr, cc = np.nonzero(nav)
        idx = -np.ones((H, W), dtype=np.int64)
        idx[rr, cc] = np.arange(len(rr))
        d = self.dist_field
        infl = 253.0 * np.exp(-self.ROUTE_COST_SCALING
                              * np.maximum(d - self.ROUTE_INSCRIBED_M, 0.0))
        shape = 1.0 + np.where(d < self.ROUTE_INFLATION_M, infl, 0.0) / self.NAVFN_NEUTRAL_COST
        rows, cols, wts = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r2, c2 = rr + dr, cc + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            r1, c1, r2, c2 = rr[ok], cc[ok], r2[ok], c2[ok]
            ok = nav[r2, c2]
            r1, c1, r2, c2 = r1[ok], c1[ok], r2[ok], c2[ok]
            step = self.resolution * (np.sqrt(2.0) if dr and dc else 1.0)
            w = step * 0.5 * (shape[r1, c1] + shape[r2, c2])
            a, b = idx[r1, c1], idx[r2, c2]
            rows += [a, b]; cols += [b, a]; wts += [w, w]
        n = len(rr)
        self._route_graph = csr_matrix(
            (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n))
        self._route_rc = np.stack([rr, cc], axis=1)
        self._route_idx = idx
        self._route_clr = self.dist_field[rr, cc]
        self._route_valid = self.valid_mask[rr, cc]
        self._route_corr = self.corridor_mask[rr, cc]
        self._route_gate = np.zeros(n, dtype=bool)
        if self.gateway_rc is not None and len(self.gateway_rc):
            g = idx[self.gateway_rc[:, 0], self.gateway_rc[:, 1]]
            self._route_gate[g[g >= 0]] = True

    def _route_tree(self, s):
        """Cheapest-route tree from node ``s``.

        Returns (pred, length_m, min_clearance_m, crosses_throat, corridor_m),
        each per node, for the route to that node. Path sums are taken by pointer
        jumping over the predecessor tree, so the whole map costs ~12 vector
        passes instead of a Python walk per goal.
        """
        from scipy.sparse.csgraph import dijkstra
        cost, pred = dijkstra(self._route_graph, indices=s,
                              return_predecessors=True)
        n = len(cost)
        reached = np.isfinite(cost)
        anc = np.where(pred >= 0, pred, np.arange(n))
        rc = self._route_rc
        d = rc - rc[anc]
        val = self.resolution * np.hypot(d[:, 0], d[:, 1])
        val[~reached] = np.inf
        cor = np.where(self._route_corr, val, 0.0)
        cor[~reached] = 0.0
        mn = self._route_clr.copy()
        gt = self._route_gate.copy()
        a = anc.copy()
        for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
            val = val + val[a]
            cor = cor + cor[a]
            mn = np.minimum(mn, mn[a])
            gt = gt | gt[a]
            a = a[a]
        return pred, val, mn, gt, cor

    def sample_route_probe_pose(
        self,
        start_cells: Optional[np.ndarray] = None,
        d_min_m: float = 9.0,
        d_max_m: float = 10.9,
        goal_weight_fn=None,
        gap_band: Optional[tuple] = None,
        min_euclid_frac: float = 0.5,
        initial_turn_deg: tuple = (0.0, 180.0),
        goal_yaw_offset_deg: float = 180.0,
        max_attempts: int = 6,
        goal_node_mask: Optional[np.ndarray] = None,
        min_corridor_m: Optional[float] = None,
        corridor_after_m: float = 0.0,
        **_ignored,
    ):
        """Pair whose cheapest ROUTE is ``d_min_m``..``d_max_m`` long.

        The doorway / wall / open strata all need a straight, clear 9-11 m
        line from the start, which small rooms cannot hold: on the campaign-2
        map 19 of 50 start regions, including the north and south-east rooms,
        never produce one. Here the goal is anywhere the route reaches at the
        right length, so the start can be any valid cell and the route bends
        through doorways as Nav2's would.

        ``start_cells``      (N,2) rows/cols to draw the start from (default:
                             all valid cells).
        ``goal_weight_fn``   f(rows, cols) -> weights over candidate goals,
                             used to steer goals into under-filled regions.
                             All-zero weights fall back to uniform, and
                             meta['goal_weight_zero'] says so.
        ``gap_band``         keep only goals whose route narrows to a free
                             width (2 x clearance) in [lo, hi).
        ``min_euclid_frac``  goal at least this fraction of ``d_min_m`` away in
                             a straight line, so a route doubling back round
                             one wall does not count as a 9 m probe.
        ``goal_node_mask``   per route-graph node, where a goal may sit
                             (default: valid cells, i.e. obstacle_clearance_m).
        ``min_corridor_m``   keep only routes with at least this many metres
                             inside ``corridor_mask`` AFTER the first
                             ``corridor_after_m`` metres, i.e. after the
                             earliest point the treatment can land.
        """
        if getattr(self, "_route_graph", None) is None:
            self._build_route_graph()
        cells = self.valid_indices if start_cells is None else start_cells
        if len(cells) == 0:
            return None
        rc = self._route_rc
        for _ in range(max_attempts):
            sr, sc = cells[int(self.rng.integers(len(cells)))]
            s = int(self._route_idx[sr, sc])
            if s < 0:
                continue
            pred, length, mn, gt, cor = self._route_tree(s)
            eu = self.resolution * np.hypot(rc[:, 0] - sr, rc[:, 1] - sc)
            gmask = self._route_valid if goal_node_mask is None else goal_node_mask
            cand = (gmask & (length >= d_min_m) & (length <= d_max_m)
                    & (eu >= min_euclid_frac * d_min_m))
            if min_corridor_m is not None:
                cand &= cor >= min_corridor_m
            if gap_band is not None:
                gap = 2.0 * mn
                cand &= (gap >= gap_band[0]) & (gap < gap_band[1])
            ci = np.nonzero(cand)[0]
            if len(ci) == 0:
                continue
            w = None
            zero = False
            if goal_weight_fn is not None:
                w = np.asarray(goal_weight_fn(rc[ci, 0], rc[ci, 1]), dtype=float)
                if w.sum() <= 0:
                    w, zero = None, True
            p = None if w is None else w / w.sum()
            picked = None
            for _pick in range(15 if min_corridor_m is not None else 1):
                g = int(ci[self.rng.choice(len(ci), p=p)])
                path = [g]
                while path[-1] != s:
                    path.append(int(pred[path[-1]]))
                path = path[::-1]
                pxy = np.array([self._pixel_to_world(int(r), int(c))
                                for r, c in rc[path]])
                seg = np.hypot(*np.diff(pxy, axis=0).T)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                late_corr = float(seg[(cum[1:] > corridor_after_m)
                                      & self._route_corr[path[1:]]].sum())
                if min_corridor_m is None or late_corr >= min_corridor_m:
                    picked = g
                    break
            if picked is None:
                continue
            t = np.arange(0.0, cum[-1], 0.25)
            route_xy = np.stack([np.interp(t, cum, pxy[:, 0]),
                                 np.interp(t, cum, pxy[:, 1])], axis=1)
            k1 = min(len(route_xy) - 1, 4)          # ~1 m ahead
            k0 = max(0, len(route_xy) - 5)          # ~1 m before the goal
            bearing = float(np.arctan2(route_xy[k1, 1] - pxy[0, 1],
                                       route_xy[k1, 0] - pxy[0, 0]))
            final = float(np.arctan2(pxy[-1, 1] - route_xy[k0, 1],
                                     pxy[-1, 0] - route_xy[k0, 0]))
            turn = self._draw_turn(initial_turn_deg)
            start_yaw = float(np.mod(bearing + turn + np.pi, 2 * np.pi) - np.pi)
            gy_off = np.radians(goal_yaw_offset_deg)
            goal_yaw = float(np.mod(
                final + self.rng.uniform(-gy_off, gy_off) + np.pi, 2 * np.pi) - np.pi)

            gr, gc = rc[g]
            room_s = int(self.room_labels[sr, sc])
            room_g = int(self.room_labels[gr, gc])
            crossed = bool(gt[g]) or (room_s > 0 and room_g > 0 and room_s != room_g)
            sx, sy = pxy[0]
            gx, gy = pxy[-1]
            meta = {
                "route_length_m": round(float(length[g]), 3),
                "route_min_clearance_m": round(float(mn[g]), 3),
                "route_min_gap_m": round(2.0 * float(mn[g]), 3),
                "route_crosses_throat": bool(gt[g]),
                "route_corridor_m": round(float(cor[g]), 2),
                "route_corridor_after_apply_m": round(late_corr, 2),
                "room_start": room_s,
                "room_goal": room_g,
                "initial_turn_deg": round(float(np.degrees(turn)), 1),
                "_route_xy": route_xy,
            }
            if zero:
                meta["goal_weight_zero"] = True
            return (
                {"x": float(sx), "y": float(sy), "yaw": start_yaw},
                {"x": float(gx), "y": float(gy), "yaw": goal_yaw},
                crossed, "route", meta,
            )
        return None

    # ------------------------------------------------------------------ #
    # Pool generation
    # ------------------------------------------------------------------ #

    # Geometry of the difficulty labels. carry_radius_m (0.847) is the
    # CIRCUMSCRIBED radius of the carry polygon, so 2x it is the width the arm
    # needs to rotate freely; CARRY_WIDTH_M is its lateral extent going straight.
    CARRY_WIDTH_M = 0.97
    NARROW_GAP_M = 1.20          # carry must squeeze or retract below this
    # pal_office is open: obstacle fraction within 1 m of a route peaks at 0.09
    # and its median is 0.009, so "dense" is the top few percent of THIS map,
    # not an absolute occupancy. Recalibrate if the world changes.
    DENSE_FRAC = 0.05
    DENSE_BAND_M = 1.0
    DYNAMIC_NEAR_M = 1.5

    @staticmethod
    def _as_segments(dynamic_xy):
        """None | (x, y) | (x1, y1, x2, y2) | list of those -> list of 4-tuples."""
        if dynamic_xy is None:
            return []
        if len(dynamic_xy) and not hasattr(dynamic_xy[0], "__len__"):
            dynamic_xy = [dynamic_xy]
        out = []
        for d in dynamic_xy:
            d = [float(v) for v in d]
            out.append(tuple(d) if len(d) == 4 else (d[0], d[1], d[0], d[1]))
        return out

    def classify_difficulty(self, start, goal, meta=None, dynamic_xy=None):
        """Label a pose pair easy/difficult and say WHY.

        Difficulty is what makes the arm state matter, so each tag names a
        mechanism the tuner has to react to rather than a generic score:

          corner    the robot must turn more than 90 deg before making progress
          doorway   the straight route crosses a room-to-room throat
          narrow    the tightest passage is under NARROW_GAP_M but a tucked
                    base still fits, so the carry envelope must squeeze
          detour    the straight line is blocked outright, so the real plan has
                    to go around: a different problem from squeezing through
          dense     obstacles crowd the corridor on both sides
          dynamic   the route passes close to where the moving obstacle blocks

        Measured on the straight start->goal segment. That segment is NOT the
        planned path: where it clips a wall the gap reads ~0 and the planner
        will detour instead, which is why 'detour' and 'narrow' are separate
        tags rather than one threshold. Treat these as route-difficulty priors
        to stratify on, not as measurements of the executed path.
        Returns a dict; 'class' is easy when no tag fires.
        """
        import numpy as _np
        meta = meta or {}
        x0, y0, x1, y1 = start["x"], start["y"], goal["x"], goal["y"]
        n = max(2, int(_np.hypot(x1 - x0, y1 - y0) / self.resolution))
        xs = _np.linspace(x0, x1, n); ys = _np.linspace(y0, y1, n)

        # Straight-segment gap, kept for the 'detour' tag in every case.
        _g = [2.0 * float(self.dist_field[r, c]) for r, c in
              (self._world_to_pixel(x, y) for x, y in zip(xs, ys))
              if self._in_grid(r, c)]
        straight_gap = float(min(_g)) if _g else 0.0

        # Route-stratum pairs carry their route: measure on it instead, since
        # the straight segment of a bent route runs through walls.
        route_xy = meta.get("_route_xy")
        on_route = route_xy is not None and len(route_xy) >= 2
        if on_route:
            xs, ys = _np.asarray(route_xy)[:, 0], _np.asarray(route_xy)[:, 1]

        gaps, occ_hits, occ_tot = [], 0, 0
        band = int(round(self.DENSE_BAND_M / self.resolution))
        for x, y in zip(xs, ys):
            r, c = self._world_to_pixel(x, y)
            if not self._in_grid(r, c):
                continue
            gaps.append(2.0 * float(self.dist_field[r, c]))
            r0, r1 = max(0, r - band), min(self.free_mask.shape[0], r + band + 1)
            c0, c1 = max(0, c - band), min(self.free_mask.shape[1], c + band + 1)
            win = self.free_mask[r0:r1, c0:c1]
            occ_hits += int((~win).sum()); occ_tot += int(win.size)

        min_gap = float(min(gaps)) if gaps else 0.0
        dens = (occ_hits / occ_tot) if occ_tot else 0.0

        # Moving obstacles, each a blocking segment (x1, y1, x2, y2); a point
        # (x, y) is a zero-length segment. Distance is from the route to the
        # nearest segment, and 'dynamic_sites' says which ones are near.
        d_dyn = float("inf")
        near_sites = []
        for k, seg in enumerate(self._as_segments(dynamic_xy)):
            ax, ay, bx, by = seg
            ex, ey = bx - ax, by - ay
            L2 = ex * ex + ey * ey
            t = (_np.clip(((xs - ax) * ex + (ys - ay) * ey) / L2, 0.0, 1.0)
                 if L2 > 0 else _np.zeros_like(xs))
            d = float(_np.min(_np.hypot(xs - (ax + t * ex), ys - (ay + t * ey))))
            if d <= self.DYNAMIC_NEAR_M:
                near_sites.append(k)
            d_dyn = min(d_dyn, d)

        # Metres of the measured line inside a corridor (passage < 1.8 m).
        if getattr(self, "corridor_mask", None) is None:
            self._build_corridor_mask()
        _pts = [self._world_to_pixel(x, y) for x, y in zip(xs, ys)]
        _in = [self._in_grid(r, c) and bool(self.corridor_mask[r, c]) for r, c in _pts]
        _len = float(_np.hypot(_np.diff(xs), _np.diff(ys)).sum())
        corr_m = _len * (sum(_in) / max(len(_in), 1))

        turn = float(meta.get("initial_turn_deg", 0.0) or 0.0)
        tags = []
        if abs(turn) >= 90.0:
            tags.append("corner")
        if meta.get("stratum") == "doorway" or meta.get("crossed"):
            tags.append("doorway")
        nav_diam = 2.0 * self.nav_radius_m
        if on_route:
            # Route known: detour and narrow are separate facts about it.
            if straight_gap < nav_diam:
                tags.append("detour")
            if min_gap < self.NARROW_GAP_M:
                tags.append("narrow")
        elif min_gap < nav_diam:
            tags.append("detour")        # straight line blocked; plan must go around
        elif min_gap < self.NARROW_GAP_M:
            tags.append("narrow")        # tucked fits, carry must squeeze
        if corr_m >= self.CORRIDOR_TAG_M:
            tags.append("corridor")      # sustained squeeze, not a doorway
        if dens > self.DENSE_FRAC:
            tags.append("dense")
        if d_dyn <= self.DYNAMIC_NEAR_M:
            tags.append("dynamic")

        return {
            "class": "difficult" if tags else "easy",
            "tags": tags,
            "n_tags": len(tags),
            "min_gap_m": round(min_gap, 3),
            "straight_blocked": bool(straight_gap < 2.0 * self.nav_radius_m),
            "measured_on": "route" if on_route else "straight",
            "corridor_m": round(corr_m, 2),
            "obstacle_frac": round(dens, 4),
            "initial_turn_deg": round(abs(turn), 1),
            "dist_to_dynamic_m": (None if d_dyn == float("inf") else round(d_dyn, 3)),
            "dynamic_sites": near_sites,
        }

    def _draw_one(self, constriction, forward_distance_m, ck):
        """One (start, goal, crossed, stratum, meta) draw for the chosen mode.

        Factored out of generate_and_save so the pool generator can redraw a
        pair that a spatial quota rejects, without duplicating the branch chain.
        """
        if constriction == "rooms":
            start, goal, crossed, stratum, meta = self.sample_stratified_probe_pose(
                d_min_m=forward_distance_m,
                d_max_m=forward_distance_m + 1.9, **ck)
            return start, goal, crossed, stratum, meta
        if constriction == "enriched":
            start, goal, crossed, stratum = self.sample_probe_pose_enriched(
                forward_distance_m=forward_distance_m, **ck)
            return start, goal, crossed, stratum, None
        if constriction:
            start, goal, crossed = self.sample_constriction_probe_pose(
                forward_distance_m=forward_distance_m, **ck)
            return start, goal, crossed, None, None
        start, goal = self.sample_probe_pose(forward_distance_m=forward_distance_m)
        return start, goal, None, None, None

    def generate_and_save(
        self,
        n_poses: int,
        output_path: str = "presampled_poses.json",
        forward_distance_m: float = 3.5,
        constriction: bool = False,
        constriction_kwargs: Optional[dict] = None,
        spatial_balance: bool = False,
        balance_cell_m: float = 3.0,
        balance_oversample: int = 40,
        label_difficulty: bool = False,
        dynamic_obstacle_xy: Optional[tuple] = None,
        tight_fraction: float = 0.0,
        tight_gap_band: tuple = (0.65, 1.20),
        min_region_area_m2: float = 1.0,
        strata_tries_per_region: int = 3,
        corridor_fraction: float = 0.0,
        corridor_pose_clearance_m: float = 0.50,
        corridor_min_m: float = 2.0,
        corridor_after_m: float = 2.0,
    ) -> list[dict]:
        """
        Pre-generate n pose pairs and save them to a JSON file.

        Each entry has: {"start": {"x","y","yaw"}, "goal": {"x","y","yaw"},
        "distance": float, "crossed": bool, "stratum": str, "meta": {...}}.
        With ``spatial_balance`` it also has "share" ("tight" / "spread") and
        "start_region".

        ``constriction`` selects the probe sampler:
          "rooms"     v2 room-topology strata (recommended)
          "enriched"  v1 three-stratum sampler
          True        v1 single-constriction sampler
          False       plain forward probe

        ``spatial_balance`` spreads the pool over the map by TARGETING regions
        rather than rejecting draws. Valid start space is cut into
        ``balance_cell_m`` squares; squares with at least
        ``min_region_area_m2`` of valid area get equal start quotas, and the
        same squares get equal goal quotas. Each pair is drawn for a named
        start region:

          1. up to ``strata_tries_per_region`` stratified draws with the start
             restricted to that region, kept only if the goal's region still
             has quota (and, for the tight share, the straight gap is in band);
          2. otherwise the route stratum (``sample_route_probe_pose``), which
             picks a goal at the right ROUTE length, weighted toward goal
             regions with quota left.

        The previous version rejected draws against the quota and, after
        ``balance_oversample`` rejections, kept the last draw anyway. Because
        the straight-line strata cannot place a 9-11 m pair in small rooms,
        that fallback fired constantly and the quota never bound: 21% of goals
        landed in one 3 m cell. Targeting removes the fallback; a region that
        cannot produce a pair at all is retired and its quota moved to the
        live regions, and the log reports it.

        ``tight_fraction`` reserves that share of the pool for pairs whose
        route narrows to ``tight_gap_band`` metres of free width. That band is
        where the arm state decides the outcome: a tucked base (0.65 m) fits,
        a carry envelope (0.97 m) has to squeeze or give up. With balancing,
        the tight share has its own start quotas over the regions that CAN
        produce a tight route, so it spreads over every tight place on the map
        instead of piling into the few nearest the sampler's favourite
        throats.

        ``corridor_fraction`` reserves that share for pairs whose route runs
        at least ``corridor_min_m`` through a corridor (passage narrower than
        CORRIDOR_MAX_WIDTH_M) after the first ``corridor_after_m`` metres, the
        earliest the treatment can land. A doorway is a short squeeze; a
        corridor holds the carry envelope tight for metres and allows no turn,
        so it is a different mechanism and needs its own data. Corridors cannot
        hold a pose with ``obstacle_clearance_m`` (0.90 m needs 1.8 m of
        width), so for this share start and goal may also sit in a corridor
        with ``corridor_pose_clearance_m``. That is safe because every probe
        spawns under the baseline (tucked) arm; the carry footprint only
        appears after the trigger. The corridor share is exempt from the start
        quotas (corridors are few) but its goals still count against them.

        For "rooms", ``forward_distance_m`` is the lower end of the pair
        separation and the upper end is +1.9 m. The orchestrator rejects any
        plan shorter than ``guard + 0.4`` with ``guard = v_max * H + 0.5``.
        """
        import json

        ck = dict(constriction_kwargs or {})
        lo_gap, hi_gap = tight_gap_band
        n_tight_target = int(round(tight_fraction * n_poses))
        n_corr_target = int(round(corridor_fraction * n_poses))
        if n_tight_target + n_corr_target > n_poses:
            raise ValueError("tight_fraction + corridor_fraction exceeds 1")
        d_min = forward_distance_m
        d_max = forward_distance_m + 1.9

        def _straight_min_gap(start, goal):
            """Narrowest free width on the straight start->goal segment."""
            n = max(2, int(np.hypot(goal["x"] - start["x"],
                                    goal["y"] - start["y"]) / self.resolution))
            g = []
            for x, y in zip(np.linspace(start["x"], goal["x"], n),
                            np.linspace(start["y"], goal["y"], n)):
                r, c = self._world_to_pixel(x, y)
                if self._in_grid(r, c):
                    g.append(2.0 * float(self.dist_field[r, c]))
            return min(g) if g else 0.0

        def _gap_of(start, goal, meta):
            if meta and "route_min_gap_m" in meta:
                return meta["route_min_gap_m"]
            return _straight_min_gap(start, goal)

        _corr_ctx = {}

        def _draw_corridor(goal_weight_fn=None, mode="end"):
            """One corridor-share pair, or None.

            mode  end      route ends inside the corridor (robot enters it late,
                           likely after the treatment landed)
                  through  start and goal outside: the robot traverses it and
                           has to get out the other side
                  start    spawn inside the corridor, so the arm can switch to
                           carry while the robot is still in it
            """
            if not _corr_ctx:
                if getattr(self, "_route_graph", None) is None:
                    self._build_route_graph()
                pose_ok = (self.free_mask & (self.dist_field >= corridor_pose_clearance_m)
                           & (self.valid_mask | self.corridor_mask))
                ko = getattr(self, "keepout_mask", None)
                if ko is not None:
                    pose_ok &= ~ko
                near = distance_transform_edt(~self.corridor_mask) * self.resolution <= d_max
                rc = self._route_rc
                in_corr = self.corridor_mask[rc[:, 0], rc[:, 1]]
                ok_n = pose_ok[rc[:, 0], rc[:, 1]]
                _corr_ctx.update({
                    "end": (np.argwhere(pose_ok & near), ok_n & in_corr),
                    "through": (np.argwhere(pose_ok & near & ~self.corridor_mask),
                                ok_n & ~in_corr),
                    "start": (np.argwhere(pose_ok & self.corridor_mask), ok_n),
                })
            starts, gmask = _corr_ctx[mode]
            out = self.sample_route_probe_pose(
                start_cells=starts, d_min_m=d_min, d_max_m=d_max,
                goal_weight_fn=goal_weight_fn, goal_node_mask=gmask,
                min_corridor_m=corridor_min_m, corridor_after_m=corridor_after_m,
                max_attempts=12,
                **{k: v for k, v in ck.items() if k in ("initial_turn_deg",)})
            if out is None:
                return None
            out[4]["corridor_mode"] = mode
            return out[:3] + ("corridor",) + out[4:]

        _corr_modes = ["end", "through", "start"]
        _corr_dead = set()
        _corr_i = [0]

        def _next_corridor(goal_weight_fn=None):
            """Cycle the corridor modes; retire a mode that cannot produce."""
            for _ in range(len(_corr_modes)):
                live = [m for m in _corr_modes if m not in _corr_dead]
                if not live:
                    return None
                mode = live[_corr_i[0] % len(live)]
                _corr_i[0] += 1
                for _try in range(4):
                    out = _draw_corridor(goal_weight_fn, mode)
                    if out is not None:
                        return out
                _corr_dead.add(mode)
                logger.warning(f"Corridor mode '{mode}' produced nothing; retired")
            return None

        draws = []          # (start, goal, crossed, stratum, meta, share, region)
        region_of = None
        stats = {"strata": 0, "route": 0, "goal_overflow": 0}
        retired = {"tight": [], "spread": []}
        n_corr_failed = 0

        if spatial_balance:
            if self.nav_mask is None:
                raise RuntimeError("spatial_balance needs load_map() first")
            step = max(1, int(round(balance_cell_m / self.resolution)))

            def region_of(x, y):
                r, c = self._world_to_pixel(x, y)
                return (int(r // step), int(c // step))

            V = self.valid_indices
            keys = (V[:, 0] // step) * 100000 + V[:, 1] // step
            uniq, counts = np.unique(keys, return_counts=True)
            min_cells = int(np.ceil(min_region_area_m2 / self.resolution ** 2))
            region_cells = {}
            for k, n in zip(uniq, counts):
                if n >= min_cells:
                    region_cells[(int(k // 100000), int(k % 100000))] = V[keys == k]
            regions = sorted(region_cells)
            R = len(regions)
            if R == 0:
                raise RuntimeError("no region has enough valid area")

            def _split(n):
                """n spread as evenly as possible over the regions."""
                q = {k: n // R for k in regions}
                for k in self.rng.permutation(R)[: n - (n // R) * R]:
                    q[regions[int(k)]] += 1
                return q

            goal_left = _split(n_poses)
            start_left = {"tight": _split(n_tight_target),
                          "spread": _split(n_poses - n_tight_target - n_corr_target)}
            logger.info(
                f"Spatial balancing (targeted): {R} regions of {balance_cell_m} m "
                f"with >= {min_region_area_m2} m^2 valid start area; "
                f"{n_tight_target} tight + {n_corr_target} corridor + "
                f"{n_poses - n_tight_target - n_corr_target} spread pairs")

            # Region shapes for the goal-weight lookup.
            def _goal_weights(rows, cols):
                return np.array([max(goal_left.get((int(r // step), int(c // step)), 0), 0)
                                 for r, c in zip(rows, cols)], dtype=float)

            _sw = ck.get("strata_weights") or {"doorway": 0.40, "wall": 0.25, "open": 0.35}
            open_share = float(_sw.get("open", 0.0)) / max(sum(_sw.values()), 1e-9)
            open_gap = 2.0 * (self.carry_radius_m + 0.15)
            door_share = float(_sw.get("doorway", 0.0)) / max(sum(_sw.values()), 1e-9)
            no_door = set()

            # Draw stratified first, sized to the target region.
            ck_t = dict(ck)
            ck_t.setdefault("max_attempts", 200)

            def _try_region(region, share):
                cells = region_cells[region]
                tight = share == "tight"
                if constriction:
                    full = self.valid_indices
                    try:
                        self.valid_indices = cells
                        # the doorway stratum anchors on a throat, not on
                        # valid_indices, so it needs the region as a mask
                        sm = np.zeros(self.valid_mask.shape, dtype=bool)
                        sm[cells[:, 0], cells[:, 1]] = True
                        self._start_mask = sm
                        # Doorway first, at its stratum weight and with its
                        # own budget: inside the generic draw it shares a
                        # 200-attempt cap that a region-restricted throat
                        # draw almost never meets.
                        if (constriction == "rooms" and region not in no_door
                                and self.rng.random() < door_share):
                            out = self.sample_doorway_probe_pose(
                                d_min_m=d_min, d_max_m=d_max, max_attempts=4000,
                                **{k: v for k, v in ck.items()
                                   if k in ("initial_turn_deg",)})
                            if out is None:
                                no_door.add(region)
                            else:
                                s, g, cr, st, me = out
                                if (goal_left.get(region_of(g["x"], g["y"]), 0) > 0
                                        and (not tight or lo_gap <= _straight_min_gap(s, g) < hi_gap)):
                                    stats["strata"] += 1
                                    return out
                        for _ in range(strata_tries_per_region):
                            try:
                                s, g, cr, st, me = self._draw_one(
                                    constriction, forward_distance_m, ck_t)
                            except RuntimeError:
                                # the plain-probe fallback raises when the
                                # region has no straight pair at all
                                break
                            if st in ("fallback", None) and constriction == "rooms":
                                continue
                            if region_of(s["x"], s["y"]) != region:
                                continue
                            if goal_left.get(region_of(g["x"], g["y"]), 0) <= 0:
                                continue
                            if tight and not (lo_gap <= _straight_min_gap(s, g) < hi_gap):
                                continue
                            stats["strata"] += 1
                            return s, g, cr, st, me
                    finally:
                        self.valid_indices = full
                        self._start_mask = None
                # Route fallback. For the spread share, keep the open control
                # at its stratum weight: the whole route clears the carry
                # envelope plus margin, as sample_open_probe_pose requires.
                rk = {k: v for k, v in ck.items() if k in ("initial_turn_deg",)}
                bands = []
                if tight:
                    bands = [(lo_gap, hi_gap)]
                else:
                    if self.rng.random() < open_share:
                        bands.append((open_gap, np.inf))
                    bands.append(None)
                out = None
                for band in bands:
                    out = self.sample_route_probe_pose(
                        start_cells=cells, d_min_m=d_min, d_max_m=d_max,
                        goal_weight_fn=_goal_weights, gap_band=band, **rk)
                    if out is not None:
                        if band is not None and not tight:
                            out = out[:3] + ("route_open",) + out[4:]
                        break
                if out is None:
                    return None
                stats["route"] += 1
                if out[4].get("goal_weight_zero"):
                    stats["goal_overflow"] += 1
                return out

            # Job list: one job per quota unit, shuffled so the two shares
            # interleave in the pool instead of sitting in blocks.
            jobs = [(share, k) for share in ("tight", "spread")
                    for k in regions for _ in range(start_left[share][k])]
            jobs += [("corridor", None)] * n_corr_target
            jobs = [jobs[int(i)] for i in self.rng.permutation(len(jobs))]
            dead = {"tight": set(), "spread": set()}
            assigned = {share: dict(start_left[share]) for share in start_left}
            while jobs:
                share, region = jobs.pop()
                if share == "corridor":
                    out = _next_corridor(_goal_weights)
                    if out is None:
                        n_corr_failed += 1
                        if n_corr_failed > 10 * max(n_corr_target, 1):
                            logger.warning("Corridor share starved; giving up on it")
                            jobs = [j for j in jobs if j[0] != "corridor"]
                        else:
                            jobs.insert(0, (share, region))
                        continue
                    s, g, cr, st, me = out
                    gr = region_of(g["x"], g["y"])
                    if gr in goal_left:
                        goal_left[gr] -= 1
                    draws.append((s, g, cr, st, me, share,
                                  region_of(s["x"], s["y"])))
                    continue
                out = None
                if region not in dead[share]:
                    out = _try_region(region, share)
                if out is None:
                    if region not in dead[share]:
                        dead[share].add(region)
                        retired[share].append(region)
                    alive = [k for k in regions if k not in dead[share]]
                    if not alive:
                        logger.warning(f"No region can produce a {share} pair; "
                                       f"pool will be short by one")
                        continue
                    # move the unit to the live region holding the fewest
                    new = min(alive, key=lambda k: (assigned[share][k], self.rng.random()))
                    assigned[share][new] += 1
                    jobs.insert(0, (share, new))
                    continue
                s, g, cr, st, me = out
                gr = region_of(g["x"], g["y"])
                if gr in goal_left:
                    goal_left[gr] -= 1
                draws.append((s, g, cr, st, me, share, region))
        else:
            n_tight = 0
            for _ in range(n_corr_target):
                out = None
                for _try in range(10):
                    out = _next_corridor()
                    if out is not None:
                        break
                if out is not None:
                    draws.append(tuple(out) + ("corridor", None))
            for i in range(n_poses - len(draws)):
                want_tight = n_tight < n_tight_target
                _tries = balance_oversample if want_tight else 1
                got_tight = False
                for _attempt in range(_tries):
                    s, g, cr, st, me = self._draw_one(constriction, forward_distance_m, ck)
                    if not want_tight:
                        break
                    if lo_gap <= _gap_of(s, g, me) < hi_gap:
                        got_tight = True
                        break
                n_tight += int(got_tight)
                draws.append((s, g, cr, st, me,
                              "tight" if got_tight else "spread", None))

        poses = []
        n_crossed = 0
        for i, (start, goal, crossed, stratum, meta, share, region) in enumerate(draws):
            if crossed is not None:
                n_crossed += int(crossed)
            dist = float(np.hypot(goal["x"] - start["x"], goal["y"] - start["y"]))
            entry = {"id": i, "start": start, "goal": goal,
                     "distance": round(dist, 3)}
            if crossed is not None:
                entry["crossed"] = bool(crossed)
            if stratum is not None:
                entry["stratum"] = stratum
            if tight_fraction > 0 or spatial_balance:
                entry["share"] = share
            if region is not None:
                entry["start_region"] = list(region)
            if label_difficulty:
                _m = dict(meta or {})
                _m.setdefault("stratum", stratum)
                _m.setdefault("crossed", crossed)
                entry["difficulty"] = self.classify_difficulty(
                    start, goal, _m, dynamic_xy=dynamic_obstacle_xy)
            if meta:
                entry["meta"] = {k: v for k, v in meta.items() if not k.startswith("_")}
            poses.append(entry)

        with open(output_path, "w") as f:
            json.dump(poses, f, indent=2)

        logger.info(f"Saved {len(poses)} pose pairs to {output_path}")

        if spatial_balance:
            from collections import Counter as _C
            sc = _C(region_of(p["start"]["x"], p["start"]["y"]) for p in poses)
            gc = _C(region_of(p["goal"]["x"], p["goal"]["y"]) for p in poses)
            top_s = sc.most_common(1)[0][1] / max(len(poses), 1)
            top_g = gc.most_common(1)[0][1] / max(len(poses), 1)
            logger.info(
                f"Spatial balance: starts over {len(sc)} regions "
                f"(largest {100*top_s:.1f}%), goals over {len(gc)} regions "
                f"(largest {100*top_g:.1f}%). Draws: {stats['strata']} stratified, "
                f"{stats['route']} route; {stats['goal_overflow']} goals had no "
                f"under-quota region in reach")
            for share in ("tight", "spread"):
                if retired[share]:
                    logger.info(f"Regions retired for the {share} share "
                                f"(cannot produce one): {len(retired[share])}")
        n_tight = sum(1 for p in poses if p.get("share") == "tight")
        from collections import Counter
        if constriction in ("rooms", "enriched"):
            strata = Counter(p.get("stratum", "?") for p in poses)
            logger.info(f"Crossing: {n_crossed}/{n_poses} "
                        f"({100*n_crossed/max(n_poses,1):.0f}%). Strata: {dict(strata)}")
            n_fb = strata.get("fallback", 0)
            if n_fb > 0.05 * n_poses:
                logger.warning(
                    f"{n_fb} pairs fell back to a plain probe pose. Check "
                    "room_split_clearance_m and the distance band.")
        elif constriction:
            logger.info(f"Constriction-crossing: {n_crossed}/{n_poses} "
                        f"({100*n_crossed/max(n_poses,1):.0f}%); the rest fell back "
                        "to ordinary probe poses.")

        if tight_fraction > 0:
            logger.info(f"Tight routes: {n_tight}/{n_tight_target} requested "
                        f"in band {tight_gap_band} m")

        if label_difficulty:
            from collections import Counter as _C2
            cls = _C2(p["difficulty"]["class"] for p in poses)
            tags = _C2(t for p in poses for t in p["difficulty"]["tags"])
            logger.info(f"Difficulty: {dict(cls)}  tags={dict(tags)}")

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