#!/usr/bin/env python3
"""
Nav2 Parameter Space for RCT (Paper 2: online causal tuner).

Defines the treatment variables (C_t) for the randomized controlled trial.
Each trial samples a random configuration from this space; randomization of
C_t independently of the environment is the core causal-identification
strategy (do(C=c) is valid because c is assigned by LHS, not by the scene).

============================================================================
FINALIZED CONFIGURATION SPACE — 6 parameters, organized by causal channel
============================================================================
Confirmed against the live TIAGo stack (nav2_mppi_controller, MPPI):

  Channel        | Variable                   | ROS param
  ---------------|----------------------------|------------------------------------------
  dynamics       | max translational velocity | controller_server FollowPath.vx_max
  dynamics       | max rotational velocity    | controller_server FollowPath.wz_max
  obstacle resp. | MPPI obstacle critic weight| controller_server FollowPath.CostCritic.cost_weight
  lookahead      | prediction horizon         | controller_server FollowPath.time_steps
  geometry       | inflation radius           | {local,global}_costmap inflation_layer.inflation_radius
  geometry       | arm / footprint state      | {local,global}_costmap footprint  (categorical)

Notes specific to THIS stack:
- The obstacle critic is `CostCritic` (not `ObstaclesCritic`); newer Nav2 MPPI
  reads costmap cost directly. It also exposes CostCritic.consider_footprint,
  which is what makes the arm/footprint variable couple into obstacle response.
- Prediction horizon in seconds = time_steps * model_dt. We vary `time_steps`
  and hold `model_dt` fixed (see HELD FIXED), so horizon scales linearly and
  stays interpretable.
- inflation_radius is treated as ONE causal knob applied to BOTH local and
  global costmaps (linked via extra_targets) so the geometry stays consistent.
  To make it local-only, drop the extra_target on that ParamDef.

============================================================================
HELD FIXED (NOT randomized) — by design / per supervisor + Paper 1
============================================================================
- Global planner: FIXED (not changed across trials).
- Controller plugin: FIXED to MPPI (nav2_mppi_controller). Type never changes.
- controller_server.controller_frequency: FIXED.
- FollowPath.model_dt, batch_size, iteration_count, temperature, gamma,
  motion_model, and the critic *list* (FollowPath.critics): FIXED — only the
  obstacle (CostCritic) weight is a treatment; all other critic weights stay
  at their tuned defaults.
- cost_scaling_factor: EXCLUDED (Paper 1 found no direct collision effect).
- min_vel_x / planner tolerance / controller_frequency: EXCLUDED (not in C_t).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamDef:
    """Definition of a single parameter to randomize."""

    node: str             # Logical label, used for CSV headers (e.g. "controller_server")
    name: str             # Parameter as addressed within the ROS node
    param_type: str       # "continuous" | "discrete" | "categorical"
    low: float = 0.0      # Min value (continuous/discrete)
    high: float = 1.0     # Max value (continuous/discrete)
    choices: list = field(default_factory=list)  # For categorical
    log_scale: bool = False                       # Sample in log space
    ros_node: str = ""    # Actual ROS node for `ros2 param set` (no leading slash).
                          # Defaults to `node`. Needed for nested costmap nodes.
    channel: str = ""     # Causal channel label (interpretability / paper bookkeeping)
    apply_via: str = "param_set"  # "param_set" | "footprint"
                          #   param_set : ros2 param set + read-back
                          #   footprint : categorical -> swap footprint polygon preset
    extra_targets: list = field(default_factory=list)
                          # [(ros_node, name), ...] that receive the SAME sampled value.
                          # Used to keep one causal knob applied to >1 ROS param
                          # (e.g. inflation radius on local AND global costmap).
    presets: dict = field(default_factory=dict)
                          # For categorical applied via footprint: label -> concrete value.

    def __post_init__(self):
        if not self.ros_node:
            self.ros_node = self.node


# ── Arm / footprint configurations ──────────────────────────────────────────
# Each level of the arm variable couples a costmap footprint polygon with a
# physical arm joint target. The costmap polygon is set via `ros2 param set`;
# the arm is moved per-episode by the orchestrator (see OrchestratorConfig.move_arm).
#
# Footprint polygons are in base_link (x forward, y left):
#   - "offer":     arm reaching forward  -> long +x footprint
#   - "carry_bag": bag held to the side  -> extended -y footprint
#
# joints: the 7 arm joint targets (radians), in ARM_JOINT_NAMES order.
#   *** PLACEHOLDER — CAPTURE ON YOUR ROBOT ***
#   Pose the arm into each position, then `ros2 topic echo /joint_states --once`
#   and copy arm_1_joint..arm_7_joint here. Until filled (left as None), the
#   orchestrator will set the footprint but SKIP the arm motion with a warning.
ARM_JOINT_NAMES = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
]


# joints: 7 arm joint targets (radians), captured live and confirmed reachable.
ARM_CONFIGS = {
    "tucked": {
        "footprint": (
            "[[0.30, 0.0], [0.21, 0.21], [0.0, 0.30], [-0.21, 0.21], "
            "[-0.30, 0.0], [-0.21, -0.21], [0.0, -0.30], [0.21, -0.21]]"
        ),
        "mode": "play_motion",
        "motion_name": "home",
        "joints": [0.50, -1.34, -0.48, 1.94, -1.49, 1.37, 0.0],
    },
    "carry": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641], [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138], [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
        "mode": "joint_trajectory",
        "joints": [0, 0.15, -0.5, 1.2, 0.0, 0.8, 0.0],
         
    },
}


class ParameterSpace:
    """
    Defines and samples from the Nav2 parameter space (the treatment C_t).

    sample()      -> one uniform-random config (online / per-trial draw)
    sample_lhs(n) -> n configs via Latin Hypercube Sampling (pre-generated,
                     reproducible, good coverage with few trials). LHS covers
                     the continuous + discrete dims jointly and stratifies the
                     categorical (arm) so it stays balanced across the batch.
    """

    # ── Finalized 6-parameter space ─────────────────────────────────────────
    DEFAULT_PARAMS = [
        # ── Dynamics channel (MPPI velocity caps) ───────────────────────────
        ParamDef(
            node="controller_server",
            name="FollowPath.vx_max",
            param_type="continuous",
            low=0.15, high=0.65,            # CONFIRM vs TIAGo hardware/config max
            channel="dynamics",
        ),
        ParamDef(
            node="controller_server",
            name="FollowPath.wz_max",
            param_type="continuous",
            low=0.4, high=1.8,              # MPPI default wz_max ~1.9
            channel="dynamics",
        ),

        # ── Obstacle-response channel (MPPI obstacle critic) ────────────────
        ParamDef(
            node="controller_server",
            name="FollowPath.CostCritic.cost_weight",
            param_type="continuous",
            low=1.0, high=12.0,             # default ~3.81; brackets it both ways
            channel="obstacle_response",
        ),

        # ── Lookahead channel (prediction horizon) ──────────────────────────
        ParamDef(
            node="controller_server",
            name="FollowPath.time_steps",
            param_type="discrete",
            low=30, high=90,                # horizon_sec = time_steps * model_dt (model_dt fixed)
            channel="lookahead",
        ),

        # ── Geometry channel: inflation radius (ONE knob, local+global) ─────
        ParamDef(
            node="local_costmap",
            ros_node="local_costmap/local_costmap",
            name="inflation_layer.inflation_radius",
            param_type="continuous",
            low=0.25, high=0.8,             # live value was 0.55; per-episode (cache rebuild)
            channel="geometry",
            extra_targets=[("global_costmap/global_costmap",
                            "inflation_layer.inflation_radius")],
        ),

        # ── Geometry channel: arm / footprint state (categorical) ───────────
        ParamDef(
            node="local_costmap",
            ros_node="local_costmap/local_costmap",
            name="footprint",
            param_type="categorical",
            choices=list(ARM_CONFIGS.keys()),
            channel="footprint",
            apply_via="footprint",
            presets={k: v["footprint"] for k, v in ARM_CONFIGS.items()},
            extra_targets=[("global_costmap/global_costmap", "footprint")],
        ),
    ]

    def __init__(self, params: list[ParamDef] | None = None, seed: int | None = None):
        self.params = params if params is not None else [
            ParamDef(**{**p.__dict__}) for p in self.DEFAULT_PARAMS
        ]
        self.rng = np.random.default_rng(seed)

    def add_param(self, param: ParamDef):
        self.params.append(param)

    def remove_param(self, node: str, name: str):
        self.params = [p for p in self.params if not (p.node == node and p.name == name)]

    # ── Sampling ────────────────────────────────────────────────────────────
    def sample(self) -> dict[str, dict[str, Any]]:
        """One uniform-random configuration, nested dict keyed by node label."""
        config: dict[str, dict[str, Any]] = {}
        for p in self.params:
            config.setdefault(p.node, {})[p.name] = self._sample_param(p)
        return config

    def sample_lhs(self, n_samples: int, optimization: str | None = "random-cd"
                   ) -> list[dict[str, dict[str, Any]]]:
        """
        Latin Hypercube Sampling over continuous + discrete dims, with the
        categorical (arm) stratified so it is ~balanced across the batch.
        Use this to pre-generate all configs for a reproducible run.

        `optimization` improves space-filling to avoid clustered / near-duplicate
        points (plain LHS only controls 1-D spacing, not joint spacing):
          - "random-cd" : minimize centered discrepancy (recommended)
          - "lloyd"     : Lloyd's relaxation toward maximin
          - None        : plain LHS
        Falls back to plain LHS if the installed scipy is too old to support it.
        """
        from scipy.stats import qmc

        grid_params = [p for p in self.params
                       if p.param_type in ("continuous", "discrete")]
        cat_params = [p for p in self.params if p.param_type == "categorical"]

        try:
            sampler = qmc.LatinHypercube(
                d=len(grid_params), seed=self.rng, optimization=optimization
            )
        except TypeError:
            # Old scipy without the `optimization` kwarg
            sampler = qmc.LatinHypercube(d=len(grid_params), seed=self.rng)
        unit = sampler.random(n=n_samples)

        # Pre-build balanced, shuffled assignments for each categorical dim.
        cat_assign: dict[int, list] = {}
        for ci, p in enumerate(cat_params):
            reps = int(np.ceil(n_samples / len(p.choices)))
            col = (p.choices * reps)[:n_samples]
            self.rng.shuffle(col)
            cat_assign[ci] = col

        configs = []
        for i in range(n_samples):
            config: dict[str, dict[str, Any]] = {}
            for gi, p in enumerate(grid_params):
                u = float(unit[i, gi])
                if p.param_type == "continuous":
                    if p.log_scale:
                        val = float(np.exp(np.log(p.low) + u * (np.log(p.high) - np.log(p.low))))
                    else:
                        val = float(p.low + u * (p.high - p.low))
                else:  # discrete: map unit interval onto integer bins inclusive
                    n_bins = int(p.high) - int(p.low) + 1
                    val = int(p.low) + min(int(u * n_bins), n_bins - 1)
                config.setdefault(p.node, {})[p.name] = val
            for ci, p in enumerate(cat_params):
                config.setdefault(p.node, {})[p.name] = cat_assign[ci][i]
            configs.append(config)
        return configs

    # ── Coverage / near-duplicate diagnostics ───────────────────────────────
    def sample_maximin(self, n_samples: int, oversample: int = 4,
                       optimization: str | None = None) -> list[dict]:
        """
        Space-filling design with a GUARANTEED large minimum separation.

        Draws an oversampled LHS pool, then greedily selects the most-separated
        points (farthest-point sampling), balanced across the categorical
        dimension(s). Eliminates near-duplicate configs. Marginals are slightly
        less uniform than plain LHS, but every parameter's range stays fully and
        densely covered. For the paper: "Latin Hypercube pool refined by maximin
        farthest-point selection to enforce a minimum inter-config separation."
        """
        from collections import defaultdict

        cat_params = [p for p in self.params if p.param_type == "categorical"]
        pool = self.sample_lhs(n_samples * oversample, optimization=optimization)
        X = self._to_unit_matrix(pool)

        # Group pool points by their joint categorical assignment (for balance).
        if cat_params:
            def key(c):
                return tuple(c[p.node][p.name] for p in cat_params)
            groups = defaultdict(list)
            for i, c in enumerate(pool):
                groups[key(c)].append(i)
        else:
            groups = {(): list(range(len(pool)))}

        keys = list(groups)
        per = [n_samples // len(keys)] * len(keys)
        for r in range(n_samples - sum(per)):
            per[r] += 1

        chosen: list[int] = []
        for k_key, k in zip(keys, per):
            idx = np.asarray(groups[k_key])
            if k <= 0 or len(idx) == 0:
                continue
            Xg = X[idx]
            # Start from the point nearest the group centroid (avoids edge bias).
            start = int(np.argmin(np.linalg.norm(Xg - Xg.mean(axis=0), axis=1)))
            sel = [start]
            mind = np.linalg.norm(Xg - Xg[start], axis=1)
            for _ in range(min(k, len(idx)) - 1):
                j = int(np.argmax(mind))
                sel.append(j)
                mind = np.minimum(mind, np.linalg.norm(Xg - Xg[j], axis=1))
            chosen.extend(int(idx[s]) for s in sel)

        self.rng.shuffle(chosen)
        return [pool[i] for i in chosen]

    def _to_unit_matrix(self, configs: list[dict]) -> np.ndarray:
        """Map a list of configs to an [n, d] matrix with every axis in [0, 1]
        (continuous/discrete min-max scaled; categorical as evenly-spaced codes)."""
        rows = []
        for c in configs:
            row = []
            for p in self.params:
                v = c[p.node][p.name]
                if p.param_type == "categorical":
                    k = max(len(p.choices) - 1, 1)
                    row.append(p.choices.index(v) / k)
                else:
                    span = (p.high - p.low) or 1.0
                    row.append((float(v) - p.low) / span)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def distance_report(self, configs: list[dict]) -> dict:
        """Nearest-neighbour distance stats (normalized space) to detect
        near-duplicates. min_nn is the closest any two configs get; for ~3000
        points in this space it should be well above ~0.05."""
        X = self._to_unit_matrix(configs)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(X)
            d, _ = tree.query(X, k=2)  # k=1 is self (dist 0)
            nn = d[:, 1]
        except Exception:
            # Fallback: brute force (fine for a few thousand points)
            from scipy.spatial.distance import pdist, squareform
            D = squareform(pdist(X))
            np.fill_diagonal(D, np.inf)
            nn = D.min(axis=1)
        return {
            "n": len(configs),
            "dims": X.shape[1],
            "min_nn": float(nn.min()),
            "p1_nn": float(np.percentile(nn, 1)),
            "median_nn": float(np.median(nn)),
            "n_pairs_below_0.05": int((nn < 0.05).sum()),
        }

    def _sample_param(self, p: ParamDef) -> Any:
        if p.param_type == "continuous":
            if p.log_scale:
                return float(np.exp(self.rng.uniform(np.log(p.low), np.log(p.high))))
            return float(self.rng.uniform(p.low, p.high))
        if p.param_type == "discrete":
            return int(self.rng.integers(int(p.low), int(p.high) + 1))
        if p.param_type == "categorical":
            return str(self.rng.choice(p.choices))
        raise ValueError(f"Unknown param_type: {p.param_type}")

    # ── CSV helpers ─────────────────────────────────────────────────────────
    def get_param_names(self) -> list[str]:
        return [f"{p.node}__{p.name}" for p in self.params]

    def flatten(self, config: dict) -> dict[str, Any]:
        flat = {}
        for node, params in config.items():
            for name, value in params.items():
                flat[f"{node}__{name}"] = value
        return flat

    # ── Pre-generation (reproducible, applied one-per-trial) ─────────────────
    def generate_and_save(
        self, n_configs: int, output_path: str = "presampled_configs.json",
        method: str = "maximin", oversample: int = 4,
    ) -> list[dict]:
        """
        Pre-generate n configurations and save to JSON.

        method="maximin" (default): space-filling with a guaranteed minimum
        separation (no near-duplicates). method="lhs": plain optimized LHS.

        Each entry: {"id": i, "config": {node: {name: value}}}. The orchestrator
        applies config i to trial i, so the dataset is fully reproducible from
        this file + the pose file (no fresh randomness at run time).
        """
        import json

        if method == "maximin":
            configs = self.sample_maximin(n_configs, oversample=oversample)
        else:
            configs = self.sample_lhs(n_configs)
        records = [{"id": i, "config": c} for i, c in enumerate(configs)]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        return records

    @staticmethod
    def load_presampled(path: str) -> list[dict]:
        """Load pre-generated configs. Returns a list of nested config dicts.

        Accepts either the wrapped form [{"id", "config"}, ...] or a bare
        list of config dicts.
        """
        import json

        with open(path) as f:
            data = json.load(f)
        return [d["config"] if isinstance(d, dict) and "config" in d else d for d in data]
