#!/usr/bin/env python3
"""
RCT Orchestrator for PAL TIAGo Docker Environment.

Outer-loop controller:
1. Samples random Nav2 parameter configs (the causal intervention C_t)
2. Samples random initial and goal poses from map free space
3. Applies parameters via `ros2 param set` (dynamic reconfigure)
4. Teleports robot in Gazebo, sets AMCL initial pose
5. Runs navigation trial via NavigateToPose action
6. Records outcome data to CSV with checkpoint/resume

Design notes:
- PAL Module Manager is NOT available in public sim Docker images.
  Nav2 nodes are launched via pmb2_2dnav and stay alive between trials.
- All parameter changes use dynamic reconfigure (`ros2 param set`),
  confirmed working on the actual container.
- No file-based config override or module restart needed.
"""

import csv
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory

from rct_collector.scripts.param_space import ParameterSpace, ARM_CONFIGS, ARM_JOINT_NAMES
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.trial_runner import TrialRunner, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the RCT orchestrator."""

    num_trials: int = 3000
    trial_timeout_sec: float = 180.0
    cooldown_sec: float = 3.0

    map_yaml_path: str = ""
    map_frame: str = "map"

    min_goal_distance: float = 3.0
    max_goal_distance: float = 15.0
    obstacle_clearance_m: float = 0.5
    sampling_bounds: Optional[dict] = None

    output_dir: str = "./rct_data"
    checkpoint_file: str = "checkpoint.json"
    results_csv: str = "rct_results.csv"

    max_consecutive_failures: int = 10
    reset_gazebo_between_trials: bool = True

    gazebo_robot_model: str = "tiago"
    scan_topic: str = "/scan_raw"
    odom_topic: str = "/mobile_base_controller/odom"
    collision_threshold: float = 0.15

    collect_risk_features: bool = False
    risk_topic: str = "/risk_state"

    presampled_poses_path: Optional[str] = None
    presampled_configs_path: Optional[str] = None
    seed: Optional[int] = None

    # --- Pre-generated pose/config handling -------------------------------
    # How trial i picks its pose from the pre-sampled pool:
    #   "index"    : pose[i] (the old behaviour). Pose is then a deterministic
    #                function of trial order, i.e. perfectly confounded with
    #                time — anything that drifts over the run (thermal, memory,
    #                map/localization state) is aliased onto the pose factor.
    #   "shuffled" : a seeded permutation of the pool, reshuffled each time the
    #                pool is exhausted. Pose stays balanced but is no longer
    #                collinear with trial index. RECOMMENDED.
    pose_assignment: str = "shuffled"
    # If the pool is smaller than num_trials: reuse it (True, cycling through
    # fresh permutations) or abort (False). The old code silently *clamped
    # num_trials to the pool size*, so `num_trials: 3000` with a 20-pose file
    # quietly produced 20 trials.
    allow_pose_reuse: bool = True
    allow_config_reuse: bool = False   # configs are the treatment: reuse must be deliberate

    # --- Parameter application / verification -----------------------------
    param_service_timeout_sec: float = 5.0
    param_readback_attempts: int = 3      # retries before recording READBACK_FAILED
    param_readback_backoff_sec: float = 0.25
    param_settle_sec: float = 0.0         # pause after set, before read-back
    # Abort the run if a trial's treatment could not be applied at all. Keeps a
    # silent mis-specification from poisoning thousands of rows.
    max_consecutive_integrity_failures: int = 5

    # --- Goal tolerance (recorded per row so success is re-derivable) ------
    xy_goal_tolerance: float = 0.35
    yaw_goal_tolerance: float = 0.65
    check_goal_tolerance_against_nav2: bool = True
    # End a trial as soon as GROUND TRUTH enters the goal tolerance.
    #
    # Leave False for the dual-outcome design. Stopping here fires exactly on
    # the trials where success_true == 1, so Nav2 never renders its own verdict
    # on those trials and success_believed goes missing non-randomly — which
    # destroys the true-vs-believed comparison in the one cell that matters.
    # Arrival is still detected and timestamped either way
    # (gt_ever_within_tolerance, t_first_within_tolerance).
    stop_when_within_tolerance: bool = False

    # --- LiDAR self-return filtering (see TrialRunner) ---------------------
    scan_self_filter_radius_m: float = 0.30

    # --- Ground-truth rate guard -------------------------------------------
    # Startup check on /gazebo/model_states. gazebo_ros_state defaults to 1 Hz,
    # which quantises path_length_m, final_xy_error and min_obstacle_distance.
    # 0.0 disables the check.
    gt_min_rate_hz: float = 20.0

    # --- Stall detection ---------------------------------------------------
    # End a trial as STUCK once the ground-truth pose has been static this long,
    # rather than burning the full trial_timeout_sec on a robot that is not
    # going to recover (this BT has no recovery nodes). 0.0 disables the
    # termination; longest_stall_sec is still recorded either way.
    no_progress_timeout_sec: float = 20.0
    no_progress_dist_m: float = 0.10
    no_progress_yaw_rad: float = 0.20

    # Constants of the r_grad DEFINITION, forwarded to environment_risk_node so
    # a single experiment YAML controls both processes. Deliberately NOT the
    # inflation_radius / cost_scaling_factor treatment values.
    risk_inflation_radius_m: float = 0.30
    risk_cost_scaling_factor: float = 10.0
    scan_angle_mask_deg: Optional[list] = None

    # Arm control (per-episode physical arm move to match the footprint variable)
    move_arm: bool = False
    arm_control_mode: str = "joint_trajectory"  # default mode; per-pose "mode" overrides
    arm_action: str = "/arm_controller/follow_joint_trajectory"
    play_motion_action: str = "/play_motion2"
    arm_move_time_sec: float = 4.0
    arm_move_on_change_only: bool = True  # only re-move the arm when the label changes
    # Confirm the arm physically arrived by comparing /joint_states against the
    # ARM_CONFIGS joint targets. An action reporting SUCCEEDED is not proof.
    verify_arm_joints: bool = True
    arm_joint_tolerance_rad: float = 0.15
    arm_settle_sec: float = 1.0

    # --- Campaign selection ------------------------------------------------
    # "B" (default): current full-mission behavior, unchanged.
    # "A": short decision-point collision probes (see _run_single_probe). The
    # baseline these probes snapshot R under is CAPTURED from Nav2's live
    # launch-default parameter values at startup (_capture_baseline_config),
    # not a hardcoded profile — its specific values don't matter for
    # identification, only that it is fixed and applied before every R
    # snapshot.
    campaign: str = "B"
    horizon_sec: float = 3.0            # how long to watch for a collision after do(C=c)
    baseline_settle_sec: float = 2.0    # how long to drive under baseline before snapshotting R
    washout_sec: float = 2.0            # pause after reverting to baseline, before the next probe
    randomize_arm: bool = True          # if False, c's footprint is forced to the baseline's label
    probe_forward_distance_m: float = 2.5   # local goToPose target, straight ahead of the probe start
    cmd_vel_topic: str = "/mobile_base_controller/cmd_vel"  # only used by --baseline-nudge
    # See TrialRunner._publish_nudge(): under the active-drive probe design the
    # robot is already moving via MPPI during baseline_settle_sec, so R should
    # be non-degenerate without this. Documented escape hatch, default off —
    # flagged rather than silently decided either way.
    baseline_nudge: bool = False


class RCTOrchestrator:

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.param_space = ParameterSpace(seed=config.seed)
        self.pose_sampler: Optional[PoseSampler] = None
        self.trial_runner: Optional[TrialRunner] = None

        self.presampled_configs: Optional[list] = None
        self.presampled_poses: Optional[list] = None

        self.completed_trials: int = 0
        self.consecutive_failures: int = 0
        self.consecutive_integrity_failures: int = 0
        self.results: list[dict] = []
        self._shutdown_requested = False
        self._last_param_outcomes: list = []          # list[ParamOutcome]
        self._last_arm_status: dict = {}
        self._current_arm_label: Optional[str] = None  # last successfully applied arm pose

        # Unique id for THIS process invocation. Every artifact written by this
        # run is namespaced with it, so re-running trial N never overwrites the
        # JSON/plot belonging to an earlier run of trial N.
        self.run_id: str = datetime.now().strftime("%Y%m%dT%H%M%S")

        self.param_applier = None
        self._param_node = None
        self._pose_index_map: Optional[list] = None
        self._config_index_map: Optional[list] = None

        # Campaign A only.
        self._baseline_config: Optional[dict] = None       # captured Nav2 launch defaults, {node:{name:value}}
        self._last_baseline_outcomes: list = []             # list[ParamOutcome], from the most recent reassert

        os.makedirs(self.config.output_dir, exist_ok=True)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.warning(f"Signal {signum} received, saving checkpoint...")
        self._shutdown_requested = True

    def initialize(self):
        if not self.config.map_yaml_path:
            raise ValueError("map_yaml_path must be set")

        # A null seed silently disables every reproducibility guarantee in this
        # package: the assignment map records "seed": null, the pose/config
        # permutations cannot be replayed, and configs fall back to per-trial
        # uniform sampling instead of the maximin/LHS coverage the parameter
        # space provides. The smoke run drifted to 14 tucked / 6 carry this way.
        if self.config.seed is None:
            raise ValueError(
                "seed is None. The run would not be reproducible and the "
                "assignment map could not be replayed. Set `seed:` in the "
                "experiment config (any integer)."
            )

        logger.info(f"Loading map from {self.config.map_yaml_path}")
        self.pose_sampler = PoseSampler(
            map_yaml_path=self.config.map_yaml_path,
            obstacle_clearance_m=self.config.obstacle_clearance_m,
            min_goal_distance=self.config.min_goal_distance,
            max_goal_distance=self.config.max_goal_distance,
            sampling_bounds=self.config.sampling_bounds,
            seed=self.config.seed,
        )
        self.pose_sampler.load_map()

        self._load_presampled()
        bt_path = os.path.join(get_package_share_directory('rct_collector'), 'behavior_trees', 'navigate_w_replanning_only.xml')
        self.trial_runner = TrialRunner(
            timeout_sec=self.config.trial_timeout_sec,
            collision_threshold=self.config.collision_threshold,
            collect_risk_features=self.config.collect_risk_features,
            risk_topic=self.config.risk_topic,
            scan_topic=self.config.scan_topic,
            odom_topic=self.config.odom_topic,
            gazebo_robot_model=self.config.gazebo_robot_model,
            output_dir=self.config.output_dir,
            map_yaml_path=self.config.map_yaml_path,
            bt_xml_path=bt_path,
            run_id=self.run_id,
            xy_goal_tolerance=self.config.xy_goal_tolerance,
            yaw_goal_tolerance=self.config.yaw_goal_tolerance,
            stop_when_within_tolerance=self.config.stop_when_within_tolerance,
            scan_self_filter_radius_m=self.config.scan_self_filter_radius_m,
            scan_angle_mask_deg=self.config.scan_angle_mask_deg,
            gt_min_rate_hz=self.config.gt_min_rate_hz,
            no_progress_timeout_sec=self.config.no_progress_timeout_sec,
            no_progress_dist_m=self.config.no_progress_dist_m,
            no_progress_yaw_rad=self.config.no_progress_yaw_rad,
            cmd_vel_topic=self.config.cmd_vel_topic,
        )

        # TrialRunner has now initialised rclpy; build the parameter client on
        # a dedicated node. It is deliberately NOT added to the recorder's
        # executor — ParamApplier spins it itself via spin_until_future_complete.
        self._init_param_applier()

        self._verify_nav2_running()
        if self.config.campaign == "A":
            # Goal tolerance is a mission-only concept (Campaign B's early-stop
            # / success check); probes don't have one.
            self._baseline_config = self._capture_baseline_config()
            logger.info(
                "Campaign A baseline (captured from Nav2's live launch-default "
                f"values, NOT a hardcoded profile): "
                f"{self.param_space.flatten(self._baseline_config)}")
            self._save_assignment_maps()   # re-write with baseline_config now known
        elif self.config.check_goal_tolerance_against_nav2:
            self._verify_goal_tolerance()

    def _init_param_applier(self):
        import rclpy
        from rclpy.node import Node

        from rct_collector.scripts.param_applier import ParamApplier

        if not rclpy.ok():
            rclpy.init()
        self._param_node = Node("rct_param_applier")
        self.param_applier = ParamApplier(
            self._param_node,
            service_timeout_sec=self.config.param_service_timeout_sec,
            readback_attempts=self.config.param_readback_attempts,
            readback_backoff_sec=self.config.param_readback_backoff_sec,
            settle_sec=self.config.param_settle_sec,
        )
        logger.info("Parameter applier ready (rclpy service clients) ✓")

    def _verify_goal_tolerance(self):
        """Warn loudly if the runner's success criterion disagrees with Nav2's.

        The runner can end a trial early once it is 'within tolerance'. If that
        threshold differs from the controller's goal_checker, the runner and
        Nav2 disagree about what SUCCESS means and the outcome variable becomes
        a mixture of two definitions.
        """
        if self.param_applier is None:
            return
        for name, ours in (("xy_goal_tolerance", self.config.xy_goal_tolerance),
                           ("yaw_goal_tolerance", self.config.yaw_goal_tolerance)):
            found = None
            for path in (f"general_goal_checker.{name}", f"goal_checker.{name}", name):
                val, _t, err = self.param_applier.get("controller_server", path,
                                                      timeout_sec=3.0)
                if err == "" and val is not None:
                    found = (path, float(val))
                    break
            if found is None:
                logger.warning(
                    f"  Could not read Nav2's {name}; cannot confirm the runner's "
                    f"success criterion matches the controller's."
                )
                continue
            path, nav2_val = found
            if abs(nav2_val - ours) > 1e-6:
                raise RuntimeError(
                    f"GOAL TOLERANCE MISMATCH: runner {name}={ours} but Nav2 "
                    f"{path}={nav2_val}. SUCCESS would mean different things to "
                    f"the runner and to Nav2, and the recorded tolerance columns "
                    f"would not let anyone re-derive the outcome label. This was "
                    f"only a warning during the smoke run (runner 0.25 vs Nav2 "
                    f"0.45 for yaw) and every SUCCESS row was decided by a "
                    f"threshold that is not in the CSV. Align the two in "
                    f"nav2_params.yaml, or set check_goal_tolerance_against_nav2: "
                    f"false to collect anyway."
                )
            logger.info(f"  {name} matches Nav2 ({nav2_val}) ✓")

    def _capture_baseline_config(self) -> dict:
        """Capture the FIXED baseline Campaign A snapshots R under, by reading
        back Nav2's live launch-default value for every parameter param_space
        controls. Not a hardcoded profile: a probe's baseline only needs to be
        config-independent and identical every probe, and re-deriving it from
        whatever Nav2 actually booted with is both more honest and impossible
        to let drift out of sync with the live stack.

        Startup-fatal on any read failure — Campaign A cannot begin without
        confirming what "baseline" concretely means for this run.
        """
        from rct_collector.scripts.param_applier import values_match

        baseline: dict = {}
        for p in self.param_space.params:
            value, _declared_type, err = self.param_applier.get(p.ros_node, p.name)
            if err:
                raise RuntimeError(
                    f"Could not read back the live value of {p.ros_node}/{p.name} "
                    f"to establish Campaign A's baseline ({err}). Refusing to "
                    "start with a guessed baseline."
                )
            if p.apply_via == "footprint":
                # Store a LABEL ("tucked"/"carry"), consistent with every other
                # config dict in this codebase, not the raw polygon string —
                # reverse-lookup which ARM_CONFIGS preset the live polygon
                # matches.
                label = next(
                    (lbl for lbl, poly in p.presets.items()
                     if values_match(value, poly, "footprint")), None)
                if label is None:
                    logger.warning(
                        f"  Captured baseline footprint on {p.ros_node} does not "
                        "match any known ARM_CONFIGS preset; storing the raw "
                        "polygon. Physical arm moves to 'baseline' will be "
                        "skipped (no known joint target) — see _move_arm().")
                    label = value
                value = label
            baseline.setdefault(p.node, {})[p.name] = value
        return baseline

    def _verify_nav2_running(self):
        logger.info("Checking Nav2 is running...")
        result = subprocess.run(
            ["ros2", "action", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if "/navigate_to_pose" not in result.stdout:
            raise RuntimeError(
                "/navigate_to_pose action not found. Launch Nav2 first:\n"
                "  ros2 launch gazebo_simulation gazebo_with_navigation.launch.py"
            )
        logger.info("  /navigate_to_pose found ✓")

        # Smoke-test dynamic reconfig
        result = subprocess.run(
            ["ros2", "param", "get", "/controller_server", "controller_frequency"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"  Dynamic reconfig works ✓ ({result.stdout.strip()})")
        else:
            logger.warning("  Could not read /controller_server params")

    def _load_presampled(self):
        """Load pre-generated configs/poses and build an explicit index map.

        The old behaviour was to silently clamp ``num_trials`` down to the size
        of the smallest pool, so a 20-entry pose file turned ``num_trials: 3000``
        into a 20-trial run with only a warning in the log. It also used
        ``pose[trial_id - 1]``, making the pose factor a deterministic function
        of trial order and therefore collinear with anything that drifts over
        the run.

        Now: pools smaller than num_trials are either reused through fresh
        seeded permutations (``allow_*_reuse: true``) or raise. Assignment is
        shuffled by default so pose is balanced but not confounded with time.
        The realised maps are written to disk for reproducibility.
        """
        sources = [
            ("presampled_configs", self.config.presampled_configs_path,
             ParameterSpace.load_presampled, "pre-generated configs"),
            ("presampled_poses", self.config.presampled_poses_path,
             PoseSampler.load_presampled, "pre-sampled poses"),
        ]
        for attr, path, loader, label in sources:
            if path:
                loaded = loader(path)
                setattr(self, attr, loaded)
                logger.info(f"Loaded {len(loaded)} {label} from {path}")

        rng = np.random.default_rng(self.config.seed)
        n = self.config.num_trials

        self._pose_index_map = self._build_index_map(
            "poses", self.presampled_poses, n, self.config.allow_pose_reuse, rng)
        self._config_index_map = self._build_index_map(
            "configs", self.presampled_configs, n, self.config.allow_config_reuse, rng)

        self._save_assignment_maps()

    def _build_index_map(self, label: str, pool: Optional[list], n_trials: int,
                         allow_reuse: bool, rng) -> Optional[list]:
        """Map trial index -> pool index for the whole run."""
        if pool is None:
            return None
        pool_n = len(pool)
        if pool_n == 0:
            raise ValueError(f"Pre-generated {label} pool is empty.")

        if pool_n < n_trials and not allow_reuse:
            raise ValueError(
                f"num_trials ({n_trials}) exceeds the {pool_n} pre-generated "
                f"{label} available, and reuse is disabled. Generate at least "
                f"{n_trials} entries, or set allow_{label[:-1]}_reuse: true. "
                f"(The previous behaviour silently shortened the run to "
                f"{pool_n} trials.)"
            )
        if pool_n < n_trials:
            logger.warning(
                f"Only {pool_n} pre-generated {label} for {n_trials} trials; "
                f"the pool will be reused ~{n_trials / pool_n:.1f}x. Each "
                f"{label[:-1]} therefore appears as a repeated level — treat it "
                f"as a blocking factor in the analysis, not as i.i.d. sampling."
            )

        if self.config.pose_assignment == "index":
            return [i % pool_n for i in range(n_trials)]

        # Shuffled: concatenate independent permutations of the pool until the
        # run is covered. Guarantees near-equal usage of every entry while
        # decorrelating entry identity from trial order.
        out: list = []
        while len(out) < n_trials:
            out.extend(rng.permutation(pool_n).tolist())
        return out[:n_trials]

    def _save_assignment_maps(self):
        """Persist the realised trial -> pool-entry assignment for reproducibility."""
        path = os.path.join(self.config.output_dir,
                            f"assignment_map_{self.run_id}.json")
        try:
            with open(path, "w") as f:
                json.dump({
                    "run_id": self.run_id,
                    "seed": self.config.seed,
                    "num_trials": self.config.num_trials,
                    "pose_assignment": self.config.pose_assignment,
                    "pose_index_map": self._pose_index_map,
                    "config_index_map": self._config_index_map,
                    "campaign": self.config.campaign,
                    # Campaign A provenance: None until _capture_baseline_config()
                    # has run (this is called once before that, from
                    # _load_presampled(), and again right after, in initialize()).
                    "baseline_config": self._baseline_config,
                }, f)
            logger.info(f"Assignment map written to {path}")
        except OSError as e:
            logger.warning(f"Could not write assignment map: {e}")

    def run(self):
        self._load_checkpoint()
        start_trial = self.completed_trials
        unit = "probes" if self.config.campaign == "A" else "trials"
        logger.info(f"RCT collection ({self.config.campaign}): "
                   f"{unit} {start_trial+1}..{self.config.num_trials}")

        for trial_idx in range(start_trial, self.config.num_trials):
            if self._shutdown_requested:
                break

            trial_id = trial_idx + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"{unit.upper()[:-1]} {trial_id}/{self.config.num_trials}")
            logger.info(f"{'='*60}")

            try:
                if self.config.campaign == "A":
                    result = self._run_single_probe(trial_id)
                else:
                    result = self._run_single_trial(trial_id)
                self._record_result(trial_id, result)
                self.consecutive_failures = 0

                # A run where the treatment repeatedly fails to reach the stack
                # is producing rows whose recorded C_t is not the applied C_t.
                # Better to stop than to fill a CSV with unusable trials. For
                # Campaign A, self._last_param_outcomes is the C (treatment)
                # apply's outcomes — baseline-reassert breakage is tracked
                # separately (see _run_single_probe / baseline_valid) and
                # already folds into that row's treatment_valid via
                # _record_result, but does not by itself trip this
                # consecutive-failure guard the same way a broken c does.
                broken = [o for o in self._last_param_outcomes if o.breaks_integrity]
                if broken:
                    self.consecutive_integrity_failures += 1
                    if (self.consecutive_integrity_failures
                            >= self.config.max_consecutive_integrity_failures):
                        logger.critical(
                            f"{self.consecutive_integrity_failures} consecutive trials "
                            f"with an unapplied/mismatched treatment "
                            f"({[o.key for o in broken]}). Stopping — fix the stack "
                            f"before collecting further."
                        )
                        break
                else:
                    self.consecutive_integrity_failures = 0
            except Exception as e:
                logger.error(f"Trial {trial_id} EXCEPTION: {e}", exc_info=True)
                self.consecutive_failures += 1
                self._record_failure(trial_id, str(e))
                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.critical(f"{self.config.max_consecutive_failures} consecutive failures. Stopping.")
                    break

            self.completed_trials = trial_id
            self._save_checkpoint()

            if trial_idx < self.config.num_trials - 1:
                time.sleep(self.config.cooldown_sec)

        self._save_final_results()
        self.shutdown()
        logger.info(f"Done. {self.completed_trials} trials recorded.")

    def _run_single_trial(self, trial_id: int) -> TrialResult:
        idx = trial_id - 1  # trials are 1-indexed; lists are 0-indexed
        self._last_pool_indices = {"config": None, "pose": None}

        # 1. Get this trial's config: pre-generated if available, else fresh
        if self.presampled_configs is not None:
            cfg_idx = (self._config_index_map[idx]
                       if self._config_index_map is not None else idx)
            self._last_pool_indices["config"] = cfg_idx
            params = self.presampled_configs[cfg_idx]
            logger.info(f"  Config (presampled #{cfg_idx}): {self.param_space.flatten(params)}")
        else:
            params = self.param_space.sample()
            logger.info(f"  Config (fresh): {self.param_space.flatten(params)}")

        # 1a. Reset Gazebo. When we rely on arm persistence (move_arm +
        #     move-on-change), /reset_world is SKIPPED because it resets joint
        #     states and would snap the arm back every trial. The base is
        #     repositioned by the per-trial teleport instead.
        arm_persistence = self.config.move_arm and self.config.arm_move_on_change_only
        if self.config.reset_gazebo_between_trials and not arm_persistence:
            self._reset_gazebo()

        # 1b. Move the physical arm — only when needed (first trial, label change,
        #     or after a failed/unknown move). When not relying on persistence,
        #     the arm is re-commanded every trial.
        self._set_arm_for_config(params)

        # 2. Apply via dynamic reconfigure, with read-back verification
        outcomes = self._apply_params(params)
        self._last_param_outcomes = outcomes
        self._log_param_outcomes(outcomes)

        # 3. Get this trial's poses: pre-generated if available, else fresh
        if self.presampled_poses is not None:
            pose_idx = (self._pose_index_map[idx]
                        if self._pose_index_map is not None else idx)
            self._last_pool_indices["pose"] = pose_idx
            entry = self.presampled_poses[pose_idx]
            start_pose, goal_pose = entry["start"], entry["goal"]
            logger.info(f"  Pose (presampled #{pose_idx})")
        else:
            start_pose, goal_pose = self.pose_sampler.sample_start_goal()
        logger.info(f"  Start: ({start_pose['x']:.2f}, {start_pose['y']:.2f})")
        logger.info(f"  Goal:  ({goal_pose['x']:.2f}, {goal_pose['y']:.2f})")

        # 4. Run navigation
        return self.trial_runner.run_trial(
            trial_id=trial_id,
            start_pose=start_pose,
            goal_pose=goal_pose,
            params=params,
        )

    def _run_single_probe(self, probe_id: int) -> TrialResult:
        """Execute one Campaign A decision-point probe and return its row.

        Step numbering matches the design this was reviewed against:
          1. Re-assert the captured baseline (belt — never skipped; this, not
             the end-of-probe revert in step 7, is what actually guarantees R
             is snapshotted under a fixed, treatment-independent state rather
             than whatever `c` the PREVIOUS probe left the stack in).
          2. Sample a pose, start driving toward a short local goal under
             baseline.
          3. Settle under baseline, watched for a baseline-phase collision.
          4. Snapshot R (frozen, pre-treatment).
          5. Sample + apply the random config c = do(C=c).
          6. Watch horizon_sec for a collision, on a clean window.
          7. Revert to baseline (suspenders) + washout.
          8. Write the row.
        """
        idx = probe_id - 1
        self._last_pool_indices = {"config": None, "pose": None}
        result = TrialResult(
            trial_id=probe_id, campaign="A", run_id=self.run_id,
            horizon_sec=self.config.horizon_sec,
            baseline_settle_sec=self.config.baseline_settle_sec,
            washout_sec=self.config.washout_sec,
        )

        # 1. Re-apply the captured baseline.
        baseline_flat = self.param_space.flatten(self._baseline_config)
        logger.info(f"  Probe {probe_id}: re-asserting baseline {baseline_flat}")
        self._last_baseline_outcomes = self._apply_params(self._baseline_config)
        self._log_param_outcomes(self._last_baseline_outcomes)
        self._set_arm_for_config(self._baseline_config)
        result.baseline_valid = int(
            not any(o.breaks_integrity for o in self._last_baseline_outcomes))
        result.baseline_config = baseline_flat

        # 2. Pose + start driving under baseline.
        if self.presampled_poses is not None:
            pose_idx = (self._pose_index_map[idx]
                        if self._pose_index_map is not None else idx)
            self._last_pool_indices["pose"] = pose_idx
            entry = self.presampled_poses[pose_idx]
            start_pose, goal_pose = entry["start"], entry["goal"]
        else:
            start_pose, goal_pose = self.pose_sampler.sample_probe_pose(
                self.config.probe_forward_distance_m)
        result.start_x, result.start_y, result.start_yaw = (
            start_pose["x"], start_pose["y"], start_pose["yaw"])
        result.probe_goal_pose = goal_pose
        logger.info(
            f"  Probe {probe_id}: start ({start_pose['x']:.2f}, "
            f"{start_pose['y']:.2f}) -> local goal ({goal_pose['x']:.2f}, "
            f"{goal_pose['y']:.2f})")

        result.teleport_ok = int(self.trial_runner.start_probe_drive(
            start_pose, goal_pose, nudge=self.config.baseline_nudge))

        # 3. Settle under baseline, watched for a baseline-phase collision.
        baseline_collided, baseline_collision_t = self.trial_runner.watch_probe_collision(
            self.config.baseline_settle_sec)
        if baseline_collided:
            logger.warning(
                f"  Probe {probe_id}: collision DURING baseline settle "
                f"(t={baseline_collision_t:.2f}s) — R was never validly "
                "snapshotted under a clean pre-treatment state. Aborting probe.")
            self.trial_runner.stop_probe_drive()
            result.status = "BASELINE_COLLISION"
            result.failure_reason = "BASELINE_COLLISION"
            result.collision = True
            result.y_h = None
            self._last_param_outcomes = []
            self._finalize_probe(result, probe_id)
            time.sleep(self.config.washout_sec)
            return result

        # 4. Snapshot R — the frozen pre-treatment state.
        result.risk_state_snapshot = self.trial_runner.get_risk_snapshot()
        result.r_snapshot_time = time.time()
        result.localization_error_m = self.trial_runner.get_localization_error()
        if result.risk_state_snapshot is None:
            logger.warning(
                f"  Probe {probe_id}: no risk state available at snapshot "
                "time (collect_risk_features off, or environment_risk_node "
                "not publishing yet?).")

        # 5. Sample + apply the random config c = do(C=c).
        if self.presampled_configs is not None:
            cfg_idx = (self._config_index_map[idx]
                       if self._config_index_map is not None else idx)
            self._last_pool_indices["config"] = cfg_idx
            params = self.presampled_configs[cfg_idx]
        else:
            params = self.param_space.sample()

        if not self.config.randomize_arm:
            arm_pd = next((p for p in self.param_space.params
                           if getattr(p, "apply_via", "") == "footprint"), None)
            if arm_pd is not None:
                baseline_label = self._baseline_config.get(arm_pd.node, {}).get(arm_pd.name)
                if baseline_label is not None:
                    params = {**params, arm_pd.node: {**params[arm_pd.node],
                                                        arm_pd.name: baseline_label}}

        logger.info(f"  Probe {probe_id}: applying c = {self.param_space.flatten(params)}")
        outcomes = self._apply_params(params)
        self._last_param_outcomes = outcomes
        self._log_param_outcomes(outcomes)
        self._set_arm_for_config(params)
        result.c_apply_time = time.time()
        assert result.c_apply_time > result.r_snapshot_time, (
            "do(C=c) timestamped before the R snapshot — pre-treatment-R "
            "invariant violated.")
        result.params = self.param_space.flatten(params)

        # 6. Watch horizon_sec for collision, on a clean window.
        self.trial_runner.reset_probe_collision_state()
        y_h, collision_t = self.trial_runner.watch_probe_collision(self.config.horizon_sec)
        self.trial_runner.stop_probe_drive()
        result.y_h = y_h
        result.collision = bool(y_h)
        result.collision_time_sec = collision_t
        result.status = "COLLISION" if y_h else "PROBE_COMPLETE"

        # Scoped to exactly this horizon window (reset_probe_collision_state()
        # just above is the start of the scope) — confirms y_h=0 means "no
        # collision", not "the channel went silent and nothing was observed".
        result.collision_msgs_seen, result.collision_channel_silent = (
            self.trial_runner.get_collision_channel_status())
        if result.collision_channel_silent:
            logger.error(
                f"  Probe {probe_id}: no messages on /gazebo/collision during "
                "the horizon window — y_h=0 here would mean 'not observed', "
                "not 'no collision'. Row flagged collision_channel_silent=1.")

        # 7. Revert to baseline (suspenders — the correctness mechanism is
        #    step 1 of the NEXT probe, not this) + washout.
        revert_outcomes = self._apply_params(self._baseline_config)
        self._set_arm_for_config(self._baseline_config)
        if any(o.breaks_integrity for o in revert_outcomes):
            logger.warning(
                f"  Probe {probe_id}: post-probe baseline revert did not "
                "fully apply. The next probe's step-1 re-assert is what "
                "actually matters for correctness, but flagging here too.")
        time.sleep(self.config.washout_sec)

        # 8. Write the row.
        self._finalize_probe(result, probe_id)
        return result

    def _finalize_probe(self, result: TrialResult, probe_id: int) -> None:
        """Write a Campaign A probe's lean JSON (no time-series — see
        TrialRunner.write_probe_json) and stamp result.json_path."""
        payload = {
            "trial_id": probe_id,
            "campaign": "A",
            "run_id": self.run_id,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "y_h": result.y_h,
            "collision_time_sec": result.collision_time_sec,
            "r_snapshot_time": result.r_snapshot_time,
            "c_apply_time": result.c_apply_time,
            "horizon_sec": result.horizon_sec,
            "baseline_settle_sec": result.baseline_settle_sec,
            "washout_sec": result.washout_sec,
            "start_pose": {"x": result.start_x, "y": result.start_y, "yaw": result.start_yaw},
            "probe_goal_pose": result.probe_goal_pose,
            "baseline_config": result.baseline_config,
            "nav2_config": result.params,
            "risk_state_snapshot": result.risk_state_snapshot,
            "baseline_valid": result.baseline_valid,
            "teleport_ok": result.teleport_ok,
            "localization_error_m": result.localization_error_m,
            "collision_msgs_seen": result.collision_msgs_seen,
            "collision_channel_silent": result.collision_channel_silent,
        }
        result.json_path = self.trial_runner.write_probe_json(payload, probe_id)

    def _apply_params(self, params: dict) -> list:
        """Set every treatment parameter and classify how each one went.

        Returns a list of ParamOutcome (one per ROS target, so a knob linked to
        both costmaps via extra_targets yields two entries).

        Why this matters for the RCT: the CSV records the value we *intended*
        to apply. If a set is silently rejected, that row claims do(C=x) for a
        trial in which C never changed, which biases that parameter's estimated
        effect toward zero. The read-back is the manipulation check.

        What changed from the original: verification now distinguishes "the node
        refused / disagreed" (treatment integrity broken) from "the node
        accepted but we failed to read it back" (a tooling hiccup, trial still
        usable). Previously both wrote the same `params_unverified` flag, which
        made 53% of the smoke run look invalid when in fact zero sets had been
        rejected and zero values mismatched.

        Two special cases from the finalized space:
        - extra_targets: one sampled value applied to several ROS params (e.g.
          inflation radius / footprint on BOTH local and global costmaps), so it
          stays a single causal knob. Every target must verify.
        - apply_via == "footprint": the sampled value is a label ("tucked" /
          "carry") resolved to a concrete polygon preset before setting.

        NOTE: read-back confirms the *parameter* changed. For inflation_radius
        and footprint it does NOT by itself prove the costmap *cost cache* /
        collision geometry was rebuilt — those are per-episode and verified
        behaviorally once (see README: inflation gradient).
        """
        outcomes = []
        for p in self.param_space.params:
            try:
                value = params[p.node][p.name]
            except KeyError:
                continue

            apply_via = getattr(p, "apply_via", "param_set")
            extra = getattr(p, "extra_targets", None) or []
            ros_node = getattr(p, "ros_node", None) or p.node

            if apply_via == "footprint":
                presets = getattr(p, "presets", None) or {}
                value = presets.get(value, value)
                param_type = "footprint"
            else:
                param_type = p.param_type

            for en, enm in [(ros_node, p.name)] + list(extra):
                out = self.param_applier.set_and_verify(en, enm, value, param_type)
                # Remember which logical knob this ROS target belongs to, so the
                # CSV can carry a param_actual__<knob> column.
                out.logical_key = f"{p.node}__{p.name}"
                outcomes.append(out)
        return outcomes

    def _log_param_outcomes(self, outcomes: list):
        from rct_collector.scripts import param_applier as pa

        bad = [o for o in outcomes if o.breaks_integrity]
        soft = [o for o in outcomes if o.outcome == pa.READBACK_FAILED]

        if bad:
            logger.error(
                f"  TREATMENT NOT APPLIED for {len(bad)} target(s): "
                + "; ".join(f"{o.key} [{o.outcome}] {o.detail}" for o in bad)
                + ". Row flagged treatment_valid=0 — exclude from causal estimates."
            )
        if soft:
            logger.warning(
                f"  {len(soft)} target(s) set OK but could not be read back after "
                f"{self.config.param_readback_attempts} attempts: "
                + "; ".join(o.key for o in soft)
                + ". Treatment is most likely correct; row stays usable "
                  "(treatment_valid=1, config_verified=0)."
            )
        if not bad and not soft:
            total_ms = 1000.0 * sum(o.elapsed_sec for o in outcomes)
            logger.info(
                f"  All {len(outcomes)} params set and verified ✓ ({total_ms:.0f} ms)")

    def _set_arm_for_config(self, params: dict) -> None:
        """Resolve the footprint/arm label from `params`, move the physical
        arm if needed (respecting arm_move_on_change_only persistence), and
        verify via /joint_states. Fills self._last_arm_status. A no-op
        (status stays blank) when move_arm=False.

        Extracted from _run_single_trial's former inline arm-handling block —
        pure extraction, same calls in the same order under the same
        conditions — so Campaign A probes (which call this twice per probe:
        once for the baseline label, once for c's label) share the exact same
        persistence/verification logic Campaign B missions use, rather than a
        second implementation that could drift from it.
        """
        self._last_arm_status = {
            "arm_requested_label": "",
            "arm_move_attempted": 0,
            "arm_move_skipped_persistent": 0,
            "arm_verified": "",       # 1 / 0 / "" when not applicable
            "arm_max_joint_error_rad": "",
            "arm_detail": "",
        }
        if not self.config.move_arm:
            return

        arm_pd = next(
            (p for p in self.param_space.params
             if getattr(p, "apply_via", "") == "footprint"), None
        )
        if arm_pd is None:
            return

        arm_persistence = self.config.move_arm and self.config.arm_move_on_change_only
        label = params[arm_pd.node][arm_pd.name]
        self._last_arm_status["arm_requested_label"] = label
        if arm_persistence and label == self._current_arm_label:
            logger.info(f"  Arm already in '{label}' — skipping move")
            self._last_arm_status["arm_move_skipped_persistent"] = 1
            # Persistence is an assumption, not an observation: confirm
            # the arm is still where we left it.
            self._confirm_arm_pose(label)
        else:
            self._last_arm_status["arm_move_attempted"] = 1
            if self._move_arm(label):
                self._current_arm_label = label
            else:
                self._current_arm_label = None  # unknown -> force re-move next time
            self._confirm_arm_pose(label)

        if self._last_arm_status["arm_verified"] == 0:
            logger.warning(
                f"  Arm did not reach '{label}' — physical geometry does "
                f"NOT match the footprint treatment for this row. "
                f"Row will be flagged arm_verified=0."
            )
            self._current_arm_label = None

    def _confirm_arm_pose(self, label: str):
        """Check /joint_states against the ARM_CONFIGS joint targets for `label`.

        The footprint parameter is only a *proxy* for the physical arm pose. If
        the arm never moved, the costmap says "carry" while the robot is still
        tucked, and the trial silently measures the wrong treatment. The old
        code logged a warning and moved on; nothing reached the CSV, so these
        trials were indistinguishable from clean ones during analysis.

        An action reporting SUCCEEDED is not sufficient evidence either, so the
        joint positions themselves are compared. Fills self._last_arm_status.
        """
        st = self._last_arm_status
        if not self.config.verify_arm_joints:
            return

        cfg = ARM_CONFIGS.get(label) or {}
        target = cfg.get("joints")
        if not target:
            st["arm_detail"] = "no_joint_target_defined"
            logger.warning(
                f"  Arm config '{label}' has no joint targets; cannot verify the "
                f"physical pose matches the footprint treatment."
            )
            return

        if self.config.arm_settle_sec > 0:
            time.sleep(self.config.arm_settle_sec)

        actual = self.trial_runner.get_arm_joint_positions(ARM_JOINT_NAMES)
        if actual is None:
            st["arm_verified"] = 0
            st["arm_detail"] = "no_joint_states"
            logger.warning("  No /joint_states received; cannot verify arm pose.")
            return

        missing = [n for n in ARM_JOINT_NAMES if n not in actual]
        if missing:
            st["arm_verified"] = 0
            st["arm_detail"] = f"missing_joints:{','.join(missing)}"
            return

        errs = [abs(actual[n] - float(t)) for n, t in zip(ARM_JOINT_NAMES, target)]
        max_err = max(errs)
        st["arm_max_joint_error_rad"] = round(max_err, 4)
        st["arm_verified"] = int(max_err <= self.config.arm_joint_tolerance_rad)
        if not st["arm_verified"]:
            worst = ARM_JOINT_NAMES[int(np.argmax(errs))]
            st["arm_detail"] = f"worst={worst}:{max_err:.3f}rad"
        else:
            logger.info(f"  Arm pose '{label}' verified ✓ (max err {max_err:.3f} rad)")

    def _move_arm(self, label: str) -> bool:
        """Move TIAGo's arm to the configuration for `label` and wait for it to
        finish. Each pose picks its own control mode (cfg["mode"]): "play_motion"
        uses play_motion2 (collision-free named motion, e.g. "home"); else a
        FollowJointTrajectory on /arm_controller. If play_motion fails and joints
        are available, falls back to a joint trajectory.
        """
        cfg = ARM_CONFIGS.get(label)
        if cfg is None:
            logger.warning(f"  No arm config defined for '{label}'")
            return False

        mode = cfg.get("mode", self.config.arm_control_mode)

        if mode == "play_motion":
            if self._send_play_motion(cfg.get("motion_name", label), label):
                return True
            if cfg.get("joints"):
                logger.info(f"  play_motion for '{label}' failed; trying joint trajectory")
                return self._send_joint_trajectory(cfg["joints"], label)
            return False

        return self._send_joint_trajectory(cfg.get("joints"), label)

    def _send_play_motion(self, motion: str, label: str) -> bool:
        goal = f"{{motion_name: {motion}, skip_planning: false}}"
        cmd = ["ros2", "action", "send_goal", self.config.play_motion_action,
               "play_motion2_msgs/action/PlayMotion", goal]
        return self._send_arm_goal(cmd, label)

    def _send_joint_trajectory(self, joints, label: str) -> bool:
        if not joints:
            logger.warning(
                f"  Arm joints for '{label}' are unset — skipping arm motion."
            )
            return False
        if len(joints) != len(ARM_JOINT_NAMES):
            logger.warning(
                f"  Arm config '{label}' has {len(joints)} joints, expected "
                f"{len(ARM_JOINT_NAMES)}."
            )
            return False
        names = ", ".join(ARM_JOINT_NAMES)
        pos = ", ".join(str(float(v)) for v in joints)
        t = int(self.config.arm_move_time_sec)
        goal = (
            f"{{trajectory: {{joint_names: [{names}], "
            f"points: [{{positions: [{pos}], time_from_start: {{sec: {t}}}}}]}}}}"
        )
        cmd = ["ros2", "action", "send_goal", self.config.arm_action,
               "control_msgs/action/FollowJointTrajectory", goal]
        return self._send_arm_goal(cmd, label)

    def _send_arm_goal(self, cmd: list, label: str) -> bool:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.arm_move_time_sec + 20,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"  Arm motion to '{label}' timed out")
            return False
        if r.returncode == 0 and "SUCCEEDED" in r.stdout:
            logger.info(f"  Arm moved to '{label}' ✓")
            return True
        logger.warning(
            f"  Arm motion to '{label}' failed: {(r.stdout + r.stderr).strip()[-200:]}"
        )
        return False

    def _reset_gazebo(self):
        try:
            subprocess.run(
                ["ros2", "service", "call", "/reset_world", "std_srvs/srv/Empty", "{}"],
                capture_output=True, timeout=10,
            )
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  Gazebo reset failed: {e}")

    # ── Recording ───────────────────────────────────────────────────────

    def _record_result(self, trial_id: int, result: TrialResult):
        from rct_collector.scripts import param_applier as pa

        row = {"run_id": self.run_id, "trial_id": trial_id,
               "timestamp": datetime.now().isoformat(), **result.to_dict()}

        # --- provenance: which pool entry produced this trial ---------------
        pools = getattr(self, "_last_pool_indices", {}) or {}
        row["config_pool_index"] = pools.get("config", "")
        row["pose_pool_index"] = pools.get("pose", "")

        # --- treatment fidelity --------------------------------------------
        outcomes = self._last_param_outcomes
        by_outcome: dict[str, list] = {}
        for o in outcomes:
            by_outcome.setdefault(o.outcome, []).append(o.key)

        rejected = by_outcome.get(pa.SET_REJECTED, []) + by_outcome.get(pa.SET_TIMEOUT, [])
        rejected += by_outcome.get(pa.NO_SERVICE, [])
        mismatched = by_outcome.get(pa.MISMATCH, [])
        readback_failed = by_outcome.get(pa.READBACK_FAILED, [])
        not_ok = rejected + mismatched + readback_failed

        row["params_rejected"] = ";".join(rejected)
        row["params_rejected_count"] = len(rejected)
        row["params_mismatched"] = ";".join(mismatched)
        row["params_mismatched_count"] = len(mismatched)
        row["params_readback_failed"] = ";".join(readback_failed)
        row["params_readback_failed_count"] = len(readback_failed)

        # Backwards-compatible names, but now meaning only "not fully confirmed".
        row["params_unverified"] = ";".join(not_ok)
        row["params_unverified_count"] = len(not_ok)

        # config_verified : every target confirmed by read-back (strict).
        # treatment_valid : nothing was rejected or read back wrong. This is the
        #                   column to filter on for causal analysis — a failed
        #                   read-back alone does not invalidate a trial.
        row["config_verified"] = int(len(not_ok) == 0)
        row["treatment_valid"] = int(len(rejected) == 0 and len(mismatched) == 0)
        # Campaign A: a probe whose BASELINE reassert didn't fully apply is
        # just as unusable as one whose c didn't — R would not have been
        # snapshotted under the fixed pre-treatment state the campaign
        # requires. baseline_valid is scored in _run_single_probe / to_dict().
        if result.campaign == "A" and not result.baseline_valid:
            row["treatment_valid"] = 0
        # A probe aborted during baseline settle never reaches step 5 — c was
        # never applied at all (self._last_param_outcomes is empty for that
        # row, which would otherwise read as vacuously "nothing rejected").
        # treatment_valid must say "not applicable", not "valid".
        if result.campaign == "A" and result.status == "BASELINE_COLLISION":
            row["treatment_valid"] = 0
        row["config_valid"] = row["treatment_valid"]   # legacy alias
        row["param_apply_detail"] = ";".join(
            f"{o.key}={o.outcome}" for o in outcomes if o.outcome != pa.OK)
        row["param_apply_sec"] = round(sum(o.elapsed_sec for o in outcomes), 3)

        # Read-back values alongside the intended ones, so a mismatch is
        # recoverable in analysis instead of merely flagged.
        for o in outcomes:
            if o.actual is not None:
                row[f"param_actual__{o.logical_key}"] = o.actual

        # --- arm / footprint fidelity ---------------------------------------
        row.update(self._last_arm_status or {})

        # --- success criterion provenance ------------------------------------
        # The three outcome columns (success_true, believed_within_tolerance,
        # success_believed) are scored inside TrialResult._score_outcomes() and
        # arrive via to_dict(); they are NOT recomputed here, so there is one
        # definition of success in the codebase rather than two.
        row["xy_goal_tolerance"] = self.config.xy_goal_tolerance
        row["yaw_goal_tolerance"] = self.config.yaw_goal_tolerance
        # Legacy alias kept so older analysis scripts keep working. Identical to
        # success_true; prefer that name in new work.
        row["within_goal_tolerance"] = (
            "" if result.success_true is None else int(result.success_true))

        if result.outcome_agreement == "FALSE_SUCCESS":
            logger.warning(
                f"  Trial {trial_id} scored FALSE_SUCCESS "
                f"(true error {result.final_xy_error:.3f} m vs believed "
                f"{result.believed_final_xy_error:.3f} m). Do NOT use "
                "success_believed as the outcome for causal estimates.")

        self.results.append(row)
        self._append_csv(row)

    def _record_failure(self, trial_id: int, error_msg: str):
        row = {"run_id": self.run_id, "trial_id": trial_id,
               "timestamp": datetime.now().isoformat(),
               "status": "EXCEPTION", "error": error_msg,
               "failure_reason": "RUNNER_EXCEPTION",
               "treatment_valid": 0, "config_valid": 0, "config_verified": 0}
        self.results.append(row)
        self._append_csv(row)

    def _append_csv(self, row: dict):
        csv_path = os.path.join(self.config.output_dir, self.config.results_csv)
        exists = os.path.exists(csv_path)

        if not exists:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                w.writeheader()
                w.writerow(row)
        else:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                new_keys = set(row.keys()) - set(headers)
                if new_keys:
                    rows = list(reader)  # only need the full contents when re-writing

            if new_keys:
                # Extend headers (rare — only first few trials)
                all_h = sorted(set(headers) | set(row.keys()))
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=all_h)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                    w.writerow(row)
            else:
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=headers).writerow(row)

    # ── Checkpoint ──────────────────────────────────────────────────────

    def _save_checkpoint(self):
        path = os.path.join(self.config.output_dir, self.config.checkpoint_file)
        with open(path, "w") as f:
            json.dump({
                "completed_trials": self.completed_trials,
                "timestamp": datetime.now().isoformat(),
                "map": self.config.map_yaml_path,
                "run_id": self.run_id,
                "seed": self.config.seed,
            }, f, indent=2)

    def _load_checkpoint(self):
        path = os.path.join(self.config.output_dir, self.config.checkpoint_file)
        if os.path.exists(path):
            with open(path) as f:
                self.completed_trials = json.load(f)["completed_trials"]
            logger.info(f"Resumed from checkpoint: {self.completed_trials} done")
        else:
            logger.info("No checkpoint, starting fresh")

    def _save_final_results(self):
        """Recompute the summary from the CSV on disk, not from memory.

        The old version counted statuses in ``self.results`` (only the rows this
        process produced) while reporting ``total = self.completed_trials``
        (which comes from the checkpoint and includes earlier sessions). On any
        resumed run the two disagreed — the smoke output claimed total 10 with
        statuses summing to 5, and COLLISION 0 while the CSV held 3.
        """
        csv_path = os.path.join(self.config.output_dir, self.config.results_csv)
        rows: list[dict] = []
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

        def n(pred) -> int:
            return sum(1 for r in rows if pred(r))

        def truthy(r, key) -> bool:
            return str(r.get(key, "")).strip() in ("1", "1.0", "True", "true")

        def _flt(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        statuses = sorted({(r.get("status") or "UNKNOWN") for r in rows})
        summary = {
            "run_id": self.run_id,
            "generated_at": datetime.now().isoformat(),
            "source_csv": csv_path,
            "rows_in_csv": len(rows),
            "completed_trials_this_session": self.completed_trials,
            "by_status": {st: n(lambda r, st=st: (r.get("status") or "UNKNOWN") == st)
                          for st in statuses},
            "collisions": n(lambda r: str(r.get("collision", "")).strip() in ("1", "1.0", "True")),
            "treatment_valid": n(lambda r: truthy(r, "treatment_valid")),
            "treatment_invalid": n(lambda r: not truthy(r, "treatment_valid")),
            "config_verified": n(lambda r: truthy(r, "config_verified")),
            "readback_failures": n(
                lambda r: (r.get("params_readback_failed_count") or "0") not in ("0", "")),
            "arm_unverified": n(lambda r: str(r.get("arm_verified", "")).strip() == "0"),
            # Rows whose start pose was never confirmed in Gazebo.
            "teleport_unverified": n(
                lambda r: str(r.get("teleport_ok", "1")).strip() == "0"),
            # Rows where NOTHING was published on /gazebo/collision, so
            # collision=0 means "not observed", not "did not happen".
            "collision_channel_silent": n(
                lambda r: truthy(r, "collision_channel_silent")),
            # Pose-quantisation regression guard. Should be ~0 with ground
            # truth; a non-zero count means the pose source reverted to AMCL.
            "quantised_pose_trials": n(
                lambda r: _flt(r.get("unique_pose_fraction")) is not None
                and _flt(r.get("unique_pose_fraction")) < 0.5),
            # Failure taxonomy: lets infrastructure failures be told apart from
            # genuine navigation failures without re-reading every row.
            "by_failure_reason": {
                fr: n(lambda r, fr=fr: (r.get("failure_reason") or "") == fr)
                for fr in sorted({(r.get("failure_reason") or "") for r in rows})
            },
            # ── dual outcome ────────────────────────────────────────────────
            # success_true is the estimand. success_believed is what the robot
            # thought. A large false_success count means an analysis scored on
            # the belief would have been measuring overconfidence, not arrival.
            "success_true": n(lambda r: truthy(r, "success_true")),
            "believed_within_tolerance": n(
                lambda r: truthy(r, "believed_within_tolerance")),
            "nav2_reported_success": n(lambda r: truthy(r, "success_believed")),
            "belief_censored": n(lambda r: truthy(r, "belief_censored")),
            "outcome_agreement": {
                oa: n(lambda r, oa=oa: (r.get("outcome_agreement") or "") == oa)
                for oa in sorted({(r.get("outcome_agreement") or "") for r in rows})
            },
            # Robot believed it arrived and did not. The dangerous cell.
            "false_success": n(
                lambda r: (r.get("outcome_agreement") or "") == "FALSE_SUCCESS"),
            "missed_success": n(
                lambda r: (r.get("outcome_agreement") or "") == "MISSED_SUCCESS"),
            # Arrived at some point but did not end inside tolerance.
            "arrived_then_left": n(
                lambda r: truthy(r, "gt_ever_within_tolerance")
                and not truthy(r, "success_true")),
            # Nav2's action result disagreeing with Nav2's OWN goal checker is a
            # different fault from localization drift and needs separate triage.
            "nav2_result_inconsistent": n(
                lambda r: truthy(r, "success_believed")
                and str(r.get("believed_within_tolerance", "")).strip() == "0"),
            "mean_belief_error_gap_m": (
                round(sum(g for g in (_flt(r.get("belief_error_gap_m")) for r in rows)
                          if g is not None)
                      / max(sum(1 for r in rows
                                if _flt(r.get("belief_error_gap_m")) is not None), 1), 4)),
            "usable_for_causal_analysis": n(
                lambda r: truthy(r, "treatment_valid")
                and str(r.get("arm_verified", "")).strip() != "0"
                and str(r.get("teleport_ok", "1")).strip() != "0"
                and not truthy(r, "collision_channel_silent")
                and (_flt(r.get("unique_pose_fraction")) or 1.0) >= 0.5),
        }
        # Sanity: the status buckets must account for every row.
        summary["status_counts_sum"] = sum(summary["by_status"].values())
        summary["status_counts_consistent"] = (
            summary["status_counts_sum"] == summary["rows_in_csv"])

        path = os.path.join(self.config.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    def shutdown(self):
        """Release the parameter-applier node. Safe to call more than once."""
        if self._param_node is not None:
            try:
                self._param_node.destroy_node()
            except Exception:
                pass
            self._param_node = None