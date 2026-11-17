#!/usr/bin/env python3
"""
RCT Data Collection Entry Point.

Usage:
    # Step 1: Visualize sampling area (no ROS needed)
    python3 -m rct_collector.scripts.main --map /path/to/map.yaml --visualize-only

    # Step 2: Pre-generate poses, inspect, save to JSON
    python3 -m rct_collector.scripts.main --map /path/to/map.yaml --generate-poses 3000

    # Step 3: Visualize the pre-generated poses
    python3 -m rct_collector.scripts.main --map /path/to/map.yaml --visualize-only \
        --presampled-poses ./rct_data/presampled_poses.json

    # Step 4: Run trials using the pre-generated poses
    python3 -m rct_collector.scripts.main --map /path/to/map.yaml --trials 3000 \
        --presampled-poses ./rct_data/presaSmpled_poses.json

    # Resume from checkpoint
    python3 -m rct_collector.scripts.main --map /path/to/map.yaml --resume --output ./rct_data \
        --presampled-poses ./rct_data/presampled_poses.json
"""

import argparse
import logging
import os
import sys

import yaml

from rct_collector.scripts.orchestrator import OrchestratorConfig, RCTOrchestrator

# Module-level so helpers such as _config_from_yaml can log (it used to raise
# NameError on any unknown YAML key).
logger = logging.getLogger("rct_collector.main")

# Process exit codes per stop reason, for a supervisor to act on.
EXIT_CODES = {"OK": 0, "STOP_AT": 0, "INTERRUPTED": 2, "SIM_STALLED": 3,
              "EXCEPTION": 4, "INTEGRITY_FAILURES": 5, "CONSECUTIVE_FAILURES": 6}


def setup_logging(log_level: str = "INFO", log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args():
    p = argparse.ArgumentParser(description="RCT Data Collector for Nav2")
    p.add_argument("--map", type=str, help="Path to map YAML file")
    p.add_argument("--trials", type=int, default=3000)
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: the config's output_dir, "
                        "else ./rct_data). Only overrides the YAML when given.")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--cooldown", type=float, default=3.0)
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--footprint-clearance", type=float, default=0.0,
                   help="inscribed radius of the largest (carry) footprint; "
                        "when >0 the pose pool is carry-feasible and shared "
                        "across both arm states (simple circular check)")
    p.add_argument("--carry-footprint", type=str, default=None,
                   help="carry footprint polygon as a '[[x,y],...]' string "
                        "(robot frame). When set, poses are checked "
                        "orientation-aware: the footprint must fit at the "
                        "sampled yaw. Supersedes --footprint-clearance.")
    p.add_argument("--footprint-yaw-bins", type=int, default=72,
                   help="yaw discretisation for the orientation-aware check")
    p.add_argument("--footprint-margin", type=float, default=0.0,
                   help="extra safety margin (m) added around the footprint")
    p.add_argument("--min-distance", type=float, default=10.0)
    p.add_argument("--max-distance", type=float, default=20.0)
    p.add_argument("--config", type=str, help="YAML config file")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-at", type=int, default=None, metavar="PROBE_ID",
                   help="Stop cleanly after this probe id (chunk boundary). "
                        "num_trials, and so the checkpoint identity, is unchanged.")
    p.add_argument("--attempt-kind", choices=["rerun", "replicate"], default="rerun",
                   help="tag for --rerun-trials rows: rerun (replaces a quarantined "
                        "attempt) or replicate (noise measurement, never accepted)")
    p.add_argument("--rerun-trials", type=str, default=None, metavar="ID,ID,...",
                   help="Re-run these already-completed probe ids (rows tagged "
                        "attempt_kind=rerun); the checkpoint is not moved. Needs --resume.")
    p.add_argument("--visualize-only", action="store_true")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Number of sample pairs to show in visualization (default: 20)")
    p.add_argument("--generate-poses", type=int, default=None, metavar="N",
                   help="Pre-generate N pose pairs and save to JSON (no ROS needed)")
    p.add_argument("--presampled-poses", type=str, default=None,
                   help="Path to pre-generated poses JSON file")
    p.add_argument("--generate-configs", type=int, default=None, metavar="N",
                   help="Pre-generate N configs and save to JSON (no ROS/map needed)")
    p.add_argument("--presampled-configs", type=str, default=None,
                   help="Path to pre-generated configs JSON file")
    p.add_argument("--sampling", choices=["maximin", "lhs"], default="maximin",
                   help="Config sampler: maximin (no near-duplicates, default) or lhs")
    p.add_argument("--oversample", type=int, default=4,
                   help="Pool multiplier for maximin selection (default 4)")
    p.add_argument("--move-arm", action="store_true",
                   help="Physically move TIAGo's arm per trial to match the footprint")
    p.add_argument("--arm-control-mode", choices=["joint_trajectory", "play_motion"],
                   default="joint_trajectory")
    p.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    p.add_argument("--arm-move-time", type=float, default=4.0)
    p.add_argument("--arm-move-every-trial", action="store_true",
                   help="Re-command the arm every trial instead of only on label change")
    p.add_argument("--no-gazebo-reset", action="store_true",
                   help="Disable /reset_world between trials")
    p.add_argument("--collect-risk", action="store_true")
    p.add_argument("--scan-topic", default="/scan_raw")
    p.add_argument("--odom-topic", default="/mobile_base_controller/odom")
    p.add_argument("--robot-model", default="tiago", help="Gazebo model name")
    p.add_argument("--collision-threshold", type=float, default=0.15)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--world-name", type=str, default=None, help="Name of the world for dataset metadata")
    p.add_argument("--seed", type=int, default=None)

    # --- Probe Data Collection -----------------------
    p.add_argument("--n-probes", type=int, default=None,
                   help="Number of decision-point probes to run")
    p.add_argument("--horizon-sec", type=float, default=3.0,
                   help="Seconds to watch for a collision after candidate config is applied")
    p.add_argument("--baseline-settle-sec", type=float, default=2.0,
                   help="Seconds to drive under baseline config before snapshotting risk context R")
    p.add_argument("--washout-sec", type=float, default=2.0,
                   help="Pause after reverting to baseline, before next probe")
    p.add_argument("--probe-goal-distance", type=float, default=2.5,
                   help="Distance (m) of short local goToPose goal straight ahead")
    p.add_argument("--constriction", type=str, default=None,
                   choices=["true", "enriched", "rooms"],
                   help="Pose generation mode: 'true' for constriction-crossing, "
                        "'enriched' for 3-stratum enriched sampling, 'rooms' for v2 room topology")
    p.add_argument("--doorway-weight", type=float, default=0.40,
                   help="Weight for doorway stratum in v2 rooms pose sampling")
    p.add_argument("--wall-weight", type=float, default=0.25,
                   help="Weight for wall-following stratum in v2 rooms pose sampling")
    p.add_argument("--open-weight", type=float, default=0.35,
                   help="Weight for open control stratum in v2 rooms pose sampling")
    # --- pose-pool shape (campaign 2) ---
    p.add_argument("--spatial-balance", action="store_true",
                   help="Target equal start/goal quotas over --balance-cell squares "
                        "of valid space (route stratum fills rooms a straight pair "
                        "cannot reach)")
    p.add_argument("--balance-cell", type=float, default=3.0,
                   help="Side (m) of the spatial-balance squares")
    p.add_argument("--min-region-area", type=float, default=1.0,
                   help="Valid start area (m^2) a square needs to get a quota")
    p.add_argument("--tight-fraction", type=float, default=0.0,
                   help="Share of the pool whose route narrows into --tight-gap-band")
    p.add_argument("--tight-gap-band", type=float, nargs=2, default=[0.65, 1.20],
                   metavar=("LO", "HI"), help="Free-width band (m) for tight routes")
    p.add_argument("--corridor-fraction", type=float, default=0.0,
                   help="Share of the pool whose route runs through a corridor "
                        "(passage < 1.8 m) after the earliest treatment point")
    p.add_argument("--corridor-pose-clearance", type=float, default=0.50,
                   help="Start/goal clearance (m) allowed inside corridors for the "
                        "corridor share; probes spawn tucked (radius 0.325)")
    p.add_argument("--corridor-min-len", type=float, default=2.0,
                   help="Route metres required inside a corridor")
    p.add_argument("--corridor-after", type=float, default=2.0,
                   help="Corridor metres only count after this route distance (m)")
    p.add_argument("--label-difficulty", action="store_true",
                   help="Attach easy/difficult labels and mechanism tags to each pair")
    p.add_argument("--dynamic-obstacle-xy", type=float, nargs=2, action="append",
                   default=None, metavar=("X", "Y"),
                   help="Blocking position of a small moving obstacle (repeatable)")
    p.add_argument("--keepout-sweep", type=float, nargs=7, action="append", default=None,
                   metavar=("SX", "SY", "BX", "BY", "YAW", "L", "W"),
                   help="moving obstacle: box L x W at heading YAW sliding (SX,SY)->(BX,BY); "
                        "no start/goal pose within --keepout-margin of its swept area "
                        "(repeatable)")
    p.add_argument("--keepout-margin", type=float, default=None,
                   help="keep-out margin around each sweep (default: --clearance)")
    p.add_argument("--dynamic-obstacle-seg", type=float, nargs=4, action="append",
                   default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Blocking extent of a long moving obstacle, e.g. a board "
                        "(repeatable); routes within 1.5 m are tagged 'dynamic'")
    p.add_argument("--no-randomize-arm", dest="randomize_arm",
                   action="store_false", default=True,
                   help="Hold the arm at the baseline's captured label for every "
                        "probe's c, instead of randomizing carry/tucked (default: randomize)")
    p.add_argument("--baseline-nudge", action="store_true", default=False,
                   help="Publish a short forward cmd_vel burst under baseline before "
                        "the R snapshot. Off by default — under the active-drive "
                        "probe design the robot is already moving via MPPI during "
                        "baseline-settle-sec, so R should be non-degenerate without "
                        "it; this is a documented escape hatch, not the primary mechanism.")
    p.add_argument("--xy-goal-tolerance", type=float, default=0.25,
                   help="Runner XY goal tolerance (default 0.25)")
    p.add_argument("--yaw-goal-tolerance", type=float, default=0.9,
                   help="Runner Yaw goal tolerance (default 0.9, matching Nav2's general_goal_checker)")
    p.add_argument("--no-check-goal-tolerance", dest="check_goal_tolerance",
                   action="store_false", default=True,
                   help="Skip checking goal tolerance against Nav2 live parameters")
    p.add_argument("--footprint-policy", choices=["instant", "track_arm", "hold_tucked"], default="instant",
                   help="Costmap footprint update policy: instant (default), track_arm, or hold_tucked")
    p.add_argument("--trigger-early-frac", type=float, default=1.0,
                   help="Fraction of admissible band for trigger: 1.0 = uniform random over entire drive (default 1.0)")
    p.add_argument("--apply-mode", choices=["time", "distance", "early_time"], default="time",
                   help="Treatment trigger mode: time (default, random wall clock delay) or distance")
    p.add_argument("--t-apply-min", type=float, default=4.0,
                   help="Minimum seconds after drive start before applying treatment (default 4.0)")
    p.add_argument("--t-apply-max", type=float, default=8.0,
                   help="Maximum seconds after drive start before applying treatment (default 8.0)")
    p.add_argument("--no-start-horizon-after-arm-move", dest="start_horizon_after_arm_move",
                   action="store_false", default=True,
                   help="Do not delay horizon_sec countdown until arm reaches carry pose")
    return p.parse_args()


def _config_from_yaml(raw: dict) -> OrchestratorConfig:
    """Build an OrchestratorConfig from a YAML dict, tolerating extra keys.

    The previous code did ``OrchestratorConfig(**yaml.safe_load(f))``, which
    raises TypeError on any key the dataclass does not declare. The shipped
    default_config.yaml contains several such keys (pal_config_dir,
    override_filename, nav_module_name, use_dynamic_reconfig,
    restart_nav_module, retry_on_nav_failure, risk_feature_topics), so loading
    it by --config could not work. Unknown keys are now reported and skipped
    rather than aborting the run, and the nested risk_feature_topics block is
    flattened onto the flat scan_topic/costmap fields it corresponds to.
    """
    import dataclasses

    raw = dict(raw)

    # Nested convenience block -> flat fields.
    topics = raw.pop("risk_feature_topics", None) or {}
    if isinstance(topics, dict):
        if "scan" in topics and "scan_topic" not in raw:
            raw["scan_topic"] = topics["scan"]
        if "risk_state" in topics and "risk_topic" not in raw:
            raw["risk_topic"] = topics["risk_state"]

    known = {f.name for f in dataclasses.fields(OrchestratorConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        logger.warning(
            "Ignoring %d unrecognised config key(s): %s. They are not fields on "
            "OrchestratorConfig — check for typos or stale settings.",
            len(unknown), ", ".join(unknown),
        )
    return OrchestratorConfig(**{k: v for k, v in raw.items() if k in known})


def main():
    args = parse_args()

    is_offline_mode = (
        args.visualize_only
        or args.generate_poses is not None
        or args.generate_configs is not None
    )
    # Resolve the output directory once: CLI, else the YAML's output_dir,
    # else ./rct_data. --output used to default to ./rct_data and therefore
    # ALWAYS override the YAML, so runs landed in the working directory.
    if args.output is None:
        _yaml_out = None
        if args.config:
            try:
                with open(args.config) as _f:
                    _yaml_out = (yaml.safe_load(_f) or {}).get("output_dir")
            except OSError:
                _yaml_out = None
        args.output = _yaml_out or "./rct_data"
    if not is_offline_mode:
        os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, "rct.log") if not is_offline_mode else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)

    # ── Offline modes (no ROS needed) ────────────────────────────────

    if is_offline_mode:
        # ── Config generation needs no map or ROS ───────────────
        if args.generate_configs is not None:
            from rct_collector.scripts.param_space import ParameterSpace

            os.makedirs(args.output, exist_ok=True)
            space = ParameterSpace(seed=args.seed)
            path = os.path.join(args.output, "presampled_configs.json")
            records = space.generate_and_save(
                args.generate_configs, path, method=args.sampling,
                oversample=args.oversample,
            )
            logger.info(
                f"Saved {len(records)} pre-generated configs to {path} "
                f"(method={args.sampling})"
            )

            report = space.distance_report([r["config"] for r in records])
            logger.info(
                "Coverage check (normalized nearest-neighbour distance): "
                f"min={report['min_nn']:.3f}, 1st-pctile={report['p1_nn']:.3f}, "
                f"median={report['median_nn']:.3f} over {report['dims']} dims"
            )
            if report["n_pairs_below_0.05"] > 0:
                logger.warning(
                    f"{report['n_pairs_below_0.05']} config(s) have a neighbour closer "
                    f"than 0.05 (normalized) — possible near-duplicates. Consider a "
                    f"different seed or fewer trials."
                )
            else:
                logger.info("No near-duplicate configs detected (all neighbours > 0.05). ✓")
            return

        if not args.map:
            logger.error("--map required for --visualize-only / --generate-poses")
            sys.exit(1)

        from rct_collector.scripts.pose_sampler import PoseSampler

        sampler = PoseSampler(
            map_yaml_path=args.map,
            obstacle_clearance_m=args.clearance,
            min_goal_distance=args.min_distance,
            max_goal_distance=args.max_distance,
            seed=args.seed,
        )
        # PoseSampler has no footprint-aware check; these flags were passed
        # through and crashed --generate-poses with a TypeError.
        if (args.footprint_clearance or args.carry_footprint
                or args.footprint_margin):
            logger.warning("--footprint-clearance/--carry-footprint/--footprint-margin "
                           "are not implemented by PoseSampler and are ignored; "
                           "use --clearance for pose clearance")
        sampler.load_map()
        if args.keepout_sweep:
            sampler.apply_keepout_sweeps(
                [tuple(v) for v in args.keepout_sweep],
                args.keepout_margin if args.keepout_margin is not None else args.clearance)
        os.makedirs(args.output, exist_ok=True)

        # Generate poses mode
        if args.generate_poses is not None:
            poses_path = os.path.join(args.output, "presampled_poses.json")
            constriction_val = args.constriction if args.constriction else False
            if constriction_val == "true":
                constriction_val = True
            ckwargs = {
                "strata_weights": {
                    "doorway": args.doorway_weight,
                    "wall": args.wall_weight,
                    "open": args.open_weight,
                }
            }
            poses = sampler.generate_and_save(
                args.generate_poses, poses_path,
                forward_distance_m=args.probe_goal_distance,
                constriction=constriction_val,
                constriction_kwargs=ckwargs,
                spatial_balance=args.spatial_balance,
                balance_cell_m=args.balance_cell,
                min_region_area_m2=args.min_region_area,
                tight_fraction=args.tight_fraction,
                tight_gap_band=tuple(args.tight_gap_band),
                corridor_fraction=args.corridor_fraction,
                corridor_pose_clearance_m=args.corridor_pose_clearance,
                corridor_min_m=args.corridor_min_len,
                corridor_after_m=args.corridor_after,
                label_difficulty=args.label_difficulty,
                dynamic_obstacle_xy=([tuple(v) for v in (args.dynamic_obstacle_xy or [])]
                                     + [tuple(v) for v in (args.dynamic_obstacle_seg or [])]
                                     or None),
            )

            # Also save a visualization with the generated poses
            vis_path = os.path.join(args.output, "presampled_poses.png")
            sampler.visualize_sampling_area(vis_path, presampled_poses=poses)
            logger.info(f"Visualization saved to {vis_path}")
            return

        # Visualize-only mode
        if args.visualize_only:
            vis_path = os.path.join(args.output, "sampling_area.png")

            if args.presampled_poses:
                # Visualize an existing poses file
                poses = PoseSampler.load_presampled(args.presampled_poses)
            else:
                # Generate poses, save them, then visualize
                poses_path = os.path.join(args.output, "presampled_poses.json")
                poses = sampler.generate_and_save(args.n_samples, poses_path)

            sampler.visualize_sampling_area(vis_path, presampled_poses=poses)
            logger.info(f"Saved visualization to {vis_path}")
            return

    # ── Online mode (ROS required) ───────────────────────────────────

    # Build config
    if args.config:
        with open(args.config) as f:
            config = _config_from_yaml(yaml.safe_load(f) or {})
        # Apply explicit CLI overrides if supplied
        if args.presampled_poses:
            config.presampled_poses_path = args.presampled_poses
        if args.presampled_configs:
            config.presampled_configs_path = args.presampled_configs
        if args.output:
            config.output_dir = args.output
        if args.map:
            config.map_yaml_path = args.map
        if args.n_probes:
            config.num_trials = args.n_probes
        config.resume = bool(args.resume)

    else:
        num_trials = args.n_probes if args.n_probes else 500
        world_name = args.world_name
        if not world_name and args.map:
            world_name = os.path.basename(os.path.dirname(os.path.abspath(args.map)))
            if not world_name or world_name in (".", "maps"):
                world_name = "pal_office"

        config = OrchestratorConfig(
            num_trials=num_trials,
            world_name=world_name or "pal_office",
            horizon_sec=args.horizon_sec,
            baseline_settle_sec=args.baseline_settle_sec,
            washout_sec=args.washout_sec,
            probe_forward_distance_m=args.probe_goal_distance,
            randomize_arm=args.randomize_arm,
            baseline_nudge=args.baseline_nudge,
            trial_timeout_sec=args.timeout,
            cooldown_sec=args.cooldown,
            map_yaml_path=args.map or "",
            obstacle_clearance_m=args.clearance,
            footprint_clearance_m=args.footprint_clearance,
            min_goal_distance=args.min_distance,
            max_goal_distance=args.max_distance,
            output_dir=args.output,
            scan_topic=args.scan_topic,
            odom_topic=args.odom_topic,
            gazebo_robot_model=args.robot_model,
            collision_threshold=args.collision_threshold,
            collect_risk_features=args.collect_risk,
            presampled_poses_path=args.presampled_poses,
            presampled_configs_path=args.presampled_configs,
            seed=args.seed,
            move_arm=args.move_arm,
            arm_control_mode=args.arm_control_mode,
            arm_action=args.arm_action,
            arm_move_time_sec=args.arm_move_time,
            arm_move_on_change_only=not args.arm_move_every_trial,
            reset_gazebo_between_trials=not args.no_gazebo_reset,
            xy_goal_tolerance=args.xy_goal_tolerance,
            yaw_goal_tolerance=args.yaw_goal_tolerance,
            check_goal_tolerance_against_nav2=args.check_goal_tolerance,
            footprint_policy=args.footprint_policy,
            trigger_early_frac=args.trigger_early_frac,
            apply_mode=args.apply_mode,
            t_apply_min_sec=args.t_apply_min,
            t_apply_max_sec=args.t_apply_max,
            start_horizon_after_arm_move=args.start_horizon_after_arm_move,
        )

    if not config.map_yaml_path and not args.resume:
        logger.error("--map required (or use --resume)")
        sys.exit(1)

    config.resume = bool(args.resume)
    if args.stop_at:
        config.stop_at_trial = int(args.stop_at)
    if args.rerun_trials:
        config.rerun_trials = [int(t) for t in args.rerun_trials.split(",") if t.strip()]
        config.rerun_kind = args.attempt_kind
        if not args.resume:
            logger.error("--rerun-trials needs --resume (it re-runs probes of an existing run)")
            sys.exit(1)
    orchestrator = RCTOrchestrator(config)
    orchestrator.initialize()
    orchestrator.run()
    reason = getattr(orchestrator, "exit_reason", "OK")
    code = EXIT_CODES.get(reason, 1)
    logger.info(f"exit_reason={reason} -> exit code {code}")
    # Every data file is closed by now (CSV/JSON via with-blocks, checkpoint
    # fsync'd). os._exit skips interpreter teardown, where rclpy/C++ threads
    # aborted with SIGABRT (exit 134) AFTER a clean run, which a supervisor
    # would have read as a crash.
    logging.shutdown()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()