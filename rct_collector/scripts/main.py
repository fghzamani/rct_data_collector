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
    p.add_argument("--output", type=str, default="./rct_data")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--cooldown", type=float, default=3.0)
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--min-distance", type=float, default=3.0)
    p.add_argument("--max-distance", type=float, default=15.0)
    p.add_argument("--config", type=str, help="YAML config file")
    p.add_argument("--resume", action="store_true")
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
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    is_offline_mode = (
        args.visualize_only
        or args.generate_poses is not None
        or args.generate_configs is not None
    )
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
        sampler.load_map()
        os.makedirs(args.output, exist_ok=True)

        # Generate poses mode
        if args.generate_poses is not None:
            poses_path = os.path.join(args.output, "presampled_poses.json")
            poses = sampler.generate_and_save(args.generate_poses, poses_path)

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
            config = OrchestratorConfig(**yaml.safe_load(f))
    else:
        config = OrchestratorConfig(
            num_trials=args.trials,
            trial_timeout_sec=args.timeout,
            cooldown_sec=args.cooldown,
            map_yaml_path=args.map or "",
            obstacle_clearance_m=args.clearance,
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
        )

    if not config.map_yaml_path and not args.resume:
        logger.error("--map required (or use --resume)")
        sys.exit(1)

    orchestrator = RCTOrchestrator(config)
    orchestrator.initialize()
    orchestrator.run()


if __name__ == "__main__":
    main()
