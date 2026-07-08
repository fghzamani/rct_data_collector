#!/usr/bin/env python3
"""
Launch file for RCT data collection.

This launch file is an OPTIONAL alternative to running the orchestrator
directly via `python3 -m rct_collector.scripts.main`. Use it if you prefer
ros2 launch semantics or want to integrate with other launch files.

It assumes Nav2 (and Gazebo) are already running. With collect_risk:=true it
also starts the risk_state_node peer (publishes /risk_state at 10 Hz) and passes
--collect-risk to the orchestrator so trial_runner records the risk vector.

Usage:
    ros2 launch rct_collector rct_collection.launch.py \
        map_yaml:=/path/to/map.yaml \
        num_trials:=3000 \
        output_dir:=./rct_data \
        collect_risk:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    """Resolve args, then build the orchestrator command (appending
    --collect-risk when requested) and optionally start the risk node."""
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    collect_risk = _truthy(cfg("collect_risk"))

    orchestrator_cmd = [
        "python3", "-m", "rct_collector.scripts.main",
        "--map", cfg("map_yaml"),
        "--trials", cfg("num_trials"),
        "--output", cfg("output_dir"),
        "--timeout", cfg("timeout"),
    ]
    if collect_risk:
        orchestrator_cmd.append("--collect-risk")

    actions = []

    # Risk-state node — only when collecting risk features. Started before the
    # orchestrator so /risk_state is already flowing by the first trial.
    if collect_risk:
        actions.append(
            Node(
                package="rct_collector",
                executable="risk_state_node",
                name="optimized_risk_state_node",
                output="screen",
                parameters=[{
                    "scan_topic": cfg("scan_topic"),
                    "odom_topic": cfg("odom_topic"),
                    "pose_topic": cfg("pose_topic"),
                    "costmap_topic": cfg("costmap_topic"),
                    "path_topic": cfg("path_topic"),
                }],
            )
        )

    actions.append(ExecuteProcess(cmd=orchestrator_cmd, output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_yaml",
                description="Path to the map YAML file",
            ),
            DeclareLaunchArgument(
                "num_trials",
                default_value="3000",
                description="Number of RCT trials to run",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="./rct_data",
                description="Output directory for results",
            ),
            DeclareLaunchArgument(
                "timeout",
                default_value="180.0",
                description="Timeout per trial in seconds",
            ),
            DeclareLaunchArgument(
                "config",
                default_value="",
                description="Path to config YAML (overrides other args)",
            ),
            DeclareLaunchArgument(
                "collect_risk",
                default_value="false",
                description="Start risk_state_node and record the /risk_state vector",
            ),
            # Risk-node topic overrides (defaults match the TIAGo stack).
            DeclareLaunchArgument(
                "scan_topic", default_value="/scan_raw",
                description="LiDAR topic for the risk node",
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/mobile_base_controller/odom",
                description="Odometry topic for the risk node",
            ),
            DeclareLaunchArgument(
                "pose_topic", default_value="/amcl_pose",
                description="AMCL pose topic for the risk node",
            ),
            DeclareLaunchArgument(
                "costmap_topic", default_value="/local_costmap/costmap_raw",
                description="Local costmap topic for the risk node",
            ),
            DeclareLaunchArgument(
                "path_topic", default_value="/plan",
                description="Global plan topic for the risk node",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
