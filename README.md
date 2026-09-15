# rct_data_collector

A ROS 2 package for collecting randomized controlled trial (RCT) decision probes in Gazebo.

This package collects short decision-point probes:
1. Drives the robot under a fixed baseline navigation configuration.
2. Freezes a snapshot of the local risk context $R_t$ (corridor width, obstacle clearance, TTC).
3. Swaps to a randomly sampled candidate configuration $c$ mid-drive.
4. Monitors a short time horizon ($H = 2.8\,\text{s}$) to record collision ($Y^H$), stall, and progress outcomes.

Outputs a dataset CSV (`rct_results.csv`) and per-probe JSON logs for offline causal model training.

---

## Quick Start Guide

### Run Probe Collection

```bash
ros2 run rct_collector rct_collect \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --n-probes 500 \
  --horizon-sec 3.0 --baseline-settle-sec 2.0 \
  --output ./rct_data \
  --presampled-poses ./rct_data/presampled_poses.json \
  --collect-risk --move-arm --seed 42
```

---

## Required Gazebo Plugins for Custom Worlds

Add both plugins as children of `<world>` in your SDF/World file:

```xml
<world name="your_world">
  <!-- 1. Teleport Control (/gazebo/set_entity_state) -->
  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
    <ros><namespace>/gazebo</namespace></ros>
    <update_rate>1.0</update_rate>
  </plugin>

  <!-- 2. Ground-Truth Collision Monitor (/gazebo/collision) -->
  <plugin name="collision_monitor" filename="libcollision_monitor.so">
    <ros><namespace>/gazebo</namespace></ros>
    <robot_name>tiago</robot_name>
    <ignore_models>ground_plane</ignore_models>
    <force_threshold>1.0</force_threshold>
    <publish_rate>50</publish_rate>
  </plugin>
</world>
```

---

## Generate Presampled Probe Poses

```bash
python3 rct_collector/scripts/main.py \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --generate-poses 15000 \
  --clearance 0.35 \
  --output ./rct_data
```

---

## Setup & Build

Build from your ROS 2 workspace root:

```bash
cd <your_workspace_root>
colcon build --packages-select gazebo_collision_monitor rct_collector
source install/setup.bash
```
