# rct_data_collector
A ROS 2 package for collecting randomized controlled trial (RCT) dataset probes and full navigation trials in Gazebo.

---

## Usage & Commands

### Run a Single Test Trial
To run a test trial, run:

```bash
ros2 run rct_collector rct_collect \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --trials 20 \
  --output ./rct_data_smoke \
  --presampled-poses ./rct_data/presampled_poses.json \
  --move-arm --collect-risk \
  --timeout 150 --log-level DEBUG --seed 42
```

---

## Campaigns: A vs B

The collector runs two coexisting experiment shapes, selected with `--campaign {A,B}` (default `B`). Every row in the CSV/JSON carries a `campaign` column so downstream analysis never accidentally merges them.

- **Campaign B** (default): a full point-to-point mission per trial, one random Nav2/arm config sampled up front and held for the whole mission. Feeds an efficiency model `E[S | do(C), R]`.
- **Campaign A**: short "decision-point" collision probes. Each probe drives the robot a short distance under a **fixed baseline config**, freezes a snapshot of the risk-state `R`, then swaps to a **random config `c`** mid-drive and watches a short horizon for a Gazebo collision `Y^H`. Each probe writes exactly one `(c, R, Y^H)` row, feeding a collision model `P(Y^H | do(C), R)`.

### Run Campaign A Probes

```bash
ros2 run rct_collector rct_collect \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --campaign A --n-probes 500 \
  --horizon-sec 3.0 --baseline-settle-sec 2.0 --washout-sec 2.0 \
  --output ./rct_data_campaign_a --collect-risk --seed 42 --move-arm --presampled-poses ./rct_data/presampled_poses.json
```

---

## Required Gazebo Plugins for Custom Worlds

If you use your own world file, add **both** plugins as direct children of `<world>` (not inside any `<model>`). Without them, the robot won't reposition between trials and collisions won't be recorded.

```xml
<world name="your_world">

  <!-- 1. Pose control: teleport robot to trial start pose (/gazebo/set_entity_state) -->
  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
    <ros>
      <namespace>/gazebo</namespace>
    </ros>
    <update_rate>1.0</update_rate>
  </plugin>

  <!-- 2. Ground-truth collisions: /gazebo/collision + /gazebo/collision_info -->
  <plugin name="collision_monitor" filename="libcollision_monitor.so">
    <ros>
      <namespace>/gazebo</namespace>
    </ros>
    <robot_name>tiago</robot_name>
    <ignore_models>ground_plane</ignore_models>
    <force_threshold>1.0</force_threshold>
    <publish_rate>50</publish_rate>
  </plugin>

</world>
```

---

## Generate Presampled Poses Randomly

```bash
python3 rct_collector/scripts/main.py \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --generate-poses 15000 \
  --clearance 0.35 \
  --campaign A \
  --output ./rct_data_campaign_a_pal_office
```

---

## Setup & Build

1. Build and source the collision-monitor and collector packages from your workspace root:
   ```bash
   cd <your_workspace_root>
   colcon build --packages-select gazebo_collision_monitor rct_collector
   source install/setup.bash
   ```
