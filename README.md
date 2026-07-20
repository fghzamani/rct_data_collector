# rct_data_collector
This repository is a ros2 package for collecting randomized controlled trials in gazebo.

## Usage

### Run a Single Test Trial
To run a single test trial, run the following command:

```bash
ros2 run rct_collector rct_collect \
  --map /opt/pal/alum/share/pal_maps/maps/pal_office/map.yaml \
  --trials 1 \
  --output /home/user/exchange/tiago_wr/rct_data_smoke \
  --presampled-poses /home/user/exchange/tiago_wr/src/rct_collector/rct_data/presampled_poses.json \
  --move-arm --collect-risk \
  --timeout 180 --log-level DEBUG
```

## Required Gazebo plugins for custom worlds

If you use your own world file, add **both** plugins as direct children of `<world>` (not inside any `<model>`). Without them, the robot won't reposition between trials and collisions won't be recorded.

```xml
<world name="your_world">

  <!-- ... your world contents ... -->

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

### Setup

1. Build and source the collision-monitor package (Gazebo must be launched from this sourced shell, or the `.so` won't be found):
   ```bash
   colcon build --packages-select gazebo_collision_monitor
   source install/setup.bash
   ```

2. Set `<robot_name>` to your robot's exact model name:
   ```bash
   ros2 service call /gazebo/get_model_list gazebo_msgs/srv/GetModelList '{}'
   ```

3. Add any non-collision static models (floor, map mesh) to `<ignore_models>`, space-separated. Only `ground_plane` is ignored by default.

4. Verify both are live after launch:
   ```bash
   ros2 service list | grep set_entity_state     # plugin 1
   ros2 topic echo /gazebo/collision --once       # plugin 2
   ```