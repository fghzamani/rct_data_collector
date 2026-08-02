# rct_data_collector
This repository is a ros2 package for collecting randomized controlled trials in gazebo.

## Usage

### Run a Single Test Trial
To run a single test trial, run the following command:

```bash
ros2 run rct_collector rct_collect   --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml   --trials 20   --output /home/forough/phd_projects/online_tuner/rct_data_smoke   --presampled-poses /home/forough/phd_projects/online_tuner/src/rct_data_collector/rct_data/presampled_poses.json   --move-arm --collect-risk   --timeout 150 --log-level DEBUG --seed 42

```

### Interactive Motion Visualizer
To visualize the recorded trial trajectory, motion, heading, and footprint over time on the map:

```bash
# Direct module execution:
python3 -m rct_collector.scripts.visualizer --json /path/to/trial.json

# Or after building with colcon / installing:
ros2 run rct_collector rct_visualize --json /path/to/trial.json
```
This opens a web-based dashboard with:
1. Time slider to step/play the robot motion over time.
2. File path input and browser to load any recorded `.json` trial.
3. Map rendering with start/goal poses, global planner path, executed path, and footprint polygon overlays.

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

## generate presampled poses randomly
python3 main.py   --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml   --generate-poses 100   --carry-footprint '[[-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641], [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138], [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]'   --output /home/forough/phd_projects/online_tuner/src/rct_data_collector/rct_data   --min-distance 10.0 --max-distance 20.0   --seed 42 



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

  ###Remember
  - xhost +local:root in your own system before runing the docker container to give acess to the GUI.


| `failure_reason` | Detected by | Answers |
|---|---|---|
| `PLANNING_FAILED_INITIAL` | pre-nav `getPath()` returned no path | goal unreachable from start |
| `BT_PLANNER_FAILED` | `ComputePathToPose` → FAILURE on `/behavior_tree_log` | planner failed mid-run |
| `BT_CONTROLLER_FAILED` | `FollowPath` → FAILURE on `/behavior_tree_log` | MPPI gave up |
| `BT_OTHER_FAILED` | any other BT node → FAILURE | container/unknown node |
| `STUCK_NO_PROGRESS` | stall detector in the recording loop | robot stopped moving |
| `RUNNER_TIMEOUT` | `timeout_sec` in the recording loop | ran out of clock |
| `COLLISION` | `/gazebo/collision` contact | physical hit |
| `RUNNER_EXCEPTION` | orchestrator `_record_failure` | harness crashed |
| `NONE` | no failure detected | trial ran to completion |
| `UNKNOWN` | Nav2 said FAILED, BT log was silent | diagnostic gap |
  
