# rct_data_collector
This repository is a ros2 package for collecting randomized controlled trials in gazebo.

## Usage

### Run a Single Test Trial
To run a single test trial, run the following command:

```bash
ros2 run rct_collector rct_collect   --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml   --trials 20   --output /home/forough/phd_projects/online_tuner/rct_data_smoke   --presampled-poses /home/forough/phd_projects/online_tuner/src/rct_data_collector/rct_data/presampled_poses.json   --move-arm --collect-risk   --timeout 150 --log-level DEBUG --seed 42

```

## Campaigns: A vs B

The collector runs two coexisting experiment shapes, selected with `--campaign
{A,B}` (default `B`). Every row in the CSV/JSON carries a `campaign` column so
downstream analysis never accidentally merges them — **A and B rows feed two
different models and are never merged row-wise.**

- **Campaign B** (default, unchanged): a full point-to-point mission per
  trial, one random Nav2/arm config sampled up front and held for the whole
  mission. Feeds an efficiency model `E[S | do(C), R]`.
- **Campaign A**: short "decision-point" collision probes. Each probe drives
  the robot a short distance under a **fixed baseline config**, freezes a
  snapshot of the risk-state `R`, then swaps to a **random config `c`**
  mid-drive and watches a short horizon for a Gazebo collision `Y^H`. Each
  probe writes exactly one `(c, R, Y^H)` row, feeding a collision model
  `P(Y^H | do(C), R)`.

### The one causal rule Campaign A is built around

`R` must be recorded **before** the random config is applied, while the robot
is under a **fixed baseline config** — never after `do(C=c)` has begun acting.
If `R` were read after `c` started acting, it would be a downstream effect of
the treatment (a mediator), not a pre-treatment covariate, and the collision
estimate would be invalid. This is why a probe cannot be embedded inside a
mission, and why every probe re-applies the baseline and re-drives from it
before each `R` snapshot — never reuses a mid-mission `R`.

The baseline is **not** a hand-picked profile: at startup the orchestrator
reads back Nav2's own live launch-default values for exactly the parameters
`param_space` controls (`_capture_baseline_config()`) and re-asserts that
*exact* captured baseline at the start of **every** probe, before the R
snapshot — not just relying on the end-of-previous-probe revert. Its specific
values don't matter for identification, only that it is fixed and
config-independent every time R is measured under it.

### Run Campaign A probes

```bash
ros2 run rct_collector rct_collect \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --campaign A --n-probes 500 \
  --horizon-sec 3.0 --baseline-settle-sec 2.0 --washout-sec 2.0 \
  --output ./rct_data_campaign_a --collect-risk --seed 42 --move-arm --presampled-poses /home/forough/phd_projects/online_tuner/src/rct_data_collector/rct_data/presampled_poses.json
```

`--no-randomize-arm` holds the arm at the baseline's captured label for every
probe instead of randomizing carry/tucked; `--baseline-nudge` adds a short
open-loop forward `cmd_vel` burst before the R snapshot (off by default — the
robot is already moving under Nav2 during `--baseline-settle-sec`, so `R`
should be non-degenerate without it).

Campaign A rows leave mission-only columns (`travel_time_sec`, path lengths,
`success_true`, replan history, etc.) blank, and their per-trial JSON is
deliberately lean — no time-series trajectory, since anything recorded after
`do(C=c)` begins is post-treatment.

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
cd ~/phd_projects/online_tuner/src/rct_data_collector/rct_collector/scripts/

python3 main.py \
  --map /opt/ros/humble/share/pal_maps/maps/pal_office/map.yaml \
  --generate-poses 15000 \
  --clearance 0.35 \
  --campaign A \
  --output /home/forough/phd_projects/online_tuner/rct_data_campaign_a_pal_office


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
| `BASELINE_COLLISION` | Campaign A only: contact during the pre-`R`-snapshot baseline settle | probe aborted, `R`/`y_h` never valid |
| `RUNNER_EXCEPTION` | orchestrator `_record_failure` | harness crashed |
| `NONE` | no failure detected | trial ran to completion |
| `UNKNOWN` | Nav2 said FAILED, BT log was silent | diagnostic gap |
  
