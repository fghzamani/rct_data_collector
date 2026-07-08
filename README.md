# RCT Collector for Nav2 Parameter Tuning (PAL TIAGo)

Randomized Controlled Trial data collector for causal inference on Nav2 navigation parameters. Designed to run inside PAL Robotics' TIAGo Docker container (PAL OS 25.01).

## What This Does

For each trial:
1. **Samples random Nav2 parameters** (the causal intervention C_t) — controller frequency, velocity limits, inflation radii, cost scaling, etc.
2. **Samples random start and goal poses** from free space on your map
3. **Applies parameters** via PAL's layered config system (`~/.pal/config/99_rct_override.yaml`) and restarts the navigation module
4. **Teleports the robot** in Gazebo to the start pose, sets AMCL initial pose
5. **Runs navigation** via the `NavigateToPose` action and monitors for collision, timeout, success
6. **Logs outcomes** (Y_t): collision flag, travel time, path length, min obstacle distance, goal status
7. **Checkpoints** progress to disk so you can resume after interruption

The randomization of both parameters and poses ensures the resulting dataset satisfies the assumptions for causal identification (no confounding between configuration and environment).

## Why PAL Config Override Instead of Dynamic Reconfigure

The PAL OS 25.01 documentation specifies a layered YAML config system where files in `~/.pal/config/` override defaults. Files prefixed `90`–`99` are reserved for user overrides and load last. This package uses **both** approaches as belt-and-suspenders:

1. **`99_rct_override.yaml`** — Written before each trial. Guarantees the parameter is set correctly even for params that don't support dynamic reconfigure. Requires `pal module restart navigation` (~5-8s).
2. **`ros2 param set`** — Applied after restart as a backup. Faster, but not all Nav2 params take effect dynamically.

If you want speed over robustness (e.g., testing only dynamically-reconfigurable params), use `--no-restart` to skip the module restart.

## Package Structure

```
rct_collector/
├── rct_collector/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── orchestrator.py      # Main loop: trial sequencing, checkpoint/resume
│   ├── param_space.py       # Nav2 parameter definitions and sampling
│   ├── pose_sampler.py      # Random pose sampling from map free space
│   └── trial_runner.py      # ROS 2 node: initial pose, nav goal, collision detection
├── config/
│   └── default_config.yaml  # Example configuration
├── launch/
│   └── rct_collection.launch.py
├── package.xml
├── setup.py
└── setup.cfg
```

## Setup

### Inside the PAL Docker container:

```bash
# 1. Copy this package to your workspace
cp -r rct_collector ~/ros2_ws/src/

# 2. Install Python dependencies (scipy is needed for LHS sampling)
pip3 install scipy Pillow --break-system-packages

# 3. Build
cd ~/ros2_ws
colcon build --packages-select rct_collector
source install/setup.bash

# 4. Create PAL config directory if it doesn't exist
mkdir -p ~/.pal/config
```

### Volume mounting from host:

```bash
docker run -it --rm \
  -v /path/to/your/maps:/maps:ro \
  -v /path/to/rct_data:/rct_data \
  -v /path/to/rct_collector:/root/ros2_ws/src/rct_collector:ro \
  your_pal_tiago_image bash
```

## Usage

### 1. Visualize the sampling area first (dry run)

```bash
python3 -m rct_collector.main \
  --map /maps/my_world.yaml \
  --visualize-only \
  --output /rct_data
```

This saves `sampling_area.png` showing which areas the robot can start/end in. Check that the green region makes sense for your map.

### 2. Run the full collection

```bash
python3 -m rct_collector.main \
  --map /maps/my_world.yaml \
  --trials 3000 \
  --output /rct_data \
  --timeout 180 \
  --log-level INFO
```

### 3. Resume after interruption

```bash
python3 -m rct_collector.main \
  --map /maps/my_world.yaml \
  --resume \
  --output /rct_data
```

### 4. Using a config file

```bash
python3 -m rct_collector.main --config my_experiment.yaml
```

### 5. Fast mode (no module restart, dynamic reconfig only)

```bash
python3 -m rct_collector.main \
  --map /maps/my_world.yaml \
  --trials 100 \
  --no-restart \
  --cooldown 2.0
```

## Customizing the Parameter Space

Edit `rct_collector/param_space.py` to change which Nav2 parameters are randomized. The default space covers:

| Node | Parameter | Range | Notes |
|------|-----------|-------|-------|
| controller_server | controller_frequency | [5, 30] Hz | |
| controller_server | FollowPath.max_vel_x | [0.1, 0.8] m/s | TIAGo max ~0.8 |
| controller_server | FollowPath.max_vel_theta | [0.3, 1.5] rad/s | |
| controller_server | FollowPath.min_vel_x | [-0.1, 0.05] m/s | |
| planner_server | GridBased.tolerance | [0.1, 1.0] m | |
| local_costmap | inflation_radius | [0.2, 1.5] m | |
| local_costmap | cost_scaling_factor | [1, 15] | |
| global_costmap | inflation_radius | [0.2, 1.5] m | |
| global_costmap | cost_scaling_factor | [1, 15] | |

To add a parameter:

```python
from rct_collector.param_space import ParameterSpace, ParamDef

space = ParameterSpace()
space.add_param(ParamDef(
    node="controller_server",
    name="FollowPath.xy_goal_tolerance",
    param_type="continuous",
    low=0.05,
    high=0.5,
))
```

### MPPI-specific parameters

If your PAL image uses the MPPI controller, you may want to add:

```python
ParamDef(node="controller_server", name="FollowPath.iteration_count", param_type="discrete", low=1, high=5)
ParamDef(node="controller_server", name="FollowPath.batch_size", param_type="discrete", low=500, high=3000)
ParamDef(node="controller_server", name="FollowPath.time_steps", param_type="discrete", low=20, high=80)
```

## Output Format

### `rct_results.csv`

Each row is one trial. Columns:

```
trial_id, timestamp, start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw,
status, collision, travel_time_sec, path_length_m, goal_distance_remaining,
min_obstacle_distance, param__controller_server__controller_frequency,
param__controller_server__FollowPath.max_vel_x, ...
```

### `checkpoint.json`

```json
{
  "completed_trials": 1542,
  "timestamp": "2026-05-04T14:30:00",
  "config": { ... }
}
```

## Important Notes for Your RCT Design

1. **Don't retry failed trials.** A navigation failure (ABORTED, TIMEOUT) under a specific parameter configuration is valid outcome data — it tells your causal model that those parameters cause failures. The `retry_on_nav_failure` flag is `false` by default for this reason.

2. **Collision detection** uses the laser scan minimum range with a configurable threshold (default 0.15m). Adjust `collision_threshold` in `TrialRunnerNode` based on TIAGo's actual footprint radius.

3. **Module restart cost.** Each `pal module restart navigation` takes ~5-8s. For 3000 trials, that's ~4-7 extra hours. If your parameter set is fully dynamically reconfigurable, use `--no-restart` to skip this. Test with a small batch first to verify params actually change.

4. **Pose sampling bounds.** If your map has large unreachable areas (behind walls that Nav2 can't path through), set `sampling_bounds` in the config to restrict start/goal to the reachable region. Otherwise you'll get many ABORTED trials that aren't informative.

5. **Gazebo model name.** The teleport function assumes the robot model in Gazebo is named `"tiago"`. Check with `ros2 service call /gazebo/get_model_list ...` and update `_teleport_robot()` if different.

6. **Scan topic.** PAL TIAGo may publish laser data on `/scan_raw` (raw) or `/scan` (filtered). Check which topic is active in your container and update `risk_feature_topics` accordingly.

## Time Estimate

| Trials | With restart (~13s/trial) | Without restart (~65s/trial avg) |
|--------|---------------------------|----------------------------------|
| 100    | ~22 min                   | ~1.8 hours                       |
| 1000   | ~3.6 hours                | ~18 hours                        |
| 3000   | ~11 hours                 | ~54 hours                        |

The "with restart" estimate assumes ~5s restart + ~3s localization + ~5s cooldown overhead on top of actual navigation time. The "without restart" estimate assumes navigation dominates and averages ~60s per trial.
