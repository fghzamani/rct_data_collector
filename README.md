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
