---
pretty_name: Indomitus Rover Traversability Bags
license: mit
task_categories:
  - robotics
tags:
  - ros2
  - rosbag
  - mcap
  - rgb-d
  - zed2i
  - imu
  - odometry
  - traversability
size_categories:
  - n<1K
viewer: false
---

# Indomitus Rover Traversability Bags

ROS 2 bags (`.mcap`) recorded on the Indomitus rover of [UCU Space Robotics](https://github.com/UCUSpaceRobotics) while driving over different terrain: flat ground with bumps, single stones and a stony area. The rides were driven at several speeds.

The dataset is used to train a self-supervised traversability model. The camera images are the model input. The IMU and odometry provide the labels (slip and chassis vibration). Code for preprocessing training and evaluating model are in the [GitHub repo](https://github.com/UCUSpaceRobotics/traversability-costmap).

## Contents

Each bag contains:

| Topic | Type | Rate |
|---|---|---|
| `/rover/zed2i/rgb/image_rect_color/compressed` | `sensor_msgs/CompressedImage` | 7 Hz |
| `/rover/zed2i/depth/depth_registered/compressedDepth` | `sensor_msgs/CompressedImage` | 7 Hz |
| `/rover/zed2i/rgb/camera_info`, `/rover/zed2i/depth/camera_info` | `sensor_msgs/CameraInfo` | 7 Hz |
| `/rover/zed2i/imu/data` | `sensor_msgs/Imu` | 100 Hz |
| `/rover/zed2i/odom` | `nav_msgs/Odometry` (ZED visual odometry) | 20 Hz |
| `/rover/wheels/odom` | `nav_msgs/Odometry` (wheel odometry) | 100 Hz |
| `/rover/odom` | `nav_msgs/Odometry` (EKF fused) | 20 Hz |
| `/rover/joint_states` | `sensor_msgs/JointState` | 100 Hz |
| `/rover/cmd_vel` | `geometry_msgs/Twist` | 50 Hz, when driven |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | |

Recordings are grouped into **sessions**. One session is one recording day at one location, and it has a single camera configuration. Each session has a `session.yaml`.

## Structure

```
├── README.md
└── raw/
    └── <session>/                # <YYYY-MM-DD>_<location>, e.g. 2026-10-19_ucu-cinema
        ├── session.yaml          # date, location, lighting, surface description, notes
        ├── zed2i_record.yaml     # ZED 2i camera config used for this session
        └── <bag>/                # one ROS 2 bag
            ├── metadata.yaml
            └── <bag>_0.mcap
```

Bag names follow `<ride_type>_<speed>[_<index>]_<YYYYMMDD>_<HHMMSS>`, for example `mixed_05ms_2_20261019_160254` is the second `mixed` ride at a maximum of 0.5 m/s, started at 16:02:54 UTC. The ride types, the full naming convention and the train / val split of each bag are described in [data/README.md](https://github.com/UCUSpaceRobotics/traversability-costmap/blob/main/data/README.md) and [data/manifest.csv](https://github.com/UCUSpaceRobotics/traversability-costmap/blob/main/data/manifest.csv) on GitHub.

## Download

With the code repo, use [scripts/download_data.py](https://github.com/UCUSpaceRobotics/traversability-costmap/blob/main/scripts/download_data.py). It selects bags by split, session or ride type from the manifest.

Without it, use the `hf` CLI:

```bash
hf download ucu-space-robotics/traversability-costmap --repo-type dataset --local-dir data                                          # everything
hf download ucu-space-robotics/traversability-costmap --repo-type dataset --local-dir data --include "raw/2026-10-19_ucu-cinema/*"  # one session
```

## Reading the bags

The bags can be opened with [rosbags](https://pypi.org/project/rosbags/) without a ROS installation:

```python
from pathlib import Path
from rosbags.highlevel import AnyReader

with AnyReader([Path("data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249")]) as reader:
    conns = [c for c in reader.connections if c.topic == "/rover/zed2i/imu/data"]
    for conn, t, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        print(t, msg.linear_acceleration)
```

With ROS 2 installed, use `ros2 bag play <bag folder>` or `ros2 bag info <bag folder>`.
