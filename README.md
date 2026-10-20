# Traversability Costmap

Self-supervised terrain traversability prediction and ROS 2 Nav2 costmap generation for the Indomitus rover.


## Demo



## Team

| Name | GitHub |
|---|---|
| Bohdan Zasymovych | [@BohdanZasymovych](https://github.com/BohdanZasymovych) |
| Mykyta Yagoda | [@yagodanr](https://github.com/yagodanr) |
| Olena Yakovenko | [@OlenaYakovenk0](https://github.com/OlenaYakovenk0) |



## Problem and motivation

Autonomous off-road navigation typically relies on geometric occupancy grids, which fail to capture the physical properties of the terrain—visually flat sand might cause severe wheel slip, while safe, tall grass is often falsely flagged as an impassable obstacle.This project asks a practical question: can we predict terrain traversability (expected slip and chassis vibration) directly from RGB-D images using self-supervised learning? Developed for the Indomitus rover to compete in the European Rover Challenge (ERC), the model learns from the rover's own IMU and odometry data, eliminating the need for manual labeling. The end goal is to generate a dynamic 2-channel costmap that integrates seamlessly into the ROS 2 Nav2 stack, allowing the rover to proactively avoid treacherous terrain and minimizing the need for manual operator intervention.   



## Data

The dataset is a set of ROS 2 bags (`.mcap`) recorded by teleoperating the Indomitus rover over different terrain: flat ground with bumps, single stones and a stony area. The rides were driven at several speeds. Each bag contains:

- ZED 2i RGB and depth images
- IMU readings
- odometry from the ZED, the wheels and the fused estimate
- wheel joint states
- velocity commands

The camera images are the model input. The IMU and odometry provide the self-supervised labels (slip and vibration).

The bags are hosted on Hugging Face at [ucu-space-robotics/traversability-costmap](https://huggingface.co/datasets/ucu-space-robotics/traversability-costmap). [data/manifest.csv](data/manifest.csv) lists each bag and its train/val split. See [data/README.md](data/README.md) for how to download the data, the directory layout, ride types, bag naming, and how to generate per-bag quality reports.


## Approach



## Results



## Quick start



## Repository structure



## Reproducibility



## Development



## AI usage



## Contribution statement

| Member | Responsible for |
| --- | --- |
| Bohdan Zasymovych | Dataset gathering |
| Mykyta Yagoda | `<specific directories/files, specific tasks>` |
| Olena Yakovenko | `<specific directories/files, specific tasks>` |

All members agree with the above distribution.



## License and citation

|  | License | Notes |
| --- | --- | --- |
| Code (this repo) | MIT — see [`LICENSE`](https://www.google.com/search?q=LICENSE&utm_source=gemini) | Free to reuse, including commercially |