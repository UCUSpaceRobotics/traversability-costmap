# Traversability Costmap

Self-supervised terrain traversability prediction and ROS 2 Nav2 costmap generation for the Indomitus rover.


## Demo



## Team

| Name | GitHub | Role |
|---|---|---|
| Bohdan Zasymovych | [@BohdanZasymovych](https://github.com/BohdanZasymovych) |
| <Team Member 2> | [@username](https://github.com/username) |
| <Team Member 3> | [@username](https://github.com/username) |



## Problem and motivation

Autonomous off-road navigation typically relies on geometric occupancy grids, which fail to capture the physical properties of the terrain—visually flat sand might cause severe wheel slip, while safe, tall grass is often falsely flagged as an impassable obstacle.This project asks a practical question: can we predict terrain traversability (expected slip and chassis vibration) directly from RGB-D images using self-supervised learning? Developed for the Indomitus rover to compete in the European Rover Challenge (ERC), the model learns from the rover's own IMU and odometry data, eliminating the need for manual labeling. The end goal is to generate a dynamic 2-channel costmap that integrates seamlessly into the ROS 2 Nav2 stack, allowing the rover to proactively avoid treacherous terrain and minimizing the need for manual operator intervention.   



## Data



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
| Bohdan Zasymovych | `<specific directories/files, specific tasks>` |
| <Name 2> | `<specific directories/files, specific tasks>` |
| <Name 3> | `<specific directories/files, specific tasks>` |

All members agree with the above distribution.



## License and citation

|  | License | Notes |
| --- | --- | --- |
| Code (this repo) | MIT — see [`LICENSE`](https://www.google.com/search?q=LICENSE&utm_source=gemini) | Free to reuse, including commercially |