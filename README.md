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

The model is [WayFASTER](https://github.com/matval/wayfaster) (a copy lives in [wayfaster/](wayfaster/)), a temporal network that takes a short sequence of RGB-D frames and predicts a 2-channel traversability map on a 100×100 grid in front of the rover. Input per clip: 6 RGB images (180×320), a depth-derived voxel cloud, camera intrinsics and camera-to-base extrinsics.

Pipeline:

1. **Record** ROS 2 bags on the rover ([scripts/record_bag.sh](scripts/record_bag.sh), [config/recording.yaml](config/recording.yaml)).
2. **Preprocess** the bags ([preprocessing/](preprocessing/)): synchronize images with odometry, resolve the TF chain, and export WayFASTER-style `rosbags.csv` / `images.csv` / `states.csv` plus per-clip `.npz` tensors.
3. **Self-supervised labels**: receding horizon estimation (RHE, as in WayFASTER) compares commanded motion (`/rover/cmd_vel`) with measured odometry over 3-second windows to estimate per-window slip factors (`mu`, `nu`), giving labels without manual annotation. See [RHE setup and limitations](preprocessing/src/labels/RHE_SETUP_AND_VALIDATION.md). These labels are experimental and still need physical validation.
4. **Inference / fine-tuning**: run the pretrained WayFASTER checkpoint with [scripts/infer_wayfaster.py](scripts/infer_wayfaster.py), then fine-tune on rover data.
5. **Costmap**: publish the predicted map as a 2-channel ROS 2 Nav2 costmap layer (planned, not implemented yet).

Details of the preprocessing CLI are in [preprocessing/PREPROCESS_WAYFASTER_README.md](preprocessing/PREPROCESS_WAYFASTER_README.md).


## Results



## Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # preprocessing deps; install PyTorch separately (CUDA/ROCm)

# 2. Data and pretrained weights
python scripts/download_data.py          # see data/README.md for options
scripts/download_weights.sh              # -> pretrained/wayfaster_best.ckpt

# 3. Convert one bag to a 6-frame inference clip
python preprocessing/preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --output wayfaster/dataset/rover --split inference --tf-policy prefer-dynamic
python preprocessing/preprocess_wayfaster.py make-npz \
  --recording wayfaster/dataset/rover/inference/2026-10-19_ucu-cinema__flat_05ms_1_20261019_152249 \
  --config wayfaster/configs/temporal_model.yaml --index 0 \
  --output wayfaster/prepared/flat_05ms_clip0.npz

# 4. Predict
python scripts/infer_wayfaster.py \
  --input-npz wayfaster/prepared/flat_05ms_clip0.npz --output-dir out/flat_05ms
```

`infer_wayfaster.py` expects weights-only checkpoint `wayfaster/pretrained/wayfaster_weights_only.pt` by default; pass `--checkpoint` to override. The TF policy is an explicit choice, see the preprocessing README before picking one.


## Repository structure

```text
config/          rover recording config
data/            manifest, data docs, raw bag metadata (.mcap files are git-ignored)
preprocessing/   MCAP -> WayFASTER converter, RHE label pipeline
scripts/         data/weights download, bag recording, bag reports, inference
tests/           unit tests for preprocessing and RHE
wayfaster/       vendored WayFASTER model, training code and configs
ai-context/      notes for AI assistants working on the repo
```


## Reproducibility

- Dependencies are pinned in [requirements.txt](requirements.txt); the PyTorch build (CUDA or ROCm) is installed separately.
- Data is versioned on Hugging Face, with the train/val split fixed in [data/manifest.csv](data/manifest.csv). Splits are per recording, never per clip.
- The converter never overwrites outputs and fails instead of fabricating missing data, so use a new output directory per experiment.
- Training seeds and final fine-tuning configs: _TBD_.


## Development

```bash
pip install pytest
pytest tests/
```

Per-bag quality reports: `python scripts/generate_bags_report.py` (output in the git-ignored `reports/`). Work happens on feature branches merged through pull requests.


## AI usage

AI tools were used for the following tasks:

- **Code generation:** AI-generated code was reviewed manually and tested before being merged.
- **Documentation:** AI helped draft and edit the README and other docs.
- **Theory consultations:** AI was used to explain concepts and methods relevant to the project.

## Contribution statement

| Member | Responsible for |
| --- | --- |
| Bohdan Zasymovych | Dataset gathering, ROS 2  |
| Mykyta Yagoda | Dataset preprocessing |
| Olena Yakovenko | Model training |

All members agree with the above distribution.



## License and citation

|  | License | Notes |
| --- | --- | --- |
| Code (this repo) | MIT — see [`LICENSE`](https://www.google.com/search?q=LICENSE&utm_source=gemini) | Free to reuse, including commercially |
| Original WayFASTER repo | MIT — see [`wayfaster/LICENSE`](https://github.com/matval/wayfaster) | Free to reuse, including commercially |