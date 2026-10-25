# UCU rover MCAP → WayFASTER preprocessing

A standalone CLI that prepares your UCU rover ROS2 `.mcap` recordings for:

1. **Pretrained inference**: RGB-D sequence, K, camera poses, depth-derived voxel cloud → `input.npz`.
2. **Fine-tuning**: exact original WayFASTER `rosbags.csv`, `images.csv`, `states.csv` layout **only if you provide calibrated μ/ν labels**.

The converter does **not fabricate** traversability training targets. Its unlabelled outputs contain `states_unlabeled.csv`, not a deceptive `states.csv` with constant targets. Do not train until the experimental μ/ν methodology has been decided and validated.

## Prerequisites

Use the existing ROCm / Python 3.11 `uv` environment. The converter itself does not import PyTorch.

```bash
cd ~/AI/RGB-D_Traversability_Map
uv pip install rosbags pyyaml
python -c 'import rosbags, cv2, numpy, scipy, yaml; print("OK")'
```

If an import is missing, use `uv pip install <package>` (OpenCV's import is `cv2`, its package is generally `opencv-python` or `opencv-python-headless`). Avoid reinstalling your working ROCm-enabled PyTorch.

Copy `preprocess_wayfaster.py` into `~/AI/RGB-D_Traversability_Map/`.

## Step 1 — Inspect one MCAP

```bash
python preprocess_wayfaster.py inspect \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249
```

Check that the input topic names are present, `CameraInfo` specifies the optical camera frame, `/rover/odom` is populated, and the printed **static TF** edges connect your requested base frame to the camera optical frame. By default, the target base frame is `base_footprint`. If `base_link` is the correct base and has the required static TF chain, pass `--base-frame base_link` to `convert`. Do **not** use an identity camera extrinsic unless it is your verified calibration.

If the transform path exists only in `/tf` (dynamic) and not in `/tf_static`, this version intentionally refuses to guess it. Provide a calibrated camera-to-base 4×4 JSON matrix with `--cam2base-json /path/to/matrix.json`, or extend the TF resolver after inspecting the actual TF tree.

## Step 2 — Convert one recording (no labels yet)

```bash
python preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --output wayfaster/dataset/rover \
  --split inference \
  --max-frames 30
```

Result:

```text
wayfaster/dataset/rover/
└── inference/
    ├── inference_bags.csv
    ├── rosbags.csv                # header-only until labels are supplied
    └── 2026-10-19_ucu-cinema__flat_05ms_1_20261019_152249/
        ├── images.csv
        ├── states_unlabeled.csv
        ├── preprocessing.json
        ├── rgb/000000.png ...
        └── depth/000000.png ...  # uint16 millimetres, zero = invalid
```

For a full export, remove `--max-frames 30`.

The default `--sync-tolerance-ms 30` pairs RGB/depth by **ROS header timestamps**. The converter reads MCAP twice (metadata, then streaming images) so it need not load every frame into RAM. The output tracks which sources and transforms were used.

## Step 3 — Generate inference tensors for a six-frame clip

```bash
python preprocess_wayfaster.py make-npz \
  --recording wayfaster/dataset/rover/inference/2026-10-19_ucu-cinema__flat_05ms_1_20261019_152249 \
  --config wayfaster/configs/temporal_model.yaml \
  --index 0 \
  --output wayfaster/prepared/flat_05ms_clip0.npz
```

`--index 1` selects the next overlapping window. `--pixel-step 1` matches the original dataloader's dense depth voxelization; a higher step is faster but produces approximate occupancy grids.

The `.npz` keys are:

| Array | Shape with the provided config |
|---|---|
| `color_img` | `[1,6,3,180,320]` |
| `pcloud` | `[1,6,15,100,100]` |
| `intrinsics` | `[1,6,3,3]` |
| `extrinsics` | `[1,6,4,4]` |
| `timestamps` | `[6]` in seconds |

`pcloud` and extrinsics use the indexing and geometry of the attached official dataloader. Camera intrinsics are resized for the model input **after** voxelization, as in the original loader.

Using your previously created runner:

```bash
cd wayfaster
ROCR_VISIBLE_DEVICES=0 python wayfaster_runner.py predict \
  --input-npz prepared/flat_05ms_clip0.npz \
  --output-dir results/flat_05ms_clip0
```

This path does not require `states.csv`, because it bypasses the original training-oriented dataset loader.

## Step 4 — Make a trainable dataset (requires labels)

For every recording, prepare a separate label file:

```text
my_labels/
├── flat_05ms_1_20261019_152249.csv
└── mixed_05ms_1_20261019_153122.csv
```

Each file needs **timestamp,mu,nu**:

```csv
timestamp,mu,nu
1790000000.0,0.6,0.8
1790000000.1,0.5,0.7
```

Those two rows are illustrative only. They are **not measured traversability labels**. The values must correspond to your selected method of turning rover motion, slip and vibration into meaningful μ and ν targets in `[0,1]`.

`--labels-dir` expects a label CSV per bag (basename `<bagname>.csv`). All odometry timestamps must be covered by nearby labels; the default interpolation tolerance is 0.11 s (`--max-label-gap`). Without matching supervision, the converter will not create a trainable `states.csv`.

Use the original `data/manifest.csv` from the *UCU GitHub repository* to respect recording-level train/validation splits. It is **not present in the Hugging Face raw data download**.

```bash
curl -fL -o data/manifest.csv \
  https://raw.githubusercontent.com/UCUSpaceRobotics/traversability-costmap/main/data/manifest.csv
```

**Currently, the 2026-10-19 manifest marks its usable bags as train and the test bag as excluded.** Before evaluating, reserve at least one entire recording as `val` in a copy of the manifest, or deliberately convert that recording separately into `val`. Avoid splitting overlapping frames from a single recording between train and validation.

```bash
python preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema \
  --output wayfaster/dataset/rover \
  --manifest data/manifest.csv \
  --labels-dir my_labels \
  --require-labels \
  --future-seconds 30
```

`--future-seconds 30` is tied to the original config's `HORIZON: 300` and `DT: 0.1` (30 seconds of future trajectory). If you change those training settings, adjust this accordingly.

The script creates a `states.csv` only for recordings with valid labels, and includes only those recordings in `<split>/rosbags.csv`.

Then point the WayFASTER config or runner at the dataset roots:

```text
wayfaster/dataset/rover/train/
wayfaster/dataset/rover/val/
```

Example training command using the earlier runner (after you have at least one labelled bag in each split):

```bash
cd wayfaster
ROCR_VISIBLE_DEVICES=0 python wayfaster_runner.py finetune \
  --train-data dataset/rover/train \
  --valid-data dataset/rover/val \
  --epochs 1 --batch-size 1 --limit-train-batches 5 --limit-val-batches 2
```

## Important limitations

- **Actual MCAP not available in this chat.** The script was tested on synthetic ROS-message streams and synthetic RGB/depth recordings, including an integration check using the attached official `Dataset` class. You must verify it on your exact TF tree, image format and timestamps.
- **TF:** Only fixed camera extrinsics from `/tf_static` are followed automatically. Use `--cam2base-json` when the correct transform cannot be inferred from static TF.
- **Labels:** The source data contains IMU/wheel/ZED information but not a validated WayFASTER μ/ν label mapping. No provisional values are silently inserted.
- **Timing:** Camera/odometry are matched by header times. Bad clock offsets or large sensor gaps require inspection before training.
- **Model behavior:** Successful shape checks do not prove calibration, physical meaning, or accuracy. Compare predictions to known terrain and validate the μ/ν interpretation before fine-tuning.
