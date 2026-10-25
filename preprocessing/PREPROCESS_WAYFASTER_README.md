# UCU rover MCAP → WayFASTER preprocessing

A standalone CLI that prepares your UCU rover ROS2 `.mcap` recordings for:

1. **Pretrained inference**: RGB-D sequence, K, camera poses, depth-derived voxel cloud → `input.npz`.
2. **Fine-tuning**: original WayFASTER `rosbags.csv`, `images.csv`, `states.csv` layout with supplied labels or experimental odometry-based `--rhe` labels.

The converter does **not fabricate** traversability training targets. Without supplied labels or `--rhe`, it writes `states_unlabeled.csv`. RHE exports only continuous, fully labeled clips with six-image context and a complete 30-second future trajectory. See [RHE setup and limitations](src/labels/RHE_SETUP_AND_VALIDATION.md); experimental labels still require physical validation.

## Prerequisites

Use the existing ROCm / Python 3.11 `uv` environment. The converter itself does not import PyTorch.

```bash
cd ~/AI/RGB-D_Traversability_Map/traversability-costmap
uv pip install rosbags pyyaml
python -c 'import rosbags, cv2, numpy, scipy, yaml; print("OK")'
```

If an import is missing, use `uv pip install <package>` (OpenCV's import is `cv2`, its package is generally `opencv-python` or `opencv-python-headless`). Avoid reinstalling your working ROCm-enabled PyTorch.

Run the entrypoint from this repository; do not copy it away from its `src/` modules.

## Step 1 — Inspect one MCAP

```bash
python preprocessing/preprocess_wayfaster.py inspect \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249
```

Inspect the optical frame, odometry parent/child frames, both TF inventories, and the reported static/dynamic discrepancies. The default pose semantics must be `^odom T_base_footprint`, from `/rover/odom`. Selecting another base requires odometry that actually reports that base; changing a flag does not transform the pose.

### Transform authority and frame semantics

ROS TF `(parent, child)` stores `^parent T_child`, mapping child coordinates into the parent frame. The image points are in the CameraInfo optical frame. For this rover:

```text
^base_footprint T_camera(t)
    = ^base_footprint T_base_link(t) @ ^base_link T_camera(t)
```

The selected odometry is the pose of `base_footprint`, not `base_link` or the camera. Its roll/pitch are zero in the audited recording; body tilt must therefore come from the camera TF chain, not be discarded or applied twice. Per-image transforms are stored in `images.csv`.

The recording contains the same body edge in `/tf_static` and `/tf`, with substantially different orientations. This data alone does not establish which publisher is physically calibrated/authoritative. Verify the robot TF publishers, URDF/mount calibration, and body attitude against an independent physical reference before choosing:

- `--tf-policy require-explicit` (default): reject any camera-chain edge present in both TF sources.
- `--tf-policy prefer-dynamic`: use dynamic samples on overlapping edges; use static edges where only static exists.
- `--tf-policy prefer-static`: use static samples on overlapping edges; dynamic-only edges still require timestamped lookup. This is a comparison/diagnostic policy, not a claim that ignoring body attitude is valid.

Dynamic edges use linear translation interpolation and quaternion SLERP at each RGB acquisition timestamp. They require bracketing samples, with no extrapolation and a default maximum bracket of `--max-tf-gap 0.2` seconds. Identical TF retransmissions are deduplicated; contradictory same-time TF, changing static TF, multiple parents, and cycles are rejected.

`--cam2base-json` explicitly bypasses TF with a verified rigid 4×4 calibration. A constant override cannot represent changing body-to-footprint orientation unless it is genuinely constant in the selected base frame. Identity is never substituted for missing TF.

## Step 2 — Convert one recording (no labels yet)

```bash
python preprocessing/preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --output wayfaster/dataset/rover \
  --split inference \
  --tf-policy prefer-dynamic \
  --max-frames 30
```

The example explicitly selects dynamic authority for comparison; verify that choice before research use. Result:

```text
wayfaster/dataset/rover/
└── inference/
    ├── inference_bags.csv
    ├── rosbags.csv                # header-only until labels are supplied
    └── 2026-10-19_ucu-cinema__flat_05ms_1_20261019_152249/
        ├── images.csv
        ├── states_unlabeled.csv
        ├── preprocessing.json
        ├── synchronization.csv
        ├── rgb/000000.png ...
        └── depth/000000.png ...  # uint16 millimetres, zero = invalid
```

For a full export, remove `--max-frames 30`.

The default `--sync-tolerance-ms 0` requires **exact ROS acquisition timestamps** (2,074 exact pairs were found in the audited recording). Set a nonzero tolerance only after validating sensor timing; it permits a bounded residual timing/motion approximation, not motion compensation between RGB and depth. Matching is deliberately simple, first-compatible FIFO, not a nearest-neighbor optimizer.

The first pass validates the complete image streams and telemetry, even for `--max-frames`; duplicate/out-of-order timestamps and excessive gaps fail instead of being silently sorted or deduplicated. Acquisition timestamps must exist and be positive: safety-critical image/odometry/TF lookup does not substitute bag receive time. Synchronization buffers are bounded by `--max-sync-buffer 32`. Unmatched frames are counted, and each exported pair's exact nanosecond timestamps, signed depth-minus-RGB error, and odometry support distance are recorded in `synchronization.csv`.

### Odometry and continuity limits

Defaults (all configurable on the CLI):

| Option | Default | Meaning |
|---|---:|---|
| `--odom-frame` | `odom` | Required pose parent |
| `--base-frame` | `base_footprint` | Required pose child and camera target base |
| `--max-odom-gap` | 0.2 s | Maximum adjacent odometry interval |
| `--max-odom-offset` | 0.075 s | Maximum nearest supporting odometry distance |
| `--max-position-step` | 0.5 m | Maximum adjacent pose translation; rejects large jumps/resets |
| `--max-rotation-step` | 0.5 rad | Maximum adjacent orientation change |
| `--quaternion-tolerance` | 0.001 | Maximum absolute quaternion-norm error |
| `--max-frame-gap` | 0.5 s | Maximum raw-image/exported/selected-sequence interval |

Positions, all six twist components, and quaternions must be finite. Quaternions outside the norm tolerance are rejected; only small validated roundoff is normalized. Thresholds are integrity guards, not a validated rover dynamics model: review them for each recording regime. Increasing a threshold to hide a reset is not a correction.

Position/velocity are linearly interpolated and orientation uses SLERP at the RGB timestamp. Image-time states are inserted alongside measured odometry in the exported state CSV so the original loader can use the same poses. WayFASTER's final-frame yaw-aligned planar reference convention is retained, including its handling of vertical translation and Euler angles; this is not a new general 6-DoF model.

The exported `action` remains measured odometry `linear.x/angular.z`, **not commands**. Integrity checks do not establish fused-odometry independence from wheel measurements. RHE reads commands separately and masks anomalous measurements/intervals rather than repairing odometry.

### Safe publication and reruns

Conversion refuses existing recording destinations, including legacy/partial directories. Use a new output root for corrected TF or validation settings; old datasets are never silently upgraded or overwritten. Files are written into a private staging directory and published only after success. Ordinary failures remove staging; a process crash may leave hidden staging/lock files, but they are not published as recordings. Before manually cleaning a stale lock, verify that no writer is running.

Completed recordings carry `schema_version: 2` and `complete: true` in metadata. `make-npz` rejects older/incomplete exports and ambiguous labeled/unlabeled state files. NPZ publication is atomic and refuses existing destinations. Split CSV updates preserve previous entries and are individually atomic; they are not a multi-file dataset transaction. Back up datasets and use separate output roots for independent experiments.

## Step 3 — Generate inference tensors for a six-frame clip

```bash
python preprocessing/preprocess_wayfaster.py make-npz \
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
| `timestamps_ns` | `[6]` exact acquisition nanoseconds, int64 |

`pcloud` and extrinsics use the indexing and geometry of the attached official dataloader. Camera intrinsics are resized for the model input **after** voxelization, as in the original loader.

The index must be nonnegative, frame count must equal `MODEL.TIME_LENGTH`, timestamps must be strictly increasing, and the selected clip must satisfy `--max-frame-gap`. Recording boundaries are never crossed. NPZ export revalidates the full state trajectory before sampling poses.

Run the dedicated inference script from the repository root, using your existing
PyTorch/torchvision environment (including ROCm support on the AMD machine):

```bash
ROCR_VISIBLE_DEVICES=0 python scripts/infer_wayfaster.py \
  --input-npz wayfaster/prepared/flat_05ms_clip0.npz \
  --output-dir /tmp/opencode/flat_05ms-prediction
```

Defaults use `wayfaster/configs/temporal_model.yaml` and
`wayfaster/pretrained/wayfaster_weights_only.pt`. Override them with `--config`
and `--checkpoint`. `--device auto` uses CUDA/ROCm when available and otherwise
CPU; use `--device cpu` or `--device cuda` to select explicitly. Checkpoints are
loaded with `weights_only=True`, and all model weights must match exactly.

The new output directory contains `prediction.npz` (a `[1,2,100,100]`
`traversability` tensor plus `timestamps_ns`), `prediction.png` (last RGB and both
map channels), and `summary.json` (input, checkpoint, configuration, device and
forward-pass duration). Maps are shown in raw model index order, not a calibrated
navigation display. Existing output directories are refused, and failures do not
publish partial results.

This path does not require `states.csv`, because it bypasses the original training-oriented dataset loader. It runs the selected checkpoint without fine-tuning on rover labels.

## Step 4 — Make a trainable dataset (requires labels)

To estimate labels from recorded commands and fused odometry using the paper-inspired
reduced RHE, use the existing converter with `--rhe` instead of `--labels-dir`:

```bash
python preprocessing/preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --output /tmp/opencode/rover-rhe \
  --split train --rhe --future-seconds 30 --tf-policy prefer-dynamic
```

Use a new output directory. Diagnostics are saved even when no clip survives the
strict coverage requirement. Add `--require-labels` to make zero exported clips an
error. Low-excitation fits are accepted only when both coefficients are numerically
identifiable from measurements; priors do not establish observability. Do not fill
unknown coefficients or relax anomaly checks just to create training data. See the
[RHE guide](src/labels/RHE_SETUP_AND_VALIDATION.md) for masks, defaults and caveats.

Alternatively, supply externally generated, validated labels as follows.

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
python preprocessing/preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema \
  --output wayfaster/dataset/rover \
  --manifest data/manifest.csv \
  --labels-dir my_labels \
  --require-labels \
  --tf-policy prefer-dynamic \
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

- **Validation:** Run `python -m unittest discover -s tests -v`. Small real-bag exports and reference-loader comparisons have also been exercised; this does not validate physical calibration, all recordings, or navigation performance.
- **TF:** The authority choice remains an experimental calibration decision. It is recorded along with the chosen chain and source discrepancies. Do not use old static-only results as if they were corrected.
- **Labels:** The source data contains IMU/wheel/ZED information but not a validated WayFASTER μ/ν label mapping. No provisional values are silently inserted.
- **Timing:** Camera/odometry are matched by header times. Bad clock offsets or large sensor gaps require inspection before training.
- **Model behavior:** Successful shape checks do not prove calibration, physical meaning, or accuracy. Compare predictions to known terrain and validate the μ/ν interpretation before fine-tuning.
