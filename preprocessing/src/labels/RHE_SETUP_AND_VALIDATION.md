# Odometry-based WayFASTER RHE

Run from the repository root. RHE is integrated into the existing converter:

```bash
python preprocessing/preprocess_wayfaster.py convert \
  --input data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --output /tmp/opencode/rover-rhe \
  --split train --rhe --future-seconds 30 --tf-policy prefer-dynamic
```

Use a **new output directory** for each experiment; existing recordings and RHE
diagnostics are not overwritten. The TF policy is an explicit experimental
choice, not proof of physical calibration. Add `--require-labels` to fail when a
bag cannot produce any training clips. Otherwise zero clips is a reported result,
not a reason to fill missing labels or relax thresholds.

For a session, replace `--input` with the session directory and supply
`--manifest data/manifest.csv`. All clips from one recording retain that recording's
split. Reserve a whole recording for validation: the current manifest has no val
bags. Never split clips from the same recording between train and validation.

## Estimator

Inputs are `/rover/cmd_vel` commands and `/rover/odom` position and ROS planar yaw.
The discrete model is:

```text
x[i+1]   = x[i]   + mu * integral(v_cmd) * cos(yaw[i])
y[i+1]   = y[i]   + mu * integral(v_cmd) * sin(yaw[i])
yaw[i+1] = yaw[i] + nu * integral(omega_cmd)
0 <= mu, nu <= 1
```

Command integrals use zero-order hold over the complete odometry interval. Body
heading is `atan2(R[1,0], R[0,0])`, including when roll/pitch are nonzero. The RHE
does **not** smooth, interpolate, sort or repair odometry. Only the existing
converter interpolates valid poses at image timestamps, as before.

Each full 3-second window has five unknowns: initial x/y/yaw and constant mu/nu.
Rolling out the equality-constrained dynamics eliminates trajectory variables
without changing that discrete constrained problem. SciPy solves weighted
**quadratic** least squares. Wrapped angular residuals are used. The first window
has no fabricated initial-state prior; subsequent accepted windows use the previous
rollout **at the new window's start**, plus the previous accepted parameters.
Priors are discarded across invalid/unobservable windows. There are no fixed
minimum-distance or minimum-turn gates: low-excitation windows are fitted too.
After fitting, a coefficient is accepted only if removing its column reduces the
numerical rank of the **measurement-only** Jacobian, accounting for the unknown
initial state and the other coefficient. Prior rows are excluded from this check:
a prior cannot make zero-input mu or nu observable. Unidentifiable coefficients
remain NaN and cannot enter training clips.

Unlike paper Eq. 2, this uses fused odometry rather than independent GNSS position
and IMU compass heading. There is no compass-offset parameter. Fused odometry
may contain wheel-derived errors; low residuals do not establish actual traction.
`Twist` commands have no header, so bag receive timestamps approximate application
times. Command latency/scaling and the physical interpretation of mu/nu still need
independent validation. The endpoint estimate represents a short preceding window,
not an instantaneous terrain measurement.

## Anomaly and window validity

The detector returns boolean measurement and adjacent-interval masks. It rejects
missing/invalid/nonincreasing odometry timestamps, >1-second header/receive clock
disagreement, nonfinite values, invalid quaternions, excessive gaps, absolute
twist/pose speeds, and pose increments inconsistent with integrated measured twist.
No comparison with commanded velocity is used to classify corruption.

Tunable converter options:

| Option | Default |
|---|---:|
| `--rhe-horizon-seconds` | 3 s |
| `--max-cmd-age` (anywhere in a held interval) | 0.15 s |
| `--max-odom-gap` | 0.2 s |
| `--rhe-max-speed` (twist and XY displacement / dt) | 1 m/s |
| `--rhe-max-yaw-rate` (twist and heading increment / dt) | 1.5 rad/s |
| `--rhe-max-position-error` (pose increment versus measured twist) | 0.05 m |
| `--rhe-max-heading-error` (heading increment versus measured twist) | 0.10 rad |

These are configurable integrity limits, not a calibrated corruption classifier.
Slow pose drift, plausible-but-wrong fused motion, and systematic scale errors may
pass. Inspect the affected measurements rather than tuning solely for clip yield.

The fit uses sigma_xy=0.1 m, sigma_yaw=0.15 rad, parameter-prior sigma=0.5,
and rejects RMS XY >0.25 m or yaw >0.30 rad. These prototype weights/gates and
the identifiability rule are recorded in diagnostics. Numerical identifiability
is **not** a confidence bound: very weak nonzero inputs may produce noisy or
prior-dominated estimates even when the Jacobian has full rank.

## Outputs and coverage

`<output>/rhe/<session>__<bag>/` contains:

- `observations.csv`: unchanged source-derived pose/twist, command integrals,
  measurement/interval validity and rejection reasons.
- `estimates.csv`: every endpoint, exact timestamp, mu/nu (NaN when unavailable),
  joint/per-coefficient validity and identifiability, measurement-Jacobian rank,
  fit diagnostics and rejection reasons.
- `summary.json`: anomaly counts/reasons, accepted/rejected windows, longest valid
  run, candidate clips, exported clip count/durations, and skipped short clips.

Every window overlapping a bad measurement, bad motion interval or stale command
is rejected. Estimation resumes only after an entire clean horizon. Consecutive
accepted endpoints form separate clips; no invalid endpoint is bridged. Labels
may be interpolated **only inside one accepted run**, never across invalid regions.

Clips use the existing `states.csv`, `images.csv`, RGB/depth PNG and `rosbags.csv`
layout under `<output>/<split>/<session>__<bag>__clipNNN/`. `action` remains measured
odometry linear.x/angular.z, not commanded motion. The trainer is unchanged.
All images are inside the accepted interval and at least 30 seconds before its
end. At least six images are required; thus every six-image history has complete
future trajectory/label support. No state or image crosses a rejected boundary.

```bash
python -m unittest discover -s tests -v
# Optional audit of an existing conversion, including its image-time states:
python preprocessing/src/labels/rhe_pipeline.py audit \
  --bag data/raw/2026-10-19_ucu-cinema/flat_05ms_1_20261019_152249 \
  --preprocessed wayfaster/dataset/rover/inference/2026-10-19_ucu-cinema__flat_05ms_1_20261019_152249
```

Tests cover known synthetic coefficients including low-excitation fits, rejection
of zero-input coefficients even with priors, yaw with roll/pitch, interior command
staleness, velocity/pose/heading/time anomalies, complete window rejection, clip
segmentation, the 30-second future requirement, and an end-to-end supervised CLI
export using synthetic telemetry. They do not establish physical label validity.
