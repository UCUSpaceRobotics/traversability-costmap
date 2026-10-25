#!/usr/bin/env python3
"""Odometry-based adaptation of WayFASTER Eq. 1/2, also used by convert --rhe.

Observations are fused odometry, not independent GNSS position/IMU compass heading.
Consequently there is no compass-offset parameter. Eliminating the trajectory via
the exact discrete dynamics leaves only initial x/y/yaw and constant mu/nu.
These are model-based experimental labels, not independently verified traction.

Uses the existing preprocessing dependencies (NumPy, SciPy, rosbags, OpenCV).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation import (ValidationLimits, format_timestamp, interpolate_state,
                            parse_timestamp_ns)
from src.base import normalize_frame

ODOM_TOPIC = '/rover/odom'
CMD_TOPIC = '/rover/cmd_vel'
WHEEL_TOPIC = '/rover/wheels/odom'
ZED_TOPIC = '/rover/zed2i/odom'


@dataclass(frozen=True)
class Odom:
    t_ns: int
    rx_ns: int
    frame: str
    child: str
    p: np.ndarray      # x, y, z in parent frame
    q: np.ndarray      # w, x, y, z
    measured: np.ndarray  # vx, wz in child frame (assumed ROS Odometry convention)
    used_receive_stamp: bool


@dataclass(frozen=True)
class Command:
    t_ns: int
    v: float
    omega: float
    timestamp_source: str


def header_ns(message, receive_ns: int) -> tuple[int, bool]:
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is not None:
        ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if ns > 0 and 0 <= int(stamp.nanosec) < 1_000_000_000:
            return ns, False
    return int(receive_ns), True


def get_odom(message, receive_ns: int) -> Odom:
    timestamp, fallback = header_ns(message, receive_ns)
    pos, quat = message.pose.pose.position, message.pose.pose.orientation
    twist = message.twist.twist
    return Odom(
        t_ns=timestamp,
        rx_ns=int(receive_ns),
        frame=str(message.header.frame_id).lstrip('/'),
        child=str(message.child_frame_id).lstrip('/'),
        p=np.asarray([pos.x, pos.y, pos.z], dtype=np.float64),
        q=np.asarray([quat.w, quat.x, quat.y, quat.z], dtype=np.float64),
        measured=np.asarray([twist.linear.x, twist.angular.z], dtype=np.float64),
        used_receive_stamp=fallback,
    )


def get_command(message, receive_ns: int) -> Command:
    # /rover/cmd_vel is geometry_msgs/Twist (no Header): bag receive time is
    # the only time available. For TwistStamped, use its header stamp instead.
    if hasattr(message, 'twist') and hasattr(message, 'header'):
        t, fallback = header_ns(message, receive_ns)
        twist = message.twist
        source = 'receive' if fallback else 'header'
    else:
        t = int(receive_ns)
        twist = message
        source = 'receive'
    return Command(t, float(twist.linear.x), float(twist.angular.z), source)


def read_bag(bag: Path, odom_topic: str = ODOM_TOPIC, cmd_topic: str = CMD_TOPIC):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError('Install MCAP support: uv pip install rosbags') from exc
    odom, commands, topic_info = [], [], Counter()
    topics = {odom_topic, cmd_topic, WHEEL_TOPIC, ZED_TOPIC}
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, receive_ns, raw in reader.messages(connections=conns):
            topic_info[conn.topic] += 1
            if conn.topic not in (odom_topic, cmd_topic):
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == odom_topic:
                odom.append(get_odom(msg, receive_ns))
            else:
                commands.append(get_command(msg, receive_ns))
    return odom, commands, dict(topic_info)


def _normalize_inputs(odom: list[Odom], commands: list[Command], parent: str, child: str):
    errors, warnings = [], []
    if len(odom) < 3:
        errors.append(f'Not enough {ODOM_TOPIC} messages ({len(odom)}).')
    if len(commands) < 2:
        errors.append(f'Not enough {CMD_TOPIC} messages ({len(commands)}).')
    for name, stream in [('odometry', odom), ('commands', commands)]:
        ts = np.array([x.t_ns for x in stream], dtype=np.int64)
        if len(ts) > 1:
            if np.any(np.diff(ts) < 0):
                errors.append(f'{name} acquisition timestamps are out of order.')
            if np.any(np.diff(ts) == 0):
                errors.append(f'{name} acquisition timestamps are duplicated.')
    pairs = Counter((x.frame, x.child) for x in odom)
    if pairs and pairs != {(parent, child): len(odom)}:
        errors.append(f'Odometry frame(s) {dict(pairs)}; expected parent={parent!r}, child={child!r}.')
    bad = []
    for i, state in enumerate(odom):
        norm = float(np.linalg.norm(state.q))
        if (not np.isfinite(state.p).all() or not np.isfinite(state.q).all()
                or not np.isfinite(state.measured).all() or norm < 1e-9 or abs(norm - 1) > 0.02):
            bad.append(i)
    if bad:
        errors.append(f'{len(bad)} odometry rows have non-finite values or invalid quaternions (first: {bad[:5]}).')
    if any(not np.isfinite([c.v, c.omega]).all() for c in commands):
        errors.append('Nonfinite command velocities found.')
    if odom:
        n_fallback = sum(x.used_receive_stamp for x in odom)
        if n_fallback:
            warnings.append(f'{n_fallback} odometry messages have missing header stamps; receive time used.')
    if commands and all(c.timestamp_source == 'receive' for c in commands):
        warnings.append('Commands have no acquisition header; their MCAP receive timestamps are used. Account for command latency.')
    if odom and commands:
        if odom[-1].t_ns < commands[0].t_ns or commands[-1].t_ns < odom[0].t_ns:
            errors.append('Odometry and command timestamps have no overlapping range; clock mismatch possible.')
        max_cmd = max(abs(c.v) for c in commands)
        max_measured = max(abs(x.measured[0]) for x in odom)
        if max_cmd > 0.05 and max_measured > 1.5 * max_cmd + 0.1:
            warnings.append(f'Peak measured vx={max_measured:.3f} m/s exceeds peak |cmd_vel.x|={max_cmd:.3f} m/s; investigate frames, scaling, and latency.')
    return errors, warnings


def csv_vector(text: str, size: int) -> np.ndarray:
    a = np.fromstring(str(text).replace('[', ' ').replace(']', ' '), sep=' ')
    if a.size != size:
        raise ValueError(f'Expected {size} vector elements, got {a.size}: {text}')
    return a


def compare_preprocessed(states_csv: Path, odom: list[Odom], tol_ns: int = 1000):
    """Check the existing converted CSV mirrors /rover/odom; action is measured twist."""
    if not states_csv.exists():
        raise FileNotFoundError(states_csv)
    if not odom:
        return {'error': 'No source odometry'}
    with states_csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ts = np.asarray([x.t_ns for x in odom], dtype=np.int64)
    states = [(x.t_ns, x.p, x.q, x.measured) for x in odom]
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        return {'error': 'Source odometry must have unique, increasing timestamps'}
    largest = {'position_m': 0.0, 'quaternion': 0.0, 'action': 0.0, 'timestamp_ns': 0}
    missing, parsed, interpolated, bad_rows = 0, 0, 0, []
    for i, row in enumerate(rows):
        try:
            t = parse_timestamp_ns(row['timestamp'])
            j = bisect.bisect_left(ts, t)
            js = [k for k in (j-1, j) if 0 <= k < len(ts)]
            if not js:
                missing += 1
                continue
            j = min(js, key=lambda k: abs(int(ts[k]) - t))
            ns_error = abs(int(ts[j]) - t)
            expected, _ = interpolate_state(states, t, ValidationLimits())
            interpolated += int(t != ts[j])
            dp = float(np.max(np.abs(csv_vector(row['position'], 3) - expected[1])))
            q = csv_vector(row['quaternion'], 4)
            expected_q = expected[2] / np.linalg.norm(expected[2])
            dq = float(min(np.max(np.abs(q - expected_q)), np.max(np.abs(q + expected_q))))
            da = float(np.max(np.abs(csv_vector(row['action'], 2) - expected[3])))
            largest['position_m'] = max(largest['position_m'], dp)
            largest['quaternion'] = max(largest['quaternion'], dq)
            largest['action'] = max(largest['action'], da)
            largest['timestamp_ns'] = max(largest['timestamp_ns'], ns_error if ns_error <= tol_ns else 0)
            if max(dp, dq, da) > 1e-6:
                bad_rows.append(i)
            parsed += 1
        except (ValueError, KeyError) as exc:
            bad_rows.append(f'{i}: {exc}')
    return {
        'csv_rows': len(rows), 'matched_to_raw_odom': parsed - interpolated,
        'matched_interpolated_states': interpolated,
        'unmatched_timestamps': missing, 'mismatching_rows': len(bad_rows),
        'first_mismatches': bad_rows[:5], 'maximum_absolute_error': largest,
        'note': 'states.csv action is /rover/odom.twist (MEASURED), not /rover/cmd_vel (COMMANDED).',
    }


def nearest_hold_commands(odom_t: np.ndarray, cmd_t: np.ndarray, cmd_v: np.ndarray, cmd_w: np.ndarray):
    """Zero-order hold command at each odometry sample (no future-command leakage)."""
    j = np.searchsorted(cmd_t, odom_t, side='right') - 1
    have = j >= 0
    safe = np.clip(j, 0, len(cmd_t)-1)
    age = np.full(len(odom_t), np.inf, dtype=np.float64)
    age[have] = (odom_t[have] - cmd_t[safe[have]]) / 1e9
    v = np.where(have, cmd_v[safe], np.nan)
    w = np.where(have, cmd_w[safe], np.nan)
    return v, w, age


def integrate_commands_zoh(start_ns: int, end_ns: int, cmd_t: np.ndarray, cmd_v: np.ndarray,
                           cmd_w: np.ndarray) -> tuple[float, float]:
    """Integrate [v, omega] over [start,end) in SI units, using bag-stamped command events."""
    if start_ns >= end_ns:
        raise ValueError('Expected positive time interval')
    j = int(np.searchsorted(cmd_t, start_ns, side='right') - 1)
    if j < 0:
        return math.nan, math.nan
    distance = heading = 0.0
    cursor = start_ns
    while cursor < end_ns:
        next_event = int(cmd_t[j+1]) if j + 1 < len(cmd_t) else end_ns
        stop = min(end_ns, next_event)
        dt = (stop - cursor) / 1e9
        distance += float(cmd_v[j]) * dt
        heading += float(cmd_w[j]) * dt
        cursor = stop
        if cursor < end_ns:
            j += 1
    return distance, heading


def planar_yaw(q):
    """ROS quaternion order w,x,y,z; heading of the body x axis in the XY plane."""
    w, x, y, z = np.asarray(q, dtype=float).T
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def detect_odom_anomalies(odom, max_odom_gap=.2, max_speed=1., max_yaw_rate=1.5,
                          max_position_error=.05, max_heading_error=.10):
    """Return boolean measurement/interval masks and reasons. Never repair data.

    Absolute speed limits and pose-versus-twist consistency are independent of
    commanded motion. Interval i connects measurements i and i+1.
    """
    reasons = [[] for _ in odom]
    intervals = [[] for _ in odom]
    yaw = planar_yaw([s.q for s in odom])
    for i, s in enumerate(odom):
        if (s.used_receive_stamp or s.t_ns <= 0 or
                abs(s.rx_ns - s.t_ns) > 1_000_000_000 or
                (i and s.t_ns <= odom[i-1].t_ns)):
            reasons[i].append('timestamp')
        if (not np.isfinite(np.r_[s.p, s.q, s.measured]).all() or
                abs(np.linalg.norm(s.q) - 1) > .001):
            reasons[i].append('invalid_values')
        if abs(s.measured[0]) > max_speed:
            reasons[i].append('velocity_spike')
        if abs(s.measured[1]) > max_yaw_rate:
            reasons[i].append('yaw_rate_spike')
    valid = np.array([not r for r in reasons], dtype=bool)
    for i in range(len(odom)-1):
        a, b = odom[i:i+2]
        dt = (b.t_ns - a.t_ns) / 1e9
        if not valid[i] or not valid[i+1]:
            intervals[i].append('invalid_measurement')
        if not 0 < dt <= max_odom_gap:
            intervals[i].append('odom_gap')
        if intervals[i]:
            continue
        delta = b.p[:2] - a.p[:2]
        heading_delta = float(wrap_angle(yaw[i+1] - yaw[i]))
        velocity = np.array([a.measured[0], b.measured[0]])
        expected_delta = dt/2 * np.sum(velocity[:, None] *
                                      np.column_stack((np.cos(yaw[i:i+2]), np.sin(yaw[i:i+2]))), axis=0)
        if (np.linalg.norm(delta) > max_speed * dt or
                np.linalg.norm(delta - expected_delta) > max_position_error):
            intervals[i].append('position_discontinuity')
        expected_heading = dt/2 * (a.measured[1] + b.measured[1])
        if (abs(heading_delta) > max_yaw_rate * dt or
                abs(wrap_angle(heading_delta - expected_heading)) > max_heading_error):
            intervals[i].append('heading_discontinuity')
    if intervals:
        intervals[-1].append('end_of_recording')
    interval_valid = np.array([not r for r in intervals], dtype=bool)
    return valid, interval_valid, reasons, intervals


def command_interval_fresh(start_ns, end_ns, cmd_t, max_age):
    """Check held-command age up to every event, not just odometry endpoints."""
    first = int(np.searchsorted(cmd_t, start_ns, side='right') - 1)
    if first < 0:
        return False
    stop = int(np.searchsorted(cmd_t, end_ns, side='left'))
    ends = np.r_[cmd_t[first+1:stop], end_ns]
    return bool(np.all(ends - cmd_t[first:stop] <= max_age * 1e9))


def produce_samples(odom: list[Odom], commands: list[Command], max_cmd_age: float,
                    max_odom_gap: float, **anomaly_limits) -> list[dict]:
    """Materialize time-aligned data including real command integrals between odom samples."""
    ot = np.asarray([s.t_ns for s in odom], dtype=np.int64)
    ct = np.asarray([c.t_ns for c in commands], dtype=np.int64)
    vv = np.asarray([c.v for c in commands], dtype=float)
    ww = np.asarray([c.omega for c in commands], dtype=float)
    v, w, age = nearest_hold_commands(ot, ct, vv, ww)
    yaw = planar_yaw([s.q for s in odom])
    valid, interval_valid, reasons, interval_reasons = detect_odom_anomalies(
        odom, max_odom_gap, **anomaly_limits)
    rows: list[dict] = []
    for i, s in enumerate(odom):
        dt = (int(ot[i+1]) - int(ot[i])) / 1e9 if i < len(odom)-1 else math.nan
        good_cmd = bool(0 <= age[i] <= max_cmd_age)
        if i < len(odom)-1:
            if dt > 0:
                forward, turn = integrate_commands_zoh(int(ot[i]), int(ot[i+1]), ct, vv, ww)
            else:
                forward = turn = math.nan
            fresh = dt > 0 and command_interval_fresh(int(ot[i]), int(ot[i+1]), ct, max_cmd_age)
            if not fresh:
                interval_reasons[i].append('stale_command')
            good_interval = bool(interval_valid[i] and fresh and np.isfinite([forward, turn]).all())
        else:
            forward = turn = math.nan
            good_interval = False
        rows.append(dict(
            timestamp_ns=int(ot[i]), timestamp_s=format_timestamp(ot[i]),
            x_m=float(s.p[0]), y_m=float(s.p[1]), z_m=float(s.p[2]),
            yaw_rad=float(yaw[i]), measured_v_mps=float(s.measured[0]),
            measured_omega_radps=float(s.measured[1]),
            cmd_v_mps=float(v[i]), cmd_omega_radps=float(w[i]),
            command_age_s=float(age[i]), dt_next_s=float(dt),
            drive_integral_next_m=float(forward), turn_integral_next_rad=float(turn),
            valid_command=int(good_cmd), valid_interval=int(good_interval),
            valid_measurement=int(valid[i]), valid_odom_interval=int(interval_valid[i]),
            measurement_reasons=';'.join(reasons[i]), interval_reasons=';'.join(interval_reasons[i]),
        ))
    return rows


def summarize(odom, commands, topics, comparison, samples, errors, warnings, args):
    out = {
        'bag': str(args.bag), 'topics': topics,
        'selected_odom_topic': args.odom_topic, 'selected_cmd_topic': args.cmd_topic,
        'expected_frames': {'parent': args.parent_frame, 'child': args.child_frame},
        'odom_frame_counts': {f'{a}->{b}': n for (a,b),n in Counter((x.frame,x.child) for x in odom).items()},
        'odom_count': len(odom), 'commands_count': len(commands),
        'odometry_header_fallback': sum(x.used_receive_stamp for x in odom),
        'cmd_timestamp_sources': dict(Counter(c.timestamp_source for c in commands)),
        'raw_csv_comparison': comparison,
        'errors': errors.copy(), 'warnings': warnings.copy(),
    }
    if len(odom)>1:
        gaps = np.diff([x.t_ns for x in odom]) / 1e9
        out['odom_period_s'] = {'median': float(np.median(gaps)), 'p99': float(np.percentile(gaps,99)), 'maximum': float(np.max(gaps))}
        head_lag = (np.asarray([x.rx_ns - x.t_ns for x in odom]) / 1e9)
        out['odom_receive_minus_header_s'] = {'median': float(np.median(head_lag)), 'p99': float(np.percentile(head_lag, 99))}
        if np.median(np.abs(head_lag)) > 1.0:
            errors.append('Odometry header and bag receive timestamps differ by >1 second in median; command timestamps are receive-time. Clock alignment must be fixed before RHE.')
        elif np.percentile(np.abs(head_lag),99) > 0.15:
            warnings.append('Large tail in odometry receive-minus-header latency; investigate timing before RHE.')
    if len(commands)>1:
        out['cmd_period_s'] = {'median': float(np.median(np.diff([x.t_ns for x in commands])/1e9))}
    if samples:
        ages = np.asarray([r['command_age_s'] for r in samples]); finite_ages = ages[np.isfinite(ages)]
        out['associated_command_age_s'] = {'median': float(np.median(finite_ages)) if len(finite_ages) else None,
                                          'p99': float(np.percentile(finite_ages,99)) if len(finite_ages) else None}
        out['valid_command_rows'] = sum(x['valid_command'] for x in samples)
        out['valid_intervals'] = sum(x['valid_interval'] for x in samples)
        cmd_v = np.asarray([r['cmd_v_mps'] for r in samples], dtype=float)
        meas_v = np.asarray([r['measured_v_mps'] for r in samples], dtype=float)
        command_valid = np.isfinite(cmd_v) & (ages >= 0) & (ages <= args.max_cmd_age)
        if command_valid.any():
            delta_v = meas_v[command_valid] - cmd_v[command_valid]
            out['measured_minus_command_v_mps'] = {
                'median': float(np.median(delta_v)),
                'p95_absolute': float(np.percentile(np.abs(delta_v),95)),
                'maximum_absolute': float(np.max(np.abs(delta_v)))}
        if len(odom)>1:
            valid_pairs = np.asarray([bool(r['valid_interval']) for r in samples[:-1]])
            if valid_pairs.any():
                p=np.asarray([s.p[:2] for s in odom], dtype=float)
                yaw=np.asarray([r['yaw_rad'] for r in samples])
                delta_p=np.diff(p,axis=0)
                dts=np.diff(np.asarray([s.t_ns for s in odom],dtype=np.int64))/1e9
                measured_forward_speed=(delta_p[:,0]*np.cos(yaw[:-1])+delta_p[:,1]*np.sin(yaw[:-1]))/dts
                v_error=measured_forward_speed[valid_pairs]-meas_v[:-1][valid_pairs]
                out['pose_delta_forward_minus_odom_twist_mps']={
                    'median':float(np.median(v_error)),
                    'p95_absolute':float(np.percentile(np.abs(v_error),95))}
                if np.percentile(np.abs(v_error),95)>0.5:
                    warnings.append('Pose-derived forward speed and odometry twist disagree substantially; verify frame and odometry conventions before RHE.')
    if comparison:
        if comparison.get('error') or comparison.get('unmatched_timestamps') or comparison.get('mismatching_rows'):
            errors.append('Existing preprocessed states CSV does not match raw source odometry.')
    out['errors'] = errors
    out['warnings'] = warnings
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Refusing to write empty CSV')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_or_extract(args):
    odom, commands, topics = read_bag(args.bag, args.odom_topic, args.cmd_topic)
    errors, warnings = _normalize_inputs(odom, commands, args.parent_frame, args.child_frame)
    comparison = None
    if args.preprocessed:
        states = args.preprocessed
        if states.is_dir():
            has_labeled = (states / 'states.csv').exists()
            has_unlabeled = (states / 'states_unlabeled.csv').exists()
            if has_labeled and has_unlabeled:
                errors.append('Both states.csv and states_unlabeled.csv exist: stale supervision may be present; refuse ambiguous output.')
            states = states / ('states_unlabeled.csv' if has_unlabeled else 'states.csv')
        comparison = compare_preprocessed(states, odom)
    samples=[]
    if not errors:
        samples = produce_samples(odom, commands, args.max_cmd_age, args.max_odom_gap)
        if sum(r['valid_interval'] for r in samples) < args.min_valid_intervals:
            errors.append('Too few valid command/odometry intervals for estimation: '
                          f'{sum(r["valid_interval"] for r in samples)} < {args.min_valid_intervals}')
    report = summarize(odom, commands, topics, comparison, samples, errors, warnings, args)
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+'\n')
    if args.command == 'extract' and not errors:
        write_csv(args.output, samples)
        print(f'Saved {len(samples)} raw-derived observation rows to: {args.output}')
        print('NOTE: cmd_* fields are commanded; measured_* are from odometry; these are not traversability labels.')
    if errors:
        raise SystemExit(2)


def wrap_angle(error: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(error), np.cos(error))


def rollout(initial: np.ndarray, mu: float, nu: float, ds: np.ndarray, dphi: np.ndarray) -> np.ndarray:
    """Discrete WayFASTER Eq.1 dynamics; controls are interval-integrated cmd_vel."""
    yaw = initial[2] + nu * np.r_[0., np.cumsum(dphi)]
    return np.column_stack((initial[0] + mu*np.r_[0., np.cumsum(ds*np.cos(yaw[:-1]))],
                            initial[1] + mu*np.r_[0., np.cumsum(ds*np.sin(yaw[:-1]))], yaw))


def solve_window(rows: list[dict], previous: tuple[float,float] | None,
                 sigma_xy: float, sigma_yaw: float, prior_sigma: float,
                 max_rms_xy: float, max_rms_yaw: float, initial_prior=None) -> dict:
    t = np.asarray([r['timestamp_ns'] for r in rows], dtype=np.int64)
    observed = np.asarray([[r['x_m'],r['y_m'],r['yaw_rad']] for r in rows], dtype=float)
    ds = np.asarray([r['drive_integral_next_m'] for r in rows[:-1]], dtype=float)
    dphi = np.asarray([r['turn_integral_next_rad'] for r in rows[:-1]], dtype=float)
    excitation_linear = float(np.sum(np.abs(ds)))
    excitation_angular = float(np.sum(np.abs(dphi)))
    # Optimization over initial state and constant mu,nu. Measurement model is fused odom
    # rather than paper's GNSS/IMU measurement, and compass bias is omitted.
    initial = observed[0] if initial_prior is None else initial_prior
    guess = [*initial, *(previous if previous is not None else (0.9, 0.9))]
    lower = [-np.inf,-np.inf,-np.inf,0,0]
    upper = [np.inf,np.inf,np.inf,1,1]

    def residual(z):
        predicted = rollout(z[:3], z[3], z[4], ds, dphi)
        r = np.column_stack(((predicted[:, :2]-observed[:, :2])/sigma_xy,
                             wrap_angle(predicted[:,2]-observed[:,2])/sigma_yaw)).ravel()
        p = []
        # The first observation is already in r. Do not count it again as an
        # artificial arrival prior. Later priors come from the previous rollout
        # at this window's start, not from the previous window's terminal pose.
        if initial_prior is not None:
            p.extend((z[:2] - initial_prior[:2]) / sigma_xy)
            p.append(float(wrap_angle(z[2] - initial_prior[2])) / sigma_yaw)
        if previous is not None:
            p.extend((z[3:5] - previous) / prior_sigma)
        return np.concatenate((r, np.asarray(p)))

    result = least_squares(residual, np.asarray(guess), bounds=(lower,upper),
                           method='trf', loss='linear', max_nfev=200,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    pred = rollout(result.x[:3], result.x[3], result.x[4], ds, dphi)
    e_xy = np.linalg.norm(pred[:,:2]-observed[:,:2], axis=1)
    e_yaw = wrap_angle(pred[:,2]-observed[:,2])
    rms_xy = float(np.sqrt(np.mean(e_xy**2)))
    rms_yaw = float(np.sqrt(np.mean(e_yaw**2)))
    # Test identifiability using measurements ONLY. Including arrival/parameter
    # priors would falsely make a zero-input coefficient appear observable.
    # A coefficient is identifiable iff removing its column reduces the rank,
    # accounting for the three unknown initial-state components and its partner.
    jacobian = result.jac[:observed.size]
    tolerance = (np.linalg.norm(jacobian, ord=2) * max(jacobian.shape) *
                 np.finfo(float).eps)
    rank = int(np.linalg.matrix_rank(jacobian, tol=tolerance))
    mu_identifiable = rank > np.linalg.matrix_rank(jacobian[:, [0, 1, 2, 4]], tol=tolerance)
    nu_identifiable = rank > np.linalg.matrix_rank(jacobian[:, :4], tol=tolerance)
    quality = bool(result.success and rms_xy <= max_rms_xy and rms_yaw <= max_rms_yaw)
    mu_valid = bool(quality and mu_identifiable)
    nu_valid = bool(quality and nu_identifiable)
    return {
        'timestamp_ns':int(t[-1]), 'timestamp':format_timestamp(t[-1]),
        'mu':float(result.x[3]) if mu_valid else math.nan,
        'nu':float(result.x[4]) if nu_valid else math.nan,
        'mu_valid':int(mu_valid), 'nu_valid':int(nu_valid),
        'mu_identifiable':int(mu_identifiable), 'nu_identifiable':int(nu_identifiable),
        'measurement_jacobian_rank':rank,
        'linear_excitation_m':excitation_linear, 'angular_excitation_rad':excitation_angular,
        'rms_xy_m':rms_xy, 'rms_yaw_rad':rms_yaw,
        'optimizer_success':int(result.success), 'optimizer_message':str(result.message),
        'window_samples':len(rows), 'window_duration_s':float((t[-1]-t[0])/1e9),
        'valid':int(mu_valid and nu_valid),
        'rejection_reasons':';'.join(
            ([] if result.success else ['optimizer_failure']) +
            ([] if rms_xy <= max_rms_xy and rms_yaw <= max_rms_yaw else ['poor_fit']) +
            ([] if mu_identifiable else ['unidentifiable_mu']) +
            ([] if nu_identifiable else ['unidentifiable_nu'])),
        '_trajectory':pred,
    }


def estimate_rows(data, horizon_seconds=3., stride=1, sigma_xy=.1, sigma_yaw=.15,
                  prior_sigma=.5, max_rms_xy=.25, max_rms_yaw=.30):
    """One result per sampled endpoint, including every rejected window.

    A valid run restarts after any bad measurement/interval. This avoids sorting,
    repairing, or searching across invalid/out-of-order acquisition timestamps.
    """
    windows = []
    previous = None
    run_start = 0
    horizon_reason = 'incomplete_valid_horizon'
    ts = np.array([int(r['timestamp_ns']) for r in data], dtype=np.int64)
    for end, row in enumerate(data):
        reason = ''
        if not int(row.get('valid_measurement', 1)):
            run_start = end + 1
            reason = row.get('measurement_reasons', '') or 'invalid_measurement'
            horizon_reason = 'invalid_measurement_in_horizon'
        elif end and not int(data[end-1]['valid_interval']):
            run_start = end
            reason = data[end-1].get('interval_reasons', '') or 'invalid_interval'
            horizon_reason = 'invalid_interval_in_horizon'
        if reason:
            previous = None
        if end % stride:
            continue
        result = dict(timestamp_ns=int(ts[end]), timestamp=format_timestamp(ts[end]),
                      mu=math.nan, nu=math.nan, mu_valid=0, nu_valid=0, valid=0,
                      mu_identifiable=0, nu_identifiable=0, measurement_jacobian_rank=0,
                      rejection_reasons=reason or horizon_reason,
                      linear_excitation_m=math.nan, angular_excitation_rad=math.nan,
                      rms_xy_m=math.nan, rms_yaw_rad=math.nan, optimizer_success=0,
                      optimizer_message='', window_samples=0, window_duration_s=0.)
        if not reason and run_start <= end and ts[end] - ts[run_start] >= horizon_seconds * 1e9:
            start = run_start + int(np.searchsorted(ts[run_start:end+1],
                         int(ts[end]) - int(horizon_seconds*1e9), side='right')) - 1
            # An accepted prior must still cover the new initial timestamp.
            prior_state = None
            parameters = None
            if previous is not None and previous[0] <= start <= previous[1]:
                prior_state = previous[2][start - previous[0]]
                parameters = previous[3]
            result = solve_window(data[start:end+1], parameters, sigma_xy, sigma_yaw,
                                  prior_sigma, max_rms_xy, max_rms_yaw, prior_state)
            trajectory = result.pop('_trajectory')
            previous = (start, end, trajectory, (result['mu'], result['nu'])) if result['valid'] else None
        windows.append(result)
    return windows


def continuous_label_segments(estimates, future_seconds=30.):
    """Never bridge a rejected endpoint, including an isolated rejected sample."""
    segments, run = [], []
    for row in [*estimates, {'valid': 0}]:
        if row['valid']:
            run.append(row)
        else:
            if len(run) >= 2 and (run[-1]['timestamp_ns'] - run[0]['timestamp_ns']) / 1e9 >= future_seconds:
                segments.append(run)
            run = []
    return segments


def generate_rhe(bag, args, output):
    """Extract telemetry once, estimate, and save small per-recording diagnostics."""
    odom, commands, _ = read_bag(bag, args.odom_topic)
    if len(odom) < 3 or len(commands) < 2:
        raise ValueError('RHE needs odometry and /rover/cmd_vel')
    if any((s.frame, s.child) != (normalize_frame(args.odom_frame), normalize_frame(args.base_frame)) for s in odom):
        raise ValueError('RHE odometry parent/child frames do not match the selected base')
    ct = np.array([c.t_ns for c in commands], dtype=np.int64)
    if np.any(np.diff(ct) <= 0) or not np.isfinite([[c.v, c.omega] for c in commands]).all():
        raise ValueError('Command timestamps must increase strictly and controls must be finite')
    limits = dict(max_speed=args.rhe_max_speed, max_yaw_rate=args.rhe_max_yaw_rate,
                  max_position_error=args.rhe_max_position_error,
                  max_heading_error=args.rhe_max_heading_error)
    rows = produce_samples(odom, commands, args.max_cmd_age, args.max_odom_gap, **limits)
    estimates = estimate_rows(rows, horizon_seconds=args.rhe_horizon_seconds)
    segments = continuous_label_segments(estimates, args.future_seconds)
    all_segments = continuous_label_segments(estimates, 0.)
    summary = dict(
        source_bag=str(bag.resolve()), method='odometry-based WayFASTER adaptation; no GNSS/compass bias',
        settings=dict(**limits, max_cmd_age=args.max_cmd_age, max_odom_gap=args.max_odom_gap,
                      horizon_seconds=args.rhe_horizon_seconds, future_seconds=args.future_seconds,
                      sigma_xy=.1, sigma_yaw=.15, prior_sigma=.5,
                      identifiability='measurement-only Jacobian numerical rank; priors excluded',
                      max_rms_xy=.25, max_rms_yaw=.30),
        measurements=len(rows), invalid_measurements=sum(not r['valid_measurement'] for r in rows),
        invalid_motion_intervals=sum(not r['valid_odom_interval'] for r in rows[:-1]),
        anomaly_reasons=dict(Counter(reason for r in rows for reason in
                            (r['measurement_reasons'] + ';' + r['interval_reasons']).split(';')
                            if reason and reason not in ('end_of_recording', 'stale_command', 'invalid_measurement'))),
        accepted_windows=sum(r['valid'] for r in estimates),
        rejected_windows=sum(not r['valid'] for r in estimates),
        window_rejection_reasons=dict(Counter(reason for r in estimates
                                              for reason in r['rejection_reasons'].split(';') if reason)),
        valid_runs=len(all_segments),
        longest_valid_run_s=max(((s[-1]['timestamp_ns']-s[0]['timestamp_ns'])/1e9
                                 for s in all_segments), default=0.),
        candidate_clips=len(segments), exported_clip_count=0, exported_duration_s=0.,
        exported_clips=[], skipped_clips=[])
    from src.output import staged_recording
    with staged_recording(output) as stage:
        write_csv(stage / 'observations.csv', rows)
        write_csv(stage / 'estimates.csv', estimates)
        (stage / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return segments, summary


def estimate(args):
    with args.input.open(newline='') as f:
        data=list(csv.DictReader(f))
    if len(data)<3: raise ValueError('Need >=3 observations')
    expected=['timestamp_ns','x_m','y_m','yaw_rad','drive_integral_next_m',
              'turn_integral_next_rad','valid_interval']
    missing=[f for f in expected if f not in data[0]]
    if missing: raise ValueError(f'Missing columns: {missing}')
    for row in data:
        for key in expected + ['valid_measurement']:
            if key in row:
                row[key] = int(row[key]) if key in ('timestamp_ns', 'valid_interval', 'valid_measurement') else float(row[key])
    windows = estimate_rows(data, args.horizon_seconds, args.stride, args.sigma_xy, args.sigma_yaw,
                            args.prior_sigma, args.max_rms_xy, args.max_rms_yaw)
    write_csv(args.output,windows)
    print(f'Saved {len(windows)} experimental RHE windows to {args.output}')
    print(f'Both parameters observable and low-residual: {sum(r["mu_valid"] and r["nu_valid"] for r in windows)}')
    print('Experimental coefficients. Use convert --rhe for guarded continuous training clips.')



def plot(args):
    """Generate straightforward sensor-alignment diagnostics from extracted CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.input.open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('No extracted measurements')
    time_ns=np.asarray([int(r['timestamp_ns']) for r in rows],dtype=np.int64)
    time_s=(time_ns-time_ns[0])/1e9
    args.output_dir.mkdir(parents=True,exist_ok=True)

    def plot_timeseries(filename,title,ylabel,keys):
        fig,ax=plt.subplots(figsize=(12,4))
        for name,label in keys:
            v=np.asarray([float(r[name]) for r in rows])
            ax.plot(time_s,v,label=label,linewidth=.8)
        ax.set(xlabel='Time since first odometry sample (s)',ylabel=ylabel,title=title)
        ax.grid(True,alpha=.25);ax.legend(loc='best')
        fig.tight_layout();fig.savefig(args.output_dir/filename,dpi=150);plt.close(fig)

    plot_timeseries('linear_velocity.png','Commanded vs measured linear velocity','v (m/s)',
                    [('cmd_v_mps','/rover/cmd_vel'),('measured_v_mps','/rover/odom twist')])
    plot_timeseries('angular_velocity.png','Commanded vs measured angular velocity','omega (rad/s)',
                    [('cmd_omega_radps','/rover/cmd_vel'),('measured_omega_radps','/rover/odom twist')])
    fig,ax=plt.subplots(figsize=(12,3))
    age=np.asarray([float(r['command_age_s']) for r in rows]);age[~np.isfinite(age)]=np.nan
    ax.plot(time_s,age,linewidth=.8);ax.axhline(args.max_cmd_age,color='red',linestyle='--',label='Maximum command age')
    ax.set(xlabel='Time (s)',ylabel='Command age (s)',title='Time since latest cmd_vel')
    ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(args.output_dir/'command_age.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot([float(r['x_m']) for r in rows],[float(r['y_m']) for r in rows],linewidth=.8)
    ax.set(xlabel='Odometry x (m)',ylabel='Odometry y (m)',title='Observed rover trajectory')
    ax.axis('equal');ax.grid(True,alpha=.25);fig.tight_layout()
    fig.savefig(args.output_dir/'trajectory.png',dpi=150);plt.close(fig)
    print(f'Saved four diagnostic plots under {args.output_dir}')

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','extract'):
        a=sub.add_parser(name,help='Check bag and (optional) converted states; extract command/pose timeline')
        a.add_argument('--bag',type=Path,required=True,help='ROS 2 bag DIRECTORY containing metadata.yaml')
        a.add_argument('--preprocessed',type=Path,help='Existing converted recording DIRECTORY or states_unlabeled.csv')
        a.add_argument('--odom-topic',default=ODOM_TOPIC)
        a.add_argument('--cmd-topic',default=CMD_TOPIC)
        a.add_argument('--parent-frame',default='odom')
        a.add_argument('--child-frame',default='base_footprint')
        a.add_argument('--max-cmd-age',type=float,default=0.15,help='Maximum age of last cmd_vel at odom time, seconds')
        a.add_argument('--max-odom-gap',type=float,default=0.15,help='Maximum acceptable adjacent odometry interval, seconds')
        a.add_argument('--min-valid-intervals',type=int,default=10)
        a.add_argument('--report',type=Path,help='Write JSON report')
        if name=='extract':a.add_argument('--output',type=Path,required=True)
        a.set_defaults(func=audit_or_extract)
    v=sub.add_parser('plot',help='Create velocity, timing, and path diagnostic PNGs')
    v.add_argument('--input',type=Path,required=True)
    v.add_argument('--output-dir',type=Path,required=True)
    v.add_argument('--max-cmd-age',type=float,default=.15)
    v.set_defaults(func=plot)
    e=sub.add_parser('estimate',help='EXPERIMENTAL reduced moving-horizon fit; NOT paper-equivalent labels')
    e.add_argument('--input',type=Path,required=True)
    e.add_argument('--output',type=Path,required=True)
    e.add_argument('--horizon-seconds',type=float,default=3.0)
    e.add_argument('--stride',type=int,default=5)
    e.add_argument('--sigma-xy',type=float,default=0.1)
    e.add_argument('--sigma-yaw',type=float,default=0.15)
    e.add_argument('--prior-sigma',type=float,default=0.5)
    e.add_argument('--max-rms-xy',type=float,default=0.25)
    e.add_argument('--max-rms-yaw',type=float,default=0.30)
    e.set_defaults(func=estimate)
    return p


def main():
    args=parser().parse_args()
    if args.command=='estimate':
        for n in ('horizon_seconds','sigma_xy','sigma_yaw','prior_sigma','max_rms_xy','max_rms_yaw'):
            if not np.isfinite(getattr(args,n)) or getattr(args,n)<=0:raise SystemExit(f'--{n.replace("_","-")} must be finite and >0')
        if args.stride<1:raise SystemExit('stride must be >=1')
    elif args.command in ('audit','extract'):
        if args.max_cmd_age<=0 or args.max_odom_gap<=0:raise SystemExit('max ages and gaps must be positive')
        if not args.bag.is_dir():raise SystemExit(f'Bag directory missing: {args.bag}')
    args.func(args)


if __name__=='__main__':main()
