"""Fail-fast numerical and timing checks shared by conversion and NPZ export."""
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class ValidationLimits:
    max_odom_gap: float = 0.2
    max_odom_offset: float = 0.075
    max_position_step: float = 0.5
    max_rotation_step: float = 0.5
    quaternion_tolerance: float = 0.001
    max_frame_gap: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and > 0')

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name, field.default)
                      for name, field in cls.__dataclass_fields__.items()})


def acquisition_ns(msg, topic):
    """Safety-critical streams require acquisition time, not receive-time fallback."""
    stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
    if stamp is None or not 0 <= int(stamp.nanosec) < 1_000_000_000:
        raise ValueError(f'{topic}: missing or invalid acquisition timestamp')
    result = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if result <= 0:
        raise ValueError(f'{topic}: acquisition timestamp must be positive')
    return result


def format_timestamp(ns):
    seconds, fraction = divmod(int(ns), 1_000_000_000)
    return f'{seconds}.{fraction:09d}'


def parse_timestamp_ns(text):
    value = Decimal(str(text)) * 1_000_000_000
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f'Invalid nanosecond timestamp: {text!r}')
    return int(value)


def validate_times(times, name, max_gap_s=None):
    times = np.asarray(times)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError(f'{name}: need positive, finite, nonempty timestamps')
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError(f'{name}: timestamps must be strictly increasing; duplicate/out-of-order stamp')
    if max_gap_s is not None and np.any(gaps > max_gap_s * 1e9):
        raise ValueError(f'{name}: timestamp gap exceeds {max_gap_s}s')


def validate_states(states, limits):
    times = np.array([s[0] for s in states], dtype=np.int64)
    validate_times(times, 'odometry', limits.max_odom_gap)
    p = np.asarray([s[1] for s in states], dtype=float)
    q = np.asarray([s[2] for s in states], dtype=float)
    a = np.asarray([s[3] for s in states], dtype=float)
    if p.shape != (len(times), 3) or q.shape != (len(times), 4) or a.shape != (len(times), 2):
        raise ValueError('Malformed odometry position/quaternion/action arrays')
    if not all(np.isfinite(x).all() for x in (p, q, a)):
        raise ValueError('Odometry position, quaternion and velocity must be finite')
    norms = np.linalg.norm(q, axis=1)
    if np.any(np.abs(norms - 1) > limits.quaternion_tolerance):
        raise ValueError('Odometry quaternion is not unit length within tolerance')
    # Normalize only the already-validated floating-point roundoff.
    q = q / norms[:, None]
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) > limits.max_position_step):
        raise ValueError('Odometry position discontinuity/reset exceeds max-position-step')
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    if len(q) > 1 and np.any((rotations[:-1].inv() * rotations[1:]).magnitude() > limits.max_rotation_step):
        raise ValueError('Odometry orientation discontinuity/reset exceeds max-rotation-step')
    return [(int(t), pp, qq, aa) for t, pp, qq, aa in zip(times, p, q, a)]


def interpolate_state(states, timestamp, limits):
    """Interpolate pose at acquisition time, with bounded support; never extrapolate."""
    # Callers validate the whole trajectory once before using this function.
    import bisect
    right = bisect.bisect_left(states, int(timestamp), key=lambda s: s[0])
    if right < len(states) and states[right][0] == timestamp:
        return states[right], 0
    if right == 0 or right == len(states):
        raise ValueError(f'Image at {timestamp} lies outside odometry coverage')
    left, end = states[right - 1], states[right]
    span = end[0] - left[0]
    nearest = min(timestamp - left[0], end[0] - timestamp)
    if span > limits.max_odom_gap * 1e9 or nearest > limits.max_odom_offset * 1e9:
        raise ValueError(f'Image at {timestamp}: odometry gap/offset exceeds configured limits')
    alpha = (timestamp - left[0]) / span
    rotations = Rotation.from_quat(np.array([left[2], end[2]])[:, [1, 2, 3, 0]])
    q = Slerp([0., 1.], rotations)(alpha).as_quat()[[3, 0, 1, 2]]
    return (int(timestamp), (1-alpha)*left[1] + alpha*end[1], q,
            (1-alpha)*left[3] + alpha*end[3]), int(nearest)


def validate_sequence(times, index, count, expected_count, max_gap_s):
    if index < 0:
        raise ValueError('Sequence index must be >= 0')
    if count != expected_count or count < 1:
        raise ValueError(f'Sequence must contain MODEL.TIME_LENGTH={expected_count} frames')
    validate_times(times, 'images')
    selected = times[index:index + count]
    if len(selected) != count:
        raise ValueError(f'Not enough image frames: requested {count} at index {index}')
    validate_times(selected, 'image sequence', max_gap_s)
    return selected
