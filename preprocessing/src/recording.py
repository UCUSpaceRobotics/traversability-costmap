"""First-pass extraction, retaining telemetry and TF but no decoded images."""
from dataclasses import dataclass

import numpy as np

from .base import (CAMERA_TOPIC, DEPTH_TOPIC, DYNAMIC_TF_TOPIC, RGB_TOPIC,
                   STATIC_TF_TOPIC, normalize_frame, tf_to_matrix)
from .transforms import TransformBuffer
from .validation import ValidationLimits, acquisition_ns, validate_states


@dataclass
class RecordingMetadata:
    transforms: TransformBuffer
    states: list
    intrinsics: np.ndarray
    camera_frame: str
    odom_frame: str
    base_frame: str
    image_streams: dict


def read_metadata(reader, odom_topic, base_frame='base_footprint', odom_frame='odom', limits=None,
                  state_range=None):
    limits = limits or ValidationLimits()
    chosen = {CAMERA_TOPIC, odom_topic, STATIC_TF_TOPIC, DYNAMIC_TF_TOPIC, RGB_TOPIC, DEPTH_TOPIC}
    available = {c.topic for c in reader.connections}
    missing = {CAMERA_TOPIC, odom_topic, RGB_TOPIC, DEPTH_TOPIC} - available
    if missing:
        raise ValueError(f'Missing required topics: {sorted(missing)}')
    transforms = TransformBuffer(); states = []; K = None; camera_frame = None
    streams = {topic: {'count': 0, 'first_ns': None, 'last_ns': None, 'max_gap_ns': 0}
               for topic in (RGB_TOPIC, DEPTH_TOPIC)}
    for conn, rx_time, raw in reader.messages(connections=[c for c in reader.connections if c.topic in chosen]):
        msg = reader.deserialize(raw, conn.msgtype)
        if conn.topic == CAMERA_TOPIC:
            matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
            frame = normalize_frame(msg.header.frame_id)
            if not frame or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
                raise ValueError('Invalid CameraInfo intrinsics/frame')
            if K is not None and (frame != camera_frame or not np.array_equal(K, matrix)):
                raise ValueError('Camera calibration/frame changed within recording')
            K, camera_frame = matrix, frame
        elif conn.topic in (STATIC_TF_TOPIC, DYNAMIC_TF_TOPIC):
            for tf in msg.transforms:
                stamp = acquisition_ns(tf, conn.topic) if conn.topic == DYNAMIC_TF_TOPIC else None
                transforms.add(tf.header.frame_id, tf.child_frame_id, tf_to_matrix(tf.transform), stamp)
        elif conn.topic == odom_topic:
            # RHE selects an already-validated continuous run. Bad telemetry
            # elsewhere in the bag must not be repaired or included in this clip.
            if state_range is not None:
                stamp = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
                if not state_range[0] <= stamp <= state_range[1]:
                    continue
            parent, child = normalize_frame(msg.header.frame_id), normalize_frame(msg.child_frame_id)
            if parent != normalize_frame(odom_frame) or child != normalize_frame(base_frame):
                raise ValueError(f'{odom_topic}: expected pose ^{odom_frame} T_{base_frame}, '
                                 f'got parent={parent!r}, child={child!r}')
            stamp = acquisition_ns(msg, odom_topic)
            if states and stamp <= states[-1][0]:
                raise ValueError('Odometry timestamps must be strictly increasing; duplicate/out-of-order stamp')
            pose, twist = msg.pose.pose, msg.twist.twist
            p, q = pose.position, pose.orientation
            # All velocity components must be finite, not just exported x/z.
            velocities = [twist.linear.x, twist.linear.y, twist.linear.z,
                          twist.angular.x, twist.angular.y, twist.angular.z]
            if not np.isfinite(velocities).all():
                raise ValueError('Odometry velocity components must all be finite')
            states.append((stamp, np.array([p.x, p.y, p.z]),
                           np.array([q.w, q.x, q.y, q.z]),
                           np.array([twist.linear.x, twist.angular.z])))
        else:
            stamp = acquisition_ns(msg, conn.topic)
            frame = normalize_frame(msg.header.frame_id)
            item = streams[conn.topic]
            if 'frame' in item and item['frame'] != frame:
                raise ValueError(f'{conn.topic}: image frame changed')
            item['frame'] = frame
            if item['last_ns'] is not None:
                gap = stamp - item['last_ns']
                if gap <= 0:
                    raise ValueError(f'{conn.topic}: duplicate/out-of-order acquisition timestamp')
                if gap > limits.max_frame_gap * 1e9:
                    raise ValueError(f'{conn.topic}: frame gap exceeds {limits.max_frame_gap}s')
                item['max_gap_ns'] = max(item['max_gap_ns'], gap)
            if item['first_ns'] is None:
                item['first_ns'] = stamp
            item['last_ns'] = stamp; item['count'] += 1
    if K is None or not states or any(not s['count'] for s in streams.values()):
        raise ValueError('Missing/empty camera calibration, odometry, RGB or depth stream')
    for topic, item in streams.items():
        if item['frame'] != camera_frame:
            raise ValueError(f'{topic}: expected CameraInfo optical frame {camera_frame!r}, got {item["frame"]!r}')
    states = validate_states(states, limits)
    transforms.finalize()
    return RecordingMetadata(transforms, states, K, camera_frame,
                             normalize_frame(odom_frame), normalize_frame(base_frame), streams)
