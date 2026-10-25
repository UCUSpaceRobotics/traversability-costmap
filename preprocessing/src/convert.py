"""Validated MCAP conversion, published only after the whole recording succeeds."""
import csv
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from .base import (DEPTH_TOPIC, RGB_TOPIC, interpolated_labels, matrix_csv,
                   normalize_frame, save_depth_png, vector_csv)
from .output import staged_recording
from .recording import read_metadata
from .synchronization import iter_rgbd_pairs
from .transforms import validate_transform
from .validation import ValidationLimits, format_timestamp, interpolate_state


def read_manifest(file: Path | None) -> dict:
    if file is None:
        return {}
    with file.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return {(r['session'], r['bag']): r['split'] for r in reader}


def convert_bag(bag: Path, output: Path, args, label_file: Path | None) -> dict:
    limits = ValidationLimits.from_args(args)
    with staged_recording(output) as stage:
        return _convert_bag(bag, stage, args, label_file, limits)


def _convert_bag(bag, output, args, label_file, limits):
    rgb_dir, depth_dir = output / 'rgb', output / 'depth'
    rgb_dir.mkdir(); depth_dir.mkdir()
    with AnyReader([bag]) as reader:
        metadata = read_metadata(reader, args.odom_topic, args.base_frame, args.odom_frame, limits)
        states, K = metadata.states, metadata.intrinsics
        camera_frame = normalize_frame(args.camera_frame or metadata.camera_frame)
        if camera_frame != metadata.camera_frame:
            raise ValueError('Camera override must match the recorded CameraInfo/image optical frame; '
                             'changing its name does not rotate image coordinates')
        if args.cam2base_json:
            override = validate_transform(json.loads(args.cam2base_json.read_text()))
            path = []
            print(f'[TF] Explicit calibrated JSON override: {args.cam2base_json}')
        else:
            override = None
            path = metadata.transforms.camera_path(camera_frame, args.base_frame, args.tf_policy)
            chain = ', '.join(f'{edge[0]}->{edge[1]}={source}' for edge, _, source in path)
            print(f'[TF] policy={args.tf_policy}; ^{args.base_frame} T_{camera_frame}: {chain}')
            if args.tf_policy == 'prefer-static' and any(
                    edge in metadata.transforms.conflicts for edge, _, _ in path):
                print('[WARNING] Static authority explicitly selected on overlapping camera edges; '
                      'recorded dynamic body attitude may be ignored. Verify physical calibration.')
        rows, sync_rows, image_states = [], [], {}
        stats = {}
        first_transform = None
        min_ts = states[0][0]
        max_ts = states[-1][0] - int(args.future_seconds * 1e9)
        pairs = iter_rgbd_pairs(reader, int(args.sync_tolerance_ms * 1e6),
                               int(limits.max_frame_gap * 1e9), stats, args.max_sync_buffer)
        eligible = 0
        for rt, rm, dt, dm in pairs:
            if rt < min_ts or rt >= max_ts:
                continue
            state, odom_offset = interpolate_state(states, rt, limits)
            # Depth must have valid motion support as well, even for nonzero pairing tolerance.
            interpolate_state(states, dt, limits)
            eligible += 1
            if (eligible - 1) % args.frame_step:
                continue
            T = override if override is not None else metadata.transforms.at(path, rt, args.max_tf_gap)
            if override is None and dt != rt:
                metadata.transforms.at(path, dt, args.max_tf_gap)
            color = cv2.imdecode(np.frombuffer(bytes(rm.data), np.uint8), cv2.IMREAD_COLOR)
            if color is None:
                raise ValueError(f'{bag.name}: RGB decode failed at {rt}')
            index = len(rows)
            rgb_name, depth_name = f'rgb/{index:06d}.png', f'depth/{index:06d}.png'
            if not cv2.imwrite(str(output / rgb_name), color):
                raise OSError(f'Could not save {rgb_name}')
            save_depth_png(dm, output / depth_name)
            depth = cv2.imread(str(output / depth_name), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape != color.shape[:2]:
                raise ValueError('Registered depth and RGB dimensions differ')
            if rows:
                previous_ns = sync_rows[-1][0]
                if rt - previous_ns > limits.max_frame_gap * 1e9:
                    raise ValueError('Exported frame gap exceeds --max-frame-gap (including frame-step)')
            rows.append((format_timestamp(rt), rgb_name, depth_name, matrix_csv(K), matrix_csv(T)))
            sync_rows.append((rt, dt, dt-rt, odom_offset))
            image_states[rt] = state
            if first_transform is None:
                first_transform = T
            if args.max_frames and len(rows) >= args.max_frames:
                break
        if len(rows) < args.min_frames:
            raise ValueError(f'{bag.name}: only {len(rows)} usable RGB-D frames (need {args.min_frames})')

        # Add interpolated image-time poses so the unchanged official loader can use
        # the same sampled poses instead of ceiling to a later raw odometry message.
        combined = {s[0]: s for s in states}
        combined.update(image_states)
        exported_states = sorted(combined.values(), key=lambda s: s[0])
        stamps = np.array([s[0] for s in exported_states], dtype=np.float64) / 1e9
        labels = interpolated_labels(stamps, label_file, args.max_label_gap) if label_file else None
        with (output / ('states.csv' if labels is not None else 'states_unlabeled.csv')).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'position', 'quaternion', 'action'] + (['traversability'] if labels is not None else []))
            for i, (ns, pos, quat, action) in enumerate(exported_states):
                fields = [format_timestamp(ns), vector_csv(pos), vector_csv(quat), vector_csv(action)]
                if labels is not None:
                    fields.append(vector_csv(labels[i]))
                writer.writerow(fields)
        with (output / 'images.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'image', 'depth', 'intrinsics', 'cam2base'])
            writer.writerows(rows)
        with (output / 'synchronization.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['rgb_timestamp_ns', 'depth_timestamp_ns', 'depth_minus_rgb_ns', 'nearest_odom_offset_ns'])
            writer.writerows(sync_rows)
        info = {
            'schema_version': 2, 'complete': True, 'source_bag': str(bag.resolve()),
            'odometry_topic': args.odom_topic, 'odom_frame': metadata.odom_frame,
            'base_frame': metadata.base_frame, 'camera_frame': camera_frame,
            'odometry_sampling': 'linear position/velocity + quaternion SLERP at RGB acquisition time',
            'action_semantics': 'measured odometry linear.x/angular.z; not commanded motion',
            'rgb_topic': RGB_TOPIC, 'depth_topic': DEPTH_TOPIC,
            'rgbd_pairs_written': len(rows), 'odometry_rows': len(states),
            'exported_state_rows': len(exported_states), 'validation_limits': asdict(limits),
            'sync_tolerance_ms': args.sync_tolerance_ms, 'max_sync_buffer': args.max_sync_buffer,
            'synchronization': stats, 'sync_statistics_scope': 'through final exported frame',
            'max_abs_sync_error_ns': max(abs(s[2]) for s in sync_rows),
            'max_nearest_odom_offset_ns': max(s[3] for s in sync_rows),
            'image_streams': metadata.image_streams,
            'tf_policy': 'calibrated-json-override' if override is not None else args.tf_policy,
            'tf_chain': metadata.transforms.describe(path),
            'tf_overlapping_edges': [list(e) for e in metadata.transforms.conflicts],
            'tf_conflict_summary': metadata.transforms.conflict_summary(),
            'max_tf_gap': args.max_tf_gap, 'cam2base_per_frame': True,
            'cam2base_first_frame': first_transform.tolist(), 'intrinsics': K.tolist(),
            'frame_step': args.frame_step, 'max_frames': args.max_frames,
            'future_seconds': args.future_seconds,
            'depth_png_units': 'uint16 millimetres, 0 = invalid',
            'labels': str(label_file.resolve()) if label_file else None,
            'trainable': labels is not None,
        }
        (output / 'preprocessing.json').write_text(json.dumps(info, indent=2))
        return info
