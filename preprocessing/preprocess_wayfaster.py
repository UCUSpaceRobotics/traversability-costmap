#!/usr/bin/env python3
"""Convert UCU rover ROS2 MCAP recordings to WayFASTER's original CSV/image format.

Commands:
    convert: MCAP -> images.csv + RGB/depth PNGs + states.csv (with labels or --rhe)
    make-npz: converted recording -> inference-ready WayFASTER tensors (.npz)
    inspect: show recorded topics and available TF frames

The original WayFASTER training loader expects traversability [mu, nu] supervision.
Supply --labels-dir with timestamp,mu,nu CSVs, or --rhe for experimental
odometry-based estimates and strictly continuous, fully supervised clips.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader

from src.base import ODOM_TOPIC, resolve_bags
from src.convert import InsufficientFramesError, convert_bag, read_manifest
from src.labels.rhe_pipeline import generate_rhe
from src.make_npz import decode_matrix, get_extrinsics_original, voxelize_mm
from src.output import atomic_write_text, save_npz_exclusive, exclusive_lock
from src.recording import read_metadata
from src.transforms import validate_transform
from src.validation import (ValidationLimits, parse_timestamp_ns, validate_sequence,
                            validate_states)


def run_convert(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.output / '.conversion.lock'):
        _run_convert(args)


def _run_convert(args) -> None:
    manifest = read_manifest(args.manifest)
    bags = resolve_bags(args.input)
    groups = {'train': [], 'val': [], 'inference': []}
    errors = []
    for bag in bags:
        session = bag.parent.name
        split = manifest.get((session, bag.name), args.split)
        if split == 'exclude' and not args.include_excluded:
            print(f'[SKIP] excluded: {bag.name}')
            continue
        if split == 'exclude':
            split = 'inference'
        if split not in groups:
            raise ValueError(f'Unknown manifest split {split!r} for {bag.name}')
        name = f'{session}__{bag.name}'
        label_file = ((args.labels_dir / f'{bag.name}.csv') if args.labels_dir else None)
        if label_file and not label_file.exists():
            label_file = None
        if args.require_labels and label_file is None and not args.rhe:
            errors.append(f'{bag.name}: missing label CSV; cannot train')
            print('[ERROR]', errors[-1])
            continue
        target = args.output / split / name
        print(f'\n[CONVERT] {bag} -> {target}')
        try:
            if args.rhe:
                diagnostic_dir = args.output / 'rhe' / name
                segments, summary = generate_rhe(bag, args, diagnostic_dir)
                try:
                    for i, segment in enumerate(segments):
                        clip_name = f'{name}__clip{i:03d}'
                        try:
                            info = convert_bag(bag, args.output / split / clip_name, args, None, segment)
                        except InsufficientFramesError as exc:
                            # A sufficiently long labeled run can still lack six
                            # images before its future cutoff. Nothing is published.
                            summary['skipped_clips'].append(dict(name=clip_name, reason=str(exc)))
                            continue
                        groups[split].append((clip_name, True))
                        summary['exported_clips'].append(dict(
                            name=clip_name, start=segment[0]['timestamp'], end=segment[-1]['timestamp'],
                            duration_s=(segment[-1]['timestamp_ns']-segment[0]['timestamp_ns'])/1e9,
                            image_frames=info['rgbd_pairs_written']))
                finally:
                    summary['exported_clip_count'] = len(summary['exported_clips'])
                    summary['exported_duration_s'] = sum(c['duration_s'] for c in summary['exported_clips'])
                    atomic_write_text(diagnostic_dir / 'summary.json', json.dumps(summary, indent=2) + '\n')
                    print('[RHE]', json.dumps(summary, indent=2))
                if args.require_labels and not summary['exported_clips']:
                    raise ValueError('No RHE clips have complete labels, image history and future coverage')
            else:
                info = convert_bag(bag, target, args, label_file)
                print(f"[OK] {info['rgbd_pairs_written']} synchronized pairs; "
                      f"{info['odometry_rows']} odom rows; "
                      f"training labels: {'yes' if info['trainable'] else 'NO'}")
                groups[split].append((name, info['trainable']))
        except Exception as exc:
            print(f'[ERROR] {bag.name}: {exc}', file=sys.stderr)
            errors.append(f'{bag.name}: {exc}')
    for split, items in groups.items():
        directory = args.output / split
        if not items:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for filename, names in (
                ('rosbags.csv', [name for name, ready in items if ready]),
                ('inference_bags.csv', [name for name, _ in items])):
            destination = directory / filename
            existing = []
            if destination.exists():
                with destination.open(newline='') as handle:
                    existing = [row['bag'] for row in csv.DictReader(handle)]
            text = io.StringIO(newline='')
            writer = csv.writer(text)
            writer.writerow(['bag'])
            writer.writerows([[name] for name in sorted(set(existing + names))])
            atomic_write_text(destination, text.getvalue())
    print(f'\nExported {sum(len(x) for x in groups.values())} recordings/clips from {len(bags)} bags. Errors: {len(errors)}')
    if errors:
        for err in errors:
            print(' -', err)
        raise SystemExit(1)
    if all(not label_ready for records in groups.values() for _, label_ready in records):
        if args.rhe:
            print('No training clips exported. Inspect the per-bag rhe/*/summary.json; no invalid labels were filled.')
        else:
            print('No supervised labels supplied: inference data is ready; training is NOT ready.')
            print('Create one <bag_name>.csv per bag with timestamp,mu,nu in --labels-dir to enable training.')


def run_inspect(args) -> None:
    bag = resolve_bags(args.input)[0]
    with AnyReader([bag]) as reader:
        print('Recording:', bag)
        print('Duration:', reader.duration / 1e9, 's')
        print('Message count:', reader.message_count)
        for c in reader.connections:
            print(f'  {c.topic:70} {c.msgtype}')
        metadata = read_metadata(reader, args.odom_topic, args.base_frame, args.odom_frame,
                                 ValidationLimits.from_args(args))
        print('Camera frame:', metadata.camera_frame)
        print('Camera K:', metadata.intrinsics)
        print('Odom pose semantics:', f'^{metadata.odom_frame} T_{metadata.base_frame}')
        print('Odom states:', len(metadata.states))
        print('Static TF edges:')
        for parent, child in sorted(metadata.transforms.static):
            print(f'  {parent} -> {child}')
        print('Dynamic TF edges:')
        for (parent, child), samples in sorted(metadata.transforms.dynamic.items()):
            print(f'  {parent} -> {child}: {len(samples)} samples')
        print('Overlapping static/dynamic edges (authority must be chosen explicitly):')
        for parent, child in metadata.transforms.conflicts:
            print(f'  {parent} -> {child}')
        for difference in metadata.transforms.conflict_summary():
            print('  Static/dynamic discrepancy:', difference)
        for policy in ('prefer-static', 'prefer-dynamic'):
            path = metadata.transforms.camera_path(metadata.camera_frame, args.base_frame, policy)
            print(f'Camera chain with {policy}:', metadata.transforms.describe(path))

def run_make_npz(args) -> None:
    record = args.recording
    limits = ValidationLimits.from_args(args)
    provenance_file = record / 'preprocessing.json'
    if not provenance_file.exists():
        raise ValueError('Missing preprocessing.json; reconvert with validated preprocessing')
    provenance = json.loads(provenance_file.read_text())
    if provenance.get('schema_version') != 2 or provenance.get('complete') is not True:
        raise ValueError('Legacy/incomplete recording: reconvert with validated preprocessing')
    if provenance.get('base_frame') != args.base_frame or provenance.get('odom_frame') != args.odom_frame:
        raise ValueError('Converted recording odometry/base frames do not match requested frames')
    with (record / 'images.csv').open(newline='') as f:
        image_rows = list(csv.DictReader(f))
    state_files = [p for p in (record / 'states.csv', record / 'states_unlabeled.csv') if p.exists()]
    if len(state_files) != 1:
        raise ValueError('Recording must contain exactly one labeled or unlabeled states CSV; stale/partial output')
    state_file = state_files[0]
    if (state_file.name == 'states.csv') != bool(provenance.get('trainable')):
        raise ValueError('Supervision metadata and states CSV disagree')
    with state_file.open(newline='') as f:
        state_rows = list(csv.DictReader(f))
    config = yaml.safe_load(args.config.read_text())['MODEL']
    T = args.frames or int(config['TIME_LENGTH'])
    all_image_ns = np.array([parse_timestamp_ns(r['timestamp']) for r in image_rows], dtype=np.int64)
    image_ns = validate_sequence(all_image_ns, args.index, T, int(config['TIME_LENGTH']), limits.max_frame_gap)
    selected = image_rows[args.index:args.index + T]
    state_ns = np.array([parse_timestamp_ns(r['timestamp']) for r in state_rows], dtype=np.int64)
    state_pos = np.asarray([decode_matrix(r['position'], (3,)) for r in state_rows])
    state_quat = np.asarray([decode_matrix(r['quaternion'], (4,)) for r in state_rows])
    state_actions = np.asarray([decode_matrix(r['action'], (2,)) for r in state_rows])
    validate_states(list(zip(state_ns, state_pos, state_quat, state_actions)), limits)
    image_time = image_ns.astype(np.float64) / 1e9
    K = [decode_matrix(r['intrinsics'], (3, 3)) for r in selected]
    if any(not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-8 for k in K):
        raise ValueError('Invalid camera intrinsics in images.csv')
    T_base_cam = [validate_transform(decode_matrix(r['cam2base'], (4, 4))) for r in selected]
    extrinsics = get_extrinsics_original(image_ns, state_ns, state_pos, state_quat, T_base_cam,
                                         limits, timestamps_ns=True)
    out_w, out_h = config['INPUT_SIZE']
    rgb_tensors, cloud_tensors, K_tensors = [], [], []
    for row, k, extr in zip(selected, K, extrinsics):
        rgb = cv2.imread(str(record / row['image']), cv2.IMREAD_COLOR)
        dep = cv2.imread(str(record / row['depth']), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None or dep.ndim != 2 or dep.dtype != np.uint16:
            raise ValueError(f'Missing RGB or uint16 depth at {row["timestamp"]}')
        if dep.shape != rgb.shape[:2]:
            raise ValueError('RGB and registered depth dimensions differ')
        # Original loader voxelizes full-resolution depth before resizing RGB.
        cloud_tensors.append(voxelize_mm(dep, k, extr, config['GRID_BOUNDS'], args.pixel_step))
        h, w = rgb.shape[:2]
        if (w, h) != (out_w, out_h):
            k = k.copy()
            k[0, :] *= out_w / w
            k[1, :] *= out_h / h
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        K_tensors.append(k.astype(np.float32))
    destination = args.output
    save_npz_exclusive(destination,
                        color_img=np.stack(rgb_tensors)[None],
                        pcloud=np.stack(cloud_tensors)[None],
                        intrinsics=np.stack(K_tensors)[None],
                        extrinsics=np.stack(extrinsics).astype(np.float32)[None],
                        timestamps=image_time, timestamps_ns=image_ns)
    print('Saved inference input:', destination)
    print('  color_img:', (1, T, 3, out_h, out_w))
    print('  pcloud:', (1, T, *cloud_tensors[0].shape))
    print('  intrinsics:', (1, T, 3, 3), 'extrinsics:', (1, T, 4, 4))
    if args.pixel_step > 1:
        print(f'Warning: pixel-step={args.pixel_step} makes the voxel map approximate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('inspect', help='Show topics, validated odometry and static/dynamic TF conflicts')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--odom-topic', default=ODOM_TOPIC)
    add_validation_arguments(p)
    p.set_defaults(func=run_inspect)

    p = commands.add_parser('convert', help='Convert MCAP bag(s) to WayFASTER CSV + images')
    p.add_argument('--input', type=Path, required=True, help='A bag folder, session folder, or all data/raw/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Original rover data/manifest.csv with train/val/exclude splits')
    p.add_argument('--split', choices=['train', 'val', 'inference'], default='inference',
                   help='Used if no manifest row exists; defaults to inference')
    p.add_argument('--include-excluded', action='store_true')
    p.add_argument('--labels-dir', type=Path, help='Directory of <bag_name>.csv files with timestamp,mu,nu')
    p.add_argument('--rhe', action='store_true', help='Estimate labels and export only continuous supervised clips')
    p.add_argument('--rhe-horizon-seconds', type=float, default=3.)
    p.add_argument('--max-cmd-age', type=float, default=.15, help='Maximum held command age anywhere in an interval')
    p.add_argument('--rhe-max-speed', type=float, default=1., help='Absolute observed linear/pose speed limit, m/s')
    p.add_argument('--rhe-max-yaw-rate', type=float, default=1.5, help='Absolute observed yaw/pose rate limit, rad/s')
    p.add_argument('--rhe-max-position-error', type=float, default=.05,
                   help='Maximum adjacent XY displacement minus integrated measured twist, m')
    p.add_argument('--rhe-max-heading-error', type=float, default=.10,
                   help='Maximum adjacent yaw change minus integrated measured yaw rate, rad')
    p.add_argument('--require-labels', action='store_true', help='Fail if any included bag lacks labels')
    p.add_argument('--max-label-gap', type=float, default=0.11)
    p.add_argument('--odom-topic', default=ODOM_TOPIC)
    add_validation_arguments(p)
    p.add_argument('--camera-frame', help='Override CameraInfo optical frame name if necessary')
    p.add_argument('--cam2base-json', type=Path, help='Explicit calibrated rigid camera-to-base override')
    p.add_argument('--tf-policy', choices=['require-explicit', 'prefer-dynamic', 'prefer-static'],
                   default='require-explicit', help='Authority for overlapping TF edges; default rejects conflicts')
    p.add_argument('--max-tf-gap', type=float, default=0.2, help='Maximum interpolation bracket in dynamic TF, seconds')
    p.add_argument('--sync-tolerance-ms', type=float, default=0.0,
                   help='Maximum RGB/depth acquisition-time offset; default requires exact stamps')
    p.add_argument('--max-sync-buffer', type=int, default=32, help='Maximum pending compressed messages per stream')
    p.add_argument('--future-seconds', type=float, default=0.0,
                   help='Discard RGB-D samples with less than this many future odom seconds. '
                        'For training, normally TRAINING.HORIZON * TRAINING.DT.')
    p.add_argument('--frame-step', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0, help='For development/test; 0 means all')
    p.add_argument('--min-frames', type=int, default=6)
    p.set_defaults(func=run_convert)

    p = commands.add_parser('make-npz', help='Export a preprocessed inference clip from a converted recording')
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True, help='WayFASTER temporal_model.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--frames', type=int, default=0, help='Defaults to MODEL.TIME_LENGTH in config')
    p.add_argument('--pixel-step', type=int, default=1,
                   help='Depth voxelization stride. 1 matches official loader.')
    add_validation_arguments(p)
    p.set_defaults(func=run_make_npz)

    args = parser.parse_args()
    if getattr(args, 'rhe', False):
        if args.labels_dir or args.odom_topic != ODOM_TOPIC:
            parser.error('--rhe uses /rover/odom and cannot be combined with --labels-dir')
        if args.future_seconds == 0:
            args.future_seconds = 30.
        if args.future_seconds < 30 or args.min_frames < 6:
            parser.error('--rhe requires >=30 future seconds and >=6 image frames')
    for name in ('rhe_horizon_seconds', 'max_cmd_age', 'rhe_max_speed', 'rhe_max_yaw_rate',
                 'rhe_max_position_error', 'rhe_max_heading_error'):
        value = getattr(args, name, 1.)
        if not np.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and > 0')
    if getattr(args, 'frame_step', 1) < 1 or getattr(args, 'pixel_step', 1) < 1:
        parser.error('--frame-step and --pixel-step must be >=1')
    try:
        ValidationLimits.from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ('sync_tolerance_ms', 'future_seconds', 'max_label_gap'):
        value = getattr(args, name, 0)
        if not np.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and >= 0')
    if hasattr(args, 'max_tf_gap') and (not np.isfinite(args.max_tf_gap) or args.max_tf_gap <= 0):
        parser.error('--max-tf-gap must be finite and > 0')
    if getattr(args, 'max_sync_buffer', 1) < 1 or getattr(args, 'min_frames', 1) < 1:
        parser.error('--max-sync-buffer and --min-frames must be >= 1')
    if getattr(args, 'max_frames', 0) < 0 or getattr(args, 'index', 0) < 0 or getattr(args, 'frames', 0) < 0:
        parser.error('--max-frames, --index and --frames must be >= 0')
    if args.command == 'make-npz' and args.frames == 0:
        args.frames = int(yaml.safe_load(args.config.read_text())['MODEL']['TIME_LENGTH'])
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'[ERROR] {exc}\n')


def add_validation_arguments(parser):
    parser.add_argument('--base-frame', default='base_footprint', help='Odometry child and target camera base frame')
    parser.add_argument('--odom-frame', default='odom', help='Required odometry parent frame')
    for name, field in ValidationLimits.__dataclass_fields__.items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=field.default,
                            help=f'Integrity limit ({name}); default {field.default}. '
                                 'Times in seconds, position step in metres, rotation step in radians.')


if __name__ == '__main__':
    main()
