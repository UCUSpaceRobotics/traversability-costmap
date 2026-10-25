#!/usr/bin/env python3
"""Convert UCU rover ROS2 MCAP recordings to WayFASTER's original CSV/image format.

Commands:
    convert: MCAP -> images.csv + RGB/depth PNGs + states.csv (only with labels)
    make-npz: converted recording -> inference-ready WayFASTER tensors (.npz)
    inspect: show recorded topics and available TF frames

The original WayFASTER training loader expects traversability [mu, nu] supervision.
This script DOES NOT invent these values. Add --labels-dir containing per-bag
CSV files with timestamp,mu,nu to produce trainable states.csv files.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader

from src.base import *
from src.convert import convert_bag, read_manifest
from src.make_npz import decode_matrix, get_extrinsics_original, voxelize_mm


def run_convert(args) -> None:
    manifest = read_manifest(args.manifest)
    bags = resolve_bags(args.input)
    groups = {'train': [], 'val': [], 'inference': []}
    errors = []
    for bag in bags:
        session = bag.parent.name if bag.parent != args.input else 'session'
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
        if args.require_labels and label_file is None:
            errors.append(f'{bag.name}: missing label CSV; cannot train')
            print('[ERROR]', errors[-1])
            continue
        target = args.output / split / name
        print(f'\n[CONVERT] {bag} -> {target}')
        try:
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
        with (directory / 'rosbags.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bag'])
            writer.writerows([[name] for name, label_ready in items if label_ready])
        with (directory / 'inference_bags.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bag'])
            writer.writerows([[name] for name, _ in items])
    print(f'\nConverted {sum(len(x) for x in groups.values())}/{len(bags)} recordings. Errors: {len(errors)}')
    if errors:
        for err in errors:
            print(' -', err)
        raise SystemExit(1)
    if all(not label_ready for records in groups.values() for _, label_ready in records):
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
        edges, states, K, frame = grab_metadata(reader, args.odom_topic)
        print('Camera frame:', frame)
        print('Camera K:', K)
        print('Odom states:', len(states))
        print('Static TF edges:')
        for parent, child in sorted(edges):
            print(f'  {parent} -> {child}')

def run_make_npz(args) -> None:
    record = args.recording
    with (record / 'images.csv').open(newline='') as f:
        image_rows = list(csv.DictReader(f))
    state_file = record / 'states.csv'
    if not state_file.exists():
        state_file = record / 'states_unlabeled.csv'
    with state_file.open(newline='') as f:
        state_rows = list(csv.DictReader(f))
    T = args.frames or 6
    selected = image_rows[args.index:args.index + T]
    if len(selected) != T:
        raise ValueError(f'Not enough image frames: requested {T} at index {args.index}')
    state_time = np.asarray([float(r['timestamp']) for r in state_rows])
    state_pos = np.asarray([decode_matrix(r['position'], (3,)) for r in state_rows])
    state_quat = np.asarray([decode_matrix(r['quaternion'], (4,)) for r in state_rows])
    image_time = np.asarray([float(r['timestamp']) for r in selected])
    K = [decode_matrix(r['intrinsics'], (3, 3)) for r in selected]
    T_base_cam = [decode_matrix(r['cam2base'], (4, 4)) for r in selected]
    extrinsics = get_extrinsics_original(image_time, state_time, state_pos, state_quat, T_base_cam)
    config = yaml.safe_load(args.config.read_text())['MODEL']
    out_w, out_h = config['INPUT_SIZE']
    rgb_tensors, cloud_tensors, K_tensors = [], [], []
    for row, k, extr in zip(selected, K, extrinsics):
        rgb = cv2.imread(str(record / row['image']), cv2.IMREAD_COLOR)
        dep = cv2.imread(str(record / row['depth']), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None or dep.ndim != 2 or dep.dtype != np.uint16:
            raise ValueError(f'Missing RGB or uint16 depth at {row["timestamp"]}')
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination,
                        color_img=np.stack(rgb_tensors)[None],
                        pcloud=np.stack(cloud_tensors)[None],
                        intrinsics=np.stack(K_tensors)[None],
                        extrinsics=np.stack(extrinsics).astype(np.float32)[None],
                        timestamps=image_time)
    print('Saved inference input:', destination)
    print('  color_img:', (1, T, 3, out_h, out_w))
    print('  pcloud:', (1, T, *cloud_tensors[0].shape))
    print('  intrinsics:', (1, T, 3, 3), 'extrinsics:', (1, T, 4, 4))
    if args.pixel_step > 1:
        print(f'Warning: pixel-step={args.pixel_step} makes the voxel map approximate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('inspect', help='Show recorded topics, odometry and TF static frames')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--odom-topic', default=ODOM_TOPIC)
    p.set_defaults(func=run_inspect)

    p = commands.add_parser('convert', help='Convert MCAP bag(s) to WayFASTER CSV + images')
    p.add_argument('--input', type=Path, required=True, help='A bag folder, session folder, or all data/raw/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Original rover data/manifest.csv with train/val/exclude splits')
    p.add_argument('--split', choices=['train', 'val', 'inference'], default='inference',
                   help='Used if no manifest row exists; defaults to inference')
    p.add_argument('--include-excluded', action='store_true')
    p.add_argument('--labels-dir', type=Path, help='Directory of <bag_name>.csv files with timestamp,mu,nu')
    p.add_argument('--require-labels', action='store_true', help='Fail if any included bag lacks labels')
    p.add_argument('--max-label-gap', type=float, default=0.11)
    p.add_argument('--odom-topic', default=ODOM_TOPIC)
    p.add_argument('--base-frame', default='base_footprint')
    p.add_argument('--camera-frame', help='Override CameraInfo optical frame name if necessary')
    p.add_argument('--cam2base-json', type=Path, help='Override static TF with calibrated 4x4 matrix JSON')
    p.add_argument('--sync-tolerance-ms', type=float, default=30.0)
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
    p.set_defaults(func=run_make_npz)

    args = parser.parse_args()
    if getattr(args, 'frame_step', 1) < 1 or getattr(args, 'pixel_step', 1) < 1:
        parser.error('--frame-step and --pixel-step must be >=1')
    if args.command == 'make-npz' and args.frames == 0:
        args.frames = int(yaml.safe_load(args.config.read_text())['MODEL']['TIME_LENGTH'])
    args.func(args)


if __name__ == '__main__':
    main()
