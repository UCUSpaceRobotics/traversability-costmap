"""Run with: python -m unittest discover -s tests -v (no ROS installation needed)."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'preprocessing'))
from src.base import CAMERA_TOPIC, DEPTH_TOPIC, RGB_TOPIC, ODOM_TOPIC, STATIC_TF_TOPIC, DYNAMIC_TF_TOPIC
from src.convert import convert_bag
from src.make_npz import get_extrinsics_original
from src.output import save_npz_exclusive, staged_recording
from src.recording import read_metadata
from src.synchronization import iter_rgbd_pairs
from src.transforms import TransformBuffer, validate_transform
from src.validation import (ValidationLimits, acquisition_ns, format_timestamp,
                            interpolate_state, parse_timestamp_ns, validate_sequence, validate_states)

SPEC = importlib.util.spec_from_file_location('preprocess_cli', ROOT / 'preprocessing/preprocess_wayfaster.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)
SPEC = importlib.util.spec_from_file_location('inference_cli', ROOT / 'scripts/infer_wayfaster.py')
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)
EPOCH = 1_790_000_000_000_000_000


def header(t, frame='optical'):
    return NS(stamp=NS(sec=t // 10**9, nanosec=t % 10**9), frame_id=frame)


def options(**changes):
    values = dict(odom_topic=ODOM_TOPIC, odom_frame='odom', base_frame='base_footprint',
                  camera_frame=None, cam2base_json=None, tf_policy='prefer-dynamic',
                  max_tf_gap=.2, sync_tolerance_ms=0., max_sync_buffer=32,
                  future_seconds=0., frame_step=1, max_frames=6, min_frames=6,
                  max_label_gap=.11)
    values.update(changes)
    return NS(**values)


def numeric_states():
    return [(EPOCH+i*50_000_000, np.array([i*.01, 0., 0.]),
             np.array([1., 0., 0., 0.]), np.array([.2, .0])) for i in range(31)]


def odom(s):
    t, p, q, a = s
    return NS(header=header(t, 'odom'), child_frame_id='base_footprint',
              pose=NS(pose=NS(position=NS(x=p[0], y=p[1], z=p[2]),
                              orientation=NS(w=q[0], x=q[1], y=q[2], z=q[3]))),
              twist=NS(twist=NS(linear=NS(x=a[0], y=0., z=0.), angular=NS(x=0., y=0., z=a[1]))))


def tf(t, parent, child, pitch=0.):
    q = Rotation.from_euler('y', pitch).as_quat()
    return NS(header=header(t, parent), child_frame_id=child,
              transform=NS(translation=NS(x=.2, y=0., z=.5),
                           rotation=NS(x=q[0], y=q[1], z=q[2], w=q[3])))


def image(t, depth=False):
    data = np.full((8, 10), 1000, np.uint16) if depth else np.full((8, 10, 3), 90, np.uint8)
    _, data = cv2.imencode('.png', data)
    return NS(header=header(t), data=data, format='16UC1; compressedDepth' if depth else 'bgr8; png')


class Reader:
    def __init__(self, entries):
        self.entries = entries
        self.connections = [NS(topic=t, msgtype='test') for t in sorted({t for t, m in entries})]

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def deserialize(self, raw, msgtype): return raw

    def messages(self, connections):
        selected = {c.topic: c for c in connections}
        for i, (topic, msg) in enumerate(self.entries):
            if topic in selected:
                yield selected[topic], i, msg


def recording_entries():
    entries = [(CAMERA_TOPIC, NS(header=header(EPOCH), k=np.diag([10., 10., 1.]).ravel()))]
    entries.append((STATIC_TF_TOPIC, NS(transforms=[tf(EPOCH, 'base_footprint', 'optical')])) )
    for s in numeric_states():
        entries += [(ODOM_TOPIC, odom(s)),
                    (DYNAMIC_TF_TOPIC, NS(transforms=[tf(s[0], 'base_footprint', 'optical', (s[0]-EPOCH)*1e-10)]))]
    for i in range(6):
        t = EPOCH + 525_000_000 + i*50_000_000
        entries += [(RGB_TOPIC, image(t)), (DEPTH_TOPIC, image(t, True))]
    return entries


class SafetyTests(unittest.TestCase):
    def test_epoch_nanosecond_roundtrip_and_missing_header(self):
        t = EPOCH+123456789
        self.assertEqual(parse_timestamp_ns(format_timestamp(t)), t)
        with self.assertRaises(ValueError): acquisition_ns(NS(), 'rgb')
        with self.assertRaises(ValueError): acquisition_ns(NS(header=header(0)), 'rgb')

    def test_transform_authority_interpolation_and_inverse(self):
        buffer = TransformBuffer()
        fixed = np.eye(4); fixed[:3, 3] = [1, 2, 3]
        buffer.add('base', 'mount', fixed)
        buffer.add('mount', 'camera', np.eye(4))
        a = np.eye(4); b = np.eye(4)
        b[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix(); b[0, 3] = 2
        buffer.add('mount', 'camera', a, EPOCH)
        buffer.add('mount', 'camera', b, EPOCH+100_000_000)
        buffer.finalize()
        with self.assertRaisesRegex(ValueError, 'authority conflict'): buffer.camera_path('camera', 'base')
        dynamic = buffer.camera_path('camera', 'base', 'prefer-dynamic')
        actual = buffer.at(dynamic, EPOCH+50_000_000, .2)
        midpoint = np.eye(4); midpoint[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix(); midpoint[0, 3] = 1
        np.testing.assert_allclose(actual, fixed @ midpoint, atol=1e-14)
        inverse = buffer.at(buffer.camera_path('base', 'camera', 'prefer-dynamic'), EPOCH+50_000_000, .2)
        np.testing.assert_allclose(actual @ inverse, np.eye(4), atol=1e-14)
        np.testing.assert_array_equal(buffer.at(buffer.camera_path('camera', 'base', 'prefer-static'), EPOCH, .2), fixed)
        with self.assertRaises(ValueError): buffer.at(dynamic, EPOCH-1, .2)
        with self.assertRaises(ValueError): buffer.at(dynamic, EPOCH+50_000_000, .01)

    def test_rigid_transform_required(self):
        with self.assertRaises(ValueError): validate_transform(np.diag([2., 1, 1, 1]))

    def test_tf_retransmissions_conflicts_and_topology(self):
        buffer = TransformBuffer()
        buffer.add('a', 'b', np.eye(4), EPOCH)
        buffer.add('a', 'b', np.eye(4), EPOCH)
        buffer.finalize()
        self.assertEqual(len(buffer.dynamic[('a', 'b')]), 1)
        buffer = TransformBuffer(); moved = np.eye(4); moved[0, 3] = 1
        buffer.add('a', 'b', np.eye(4), EPOCH); buffer.add('a', 'b', moved, EPOCH)
        with self.assertRaisesRegex(ValueError, 'Conflicting dynamic'): buffer.finalize()
        for edges in [[('a', 'b'), ('c', 'b')], [('a', 'b'), ('b', 'a')]]:
            buffer = TransformBuffer()
            for parent, child in edges: buffer.add(parent, child, np.eye(4))
            with self.assertRaises(ValueError): buffer.finalize()

    def test_odometry_integrity(self):
        good = numeric_states()
        validate_states(good, ValidationLimits())
        for index, value in [(1, np.array([np.nan, 0, 0])), (2, np.zeros(4)), (3, np.array([np.inf, 0]))]:
            bad = list(good); row = list(bad[1]); row[index] = value; bad[1] = tuple(row)
            with self.assertRaises(ValueError): validate_states(bad, ValidationLimits())
        for times in ([good[0], good[0]], [good[1], good[0]], [good[0], good[10]]):
            with self.assertRaises(ValueError): validate_states(times, ValidationLimits())
        bad = list(good); bad[1] = (bad[1][0], np.array([1., 0, 0]), bad[1][2], bad[1][3])
        with self.assertRaises(ValueError): validate_states(bad, ValidationLimits())
        bad = list(good); q = Rotation.from_euler('z', 1.).as_quat()[[3, 0, 1, 2]]
        bad[1] = (bad[1][0], bad[1][1], q, bad[1][3])
        with self.assertRaises(ValueError): validate_states(bad, ValidationLimits())

    def test_pose_interpolation_and_offset(self):
        s, offset = interpolate_state(numeric_states(), EPOCH+25_000_000, ValidationLimits())
        np.testing.assert_allclose(s[1], [.005, 0, 0]); self.assertEqual(offset, 25_000_000)
        with self.assertRaises(ValueError): interpolate_state(numeric_states(), EPOCH+25_000_000, ValidationLimits(max_odom_offset=.01))
        with self.assertRaises(ValueError): interpolate_state(numeric_states(), EPOCH-1, ValidationLimits())

    def test_frames_and_all_velocity_components(self):
        for attribute in ('parent', 'child', 'velocity'):
            entries = recording_entries()
            msg = next(m for t, m in entries if t == ODOM_TOPIC)
            if attribute == 'parent': msg.header.frame_id = 'map'
            elif attribute == 'child': msg.child_frame_id = 'camera'
            else: msg.twist.twist.linear.y = float('nan')
            with self.assertRaises(ValueError): read_metadata(Reader(entries), ODOM_TOPIC)

    def test_sync_matching_offsets_missing_duplicates_order_and_buffer(self):
        def match(entries, tolerance=0, pending=32):
            stats = {}
            pairs = list(iter_rgbd_pairs(Reader(entries), tolerance, 500_000_000, stats, pending))
            return pairs, stats
        t = EPOCH
        pairs, stats = match([(RGB_TOPIC, image(t)), (DEPTH_TOPIC, image(t)), (RGB_TOPIC, image(t+50_000_000))])
        self.assertEqual(len(pairs), 1); self.assertEqual(stats['unmatched_rgb'], 1)
        self.assertEqual(len(match([(RGB_TOPIC, image(t)), (DEPTH_TOPIC, image(t+1000))], 1000)[0]), 1)
        self.assertEqual(len(match([(RGB_TOPIC, image(t)), (DEPTH_TOPIC, image(t+1001))], 1000)[0]), 0)
        for entries in ([(RGB_TOPIC, image(t)), (RGB_TOPIC, image(t))],
                        [(RGB_TOPIC, image(t+1)), (RGB_TOPIC, image(t))],
                        [(RGB_TOPIC, image(t)), (RGB_TOPIC, image(t+600_000_000))]):
            with self.assertRaises(ValueError): match(entries + [(DEPTH_TOPIC, image(t))])
        with self.assertRaisesRegex(ValueError, 'buffer'):
            match([(RGB_TOPIC, image(t+i)) for i in range(4)] + [(DEPTH_TOPIC, image(t))], pending=2)

    def test_sequence_size_indices_duplicates_and_gaps(self):
        times = np.arange(10, dtype=np.int64)*100_000_000 + EPOCH
        validate_sequence(times, 0, 6, 6, .5)
        for index, count in [(-1, 6), (8, 6), (0, 5)]:
            with self.assertRaises(ValueError): validate_sequence(times, index, count, 6, .5)
        times[3:] += 10_000_000_000
        with self.assertRaises(ValueError): validate_sequence(times, 0, 6, 6, .5)
        times[1] = times[0]
        with self.assertRaises(ValueError): validate_sequence(times, 0, 6, 6, .5)

    def test_preflight_checks_entire_recording_not_only_selected_clip(self):
        entries = recording_entries()
        first = next(m for t, m in entries if t == RGB_TOPIC)
        entries.append((RGB_TOPIC, first))
        with self.assertRaisesRegex(ValueError, 'duplicate/out-of-order'):
            read_metadata(Reader(entries), ODOM_TOPIC)

    def test_wayfaster_planar_geometry_independently(self):
        times = np.array([EPOCH, EPOCH+50_000_000], np.int64)
        p = np.array([[0, 0, 0], [.01, .01, 0.]])
        rotations = Rotation.from_euler('z', np.array([[.2], [.3]])); q = rotations.as_quat()[:, [3, 0, 1, 2]]
        camera = np.eye(4); camera[:3, 3] = [.3, .1, .5]
        actual = get_extrinsics_original(times, times, p, q, [camera]*2, timestamps_ns=True)
        reference_inv = np.eye(4); reference_inv[:3, :3] = rotations[-1].inv().as_matrix(); reference_inv[:3, 3] = -reference_inv[:3, :3] @ p[-1]
        for i in range(2):
            pose = np.eye(4); pose[:3, :3] = rotations[i].as_matrix(); pose[:3, 3] = p[i]
            np.testing.assert_allclose(actual[i], reference_inv @ pose @ camera, atol=1e-14)

    def test_atomic_publication_failure_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'record'
            with self.assertRaises(RuntimeError):
                with staged_recording(dest) as stage:
                    (stage/'states.csv').write_text('partial'); raise RuntimeError('interrupted')
            self.assertFalse(dest.exists()); self.assertEqual(list(Path(temp).iterdir()), [])
            with staged_recording(dest) as stage: (stage/'complete').write_text('yes')
            with self.assertRaises(FileExistsError):
                with staged_recording(dest): self.fail('must not overwrite')
            self.assertEqual((dest/'complete').read_text(), 'yes')
            npz = Path(temp)/'sample.npz'; save_npz_exclusive(npz, x=np.ones(2))
            with self.assertRaises(FileExistsError): save_npz_exclusive(npz, x=np.zeros(2))
            np.testing.assert_array_equal(np.load(npz)['x'], np.ones(2))

    def test_conversion_and_npz_integration(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'record'
            with patch('src.convert.AnyReader', side_effect=lambda _: Reader(recording_entries())):
                info = convert_bag(Path('synthetic'), dest, options(), None)
                self.assertTrue(info['complete']); self.assertEqual(info['max_abs_sync_error_ns'], 0)
                with self.assertRaises(FileExistsError): convert_bag(Path('synthetic'), dest, options(), None)
            self.assertFalse((dest/'states.csv').exists())
            with (dest/'images.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertNotEqual(rows[0]['cam2base'], rows[-1]['cam2base'])
            config = Path(temp)/'config.yaml'
            config.write_text('MODEL:\n  TIME_LENGTH: 6\n  INPUT_SIZE: [10, 8]\n  GRID_BOUNDS:\n    xbound: [-2, 8, 0.1]\n    ybound: [-5, 5, 0.1]\n    zbound: [-1, 2, 0.2]\n')
            args = NS(recording=dest, config=config, frames=6, index=0, output=Path(temp)/'sample.npz', pixel_step=1, base_frame='base_footprint', odom_frame='odom')
            CLI.run_make_npz(args)
            with np.load(args.output) as sample:
                self.assertEqual(sample['pcloud'].shape, (1, 6, 15, 100, 100))
                self.assertEqual(sample['timestamps_ns'].dtype, np.int64)
                for name in ('color_img', 'pcloud', 'extrinsics', 'intrinsics'):
                    self.assertTrue(np.isfinite(sample[name]).all())
            (dest/'states.csv').write_text('stale')
            with self.assertRaisesRegex(ValueError, 'exactly one'): CLI.run_make_npz(args)

    def test_conversion_conflict_and_write_failure_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, args in [('conflict', options(tf_policy='require-explicit')), ('write', options())]:
                dest = Path(temp)/name
                with patch('src.convert.AnyReader', side_effect=lambda _: Reader(recording_entries())):
                    if name == 'conflict':
                        with self.assertRaisesRegex(ValueError, 'authority conflict'): convert_bag(Path('synthetic'), dest, args, None)
                    else:
                        with patch('src.convert.cv2.imwrite', return_value=False):
                            with self.assertRaises(OSError): convert_bag(Path('synthetic'), dest, args, None)
                self.assertFalse(dest.exists()); self.assertEqual(list(Path(temp).iterdir()), [])

    def test_labeled_to_unlabeled_rerun_cannot_reuse_stale_supervision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dest = root/'record'; label = root/'labels.csv'
            with label.open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(['timestamp', 'mu', 'nu'])
                writer.writerows([[format_timestamp(s[0]), .4, .7] for s in numeric_states()])
            with patch('src.convert.AnyReader', side_effect=lambda _: Reader(recording_entries())):
                convert_bag(Path('synthetic'), dest, options(), label)
                original = (dest/'states.csv').read_bytes()
                with self.assertRaises(FileExistsError): convert_bag(Path('synthetic'), dest, options(), None)
                self.assertEqual(original, (dest/'states.csv').read_bytes())
                self.assertFalse((dest/'states_unlabeled.csv').exists())

    def test_rhe_clip_has_history_and_full_thirty_second_future(self):
        start, end = EPOCH+5_000_000_000, EPOCH+40_000_000_000
        entries = [(CAMERA_TOPIC, NS(header=header(EPOCH), k=np.diag([10., 10., 1.]).ravel())),
                   (STATIC_TF_TOPIC, NS(transforms=[tf(EPOCH, 'base_footprint', 'optical')]))]
        states = [(EPOCH+i*50_000_000, np.array([i*.01, 0., 0.]), np.array([1., 0., 0., 0.]),
                   np.array([.2, .0])) for i in range(821)]
        for state in states:
            message = odom(state)
            if state[0] < start or state[0] > end:
                message.twist.twist.linear.x = 1.425  # rejected region is never exported
            entries.append((ODOM_TOPIC, message))
        for i in range(30):
            t = start+500_000_000+i*200_000_000
            entries.extend([(RGB_TOPIC, image(t)), (DEPTH_TOPIC, image(t, True))])
        labels = [dict(timestamp_ns=s[0], valid=1, mu=.7, nu=.4) for s in states if start <= s[0] <= end]
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'clip'
            with patch('src.convert.AnyReader', side_effect=lambda _: Reader(entries)):
                info = convert_bag(Path('synthetic'), dest,
                                   options(future_seconds=30., max_frames=0), None, labels)
            self.assertTrue(info['trainable'])
            self.assertEqual(info['rhe_clip_interval_ns'], (start, end))
            with (dest/'images.csv').open() as f:
                stamps = [parse_timestamp_ns(r['timestamp']) for r in csv.DictReader(f)]
            self.assertGreaterEqual(len(stamps), 6)
            # Every possible six-frame sample has complete future support, even
            # if the trainer anchors its trajectory at the last history image.
            for i in range(len(stamps)-5):
                self.assertLessEqual(stamps[i+5]+30_000_000_000, end)
            with (dest/'states.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(parse_timestamp_ns(rows[0]['timestamp']), start)
            self.assertEqual(parse_timestamp_ns(rows[-1]['timestamp']), end)
            self.assertTrue(all(r['action'] == '[0.2 0]' for r in rows))
            self.assertTrue(all(r['traversability'] == '[0.7 0.4]' for r in rows))
            short = [r for r in labels if r['timestamp_ns'] <= start+30_600_000_000]
            with patch('src.convert.AnyReader', side_effect=lambda _: Reader(entries)):
                with self.assertRaisesRegex(ValueError, 'usable RGB-D frames'):
                    convert_bag(Path('synthetic'), Path(temp)/'short',
                                options(future_seconds=30., max_frames=0), None, short)
            self.assertFalse((Path(temp)/'short').exists())

    def test_rhe_cli_exports_original_dataset_layout(self):
        from src.labels.rhe_pipeline import Command, Odom, rollout
        steps = 800
        poses = rollout(np.zeros(3), .7, .5, np.full(steps, .4*.05), np.full(steps, .2*.05))
        source, commands, states = [], [], []
        for i, (x, y, yaw) in enumerate(poses):
            stamp = EPOCH+i*50_000_000
            p = np.array([x, y, 0.])
            q = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
            a = np.array([.28, .1])
            source.append(Odom(stamp, stamp+1_000_000, 'odom', 'base_footprint', p, q, a, False))
            commands.append(Command(stamp, .4, .2, 'receive'))
            states.append((stamp, p, q, a))
        entries = [(CAMERA_TOPIC, NS(header=header(EPOCH), k=np.diag([10., 10., 1.]).ravel())),
                   (STATIC_TF_TOPIC, NS(transforms=[tf(EPOCH, 'base_footprint', 'optical')]))]
        entries.extend((ODOM_TOPIC, odom(s)) for s in states)
        for i in range(180):
            stamp = EPOCH+4_000_000_000+i*200_000_000
            entries.extend([(RGB_TOPIC, image(stamp)), (DEPTH_TOPIC, image(stamp, True))])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'output'
            bag = Path(temp)/'session'/'bag'
            argv = ['preprocess_wayfaster.py', 'convert', '--input', str(bag), '--output', str(output),
                    '--split', 'train', '--rhe', '--require-labels']
            with patch.object(sys, 'argv', argv), patch.object(CLI, 'resolve_bags', return_value=[bag]), \
                    patch('src.labels.rhe_pipeline.read_bag', return_value=(source, commands, {})), \
                    patch('src.convert.AnyReader', side_effect=lambda _: Reader(entries)):
                CLI.main()
            with (output/'train'/'rosbags.csv').open() as f:
                clips = list(csv.DictReader(f))
            self.assertEqual(len(clips), 1)
            clip = output/'train'/clips[0]['bag']
            self.assertTrue((clip/'states.csv').exists())
            self.assertFalse((clip/'states_unlabeled.csv').exists())
            summary = json.loads((output/'rhe'/'session__bag'/'summary.json').read_text())
            self.assertEqual(summary['invalid_measurements'], 0)
            self.assertEqual(summary['invalid_motion_intervals'], 0)
            self.assertEqual(summary['accepted_windows'], 741)
            self.assertEqual(summary['exported_clip_count'], 1)
            self.assertAlmostEqual(summary['exported_duration_s'], 37.)
            with (output/'rhe'/'session__bag'/'estimates.csv').open() as f:
                results = list(csv.DictReader(f))
            self.assertEqual(len(results), 801)
            for result in results[60:]:
                self.assertAlmostEqual(float(result['mu']), .7, places=4)
                self.assertAlmostEqual(float(result['nu']), .5, places=4)

    def test_legacy_recording_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'preprocessing.json').write_text('{"trainable": false}')
            with self.assertRaisesRegex(ValueError, 'Legacy/incomplete'):
                CLI.run_make_npz(NS(recording=root))

    def test_inference_tensor_validation(self):
        config = dict(TIME_LENGTH=6, INPUT_SIZE=[10, 8], GRID_BOUNDS=dict(
            xbound=[0, 4, 1], ybound=[0, 5, 1], zbound=[0, 2, 1]))
        arrays = dict(color_img=np.zeros((1, 6, 3, 8, 10), np.float32),
                      pcloud=np.zeros((1, 6, 2, 4, 5), np.float32),
                      intrinsics=np.tile(np.eye(3), (1, 6, 1, 1)),
                      extrinsics=np.tile(np.eye(4), (1, 6, 1, 1)),
                      timestamps_ns=EPOCH+np.arange(6, dtype=np.int64)*50_000_000)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'clip.npz'
            np.savez(path, **arrays)
            loaded = INFERENCE.load_inputs(path, config)
            self.assertEqual(loaded['color_img'].dtype, np.float32)
            np.testing.assert_array_equal(loaded['timestamps_ns'], arrays['timestamps_ns'])
            for key, value in (
                    ('color_img', arrays['color_img'][:, :-1]),
                    ('color_img', np.full_like(arrays['color_img'], np.nan)),
                    ('color_img', np.full_like(arrays['color_img'], 2.)),
                    ('pcloud', arrays['pcloud'][..., :-1]),
                    ('intrinsics', np.zeros_like(arrays['intrinsics'])),
                    ('extrinsics', np.zeros_like(arrays['extrinsics'])),
                    ('timestamps_ns', np.full(6, EPOCH, dtype=np.int64)),
                    ('timestamps_ns', arrays['timestamps_ns'].astype(float))):
                np.savez(path, **{**arrays, key: value})
                with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                    INFERENCE.load_inputs(path, config)

    def test_inference_refuses_existing_output_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'prediction'
            output.mkdir()
            (output/'keep.txt').write_text('original')
            with self.assertRaises(FileExistsError):
                INFERENCE.run_inference(NS(output_dir=output))
            self.assertEqual((output/'keep.txt').read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
