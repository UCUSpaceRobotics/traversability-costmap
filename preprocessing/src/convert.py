

import json

from rosbags.highlevel import AnyReader

from src.base import *

def read_manifest(file: Path | None) -> dict:
    if file is None:
        return {}
    with file.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return {(r['session'], r['bag']): r['split'] for r in reader}


def convert_bag(bag: Path, output: Path, args, label_file: Path | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rgb_dir, depth_dir = output / 'rgb', output / 'depth'
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    with AnyReader([bag]) as reader:
        edges, states, K, camera_frame = grab_metadata(reader, args.odom_topic)
        if not states or K is None:
            raise ValueError(f'{bag.name}: Missing odometry ({args.odom_topic}) or RGB CameraInfo')
        if not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-8:
            raise ValueError('Invalid camera intrinsics')
        if args.cam2base_json:
            T = np.asarray(json.loads(args.cam2base_json.read_text()), dtype=np.float64)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError('--cam2base-json must contain a valid 4x4 homogeneous matrix')
        else:
            T = cam2base_from_tf(edges, args.camera_frame or camera_frame, args.base_frame)
        stamps = np.array([s[0] for s in states], dtype=np.float64) / 1e9
        labels = interpolated_labels(stamps, label_file, args.max_label_gap) if label_file else None
        with (output / ('states.csv' if labels is not None else 'states_unlabeled.csv')).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'position', 'quaternion', 'action'] + (['traversability'] if labels is not None else []))
            for i, row in enumerate(states):
                ns, pos, quat, action = row
                fields = [f'{ns / 1e9:.9f}', vector_csv(pos), vector_csv(quat), vector_csv(action)]
                if labels is not None:
                    fields.append(vector_csv(labels[i]))
                writer.writerow(fields)
        # Second pass: bounded-memory message synchronization based on header timestamps.
        paired = 0
        skipped_rgb = skipped_depth = 0
        rgb_queue, depth_queue = deque(), deque()
        rows = []
        connections = [c for c in reader.connections if c.topic in {RGB_TOPIC, DEPTH_TOPIC}]
        tol_ns = int(args.sync_tolerance_ms * 1e6)
        min_ts, max_ts = states[0][0], states[-1][0] - int(args.future_seconds * 1e9)
        for conn, rx_time, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            ts = timestamp_ns(msg, rx_time)
            if conn.topic == RGB_TOPIC:
                rgb_queue.append((ts, msg))
            else:
                depth_queue.append((ts, msg))
            while rgb_queue and depth_queue:
                rt, rm = rgb_queue[0]
                dt, dm = depth_queue[0]
                if abs(rt - dt) > tol_ns:
                    if rt < dt:
                        rgb_queue.popleft()
                        skipped_rgb += 1
                    else:
                        depth_queue.popleft()
                        skipped_depth += 1
                    continue
                rgb_queue.popleft()
                depth_queue.popleft()
                if rt < min_ts or rt >= max_ts:
                    continue
                if args.frame_step > 1 and (paired % args.frame_step) != 0:
                    paired += 1
                    continue
                color = cv2.imdecode(np.frombuffer(bytes(rm.data), np.uint8), cv2.IMREAD_COLOR)
                if color is None:
                    raise ValueError(f'{bag.name}: RGB decode failed at {rt}')
                index = len(rows)
                rgb_name, depth_name = f'rgb/{index:06d}.png', f'depth/{index:06d}.png'
                if not cv2.imwrite(str(output / rgb_name), color):
                    raise OSError(f'Could not save {rgb_name}')
                save_depth_png(dm, output / depth_name)
                rows.append((f'{rt / 1e9:.9f}', rgb_name, depth_name, matrix_csv(K), matrix_csv(T)))
                paired += 1
                if args.max_frames and len(rows) >= args.max_frames:
                    break
            if args.max_frames and len(rows) >= args.max_frames:
                break
        if len(rows) < args.min_frames:
            raise ValueError(f'{bag.name}: only {len(rows)} synchronized usable RGB-D frames '
                             f'(need >= {args.min_frames}); check timestamp sync and future horizon')
        with (output / 'images.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'image', 'depth', 'intrinsics', 'cam2base'])
            writer.writerows(rows)
        info = {
            'source_bag': str(bag), 'odometry_topic': args.odom_topic,
            'rgb_topic': RGB_TOPIC, 'depth_topic': DEPTH_TOPIC,
            'base_frame': args.base_frame, 'camera_frame': args.camera_frame or camera_frame,
            'rgbd_pairs_written': len(rows), 'odometry_rows': len(states),
            'unmatched_rgb': skipped_rgb, 'unmatched_depth': skipped_depth,
            'depth_png_units': 'uint16 millimetres, 0 = invalid',
            'labels': str(label_file) if label_file else None,
            'trainable': labels is not None,
            'cam2base': T.tolist(), 'intrinsics': K.tolist(),
        }
        (output / 'preprocessing.json').write_text(json.dumps(info, indent=2))
        return info