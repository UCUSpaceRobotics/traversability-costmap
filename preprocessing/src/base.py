import csv
from collections import deque
from pathlib import Path
import struct

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation


RGB_TOPIC = '/rover/zed2i/rgb/image_rect_color/compressed'
DEPTH_TOPIC = '/rover/zed2i/depth/depth_registered/compressedDepth'
CAMERA_TOPIC = '/rover/zed2i/rgb/camera_info'
ODOM_TOPIC = '/rover/odom'
STATIC_TF_TOPIC = '/tf_static'
DYNAMIC_TF_TOPIC = '/tf'


# =========================
# Transformations
# =========================
def timestamp_ns(msg, receive_ns: int) -> int:
    header = getattr(msg, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is not None:
        result = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if result > 0:
            return result
    return int(receive_ns)

def normalize_frame(name: str) -> str:
    return str(name).strip().lstrip('/')


def tf_to_matrix(transform) -> np.ndarray:
    tr = transform.translation
    q = transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [tr.x, tr.y, tr.z]
    return matrix


def cam2base_from_tf(edges: dict, camera_frame: str, base_frame: str) -> np.ndarray:
    """Find ^base T_camera via static TF tree, reversing edges as needed."""
    camera_frame, base_frame = map(normalize_frame, (camera_frame, base_frame))
    # A TransformStamped parent -> child represents ^parent T_child.
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (parent, child), matrix in edges.items():
        graph.setdefault(parent, []).append((child, matrix))
        graph.setdefault(child, []).append((parent, np.linalg.inv(matrix)))
    queue = deque([(base_frame, np.eye(4))])
    visited = set()
    while queue:
        node, acc = queue.popleft()  # accumulated ^base T_node
        if node == camera_frame:
            return acc
        if node in visited:
            continue
        visited.add(node)
        for next_frame, current_to_next in graph.get(node, []):
            if next_frame not in visited:
                queue.append((next_frame, acc @ current_to_next))
    edges_str = ', '.join(f'{a}->{b}' for a, b in sorted(edges))
    raise ValueError(
        f'No static TF chain from base frame {base_frame!r} to optical camera frame '
        f'{camera_frame!r}. Available static edges: {edges_str or "(none)"}. '
        'Specify the correct --base-frame/--camera-frame or provide '
        '--cam2base-json with a calibrated 4x4 transform. Never use identity by default.'
    )

def depth_from_compressed(msg) -> np.ndarray:
    """Decode ROS compressedDepth (16UC1 or 32FC1) to metres. Invalid = NaN.

    Adapted from UCUSpaceRobotics/traversability-costmap's report decoder.
    """
    data = bytes(msg.data)
    offset = data.find(b'\x89PNG')
    if offset < 0:
        raise ValueError('compressedDepth message contains no PNG signature')
    image = cv2.imdecode(np.frombuffer(data[offset:], np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError('Invalid single-channel compressedDepth PNG')
    raw = image.astype(np.float32)
    if '32FC1' in msg.format:
        if offset < 12:
            raise ValueError('Missing 12-byte depth quantization header')
        _, a, b = struct.unpack('<iff', data[:12])
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = a / (raw - b)
        depth[raw == 0] = np.nan
    elif '16UC1' in msg.format:
        depth = raw / 1000.0
    else:
        raise ValueError(f'Unsupported compressed depth format: {msg.format!r}')
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


# ==========================
# Reading ROS2 MCAP bags
# ==========================
def resolve_bags(path: Path) -> list[Path]:
    if (path / 'metadata.yaml').exists() and any(path.glob('*.mcap')):
        return [path]
    bags = sorted({p.parent for p in path.rglob('*.mcap') if (p.parent / 'metadata.yaml').exists()})
    if not bags:
        raise FileNotFoundError(f'No ROS2 MCAP bag directories under {path}')
    return bags

def grab_metadata(reader: AnyReader, odom_topic: str) -> tuple[dict, list, np.ndarray | None, str | None]:
    """Legacy inventory wrapper. Conversion must use recording.read_metadata's TF buffer."""
    from .recording import read_metadata
    metadata = read_metadata(reader, odom_topic)
    return (metadata.transforms.static, metadata.states,
            metadata.intrinsics, metadata.camera_frame)

# ==========================
# CSV label reading and writing
# ==========================
def read_label_csv(file: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with file.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'timestamp', 'mu', 'nu'}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'{file} needs timestamp,mu,nu headers')
        for row in reader:
            values = [float(row[k]) for k in ('timestamp', 'mu', 'nu')]
            if not np.isfinite(values).all() or not (0 <= values[1] <= 1 and 0 <= values[2] <= 1):
                raise ValueError(f'Invalid [timestamp,mu,nu] label row in {file}: {row}')
            rows.append(values)
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    if len(values) < 2 or not np.all(np.diff(values[:, 0]) > 0):
        raise ValueError('Need >=2 labels with strictly increasing timestamps')
    return values[:, 0], values[:, 1:]


def interpolated_labels(stamps: np.ndarray, label_file: Path, max_gap_s: float) -> np.ndarray:
    t, y = read_label_csv(label_file)
    near = np.searchsorted(t, stamps, side='left')
    right = np.clip(near, 0, len(t) - 1)
    left = np.clip(near - 1, 0, len(t) - 1)
    nearest_distance = np.minimum(np.abs(t[right] - stamps), np.abs(t[left] - stamps))
    labels = np.column_stack((np.interp(stamps, t, y[:, 0]), np.interp(stamps, t, y[:, 1])))
    okay = (stamps >= t[0]) & (stamps <= t[-1]) & (nearest_distance <= max_gap_s)
    if not okay.all():
        count = int(np.count_nonzero(~okay))
        raise ValueError(f'{label_file}: {count} odometry rows lack a nearby supervised label '
                         f'(max gap {max_gap_s}s). Generate labels covering the full recording.')
    return labels


def matrix_csv(array: np.ndarray) -> str:
    return np.array2string(np.asarray(array), separator=' ', max_line_width=100000, precision=10)

def vector_csv(array) -> str:
    return '[' + ' '.join(f'{float(v):.10g}' for v in array) + ']'


def save_depth_png(message, output: Path) -> None:
    metres = depth_from_compressed(message)
    mm = np.zeros(metres.shape, dtype=np.uint16)
    valid = np.isfinite(metres) & (metres > 0) & (metres <= 65.535)
    mm[valid] = np.rint(metres[valid] * 1000).astype(np.uint16)
    if not cv2.imwrite(str(output), mm):
        raise OSError(f'Could not save {output}')
