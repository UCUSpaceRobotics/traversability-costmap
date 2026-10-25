
import numpy as np
from scipy.spatial.transform import Rotation

from .validation import ValidationLimits, interpolate_state, validate_states

def require_fields(record: dict, fields: list[str]) -> None:
    for name in fields:
        if name not in record:
            raise ValueError(f'Missing CSV field {name!r}')


def decode_matrix(text: str, shape: tuple[int, ...]) -> np.ndarray:
    data = np.fromstring(text.replace('[', ' ').replace(']', ' '), sep=' ')
    if data.size != int(np.prod(shape)):
        raise ValueError(f'Expected shape {shape}, got {data.size} elements: {text[:80]}')
    return data.reshape(shape)


def voxelize_mm(depth_mm: np.ndarray, K: np.ndarray, extr: np.ndarray, bounds: dict,
                pixel_step: int) -> np.ndarray:
    """Replicate original WayFASTER read_pcloud indexing with bounded memory."""
    limits = [bounds[k] for k in ('xbound', 'ybound', 'zbound')]
    dx = np.asarray([x[2] for x in limits], dtype=np.float64)
    nx = np.asarray([round((x[1] - x[0]) / x[2]) for x in limits], dtype=np.int64)
    cx = np.rint(np.asarray([x[1] / x[2] - 0.5 for x in limits])).astype(np.int64)
    grid = np.zeros((int(nx[2]), int(nx[0]), int(nx[1])), dtype=np.float32)
    invK = np.linalg.inv(K)
    R, translation = extr[:3, :3], extr[:3, 3]
    h, w = depth_mm.shape
    for v_start in range(0, h, 64 * pixel_step):
        vs = np.arange(v_start, min(v_start + 64 * pixel_step, h), pixel_step)
        us = np.arange(0, w, pixel_step)
        xx, yy = np.meshgrid(us, vs)
        depth = depth_mm[np.ix_(vs, us)].astype(np.float64) * 0.001
        good = np.isfinite(depth) & (depth > 0)
        if not good.any():
            continue
        uvz = np.stack((xx[good] * depth[good], yy[good] * depth[good], depth[good]), 0)
        pts = R @ (invK @ uvz) + translation[:, None]
        indices = np.rint(cx[:, None] - pts / dx[:, None] - 0.5).astype(int)
        ok = np.all((indices >= 0) & (indices < nx[:, None]), axis=0)
        indices = indices[:, ok]
        grid[indices[2], indices[0], indices[1]] = 1.0
    return grid


def get_extrinsics_original(image_times: np.ndarray, state_times: np.ndarray,
                            positions: np.ndarray, quat_wxyz: np.ndarray,
                            cam2base: list[np.ndarray], limits=None,
                            timestamps_ns=False) -> list[np.ndarray]:
    """WayFASTER final-heading convention, with bounded image-time pose sampling.

    Conversion inserts these image-time poses in states.csv so the official
    loader can use them too. Its planar-reference rotation convention is retained.
    """
    limits = limits or ValidationLimits()
    factor = 1 if timestamps_ns else 1e9
    states = validate_states([(int(round(t * factor)), np.asarray(p), np.asarray(q), np.zeros(2))
                              for t, p, q in zip(state_times, positions, quat_wxyz)], limits)
    sampled = [interpolate_state(states, int(round(t * factor)), limits)[0] for t in image_times]
    p = np.asarray([s[1] for s in sampled])
    q = np.asarray([s[2] for s in sampled])[:, [1, 2, 3, 0]]
    if len(cam2base) != len(sampled):
        raise ValueError('Camera transforms and image timestamps have different lengths')
    euler = Rotation.from_quat(q).as_euler('zyx')
    heading_last = euler[-1, 0]
    heading_frame = Rotation.from_euler('zyx', [heading_last, 0, 0])
    p = (heading_frame.inv().as_matrix() @ (p - p[-1]).T).T
    result = []
    for i, T in enumerate(cam2base):
        base_rot = Rotation.from_euler('zyx', [0, euler[i, 1], euler[i, 2]])
        T_base = np.eye(4)
        T_base[:3, :3] = base_rot.as_matrix()
        T_head = np.eye(4)
        T_head[:3, :3] = Rotation.from_euler('zyx', [euler[i, 0] - heading_last, 0, 0]).as_matrix()
        T_head[:2, 3] = p[i, :2]
        result.append(T_head @ T_base @ T)
    return result
