"""Timestamped ^parent T_child transforms with explicit conflict authority."""
import bisect
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .base import normalize_frame


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)):
        raise ValueError('Camera/TF transform must be a finite homogeneous 4x4 matrix')
    r = matrix[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('Camera/TF transform rotation must be orthonormal with determinant +1')
    return matrix


class TransformBuffer:
    def __init__(self):
        self.static = {}
        self.dynamic = {}
        self._finalized = False

    def add(self, parent, child, matrix, timestamp=None):
        parent, child = normalize_frame(parent), normalize_frame(child)
        if not parent or not child or parent == child:
            raise ValueError('TF parent and child must be distinct nonempty frames')
        key = (parent, child)
        matrix = validate_transform(matrix)
        if timestamp is None:
            if key in self.static and not np.allclose(self.static[key], matrix, atol=1e-9, rtol=0):
                raise ValueError(f'Static TF changed for {parent}->{child}')
            self.static[key] = matrix
        else:
            if self._finalized:
                raise ValueError('Cannot append TF after finalization')
            self.dynamic.setdefault(key, []).append((int(timestamp), matrix))

    def finalize(self):
        parents = {}
        for parent, child in self.static.keys() | self.dynamic.keys():
            if child in parents and parents[child] != parent:
                raise ValueError(f'Ambiguous TF parent for {child}: {parents[child]} and {parent}')
            parents[child] = parent
        for child in parents:
            seen = set()
            node = child
            while node in parents:
                if node in seen:
                    raise ValueError(f'TF graph contains a cycle at {node}')
                seen.add(node)
                node = parents[node]
        for key, samples in self.dynamic.items():
            samples.sort(key=lambda s: s[0])
            unique = []
            for stamp, matrix in samples:
                if unique and stamp == unique[-1][0]:
                    if not np.allclose(matrix, unique[-1][1], atol=1e-9, rtol=0):
                        raise ValueError(f'Conflicting dynamic TF samples at {stamp}: {key}')
                    continue  # Identical TF retransmission, not an image observation.
                unique.append((stamp, matrix))
            self.dynamic[key] = unique
        self._finalized = True

    @property
    def conflicts(self):
        return sorted(self.static.keys() & self.dynamic.keys())

    def camera_path(self, camera_frame, base_frame, policy='require-explicit'):
        """Choose topology/authority once; evaluate dynamic edges for every image."""
        if policy not in ('require-explicit', 'prefer-static', 'prefer-dynamic'):
            raise ValueError(f'Unknown TF authority policy: {policy}')
        camera_frame, base_frame = normalize_frame(camera_frame), normalize_frame(base_frame)
        graph = {}
        for parent, child in sorted(self.static.keys() | self.dynamic.keys()):
            graph.setdefault(parent, []).append((child, (parent, child), False))
            graph.setdefault(child, []).append((parent, (parent, child), True))
        queue = deque([(base_frame, [])]); visited = set()
        while queue:
            node, path = queue.popleft()
            if node == camera_frame:
                result = []
                for edge, inverse in path:
                    overlap = edge in self.static and edge in self.dynamic
                    if overlap and policy == 'require-explicit':
                        raise ValueError(f'TF authority conflict for {edge[0]}->{edge[1]}; '
                                         'inspect frame semantics, then choose --tf-policy '
                                         'prefer-dynamic or prefer-static explicitly')
                    source = ('dynamic' if edge in self.dynamic and
                              not (overlap and policy == 'prefer-static') else 'static')
                    result.append((edge, inverse, source))
                return result
            if node in visited:
                continue
            visited.add(node)
            for other, edge, inverse in graph.get(node, []):
                if other not in visited:
                    queue.append((other, path + [(edge, inverse)]))
        raise ValueError(f'No TF chain from {base_frame!r} to {camera_frame!r}; '
                         'provide correct frames or a verified --cam2base-json')

    def at(self, path, timestamp, max_gap_s):
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('Dynamic TF maximum gap must be finite and > 0')
        if not self._finalized:
            raise ValueError('Finalize TF buffer before lookup')
        result = np.eye(4)
        for edge, inverse, source in path:
            if source == 'static':
                matrix = self.static[edge]
            else:
                samples = self.dynamic[edge]
                right = bisect.bisect_left(samples, int(timestamp), key=lambda s: s[0])
                if right < len(samples) and samples[right][0] == timestamp:
                    matrix = samples[right][1]
                else:
                    if right == 0 or right == len(samples):
                        raise ValueError(f'No bracketing dynamic TF for {edge} at {timestamp}')
                    t0, m0 = samples[right-1]; t1, m1 = samples[right]
                    if t1-t0 > max_gap_s * 1e9:
                        raise ValueError(f'Dynamic TF gap for {edge} exceeds {max_gap_s}s')
                    alpha = (timestamp-t0)/(t1-t0)
                    matrix = np.eye(4)
                    matrix[:3, :3] = Slerp([0., 1.], Rotation.from_matrix([m0[:3, :3], m1[:3, :3]]))(alpha).as_matrix()
                    matrix[:3, 3] = (1-alpha)*m0[:3, 3] + alpha*m1[:3, 3]
            result = result @ (np.linalg.inv(matrix) if inverse else matrix)
        return result

    def describe(self, path):
        return [{'parent': e[0], 'child': e[1], 'inverse': inv, 'source': source}
                for e, inv, source in path]

    def conflict_summary(self):
        result = []
        for edge in self.conflicts:
            fixed = self.static[edge]
            matrices = np.array([m for _, m in self.dynamic[edge]])
            angles = (Rotation.from_matrix(fixed[:3, :3]).inv() *
                      Rotation.from_matrix(matrices[:, :3, :3])).magnitude()
            result.append({'parent': edge[0], 'child': edge[1],
                           'max_translation_difference_m': float(np.linalg.norm(matrices[:, :3, 3] - fixed[:3, 3], axis=1).max()),
                           'max_rotation_difference_deg': float(np.rad2deg(angles).max())})
        return result
