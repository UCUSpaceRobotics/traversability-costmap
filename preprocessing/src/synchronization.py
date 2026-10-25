"""Simple bounded FIFO pairing for strictly monotonic acquisition timestamps."""
from collections import deque

from .base import RGB_TOPIC, DEPTH_TOPIC
from .validation import acquisition_ns


def iter_rgbd_pairs(reader, tolerance_ns, max_gap_ns, stats, max_pending=32):
    if tolerance_ns < 0 or max_gap_ns <= 0 or max_pending < 1:
        raise ValueError('Synchronization tolerance must be >= 0; gap and buffer limits must be > 0')
    queues = {RGB_TOPIC: deque(), DEPTH_TOPIC: deque()}
    previous = {}; stats.update(unmatched_rgb=0, unmatched_depth=0, pairs_seen=0)
    connections = [c for c in reader.connections if c.topic in queues]
    if {c.topic for c in connections} != set(queues):
        raise ValueError('Both RGB and depth topics are required')
    for conn, rx, raw in reader.messages(connections=connections):
        msg = reader.deserialize(raw, conn.msgtype)
        stamp = acquisition_ns(msg, conn.topic)
        if conn.topic in previous:
            gap = stamp - previous[conn.topic]
            if gap <= 0:
                raise ValueError(f'{conn.topic}: duplicate/out-of-order acquisition timestamp')
            if gap > max_gap_ns:
                raise ValueError(f'{conn.topic}: excessive acquisition timestamp gap')
        previous[conn.topic] = stamp
        queues[conn.topic].append((stamp, msg))
        if len(queues[conn.topic]) > max_pending:
            raise ValueError('RGB/depth synchronization buffer exceeded limit; stream missing/stalled')
        rgb, depth = queues[RGB_TOPIC], queues[DEPTH_TOPIC]
        while rgb and depth:
            rt, rm = rgb[0]; dt, dm = depth[0]
            if abs(rt-dt) <= tolerance_ns:
                rgb.popleft(); depth.popleft(); stats['pairs_seen'] += 1
                yield rt, rm, dt, dm
            elif rt < dt:
                rgb.popleft(); stats['unmatched_rgb'] += 1
            else:
                depth.popleft(); stats['unmatched_depth'] += 1
    stats['unmatched_rgb'] += len(queues[RGB_TOPIC])
    stats['unmatched_depth'] += len(queues[DEPTH_TOPIC])
