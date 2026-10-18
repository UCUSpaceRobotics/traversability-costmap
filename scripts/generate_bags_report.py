#!/usr/bin/env python3
"""Generate HTML overview reports for the ROS 2 bags in data/raw/.

Reports are written to reports/, mirroring the layout of data/raw/:
    reports/index.html                          # all sessions
    reports/<session>/index.html                # all bags of the session
    reports/<session>/<bag>/report.html         # one bag (self-contained, images embedded)
    reports/<session>/<bag>/summary.json        # machine-readable summary used by the indexes

Each bag report contains:
    - general info: duration, size, message counts, manifest row
    - integrity checks (ok / warning / error) with a short explanation
    - per-topic table: rate, jitter, gaps, start/end offsets, header-stamp latency, ...
    - topic continuity timeline (message rate per second, gaps marked)
    - speed over the ride: one panel per odometry source (vs. cmd_vel), comparison and difference
    - paths of all odometry sources, aligned at the start pose, and pairwise distances between them
    - IMU (acceleration, vibration, roll/pitch) and wheel joint plots
    - first / last RGB + depth frame vs. the start / end of the recording
    - RGB + depth previews at several moments of the ride

Usage examples (run from anywhere inside the repo):
    python scripts/generate_bags_report.py                                        # all bags
    python scripts/generate_bags_report.py --session 2026-10-17_ucu-cinema         # one session
    python scripts/generate_bags_report.py --bag flat_05ms_20261017_162915         # one bag (by name)
    python scripts/generate_bags_report.py --bag data/raw/<session>/<bag>          # one bag (by path)
    python scripts/generate_bags_report.py --skip-existing --jobs 4                # only new bags, in parallel

Requires: pip install rosbags numpy matplotlib opencv-python-headless pyyaml
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import re
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
MANIFEST = REPO_ROOT / "data" / "manifest.csv"

# --- Check settings -------------------------------------------------------------
# Topics every bag of the dataset is expected to contain.
EXPECTED_TOPICS = [
    "/rover/zed2i/rgb/image_rect_color/compressed",
    "/rover/zed2i/rgb/camera_info",
    "/rover/zed2i/depth/depth_registered/compressedDepth",
    "/rover/zed2i/depth/camera_info",
    "/rover/zed2i/imu/data",
    "/rover/zed2i/odom",
    "/rover/wheels/odom",
    "/rover/odom",
    "/rover/joint_states",
    "/rover/cmd_vel",
    "/tf",
    "/tf_static",
]
# Topics published only on input (teleop): their gaps are warnings, not errors.
EVENT_TOPICS = {"/rover/cmd_vel"}
REFERENCE_ODOM = "/rover/wheels/odom"   # reference for the drift plot and the summary distance
# Joints whose position is expected to stay constant (rockers are not measured).
CONSTANT_JOINTS = {"l_rocker_joint", "r_rocker_joint"}
BASE_FRAME = "base_footprint"     # odometry poses are converted to this frame before plotting
GAP_FACTOR = 3.0                  # interval > GAP_FACTOR * median interval is a gap ...
MIN_GAP_S = 0.25                  # ... and it must also be longer than this
BIG_GAP_S = 1.0                   # gaps longer than this are reported as errors
LATE_START_S = 2.0                # topic starting / ending this far from the bag edges -> warning
MAX_LATENCY_S = 0.5               # p99 of (receive time - header stamp) above this -> warning
STANDSTILL_SPEED = 0.02           # m/s, below this the rover is considered standing still
MIN_STANDSTILL_S = 5.0            # expected standstill at the start of each bag
MAX_PLAUSIBLE_SPEED = 3.0         # m/s, odometry jumps faster than this are pose jumps
SPEED_MARGIN = 1.25               # observed speed above max_speed * margin -> warning
GRAVITY = 9.81
MAX_DEPTH_M = 5.0                 # colour scale of depth previews
PREVIEW_W = 480                   # width of preview images in the report

# Reference palette (categorical slots in fixed order) + status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
STATUS = {"ok": "#1a8f5a", "warn": "#c98500", "error": "#d03b3b"}
STATUS_ICON = {"ok": "✓", "warn": "!", "error": "✗"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.2,
})


# --- Bag discovery --------------------------------------------------------------
@dataclass
class BagRef:
    session: str
    name: str
    path: Path

    @property
    def out_dir(self) -> Path:
        return REPORTS_DIR / self.session / self.name


def find_bags(raw_dir: Path) -> list[BagRef]:
    bags = []
    for meta in sorted(raw_dir.glob("*/*/metadata.yaml")):
        bag_dir = meta.parent
        if bag_dir.parent.name.startswith("."):
            continue
        bags.append(BagRef(bag_dir.parent.name, bag_dir.name, bag_dir))
    return bags


def select_bags(args: argparse.Namespace) -> list[BagRef]:
    all_bags = find_bags(RAW_DIR)
    if not args.session and not args.bag:
        return all_bags
    selected: dict[Path, BagRef] = {}
    for s in args.session or []:
        found = [b for b in all_bags if b.session == s]
        if not found:
            sys.exit(f"No bags found for session {s!r} in {RAW_DIR}")
        selected.update({b.path: b for b in found})
    for b in args.bag or []:
        p = Path(b).expanduser()
        if p.is_dir() and (p / "metadata.yaml").exists():
            p = p.resolve()
            selected[p] = BagRef(p.parent.name, p.name, p)
            continue
        found = [x for x in all_bags if x.name == b]
        if not found:
            sys.exit(f"Bag {b!r} not found (neither a bag directory nor a bag name in {RAW_DIR})")
        selected.update({x.path: x for x in found})
    return sorted(selected.values(), key=lambda r: (r.session, r.name))


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="") as f:
        return {r["bag"]: r for r in csv.DictReader(f)}


# --- Small math helpers ---------------------------------------------------------
def stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def quat_to_rpy(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def tf_matrix(t, q) -> np.ndarray:
    x, y, z, w = q
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = t
    return m


def se2(x, y, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def rolling(values: np.ndarray, n: int, fn=np.std) -> np.ndarray:
    """Centered rolling statistic (window of n samples)."""
    if len(values) < n or n < 2:
        return np.full(len(values), np.nan)
    win = np.lib.stride_tricks.sliding_window_view(values, n)
    out = np.full(len(values), np.nan)
    out[n // 2: n // 2 + len(win)] = fn(win, axis=1)
    return out


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_dur(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m)}:{sec:04.1f}" if m else f"{sec:.1f} s"


# --- Reading the bag ------------------------------------------------------------
@dataclass
class TopicData:
    name: str
    msgtype: str
    recv: list = field(default_factory=list)      # bag receive time, ns
    stamp: list = field(default_factory=list)     # header stamp, ns (if the message has a header)
    frame_ids: set = field(default_factory=set)


def decode_depth(msg) -> np.ndarray | None:
    """Decode a compressedDepth message into metres (float32, NaN = invalid)."""
    data = bytes(msg.data)
    png_at = data.find(b"\x89PNG")
    if png_at < 0:
        return None
    raw = cv2.imdecode(np.frombuffer(data[png_at:], np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    if "32FC1" in msg.format and png_at >= 12:
        _, quant_a, quant_b = struct.unpack("<iff", data[:12])
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = quant_a / (raw - quant_b)
        depth[raw == 0] = np.nan
    else:  # 16UC1, millimetres
        depth = raw / 1000.0
        depth[raw == 0] = np.nan
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def is_depth_topic(topic: str, fmt: str) -> bool:
    return "compressedDepth" in fmt or "depth" in topic.lower()


def read_bag(bag: BagRef, n_previews: int, frame_stats: bool) -> dict:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    topics: dict[str, TopicData] = {}
    odom: dict[str, dict] = {}
    imu: dict[str, dict] = {}
    cmd: dict[str, dict] = {}
    joints: dict[str, dict] = {}
    cam_info: dict[str, dict] = {}
    images: dict[str, dict] = {}
    tf_static: dict[str, tuple[str, np.ndarray]] = {}
    tf_dyn: dict[tuple[str, str], list] = {}
    read_error = None
    n_read = 0

    with AnyReader([bag.path], default_typestore=typestore) as reader:
        t0, t1 = reader.start_time, reader.end_time
        span = max(t1 - t0, 1)
        targets = [t0 + int(span * (i + 0.5) / n_previews) for i in range(n_previews)] if n_previews else []
        for c in reader.connections:
            topics.setdefault(c.topic, TopicData(c.topic, c.msgtype))
        try:
            for conn, t, raw in reader.messages():
                n_read += 1
                td = topics[conn.topic]
                td.recv.append(t)
                msg = reader.deserialize(raw, conn.msgtype)
                header = getattr(msg, "header", None)
                if header is not None:
                    td.stamp.append(stamp_ns(header))
                    td.frame_ids.add(header.frame_id)
                mt = conn.msgtype

                if mt == "nav_msgs/msg/Odometry":
                    d = odom.setdefault(conn.topic, {"t": [], "v": [], "child": set()})
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    tw = msg.twist.twist
                    d["t"].append(t)
                    d["v"].append((p.x, p.y, p.z, q.x, q.y, q.z, q.w, tw.linear.x, tw.linear.y, tw.angular.z))
                    d["child"].add(msg.child_frame_id)
                elif mt == "sensor_msgs/msg/Imu":
                    d = imu.setdefault(conn.topic, {"t": [], "v": []})
                    a, g, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
                    d["t"].append(t)
                    d["v"].append((a.x, a.y, a.z, g.x, g.y, g.z, q.x, q.y, q.z, q.w))
                elif mt in ("geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped"):
                    tw = msg.twist if mt.endswith("Stamped") else msg
                    d = cmd.setdefault(conn.topic, {"t": [], "v": []})
                    d["t"].append(t)
                    d["v"].append((tw.linear.x, tw.linear.y, tw.angular.z))
                elif mt == "sensor_msgs/msg/JointState":
                    d = joints.setdefault(conn.topic, {"t": [], "names": None, "name_changes": 0, "pos": [], "vel": []})
                    names = list(msg.name)
                    if d["names"] is None:
                        d["names"] = names
                    if names != d["names"]:
                        d["name_changes"] += 1
                        continue
                    n = len(names)
                    pos, vel = np.asarray(msg.position, float), np.asarray(msg.velocity, float)
                    d["t"].append(t)
                    d["pos"].append(pos if len(pos) == n else np.full(n, np.nan))
                    d["vel"].append(vel if len(vel) == n else np.full(n, np.nan))
                elif mt == "sensor_msgs/msg/CameraInfo":
                    d = cam_info.setdefault(conn.topic, {"variants": {}})
                    key = (msg.width, msg.height, tuple(np.round(msg.k, 3)), msg.distortion_model)
                    d["variants"][key] = d["variants"].get(key, 0) + 1
                elif mt == "sensor_msgs/msg/CompressedImage":
                    depth = is_depth_topic(conn.topic, msg.format)
                    d = images.setdefault(conn.topic, {
                        "depth": depth, "formats": set(), "t": [], "stamp": [], "stats": [],
                        "hashes": [], "decode_fail": 0, "shapes": set(),
                        "previews": [None] * len(targets), "first": None, "last": None,
                    })
                    d["formats"].add(msg.format)
                    s = stamp_ns(msg.header)
                    d["t"].append(t)
                    d["stamp"].append(s)
                    d["last"] = (s, t, bytes(msg.data), msg.format)
                    if d["first"] is None:
                        d["first"] = d["last"]
                    d["hashes"].append(hashlib.blake2b(bytes(msg.data), digest_size=8).digest())
                    # keep the frame closest to each preview time (by header stamp)
                    for i, tgt in enumerate(targets):
                        best = d["previews"][i]
                        if best is None or abs(s - tgt) < abs(best[0] - tgt):
                            d["previews"][i] = (s, t, bytes(msg.data), msg.format)
                    if frame_stats:
                        d["stats"].append(image_stats(msg, depth, d))
                elif mt == "tf2_msgs/msg/TFMessage":
                    for tr in msg.transforms:
                        if conn.topic == "/tf_static":
                            tt, rr = tr.transform.translation, tr.transform.rotation
                            tf_static[tr.child_frame_id] = (
                                tr.header.frame_id, tf_matrix((tt.x, tt.y, tt.z), (rr.x, rr.y, rr.z, rr.w)))
                        else:
                            tf_dyn.setdefault((tr.header.frame_id, tr.child_frame_id), []).append(t)
        except Exception as e:  # truncated / corrupted file: keep what was read
            read_error = f"{type(e).__name__}: {e} (after {n_read} messages)"

    return {
        "t0": t0, "t1": t1, "topics": topics, "odom": odom, "imu": imu, "cmd": cmd, "joints": joints,
        "cam_info": cam_info, "images": images, "tf_static": tf_static, "tf_dyn": tf_dyn,
        "targets": targets, "read_error": read_error, "n_read": n_read,
    }


def image_stats(msg, depth: bool, d: dict) -> tuple:
    """Per-frame statistics: (brightness, sharpness) for RGB, (valid fraction, median m) for depth."""
    if depth:
        img = decode_depth(msg)
        if img is None:
            d["decode_fail"] += 1
            return (np.nan, np.nan)
        d["shapes"].add(img.shape[:2])
        valid = np.isfinite(img)
        return (valid.mean(), float(np.median(img[valid])) if valid.any() else np.nan)
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        d["decode_fail"] += 1
        return (np.nan, np.nan)
    d["shapes"].add(img.shape[:2])
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return (float(img.mean()), float(cv2.Laplacian(small, cv2.CV_32F).var()))


# --- Analysis ------------------------------------------------------------------
@dataclass
class Check:
    level: str   # ok / warn / error
    title: str
    detail: str = ""


def topic_timing(td: TopicData, t0: int, t1: int) -> dict:
    recv = np.asarray(td.recv, np.int64)
    out = {"topic": td.name, "type": td.msgtype, "count": len(recv), "frames": sorted(td.frame_ids)}
    if len(recv) == 0:
        return out
    rel = (recv - t0) / 1e9
    out.update(first_s=rel[0], last_from_end_s=(t1 - recv[-1]) / 1e9, gaps=[])
    if len(recv) >= 3:
        dt = np.diff(np.sort(recv)) / 1e9
        med = float(np.median(dt))
        out.update(
            rate_hz=(len(recv) - 1) / max((recv[-1] - recv[0]) / 1e9, 1e-9),
            median_rate_hz=1 / med if med > 0 else float("inf"),
            jitter_ms=float(np.std(dt) * 1e3), max_dt_s=float(dt.max()),
        )
        thr = max(GAP_FACTOR * med, MIN_GAP_S)
        idx = np.nonzero(dt > thr)[0]
        out["gap_threshold_s"] = thr
        out["gaps"] = [(float(rel[i]), float(dt[i])) for i in idx]
    if td.stamp and len(td.stamp) == len(recv):
        st = np.asarray(td.stamp, np.int64)
        lat = (recv - st) / 1e9
        out.update(
            latency_med_ms=float(np.median(lat) * 1e3), latency_p99_ms=float(np.percentile(lat, 99) * 1e3),
            latency_min_ms=float(lat.min() * 1e3),
            stamp_backwards=int((np.diff(st) < 0).sum()), stamp_dupes=int((np.diff(st) == 0).sum()),
        )
    return out


def static_chain(tf_static: dict, parent: str, child: str) -> np.ndarray | None:
    """T_parent_child from static transforms, or None if the chain does not exist."""
    m, frame = np.eye(4), child
    for _ in range(50):
        if frame == parent:
            return m
        if frame not in tf_static:
            return None
        p, t = tf_static[frame]
        m = t @ m
        frame = p
    return None


def odom_series(d: dict, t0: int, tf_static: dict) -> dict:
    v = np.asarray(d["v"], float)
    t = (np.asarray(d["t"], np.int64) - t0) / 1e9
    x, y = v[:, 0], v[:, 1]
    _, _, yaw = quat_to_rpy(v[:, 3], v[:, 4], v[:, 5], v[:, 6])
    child = next(iter(d["child"])) if len(d["child"]) == 1 else None
    note = ""
    # Express the pose of BASE_FRAME, so all sources are comparable (e.g. ZED odom is the camera pose).
    if child and child != BASE_FRAME:
        t_base_child = static_chain(tf_static, BASE_FRAME, child)
        if t_base_child is not None:
            inv = np.linalg.inv(t_base_child)
            dx, dy = inv[0, 3], inv[1, 3]
            dyaw = math.atan2(inv[1, 0], inv[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            x, y, yaw = x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw
            note = f"pose of {child} converted to {BASE_FRAME} via tf_static"
        else:
            note = f"pose of {child} (no static TF to {BASE_FRAME})"
    # Align at the start pose, so all paths start at (0, 0) heading +x.
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw)
    ax = ay = np.full(len(x), np.nan)
    if finite.any():
        i0 = int(np.argmax(finite))
        inv0 = np.linalg.inv(se2(x[i0], y[i0], yaw[i0]))
        pts = inv0 @ np.vstack([x, y, np.ones(len(x))])
        ax, ay = pts[0], pts[1]
    step = np.hypot(np.diff(x), np.diff(y))
    dts = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        jumps = int(np.sum((step / np.maximum(dts, 1e-3) > MAX_PLAUSIBLE_SPEED) & (step > 0.05)))
    return {
        "t": t, "x": ax, "y": ay, "speed": np.hypot(v[:, 7], v[:, 8]), "wz": v[:, 9],
        "length_m": float(np.nansum(step)), "jumps": jumps,
        "nan": int((~np.isfinite(v)).any(axis=1).sum()), "child": ", ".join(sorted(d["child"])), "note": note,
    }


def analyze(bag: BagRef, raw: dict, manifest_row: dict | None) -> dict:
    t0, t1 = raw["t0"], raw["t1"]
    duration = (t1 - t0) / 1e9
    checks: list[Check] = []
    meta = yaml.safe_load((bag.path / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    meta_counts = {x["topic_metadata"]["name"]: x["message_count"] for x in meta["topics_with_message_count"]}
    files = sorted(bag.path.glob("*.mcap")) + sorted(bag.path.glob("*.db3"))
    size = sum(f.stat().st_size for f in files)

    # File-level checks
    if raw["read_error"]:
        checks.append(Check("error", "Bag could not be read to the end", raw["read_error"]))
    else:
        checks.append(Check("ok", "Bag read completely", f"{raw['n_read']} messages"))
    listed = meta.get("relative_file_paths", [])
    missing_files = [f for f in listed if not (bag.path / f).exists()]
    if missing_files:
        checks.append(Check("error", "Files listed in metadata.yaml are missing", ", ".join(missing_files)))
    if meta["message_count"] != raw["n_read"]:
        checks.append(Check("error", "Message count differs from metadata.yaml",
                            f"metadata {meta['message_count']}, read {raw['n_read']}"))

    # Topic timing
    timing = {name: topic_timing(td, t0, t1) for name, td in sorted(raw["topics"].items())}
    for name, tm in timing.items():
        tm["meta_count"] = meta_counts.get(name)
    missing = [t for t in EXPECTED_TOPICS if t not in timing or timing[t]["count"] == 0]
    checks.append(Check("error", "Expected topics missing", ", ".join(missing)) if missing
                  else Check("ok", "All expected topics present", f"{len(EXPECTED_TOPICS)} topics"))
    unexpected = [t for t in timing if t not in EXPECTED_TOPICS]
    if unexpected:
        checks.append(Check("ok", "Additional topics (not in the expected list)", ", ".join(unexpected)))

    gap_lines, big = [], False
    for name, tm in timing.items():
        for at, dt in tm.get("gaps", []):
            big |= dt > BIG_GAP_S and name not in EVENT_TOPICS
        if tm.get("gaps"):
            worst = max(tm["gaps"], key=lambda g: g[1])
            gap_lines.append(f"{name}: {len(tm['gaps'])} gap(s), longest {worst[1]:.2f} s at {worst[0]:.1f} s"
                             + (" (event-driven topic)" if name in EVENT_TOPICS else ""))
    if gap_lines:
        checks.append(Check("error" if big else "warn", "Gaps in topics", "<br>".join(gap_lines)))
    else:
        checks.append(Check("ok", "No gaps in any topic",
                            f"interval > {GAP_FACTOR:g}× median and > {MIN_GAP_S} s counts as a gap"))

    edge = [f"{n}: starts at +{tm['first_s']:.1f} s, ends {tm['last_from_end_s']:.1f} s before end"
            for n, tm in timing.items()
            if tm["count"] > 1 and (tm["first_s"] > LATE_START_S or tm["last_from_end_s"] > LATE_START_S)]
    if edge:
        checks.append(Check("warn", "Topics start late or stop early", "<br>".join(edge)))

    lat = [f"{n}: p99 {tm['latency_p99_ms']:.0f} ms" for n, tm in timing.items()
           if tm.get("latency_p99_ms", 0) > MAX_LATENCY_S * 1e3]
    neg = [f"{n}: min {tm['latency_min_ms']:.0f} ms" for n, tm in timing.items()
           if tm.get("latency_min_ms", 0) < -50]
    if lat:
        checks.append(Check("warn", "High header-stamp latency", "<br>".join(lat)))
    if neg:
        checks.append(Check("error", "Header stamps in the future (clock offset?)", "<br>".join(neg)))
    back = [f"{n}: {tm['stamp_backwards']} backwards, {tm['stamp_dupes']} duplicate"
            for n, tm in timing.items() if tm.get("stamp_backwards") or
            (tm.get("stamp_dupes") and n not in ("/tf", "/tf_static"))]
    if back:
        checks.append(Check("warn", "Non-monotonic header stamps", "<br>".join(back)))

    # Odometry
    odom = {n: odom_series(d, t0, raw["tf_static"]) for n, d in sorted(raw["odom"].items())}
    for n, o in odom.items():
        if o["nan"]:
            checks.append(Check("error", f"NaN/inf values in {n}", f"{o['nan']} messages"))
        if o["jumps"]:
            checks.append(Check("warn", f"Pose jumps in {n}",
                                f"{o['jumps']} steps faster than {MAX_PLAUSIBLE_SPEED} m/s"))
    m = re.search(r"_(\d{2})ms_", bag.name)
    max_speed = int(m.group(1)) / 10 if m else None
    for n, o in odom.items():
        rate = timing[n].get("median_rate_hz", 10)
        smooth = rolling(o["speed"], max(int(rate * 0.5), 2), np.median)  # twist is noisy, use 0.5 s median
        p99 = float(np.nanpercentile(smooth, 99)) if np.isfinite(smooth).any() else 0.0
        o["p99_speed"] = p99
        if max_speed and p99 > max_speed * SPEED_MARGIN:
            checks.append(Check("warn", f"Speed above the bag's max speed in {n}",
                                f"p99 of 0.5 s median {p99:.2f} m/s, max from bag name {max_speed} m/s"))
    if len(odom) >= 2:
        lens = {n: o["length_m"] for n, o in odom.items()}
        lo, hi = min(lens.values()), max(lens.values())
        detail = ", ".join(f"{n} {v:.1f} m" for n, v in lens.items())
        if hi > 1 and (hi - lo) / hi > 0.25:
            checks.append(Check("warn", "Path lengths of odometry sources differ by > 25 %", detail))
        else:
            checks.append(Check("ok", "Odometry path lengths agree", detail))

    # Standstill at start (needed e.g. for IMU bias estimation)
    wheel = odom.get(REFERENCE_ODOM) or (next(iter(odom.values())) if odom else None)
    standstill = None
    if wheel is not None and len(wheel["t"]):
        moving = np.nonzero(wheel["speed"] > STANDSTILL_SPEED)[0]
        standstill = float(wheel["t"][moving[0]]) if len(moving) else float(wheel["t"][-1])
        lvl = "ok" if standstill >= MIN_STANDSTILL_S else "warn"
        checks.append(Check(lvl, f"Standstill at start: {standstill:.1f} s",
                            f"expected ≥ {MIN_STANDSTILL_S:g} s (speed < {STANDSTILL_SPEED} m/s)"))

    # IMU
    imu = {}
    for n, d in raw["imu"].items():
        v = np.asarray(d["v"], float)
        t = (np.asarray(d["t"], np.int64) - t0) / 1e9
        acc = np.linalg.norm(v[:, 0:3], axis=1)
        roll, pitch, _ = quat_to_rpy(v[:, 6], v[:, 7], v[:, 8], v[:, 9])
        rate = timing[n].get("median_rate_hz", 100)
        win = max(int(round(rate * 0.5)), 2)
        frozen = int((np.diff(v[:, :6], axis=0) == 0).all(axis=1).sum())
        imu[n] = {"t": t, "acc": acc, "vib": rolling(acc, win), "gyro": np.linalg.norm(v[:, 3:6], axis=1),
                  "roll": np.degrees(roll), "pitch": np.degrees(pitch), "frozen": frozen,
                  "nan": int((~np.isfinite(v)).any(axis=1).sum())}
        g = float(np.nanmedian(acc)) if len(acc) else np.nan
        imu[n]["g"] = g
        if imu[n]["nan"]:
            checks.append(Check("error", f"NaN values in {n}", f"{imu[n]['nan']} messages"))
        if not abs(g - GRAVITY) < 0.5:
            checks.append(Check("warn", f"Median |acc| of {n} is {g:.2f} m/s²", "expected ≈ 9.81 (wrong units?)"))
        else:
            checks.append(Check("ok", f"IMU gravity check {n}", f"median |acc| = {g:.2f} m/s²"))
        if frozen > 0.01 * len(v):
            checks.append(Check("warn", f"Repeated identical IMU samples in {n}", f"{frozen} of {len(v)}"))

    # Joints
    joints = {}
    for n, d in raw["joints"].items():
        if d["name_changes"]:
            checks.append(Check("warn", f"Joint names change within {n}", f"{d['name_changes']} messages skipped"))
        if not d["t"]:
            continue
        joints[n] = {"t": (np.asarray(d["t"], np.int64) - t0) / 1e9, "names": d["names"],
                     "pos": np.vstack(d["pos"]), "vel": np.vstack(d["vel"])}
        pos = joints[n]["pos"]
        stuck = [nm for i, nm in enumerate(d["names"])
                 if nm not in CONSTANT_JOINTS and np.isfinite(pos[:, i]).any()
                 and np.nanmax(pos[:, i]) - np.nanmin(pos[:, i]) < 1e-6]
        if stuck:
            checks.append(Check("warn", f"Joint positions never change in {n}",
                                ", ".join(stuck) + " (not measured / sensor disconnected?)"))

    # Camera info
    cams = {}
    for n, d in raw["cam_info"].items():
        variants = d["variants"]
        cams[n] = [{"w": k[0], "h": k[1], "fx": k[2][0], "fy": k[2][4], "cx": k[2][2], "cy": k[2][5],
                    "model": k[3], "count": c} for k, c in variants.items()]
        if len(variants) > 1:
            checks.append(Check("warn", f"Camera info changes within {n}", f"{len(variants)} different variants"))

    # Images
    imgs = {}
    for n, d in raw["images"].items():
        t = (np.asarray(d["t"], np.int64) - t0) / 1e9
        stats = np.asarray(d["stats"], float).reshape(-1, 2)
        dupes = sum(1 for a, b in zip(d["hashes"], d["hashes"][1:]) if a == b)
        imgs[n] = {"depth": d["depth"], "t": t, "stats": stats, "formats": sorted(d["formats"]),
                   "dupes": dupes, "decode_fail": d["decode_fail"], "shapes": sorted(d["shapes"]),
                   "stamp": np.asarray(d["stamp"], np.int64), "count": len(t)}
        if d["decode_fail"]:
            checks.append(Check("error", f"Frames that cannot be decoded in {n}", f"{d['decode_fail']} frames"))
        if dupes:
            checks.append(Check("warn", f"Identical consecutive frames in {n}", f"{dupes} frames (frozen camera?)"))
        if len(stats) and not d["depth"]:
            dark = int((stats[:, 0] < 15).sum())
            bright = int((stats[:, 0] > 240).sum())
            if dark or bright:
                checks.append(Check("warn", f"Very dark / overexposed frames in {n}", f"{dark} dark, {bright} bright"))
        if len(stats) and d["depth"]:
            valid = float(np.nanmean(stats[:, 0]))
            lvl = "ok" if valid > 0.5 else "warn"
            checks.append(Check(lvl, f"Mean valid depth pixels {valid * 100:.0f} %", n))
        # image size vs camera_info
        info_topic = n.rsplit("/", 2)[0] + "/camera_info"
        for cand in (info_topic, n.split("/depth")[0] + "/depth/camera_info" if d["depth"] else ""):
            if cand in cams and d["shapes"]:
                infos = {(c["h"], c["w"]) for c in cams[cand]}
                if not set(d["shapes"]) <= infos:
                    checks.append(Check("error", f"Image size of {n} does not match {cand}",
                                        f"images {d['shapes']}, camera_info {sorted(infos)}"))
                break

    # RGB <-> depth sync
    rgb = [n for n, i in imgs.items() if not i["depth"]]
    dep = [n for n, i in imgs.items() if i["depth"]]
    sync = None
    if rgb and dep and imgs[rgb[0]]["count"] and imgs[dep[0]]["count"]:
        a, b = np.sort(imgs[rgb[0]]["stamp"]), np.sort(imgs[dep[0]]["stamp"])
        j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
        nearest = np.minimum(np.abs(a - b[j - 1]), np.abs(a - b[j])) / 1e6
        sync = {"matched": float((nearest < 1).mean()), "median_ms": float(np.median(nearest)),
                "rgb": rgb[0], "depth": dep[0]}
        lvl = "ok" if sync["matched"] > 0.95 else "warn"
        checks.append(Check(lvl, f"RGB/depth stamp sync: {sync['matched'] * 100:.1f} % exact matches",
                            f"median offset to nearest depth frame {sync['median_ms']:.1f} ms"))

    # TF
    tf_rates = {f"{p} → {c}": len(ts) / max(duration, 1e-9) for (p, c), ts in sorted(raw["tf_dyn"].items())}
    cam_frames = {f for tm in timing.values() if "Compressed" in tm["type"] for f in tm["frames"]}
    for f in sorted(cam_frames):
        if static_chain(raw["tf_static"], BASE_FRAME, f) is None and not any(
                c == f for (_, c) in raw["tf_dyn"]):
            checks.append(Check("error", f"No static TF chain {BASE_FRAME} → {f}", "needed to project images"))
        else:
            checks.append(Check("ok", f"TF chain {BASE_FRAME} → {f} available"))

    return {
        "bag": bag, "manifest": manifest_row, "duration": duration, "size": size, "files": files,
        "meta": meta, "t0": t0, "t1": t1, "n_msgs": raw["n_read"], "timing": timing, "checks": checks,
        "odom": odom, "imu": imu, "cmd": raw["cmd"], "joints": joints, "cams": cams, "imgs": imgs,
        "sync": sync, "tf_rates": tf_rates, "tf_static": raw["tf_static"], "max_speed": max_speed,
        "standstill": standstill, "raw_images": raw["images"], "targets": raw["targets"],
    }


# --- Plots ----------------------------------------------------------------------
def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def short(topic: str) -> str:
    return re.sub(r"/compressed$", "", topic.replace("/rover/", "/").replace("/image_rect_color", ""))


def plot_continuity(a: dict) -> str:
    timing = {n: tm for n, tm in a["timing"].items() if tm["count"] > 1}
    names = list(timing)
    dur = a["duration"]
    nbins = max(int(math.ceil(dur)), 1)
    edges = np.minimum(np.arange(nbins + 1, dtype=float), dur)
    widths = np.maximum(np.diff(edges), 1e-3)  # the last bin may be partial
    grid = np.full((len(names), nbins), np.nan)
    for i, n in enumerate(names):
        rel = (np.asarray(a["timing_raw"][n], np.int64) - a["t0"]) / 1e9
        counts, _ = np.histogram(rel, bins=edges)
        nominal = timing[n].get("median_rate_hz") or 1
        grid[i] = counts / (nominal * widths)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(names) + 1.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rate", ["#d03b3b", "#f3c9c4", "#cde2fb", "#2a78d6"], N=256)
    im = ax.imshow(np.clip(grid, 0, 1.25), aspect="auto", cmap=cmap, vmin=0, vmax=1.25,
                   extent=(0, nbins, len(names), 0), interpolation="nearest")
    for i, n in enumerate(names):
        for at, dt in timing[n].get("gaps", []):
            ax.plot([at, at + dt], [i + 0.5, i + 0.5], color=INK, lw=2.5, solid_capstyle="butt")
    ax.set_yticks(np.arange(len(names)) + 0.5, [short(n) for n in names])
    ax.set_xlabel("time since bag start [s]")
    ax.grid(False)
    ax.set_title("Topic continuity: messages per second / nominal rate (black bars = gaps)", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("relative rate")
    cb.outline.set_visible(False)
    return fig_to_b64(fig)


def plot_dt_hist(a: dict) -> str:
    names = [n for n, tm in a["timing"].items() if tm["count"] > 2]
    cols = 4
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 1.9 * rows), squeeze=False)
    for ax, n in zip(axes.flat, names):
        dt = np.diff(np.sort(np.asarray(a["timing_raw"][n], np.int64))) / 1e6
        ax.hist(dt, bins=60, color=SERIES[0], log=True)
        thr = a["timing"][n].get("gap_threshold_s")
        if thr:
            ax.axvline(thr * 1e3, color=STATUS["error"], lw=1, ls="--")
        ax.set_title(short(n), fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.supxlabel("interval between messages [ms] (red dashed = gap threshold)", fontsize=8, color=INK2)
    fig.tight_layout()
    return fig_to_b64(fig)


def mark_previews(ax, a):
    for i, tgt in enumerate(a["targets"]):
        x = (tgt - a["t0"]) / 1e9
        ax.axvline(x, color=MUTED, lw=0.6, ls=":")
        ax.text(x, 1.0, str(i + 1), transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                ha="center", va="bottom")


def smooth_odom(a: dict, n: str, values: np.ndarray) -> np.ndarray:
    """0.5 s rolling median, the twist of every source is noisy."""
    rate = a["timing"][n].get("median_rate_hz", 10)
    return rolling(values, max(int(rate * 0.5), 2), np.median)


def plot_speed(a: dict) -> str:
    odom = list(a["odom"].items())
    if not odom:
        return ""
    cmd = []
    for n, d in a["cmd"].items():
        v = np.asarray(d["v"], float)
        cmd.append((n, (np.asarray(d["t"], np.int64) - a["t0"]) / 1e9, np.hypot(v[:, 0], v[:, 1]), v[:, 2]))
    smooth = {n: smooth_odom(a, n, o["speed"]) for n, o in odom}
    ref_n = REFERENCE_ODOM if REFERENCE_ODOM in a["odom"] else odom[0][0]
    rows = len(odom) + 3  # one per source, comparison, difference to reference, yaw rate
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.9 * rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    top = max([a["max_speed"] or 0] + [np.nanpercentile(s, 99.5) for s in smooth.values() if np.isfinite(s).any()])
    top = top * 1.25 + 0.05

    def decorate(ax):
        if a["standstill"]:
            ax.axvspan(0, a["standstill"], color=GRID, alpha=0.6, lw=0)
        if a["max_speed"]:
            ax.axhline(a["max_speed"], color=INK2, lw=0.8, ls=":")
        for cn, t, sp, _ in cmd:
            ax.plot(t, sp, color=MUTED, lw=1, ls="--", label=f"{short(cn)} (command)")

    # one panel per source: raw twist (light) + 0.5 s median (bold) + command
    for k, (n, o) in enumerate(odom):
        ax, c = axes[k], SERIES[k % 8]
        decorate(ax)
        ax.plot(o["t"], o["speed"], color=c, lw=0.6, alpha=0.35, label="raw")
        ax.plot(o["t"], smooth[n], color=c, lw=1.6, label="0.5 s median")
        ax.set_ylim(0, top)
        ax.set_ylabel("|v| [m/s]")
        ax.set_title(short(n), loc="left", fontsize=9, pad=12 if k == 0 else 6)
        ax.legend(loc="upper right", ncols=3, fontsize=7)
    # all sources together (smoothed only)
    ax = axes[len(odom)]
    decorate(ax)
    for k, (n, o) in enumerate(odom):
        ax.plot(o["t"], smooth[n], color=SERIES[k % 8], lw=1.4, label=short(n))
    ax.set_ylim(0, top)
    ax.set_ylabel("|v| [m/s]")
    ax.set_title("All sources (0.5 s median)", loc="left", fontsize=9)
    ax.legend(loc="upper right", ncols=4, fontsize=7)
    # difference to the reference source
    ax = axes[len(odom) + 1]
    ref = a["odom"][ref_n]
    ax.axhline(0, color=INK2, lw=0.8)
    lim = [0.05]
    for k, (n, o) in enumerate(odom):
        if n == ref_n:
            continue
        diff = np.interp(ref["t"], o["t"], smooth[n], left=np.nan, right=np.nan) - smooth[ref_n]
        ax.plot(ref["t"], diff, color=SERIES[k % 8], lw=1.2, label=f"{short(n)} − {short(ref_n)}")
        if np.isfinite(diff).any():
            lim.append(np.nanpercentile(np.abs(diff), 99.5))
    ax.set_ylim(-max(lim) * 1.2, max(lim) * 1.2)
    ax.set_ylabel("Δ|v| [m/s]")
    ax.set_title(f"Speed difference to {short(ref_n)} (0.5 s median; < 0 = slower than the wheels)",
                 loc="left", fontsize=9)
    ax.legend(loc="upper right", ncols=3, fontsize=7)
    # yaw rate
    ax = axes[-1]
    ax.axhline(0, color=INK2, lw=0.8)
    for cn, t, _, wz in cmd:
        ax.plot(t, wz, color=MUTED, lw=1, ls="--", label=f"{short(cn)} (command)")
    for k, (n, o) in enumerate(odom):
        ax.plot(o["t"], smooth_odom(a, n, o["wz"]), color=SERIES[k % 8], lw=1.2, label=short(n))
    ax.set_ylabel("yaw rate [rad/s]")
    ax.set_title("Yaw rate (0.5 s median)", loc="left", fontsize=9)
    ax.legend(loc="upper right", ncols=4, fontsize=7)

    mark_previews(axes[0], a)
    axes[-1].set_xlabel("time since bag start [s]  (grey band = initial standstill, dotted = max speed from "
                        "bag name, dashed = command)")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_paths(a: dict) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    clean = [o for o in a["odom"].values() if not o["jumps"]] or list(a["odom"].values())
    for k, (n, o) in enumerate(a["odom"].items()):
        c = SERIES[k % 8]
        bad = f", {o['jumps']} pose jumps" if o["jumps"] else ""
        ax.plot(o["x"], o["y"], color=c, lw=0.6 if bad else 1.5, alpha=0.35 if bad else 1, zorder=1 if bad else 2,
                label=f"{short(n)}  ({o['length_m']:.1f} m{bad})")
        f = np.isfinite(o["x"])
        if f.any():
            ax.plot(o["x"][f][-1], o["y"][f][-1], "o", color=c, ms=7, mec="white", mew=1.5)
    ax.plot(0, 0, "s", color=INK, ms=7, mec="white", mew=1.5, label="start")
    # axis limits from sources without pose jumps, so a broken source does not hide the others
    xs = np.concatenate([o["x"] for o in clean] + [[0]])
    ys = np.concatenate([o["y"] for o in clean] + [[0]])
    pad = 0.05 * max(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys), 1)
    ax.set_xlim(np.nanmin(xs) - pad, np.nanmax(xs) + pad)
    ax.set_ylim(np.nanmin(ys) - pad, np.nanmax(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m] (start heading)")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Paths of all odometry sources ({BASE_FRAME}, aligned at start pose; dot = end)", loc="left")
    return fig_to_b64(fig)


def plot_odom_distances(a: dict) -> str:
    """Distance between the (start-aligned) positions of every pair of odometry sources over time."""
    items = list(a["odom"].items())
    if len(items) < 2:
        return ""
    # reference source first, so pairs read "<other> ↔ wheels"
    items.sort(key=lambda kv: kv[0] != REFERENCE_ODOM)
    pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    colors = {n: SERIES[k % 8] for k, n in enumerate(a["odom"])}
    fig, axes = plt.subplots(len(pairs) + 1, 1, figsize=(11, 2.0 * (len(pairs) + 1)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (i, j) in zip(axes, pairs):
        (na, oa), (nb, ob) = items[i], items[j]
        x = np.interp(oa["t"], ob["t"], ob["x"], left=np.nan, right=np.nan)
        y = np.interp(oa["t"], ob["t"], ob["y"], left=np.nan, right=np.nan)
        dist = np.hypot(x - oa["x"], y - oa["y"])
        broken = oa["jumps"] or ob["jumps"]
        ax.plot(oa["t"], dist, color=colors[nb], lw=0.6 if broken else 1.3)
        if np.isfinite(dist).any():
            fin = dist[np.isfinite(dist)]
            ax.annotate(f"end {fin[-1]:.2f} m · max {fin.max():.2f} m", (1, 1), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("distance [m]")
        ax.set_title(f"{short(nb)} ↔ {short(na)}" + ("  (source with pose jumps)" if broken else ""),
                     loc="left", fontsize=9)
    # cumulative travelled distance of each source
    ax = axes[-1]
    top = []
    for n, o in items:
        step = np.hypot(np.diff(o["x"]), np.diff(o["y"]))
        cum = np.concatenate([[0], np.nancumsum(step)])
        ax.plot(o["t"], cum, color=colors[n], lw=0.8 if o["jumps"] else 1.4,
                label=f"{short(n)} ({cum[-1]:.1f} m)")
        if not o["jumps"]:
            top.append(cum[-1])
    if top:
        ax.set_ylim(0, max(top) * 1.2 + 0.1)
    ax.set_ylabel("travelled [m]")
    ax.set_title("Cumulative travelled distance" + (" (axis scaled to sources without pose jumps)"
                                                    if len(top) < len(items) else ""), loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    axes[-1].set_xlabel("time since bag start [s]")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_imu(a: dict) -> str:
    if not a["imu"]:
        return ""
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for k, (n, d) in enumerate(a["imu"].items()):
        axes[0].plot(d["t"], d["acc"], color=SERIES[0], lw=0.6, label="|acc|")
        axes[0].axhline(GRAVITY, color=INK2, lw=0.8, ls=":")
        axes[1].plot(d["t"], d["vib"], color=SERIES[1], lw=1, label="rolling std |acc| (0.5 s)")
        axes[1].plot(d["t"], d["gyro"], color=SERIES[2], lw=0.6, label="|gyro| [rad/s]")
        axes[2].plot(d["t"], d["roll"], color=SERIES[0], lw=1, label="roll")
        axes[2].plot(d["t"], d["pitch"], color=SERIES[1], lw=1, label="pitch")
    axes[0].set_ylabel("|acc| [m/s²]")
    axes[1].set_ylabel("vibration")
    axes[2].set_ylabel("angle [°]")
    axes[2].set_xlabel("time since bag start [s]")
    for ax in axes[1:]:
        ax.legend(fontsize=8, loc="upper right", ncols=2)
    mark_previews(axes[0], a)
    axes[0].set_title(f"IMU {', '.join(short(n) for n in a['imu'])}", loc="left", pad=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_joints(a: dict) -> str:
    if not a["joints"]:
        return ""
    n, d = next(iter(a["joints"].items()))
    wheels = [i for i, nm in enumerate(d["names"]) if nm.endswith("wheel_joint")]
    steer = [i for i, nm in enumerate(d["names"]) if "mount" in nm]
    groups = [("wheel velocity [rad/s]", wheels, d["vel"]), ("steering angle [rad]", steer, d["pos"])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 2.2 * len(groups)), sharex=True, squeeze=False)
    for ax, (label, idx, arr) in zip(axes[:, 0], groups):
        for k, i in enumerate(idx):
            ax.plot(d["t"], arr[:, i], color=SERIES[k % 8], lw=0.9, label=d["names"][i])
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper right", ncols=4)
    axes[-1, 0].set_xlabel("time since bag start [s]")
    axes[0, 0].set_title(f"Joint states {short(n)}", loc="left")
    fig.tight_layout()
    return fig_to_b64(fig)


# --- Previews --------------------------------------------------------------------
def jpeg_b64(img: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    norm = np.clip(np.nan_to_num(depth, nan=0) / MAX_DEPTH_M, 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(depth)] = 0
    return img


class _Msg:  # minimal stand-in for decode_depth
    def __init__(self, data, fmt):
        self.data, self.format = data, fmt


def build_previews(a: dict) -> list[dict]:
    rgb = [n for n, i in a["raw_images"].items() if not i["depth"]]
    dep = [n for n, i in a["raw_images"].items() if i["depth"]]
    rows = []
    for k, tgt in enumerate(a["targets"]):
        row = {"idx": k + 1, "t": (tgt - a["t0"]) / 1e9, "cells": []}
        speed = None
        wheel = a["odom"].get(REFERENCE_ODOM)
        if wheel is not None and len(wheel["t"]):
            speed = float(np.interp(row["t"], wheel["t"], wheel["speed"]))
        row["speed"] = speed
        for n in rgb + dep:
            p = a["raw_images"][n]["previews"][k]
            if p is not None:
                row["cells"].append(frame_cell(a, n, p, f"{(p[0] - a['t0']) / 1e9:.2f} s"))
        rows.append(row)
    return rows


def frame_cell(a: dict, n: str, frame: tuple, caption: str) -> dict:
    """Decode one stored frame (stamp, recv, data, format) into a preview card."""
    stamp, recv, data, fmt = frame
    if a["raw_images"][n]["depth"]:
        depth = decode_depth(_Msg(data, fmt))
        img = colorize_depth(depth) if depth is not None else None
        extra = (f"valid {np.isfinite(depth).mean() * 100:.0f} %, median "
                 f"{np.nanmedian(depth):.2f} m") if depth is not None and np.isfinite(depth).any() else ""
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        extra = ""
    if img is None:
        return {"topic": short(n), "b64": "", "caption": f"{caption} · decode failed"}
    h, w = img.shape[:2]
    img = cv2.resize(img, (PREVIEW_W, int(h * PREVIEW_W / w)), interpolation=cv2.INTER_AREA)
    return {"topic": short(n), "b64": jpeg_b64(img),
            "caption": f"{caption} · {w}×{h}" + (f" · {extra}" if extra else "")}


def build_edge_frames(a: dict) -> list[dict]:
    """First and last frame of every image topic, relative to the start / end of the bag."""
    dur = a["duration"]
    first, last = {"title": "Start of recording", "cells": []}, {"title": "End of recording", "cells": []}
    names = sorted(a["raw_images"], key=lambda n: a["raw_images"][n]["depth"])  # RGB before depth
    for n in names:
        d = a["raw_images"][n]
        if d["first"] is None:
            continue
        s, r = d["first"][0], d["first"][1]
        first["cells"].append(frame_cell(a, n, d["first"],
                                         f"first frame received +{(r - a['t0']) / 1e9:.2f} s after bag start "
                                         f"(stamp +{(s - a['t0']) / 1e9:.2f} s)"))
        s, r = d["last"][0], d["last"][1]
        last["cells"].append(frame_cell(a, n, d["last"],
                                        f"last frame received at {(r - a['t0']) / 1e9:.2f} s, "
                                        f"{(a['t1'] - r) / 1e9:.2f} s before bag end ({dur:.2f} s) "
                                        f"(stamp {(s - a['t0']) / 1e9:.2f} s)"))
    return [first, last] if first["cells"] else []


def depth_colorbar_b64() -> str:
    grad = np.linspace(0, MAX_DEPTH_M, 256)[None, :].repeat(12, 0)
    img = colorize_depth(grad)
    return jpeg_b64(cv2.resize(img, (PREVIEW_W, 12)), 90)


# --- HTML ---------------------------------------------------------------------
CSS = """
:root { --bg:#f6f5f2; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df;
  --ok:#1a8f5a; --warn:#b07400; --err:#d03b3b; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 10px; }
a { color:var(--accent); } .muted { color:var(--ink2); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:10px 0; overflow-x:auto; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; margin:14px 0; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.tile .v { font-size:20px; font-weight:600; } .tile .l { color:var(--ink2); font-size:12px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th, td { padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; } td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.wrap { white-space:normal; }
.badge { display:inline-flex; align-items:center; gap:4px; font-weight:600; font-size:12px; padding:1px 8px; border-radius:99px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .error { color:var(--err); }
.badge.ok { background:#e3f3ea; } .badge.warn { background:#fbefd3; } .badge.error { background:#fbe1df; }
tr.warn td.hl { color:var(--warn); font-weight:600; } tr.error td.hl { color:var(--err); font-weight:600; }
.checks td:first-child { width:90px; }
img.plot { max-width:100%; height:auto; display:block; }
.previews { display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,480px),1fr)); gap:10px; }
.preview { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; }
.preview img { width:100%; height:auto; display:block; border-radius:6px; }
.preview .cap { font-size:12px; color:var(--ink2); margin:4px 2px 8px; }
.prow h3 { font-size:13px; margin:18px 0 6px; }
details summary { cursor:pointer; color:var(--ink2); }
"""


def esc(x) -> str:
    return html.escape(str(x))


def badge(level: str, text: str | None = None) -> str:
    text = text or {"ok": "OK", "warn": "Warning", "error": "Error"}[level]
    return f'<span class="badge {level}">{STATUS_ICON[level]} {esc(text)}</span>'


def overall(checks: list[Check]) -> str:
    levels = {c.level for c in checks}
    return "error" if "error" in levels else "warn" if "warn" in levels else "ok"


def num(v, f="{:.1f}") -> str:
    return f.format(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else "—"


def img_tag(b64: str, alt: str) -> str:
    return f'<div class="card"><img class="plot" alt="{esc(alt)}" src="data:image/png;base64,{b64}"></div>' if b64 else ""


def render_report(a: dict, plots: dict, previews: list[dict], edges: list[dict]) -> str:
    bag: BagRef = a["bag"]
    checks = sorted(a["checks"], key=lambda c: {"error": 0, "warn": 1, "ok": 2}[c.level])
    status = overall(checks)
    start = datetime.fromtimestamp(a["t0"] / 1e9, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end = datetime.fromtimestamp(a["t1"] / 1e9, timezone.utc).strftime("%H:%M:%S UTC")
    wheel = a["odom"].get(REFERENCE_ODOM)
    m = a["manifest"] or {}
    tiles = [
        (fmt_dur(a["duration"]), "duration"), (fmt_bytes(a["size"]), "size on disk"),
        (f"{a['n_msgs']:,}", "messages"), (str(len(a["timing"])), "topics"),
        (f"{wheel['length_m']:.1f} m" if wheel else "—", "distance (wheel odom)"),
        (f"{wheel['p99_speed']:.2f} m/s" if wheel else "—", "p99 speed (wheel odom)"),
        (f"{a['standstill']:.1f} s" if a["standstill"] is not None else "—", "initial standstill"),
        (m.get("split", "—") or "—", "split"),
    ]
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
         f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(bag.name)}</title>"
         f"<style>{CSS}</style></head><body><main>"]
    h.append(f'<p class="muted"><a href="../../index.html">All sessions</a> / '
             f'<a href="../index.html">{esc(bag.session)}</a></p>')
    h.append(f"<h1>{esc(bag.name)} {badge(status)}</h1>")
    h.append(f'<p class="muted">{start} – {end} · {esc(bag.path.relative_to(REPO_ROOT) if bag.path.is_relative_to(REPO_ROOT) else bag.path)} · '
             f'storage {esc(a["meta"].get("storage_identifier", "?"))}, '
             f'{len(a["files"])} file(s){", compression " + esc(a["meta"]["compression_format"]) if a["meta"].get("compression_format") else ""}'
             f' · report generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>')
    h.append('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(lbl)}</div></div>' for v, lbl in tiles) + "</div>")
    if m:
        h.append('<div class="card"><table><tr>' + "".join(f"<th>{esc(k)}</th>" for k in m) + "</tr><tr>"
                 + "".join(f"<td>{esc(v)}</td>" for v in m.values()) + "</tr></table></div>")

    h.append("<h2>Integrity checks</h2><div class='card'><table class='checks'>")
    for c in checks:
        h.append(f"<tr><td>{badge(c.level)}</td><td class='wrap'><b>{esc(c.title)}</b>"
                 f"<div class='muted'>{esc(c.detail).replace('&lt;br&gt;', '<br>')}</div></td></tr>")
    h.append("</table></div>")

    h.append("<h2>Topics</h2><div class='card'><table><tr><th>topic</th><th>type</th><th class=num>count</th>"
             "<th class=num>metadata</th><th class=num>rate Hz</th><th class=num>median Hz</th>"
             "<th class=num>jitter ms</th><th class=num>max Δt s</th><th class=num>gaps</th>"
             "<th class=num>start +s</th><th class=num>end −s</th><th class=num>latency ms (med / p99)</th>"
             "<th>frame</th></tr>")
    for n, tm in a["timing"].items():
        gaps = tm.get("gaps", [])
        lvl = "error" if any(g[1] > BIG_GAP_S for g in gaps) else "warn" if gaps else "ok"
        mismatch = tm["meta_count"] is not None and tm["meta_count"] != tm["count"]
        lat = (f"{tm['latency_med_ms']:.0f} / {tm['latency_p99_ms']:.0f}" if "latency_med_ms" in tm else "—")
        h.append(
            f"<tr class='{lvl}'><td>{esc(n)}</td><td class=muted>{esc(tm['type'])}</td><td class=num>{tm['count']:,}</td>"
            f"<td class='num {'error' if mismatch else ''}'>{num(tm['meta_count'], '{:,}')}</td>"
            f"<td class=num>{num(tm.get('rate_hz'), '{:.2f}')}</td><td class=num>{num(tm.get('median_rate_hz'), '{:.2f}')}</td>"
            f"<td class=num>{num(tm.get('jitter_ms'))}</td><td class='num hl'>{num(tm.get('max_dt_s'), '{:.3f}')}</td>"
            f"<td class='num hl'>{len(gaps)}</td><td class=num>{num(tm.get('first_s'))}</td>"
            f"<td class=num>{num(tm.get('last_from_end_s'))}</td><td class=num>{lat}</td>"
            f"<td class=muted>{esc(', '.join(tm['frames']))}</td></tr>")
    h.append("</table></div>")

    h.append("<h2>Continuity</h2>")
    h.append("<p class='muted'>Each row is a topic, each column one second. Blue = messages arrive at the "
             "nominal (median) rate, red = fewer messages than expected, black bars = gaps "
             f"(interval &gt; {GAP_FACTOR:g}× median and &gt; {MIN_GAP_S} s).</p>")
    h.append(img_tag(plots["continuity"], "topic continuity"))
    gap_rows = [(n, at, dt) for n, tm in a["timing"].items() for at, dt in tm.get("gaps", [])]
    if gap_rows:
        h.append("<details class='card'><summary>All gaps (" + str(len(gap_rows)) + ")</summary><table>"
                 "<tr><th>topic</th><th class=num>at s</th><th class=num>length s</th></tr>"
                 + "".join(f"<tr><td>{esc(n)}</td><td class=num>{at:.2f}</td><td class=num>{dt:.3f}</td></tr>"
                           for n, at, dt in sorted(gap_rows, key=lambda r: -r[2])[:300]) + "</table></details>")
    h.append("<details class='card'><summary>Distribution of message intervals</summary>"
             f"<img class='plot' alt='interval histograms' src='data:image/png;base64,{plots['dt_hist']}'></details>")

    h.append("<h2>Speed</h2>" + img_tag(plots["speed"], "speed"))
    h.append("<h2>Paths</h2>")
    notes = [f"{esc(n)}: child frame {esc(o['child'])}, {o['length_m']:.1f} m{', ' + esc(o['note']) if o['note'] else ''}"
             for n, o in a["odom"].items()]
    h.append("<p class='muted'>" + "<br>".join(notes) + "</p>")
    h.append(img_tag(plots["paths"], "paths"))
    if plots["distances"]:
        h.append("<h2>Distance between odometry sources</h2>"
                 "<p class='muted'>Distance between the start-aligned positions of each pair of sources "
                 "(accumulated drift / slip), and the travelled distance of each source.</p>"
                 + img_tag(plots["distances"], "distance between odometry sources"))
    if plots["imu"]:
        h.append("<h2>IMU</h2>" + img_tag(plots["imu"], "imu"))
    if plots["joints"]:
        h.append("<h2>Joints</h2>" + img_tag(plots["joints"], "joints"))
    if a["imgs"]:
        h.append("<h2>Camera frames</h2>")
        rows = []
        for n, i in a["imgs"].items():
            rows.append(f"<tr><td>{esc(n)}</td><td>{esc(', '.join(i['formats']))}</td><td class=num>{i['count']:,}</td>"
                        f"<td>{esc(', '.join(f'{w}×{hh}' for hh, w in i['shapes']))}</td>"
                        f"<td class=num>{i['dupes']}</td><td class=num>{i['decode_fail']}</td></tr>")
        h.append("<div class='card'><table><tr><th>topic</th><th>format</th><th class=num>frames</th><th>size</th>"
                 "<th class=num>identical consecutive</th><th class=num>decode failures</th></tr>" + "".join(rows) + "</table>")
        if a["sync"]:
            s = a["sync"]
            h.append(f"<p class='muted'>RGB/depth sync: {s['matched'] * 100:.1f} % of RGB frames have a depth frame with "
                     f"the same stamp (±1 ms), median offset {s['median_ms']:.1f} ms.</p>")
        h.append("</div>")
    cam_rows = [f"<tr><td>{esc(n)}</td><td class=num>{c['w']}×{c['h']}</td><td class=num>{c['fx']:.1f}</td>"
                f"<td class=num>{c['fy']:.1f}</td><td class=num>{c['cx']:.1f}</td><td class=num>{c['cy']:.1f}</td>"
                f"<td>{esc(c['model'])}</td><td class=num>{c['count']}</td></tr>"
                for n, cs in a["cams"].items() for c in cs]
    if cam_rows:
        h.append("<div class='card'><table><tr><th>camera_info</th><th class=num>size</th><th class=num>fx</th>"
                 "<th class=num>fy</th><th class=num>cx</th><th class=num>cy</th><th>model</th><th class=num>msgs</th></tr>"
                 + "".join(cam_rows) + "</table></div>")

    def cards(cells):
        out = ["<div class='previews'>"]
        for c in cells:
            img = f"<img alt='{esc(c['topic'])}' src='data:image/jpeg;base64,{c['b64']}'>" if c["b64"] else ""
            out.append(f"<div class='preview'>{img}<div class='cap'><b>{esc(c['topic'])}</b> · {esc(c['caption'])}</div></div>")
        return "".join(out) + "</div>"

    if previews or edges:
        h.append(f"<p class='muted'>Depth colour scale 0 – {MAX_DEPTH_M:g} m (near = red, far = blue, black = invalid):</p>"
                 f"<img alt='depth scale' style='max-width:{PREVIEW_W}px;width:100%;border-radius:4px' "
                 f"src='data:image/jpeg;base64,{depth_colorbar_b64()}'>")
    if edges:
        h.append("<h2>Start and end of recording</h2>"
                 f"<p class='muted'>Bag starts at 0.00 s (first message of any topic) and ends at {a['duration']:.2f} s "
                 "(last message of any topic). Times are bag receive times; the header stamp is in brackets.</p>")
        for e in edges:
            h.append(f"<div class='prow'><h3>{esc(e['title'])}</h3>{cards(e['cells'])}</div>")
    if previews:
        h.append("<h2>Previews</h2>")
        for row in previews:
            sp = f" · wheel speed {row['speed']:.2f} m/s" if row["speed"] is not None else ""
            h.append(f"<div class='prow'><h3>#{row['idx']} at {row['t']:.1f} s{sp}</h3>{cards(row['cells'])}</div>")

    h.append("<h2>TF</h2><div class='card'><table><tr><th>dynamic transform (/tf)</th><th class=num>rate Hz</th></tr>"
             + "".join(f"<tr><td>{esc(k)}</td><td class=num>{v:.1f}</td></tr>" for k, v in a["tf_rates"].items())
             + "</table></div>")
    h.append("</main></body></html>")
    return "\n".join(h)


def summary_of(a: dict) -> dict:
    wheel = a["odom"].get(REFERENCE_ODOM)
    return {
        "session": a["bag"].session, "bag": a["bag"].name, "status": overall(a["checks"]),
        "n_error": sum(c.level == "error" for c in a["checks"]), "n_warn": sum(c.level == "warn" for c in a["checks"]),
        "issues": [f"{c.title}" for c in a["checks"] if c.level != "ok"],
        "start": datetime.fromtimestamp(a["t0"] / 1e9, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(a["duration"], 2), "size_bytes": a["size"], "messages": a["n_msgs"],
        "topics": len(a["timing"]), "distance_m": round(wheel["length_m"], 2) if wheel else None,
        "p99_speed_ms": round(wheel["p99_speed"], 3) if wheel else None, "max_speed_ms": a["max_speed"],
        "standstill_s": round(a["standstill"], 2) if a["standstill"] is not None else None,
        "split": (a["manifest"] or {}).get("split"),
        "total_gaps": sum(len(tm.get("gaps", [])) for tm in a["timing"].values()),
        "topic_rates_hz": {n: round(tm.get("median_rate_hz", 0), 2) for n, tm in a["timing"].items()},
    }


def render_index(title: str, rows: list[dict], link, back: str | None) -> str:
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
         f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(title)}</title>"
         f"<style>{CSS}</style></head><body><main>"]
    if back:
        h.append(f"<p class='muted'><a href='{back}'>All sessions</a></p>")
    total = sum(r["duration_s"] for r in rows)
    dist = sum(r["distance_m"] or 0 for r in rows)
    h.append(f"<h1>{esc(title)}</h1><p class='muted'>{len(rows)} bags · {fmt_dur(total)} · "
             f"{fmt_bytes(sum(r['size_bytes'] for r in rows))} · {dist:.0f} m driven (wheel odom)</p>")
    h.append("<div class='card'><table><tr><th>status</th><th>session</th><th>bag</th><th>split</th><th>start (UTC)</th>"
             "<th class=num>duration</th><th class=num>size</th><th class=num>messages</th><th class=num>distance m</th>"
             "<th class=num>p99 speed / max</th><th class=num>standstill s</th><th class=num>gaps</th><th>issues</th></tr>")
    for r in rows:
        lvl = r["status"]
        txt = "OK" if lvl == "ok" else f"{r['n_error']} err, {r['n_warn']} warn"
        h.append(f"<tr><td>{badge(lvl, txt)}</td><td>{esc(r['session'])}</td>"
                 f"<td><a href='{link(r)}'>{esc(r['bag'])}</a></td><td>{esc(r['split'] or '—')}</td>"
                 f"<td>{esc(r['start'])}</td><td class=num>{fmt_dur(r['duration_s'])}</td>"
                 f"<td class=num>{fmt_bytes(r['size_bytes'])}</td><td class=num>{r['messages']:,}</td>"
                 f"<td class=num>{num(r['distance_m'])}</td>"
                 f"<td class=num>{num(r['p99_speed_ms'], '{:.2f}')} / {num(r['max_speed_ms'], '{:.1f}')}</td>"
                 f"<td class=num>{num(r['standstill_s'])}</td><td class=num>{r['total_gaps']}</td>"
                 f"<td class='wrap muted' style='min-width:260px'>{'<br>'.join(esc(i) for i in r['issues'])}</td></tr>")
    h.append("</table></div></main></body></html>")
    return "\n".join(h)


def write_indexes() -> None:
    summaries = []
    for f in sorted(REPORTS_DIR.glob("*/*/summary.json")):
        try:
            summaries.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    by_session: dict[str, list] = {}
    for s in summaries:
        by_session.setdefault(s["session"], []).append(s)
    for session, rows in by_session.items():
        rows.sort(key=lambda r: r["start"])
        (REPORTS_DIR / session / "index.html").write_text(
            render_index(session, rows, lambda r: f"{r['bag']}/report.html", "../index.html"))
    all_rows = sorted(summaries, key=lambda r: r["start"])
    (REPORTS_DIR / "index.html").write_text(
        render_index("Bag reports", all_rows, lambda r: f"{r['session']}/{r['bag']}/report.html", None))


# --- Main -----------------------------------------------------------------------
def process(bag: BagRef, n_previews: int, frame_stats: bool, manifest_row: dict | None) -> tuple[str, str]:
    t_start = time.time()
    raw = read_bag(bag, n_previews, frame_stats)
    a = analyze(bag, raw, manifest_row)
    a["timing_raw"] = {n: td.recv for n, td in raw["topics"].items()}
    plots = {
        "continuity": plot_continuity(a), "dt_hist": plot_dt_hist(a), "speed": plot_speed(a),
        "paths": plot_paths(a), "distances": plot_odom_distances(a), "imu": plot_imu(a),
        "joints": plot_joints(a),
    }
    previews = build_previews(a)
    edges = build_edge_frames(a)
    bag.out_dir.mkdir(parents=True, exist_ok=True)
    (bag.out_dir / "report.html").write_text(render_report(a, plots, previews, edges))
    summary = summary_of(a)
    (bag.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary["status"], f"{time.time() - t_start:.0f} s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", action="append", help="session folder name (repeatable)")
    p.add_argument("--bag", action="append", help="bag name or path to a bag directory (repeatable)")
    p.add_argument("--previews", type=int, default=6, help="number of RGB/depth preview moments (default: 6)")
    p.add_argument("--no-frame-stats", action="store_true",
                   help="do not decode every frame (faster, skips the brightness / valid-depth / decode checks)")
    p.add_argument("--skip-existing", action="store_true", help="skip bags that already have a report")
    p.add_argument("--jobs", type=int, default=1, help="number of bags processed in parallel (default: 1)")
    p.add_argument("--index-only", action="store_true", help="only rebuild the index pages from existing reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)
    if args.index_only:
        write_indexes()
        print(f"Indexes written to {REPORTS_DIR / 'index.html'}")
        return
    bags = select_bags(args)
    if args.skip_existing:
        bags = [b for b in bags if not (b.out_dir / "report.html").exists()]
    if not bags:
        sys.exit("No bags to process.")
    manifest = load_manifest()
    print(f"Generating reports for {len(bags)} bag(s) into {REPORTS_DIR}")

    failed = []
    task_args = [(b, args.previews, not args.no_frame_stats, manifest.get(b.name)) for b in bags]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futures = {ex.submit(process, *ta): ta[0] for ta in task_args}
            for fut in as_completed(futures):
                b = futures[fut]
                try:
                    status, took = fut.result()
                    print(f"  {STATUS_ICON[status]} {b.session}/{b.name} ({took})")
                except Exception:
                    failed.append(b)
                    print(f"  ✗ {b.session}/{b.name}: report failed\n{traceback.format_exc()}")
    else:
        for ta in task_args:
            b = ta[0]
            try:
                status, took = process(*ta)
                print(f"  {STATUS_ICON[status]} {b.session}/{b.name} ({took})")
            except Exception:
                failed.append(b)
                print(f"  ✗ {b.session}/{b.name}: report failed\n{traceback.format_exc()}")
    write_indexes()
    print(f"Index: {REPORTS_DIR / 'index.html'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
