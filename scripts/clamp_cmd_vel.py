#!/usr/bin/env python3
"""Write a copy of a bag with /rover/cmd_vel clamped the way the rover's swerve controller does.

The controller (rover_controller/swerve_controller.cpp, step 1) limits the magnitude of the
twist  m = sqrt(vx^2 + vy^2 + (d * wz)^2)  to max_linear_speed, keeping its direction
(d = rotation_scale_length = hypot(wheelbase / 2, track_width / 2)). Recorded /rover/cmd_vel is
the raw operator request, so this script scales (vx, vy, wz) by min(1, limit / m).

Only the clamp is reproduced. The controller's accel / jerk limiting and steering-alignment
scaling are not.

All other messages are copied unchanged (same topics, QoS profiles and timestamps). The source
bag is never modified.

Usage:
    python scripts/clamp_cmd_vel.py SRC_BAG DST_BAG --max-linear-speed 0.8
"""
import argparse
import math
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

WHEELBASE = 0.797    # m, from config/controllers.yaml in indomitus-rover-core
TRACK_WIDTH = 0.644  # m
CMD_VEL_TOPIC = "/rover/cmd_vel"


def clamp_twist(msg, limit: float, d: float) -> bool:
    """Scale the twist in place so its magnitude is at most `limit`. Returns True if it changed."""
    vx, vy, wz = msg.linear.x, msg.linear.y, msg.angular.z
    m = math.sqrt(vx * vx + vy * vy + (d * wz) ** 2)
    if m <= limit or m < 1e-12:
        return False
    k = limit / m
    msg.linear.x, msg.linear.y, msg.angular.z = vx * k, vy * k, wz * k
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="source bag directory")
    p.add_argument("dst", type=Path, help="output bag directory (must not exist)")
    p.add_argument("--max-linear-speed", type=float, required=True,
                   help="the controller's max_linear_speed during the recording, m/s")
    p.add_argument("--rotation-scale", type=float, default=math.hypot(WHEELBASE / 2, TRACK_WIDTH / 2),
                   help="d in the twist magnitude, m (default: hypot(wheelbase/2, track_width/2))")
    args = p.parse_args()

    if args.dst.exists():
        sys.exit(f"{args.dst} already exists")
    if args.max_linear_speed <= 0:
        sys.exit("--max-linear-speed must be positive")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    n_total = n_cmd = n_changed = 0
    with AnyReader([args.src]) as reader, Writer(args.dst, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        out = {}
        for c in reader.connections:
            out[c.id] = writer.add_connection(
                c.topic, c.msgtype, typestore=typestore,
                serialization_format=c.ext.serialization_format,
                offered_qos_profiles=c.ext.offered_qos_profiles)
        for c, ts, raw in reader.messages():
            n_total += 1
            if c.topic == CMD_VEL_TOPIC:
                n_cmd += 1
                msg = typestore.deserialize_cdr(raw, c.msgtype)
                if clamp_twist(msg, args.max_linear_speed, args.rotation_scale):
                    n_changed += 1
                    raw = typestore.serialize_cdr(msg, c.msgtype)
            writer.write(out[c.id], ts, raw)

    # Name the file like `ros2 bag record` does (<bag>_0.mcap) so the copy looks like an original bag.
    meta = args.dst / "metadata.yaml"
    for f in args.dst.glob("*.mcap"):
        new = f.with_name(f"{args.dst.name}_0.mcap")
        f.rename(new)
        meta.write_text(meta.read_text().replace(f.name, new.name))

    print(f"{args.src.name} -> {args.dst}")
    print(f"  {n_total} messages copied; {CMD_VEL_TOPIC}: {n_changed} of {n_cmd} clamped "
          f"to {args.max_linear_speed} m/s (d = {args.rotation_scale:.4f} m)")


if __name__ == "__main__":
    main()
