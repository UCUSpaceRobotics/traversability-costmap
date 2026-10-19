#!/usr/bin/env python3
"""Check topics, record a ROS 2 bag, monitor it while recording and summarise it afterwards.

Runs where the ROS 2 graph is visible (inside the rover container) with ROS 2 sourced. Usually it is
started from the laptop by scripts/record_bag.sh, but it also works on its own:

    bag_recorder.py check                                         # pre-flight checks only
    bag_recorder.py record --ride-type mixed --speed 0.5          # record until Ctrl+C
    bag_recorder.py record --ride-type flat --speed 0.8 --index 2 --duration 5m
    bag_recorder.py summary [BAG]                                 # message counts vs. expected (newest bag by default)
    bag_recorder.py list                                          # bags in the output directory
    bag_recorder.py path [BAG]                                    # absolute path of a bag (newest by default)
    bag_recorder.py stop                                          # stop a running recording, e.g. after an ssh drop

`record` runs the pre-flight checks first (topic exists with the right type, has publishers, rate close
to the expected one, enough disk space) and starts only if they pass (override with --force). While
recording it checks the topics every `checks.monitor_interval` seconds and prints warnings. After it
stops (Ctrl+C, SIGTERM, SIGHUP or --duration) it prints the summary.

Topics, expected rates and recording settings come from a YAML config (--config, default: recording.yaml
next to this script, else config/recording.yaml in the repository).

Exit codes: 0 = ok, 1 = failed (nothing recorded), 2 = recorded but the summary shows problems.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_WARN = 0, 1, 2
OK, WARN, ERR = "ok", "warn", "error"

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_CANDIDATES = [SCRIPT_DIR / "recording.yaml", SCRIPT_DIR.parent.parent / "config" / "recording.yaml"]
PID_FILE_NAME = ".bag_recorder.pid"
LOG_DIR_NAME = ".logs"
STOP_TIMEOUT_S = 60.0     # time ros2 bag record gets to finalise the bag after SIGINT


class Fail(Exception):
    """Fatal error with a message for the user."""


# --- Output ---------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


MARK = {OK: _c("32", "✓"), WARN: _c("33", "!"), ERR: _c("31", "✗")}


def info(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    info(_c("34", f"\n>>> {msg}"))


def warn(msg: str) -> None:
    info(_c("33", f"[WARN] {msg}"))


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def parse_duration(text: str) -> float:
    """'300', '90s', '5m', '1h30m' -> seconds."""
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", text)
    if not parts or "".join(n + u for n, u in parts) != text:
        raise argparse.ArgumentTypeError(f"invalid duration '{text}' (examples: 300, 90s, 5m, 1h30m)")
    return sum(float(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in parts)


# --- Config ---------------------------------------------------------------------

TOPIC_KEYS = {"type", "rate", "tolerance", "required", "latched"}


@dataclass(frozen=True)
class TopicSpec:
    name: str
    type: str | None = None          # None = take whatever the graph says
    rate: float | None = None        # expected Hz; None = event-driven, not rate-checked
    tolerance: float | None = None   # None = checks.rate_tolerance (live) / checks.count_tolerance (summary)
    required: bool = True
    latched: bool = False


@dataclass(frozen=True)
class Config:
    output_dir: Path
    ride_types: tuple[str, ...]
    ride_types_without_speed: tuple[str, ...]
    storage: str | None
    storage_preset_profile: str | None
    max_bag_duration: int
    extra_args: tuple[str, ...]
    discovery_timeout: float
    rate_window: float
    rate_tolerance: float
    monitor_interval: float
    count_tolerance: float
    min_free_gb: float
    topics: dict[str, TopicSpec]


def load_config(path: Path) -> Config:
    import yaml  # imported lazily so that --help works without PyYAML

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise Fail(f"Cannot read config {path}: {e}") from e
    try:
        naming, record, checks = raw.get("naming") or {}, raw.get("record") or {}, raw.get("checks") or {}
        topics = {}
        for name, spec in (raw.get("topics") or {}).items():
            spec = spec or {}
            if unknown := set(spec) - TOPIC_KEYS:
                raise Fail(f"{path}: unknown keys for topic {name}: {', '.join(sorted(unknown))}")
            topics[name] = TopicSpec(
                name=name,
                type=spec.get("type"),
                rate=float(spec["rate"]) if spec.get("rate") is not None else None,
                tolerance=float(spec["tolerance"]) if spec.get("tolerance") is not None else None,
                required=bool(spec.get("required", True)),
                latched=bool(spec.get("latched", False)),
            )
        if not topics:
            raise Fail(f"{path}: no topics configured")
        return Config(
            output_dir=Path(raw.get("output_dir", "~/bags")).expanduser(),
            ride_types=tuple(naming.get("ride_types", [])),
            ride_types_without_speed=tuple(naming.get("ride_types_without_speed", [])),
            storage=record.get("storage"),
            storage_preset_profile=record.get("storage_preset_profile"),
            max_bag_duration=int(record.get("max_bag_duration", 0)),
            extra_args=tuple(str(a) for a in record.get("extra_args") or []),
            discovery_timeout=float(checks.get("discovery_timeout", 5.0)),
            rate_window=float(checks.get("rate_window", 3.0)),
            rate_tolerance=float(checks.get("rate_tolerance", 0.25)),
            monitor_interval=float(checks.get("monitor_interval", 10.0)),
            count_tolerance=float(checks.get("count_tolerance", 0.1)),
            min_free_gb=float(checks.get("min_free_gb", 0.0)),
            topics=topics,
        )
    except (TypeError, ValueError, AttributeError) as e:
        raise Fail(f"{path}: invalid value: {e}") from e


def topic_specs(cfg: Config, extra_topics: list[str]) -> dict[str, TopicSpec]:
    """Configured topics plus extra --topic ones (type taken from the graph, no rate check)."""
    specs = dict(cfg.topics)
    for name in extra_topics:
        specs.setdefault(name, TopicSpec(name))
    return specs


# --- Bags on disk ---------------------------------------------------------------

def is_bag(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".") and (
        (path / "metadata.yaml").is_file() or any(path.glob("*.mcap")) or any(path.glob("*.db3")))


def find_bags(output_dir: Path) -> list[Path]:
    """Bags in output_dir, oldest first."""
    if not output_dir.is_dir():
        return []
    return sorted((p for p in output_dir.iterdir() if is_bag(p)), key=lambda p: p.stat().st_mtime)


def resolve_bag(cfg: Config, name: str | None) -> Path:
    if name is None:
        bags = find_bags(cfg.output_dir)
        if not bags:
            raise Fail(f"No bags in {cfg.output_dir}")
        return bags[-1].resolve()
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = cfg.output_dir / path
    if not is_bag(path):
        raise Fail(f"Bag not found: {path}")
    return path.resolve()


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def read_metadata(bag: Path) -> dict:
    import yaml

    meta_path = bag / "metadata.yaml"
    if not meta_path.is_file():
        raise Fail(f"{bag} has no metadata.yaml (recording not finalised?). "
                   f"Try: ros2 bag reindex {bag} -s mcap")
    return yaml.safe_load(meta_path.read_text())["rosbag2_bagfile_information"]


def bag_name(args: argparse.Namespace, cfg: Config) -> str:
    if args.name:
        return args.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [args.ride_type]
    if args.speed is not None:
        parts.append(f"{round(args.speed * 10):02d}ms")
    if args.index is not None:
        parts.append(str(args.index))
    return "_".join(parts + [stamp])


def validate_naming(args: argparse.Namespace, cfg: Config, parser: argparse.ArgumentParser) -> None:
    if args.name:
        return
    if not args.ride_type:
        parser.error("--ride-type (or --name) is required")
    if cfg.ride_types and args.ride_type not in cfg.ride_types:
        parser.error(f"--ride-type must be one of: {', '.join(cfg.ride_types)}")
    if args.ride_type in cfg.ride_types_without_speed:
        if args.speed is not None:
            parser.error(f"--speed is not used for ride type '{args.ride_type}'")
    elif args.speed is None:
        parser.error(f"--speed is required for ride type '{args.ride_type}'")
    if args.index is not None and args.index < 1:
        parser.error("--index starts from 1")


# --- Running recording (pid file) ----------------------------------------------

def running_pid(cfg: Config) -> int | None:
    """PID of a recorder that is currently recording into cfg.output_dir, if any."""
    try:
        pid = int((cfg.output_dir / PID_FILE_NAME).read_text())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists() and b"bag_recorder" not in cmdline.read_bytes():
        return None  # stale pid file, the pid was reused
    return pid


# --- ROS graph probe -------------------------------------------------------------

@dataclass
class TopicStatus:
    topic: str
    level: str
    publishers: int
    rate: float | None
    problem: str = ""


class GraphProbe:
    """rclpy node that watches topics: types, publishers and message rates (raw, no deserialisation)."""

    def __init__(self, specs: dict[str, TopicSpec]):
        import rclpy

        try:  # we handle signals ourselves
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        except ImportError:
            rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("bag_recorder_probe")
        self.specs = specs
        self._subs: dict = {}
        self._counts = dict.fromkeys(specs, 0)
        self._window_start = time.monotonic()

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            self._rclpy.spin_once(self.node, timeout_sec=min(remaining, 0.1))

    def graph(self) -> dict[str, tuple[list[str], int]]:
        """{topic: (types in the graph, number of publishers)} for every watched topic."""
        types = dict(self.node.get_topic_names_and_types())
        return {t: (types.get(t, []), self.node.count_publishers(t)) for t in self.specs}

    def wait_for_publishers(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not all(n for _, n in self.graph().values()):
            self.spin(0.2)

    def subscribe(self, graph: dict[str, tuple[list[str], int]]) -> None:
        """Subscribe to every watched topic that exists with the expected type (and is not subscribed yet)."""
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message

        # best effort + volatile matches any publisher; latched topics need reliable + transient local
        stream_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        latched_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic, spec in self.specs.items():
            types, _ = graph[topic]
            msg_type = spec.type or (types[0] if types else None)
            if topic in self._subs or msg_type not in types:
                continue
            try:
                msg_cls = get_message(msg_type)
            except (AttributeError, ModuleNotFoundError, ValueError):
                warn(f"Cannot import message type {msg_type} for {topic}, its rate is not measured")
                continue
            self._subs[topic] = self.node.create_subscription(
                msg_cls, topic, lambda _msg, t=topic: self._on_msg(t),
                latched_qos if spec.latched else stream_qos, raw=True)

    def _on_msg(self, topic: str) -> None:
        self._counts[topic] += 1

    def take_window(self) -> dict[str, tuple[int, float]]:
        """{topic: (messages, rate in Hz)} since the previous call; starts a new window."""
        now = time.monotonic()
        elapsed = max(now - self._window_start, 1e-6)
        window = {t: (c, c / elapsed) for t, c in self._counts.items()}
        self._counts = dict.fromkeys(self.specs, 0)
        self._window_start = now
        return window


def evaluate(spec: TopicSpec, types: list[str], publishers: int, count: int, rate: float,
             tolerance: float, live: bool) -> TopicStatus:
    """Status of one topic; `live` = during recording (latched messages are not re-delivered then)."""
    def bad(problem: str) -> TopicStatus:
        return TopicStatus(spec.name, ERR if spec.required else WARN, publishers, rate, problem)

    if not types:
        return bad("topic does not exist")
    if spec.type and spec.type not in types:
        return bad(f"type {', '.join(types)}, expected {spec.type}")
    if publishers == 0:
        return bad("no publishers")
    if spec.latched:
        if not live and count == 0:
            return bad("no latched message received")
    elif spec.rate is not None:
        if count == 0:
            return bad("no messages")
        tol = spec.tolerance if spec.tolerance is not None else tolerance
        if abs(rate / spec.rate - 1) > tol:
            return bad(f"rate {rate:.1f} Hz, expected {spec.rate:g} Hz ±{tol:.0%}")
    return TopicStatus(spec.name, OK, publishers, rate)


def check_topics(probe: GraphProbe, tolerance: float, live: bool) -> list[TopicStatus]:
    graph = probe.graph()
    probe.subscribe(graph)  # topics that appeared since the last check
    window = probe.take_window()
    return [evaluate(spec, *graph[t], *window[t], tolerance=tolerance, live=live) for t, spec in probe.specs.items()]


def print_status_table(statuses: list[TopicStatus], specs: dict[str, TopicSpec]) -> None:
    width = max(len(s.topic) for s in statuses)
    info(f"\n   {'topic':<{width}}  {'pubs':>4}  {'rate Hz':>7}  {'exp Hz':>6}  problem")
    for s in statuses:
        spec = specs[s.topic]
        expected = "latch" if spec.latched else ("-" if spec.rate is None else f"{spec.rate:g}")
        rate = "-" if s.rate is None or spec.latched or not s.publishers else f"{s.rate:.1f}"
        info(f"{MARK[s.level]}  {s.topic:<{width}}  {s.publishers:>4}  {rate:>7}  {expected:>6}  {s.problem}")
    info()


# --- Commands ------------------------------------------------------------------

def preflight(probe: GraphProbe, cfg: Config) -> bool:
    """Run the pre-flight checks and print the result; True if nothing blocks the recording."""
    step(f"Pre-flight checks ({len(probe.specs)} topics)")
    info(f"Waiting up to {cfg.discovery_timeout:g} s for publishers, then measuring rates for {cfg.rate_window:g} s...")
    probe.wait_for_publishers(cfg.discovery_timeout)
    probe.subscribe(probe.graph())
    probe.take_window()
    probe.spin(cfg.rate_window)
    statuses = check_topics(probe, cfg.rate_tolerance, live=False)
    print_status_table(statuses, probe.specs)

    ok = not any(s.level == ERR for s in statuses)
    n_warn = sum(s.level == WARN for s in statuses)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    free = free_bytes(cfg.output_dir)
    if free < cfg.min_free_gb * 1024**3:
        info(f"{MARK[ERR]}  only {fmt_size(free)} free in {cfg.output_dir} (minimum {cfg.min_free_gb:g} GB)")
        ok = False
    else:
        info(f"{MARK[OK]}  {fmt_size(free)} free in {cfg.output_dir}")
    if ok:
        info(_c("32", "Pre-flight checks passed") + (f" ({n_warn} warning(s))" if n_warn else ""))
    else:
        info(_c("31", "Pre-flight checks failed"))
    return ok


def cmd_check(args: argparse.Namespace, cfg: Config) -> int:
    probe = GraphProbe(topic_specs(cfg, args.topic))
    try:
        return EXIT_OK if preflight(probe, cfg) else EXIT_FAIL
    finally:
        probe.close()


def record_command(cfg: Config, out: Path, topics: list[str]) -> list[str]:
    cmd = ["ros2", "bag", "record", *topics, "--output", str(out)]
    if cfg.storage:
        cmd += ["--storage", cfg.storage]
    if cfg.storage_preset_profile:
        cmd += ["--storage-preset-profile", cfg.storage_preset_profile]
    if cfg.max_bag_duration > 0:
        cmd += ["--max-bag-duration", str(cfg.max_bag_duration)]
    return cmd + list(cfg.extra_args)


class StopRequest:
    """Turns SIGINT / SIGTERM / SIGHUP into a stop request instead of killing the recorder."""

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.reason: str | None = None
        self._previous = {s: signal.signal(s, self._handle) for s in self.SIGNALS}

    def _handle(self, signum: int, _frame) -> None:
        if signum == signal.SIGHUP:
            # terminal / ssh session is gone: keep going silently, writing to it would fail
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.dup2(devnull, sys.stderr.fileno())
        if self.reason is None:
            self.reason = signal.Signals(signum).name
            info(f"\n{signal.Signals(signum).name} received, stopping the recording...")
        else:
            info("Already stopping, wait until the bag is finalised...")

    def restore(self) -> None:
        for s, handler in self._previous.items():
            signal.signal(s, handler)


def stop_process(proc: subprocess.Popen) -> None:
    """Stop `ros2 bag record` gracefully, so that it writes metadata.yaml."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        warn(f"ros2 bag record did not stop within {STOP_TIMEOUT_S:g} s, terminating it")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


class LiveMonitor:
    """Periodic checks during recording: topics, publishers, rates and disk space."""

    def __init__(self, probe: GraphProbe | None, cfg: Config, bag: Path, duration: float | None):
        self.probe, self.cfg, self.bag, self.duration = probe, cfg, bag, duration

    def report(self, elapsed: float) -> None:
        size, free = dir_size(self.bag), free_bytes(self.cfg.output_dir)
        clock = fmt_duration(elapsed) + (f" / {fmt_duration(self.duration)}" if self.duration else "")
        line = f"[{clock}] bag {fmt_size(size)}, {fmt_size(free)} free"
        bad = []
        if self.probe is not None:
            statuses = check_topics(self.probe, self.cfg.rate_tolerance, live=True)
            bad = [s for s in statuses if s.level != OK]
            line += " | " + (_c("33", f"{len(bad)} topic(s) with problems") if bad
                             else _c("32", f"all {len(statuses)} topics OK"))
        info(line)
        for s in bad:
            warn(f"{s.topic}: {s.problem}")
        if free < self.cfg.min_free_gb * 1024**3:
            warn(f"low disk space: {fmt_size(free)} free in {self.cfg.output_dir}")


def record(cfg: Config, specs: dict[str, TopicSpec], out: Path, duration: float | None,
           probe: GraphProbe | None) -> None:
    """Run `ros2 bag record` until a signal, the duration or a crash; raise Fail on a crash."""
    cmd = record_command(cfg, out, list(specs))
    log_path = cfg.output_dir / LOG_DIR_NAME / f"{out.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = cfg.output_dir / PID_FILE_NAME

    step(f"Recording {out}")
    info(" ".join(cmd))
    info(f"recorder log: {log_path}")
    info(("Stops after " + fmt_duration(duration) + " or on Ctrl+C.") if duration else "Press Ctrl+C to stop.")
    if cfg.monitor_interval > 0:
        info(f"Checking topics every {cfg.monitor_interval:g} s.\n")

    stop = StopRequest()
    pid_file.write_text(str(os.getpid()))
    monitor = LiveMonitor(probe, cfg, out, duration)
    try:
        with log_path.open("w") as log:
            # own session: Ctrl+C reaches only us, we forward it once ros2 bag should stop
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start = last_check = time.monotonic()
        try:
            while proc.poll() is None and stop.reason is None:
                elapsed = time.monotonic() - start
                if duration and elapsed >= duration:
                    info(f"\nDuration {fmt_duration(duration)} reached, stopping the recording...")
                    break
                if probe is not None:
                    probe.spin(0.2)
                else:
                    time.sleep(0.2)
                if cfg.monitor_interval > 0 and time.monotonic() - last_check >= cfg.monitor_interval:
                    monitor.report(elapsed)
                    last_check = time.monotonic()
            crashed = proc.poll() is not None
        finally:
            stop_process(proc)
        info(f"Recording stopped after {fmt_duration(time.monotonic() - start)}")
        if crashed:
            raise Fail(f"ros2 bag record exited unexpectedly (code {proc.returncode}). "
                       f"Last lines of {log_path}:\n{tail(log_path)}")
    finally:
        pid_file.unlink(missing_ok=True)
        stop.restore()


def cmd_record(args: argparse.Namespace, cfg: Config) -> int:
    if pid := running_pid(cfg):
        raise Fail(f"Another recording is running (pid {pid}). Stop it with `bag_recorder.py stop`.")
    if not shutil.which("ros2"):
        raise Fail("`ros2` not found, source ROS 2 first")
    out = cfg.output_dir / bag_name(args, cfg)
    if out.exists():
        raise Fail(f"{out} already exists")
    specs = topic_specs(cfg, args.topic)

    probe = GraphProbe(specs)
    try:
        if not preflight(probe, cfg):
            if not args.force:
                raise Fail("Not recording (use --force to record anyway)")
            warn("--force: recording despite failed checks")
        if cfg.monitor_interval <= 0:  # nothing to watch, free the subscriptions
            probe.close()
            probe = None
        record(cfg, specs, out, args.duration, probe)
    finally:
        if probe is not None:
            probe.close()
    return EXIT_OK if summarize(out, specs, cfg) == 0 else EXIT_WARN


def summarize(bag: Path, specs: dict[str, TopicSpec], cfg: Config) -> int:
    """Print message counts vs. rate * duration; return the number of problems."""
    meta = read_metadata(bag)
    duration = meta["duration"]["nanoseconds"] / 1e9
    counts = {t["topic_metadata"]["name"]: t["message_count"] for t in meta["topics_with_message_count"]}
    start = datetime.fromtimestamp(meta["starting_time"]["nanoseconds_since_epoch"] / 1e9, timezone.utc)

    step(f"Summary of {bag.name}")
    info(f"path:      {bag}")
    info(f"start:     {start:%Y-%m-%d %H:%M:%S} UTC")
    info(f"duration:  {fmt_duration(duration)} ({duration:.1f} s)")
    info(f"size:      {fmt_size(dir_size(bag))} in {len(meta.get('relative_file_paths', []))} file(s)")
    info(f"messages:  {meta['message_count']}")

    rows, problems = [], 0
    for topic in [*specs, *(t for t in counts if t not in specs)]:
        spec, count = specs.get(topic), counts.get(topic)
        expected, ratio, level, problem = "-", "", OK, ""
        if spec is None:
            problem = "not in config"
        elif count is None:
            level, problem = (ERR if spec.required else WARN), "not in bag"
        elif spec.latched:
            expected = "≥1"
            if count < 1:
                level, problem = WARN, "no message"
        elif spec.rate is not None and duration > 0:
            exp = spec.rate * duration
            expected, ratio = f"{exp:.0f}", f"{count / exp:.0%}"
            tol = spec.tolerance if spec.tolerance is not None else cfg.count_tolerance
            if abs(count / exp - 1) > tol:
                level, problem = WARN, f"outside ±{tol:.0%}"
        problems += level != OK
        rows.append((level, topic, "-" if count is None else str(count), expected, ratio, problem))

    width = max(len(r[1]) for r in rows)
    info(f"\n   {'topic':<{width}}  {'count':>8}  {'expected':>8}  {'ratio':>5}  problem")
    for level, topic, count, expected, ratio, problem in rows:
        info(f"{MARK[level]}  {topic:<{width}}  {count:>8}  {expected:>8}  {ratio:>5}  {problem}")
    info()
    info(_c("32", "All topics match the expectations") if problems == 0
         else _c("33", f"{problems} topic(s) do not match the expectations, check the bag before using it"))
    return problems


def cmd_summary(args: argparse.Namespace, cfg: Config) -> int:
    return EXIT_OK if summarize(resolve_bag(cfg, args.bag), topic_specs(cfg, args.topic), cfg) == 0 else EXIT_WARN


def cmd_list(args: argparse.Namespace, cfg: Config) -> int:
    bags = find_bags(cfg.output_dir)
    if not bags:
        info(f"No bags in {cfg.output_dir}")
        return EXIT_OK
    recording = running_pid(cfg) is not None
    width = max(len(b.name) for b in bags)
    info(f"{cfg.output_dir}:")
    for i, bag in enumerate(bags):
        try:
            duration = fmt_duration(read_metadata(bag)["duration"]["nanoseconds"] / 1e9)
        except (Fail, KeyError, TypeError):
            duration = "recording" if recording and i == len(bags) - 1 else "no metadata"
        info(f"  {bag.name:<{width}}  {duration:>11}  {fmt_size(dir_size(bag)):>10}")
    return EXIT_OK


def cmd_path(args: argparse.Namespace, cfg: Config) -> int:
    print(resolve_bag(cfg, args.bag))
    return EXIT_OK


def cmd_stop(args: argparse.Namespace, cfg: Config) -> int:
    pid = running_pid(cfg)
    if pid is None:
        info("No recording is running")
        return EXIT_FAIL
    info(f"Stopping the recording (pid {pid})...")
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + STOP_TIMEOUT_S + 15
    while running_pid(cfg) == pid and time.monotonic() < deadline:
        time.sleep(0.5)
    if running_pid(cfg) == pid:
        raise Fail(f"Recorder (pid {pid}) is still running")
    info("Stopped. Check the bag with `bag_recorder.py summary`.")
    return EXIT_OK


# --- CLI ------------------------------------------------------------------------

def default_config() -> Path:
    return next((p for p in CONFIG_CANDIDATES if p.is_file()), CONFIG_CANDIDATES[-1])


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=default_config(),
                        help="YAML config with topics and settings (default: %(default)s)")
    common.add_argument("--out-dir", type=Path, help="bag directory, overrides output_dir from the config")
    topics = argparse.ArgumentParser(add_help=False)
    topics.add_argument("--topic", action="append", default=[], metavar="TOPIC",
                        help="extra topic to check / record, no rate check (repeatable)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("check", parents=[common, topics], help="run the pre-flight checks only")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("record", parents=[common, topics], help="check, record, monitor and summarise a bag")
    p.add_argument("--ride-type", help="ride type, first part of the bag name (see naming.ride_types)")
    p.add_argument("--speed", type=float, help="max speed in m/s (joystick fully forward), e.g. 0.5")
    p.add_argument("--index", type=int, help="number of a repeated ride with the same type and speed (1, 2, ...)")
    p.add_argument("--name", help="full bag name instead of --ride-type/--speed/--index")
    p.add_argument("--duration", type=parse_duration, metavar="TIME",
                   help="stop automatically after TIME (300, 90s, 5m, 1h30m); default: until Ctrl+C")
    p.add_argument("--monitor-interval", type=float, metavar="SEC",
                   help="seconds between checks while recording, 0 = off (default: from config)")
    p.add_argument("--force", action="store_true", help="record even if the pre-flight checks fail")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("summary", parents=[common, topics], help="message counts of a bag vs. expected")
    p.add_argument("bag", nargs="?", help="bag name or path (default: newest bag)")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("list", parents=[common], help="list the bags in the output directory")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("path", parents=[common], help="print the absolute path of a bag")
    p.add_argument("bag", nargs="?", help="bag name or path (default: newest bag)")
    p.set_defaults(func=cmd_path)

    p = sub.add_parser("stop", parents=[common], help="stop a running recording (finalises the bag)")
    p.set_defaults(func=cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.out_dir:
            cfg = replace(cfg, output_dir=args.out_dir.expanduser())
        if args.command == "record":
            validate_naming(args, cfg, parser)
            if args.monitor_interval is not None:
                cfg = replace(cfg, monitor_interval=args.monitor_interval)
        return args.func(args, cfg)
    except Fail as e:
        print(_c("31", f"[ERROR] {e}"), file=sys.stderr, flush=True)
        return EXIT_FAIL
    except KeyboardInterrupt:
        print(_c("31", "\n[ERROR] Aborted"), file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
