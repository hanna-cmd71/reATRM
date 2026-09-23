#!/usr/bin/env python3
"""Drag-teach recorder, triggered by the VT03 trigger button.

Press the trigger once to start recording and once to stop; each segment is written
as its own CSV next to the other trajectories. The output is trimmed of the still
parts at both ends, and the action table snippet is printed at the end.

    python3 teach_record.py --name LF_Rising

Then in arm_actions.yaml:

    - type: trajectory
      file: /home/ws/src/bringup/config/trajectories/LF_Rising.csv
      vel: 0.5
      acc: 0.5

Notes:
  - The trigger is safe while teaching: the calibration relatch only accepts it with
    the mode switch on the left, so it is refused in drag mode.
  - Put the arm at the start of the segment before starting. The replay begins from
    the first recorded sample.
"""

import argparse
import csv
import os
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

NUM_JOINTS = 6
TRIGGER_TOPIC = "/vt03/trigger"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "config", "trajectories"))


class Recorder(Node):
    """Sample /joint_states at a fixed rate and toggle on the trigger button."""

    def __init__(self, rate_hz: float = 200.0) -> None:
        super().__init__("arm_teach_recorder")
        self.rows: list[list[float]] = []
        self.latest: list[float] | None = None
        self.recording = False
        self.segments: list[list[list[float]]] = []
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos)
        self.create_subscription(Bool, TRIGGER_TOPIC, self._on_trigger, qos)
        self.create_timer(1.0 / rate_hz, self._sample)
        self.got_trigger = False
        self.trigger_held = False

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_joint_state(self, msg: JointState) -> None:
        values = [float("nan")] * NUM_JOINTS
        for name, position in zip(msg.name, msg.position):
            if not name.startswith("joint_"):
                continue
            index = int(name.split("_", 1)[1]) - 1
            if 0 <= index < NUM_JOINTS:
                values[index] = float(position)
        self.latest = values

    def _on_trigger(self, msg: Bool) -> None:
        self.got_trigger = True
        pressed = bool(msg.data)
        if pressed and not self.trigger_held:
            self.toggle()
        self.trigger_held = pressed

    def toggle(self) -> None:
        if not self.recording:
            self.rows = []
            self.recording = True
            print("REC  started", flush=True)
            return
        self.recording = False
        if len(self.rows) >= 20:
            self.segments.append(self.rows)
            print(f"REC  stopped ({len(self.rows)} samples)", flush=True)
        else:
            print("REC  stopped, too short, discarded", flush=True)

    def _sample(self) -> None:
        if not self.recording or self.latest is None:
            return
        if any(value != value for value in self.latest):
            return
        self.rows.append([self.now_seconds()] + list(self.latest))


def trim_still(rows, threshold: float):
    """Drop the still head and tail so the replay starts moving right away."""
    if not rows:
        return rows
    reference = rows[0][1:]
    start = 0
    for index, row in enumerate(rows):
        if max(abs(row[1 + j] - reference[j]) for j in range(NUM_JOINTS)) > threshold:
            start = index
            break
    end = len(rows) - 1
    for index in range(len(rows) - 1, -1, -1):
        if max(abs(rows[index][1 + j] - reference[j]) for j in range(NUM_JOINTS)) > threshold:
            end = index
            break
    return rows[max(0, start - 5) : min(len(rows), end + 6)]


def write_segment(path: str, rows) -> float:
    origin = rows[0][0]
    for row in rows:
        row[0] -= origin
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"] + [f"q{i}" for i in range(1, NUM_JOINTS + 1)])
        writer.writerows(rows)
    return rows[-1][0]


def next_path(directory: str, name: str, index: int) -> str:
    suffix = "" if index == 1 else f"_{index}"
    return os.path.join(directory, f"{name}{suffix}.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Drag-teach recorder, trigger toggled")
    parser.add_argument("--name", default="teach", help="output base name")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="output directory")
    parser.add_argument("--rate", type=float, default=200.0, help="sample rate, Hz")
    parser.add_argument("--trim", type=float, default=0.005, help="still threshold, rad")
    args = parser.parse_args()

    rclpy.init()
    node = Recorder(rate_hz=args.rate)
    spinning = True

    def spin():
        while spinning:
            rclpy.spin_once(node, timeout_sec=0.05)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()

    for _ in range(100):
        if node.latest is not None:
            break
        threading.Event().wait(0.1)
    if node.latest is None:
        print("no /joint_states, is start running?")
        return 1

    print(f"ready. trigger to start/stop. output -> {args.dir}/{args.name}[_n].csv")
    if not node.got_trigger:
        print(f"waiting for {TRIGGER_TOPIC} ...")
    try:
        while True:
            threading.Event().wait(0.2)
    except KeyboardInterrupt:
        pass

    spinning = False
    thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()

    if not node.got_trigger:
        print(f"never saw {TRIGGER_TOPIC}; check the value_broadcaster forward_list")
        return 1
    if not node.segments:
        print("no segment recorded")
        return 1

    print()
    for index, rows in enumerate(node.segments, start=1):
        rows = trim_still(rows, args.trim) if args.trim > 0 else rows
        path = next_path(args.dir, args.name, index)
        duration = write_segment(path, rows)
        print(f"{path}   {duration:.2f}s, {len(rows)} samples")
        for joint in range(NUM_JOINTS):
            values = [row[1 + joint] for row in rows]
            print(
                f"  q{joint + 1}: {values[0]:+.4f} -> {values[-1]:+.4f}"
                f"   [{min(values):+.4f}, {max(values):+.4f}]"
            )
    print("\narm_actions.yaml:")
    for index in range(1, len(node.segments) + 1):
        print("  - type: trajectory")
        print(f"    file: {next_path(args.dir, args.name, index)}")
        print("    vel: 0.5")
        print("    acc: 0.5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
