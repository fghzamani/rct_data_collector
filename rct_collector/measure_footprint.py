#!/usr/bin/env python3
"""measure_footprint.py — capture a matching (joints, footprint) pair for one arm pose.

Hold TIAGo's arm in the pose you want (tuck it, or extend it to the side), then
run this. It reads the *current* arm joint angles from /joint_states AND the live
TF of every arm/wrist/gripper link, projects those links onto the ground plane in
the base frame, and returns:

  * the 7 arm joint targets  -> ARM_CONFIGS[label]["joints"]
  * the measured footprint    -> ARM_CONFIGS[label]["footprint"]

Because the polygon is derived from the SAME physical pose as the joints, the two
halves of the footprint variable can't drift apart — which is the whole point:
do(F_t = <label>) then means one coherent physical state, and the costmap geometry
Nav2 plans against is the geometry the robot actually has.

Workflow
--------
  # tucked baseline (arm already home):
  ros2 run rct_collector measure_footprint --label tucked

  # lateral pose — first drive the arm out to the side, e.g.:
  ros2 action send_goal /arm_controller/follow_joint_trajectory \
      control_msgs/action/FollowJointTrajectory \
      "{trajectory: {joint_names: [arm_1_joint,arm_2_joint,arm_3_joint,arm_4_joint,arm_5_joint,arm_6_joint,arm_7_joint], \
        points: [{positions: [0.0,0.0,0.0,0.0,0.0,0.0,0.0], time_from_start: {sec: 4}}]}}"
  # ...tweak the positions, watch RViz until the arm points to the side, then:
  ros2 run rct_collector measure_footprint --label lateral

Paste the printed block into ARM_CONFIGS in scripts/param_space.py. Since the
costmap `presets` are generated from ARM_CONFIGS, updating that one dict updates
both the costmap footprint AND trial_runner's collision checker together.
"""
import argparse
import math
import re
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tf2_ros


ARM_JOINT_NAMES = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
]

# ── pure-geometry helpers (numpy only, no scipy/shapely) ─────────────────────

def convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def circle_pts(c, r, n=8):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([c[0] + r*np.cos(a), c[1] + r*np.sin(a)], axis=1)


def octagon(r, n=12):
    return circle_pts((0.0, 0.0), r, n)


def densify(a, b, step=0.04):
    d = math.hypot(b[0]-a[0], b[1]-a[1])
    k = max(1, int(d/step))
    return np.stack([np.linspace(a[0], b[0], k+1), np.linspace(a[1], b[1], k+1)], axis=1)


def drop_collinear(poly, tol=1e-3):
    out, n = [], len(poly)
    for i in range(n):
        a, b, c = poly[i-1], poly[i], poly[(i+1) % n]
        area2 = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        if area2 > tol:
            out.append(b)
    return np.array(out, dtype=float)


def inscribed_radius(poly):
    """Min distance from origin to any edge (0 if origin outside — shouldn't be)."""
    n = len(poly)
    best = float("inf")
    for i in range(n):
        a, b = poly[i], poly[(i+1) % n]
        ab = b - a
        t = np.clip(-np.dot(a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
        proj = a + t*ab
        best = min(best, math.hypot(proj[0], proj[1]))
    return best


class FootprintMeasurer(Node):
    def __init__(self, args):
        super().__init__("footprint_measurer")
        self.args = args
        self._latest_js = None
        self.create_subscription(JointState, args.joint_topic, self._js_cb, 10)
        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

    def _js_cb(self, msg):
        self._latest_js = msg

    def _discover_link_frames(self):
        """All TF frames matching the link filter (arm/wrist/gripper/hand)."""
        try:
            frames_yaml = self._tf_buf.all_frames_as_yaml()
        except Exception:
            frames_yaml = ""
        names = re.findall(r"^([\w/]+):", frames_yaml, flags=re.MULTILINE)
        pat = re.compile(self.args.link_filter)
        hits = sorted({n for n in names if pat.search(n)})
        if self.args.links:
            hits = self.args.links.split(",")
        return hits

    def run(self):
        # 1. wait for joint states + TF to populate
        deadline = self.get_clock().now().nanoseconds + int(self.args.settle_sec * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 2. arm joints from /joint_states
        joints = None
        if self._latest_js is not None:
            name_to_pos = dict(zip(self._latest_js.name, self._latest_js.position))
            if all(j in name_to_pos for j in ARM_JOINT_NAMES):
                joints = [round(float(name_to_pos[j]), 4) for j in ARM_JOINT_NAMES]
        if joints is None:
            self.get_logger().warn(
                f"Could not read all arm joints from {self.args.joint_topic}; "
                f"joints will be omitted.")

        # 3. project each arm link origin into the base frame
        base = self.args.base_frame
        link_frames = self._discover_link_frames()
        pts_arm = []
        got = []
        for frame in link_frames:
            try:
                tf = self._tf_buf.lookup_transform(
                    base, frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                pts_arm.append((x, y))
                got.append(frame)
            except Exception:
                continue

        if not pts_arm:
            self.get_logger().error(
                f"No arm-link transforms resolved in frame '{base}'. "
                f"Check --base-frame and that TF is publishing. Tried: {link_frames}")
            return

        # 4. build the point cloud: base outline (raw) + densified, buffered arm chain
        pts = list(octagon(self.args.base_radius))
        chain = np.vstack([[0.0, 0.0]] + pts_arm)   # start at base origin
        order = np.argsort(np.hypot(chain[:, 0], chain[:, 1]))  # sort outward
        chain = chain[order]
        dense = np.vstack([densify(chain[i], chain[i+1])
                           for i in range(len(chain)-1)]) if len(chain) > 1 else chain
        for p in dense:
            pts.extend(circle_pts(p, self.args.link_buffer, 6))

        hull = drop_collinear(convex_hull(np.array(pts)))

        circ = float(np.hypot(hull[:, 0], hull[:, 1]).max())
        insc = inscribed_radius(hull)
        poly_str = "[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in hull) + "]"

        # 5. emit a ready-to-paste ARM_CONFIGS entry
        joints_str = ("[" + ", ".join(str(v) for v in joints) + "]") if joints else "None  # capture failed"
        mode = "play_motion" if self.args.label == "tucked" else "joint_trajectory"
        extra = '\n        "motion_name": "home",' if self.args.label == "tucked" else ""
        print("\n" + "="*70)
        print(f"Measured pose '{self.args.label}'  (base frame: {base})")
        print(f"  links used ({len(got)}): {', '.join(got)}")
        print(f"  inscribed_radius   = {insc:.3f} m")
        print(f"  circumscribed_radius = {circ:.3f} m")
        print(f"  vertices           = {len(hull)}")
        print("="*70)
        print(f'''    "{self.args.label}": {{
        "footprint": "{poly_str}",
        "mode": "{mode}",{extra}
        "joints": {joints_str},
        }},''')
        print("="*70)
        print("Paste into ARM_CONFIGS in scripts/param_space.py. The costmap presets\n"
              "are generated from ARM_CONFIGS, so this updates costmap + checker together.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="pose name, e.g. tucked / lateral")
    p.add_argument("--base-frame", default="base_footprint",
                   help="frame the footprint is expressed in (ground-projected)")
    p.add_argument("--base-radius", type=float, default=0.275,
                   help="mobile-base radius (m) for the base outline octagon")
    p.add_argument("--link-buffer", type=float, default=0.06,
                   help="radial buffer (m) added around each arm link centerline "
                        "to approximate link thickness")
    p.add_argument("--link-filter", default=r"(arm_\d|wrist|gripper|hand|forearm)",
                   help="regex; TF frames matching it are treated as arm links")
    p.add_argument("--links", default="",
                   help="explicit comma-separated frame list (overrides --link-filter)")
    p.add_argument("--joint-topic", default="/joint_states")
    p.add_argument("--settle-sec", type=float, default=1.5,
                   help="seconds to let TF + joint_states populate before measuring")
    args = p.parse_args()

    rclpy.init()
    node = FootprintMeasurer(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()