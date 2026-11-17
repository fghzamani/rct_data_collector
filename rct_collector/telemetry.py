"""Per-probe robot telemetry, sampled on the SIM clock and saved as one .npz.

Why a separate node
-------------------
TrialRunnerNode records what the collector needs to score a probe. This node
records everything an analyst may later need to EXPLAIN one, and it can never
be added after the fact: whole-body joint states and efforts (an arm striking a
wall shows up as an effort spike, not in the planar footprint), arm link
positions for 3-D clearance, what the controller commanded versus what the base
did, the footprint each costmap actually PUBLISHED (not what we asked for),
nav feedback, every global plan, and the moving obstacles. It runs on its own
executor thread so a burst of /tf or /joint_states traffic cannot delay the
recorder's collision watch.

File format (``telemetry/probe_<id>_<run_id>.npz``)
-------------------------------------------------
  ts__<column>   float32 time series, one sample per tick (default 20 Hz sim)
  joints_q / joints_dq / joints_eff   (n, J)   names in meta["joint_names"]
  links_xyz      (n, 3L) arm link origins in ``base_frame``; names in meta
  risk           (n, R)  latest /risk_state vector at each tick
  meta_json      probe identity, stream message counts, column list
  events_json    timeline: phase marks, plans, BT failures, contacts,
                 footprint / speed-limit changes, nav recoveries

``load_probe_telemetry(path)`` returns (DataFrame-ready dict, meta, events).
Missing values are NaN; a stream that never delivered during the probe is
listed in meta["missing_streams"] and in the CSV column ``tel_missing``.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Callable, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64MultiArray, String
try:
    from visualization_msgs.msg import MarkerArray
except ImportError:              # pragma: no cover
    MarkerArray = None

try:
    from nav2_msgs.msg import SpeedLimit
except ImportError:              # pragma: no cover
    SpeedLimit = None
try:
    from nav2_msgs.action import NavigateToPose
    _NAV_FB = NavigateToPose.Impl.FeedbackMessage
except Exception:                # pragma: no cover
    _NAV_FB = None
try:
    from nav2_msgs.msg import BehaviorTreeLog
except ImportError:              # pragma: no cover
    BehaviorTreeLog = None

LASER_TO_BASE_LINK = (0.202, 0.0, 0.0)
N_SECTORS = 8                       # 45 deg each, sector 0 centred on +x (front)

# Streams whose absence makes a probe unexplainable. tel_missing lists any of
# these that delivered nothing while the probe was recording.
ESSENTIAL_STREAMS = ("model_states", "joint_states", "scan", "odom",
                     "fp_local", "fp_global")


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _poly_geom(msg: PolygonStamped) -> tuple:
    """(area m^2, diameter m) of a published footprint. Both are invariant to
    the frame: costmaps publish the footprint already placed at the robot pose
    in odom/map, so any radius 'from the origin' measures the map, not the
    footprint (the first version did exactly that)."""
    pts = [(p.x, p.y) for p in msg.polygon.points]
    if len(pts) < 3:
        return float("nan"), float("nan")
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    d = max(math.hypot(x1 - x2, y1 - y2) for x1, y1 in pts for x2, y2 in pts)
    return abs(a) / 2.0, d


_ARM_LINK = re.compile(r"(arm_|gripper|wrist|hand|tool)")


def contact_other(info: str) -> str:
    """The non-robot side of a contact: 'PAL_office', 'obs_dynamic_1', ..."""
    sides = [x.strip() for x in str(info).split("<->")]
    robot = next((x for x in sides if "_link" in x or "collision" in x), None)
    other = next((x for x in sides if x is not robot), "")
    return other.split("::")[0]


def contact_link(info: str) -> str:
    """Robot link named in a /gazebo/collision_info string, e.g.
    'wrist_ft_link_fixed_joint_lump__gripper_link_collision_2 <-> PAL_office'
    -> 'gripper_link'."""
    sides = [x.strip() for x in str(info).split("<->")]
    side = next((x for x in sides if "_link" in x or "collision" in x), sides[0])
    side = side.split("::")[-1]
    if "fixed_joint_lump__" in side:
        side = side.split("fixed_joint_lump__")[-1]
    return re.sub(r"_collision(_\d+)?$", "", side)


def _mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class ProbeTelemetry(Node):
    """Latch the newest message per stream; a sim-time timer snapshots them."""

    def __init__(self, *, rate_hz: float = 20.0, scan_topic: str = "/scan_raw",
                 odom_topic: str = "/mobile_base_controller/odom",
                 robot_model: str = "tiago",
                 obstacle_models: tuple = ("obs_dynamic_1", "obs_dynamic_2"),
                 cmd_vel_candidates: tuple = ("/cmd_vel_nav", "/cmd_vel",
                                              "/mobile_base_controller/cmd_vel_unstamped",
                                              "/mobile_base_controller/cmd_vel"),
                 joint_prefixes: tuple = ("arm_", "torso_", "gripper_", "head_"),
                 arm_links: tuple = ("arm_3_link", "arm_4_link", "arm_5_link",
                                     "arm_7_link", "arm_tool_link", "gripper_link",
                                     "payload_box"),
                 base_frame: str = "base_footprint",
                 record_arm_links: bool = True,
                 self_filter_radius_m: float = 0.30,
                 extras_fn: Optional[Callable[[], dict]] = None):
        from rclpy.parameter import Parameter as _P
        super().__init__("rct_probe_telemetry",
                         parameter_overrides=[_P("use_sim_time", _P.Type.BOOL, True)])
        self.rate_hz = float(rate_hz)
        self.robot_model = robot_model
        self.obstacle_models = tuple(obstacle_models)
        self.cmd_vel_candidates = tuple(cmd_vel_candidates)
        self.joint_prefixes = tuple(joint_prefixes)
        self.arm_links = tuple(arm_links) if record_arm_links else ()
        self.base_frame = base_frame
        self.self_filter_r = float(self_filter_radius_m)
        self.extras_fn = extras_fn

        self._lock = threading.Lock()
        self._recording = False
        self._latest: dict = {}
        self._counts: dict = {}          # stream -> messages seen while recording
        self._cols: dict = {}
        self._joints_q: list = []
        self._joints_dq: list = []
        self._joints_eff: list = []
        self._links: list = []
        self._quats: list = []
        self._risk: list = []
        self._events: list = []
        self._joint_names: Optional[list] = None
        self._risk_len: Optional[int] = None
        self._meta: dict = {}
        self._cmd_topics: list = []      # [(topic, stamped)]
        self._scan_cache = None
        self._contacts = 0
        self._last_fp = {"fp_local": None, "fp_global": None}
        self._last_sl = None
        self._last_recov = None
        self._tf_fail = 0
        self._recording = False

        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_models, sensor)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl, 5)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, sensor)
        self.create_subscription(JointState, "/joint_states", self._on_joints, sensor)
        self.create_subscription(PolygonStamped, "/local_costmap/published_footprint",
                                 lambda m: self._on_fp("fp_local", m), 5)
        self.create_subscription(PolygonStamped, "/global_costmap/published_footprint",
                                 lambda m: self._on_fp("fp_global", m), 5)
        self.create_subscription(Path, "/plan", self._on_plan, 5)
        # Local trajectories (2026-11-13). MPPI publishes them only with
        # FollowPath.visualize: true (nav_campaign2.yaml, trajectory_step = batch
        # so just ONE candidate is drawn): /trajectories carries the optimal
        # control sequence rolled out from the robot, point k at stamp + k*model_dt;
        # /transformed_global_plan is the window of the global plan it tracks.
        self.local_traj_hz = float(getattr(self, "local_traj_hz", 10.0))
        self.model_dt = 0.05
        self._last_local = -1e9
        self._last_tgp = -1e9
        if MarkerArray is not None:
            self.create_subscription(MarkerArray, "/trajectories", self._on_mppi_markers, 5)
        self.create_subscription(Path, "/transformed_global_plan", self._on_tgp, 5)
        self.create_subscription(String, "/gazebo/collision_info", self._on_contact, 20)
        self.create_subscription(Float64MultiArray, "/risk_state", self._on_risk, 5)
        if SpeedLimit is not None:
            self.create_subscription(SpeedLimit, "/speed_limit", self._on_speed_limit, 5)
        if _NAV_FB is not None:
            self.create_subscription(_NAV_FB, "/navigate_to_pose/_action/feedback",
                                     self._on_nav_fb, 5)
        if BehaviorTreeLog is not None:
            self.create_subscription(BehaviorTreeLog, "/behavior_tree_log", self._on_bt, 20)

        self._tf_buffer = None
        if self.arm_links:
            try:
                from tf2_ros import Buffer, TransformListener
                self._tf_buffer = Buffer()
                self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
            except Exception as exc:   # noqa: BLE001
                self.get_logger().warn(f"TF unavailable, arm links not recorded: {exc}")
                self.arm_links = ()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True,
                                        name="probe_telemetry")
        self._thread.start()

    # ── helpers ──────────────────────────────────────────────────────────
    def now_sec(self) -> float:
        try:
            return self.get_clock().now().nanoseconds / 1e9
        except Exception:   # noqa: BLE001
            return 0.0

    def _seen(self, stream: str) -> None:
        if self._recording:
            self._counts[stream] = self._counts.get(stream, 0) + 1

    def _event(self, kind: str, **kw) -> None:
        ev = {"t_sim": round(self.now_sec(), 4), "t_wall": round(time.time(), 4),
              "kind": kind}
        ev.update(kw)
        with self._lock:
            if self._recording:
                self._events.append(ev)

    def _resolve_cmd_topics(self) -> None:
        if self._cmd_topics:
            return
        types = dict(self.get_topic_names_and_types())
        for topic in self.cmd_vel_candidates:
            ts = types.get(topic, [])
            if "geometry_msgs/msg/TwistStamped" in ts:
                self.create_subscription(TwistStamped, topic,
                                         lambda m, t=topic: self._on_cmd(t, m.twist), 5)
                self._cmd_topics.append((topic, True))
            elif "geometry_msgs/msg/Twist" in ts:
                self.create_subscription(Twist, topic,
                                         lambda m, t=topic: self._on_cmd(t, m), 5)
                self._cmd_topics.append((topic, False))

    # ── callbacks (executor thread) ──────────────────────────────────────
    def _on_models(self, msg: ModelStates):
        names = list(msg.name)
        if self.robot_model in names:
            i = names.index(self.robot_model)
            p, q, tw = msg.pose[i].position, msg.pose[i].orientation, msg.twist[i]
            yaw = _yaw(q)
            c, s = math.cos(yaw), math.sin(yaw)
            vxw, vyw = tw.linear.x, tw.linear.y
            self._latest["gt"] = (p.x, p.y, yaw, c * vxw + s * vyw, -s * vxw + c * vyw,
                                  tw.angular.z)
        for k, name in enumerate(self.obstacle_models):
            if name in names:
                i = names.index(name)
                p, q = msg.pose[i].position, msg.pose[i].orientation
                self._latest[f"obs{k + 1}"] = (p.x, p.y, _yaw(q))
        self._seen("model_states")

    def _on_amcl(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        cov = msg.pose.covariance
        self._latest["amcl"] = (p.x, p.y, _yaw(q), math.sqrt(max(cov[0] + cov[7], 0.0)))
        self._seen("amcl")

    def _on_odom(self, msg: Odometry):
        tw = msg.twist.twist
        self._latest["odom"] = (tw.linear.x, tw.linear.y, tw.angular.z)
        self._seen("odom")

    def _on_cmd(self, topic: str, tw):
        self._latest[f"cmd:{topic}"] = (tw.linear.x, tw.linear.y, tw.angular.z, self.now_sec())
        self._seen(f"cmd:{topic}")

    def _on_scan(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float64)
        n = r.size
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        if self._scan_cache is None or self._scan_cache[0] != (msg.angle_min, msg.angle_increment, n):
            ang = msg.angle_min + np.arange(n) * msg.angle_increment + LASER_TO_BASE_LINK[2]
            self._scan_cache = ((msg.angle_min, msg.angle_increment, n), np.cos(ang), np.sin(ang))
        _, ux, uy = self._scan_cache
        px = LASER_TO_BASE_LINK[0] + r * ux
        py = LASER_TO_BASE_LINK[1] + r * uy
        d = np.hypot(px, py)
        ok &= d >= self.self_filter_r
        sectors = np.full(N_SECTORS, np.nan)
        if ok.any():
            a = np.arctan2(py[ok], px[ok])
            k = (np.floor((a + math.pi / N_SECTORS) / (2 * math.pi / N_SECTORS))
                 .astype(int) % N_SECTORS)
            dd = d[ok]
            for s in range(N_SECTORS):
                m = k == s
                if m.any():
                    sectors[s] = dd[m].min()
        self._latest["scan"] = (float(np.nanmin(sectors)) if ok.any() else float("nan"),
                                sectors)
        self._seen("scan")

    def _on_joints(self, msg: JointState):
        names = list(msg.name)
        pos = dict(zip(names, msg.position))
        vel = dict(zip(names, msg.velocity)) if msg.velocity else {}
        eff = dict(zip(names, msg.effort)) if msg.effort else {}
        store = self._latest.setdefault("joints", {})
        for n in names:
            if n.startswith(self.joint_prefixes):
                store[n] = (pos.get(n, float("nan")), vel.get(n, float("nan")),
                            eff.get(n, float("nan")))
        self._seen("joint_states")

    def _on_fp(self, which: str, msg: PolygonStamped):
        area, diam = _poly_geom(msg)
        self._latest[which] = (area, diam, self.now_sec())
        self._seen(which)
        prev = self._last_fp[which]
        if prev is None or abs(prev - area) > 0.02:
            self._last_fp[which] = area
            self._event("footprint_published", costmap=which, area_m2=round(area, 4),
                        diameter_m=round(diam, 3), n_points=len(msg.polygon.points))

    def _on_plan(self, msg: Path):
        self._seen("plan")
        if not self._recording:
            return
        xy = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses],
                      dtype=np.float64)
        length = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        keep, acc = [0] if len(xy) else [], 0.0
        for i in range(1, len(xy)):
            acc += math.hypot(*(xy[i] - xy[i - 1]))
            if acc >= 0.20 or i == len(xy) - 1:
                keep.append(i)
                acc = 0.0
        # stamp = when the planner produced it (header), t_sim (in _event) = when
        # it arrived; s_m = arc length of each kept pose along the plan.
        s_cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) \
            if len(xy) > 1 else np.zeros(len(xy))
        self._event("plan", length_m=round(length, 3), n_poses=len(xy),
                    stamp=round(_stamp(msg.header), 4),
                    frame=msg.header.frame_id,
                    xy=np.round(xy[keep], 3).tolist(),
                    s_m=np.round(s_cum[keep], 3).tolist())

    def _on_mppi_markers(self, msg):
        """MPPI optimal trajectory -> event 'local_traj': xy[k] is where the
        controller plans the robot at time stamp + k*dt (k = 0..time_steps-1)."""
        self._seen("local_traj")
        if not self._recording:
            return
        pts = [m for m in msg.markers if m.ns == "Optimal Trajectory"]
        if not pts:
            return
        now = self.now_sec()
        if now - self._last_local < 1.0 / self.local_traj_hz:
            return
        self._last_local = now
        pts.sort(key=lambda m: m.id)
        xy = np.array([(m.pose.position.x, m.pose.position.y) for m in pts])
        self._event("local_traj", stamp=round(_stamp(pts[0].header), 4), dt=self.model_dt,
                    frame=pts[0].header.frame_id, xy=np.round(xy, 3).tolist())

    def _on_tgp(self, msg: Path):
        self._seen("transformed_global_plan")
        if not self._recording or not msg.poses:
            return
        now = self.now_sec()
        if now - self._last_tgp < 0.5:          # 2 Hz is enough for the tracked window
            return
        self._last_tgp = now
        xy = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses])
        self._event("local_plan_window", stamp=round(_stamp(msg.header), 4),
                    frame=msg.header.frame_id, n_poses=len(xy),
                    xy=np.round(xy[::2], 3).tolist())

    def _on_contact(self, msg: String):
        self._contacts += 1
        self._seen("contact")
        self._event("contact", info=msg.data, link=contact_link(msg.data),
                    other=contact_other(msg.data))

    def _on_risk(self, msg: Float64MultiArray):
        self._latest["risk"] = tuple(msg.data)
        self._seen("risk")

    def _on_speed_limit(self, msg):
        v = (bool(msg.percentage), float(msg.speed_limit))
        self._latest["speed_limit"] = v
        self._seen("speed_limit")
        if v != self._last_sl:
            self._last_sl = v
            self._event("speed_limit", percentage=v[0], value=v[1])

    def _on_nav_fb(self, msg):
        fb = msg.feedback
        t = fb.navigation_time
        self._latest["nav"] = (fb.distance_remaining, float(fb.number_of_recoveries),
                               t.sec + t.nanosec * 1e-9)
        self._seen("nav_feedback")
        if fb.number_of_recoveries != self._last_recov:
            if self._last_recov is not None:
                self._event("nav_recovery", n=int(fb.number_of_recoveries))
            self._last_recov = fb.number_of_recoveries

    def _on_bt(self, msg):
        for ev in msg.event_log:
            if ev.current_status == "FAILURE":
                self._event("bt_failure", node=ev.node_name, previous=ev.previous_status)

    # ── sampling (executor thread) ───────────────────────────────────────
    def _tick(self):
        if not self._recording:
            return
        L = self._latest
        nan = float("nan")
        row = {"t_sim": self.now_sec(), "t_wall": time.time()}
        gt = L.get("gt")
        row.update(zip(("gt_x", "gt_y", "gt_yaw", "gt_vx", "gt_vy", "gt_wz"),
                       gt if gt else (nan,) * 6))
        am = L.get("amcl")
        row.update(zip(("amcl_x", "amcl_y", "amcl_yaw", "amcl_std_xy"),
                       am if am else (nan,) * 4))
        row["loc_err"] = (math.hypot(gt[0] - am[0], gt[1] - am[1])
                          if gt and am else nan)
        od = L.get("odom")
        row.update(zip(("odom_vx", "odom_vy", "odom_wz"), od if od else (nan,) * 3))
        now = row["t_sim"]
        for topic, _ in self._cmd_topics:
            key = topic.strip("/").replace("/", "__")
            c = L.get(f"cmd:{topic}")
            row[f"cmd__{key}__vx"] = c[0] if c else nan
            row[f"cmd__{key}__vy"] = c[1] if c else nan
            row[f"cmd__{key}__wz"] = c[2] if c else nan
            row[f"cmd__{key}__age"] = (now - c[3]) if c else nan
        sc = L.get("scan")
        row["scan_min"] = sc[0] if sc else nan
        for s in range(N_SECTORS):
            row[f"scan_s{s}"] = float(sc[1][s]) if sc else nan
        for which in ("fp_local", "fp_global"):
            fp = L.get(which)
            row[f"{which}_area"] = fp[0] if fp else nan
            row[f"{which}_diam"] = fp[1] if fp else nan
            row[f"{which}_age"] = (now - fp[2]) if fp else nan
        sl = L.get("speed_limit")
        row["speed_limit_value"] = sl[1] if sl else nan
        nav = L.get("nav")
        row.update(zip(("nav_dist_remaining", "nav_recoveries", "nav_time"),
                       nav if nav else (nan,) * 3))
        for k in range(len(self.obstacle_models)):
            o = L.get(f"obs{k + 1}")
            row.update(zip((f"obs{k + 1}_x", f"obs{k + 1}_y", f"obs{k + 1}_yaw"),
                           o if o else (nan,) * 3))
        row["contacts_cum"] = float(self._contacts)
        if self.extras_fn is not None:
            try:
                for k, v in (self.extras_fn() or {}).items():
                    row[k] = nan if v is None else float(v)
            except Exception:   # noqa: BLE001
                pass

        # Joints: fix the column order on the first sample of the probe.
        jt = L.get("joints", {})
        if self._joint_names is None and jt:
            self._joint_names = sorted(jt)
        jn = self._joint_names or []
        q = [jt.get(n, (nan, nan, nan)) for n in jn]

        links, quats = [], []
        if self._tf_buffer is not None:
            for link in self.arm_links:
                try:
                    tf = self._tf_buffer.lookup_transform(self.base_frame, link, Time())
                    t, r = tf.transform.translation, tf.transform.rotation
                    links += [t.x, t.y, t.z]
                    quats += [r.x, r.y, r.z, r.w]
                except Exception:   # noqa: BLE001
                    self._tf_fail += 1
                    links += [nan, nan, nan]
                    quats += [nan] * 4

        risk = L.get("risk")
        if self._risk_len is None and risk is not None:
            self._risk_len = len(risk)
        rl = self._risk_len or 0
        rv = list(risk[:rl]) + [nan] * (rl - len(risk)) if risk is not None else [nan] * rl

        with self._lock:
            if not self._recording:
                return
            n_prev = len(self._cols.get("t_sim", []))
            for k, v in row.items():
                col = self._cols.get(k)
                if col is None:
                    col = self._cols[k] = [nan] * n_prev
                col.append(v)
            for k, col in self._cols.items():
                if k not in row:
                    col.append(nan)
            self._joints_q.append([x[0] for x in q])
            self._joints_dq.append([x[1] for x in q])
            self._joints_eff.append([x[2] for x in q])
            self._links.append(links)
            self._quats.append(quats)
            self._risk.append(rv)

    # ── control (collector thread) ───────────────────────────────────────
    def start(self, probe_id: int, run_id: str, **meta) -> None:
        self._resolve_cmd_topics()
        with self._lock:
            self._cols, self._events = {}, []
            self._joints_q, self._joints_dq, self._joints_eff = [], [], []
            self._links, self._quats, self._risk = [], [], []
            self._joint_names, self._risk_len = None, None
            self._counts, self._contacts, self._tf_fail = {}, 0, 0
            self._last_fp = {"fp_local": None, "fp_global": None}
            self._last_sl, self._last_recov = None, None
            self._meta = {"probe_id": int(probe_id), "run_id": run_id,
                          "rate_hz": self.rate_hz, "base_frame": self.base_frame,
                          "arm_links": list(self.arm_links),
                          "obstacle_models": list(self.obstacle_models),
                          "cmd_topics": [t for t, _ in self._cmd_topics],
                          "scan_sectors": N_SECTORS,
                          "scan_sector_note": "sector k is centred on k*45 deg, base frame, CCW",
                          "mem_available_mb_start": round(_mem_available_mb(), 1),
                          "t_sim_start": self.now_sec(), "t_wall_start": time.time(),
                          **meta}
            self._recording = True
        self.mark("telemetry_start")

    def mark(self, kind: str, **kw) -> None:
        """Phase marker from the collector (apply, footprint push, arm goal...)."""
        self._event(kind, **{k: v for k, v in kw.items()})

    def stop(self) -> None:
        self.mark("telemetry_stop")
        with self._lock:
            self._recording = False

    def save(self, path: str, contact_t_sim: Optional[float] = None) -> dict:
        """Write the buffered probe and return its health summary (CSV columns)."""
        with self._lock:
            cols = {k: np.asarray(v, dtype=np.float64) for k, v in self._cols.items()}
            jq, jdq, jeff = self._joints_q, self._joints_dq, self._joints_eff
            links, quats = self._links, self._quats
            risk, events = self._risk, list(self._events)
            meta = dict(self._meta)
            counts = dict(self._counts)
            jn = list(self._joint_names or [])

        def _mat(rows, width):
            if not rows or width == 0:
                return np.zeros((len(rows), width), dtype=np.float32)
            return np.asarray([r if len(r) == width else [np.nan] * width for r in rows],
                              dtype=np.float32)

        missing = [s for s in ESSENTIAL_STREAMS if counts.get(s, 0) == 0]
        meta.update({"joint_names": jn, "stream_counts": counts, "missing_streams": missing,
                     "tf_lookup_failures": self._tf_fail,
                     "mem_available_mb_end": round(_mem_available_mb(), 1),
                     "columns": sorted(cols), "risk_len": self._risk_len or 0})
        summary = self._summarise(cols, jn, jeff, events, missing, contact_t_sim)
        meta["summary"] = summary

        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {f"ts__{k}": (v if k in ("t_sim", "t_wall") else v.astype(np.float32))
                  for k, v in cols.items()}
        arrays.update({
            "joints_q": _mat(jq, len(jn)), "joints_dq": _mat(jdq, len(jn)),
            "joints_eff": _mat(jeff, len(jn)),
            "links_xyz": _mat(links, 3 * len(self.arm_links)),
            "links_quat": _mat(quats, 4 * len(self.arm_links)),
            "risk": _mat(risk, self._risk_len or 0),
            "meta_json": np.array(json.dumps(meta, default=_json_default)),
            "events_json": np.array(json.dumps(events, default=_json_default)),
        })
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        from rct_collector.durable import replace_durable
        replace_durable(tmp, path)        # fsync before rename: no 0-byte files on power loss
        summary["tel_path"] = path
        return summary

    @staticmethod
    def _summarise(cols, jn, jeff, events, missing, contact_t_sim) -> dict:
        ts = cols.get("t_sim", np.zeros(0))
        tw = cols.get("t_wall", np.zeros(0))
        out = {"tel_n": int(ts.size), "tel_missing": ";".join(missing)}
        if ts.size < 2:
            out.update({"tel_rate_hz": 0.0, "tel_max_gap_s": float("nan"),
                        "tel_rtf_min_1s": float("nan"), "tel_rtf_mean": float("nan"),
                        "tel_rtf_min_1s_precontact": float("nan"),
                        "tel_max_gap_s_precontact": float("nan")})
            return out
        dur = ts[-1] - ts[0]
        out["tel_rate_hz"] = round((ts.size - 1) / dur, 2) if dur > 0 else 0.0
        out["tel_rtf_mean"] = round(dur / max(tw[-1] - tw[0], 1e-6), 3)
        first_contact = contact_t_sim
        if first_contact is None:
            ct = [e["t_sim"] for e in events if e.get("kind") == "contact"]
            first_contact = min(ct) if ct else None
        out["tel_first_contact_t"] = (round(first_contact - ts[0], 3)
                                      if first_contact is not None else "")

        def _health(mask):
            t, w = ts[mask], tw[mask]
            if t.size < 2:
                return float("nan"), float("nan")
            gap = float(np.max(np.diff(t)))
            # RTF over 1 s sim windows: worst sustained slowdown, not one jitter.
            rtfs, j = [], 0
            for i in range(t.size):
                while t[i] - t[j] > 1.0:
                    j += 1
                if t[i] - t[j] >= 0.9 and w[i] > w[j]:
                    rtfs.append((t[i] - t[j]) / (w[i] - w[j]))
            return gap, (min(rtfs) if rtfs else float("nan"))

        gap, rtf = _health(np.ones(ts.size, bool))
        out["tel_max_gap_s"] = round(gap, 3)
        out["tel_rtf_min_1s"] = round(rtf, 3)
        # The same numbers BEFORE the first contact. A collision makes the
        # physics engine slower; judging lag on the whole window would flag
        # collision probes as glitches and rerun them away (outcome selection).
        if first_contact is not None:
            gap, rtf = _health(ts < first_contact)
        out["tel_max_gap_s_precontact"] = round(gap, 3)
        out["tel_rtf_min_1s_precontact"] = round(rtf, 3)
        v = cols.get("scan_min")
        if v is not None and np.isfinite(v).any():
            out["tel_scan_min"] = round(float(np.nanmin(v)), 3)
        # Localisation error only from 1 s after the drive starts: the teleport
        # moves ground truth instantly while AMCL is re-seeded, which showed up
        # as 1-12 m "errors" that were never real.
        v = cols.get("loc_err")
        ds = [e["t_sim"] for e in events if e.get("kind") == "drive_started"]
        if v is not None:
            m = ts >= (ds[-1] + 1.0) if ds else np.ones(ts.size, bool)
            if np.isfinite(v[m]).any():
                out["tel_loc_err_max"] = round(float(np.nanmax(v[m])), 3)
            # Up to the first contact only: an impact moves the robot without
            # odometry seeing it, so AMCL loses track AFTER the outcome (seen up
            # to ~5 m). The pre-contact value is the usable covariate.
            ct = [e["t_sim"] for e in events if e.get("kind") == "contact"]
            if ct:
                m = m & (ts < ct[0])
            if np.isfinite(v[m]).any():
                out["tel_loc_err_max_precontact"] = round(float(np.nanmax(v[m])), 3)
        # Joint efforts are recorded but Gazebo's TIAGo reports 0 for every
        # joint, so arm strikes are identified from the contact plugin instead.
        contacts = [e for e in events if e.get("kind") == "contact"]
        if contacts:
            t1 = contacts[0]["t_sim"]
            first = sorted({e.get("link", "") for e in contacts if e["t_sim"] - t1 <= 0.1})
            out["tel_first_contact_links"] = ";".join(first)
            out["tel_first_contact_arm"] = int(any(_ARM_LINK.search(l) for l in first))
            out["tel_first_contact_with"] = ";".join(sorted(
                {e.get("other", "") for e in contacts if e["t_sim"] - t1 <= 0.1}))
            out["tel_contact_links"] = ";".join(sorted({e.get("link", "") for e in contacts}))
        else:
            out.update({"tel_first_contact_links": "", "tel_first_contact_arm": "",
                        "tel_contact_links": "", "tel_first_contact_with": ""})
        # Moving obstacles: did each move during the probe, where was it when
        # the treatment was applied (a pre-treatment context variable), and how
        # close did the robot come to it in the watch window (centre distance).
        ap = [e["t_sim"] for e in events if e.get("kind") == "apply"]
        we2 = [e["t_sim"] for e in events if e.get("kind") == "watch_end"]
        gx, gy = cols.get("gt_x"), cols.get("gt_y")
        k = 1
        while f"obs{k}_x" in cols:
            ox, oy = cols[f"obs{k}_x"], cols[f"obs{k}_y"]
            fin = np.isfinite(ox) & np.isfinite(oy)
            if fin.any():
                span = float(np.hypot(np.nanmax(ox) - np.nanmin(ox), np.nanmax(oy) - np.nanmin(oy)))
                out[f"tel_obs{k}_moved"] = int(span > 0.05)
                if ap and gx is not None:
                    i = min(int(np.searchsorted(ts, ap[0])), ts.size - 1)
                    out[f"tel_obs{k}_x_apply"] = round(float(ox[i]), 3)
                    out[f"tel_obs{k}_y_apply"] = round(float(oy[i]), 3)
                    out[f"tel_obs{k}_dist_apply"] = round(float(np.hypot(gx[i] - ox[i], gy[i] - oy[i])), 3)
                    j = int(np.searchsorted(ts, we2[0])) if we2 else ts.size
                    seg = slice(i, max(j, i + 1))
                    dd = np.hypot(gx[seg] - ox[seg], gy[seg] - oy[seg])
                    if np.isfinite(dd).any():
                        out[f"tel_obs{k}_min_dist_window"] = round(float(np.nanmin(dd)), 3)
            k += 1
        # What the costmaps actually PUBLISHED at the end of the watch window:
        # an independent check of footprint sync (the param read-back is the other).
        we = [e["t_sim"] for e in events if e.get("kind") == "watch_end"]
        la, ga = cols.get("fp_local_area"), cols.get("fp_global_area")
        if we and la is not None and ga is not None:
            i = int(np.searchsorted(ts, we[0])) - 1
            i = min(max(i, 0), ts.size - 1)
            out["tel_fp_local_area_end"] = round(float(la[i]), 4) if np.isfinite(la[i]) else ""
            out["tel_fp_global_area_end"] = round(float(ga[i]), 4) if np.isfinite(ga[i]) else ""
            if np.isfinite(la[i]) and np.isfinite(ga[i]):
                out["tel_fp_published_sync_end"] = int(abs(la[i] - ga[i]) < 0.02)
        out["tel_n_plans"] = sum(1 for e in events if e.get("kind") == "plan")
        out["tel_n_local_traj"] = sum(1 for e in events if e.get("kind") == "local_traj")
        out["tel_n_bt_failures"] = sum(1 for e in events if e.get("kind") == "bt_failure")
        out["tel_n_contacts"] = sum(1 for e in events if e.get("kind") == "contact")
        rec = cols.get("nav_recoveries")
        out["tel_nav_recoveries_max"] = (int(np.nanmax(rec)) if rec is not None
                                         and np.isfinite(rec).any() else "")
        return out

    def shutdown(self) -> None:
        try:
            self._executor.shutdown()
        except Exception:   # noqa: BLE001
            pass
        self._thread.join(timeout=2.0)
        try:
            self.destroy_node()
        except Exception:   # noqa: BLE001
            pass


def _json_default(o):
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_probe_telemetry(path: str):
    """(columns dict, meta, events). ``pandas.DataFrame(columns)`` gives the
    time series; joint/link/risk matrices are added as named columns."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    events = json.loads(str(z["events_json"]))
    cols = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    for i, n in enumerate(meta.get("joint_names", [])):
        cols[f"q__{n}"] = z["joints_q"][:, i]
        cols[f"dq__{n}"] = z["joints_dq"][:, i]
        cols[f"eff__{n}"] = z["joints_eff"][:, i]
    for i, n in enumerate(meta.get("arm_links", [])):
        for j, ax in enumerate("xyz"):
            if z["links_xyz"].shape[1] > 3 * i + j:
                cols[f"link__{n}__{ax}"] = z["links_xyz"][:, 3 * i + j]
    if "links_quat" in z.files:
        for i, n in enumerate(meta.get("arm_links", [])):
            for j, ax in enumerate(("qx", "qy", "qz", "qw")):
                if z["links_quat"].shape[1] > 4 * i + j:
                    cols[f"link__{n}__{ax}"] = z["links_quat"][:, 4 * i + j]
    for i in range(z["risk"].shape[1] if z["risk"].ndim == 2 else 0):
        cols[f"risk_{i}"] = z["risk"][:, i]
    return cols, meta, events
