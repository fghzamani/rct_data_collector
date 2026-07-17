# /tmp/scan_probe.py
import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class P(Node):
    def __init__(self):
        super().__init__('scan_probe')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan_raw', self.cb, qos)
        self.n = 0

    def cb(self, m):
        self.n += 1
        if self.n > 3:
            rclpy.shutdown(); return
        near = [(math.degrees(m.angle_min + i * m.angle_increment), r)
                for i, r in enumerate(m.ranges) if m.range_min < r < 0.15]
        print(f'--- scan {self.n}: range_min={m.range_min:.3f} '
              f'fov=[{math.degrees(m.angle_min):.1f}, {math.degrees(m.angle_max):.1f}] '
              f'beams={len(m.ranges)}')
        print(f'    near-field returns (<0.15 m): {len(near)}')
        for a, r in near[:20]:
            print(f'      angle={a:7.2f} deg  range={r:.4f} m')

rclpy.init(); rclpy.spin(P())