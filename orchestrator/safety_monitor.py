"""
safety_monitor.py — Reactive collision avoidance using laser scan.

Continuously monitors the forward laser scan sector. If an obstacle appears
within the danger distance, immediately publishes a zero-velocity /cmd_vel
message and flags the event. Resets when the path is clear.
"""

from typing import Optional
import math
import time
import threading
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class SafetyMonitor:
    """Reactive collision guard that preempts Nav2 via /cmd_vel zeroing.

    Does NOT cancel Nav2 goals — it force-publishes zero velocity so the
    robot stops *now*. Nav2 will timeout and re-plan. This is intentional:
    cancelling the Nav2 goal is slower and causes replanning noise.
    """

    def __init__(self, node,
                 forward_angle_deg: float = 40.0,
                 danger_distance_m: float = 0.35,
                 hysteresis_m: float = 0.10,
                 debounce_sec: float = 0.5):
        """
        Args:
            node: ROS 2 node (for creating publisher and logger).
            forward_angle_deg: Half-angle of the forward sector to scan.
            danger_distance_m: Distance at which emergency stop triggers.
            hysteresis_m: Distance above danger at which the guard resets.
            debounce_sec: Minimum time between repeated emergency stops.
        """
        self._log = node.get_logger().warn
        self._pub = node.create_publisher(Twist, "/cmd_vel", 1)
        self._half_angle = math.radians(forward_angle_deg)
        self._danger = danger_distance_m
        self._safe = danger_distance_m + hysteresis_m
        self._debounce = debounce_sec

        self._in_emergency = False
        self._last_stop_ts = 0.0
        self._lock = threading.Lock()

        # Stats exposed to diagnostics
        self.emergency_count = 0
        self.closest_object_m = float('inf')
        self.latest_scan: Optional[LaserScan] = None

    def update_scan(self, scan: LaserScan):
        """Call from your scan callback to feed the latest scan."""
        self.latest_scan = scan

    def check(self) -> bool:
        """Check the forward sector and stop if needed. Call periodically.

        Returns:
            True if an emergency stop was triggered this check.
        """
        scan = self.latest_scan
        if scan is None:
            return False

        with self._lock:
            # Find the closest object in the forward sector
            closest = float('inf')
            for i in range(len(scan.ranges)):
                angle = scan.angle_min + i * scan.angle_increment
                if abs(angle) > self._half_angle:
                    continue
                d = scan.ranges[i]
                if d <= scan.range_min or d >= scan.range_max:
                    continue
                if d < closest:
                    closest = d

            self.closest_object_m = closest

            if closest == float('inf'):
                # No valid readings in forward sector → no data
                return False

            if closest <= self._danger and not self._in_emergency:
                # Entering danger zone — emergency stop
                self._publish_zero()
                now = time.monotonic()
                if now - self._last_stop_ts > self._debounce:
                    self._log(
                        f"[SAFETY] EMERGENCY STOP — obstacle at {closest:.2f}m "
                        f"in forward sector ({math.degrees(self._half_angle):.0f}°)")
                    self._last_stop_ts = now
                self._in_emergency = True
                self.emergency_count += 1
                return True

            if closest >= self._safe and self._in_emergency:
                # Path clear — release emergency
                self._log(
                    f"[SAFETY] Path clear ({closest:.2f}m). Emergency released.")
                self._in_emergency = False
                return False

            # We're still in the hysteresis band — keep publishing zero
            if self._in_emergency:
                self._publish_zero()
                return True

            return False

    def _publish_zero(self):
        """Force-publish zero velocity to /cmd_vel."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self._pub.publish(msg)

    @property
    def is_in_emergency(self) -> bool:
        return self._in_emergency
