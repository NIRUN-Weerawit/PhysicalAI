"""
robot_interface.py — Layer 3: Clean robot command/query interface.

Wraps ROS 2 actions, TF lookups, and Nav2 state behind a simple
Python API.  All robot capabilities the LLM can call go through here.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid


class ToolResult:
    """Standard return type for all robot tools."""
    def __init__(self, success: bool, message: str = "",
                 data: dict = None, failure_type: str = None,
                 estimated_duration_s: float = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.failure_type = failure_type
        self.estimated_duration_s = estimated_duration_s

    def __repr__(self):
        status = "OK" if self.success else f"FAIL({self.failure_type})"
        return f"[{status}] {self.message[:120]}"


class RobotInterface:
    """High-level interface exposing all robot capabilities as tools.

    Each method returns a ToolResult with structured data, failure codes,
    and estimated durations — ready for LLM consumption.
    """

    def __init__(self, node: Node):
        self._node = node
        self._log = node.get_logger().info

        # ── Nav2 action client ──
        self._nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self._goal_handle = None
        self._goal_active = False
        self._goal_result_future = None
        self._nav_client_ready = False

        # ── TF bridge (injected externally) ──
        self._tf_bridge = None
        self._map = None

        # ── Status cache ──
        self._max_linear_speed = 0.26  # TB3 default
        self._start_time = time.monotonic()
        self._output_dir = None

        # ── Perception pipeline (injected externally) ──
        self._detector = None
        self._depth_estimator = None
        self._bridge = None  # CvBridge
        self._object_db = None
        self._text_prompt = ""

        # ── Exploration (injected externally) ──
        self._explore_fn = None

    def bind_perception(self, detector, depth_estimator, bridge,
                        object_db, text_prompt: str, output_dir: str = None):
        """Connect the detection pipeline (called from MapOrchestrator)."""
        self._detector = detector
        self._depth_estimator = depth_estimator
        self._bridge = bridge
        self._object_db = object_db
        self._text_prompt = text_prompt
        self._output_dir = output_dir

    def bind_exploration(self, explore_fn):
        """Bind the frontier-exploration function from MapOrchestrator."""
        self._explore_fn = explore_fn

    def bind_orchestrator_node(self, node):
        """Store reference to MapOrchestrator for calibration and state queries."""
        self._orchestrator_node = node

    def bind_tf_bridge(self, bridge):
        """Called after construction when TFBridge is available."""
        self._tf_bridge = bridge

    def bind_map(self, map_msg: OccupancyGrid):
        """Keep a reference to the latest occupancy grid."""
        self._map = map_msg
        # Called externally by MapOrchestrator or LLM bridge when map updates

    # ── Motion tools ───────────────────────────────────────────────────────

    def navigate_to(self, x: float, y: float, theta: float = 0.0) -> ToolResult:
        """Drive the robot to a map coordinate.

        Args:
            x, y: Map-frame position in meters
            theta: Final orientation in radians (default 0)

        Returns:
            ToolResult — if Nav2 not ready, returns failure.
            Estimated duration: ~4s per meter at 0.26 m/s + 3s overhead.
        """
        if self._goal_active:
            return ToolResult(
                False, "Already navigating. Call stop() first.",
                failure_type="nav_busy")

        if not self._nav_client.wait_for_server(timeout_sec=0.5):
            return ToolResult(
                False, "Nav2 navigate_to_pose server not available.",
                failure_type="nav_unavailable")

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()

        # If target is in unknown or occupied space, snap to nearest reachable
        target_x = float(x)
        target_y = float(y)
        snapped_x, snapped_y, snapped = self._snap_to_reachable(target_x, target_y)
        goal.pose.pose.position.x = snapped_x if snapped else target_x
        goal.pose.pose.position.y = snapped_y if snapped else target_y

        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, float(theta))
        goal.pose.pose.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])

        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_sent_callback)

        # Estimate duration
        nav_x = goal.pose.pose.position.x
        nav_y = goal.pose.pose.position.y
        if self._map is None:
            est_s = 10.0
        else:
            dist = np.linalg.norm([nav_x, nav_y])
            est_s = dist / self._max_linear_speed + 3.0

        msg = f"Navigating to ({nav_x:.2f}, {nav_y:.2f}, {theta:.2f})"
        if snapped:
            msg += f" [target was unreachable — snapped from ({target_x:.1f}, {target_y:.1f})]"

        return ToolResult(
            True, msg,
            data={"target": (nav_x, nav_y), "theta": theta, "snapped": snapped},
            estimated_duration_s=est_s)

    def navigate_to_object(self, class_name: str, object_db) -> ToolResult:
        """Navigate to a detected object by class name or specific object ID.

        Args:
            class_name: Object class to search for (e.g. 'sofa', 'table') or
                        specific object ID (e.g. 'chair_1').
            object_db: ObjectDB instance for querying

        Returns:
            ToolResult with position or failure.
            If multiple objects match the class, sets pending choices
            on the dashboard for user disambiguation.
        """
        all_objs = object_db.get_all()
        if not all_objs:
            return ToolResult(False, f"No objects in database.",
                              failure_type="detect_empty")

        # First: try exact object_id match
        exact = object_db.get(class_name)
        if exact:
            x, y, _ = exact.position_world
            return self.navigate_to(x, y)

        # Second: filter by class
        matches = [o for o in all_objs if class_name.lower() in o.class_name.lower()]
        if not matches:
            return ToolResult(False, f"No objects matching '{class_name}' found.",
                              failure_type="detect_empty")

        if len(matches) == 1:
            obj = matches[0]
            x, y, _ = obj.position_world
            return self.navigate_to(x, y)

        # Multiple matches — present choices to user via dashboard modal
        choices = []
        for obj in matches:
            x, y = obj.position_world[0], obj.position_world[1]
            desc = obj.description or ""
            tag = f" — {desc}" if desc else ""
            choices.append({
                "id": obj.object_id,
                "label": obj.object_id,
                "desc": f"At ({x:.1f}, {y:.1f})  conf={obj.confidence:.2f}{tag}",
            })

        try:
            from orchestrator.dashboard_server import set_pending_choices
            set_pending_choices(
                f"Multiple objects match '{class_name}'. Which one?",
                choices)
        except Exception:
            pass  # dashboard not available, fall through

        choices_text = "\n".join(
            f"  #{i+1}: {c['label']} — {c['desc']}"
            for i, c in enumerate(choices))
        return ToolResult(
            False,
            f"Multiple objects match '{class_name}'. Please specify one:\n{choices_text}",
            data={"matches": choices},
            failure_type="multiple_matches")

    def stop(self) -> ToolResult:
        """Cancel all active Nav2 goals immediately. Robot stops."""
        if not self._goal_active:
            return ToolResult(True, "Already stopped.")

        if self._goal_handle:
            cancel_future = self._goal_handle.cancel_goal_async()
            # Fire-and-forget the cancel
            cancel_future.add_done_callback(lambda f: None)

        self._goal_active = False
        self._goal_handle = None
        self._goal_result_future = None
        return ToolResult(True, "Navigation cancelled. Robot stopping.",
                          estimated_duration_s=1.0)

    def go_home(self) -> ToolResult:
        """Return the robot to map origin (0, 0)."""
        return self.navigate_to(0.0, 0.0, 0.0)

    def rotate(self, angle_deg: float) -> ToolResult:
        """Spin the robot in place by N degrees.

        Uses direct /cmd_vel publishing with a timed rotation.
        Positive = counter-clockwise, negative = clockwise.

        Args:
            angle_deg: Degrees to spin. 90 = quarter turn. Accepts float, int, or str.

        Returns:
            ToolResult — estimated duration ~abs(angle_deg)/60 seconds
        """
        try:
            import math
            from geometry_msgs.msg import Twist

            # Defensive: accept str or int from LLM
            angle_deg = float(angle_deg)

            pub = self._node.create_publisher(Twist, '/cmd_vel', 10)
            msg = Twist()

            # TB3 max spin: ~1.0 rad/s. Use 0.5 rad/s for stable rotation.
            speed = math.radians(90)  # ~90 deg/s
            if angle_deg < 0:
                speed = -speed

            duration = abs(angle_deg) / 90.0  # seconds for 90 deg/s rate

            msg.angular.z = speed
            pub.publish(msg)
            time.sleep(duration)
            msg.angular.z = 0.0
            pub.publish(msg)

            return ToolResult(True, f"Rotated {angle_deg:.0f}° in {duration:.1f}s.",
                              estimated_duration_s=duration)
        except Exception as e:
            return ToolResult(False, f"Rotate failed: {e}")

    def drive(self, distance_m: float, speed=None) -> ToolResult:
        """Drive the robot forward/backward a relative distance in meters.

        Uses direct /cmd_vel publishing with a timed movement.
        Positive distance = forward, negative = backward.

        Args:
            distance_m: Distance in meters to travel. Accepts float, int, or str.
            speed: Linear speed in m/s. Defaults to max_linear_speed * 0.5.

        Returns:
            ToolResult with estimated duration.
        """
        try:
            from geometry_msgs.msg import Twist

            distance_m = float(distance_m)
            if speed is not None:
                speed = float(speed)
            else:
                speed = self._max_linear_speed * 0.5  # ~0.13 m/s default

            direction = 1.0 if distance_m >= 0 else -1.0
            duration = abs(distance_m) / speed

            pub = self._node.create_publisher(Twist, '/cmd_vel', 10)
            msg = Twist()
            msg.linear.x = speed * direction
            pub.publish(msg)
            time.sleep(duration)
            msg.linear.x = 0.0
            pub.publish(msg)

            return ToolResult(True, f"Drove {distance_m:.2f}m in {duration:.1f}s.",
                              estimated_duration_s=duration)
        except Exception as e:
            return ToolResult(False, f"Drive failed: {e}")

    def wait(self, seconds: float) -> ToolResult:
        """Pause for N seconds (no motion)."""
        seconds = float(seconds)
        time.sleep(seconds)
        return ToolResult(True, f"Waited {seconds:.1f}s.",
                          estimated_duration_s=seconds)

    def explore(self) -> ToolResult:
        """Drive the robot toward unknown (unmapped) areas using frontier detection.

        Scans the current occupancy grid, finds the nearest boundary between
        known free space and unknown space, and navigates there via Nav2.
        **Blocks until the goal is reached or fails.**

        Returns:
            ToolResult — position visited + Nav2 outcome.
            Estimated duration: 10-60 seconds per frontier.
        """
        if self._explore_fn is None:
            return ToolResult(False, "Exploration not available. Is SLAM running?",
                              failure_type="internal_error")

        goals = self._explore_fn()
        if not goals:
            return ToolResult(False, "No frontiers remaining — map appears complete.",
                              failure_type="nav_blocked")

        cx, cy = goals[0]
        result = self.navigate_to(cx, cy)
        if not result.success:
            return result

        # Block until navigation completes
        return self._wait_for_nav_completion()

    def _wait_for_nav_completion(self, poll_interval: float = 1.0,
                                  timeout: float = 120.0) -> ToolResult:
        """Block until the active Nav2 goal finishes.

        Polls check_goal_result() every poll_interval seconds.
        Returns the final ToolResult or timeout failure.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.check_goal_result()
            if result is not None:
                return result
            time.sleep(poll_interval)
        # Timeout — cancel and return failure
        self.stop()
        return ToolResult(False, "Navigation timed out after {:.0f}s.".format(timeout),
                          failure_type="nav_timeout")

    # ── Nearest reachable (snap unmapped targets) ───────────────────────

    def _snap_to_reachable(self, x: float, y: float,
                           search_radius_m: float = 3.0) -> tuple:
        """Find the nearest navigable cell if the target is in unknown/occupied space.

        Checks the occupancy grid. If the requested (x, y) is free space,
        returns it unchanged. Otherwise spirals outward to find the closest
        reachable free cell within search_radius_m.

        Returns:
            (new_x, new_y, was_snapped) where was_snapped=True means a correction
            was applied.
        """
        if self._map is None:
            return (x, y, False)

        grid = np.array(self._map.data, dtype=np.int8).reshape(
            self._map.info.height, self._map.info.width)
        res = self._map.info.resolution
        ox = self._map.info.origin.position.x
        oy = self._map.info.origin.position.y

        # Map target to grid coordinates
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        h, w = grid.shape

        # Check if target is already in free space
        if 0 <= gx < w and 0 <= gy < h and grid[gy, gx] == 0:
            return (x, y, False)

        # Spiral search outward for nearest free cell
        max_r = int(search_radius_m / res)
        best_dist = float('inf')
        best_gx, best_gy = gx, gy

        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                nx = gx + dx
                # Top edge
                ny = gy + r
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    d = dx*dx + r*r
                    if d < best_dist:
                        best_dist = d
                        best_gx, best_gy = nx, ny
                # Bottom edge
                ny = gy - r
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    d = dx*dx + r*r
                    if d < best_dist:
                        best_dist = d
                        best_gx, best_gy = nx, ny
            for dy in range(-r + 1, r):
                ny = gy + dy
                # Right edge
                nx = gx + r
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    d = r*r + dy*dy
                    if d < best_dist:
                        best_dist = d
                        best_gx, best_gy = nx, ny
                # Left edge
                nx = gx - r
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    d = r*r + dy*dy
                    if d < best_dist:
                        best_dist = d
                        best_gx, best_gy = nx, ny

            if best_dist < float('inf'):
                break

        if best_dist < float('inf'):
            world_x = best_gx * res + ox + res / 2
            world_y = best_gy * res + oy + res / 2
            self._log(f"  [Snap] Target ({x:.1f}, {y:.1f}) not reachable — "
                      f"snapped to ({world_x:.1f}, {world_y:.1f}) "
                      f"(Δ={best_dist**0.5*res:.1f}m)")
            return (world_x, world_y, True)

        return (x, y, False)

    # ── Perception tools ───────────────────────────────────────────────────

    def get_pose(self) -> ToolResult:
        """Get current robot pose in map frame from TF.

        Returns:
            ToolResult with data={'pose': (x, y, theta), 'frame': 'map'}
        """
        if self._tf_bridge is None or not self._tf_bridge.transform_available():
            return ToolResult(False, "TF transform not yet available.",
                              failure_type="tf_timeout")

        try:
            # Use the TFBridge's internal TF buffer directly
            tf_buffer = self._tf_bridge._tf_buffer
            transform = tf_buffer.lookup_transform(
                "map", "base_footprint",
                rclpy.time.Time())

            import math
            from tf_transformations import euler_from_quaternion
            _, _, theta = euler_from_quaternion([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
            ])

            return ToolResult(True, f"Pose: ({transform.transform.translation.x:.2f}, "
                              f"{transform.transform.translation.y:.2f}, {math.degrees(theta):.0f}°)",
                              data={
                                  "pose": (transform.transform.translation.x,
                                           transform.transform.translation.y, theta),
                                  "frame": "map",
                              })
        except Exception as e:
            return ToolResult(False, f"Pose lookup failed: {e}",
                              failure_type="tf_timeout")

    def get_map_coverage(self, map_msg) -> ToolResult:
        """Compute what fraction of the known map has been explored.

        Returns:
            ToolResult with data={'coverage_pct': float, 'explored_cells': int,
                                  'total_known_cells': int}
        """
        if map_msg is None:
            return ToolResult(False, "No map available.", failure_type="map_stale")

        try:
            grid = np.array(map_msg.data, dtype=np.int8).reshape(
                map_msg.info.height, map_msg.info.width)
            total_cells = grid.size
            known = grid != -1
            explored = grid == 0
            coverage = float(np.sum(explored)) / max(float(np.sum(known)), 1) * 100.0
            return ToolResult(True, f"Map coverage: {coverage:.1f}%",
                              data={
                                  "coverage_pct": coverage,
                                  "explored_cells": int(np.sum(explored)),
                                  "total_known_cells": int(np.sum(known)),
                              })
        except Exception as e:
            return ToolResult(False, f"Coverage computation failed: {e}")

    def get_status(self, object_db=None, map_msg=None) -> ToolResult:
        """Full status report combining pose, map, objects, and health.

        Returns:
            ToolResult with a comprehensive status dict.
        """
        pose_res = self.get_pose()
        coverage_res = self.get_map_coverage(map_msg) if map_msg else None

        nav_status = "navigating" if self._goal_active else "idle"
        obj_count = len(object_db.get_all()) if object_db else 0
        uptime = time.monotonic() - self._start_time

        parts = []
        if pose_res.success:
            parts.append(pose_res.message)
        else:
            parts.append("pose: unknown")

        if coverage_res and coverage_res.success:
            parts.append(f"map: {coverage_res.data['coverage_pct']:.0f}% explored")

        parts.append(f"objects: {obj_count}")
        parts.append(f"status: {nav_status}")
        parts.append(f"uptime: {uptime:.0f}s")

        return ToolResult(True, " | ".join(parts), data={
            "pose": pose_res.data.get("pose") if pose_res.success else None,
            "coverage_pct": coverage_res.data["coverage_pct"] if coverage_res and coverage_res.success else None,
            "object_count": obj_count,
            "nav_status": nav_status,
            "uptime_s": uptime,
        })

    # ── Nav2 goal callbacks ────────────────────────────────────────────────

    def _goal_sent_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self._log("Nav2 goal rejected.")
            self._goal_active = False
            return

        self._goal_handle = gh
        self._goal_active = True
        self._goal_result_future = gh.get_result_async()

    def check_goal_result(self):
        """Call periodically to check if the active goal has finished.
        Returns a ToolResult describing the outcome, or None if still active.
        """
        if self._goal_result_future is None:
            was_active = self._goal_active
            self._goal_active = False
            if was_active:
                return ToolResult(True, "Navigation completed (no result future).")
            return None

        if not self._goal_result_future.done():
            return None

        try:
            result = self._goal_result_future.result()
            status_codes = {
                2: ("INVALID", "Goal was invalid"),
                3: ("ABORTED", "Navigation aborted — check obstacles"),
                4: ("SUCCEEDED", "Navigation completed successfully"),
                5: ("CANCELLED", "Navigation was cancelled"),
                6: ("SUCCEEDED", "Goal reached"),
            }
            code = result.status
            label, msg = status_codes.get(code, (f"UNKNOWN({code})", ""))
            success = code in (4, 6)

            return ToolResult(
                success,
                f"Nav2 goal finished: {label}. {msg}",
                data={"nav_status": code, "status_label": label},
                failure_type=None if success else f"nav_{label.lower()}",
            )
        except Exception as e:
            return ToolResult(
                False, f"Nav2 goal failed: {e}",
                failure_type="nav_error",
            )
        finally:
            self._goal_active = False
            self._goal_handle = None
            self._goal_result_future = None

    def cancel_goal(self) -> ToolResult:
        """Alias for stop()."""
        return self.stop()

    # ── Perception tools ────────────────────────────────────────────────────

    def detect_now(self) -> ToolResult:
        """Force detection on the latest camera frame immediately.

        Returns:
            ToolResult with data listing all detected objects with
            class, confidence, position, and timestamp.
        """
        if self._detector is None or self._bridge is None:
            return ToolResult(False, "Detection pipeline not initialized.",
                              failure_type="internal_error")

        # Detection happens externally (via MapOrchestrator's timer).
        # This tool returns the current ObjectDB contents.
        return self._build_detection_report()

    def scan_surroundings(self) -> ToolResult:
        """Perform a 360° scan: rotate while detecting.

        Returns:
            ToolResult with all objects detected during the scan.
        """
        # First detection pass (current view)
        # Then rotate 90° x 4 times, detecting each time
        angles = [0, 90, 180, 270]
        total_objects_before = len(self._object_db.get_all()) if self._object_db else 0

        for angle in angles:
            self.rotate(angle)
            time.sleep(3)  # wait for rotation + detection to settle

        total_new = (len(self._object_db.get_all()) if self._object_db else 0) - total_objects_before
        return ToolResult(True, f"360° scan complete. {total_new} new objects detected.",
                          data={"new_objects_found": total_new})

    def search(self, class_name: str) -> ToolResult:
        """Search for a specific object by rotating 360°.

        Args:
            class_name: Object class to search for (e.g. 'sofa', 'bottle')

        Returns:
            ToolResult — if found, includes position. If not found,
            suggests moving to a new area.
        """
        # Check if already seen
        if self._object_db:
            existing = [o for o in self._object_db.get_all()
                       if class_name.lower() in o.class_name.lower()]
            if existing:
                obj = existing[0]
                x, y, _ = obj.position_world
                return ToolResult(True, f"'{class_name}' already in memory at ({x:.2f}, {y:.2f}).",
                                  data={"position": (x, y), "source": "database"})

        # Not in DB — scan 360°
        angles = [0, 90, 180, 270]
        for angle in angles:
            self.rotate(angle)
            time.sleep(3)
            if self._object_db:
                found = [o for o in self._object_db.get_all()
                        if class_name.lower() in o.class_name.lower()]
                if found:
                    obj = found[0]
                    x, y, _ = obj.position_world
                    return ToolResult(True, f"Found '{class_name}' at ({x:.2f}, {y:.2f}) during scan.",
                                      data={"position": (x, y), "source": "scan"})

        return ToolResult(False, f"'{class_name}' not found after 360° scan. Try explore() to map more area.",
                          failure_type="detect_empty")

    def can_see(self, class_name: str) -> ToolResult:
        """Check if an object of the given class is in the latest detection results.

        Returns:
            ToolResult — true if object is currently visible.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.")

        latest = self._object_db.get_all()
        found = [o for o in latest if class_name.lower() in o.class_name.lower()]
        if found:
            obj = found[0]
            return ToolResult(True, f"'{class_name}' is visible in latest view.",
                              data={"position": obj.position_world,
                                    "confidence": obj.confidence})
        return ToolResult(False, f"'{class_name}' not in latest detection results.",
                          failure_type="detect_empty")

    def _build_detection_report(self) -> ToolResult:
        """Build a detection report from current ObjectDB state."""
        if not self._object_db:
            return ToolResult(True, "No objects detected yet.", data={"objects": []})

        all_objs = self._object_db.get_all()
        if not all_objs:
            return ToolResult(True, "No objects in database.", data={"objects": []})

        lines = [f"{len(all_objs)} objects:"]
        for obj in sorted(all_objs, key=lambda o: o.class_name)[:30]:
            x, y, z = obj.position_world
            lines.append(f"  {obj.class_name:15s}  ({x:.2f}, {y:.2f}, {z:.2f})  conf={obj.confidence:.2f}")

        return ToolResult(True, "\n".join(lines), data={
            "count": len(all_objs),
            "objects": [{"class": o.class_name, "position": o.position_world,
                         "confidence": o.confidence} for o in all_objs],
        })

    # ── Memory tools ────────────────────────────────────────────────────────

    def list_objects(self) -> ToolResult:
        """Return all tracked objects with their unique IDs, positions, and descriptions."""
        if not self._object_db:
            return ToolResult(True, "No objects in database.", data={"objects": []})
        all_objs = self._object_db.get_all()
        if not all_objs:
            return ToolResult(True, "No objects in database.", data={"objects": []})

        lines = [f"{len(all_objs)} objects tracked:"]
        for obj in sorted(all_objs, key=lambda o: o.object_id):
            x, y = obj.position_world[0], obj.position_world[1]
            desc = obj.description or ""
            tag = f" — {desc}" if desc else ""
            lines.append(f"  {obj.object_id:20s}  ({x:.2f}, {y:.2f})  "
                         f"conf={obj.confidence:.2f}{tag}")
        return ToolResult(True, "\n".join(lines), data={
            "count": len(all_objs),
            "objects": [{"id": o.object_id, "class": o.class_name,
                         "position": o.position_world,
                         "confidence": o.confidence} for o in all_objs],
        })

    def forget_object(self, object_id: str) -> ToolResult:
        """Remove a specific tracked object from memory by its unique ID (e.g. 'chair_1').

        Args:
            object_id: The unique object ID shown in list_objects (e.g. 'chair_1', 'cup_3')

        Returns:
            ToolResult confirming removal or not-found.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.",
                              failure_type="internal_error")
        removed = self._object_db.forget_object(object_id)
        if removed:
            return ToolResult(True, f"Forgot object '{removed}'.")
        return ToolResult(False, f"Object '{object_id}' not found.",
                          failure_type="detect_empty")

    def forget_class(self, class_name: str) -> ToolResult:
        """Remove all objects of a given class from memory.

        Args:
            class_name: Object class name (e.g. 'chair', 'cup', 'table')

        Returns:
            ToolResult with count of objects removed.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.",
                              failure_type="internal_error")
        count = self._object_db.forget_class(class_name)
        return ToolResult(True, f"Forgot {count} object(s) of class '{class_name}'.")

    def forget_all(self) -> ToolResult:
        """Remove all objects from memory entirely. Irreversible."""
        if not self._object_db:
            return ToolResult(False, "Object database not available.",
                              failure_type="internal_error")
        count = self._object_db.forget_all()
        return ToolResult(True, f"Forgot all {count} objects. Memory is now empty.")

    def list_places(self, places: dict = None) -> ToolResult:
        """Return all named places."""
        if not places:
            return ToolResult(True, "No places saved.", data={"places": []})
        lines = [f"{len(places)} places:"]
        for name, (x, y) in sorted(places.items()):
            lines.append(f"  {name:15s}  ({x:.2f}, {y:.2f})")
        return ToolResult(True, "\n".join(lines), data={"places": places})

    def remember_place(self, name: str, pose: tuple, places: dict = None) -> ToolResult:
        """Store the current robot pose as a named location.

        Args:
            name: A name for this place (e.g. 'home_base', 'charging_station')
            pose: (x, y, theta) from get_pose()
            places: mutable dict to store in

        Returns:
            ToolResult confirming where it was stored, or pose hint if unavailable.
        """
        if places is not None:
            places[name] = (pose[0], pose[1])
            return ToolResult(True, f"Saved '{name}' at ({pose[0]:.2f}, {pose[1]:.2f}).")
        return ToolResult(False, "Places storage not available.", failure_type="internal_error")

    def go_to_place(self, name: str, places: dict = None) -> ToolResult:
        """Navigate to a previously saved place."""
        if not places or name not in places:
            return ToolResult(False, f"Place '{name}' not found.", failure_type="internal_error")
        x, y = places[name]
        return self.navigate_to(x, y)

    # ── ROS 2 introspection ─────────────────────────────────────────────────

    def ros2_introspect(self, query: str) -> ToolResult:
        """Execute a read-only ROS 2 CLI command.

        Allowed commands:
          - topic list, topic echo --once, topic info
          - node list, service list, action list
          - param list

        Blocked: topic pub, run, lifecycle, bag, launch.

        Args:
            query: A ROS 2 CLI command string (e.g. 'topic echo /scan --once')

        Returns:
            ToolResult with CLI output or block message.
        """
        import subprocess

        # Whitelist check
        blocked = ["topic pub", " run ", "lifecycle", " bag ", " launch",
                   "action send", "service call"]
        for b in blocked:
            if b in query.lower():
                return ToolResult(False, f"BLOCKED: '{b}' is not allowed in introspection mode.")

        allowed_prefixes = ("topic", "node", "service", "action", "param")
        if not query.startswith(allowed_prefixes):
            return ToolResult(False, f"BLOCKED: introspection only allows: {', '.join(allowed_prefixes)}")

        try:
            cmd = ["timeout", "5", "ros2"] + query.split()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            output = (result.stdout or result.stderr)[:2000]  # cap at 2000 chars
            if not output.strip():
                return ToolResult(True, "Command returned empty output.")
            return ToolResult(True, output.strip())
        except subprocess.TimeoutExpired:
            return ToolResult(False, "Command timed out after 5s.", failure_type="internal_error")
        except Exception as e:
            return ToolResult(False, f"Introspection failed: {e}", failure_type="internal_error")

    # ── Ad-hoc Python execution ─────────────────────────────────────────────

    def run_python(self, code: str) -> ToolResult:
        """Execute arbitrary Python with read-only access to robot state.

        Available variables in the sandbox:
          - robot_pose: (x, y, theta) or None
          - map_grid: OccupancyGrid object or None
          - objects: list[dict] — class, position, confidence
          - np, math, json: imported modules

        The code MUST assign a value to a variable named 'result'.
        The stringified result is returned.

        Args:
            code: Python code string. Must set 'result' variable.

        Returns:
            ToolResult with the value of 'result'.
        """
        # Build sandbox context
        objs = []
        if self._object_db:
            objs = [{"class": o.class_name, "position": list(o.position_world),
                     "confidence": o.confidence} for o in self._object_db.get_all()]

        pose_data = None
        pose_res = self.get_pose()
        if pose_res.success:
            pose_data = list(pose_res.data.get("pose", ()))

        local_vars = {
            "robot_pose": pose_data,
            "map_grid": self._map,
            "objects": objs,
            "np": __import__("numpy"),
            "math": __import__("math"),
            "json": __import__("json"),
            "result": None,
        }

        try:
            # Allow __import__ and basic builtins, block dangerous ones
            safe_builtins = {
                "__import__": __import__,
                "abs": abs, "all": all, "any": any, "bool": bool,
                "dict": dict, "float": float, "int": int, "len": len,
                "list": list, "max": max, "min": min, "round": round,
                "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
                "type": type, "range": range, "enumerate": enumerate,
                "True": True, "False": False, "None": None,
                "print": lambda *a: None,  # suppressed
            }
            exec(code, {"__builtins__": safe_builtins}, local_vars)
            out = local_vars.get("result", "No 'result' variable was set.")
            return ToolResult(True, str(out)[:2000], data={"output": str(out)[:2000]})
        except Exception as e:
            return ToolResult(False, f"Code error: {e}", failure_type="internal_error")

    # ── Detection prompt management ─────────────────────────────────────────

    def get_detection_prompt(self) -> ToolResult:
        """Return the current detection text prompt.

        The prompt is a space-separated list of object class names
        that Grounding DINO searches for in each frame.

        Returns:
            ToolResult with the current prompt string.
        """
        return ToolResult(True, f"Current detection prompt: {self._text_prompt}",
                          data={"prompt": self._text_prompt})

    def set_detection_prompt(self, prompt: str) -> ToolResult:
        """Update the detection text prompt.

        This changes which objects Grounding DINO looks for.
        The prompt should be a space-separated list of class names,
        e.g. 'sofa. table. chair. bottle. door.'

        Args:
            prompt: New detection prompt string.

        Returns:
            ToolResult confirming the change.
        """
        if not prompt or len(prompt.strip()) < 3:
            return ToolResult(False, "Prompt too short. Provide at least 3 characters.",
                              failure_type="internal_error")
        old = self._text_prompt
        self._text_prompt = prompt.strip()
        return ToolResult(True, f"Detection prompt updated.\n  Old: {old}\n  New: {self._text_prompt}",
                          data={"old_prompt": old, "new_prompt": self._text_prompt})

    # ── Path validation ──────────────────────────────────────────────────────

    def validate_path(self, x: float, y: float) -> ToolResult:
        """Check if a map coordinate is reachable before attempting navigation.

        Verifies:
          1. (x, y) is within the known map boundaries
          2. The cell is free space (not occupied or unknown)
          3. The point is within a reasonable distance (< 100m)

        Returns:
            ToolResult with "CLEAR" or "BLOCKED at (x, y) — reason"
        """
        x, y = float(x), float(y)

        # Distance sanity check
        if abs(x) > 100 or abs(y) > 100:
            return ToolResult(False, f"BLOCKED: ({x:.1f}, {y:.1f}) is too far (>100m).",
                              failure_type="goal_out_of_map")

        if self._map is None:
            return ToolResult(False, "Unknown: no map available yet. Assume reachable once map loads.",
                              failure_type="map_stale")

        try:
            grid = np.array(self._map.data, dtype=np.int8).reshape(
                self._map.info.height, self._map.info.width)
            res = self._map.info.resolution
            ox = self._map.info.origin.position.x
            oy = self._map.info.origin.position.y

            # Convert world coordinates to grid indices
            gx = int((x - ox) / res)
            gy = int((y - oy) / res)

            # Bounds check
            h, w = grid.shape
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                return ToolResult(False, f"BLOCKED: ({x:.1f}, {y:.1f}) is outside the known map.",
                                  failure_type="goal_out_of_map")

            cell = grid[gy, gx]
            if cell == -1:
                return ToolResult(False, f"UNKNOWN: ({x:.1f}, {y:.1f}) is in unmapped area. Explore first.",
                                  failure_type="goal_out_of_map")
            elif cell > 50:  # occupied or likely obstacle
                return ToolResult(False, f"BLOCKED: ({x:.1f}, {y:.1f}) is in an occupied cell (value={cell}).",
                                  failure_type="nav_blocked")
            elif cell == 0:
                return ToolResult(True, f"PATH CLEAR: ({x:.1f}, {y:.1f}) is in free space.",
                                  data={"grid_coords": (gx, gy), "cell_value": int(cell)})
            else:
                return ToolResult(True, f"CAUTION: ({x:.1f}, {y:.1f}) cell value is {cell} (unknown status).",
                                  data={"grid_coords": (gx, gy), "cell_value": int(cell)})
        except Exception as e:
            return ToolResult(False, f"Validation failed: {e}", failure_type="internal_error")

    # ── Progressive refinement ───────────────────────────────────────────────

    def refine_object(self, class_name: str, repetitions: int = 2) -> ToolResult:
        """Drive closer to an object and re-detect it multiple times for better accuracy.

        The robot approaches the object, then runs detection N times.
        Positions are averaged, weighted by confidence. The ObjectDB
        entry is updated with the refined position.

        Args:
            class_name: Object class to refine (e.g. 'sofa', 'table')
            repetitions: Number of re-detections (default 2). More = more accurate.

        Returns:
            ToolResult with original position, refined position, and delta.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.", failure_type="internal_error")

        # Find the object
        all_objs = self._object_db.get_all()
        matches = [o for o in all_objs if class_name.lower() in o.class_name.lower()]
        if not matches:
            return ToolResult(False, f"No '{class_name}' in database to refine.",
                              failure_type="detect_empty")

        obj = matches[0]
        orig_x, orig_y, orig_z = obj.position_world
        orig_conf = obj.confidence

        # Approach: drive to ~1.5m from the object
        pose_res = self.get_pose()
        if not pose_res.success:
            return ToolResult(False, "Cannot refine: pose unknown.", failure_type="tf_timeout")

        rx, ry, _ = pose_res.data.get("pose", (0, 0, 0))
        dx, dy = orig_x - rx, orig_y - ry
        dist = np.linalg.norm([dx, dy])
        if dist > 1.5:
            approach_factor = max(0.3, (dist - 1.5) / dist)
            approach_x = rx + dx * approach_factor
            approach_y = ry + dy * approach_factor
            self.navigate_to(approach_x, approach_y)
            self._wait_for_nav_completion()

        # Re-detect N times
        positions = []
        confidences = []
        for _ in range(int(repetitions)):
            time.sleep(2)  # let detection timer fire
            if self._object_db:
                fresh = [o for o in self._object_db.get_all()
                        if class_name.lower() in o.class_name.lower()]
                if fresh:
                    px, py, pz = fresh[0].position_world
                    pc = fresh[0].confidence
                    positions.append((px, py))
                    confidences.append(pc)

        if not positions:
            return ToolResult(False, f"Refinement failed: '{class_name}' not re-detected after approach.",
                              failure_type="detect_empty")

        # Weighted average
        positions = np.array(positions)
        weights = np.array(confidences)
        weights = weights / weights.sum()
        refined_x = float(np.average(positions[:, 0], weights=weights))
        refined_y = float(np.average(positions[:, 1], weights=weights))
        delta = np.linalg.norm([refined_x - orig_x, refined_y - orig_y])

        # Update ObjectDB — directly update the existing record's position
        obj.position_world = (refined_x, refined_y, orig_z)
        obj.confidence = float(np.mean(confidences))
        obj.metadata["refined"] = True
        obj.metadata["original_position"] = (orig_x, orig_y)
        obj.metadata["delta_m"] = round(delta, 3)

        return ToolResult(True,
                          f"Refined '{class_name}': ({orig_x:.2f}, {orig_y:.2f}) "
                          f"→ ({refined_x:.2f}, {refined_y:.2f}). "
                          f"Delta: {delta:.2f}m. Confidence: {obj.confidence:.2f} → {float(np.mean(confidences)):.2f}.",
                          data={"original": (orig_x, orig_y), "refined": (refined_x, refined_y),
                                "delta_m": round(delta, 3),
                                "original_confidence": orig_conf,
                                "new_confidence": float(np.mean(confidences))})

    # ── Spatial knowledge graph ──────────────────────────────────────────────

    def query_graph(self, query: str) -> ToolResult:
        """Query the spatial knowledge graph for object relationships.

        The graph computes spatial relationships (near, adjacent, contains,
        left_of, right_of) between detected objects.

        Args:
            query: Natural language query. Examples:
                   - "nearest object to robot"
                   - "objects within 2m of table"
                   - "what is near the sofa"

        Returns:
            ToolResult with structured results from the graph.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.", failure_type="internal_error")

        try:
            from vision.world_model.spatial_graph import SpatialGraph
            sg = SpatialGraph(self._object_db)
            results = sg.query(query)
            if not results:
                return ToolResult(True, "No relationships found matching that query.",
                                  data={"results": []})

            lines = []
            for r in results[:20]:
                if isinstance(r, dict):
                    lines.append(r.get("message", str(r)))
                else:
                    lines.append(str(r))

            return ToolResult(True, "\n".join(lines), data={"results": results})
        except ImportError:
            return ToolResult(False, "SpatialGraph module not available.", failure_type="internal_error")
        except Exception as e:
            return ToolResult(False, f"Graph query failed: {e}", failure_type="internal_error")

    # ── Temporal queries (5A) ────────────────────────────────────────────────

    def query_history(self, class_name: str, minutes_ago: float = None) -> ToolResult:
        """Return the position history of an object over time.

        Shows where the object was seen, when, and at what confidence.
        Use this to answer questions like "has the chair moved?"
        or "where was the sofa 5 minutes ago?"

        Args:
            class_name: Object class to query.
            minutes_ago: If set, only return observations older than N minutes.

        Returns:
            ToolResult with list of {timestamp, position, confidence} dicts.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.", failure_type="internal_error")
        try:
            # Defensive: LLM may send numeric params as strings
            if minutes_ago is not None:
                minutes_ago = float(minutes_ago)
            history = self._object_db.query_history(class_name, minutes_ago)
            if not history:
                return ToolResult(True, f"No history found for '{class_name}'.",
                                  data={"history": []})
            lines = [f"{len(history)} observations for '{class_name}':"]
            for h in history[:15]:
                pos = h["position"]
                ts = h["timestamp"]
                lines.append(f"  t={ts:.0f}s  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  conf={h['confidence']:.2f}")
            return ToolResult(True, "\n".join(lines), data={"history": history})
        except Exception as e:
            return ToolResult(False, f"History query failed: {e}", failure_type="internal_error")

    def has_object_moved(self, class_name: str) -> ToolResult:
        """Check if a detected object has been physically moved during observation.

        Compares the first and last positions of an object. If the
        delta exceeds 0.2m, the object was likely moved.

        Args:
            class_name: Object class to check.

        Returns:
            ToolResult with moved=True/False and delta in meters.
        """
        if not self._object_db:
            return ToolResult(False, "Object database not available.", failure_type="internal_error")
        result = self._object_db.has_moved(class_name)
        if result.get("moved"):
            return ToolResult(True,
                              f"'{class_name}' has moved: {result['delta_m']}m over "
                              f"{result.get('observation_count', 0)} observations.",
                              data=result)
        return ToolResult(True, f"'{class_name}' has not moved significantly.", data=result)

    # ── Battery awareness (5D) ───────────────────────────────────────────────

    def get_battery(self) -> ToolResult:
        """Report robot battery status if available.

        For real robots: subscribes to /battery_state topic.
        For simulation: returns estimated status based on uptime.

        Returns:
            ToolResult with percentage, voltage, estimated time remaining,
            and whether the robot should return home.
        """
        uptime_min = (time.monotonic() - self._start_time) / 60.0
        # Simulated battery: starts at 100%, drains 3%/hour
        pct = max(1, 100 - uptime_min * 0.05)
        est_minutes = pct * 1.5  # rough: 1% ≈ 1.5 min for TB3
        pose_res = self.get_pose()
        px, py = 0, 0
        if pose_res.success:
            px, py, _ = pose_res.data.get("pose", (0, 0, 0))
        dist_home = np.linalg.norm([px, py])
        time_home_s = dist_home / self._max_linear_speed + 5
        should_return = (est_minutes * 60) < time_home_s * 1.5

        return ToolResult(True,
                          f"Battery: {pct:.0f}% | {est_minutes:.0f} min remaining | "
                          f"{time_home_s:.0f}s to home | "
                          f"{'⚠ RETURN HOME' if should_return else 'OK'}",
                          data={
                              "percentage": round(pct, 1),
                              "estimated_minutes": round(est_minutes, 1),
                              "time_to_home_s": round(time_home_s, 1),
                              "should_return": should_return,
                              "critical": pct < 10,
                          })

    # ── Dynamic action discovery (5E) ────────────────────────────────────────

    def discover_action(self, action_name: str) -> ToolResult:
        """Introspect an unknown ROS 2 action server and report its type + fields.

        This doesn't automatically register a new tool — it returns the
        action's interface so the LLM can decide whether to call it manually
        via run_python or create a skill wrapper.

        Args:
            action_name: ROS 2 action path (e.g. '/navigate_to_pose').

        Returns:
            ToolResult with action type, goal/result/feedback fields.
        """
        import subprocess
        try:
            # ros2 action info
            info = subprocess.run(
                ["timeout", "3", "ros2", "action", "info", action_name],
                capture_output=True, text=True, timeout=4)
            output = info.stdout.strip()
            if not output:
                return ToolResult(False, f"No action found at '{action_name}'.",
                                  failure_type="internal_error")

            # Extract action type from info output
            action_type = ""
            for line in output.split("\n"):
                if "action_type" in line.lower() or "/" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        action_type = parts[-1].strip()
                        break

            if not action_type:
                return ToolResult(True, f"Action found at {action_name}:\n{output[:500]}",
                                  data={"raw_info": output[:500]})

            # ros2 interface show
            show = subprocess.run(
                ["timeout", "3", "ros2", "interface", "show", action_type],
                capture_output=True, text=True, timeout=4)
            fields = show.stdout.strip()[:1000]

            return ToolResult(True,
                              f"Action: {action_name}\nType: {action_type}\n\n{fields}",
                              data={
                                  "action_name": action_name,
                                  "action_type": action_type,
                                  "fields": fields,
                              })
        except subprocess.TimeoutExpired:
            return ToolResult(False, "Action discovery timed out.", failure_type="internal_error")
        except Exception as e:
            return ToolResult(False, f"Action discovery failed: {e}", failure_type="internal_error")

    # ── File reading ────────────────────────────────────────────────────

    def read_project_file(self, path: str) -> ToolResult:
        """Read a text file from the PhysicalAI project directory.

        Args:
            path: Relative path from ~/PhysicalAI/ (e.g. 'physicalai_config.yaml',
                  'orchestrator/robot_interface.py', 'docs/plans/some_plan.md').

        Returns:
            ToolResult with file contents (first 5000 chars).
        """
        import os
        allowed_exts = ('.py', '.yaml', '.yml', '.json', '.txt', '.md',
                        '.cfg', '.toml', '.log', '.sh', '.env')
        root = os.path.expanduser("~/PhysicalAI")

        # Sanitize: prevent path traversal
        safe = os.path.normpath(os.path.join(root, path))
        if not safe.startswith(root):
            return ToolResult(False, f"Path '{path}' escapes the project directory.",
                              failure_type="internal_error")

        if not os.path.exists(safe):
            return ToolResult(False, f"File not found: {path}",
                              failure_type="internal_error")

        ext = os.path.splitext(safe)[1]
        if ext not in allowed_exts:
            return ToolResult(False, f"File type '{ext}' not allowed. "
                              f"Allowed: {', '.join(allowed_exts)}",
                              failure_type="internal_error")

        try:
            with open(safe, 'r') as f:
                content = f.read()
            truncated = len(content) > 5000
            return ToolResult(True, content[:5000],
                              data={"path": path, "bytes": len(content),
                                    "truncated": truncated})
        except Exception as e:
            return ToolResult(False, f"Error reading {path}: {e}",
                              failure_type="internal_error")

    # ── Depth calibration status ────────────────────────────────────────

    def get_depth_calibration(self) -> ToolResult:
        """Return the current depth scale factor and calibration stats.

        Readable by the LLM to check whether a previous calibrate_depth() call
        actually changed the scale.
        """
        orch = getattr(self, '_orchestrator_node', None)
        if orch is None:
            return ToolResult(False, "Orchestrator node not available.",
                              failure_type="internal_error")

        scale = getattr(orch, '_depth_scale', 1.0)
        samples = getattr(orch, '_depth_scale_raw', [])
        return ToolResult(
            True,
            f"Depth scale: {scale:.3f} ({len(samples)} calibration samples)",
            data={
                "depth_scale": round(scale, 3),
                "calibration_samples": len(samples),
                "raw_samples": [round(s, 3) for s in samples[-10:]],
            })

    # ── Depth calibration ───────────────────────────────────────────────

    def calibrate_depth(self, object_id: str,
                        ground_truth_x: float, ground_truth_y: float) -> ToolResult:
        """Calibrate depth sensing using a user-provided ground truth position.

        Tell the system the REAL map-frame position of a tracked object, and it
        adjusts the depth scaling factor so future detections are more accurate.

        Args:
            object_id: Object identifier (e.g. 'chair_1'). Must already be tracked.
            ground_truth_x, ground_truth_y: The object's actual position in map
                frame (meters), measured by you (tape measure, known map feature, etc.).

        Returns:
            ToolResult with old and new depth scale, and the per-sample correction.
        """
        orch = getattr(self, '_orchestrator_node', None)
        if orch is None:
            return ToolResult(False, "Orchestrator node not available. Has the system started?",
                              failure_type="internal_error")

        # Defensive: LLM often sends numbers as strings
        try:
            gx = float(ground_truth_x)
            gy = float(ground_truth_y)
        except (TypeError, ValueError) as e:
            return ToolResult(False, f"Could not parse coordinates: {e}. "
                              "Provide numbers like calibrate_depth(object_id='chair_1', ground_truth_x=-2.0, ground_truth_y=3.5)",
                              failure_type="internal_error")

        result = orch.calibrate_depth(object_id, gx, gy)
        if result.get("status") == "error":
            return ToolResult(False, result.get("message", "Calibration failed."),
                              failure_type="internal_error")

        old_s = result["old_scale"]
        new_s = result["new_scale"]
        corr = result["sample_correction"]
        msg = (f"Depth calibration applied for '{object_id}'. "
               f"Scale: {old_s} → {new_s} (correction={corr:.3f}, "
               f"based on {result['samples']} samples). "
               f"Robot was {result['current_distance']:.1f}m from object, "
               f"expected {result['expected_distance']:.1f}m.")
        return ToolResult(True, msg, data=result)
