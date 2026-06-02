"""
map_orchestrator.py — Main Orchestrator Node

Coordinates slam_toolbox (live SLAM), Nav2 (localization + navigation),
and Grounded SAM 2 detection.  Transforms all detections to map frame,
stores them in ObjectDB, and provides services for autonomous exploration
and object-driven navigation.
"""

import sys, os, time, signal, threading, json

import numpy as np
import cv2
import torch

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)
sys.path.insert(0, os.path.join(PHYSICALAI_ROOT, "Grounded-SAM-2"))

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.world_model.object_db import ObjectDB
from orchestrator.launcher import NavLauncher
from orchestrator.tf_bridge import TFBridge
from orchestrator.drift_monitor import DriftMonitor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from example_interfaces.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge

TEXT_PROMPT = "sphere. shelf. table. chair. human. fire hydrant. stop sign. box. cup. book. bottle. pot. trash can. furniture. sofa. desk. door. plant."
DETECT_INTERVAL = 5
FRONTIER_CHECK_INTERVAL = 2.0  # seconds between frontier searches
SAVE_MAP_ON_EXIT = os.path.join(PHYSICALAI_ROOT, "output", "orchestrator_map")


class MapOrchestrator(Node):
    """Main orchestrator: SLAM + Nav2 + object detection in one node."""

    def __init__(self):
        super().__init__("map_orchestrator")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        self._log("MapOrchestrator starting...")

        # ── Config ──
        cfg = load_vision_config(
            path=os.path.join(PHYSICALAI_ROOT, "config.json"),
            depth_source="depth_anything",
        )
        self.fx, self.fy = cfg.fx, cfg.fy
        self.cx, self.cy = cfg.cx, cfg.cy
        self.intrinsics_received = False

        # ── Detection pipeline ──
        self._log("Loading Grounded SAM 2...")
        self.detector = GroundedSAM2Wrapper(cfg)
        self._log("Model loaded.")

        # ── Depth fallback ──
        from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper
        self._log("Loading Depth Anything V2 (fallback depth)...")
        self.depth_estimator = DepthAnythingWrapper(
            encoder="vits",
            checkpoint_path=os.path.join(
                PHYSICALAI_ROOT, "depth_anything_v2", "checkpoints",
                "depth_anything_v2_vits.pth",
            ),
            device="cuda",
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
        )
        self._log("Depth Anything loaded.")

        # ── TF bridge ──
        self.tf_bridge = TFBridge(self)

        # ── Camera bridge ──
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_scan = None
        self.global_costmap = None
        self.local_costmap = None
        self.rgb_received = False
        self.depth_received = False
        self._rgb_first_time = 0.0
        self.frame_count = 0
        self.detections = []

        # ── Depth calibration ──
        self._depth_scale = 1.0        # multiplicative correction for depth readings
        self._depth_scale_raw = []     # raw correction samples for diagnostics
        self._depth_scale_max_samples = 20  # EMA window

        # ── Subs ──
        self.rgb_sub = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self.info_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self.global_costmap_callback, 10)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self.local_costmap_callback, 10)

        # ── Nav2 action client ──
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ── Goal visualization (for RViz) ──
        self._goal_viz_pub = self.create_publisher(
            PoseStamped, "/exploration_goal", 10)

        # ── Manual goal services ──
        self._list_objects_srv = self.create_service(
            Trigger, "/list_objects", self._list_objects_cb)
        self._go_to_nearest_srv = self.create_service(
            Trigger, "/go_to_nearest_object", self._go_to_nearest_cb)
        self._go_by_class_srv = self.create_service(
            Trigger, "/go_to_object_class", self._go_by_class_cb)

        # ── ObjectDB ──
        self._embedder = None
        # Persistence: env var set from physicalai_config.yaml
        db_path = os.environ.get("PHYSICALAI_PERSIST_DB", "")
        if db_path:
            self.db = ObjectDB(embedder=None, db_path=db_path, persistence=True)
            self._log(f"Object memory persisted to {db_path}")
        else:
            self.db = ObjectDB(embedder=None)
            self._log("Object memory in-memory (no persistence)")
        self.seen_this_cycle = set()
        self._warned_no_depth = False

        # ── State ──
        self.current_map: OccupancyGrid = None
        self.map_lock = threading.Lock()
        self._exploring_enabled = False
        self._goal_active = False
        self._goal_handle = None
        self._goal_result_future = None
        self._start_time = time.monotonic()

        # ── Drift monitoring (semantic loop closure) ──
        self.drift_monitor = DriftMonitor(drift_threshold=0.3, min_observations=3)

        # ── Safety monitor (reactive collision avoidance) ──
        from orchestrator.safety_monitor import SafetyMonitor
        self.safety = SafetyMonitor(self, forward_angle_deg=40.0, danger_distance_m=0.35)

        # ── Processing timer ──
        self.create_timer(0.1, self.process)

        # ── Safety check timer (runs at 20Hz, faster than Nav2's control loop) ──
        self.create_timer(0.05, self._safety_tick)

        # ── Launch SLAM + Nav2 ──
        self._nav = NavLauncher()
        self._log("Starting slam_toolbox...")
        self._nav.start_slam()
        self._log("Starting Nav2...")
        self._nav.start_nav2()

        # ── Stats ──
        self.fps_count = 0
        self.fps_start = time.perf_counter()
        self.fps_display = 0.0
        self.first_frame_time = None
        self.output_dir = os.path.join(PHYSICALAI_ROOT, "output", "orchestrator")
        os.makedirs(self.output_dir, exist_ok=True)

        self._log("MapOrchestrator ready.")

    # ── Manual goal services ─────────────────────────────────────────────────

    def _list_objects_cb(self, req, resp):
        """List all objects in the ObjectDB."""
        all_objs = self.db.get_all()
        if not all_objs:
            resp.success = True
            resp.message = "No objects in database."
            return resp

        lines = [f"{len(all_objs)} objects in database:"]
        for obj in sorted(all_objs, key=lambda o: o.class_name):
            x, y, z = obj.position_world
            conf = obj.confidence
            lines.append(f"  {obj.class_name:15s}  ({x:.2f}, {y:.2f}, {z:.2f})  conf={conf:.2f}")
        resp.success = True
        resp.message = "\n".join(lines)
        return resp

    def _go_to_nearest_cb(self, req, resp):
        """Navigate to the nearest detected object."""
        all_objs = self.db.get_all()
        if not all_objs:
            resp.success = False
            resp.message = "No objects in database to navigate to."
            return resp

        if self._goal_active:
            resp.success = False
            resp.message = "Already navigating to a goal. Cancel first or wait."
            return resp

        # Pick the first (sorted by distance from origin — rough proxy)
        obj = all_objs[0]
        x, y, z = obj.position_world
        self._log(f"Manual goal: navigating to {obj.class_name} at ({x:.2f}, {y:.2f})")
        self._send_nav_goal(x, y)
        resp.success = True
        resp.message = f"Navigating to {obj.class_name} at ({x:.2f}, {y:.2f})"
        return resp

    def _go_by_class_cb(self, req, resp):
        """Navigate to the nearest object matching a class name substring."""
        all_objs = self.db.get_all()
        if not all_objs:
            resp.success = False
            resp.message = "No objects in database."
            return resp

        # req.data = True means call with a specific class — but Trigger has no data field.
        # We use it as a simpler service to go to the highest-confidence object.
        # For class-specific, we'd need a custom service type — use /go_to_nearest_object instead.
        return self._go_to_nearest_cb(req, resp)

    def _log(self, msg: str):
        self.get_logger().info(msg)
        # Push to dashboard
        try:
            from orchestrator.dashboard_server import push_log
            push_log("info", msg)
        except Exception:
            pass

    def _warn(self, msg: str):
        self.get_logger().warn(msg)
        try:
            from orchestrator.dashboard_server import push_log
            push_log("warn", msg)
        except Exception:
            pass

    # ── Callbacks ───────────────────────────────────────────────────────────

    def info_callback(self, msg: CameraInfo):
        if not self.intrinsics_received and len(msg.k) >= 9:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])
            self.intrinsics_received = True
            self._log(f"Camera: fx={self.fx:.1f} fy={self.fy:.1f} "
                      f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not self.rgb_received:
                self.rgb_received = True
                self._rgb_first_time = time.monotonic()
                self._log(f"RGB: {self.latest_rgb.shape[1]}x{self.latest_rgb.shape[0]}")
        except Exception as e:
            self._log(f"RGB error: {e}")

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            if not self.depth_received:
                self.depth_received = True
                self._log("Depth frame received (using sim depth).")
        except Exception as e:
            self._log(f"Depth error: {e}")

    def map_callback(self, msg: OccupancyGrid):
        with self.map_lock:
            self.current_map = msg

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        if hasattr(self, 'safety'):
            self.safety.update_scan(msg)

    def global_costmap_callback(self, msg: OccupancyGrid):
        self.global_costmap = msg

    def local_costmap_callback(self, msg: OccupancyGrid):
        self.local_costmap = msg

    # ── Safety monitor tick (20Hz) ──────────────────────────────────────

    def _safety_tick(self):
        """Run the safety monitor on a fast timer. Calls emergency stop via /cmd_vel."""
        if hasattr(self, 'safety'):
            self.safety.check()

    # ── Main process loop ──────────────────────────────────────────────────

    def process(self):
        if self.latest_rgb is None:
            return

        frame_bgr = self.latest_rgb.copy()
        h, w = frame_bgr.shape[:2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        self.frame_count += 1
        is_detect = (self.frame_count % DETECT_INTERVAL == 0)
        now = time.monotonic()

        # ── Depth source ──
        if self.latest_depth is None:
            if self.rgb_received and time.monotonic() - self._rgb_first_time > 5.0:
                if not self._warned_no_depth:
                    self._warned_no_depth = True
                    self._log("No depth topic — falling back to Depth Anything V2.")
            depth_map = (
                self.depth_estimator.estimate(frame_bgr)
                if is_detect else None
            )
        else:
            depth_map = self.latest_depth.copy().astype(np.float32)

        if self.first_frame_time is None:
            self.first_frame_time = time.perf_counter()
            self._log("Processing started.")

        # ── Detection ──
        if is_detect and depth_map is not None:
            prompt = getattr(self, '_text_prompt', TEXT_PROMPT)
            self.detections = self.detector.detect(prompt, frame_bgr)
            self.seen_this_cycle = set()

            # Lazy-load embedder on first detection
            if self._embedder is None:
                try:
                    from vision.reid.embedding_matcher import EmbeddingMatcher
                    self._embedder = EmbeddingMatcher()
                    self.db._embedder = self._embedder
                    self._log("CLIP embedder loaded for object re-identification.")
                except Exception as e:
                    self._log(f"Embedder unavailable (re-ID disabled): {e}")

            for r in self.detections:
                u, v = r["centroid_2d"]
                d = 0.0
                try:
                    patch = depth_map[max(0, v - 1):min(h, v + 2),
                                      max(0, u - 1):min(w, u + 2)]
                    valid = patch[(patch > 0.001) & (~np.isnan(patch)) &
                                  (~np.isinf(patch))]
                    d = float(np.median(valid)) if len(valid) > 0 else 0.0
                    # Apply depth calibration scale
                    d *= self._depth_scale
                except Exception:
                    pass

                if d > 0.001:
                    cam_x = (u - cx) * d / fx
                    cam_y = d
                    cam_z = -(v - cy) * d / fy

                    # Transform to map frame
                    if self.tf_bridge.transform_available():
                        mx, my, mz = self.tf_bridge.point_camera_to_map(
                            cam_x, cam_y, cam_z)
                        if mx is not None:
                            pos = (mx, my, mz)
                        else:
                            pos = (cam_x, cam_y, cam_z)
                            self._log(f"  {r['class_name']}: no TF yet, "
                                      "using camera-frame position")
                    else:
                        pos = (cam_x, cam_y, cam_z)
                else:
                    pos = (0.0, 0.0, 0.0)

                # Crop RGB at bbox for CLIP embedding
                x1, y1, x2, y2 = map(int, r["bbox_xyxy"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                rgb_crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

                obs = dict(
                    camera="tb3", confidence=r["confidence"],
                    timestamp=time.monotonic(), centroid_2d=(u, v), depth=d,
                    bbox_xyxy=r["bbox_xyxy"], frame=self.frame_count,
                )

                # Add or update via CLIP-based re-identification
                obj, action = self.db.add(
                    rgb_crop=rgb_crop,
                    class_name=r["class_name"],
                    position_world=pos,
                    confidence=r["confidence"],
                    observation=obs,
                )
                self.seen_this_cycle.add(obj.object_id)

                if action == "new":
                    self._log(f"  NEW: {obj.object_id} at ({pos[0]:.2f}, {pos[1]:.2f})")

                # ── Drift monitoring ──
                if self.tf_bridge.transform_available() and pos[2] != 0.0:
                    try:
                        cam_pose = self.tf_bridge.get_camera_pose()
                        if cam_pose:
                            drift_info = self.drift_monitor.observe(
                                r["class_name"], cam_pose, pos)
                            if drift_info:
                                self._log(
                                    f"Drift detected: {drift_info['mean_error_m']}m "
                                    f"via '{r['class_name']}' "
                                    f"({drift_info['observations']} observations)")
                    except Exception:
                        pass

        # ── Build display ──
        display = frame_bgr.copy()
        for r in self.detections:
            u, v = r["centroid_2d"]
            x1, y1, x2, y2 = map(int, r["bbox_xyxy"])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(display, r["class_name"], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # ── FPS ──
        self.fps_count += 1
        if (elapsed := time.perf_counter() - self.fps_start) >= 1.0:
            self.fps_display = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.perf_counter()

        det_tag = " [DETECT]" if is_detect else ""
        cv2.putText(display,
                    f"FPS:{self.fps_display:.1f} | Det:{len(self.detections)}{det_tag}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Save every 30 frames (quietly, no log)
        if self.frame_count % 30 == 0:
            p = os.path.join(self.output_dir,
                             f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(p, display)

    # ── Exploration ──────────────────────────────────────────────────────────

    def _check_goal_result(self):
        """Check if the active Nav2 goal has completed."""
        if self._goal_result_future is None:
            self._goal_active = False
            return

        if not self._goal_result_future.done():
            return

        try:
            result = self._goal_result_future.result()
            self._log(f"Nav2 goal finished: {result.status}")
        except Exception as e:
            self._log(f"Nav2 goal failed: {e}")
        finally:
            self._goal_active = False
            self._goal_handle = None
            self._goal_result_future = None

    def _send_nav_goal(self, x: float, y: float):
        """Send a navigate_to_pose goal asynchronously."""
        if not self._nav_client.wait_for_server(timeout_sec=0.5):
            self._log("Nav2 not available yet.")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)

        from geometry_msgs.msg import Quaternion
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, 0.0)
        goal.pose.pose.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])

        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_sent_callback)

    def _goal_sent_callback(self, future):
        """Callback when a Nav2 goal is accepted."""
        gh = future.result()
        if not gh.accepted:
            self._log("Nav2 goal rejected.")
            self._goal_active = False
            return

        self._goal_handle = gh
        self._goal_active = True
        self._goal_result_future = gh.get_result_async()

    def _publish_initial_pose_once(self):
        """Publish initial pose estimate so AMCL can localize on the map.
        Publishes only once, 5 seconds after startup."""
        if hasattr(self, '_pose_published'):
            return
        if time.monotonic() - self._start_time < 5.0:
            return

        self._pose_published = True
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        from geometry_msgs.msg import Quaternion
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, 0.0)
        pose.pose.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])

        self._initial_pose_pub = self.create_publisher(
            PoseStamped, "/initialpose", 10)
        self._initial_pose_pub.publish(pose)
        self._log(f"Initial pose published at (0, 0).")

    def find_exploration_goals(self) -> list:
        """Find reachable exploration goals in FREE space near frontiers.

        Instead of returning frontier centroids (which lie in unknown cells),
        this finds the nearest FREE cell adjacent to each frontier cluster.
        This gives Nav2 a valid path-planning target.
        """
        with self.map_lock:
            if self.current_map is None:
                return []
            occ = self.current_map
            grid = np.array(occ.data, dtype=np.int8).reshape(
                occ.info.height, occ.info.width)
            res = occ.info.resolution
            ox, oy = occ.info.origin.position.x, occ.info.origin.position.y

        free = grid == 0
        unknown = grid == -1
        if not np.any(unknown):
            return []

        # Frontier = unknown adjacent to free (the boundary of known space)
        from scipy.ndimage import binary_dilation, label
        free_dilated = binary_dilation(free, iterations=1)
        frontier_mask = unknown & free_dilated

        if not np.any(frontier_mask):
            return []

        # Label frontier clusters
        labeled, n_features = label(frontier_mask)

        goals = []
        for i in range(1, n_features + 1):
            ys, xs = np.where(labeled == i)
            # For each frontier cluster, find the FREE cell closest to its centroid
            cent_y = ys.mean()
            cent_x = xs.mean()

            # Search a growing window around the centroid for a FREE cell
            found = False
            for radius in range(1, 50):
                y_min = max(0, int(round(cent_y)) - radius)
                y_max = min(grid.shape[0], int(round(cent_y)) + radius + 1)
                x_min = max(0, int(round(cent_x)) - radius)
                x_max = min(grid.shape[1], int(round(cent_x)) + radius + 1)

                window_free = free[y_min:y_max, x_min:x_max]
                free_ys, free_xs = np.where(window_free)
                if len(free_ys) > 0:
                    gy = free_ys[0] + y_min
                    gx = free_xs[0] + x_min
                    goals.append((float(gx) * res + ox, float(gy) * res + oy))
                    found = True
                    break
            if found:
                break  # just return the nearest one

        goals.sort(key=lambda c: c[0]**2 + c[1]**2)
        return goals

    # ── Depth calibration ───────────────────────────────────────────────

    def calibrate_depth(self, object_id: str,
                        ground_truth_x: float, ground_truth_y: float) -> dict:
        """Adjust depth calibration from a user-provided ground truth position.

        Works by: look up the object's last raw observation (pixel + raw depth),
        compute what depth WOULD have produced the given map-frame position,
        derive scale = expected_depth / raw_depth, and blend into the running EMA.

        Args:
            object_id: Object identifier (e.g. 'chair_1').
            ground_truth_x, ground_truth_y: Real/measured position in map frame.

        Returns:
            dict with {status, old_scale, new_scale, derived_correction, samples}.
        """
        obj = self.db.get(object_id)
        if not obj:
            return {"status": "error", "message": f"Object '{object_id}' not found."}

        # Need the last observation's centroid + raw depth
        last_obs = obj.observations[-1] if obj.observations else None
        if not last_obs:
            msg = (f"Object '{object_id}' has no observations with pixel data. "
                   "Wait for a re-detection first.")
            return {"status": "error", "message": msg}

        u, v = last_obs.get("centroid_2d", (0, 0))

        # Use the robot's current camera pose and the ground truth position
        # to compute the expected range, then derive the depth correction.
        import numpy as np
        import math

        robot_pose = self.tf_bridge.get_camera_pose() if hasattr(self, 'tf_bridge') and self.tf_bridge else None
        if not robot_pose:
            return {"status": "error", "message": "Robot pose not available for calibration."}

        rx, ry, _ = robot_pose

        # Ground truth vector from robot to object
        dx_gt = ground_truth_x - rx
        dy_gt = ground_truth_y - ry
        expected_distance = math.sqrt(dx_gt*dx_gt + dy_gt*dy_gt)

        # Object's current estimated distance (from stored position_world)
        obj_x, obj_y = obj.position_world[0], obj.position_world[1]
        current_dist = math.sqrt((obj_x - rx)**2 + (obj_y - ry)**2)

        if current_dist < 0.01 or expected_distance < 0.01:
            return {"status": "error",
                    "message": f"Distances too small: current={current_dist:.2f}m, expected={expected_distance:.2f}m"}

        # The correction: depth_scale should change by expected/current ratio
        computed_scale = self._depth_scale * (expected_distance / current_dist)

        # Sanity: clamp to [0.1, 10.0]
        computed_scale = max(0.1, min(10.0, computed_scale))

        old_scale = self._depth_scale
        n = len(self._depth_scale_raw)

        # Running median of calibration samples
        self._depth_scale_raw.append(computed_scale)
        if n > self._depth_scale_max_samples:
            self._depth_scale_raw.pop(0)
        # New scale = median of samples (robust to outliers)
        self._depth_scale = float(np.median(self._depth_scale_raw))

        self._log(
            f"[Calibrate] '{object_id}': robot→obj "
            f"current={current_dist:.2f}m, expected={expected_distance:.2f}m, "
            f"correction={computed_scale:.3f}, "
            f"scale: {old_scale:.3f} → {self._depth_scale:.3f} "
            f"(median of {len(self._depth_scale_raw)} samples)")

        return {
            "status": "ok",
            "old_scale": round(old_scale, 3),
            "new_scale": round(self._depth_scale, 3),
            "sample_correction": round(computed_scale, 3),
            "samples": len(self._depth_scale_raw),
            "current_distance": round(current_dist, 2),
            "expected_distance": round(expected_distance, 2),
        }

    # ── Cleanup ──

    def quit_callback(self):
        self._log("=== ObjectDB Snapshot ===")
        for oid, info in self.db.snapshot().items():
            self._log(
                f"  {info['class_name']:12s}  pos={info['position_world']}  "
                f"conf={info['confidence']:.2f}  "
                f"seen={info['observation_count']}x")

        self._log("Saving map...")
        os.makedirs(SAVE_MAP_ON_EXIT, exist_ok=True)
        subprocess.run(
            ["ros2", "run", "nav2_map_server", "map_saver_cli",
             "-f", SAVE_MAP_ON_EXIT],
            capture_output=True, timeout=10,
        )

        self._log("Stopping SLAM + Nav2...")
        self._nav.stop()
        self._log("Shutdown complete.")
        rclpy.shutdown()


def main():
    node = None
    try:
        rclpy.init()
        node = MapOrchestrator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            try:
                node.quit_callback()
            except Exception as e:
                print(f"[Cleanup] Error during quit_callback: {e}", flush=True)
        # Last-resort kill: clean up any remaining ROS 2 + background processes
        print("[Cleanup] Final sweep: terminating any remaining subprocesses...")
        try:
            import subprocess
            subprocess.run(
                ["pkill", "-f", "slam_toolbox"],
                capture_output=True, timeout=3)
            subprocess.run(
                ["pkill", "-f", "nav2_bringup"],
                capture_output=True, timeout=3)
        except Exception:
            pass
        print("[Cleanup] Done.")


if __name__ == "__main__":
    main()
