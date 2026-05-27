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
from vision.world_model.object_db import ObjectDB, ObjectRecord
from orchestrator.launcher import NavLauncher
from orchestrator.tf_bridge import TFBridge

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge

TEXT_PROMPT = "sphere. shelf. table. chair. human. man. woman. box."
DETECT_INTERVAL = 5
FRONTIER_CHECK_INTERVAL = 2.0  # seconds between frontier searches
SAVE_MAP_ON_EXIT = os.path.join(PHYSICALAI_ROOT, "output", "orchestrator_map")


class MapOrchestrator(Node):
    """Main orchestrator: SLAM + Nav2 + object detection in one node."""

    def __init__(self):
        super().__init__("map_orchestrator")
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
        self.rgb_received = False
        self.depth_received = False
        self._rgb_first_time = 0.0
        self.frame_count = 0
        self.detections = []

        # ── Subs ──
        self.rgb_sub = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self.info_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10)

        # ── Nav2 action client ──
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ── Goal visualization (for RViz) ──
        self._goal_viz_pub = self.create_publisher(
            PoseStamped, "/exploration_goal", 10)

        # ── ObjectDB ──
        self.db = ObjectDB(stale_timeout=5.0)
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

        # ── Processing timer ──
        self.create_timer(0.1, self.process)

        # ── Exploration timer (check every 3s for frontiers when idle) ──
        self._exploring_enabled = False  # becomes True after init
        self._goal_active = False
        self._goal_handle = None
        self._goal_result_future = None
        self.create_timer(3.0, self.exploration_tick)

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

    def _log(self, msg: str):
        self.get_logger().info(msg)

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
            self.detections = self.detector.detect(TEXT_PROMPT, frame_bgr)
            self.seen_this_cycle = set()

            for r in self.detections:
                u, v = r["centroid_2d"]
                d = 0.0
                try:
                    patch = depth_map[max(0, v - 1):min(h, v + 2),
                                      max(0, u - 1):min(w, u + 2)]
                    valid = patch[(patch > 0.001) & (~np.isnan(patch)) &
                                  (~np.isinf(patch))]
                    d = float(np.median(valid)) if len(valid) > 0 else 0.0
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
                            pos = (cam_x, cam_y, cam_z)  # fallback: camera frame
                            self._log(f"  {r['class_name']}: no TF yet, "
                                      "using camera-frame position")
                    else:
                        pos = (cam_x, cam_y, cam_z)
                else:
                    pos = (0.0, 0.0, 0.0)

                id_key = (f"{r['class_name']}_{round(pos[0],2)}_"
                          f"{round(pos[1],2)}_{round(pos[2],2)}")
                self.seen_this_cycle.add(id_key)

                obs = dict(
                    camera="tb3", confidence=r["confidence"],
                    timestamp=now, centroid_2d=(u, v), depth=d,
                    bbox_xyxy=r["bbox_xyxy"], frame=self.frame_count,
                )

                existing = self.db.get(id_key)
                if existing:
                    self.db.update(
                        id_key, pos, timestamp=now,
                        observation=obs, confidence=r["confidence"])
                else:
                    self.db.add(ObjectRecord(
                        object_id=id_key, class_name=r["class_name"],
                        position_world=pos, confidence=r["confidence"],
                        timestamp=now, first_seen=now,
                        observations=[obs],
                        metadata={"source": "orchestrator"}))

            # Evict stale
            for obj in list(self.db._objects.values()):
                if obj.object_id not in self.seen_this_cycle:
                    self.db.remove(obj.object_id)

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

        # Save every 30 frames
        if self.frame_count % 30 == 0:
            db_snap = self.db.get_all()
            p = os.path.join(self.output_dir,
                             f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(p, display)
            self._log(
                f"[frame {self.frame_count}] FPS={self.fps_display:.1f}  "
                f"det={len(self.detections)}  db={len(db_snap)} "
                f"  → {p}")

    # ── Exploration ──────────────────────────────────────────────────────────

    def exploration_tick(self):
        """Periodic exploration logic: drive toward frontiers when idle."""
        # Wait for SLAM to start building map + TF to be available
        if self.current_map is None or not self.tf_bridge.transform_available():
            return

        # Publish initial pose once so AMCL can localize
        self._publish_initial_pose_once()

        # Don't start exploring until Nav2 is ready and we've had time to init
        if not self._exploring_enabled:
            if len(self.db.get_all()) > 0 or self.frame_count > 300:
                self._exploring_enabled = True
                self._log("Exploration enabled.")
            return

        # Skip if a goal is already active
        if self._goal_active:
            self._check_goal_result()
            return

        # If we found objects, stop exploring
        if len(self.db.get_all()) > 0:
            return

        # Find and go to nearest frontier
        goals = self.find_exploration_goals()
        if not goals:
            self._log("No reachable exploration goals found.")
            return

        cx, cy = goals[0]
        self._log(f"Exploring toward frontier — goal at ({cx:.2f}, {cy:.2f})")

        # Publish goal for RViz visualization
        viz_pose = PoseStamped()
        viz_pose.header.frame_id = "map"
        viz_pose.header.stamp = self.get_clock().now().to_msg()
        viz_pose.pose.position.x = cx
        viz_pose.pose.position.y = cy
        self._goal_viz_pub.publish(viz_pose)

        self._send_nav_goal(cx, cy)

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
    rclpy.init()
    node = MapOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.quit_callback()


if __name__ == "__main__":
    main()
