#!/usr/bin/env python3
"""
tb3_detection.py — Grounded SAM 2 object detection on TurtleBot3 in Gazebo
=============================================================================

Launches or connects to a Gazebo simulation of a TurtleBot3 Waffle Pi and runs
the PhysicalAI detection pipeline (Grounded SAM 2 → 3D projection → ObjectDB)
on its onboard RGB camera stream via ROS 2 topics.

The Waffle Pi has an Intel RealSense R200-style camera with both RGB and depth
streams.  The script subscribes to both topics, runs detection every N frames,
and projects detected objects into 3D using the depth image + camera intrinsics.

USAGE
-----
    # Terminal 1 — start Gazebo + TB3 (if not already running)
    export TURTLEBOT3_MODEL=waffle_pi
    source /opt/ros/humble/setup.bash
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

    # Terminal 2 — run detection
    python3 ~/PhysicalAI/scripts/tb3_detection.py

    # Optionally drive the robot
    # Terminal 3
    export TURTLEBOT3_MODEL=waffle_pi
    ros2 run turtlebot3_teleop teleop_keyboard

DEPENDENCIES
------------
    sudo apt install ros-humble-turtlebot3-simulations

TOPICS
------
    /camera/rgb/image_raw       — RGB image (640x480, 8UC3)
    /camera/depth/image_raw     — Depth image (640x480, 32FC1)
    /camera/rgb/camera_info     — Camera intrinsics
    /odom                       — Robot odometry (for pose context)

COORDINATE CONVENTION
---------------------
    x = right, y = forward (depth), z = up  (same as live_detection.py)
    All positions are relative to the camera frame (robot-centric).
"""

import sys
import os
import json
import time
import numpy as np
import cv2
import torch
import warnings

# ── PhysicalAI pipeline ────────────────────────────────────────────────────
PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)
sys.path.insert(0, os.path.join(PHYSICALAI_ROOT, "Grounded-SAM-2"))

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.world_model.object_db import ObjectDB, ObjectRecord

warnings.filterwarnings("ignore", message=".*has been deprecated.*")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── ROS 2 ──────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ═══════════════════════════════════════════════════════════════════════════
# ROS 2 Node: camera subscriber + detection
# ═══════════════════════════════════════════════════════════════════════════
TEXT_PROMPT = "cup. bottle. book. phone. box. can. remote. mouse. keyboard. person. chair. fire hydrant. stop sign."
DETECT_INTERVAL = 5  # run detection every N received frames

class TB3DetectionNode(Node):
    def __init__(self):
        super().__init__("tb3_detection")

        # ── Config ──
        cfg = load_vision_config(
            path=os.path.join(PHYSICALAI_ROOT, "config.json"),
            depth_source="depth_anything",
        )
        # Default TB3 Waffle Pi R200 camera intrinsics (640x480)
        # These get overridden by /camera/rgb/camera_info topic
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.intrinsics_received = False

        # ── Detection pipeline ──
        self.get_logger().info("Loading Grounded SAM 2...")
        self.detector = GroundedSAM2Wrapper(cfg)
        self.get_logger().info("Model loaded.")

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_count = 0
        self.detections = []

        # ── ObjectDB ──
        self.db = ObjectDB(stale_timeout=3.0)
        self.seen_this_cycle = set()

        # ── Subscribe ──
        self.rgb_sub = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self.info_callback, 10)

        # ── Timer for processing (separate from callbacks to debounce) ──
        self.create_timer(0.1, self.process)  # 10 Hz processing

        # ── Output ──
        self.output_dir = os.path.join(PHYSICALAI_ROOT, "output", "tb3")
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Stats ──
        self.fps_count = 0
        self.fps_start = time.perf_counter()
        self.fps_display = 0.0

        self.get_logger().info(
            f"TB3 Detection Node started.\n"
            f"  Prompt: {TEXT_PROMPT}\n"
            f"  Output: {self.output_dir}/"
        )

    def info_callback(self, msg: CameraInfo):
        """Receive camera intrinsics from ROS topic."""
        if not self.intrinsics_received and msg.k == "plumb_bob":
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
            self.intrinsics_received = True
            self.get_logger().info(
                f"Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}"
            )

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def process(self):
        """Main processing loop: detect, project, display, save."""
        if self.latest_rgb is None or self.latest_depth is None:
            return

        frame_bgr = self.latest_rgb.copy()
        depth_map = self.latest_depth.copy().astype(np.float32)
        h, w = frame_bgr.shape[:2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        self.frame_count += 1
        is_detect = (self.frame_count % DETECT_INTERVAL == 0)
        now = time.monotonic()

        # ── Detection ──
        if is_detect:
            self.detections = self.detector.detect(TEXT_PROMPT, frame_bgr)
            self.seen_this_cycle = set()

            for r in self.detections:
                u, v = r["centroid_2d"]
                # Sample depth at centroid (3x3 median)
                try:
                    patch = depth_map[max(0,v-1):min(h,v+2), max(0,u-1):min(w,u+2)]
                    valid = patch[(patch > 0.001) & (~np.isnan(patch)) & (~np.isinf(patch))]
                    d = float(np.median(valid)) if len(valid) > 0 else 0.0
                except Exception:
                    d = 0.0

                if d > 0.001:
                    x_w = (u - cx) * d / fx
                    y_w = d          # depth is along camera forward axis
                    z_w = -(v - cy) * d / fy
                    r["_px"], r["_py"], r["_pz"] = x_w, y_w, z_w
                else:
                    r["_px"] = r["_py"] = r["_pz"] = 0.0

                id_key = f"{r['class_name']}_{round(r['_px'],2)}_{round(r['_py'],2)}_{round(r['_pz'],2)}"
                self.seen_this_cycle.add(id_key)

                obs = {
                    "camera": "tb3_waffle_pi",
                    "confidence": r["confidence"],
                    "timestamp": now,
                    "centroid_2d": (u, v),
                    "depth": d,
                    "bbox_xyxy": r["bbox_xyxy"],
                    "frame": self.frame_count,
                }

                existing = self.db.get(id_key)
                if existing:
                    self.db.update(id_key, (r["_px"], r["_py"], r["_pz"]),
                                   timestamp=now, observation=obs,
                                   confidence=r["confidence"])
                else:
                    new_obj = ObjectRecord(
                        object_id=id_key,
                        class_name=r["class_name"],
                        position_world=(r["_px"], r["_py"], r["_pz"]),
                        confidence=r["confidence"],
                        timestamp=now,
                        first_seen=now,
                        observations=[obs],
                        metadata={"source": "tb3_waffle_pi"},
                    )
                    self.db.add(new_obj)

            # Evict stale
            for obj in list(self.db._objects.values()):
                if obj.object_id not in self.seen_this_cycle:
                    self.db.remove(obj.object_id)

        # ── Annotate display ──
        display = frame_bgr.copy()
        for r in self.detections:
            u, v = r["centroid_2d"]
            x1, y1, x2, y2 = [int(x) for x in r["bbox_xyxy"]]
            x_w, y_w, z_w = r.get("_px", 0.0), r.get("_py", 0.0), r.get("_pz", 0.0)
            if x_w != 0.0 or y_w != 0.0:
                label = f"{r['class_name']} (x={x_w:.2f}, y={y_w:.2f}, z={z_w:.2f})m"
            else:
                label = f"{r['class_name']} (no depth)"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # ── ObjectDB overlay ──
        db_objects = self.db.get_all()
        y_off = 45
        cv2.putText(display, f"ObjectDB: {len(db_objects)} objects", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y_off += 18
        for obj in sorted(db_objects, key=lambda o: o.position_world[1]):
            x, y, z = obj.position_world
            cv2.putText(display,
                        f"  {obj.class_name:10s} x={x:.2f} y={y:.2f} z={z:.2f}",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)
            y_off += 16
            if y_off > h - 80:
                break

        # ── FPS ──
        self.fps_count += 1
        elapsed = time.perf_counter() - self.fps_start
        if elapsed >= 1.0:
            self.fps_display = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.perf_counter()

        tag = " [DETECT]" if is_detect else ""
        cv2.putText(display, f"FPS: {self.fps_display:.1f} | Detected: {len(self.detections)}{tag}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "TurtleBot3 Waffle Pi | Press 'q' to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── Show + save ──
        cv2.imshow("TB3 Detection", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info("Quit key pressed, shutting down...")
            rclpy.shutdown()
            return

        # Save every 30 frames
        if self.frame_count % 30 == 0:
            out_path = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(out_path, display)
            self.get_logger().info(
                f"[frame {self.frame_count}] FPS={self.fps_display:.1f}  "
                f"detected={len(self.detections)}  db={len(db_objects)}"
            )

    def quit_callback(self):
        self.get_logger().info("\n=== Final ObjectDB ===")
        snap = self.db.snapshot()
        if snap:
            for oid, info in snap.items():
                self.get_logger().info(
                    f"  {info['class_name']:12s}  pos={info['position_world']}  "
                    f"conf={info['confidence']:.2f}  seen={info['observation_count']}x"
                )
        rclpy.shutdown()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = TB3DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.quit_callback()


if __name__ == "__main__":
    main()
