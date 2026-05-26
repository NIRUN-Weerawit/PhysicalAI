#!/usr/bin/env python3
"""
tb3_detection.py — Grounded SAM 2 on TurtleBot3 in Gazebo
===========================================================
Subscribes to /camera/rgb/image_raw + /camera/depth/image_raw via ROS 2,
runs Grounded SAM 2 detection, projects to 3D, logs to ObjectDB.

USAGE
-----
    export TURTLEBOT3_MODEL=waffle_pi
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py  # Terminal 1
    python3 scripts/tb3_detection.py                          # Terminal 2

Output saved to ~/PhysicalAI/output/tb3/frame_*.jpg every 30 frames.
"""

import sys, os, time, json, threading
import numpy as np, cv2, torch

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)
sys.path.insert(0, os.path.join(PHYSICALAI_ROOT, "Grounded-SAM-2"))

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.world_model.object_db import ObjectDB, ObjectRecord

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

TEXT_PROMPT = "fire hydrant. stop sign. person. chair. book. bottle. cup. box."
DETECT_INTERVAL = 5
HEADLESS = not os.environ.get("DISPLAY")


class TB3DetectionNode(Node):
    def __init__(self):
        super().__init__("tb3_detection")

        cfg = load_vision_config(
            path=os.path.join(PHYSICALAI_ROOT, "config.json"),
            depth_source="depth_anything")
        self.fx, self.fy = cfg.fx, cfg.fy
        self.cx, self.cy = cfg.cx, cfg.cy
        self.intrinsics_received = False

        self._log("Loading Grounded SAM 2...")
        self.detector = GroundedSAM2Wrapper(cfg)
        self._log("Model loaded.")

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_count = 0
        self.detections = []
        self.rgb_received = False
        self.depth_received = False

        self.db = ObjectDB(stale_timeout=3.0)
        self.seen_this_cycle = set()

        self.rgb_sub = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/rgb/camera_info", self.info_callback, 10)

        self.create_timer(0.1, self.process)

        self.output_dir = os.path.join(PHYSICALAI_ROOT, "output", "tb3")
        os.makedirs(self.output_dir, exist_ok=True)
        self.fps_count = 0
        self.fps_start = time.perf_counter()
        self.fps_display = 0.0
        self.first_frame_time = None

        self._log(f"Subscribed to /camera/rgb/image_raw, /camera/depth/image_raw")
        self._log(f"Prompt: {TEXT_PROMPT}")
        self._log(f"Output: {self.output_dir}/")

        # Detect whether OpenCV actually supports GUI
        self.has_gui = not HEADLESS
        if self.has_gui:
            try:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera...", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("TB3 Detection", blank)
                cv2.waitKey(1)
            except cv2.error:
                self.has_gui = False
                self._log("OpenCV has no GUI support — running headless, "
                          "frames saved to disk.")

        self._log(f"Headless: {not self.has_gui}")

    def _log(self, msg):
        self.get_logger().info(msg)

    def info_callback(self, msg: CameraInfo):
        if not self.intrinsics_received and len(msg.k) >= 9:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])
            self.intrinsics_received = True
            self._log(f"Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                      f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if not self.rgb_received:
                self.rgb_received = True
                self._log(
                    f"First RGB frame: {self.latest_rgb.shape[1]}x{self.latest_rgb.shape[0]}")
        except Exception as e:
            self._log(f"RGB callback error: {e}")

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            if not self.depth_received:
                self.depth_received = True
                d = self.latest_depth
                valid = d[(d > 0.01) & (~np.isnan(d))]
                self._log(
                    f"First depth frame: {d.shape[1]}x{d.shape[0]}, "
                    f"range=[{float(np.min(valid)) if len(valid) else 0:.2f}, "
                    f"{float(np.max(valid)) if len(valid) else 0:.2f}]m")
        except Exception as e:
            self._log(f"Depth callback error: {e}")

    def process(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        if self.first_frame_time is None:
            self.first_frame_time = time.perf_counter()
            self._log("First processing frame — starting detection loop.")

        frame_bgr = self.latest_rgb.copy()
        depth_map = self.latest_depth.copy().astype(np.float32)
        h, w = frame_bgr.shape[:2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        self.frame_count += 1
        is_detect = (self.frame_count % DETECT_INTERVAL == 0)
        now = time.monotonic()

        if is_detect:
            self.detections = self.detector.detect(TEXT_PROMPT, frame_bgr)
            self.seen_this_cycle = set()

            for r in self.detections:
                u, v = r["centroid_2d"]
                try:
                    patch = depth_map[max(0, v - 1):min(h, v + 2),
                                      max(0, u - 1):min(w, u + 2)]
                    valid = patch[(patch > 0.001) & (~np.isnan(patch)) &
                                  (~np.isinf(patch))]
                    d = float(np.median(valid)) if len(valid) > 0 else 0.0
                except Exception:
                    d = 0.0

                if d > 0.001:
                    r["_px"] = (u - cx) * d / fx
                    r["_py"] = d
                    r["_pz"] = -(v - cy) * d / fy
                else:
                    r["_px"] = r["_py"] = r["_pz"] = 0.0

                id_key = (f"{r['class_name']}_{round(r['_px'],2)}_"
                          f"{round(r['_py'],2)}_{round(r['_pz'],2)}")
                self.seen_this_cycle.add(id_key)

                obs = dict(camera="tb3_waffle_pi", confidence=r["confidence"],
                           timestamp=now, centroid_2d=(u, v), depth=d,
                           bbox_xyxy=r["bbox_xyxy"], frame=self.frame_count)

                existing = self.db.get(id_key)
                if existing:
                    self.db.update(id_key, (r["_px"], r["_py"], r["_pz"]),
                                   timestamp=now, observation=obs,
                                   confidence=r["confidence"])
                else:
                    self.db.add(ObjectRecord(
                        object_id=id_key, class_name=r["class_name"],
                        position_world=(r["_px"], r["_py"], r["_pz"]),
                        confidence=r["confidence"], timestamp=now,
                        first_seen=now, observations=[obs],
                        metadata={"source": "tb3_waffle_pi"}))

            for obj in list(self.db._objects.values()):
                if obj.object_id not in self.seen_this_cycle:
                    self.db.remove(obj.object_id)

        # ── Build display ──
        display = frame_bgr.copy()
        for r in self.detections:
            u, v = r["centroid_2d"]
            x1, y1, x2, y2 = map(int, r["bbox_xyxy"])
            x_w, y_w, z_w = r.get("_px", 0), r.get("_py", 0), r.get("_pz", 0)
            label = (f"{r['class_name']} ({x_w:.2f},{y_w:.2f},{z_w:.2f})m"
                     if x_w or y_w else f"{r['class_name']} (no depth)")
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        db_objects = self.db.get_all()
        y_off = 45
        cv2.putText(display, f"ObjectDB: {len(db_objects)} objects",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 255), 1)
        y_off += 18
        for obj in sorted(db_objects, key=lambda o: o.position_world[1])[:10]:
            x, y, z = obj.position_world
            cv2.putText(display,
                        f"  {obj.class_name:10s} {x:.2f} {y:.2f} {z:.2f}",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (180, 180, 255), 1)
            y_off += 14

        self.fps_count += 1
        if (elapsed := time.perf_counter() - self.fps_start) >= 1.0:
            self.fps_display = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_start = time.perf_counter()

        cv2.putText(display,
                    f"FPS:{self.fps_display:.1f} | Det:{len(self.detections)}"
                    f"{' [DETECT]' if is_detect else ''}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(display, "TB3 Waffle Pi | Ctrl+C to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)

        if self.has_gui:
            cv2.imshow("TB3 Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self._log("Quit.")
                rclpy.shutdown()
                return

        if self.frame_count % 30 == 0:
            p = os.path.join(self.output_dir,
                             f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(p, display)
            self._log(
                f"[frame {self.frame_count}] FPS={self.fps_display:.1f}  "
                f"det={len(self.detections)}  db={len(db_objects)}  "
                f"→ {p}")

    def quit_callback(self):
        self._log("=== Final ObjectDB ===")
        for oid, info in self.db.snapshot().items():
            self._log(
                f"  {info['class_name']:12s}  pos={info['position_world']}  "
                f"conf={info['confidence']:.2f}  "
                f"seen={info['observation_count']}x")
        rclpy.shutdown()
        if self.has_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


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
