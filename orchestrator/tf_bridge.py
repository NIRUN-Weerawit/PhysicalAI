"""
tf_bridge.py — Transform detections from camera frame to map frame.

Uses tf2_ros.Buffer + TransformListener. Camera frame can be:
  1. Explicitly set via config.yaml → robot.camera_frame (recommended)
  2. Auto-detected from CAMERA_FRAMES list (tries each until one works)
"""
import os, numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from typing import Optional
import time

# Camera frame candidates — tried in order IF user doesn't specify one
DEFAULT_CAMERA_FRAMES = [
    "camera_link",                 # Ignition TB3
    "camera_rgb_frame",           # Gazebo TB3
    "camera_rgb_optical_frame",   # Real TB3 / many ROS robots
    "depth_camera_link",
    "base_link",                  # Fallback: no camera frame, use robot body
    "base_footprint",             # Fallback 2
]


class TFBridge(Node):
    """Look up TF transforms to convert points between frames."""

    def __init__(self, node: Node):
        self._node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._log = node.get_logger().info
        # User-provided frame takes priority; None falls back to auto-detect
        self._user_frame = os.environ.get("PHYSICALAI_CAMERA_FRAME", "").strip() or None
        # Resolved frame (None = not yet resolved)
        self._resolved_frame = None

    def _find_camera_frame(self):
        """Return the camera frame. Use user-provided frame or auto-detect."""
        # If user explicitly set it, use it directly (validate lazily on first use)
        if self._user_frame:
            self._resolved_frame = self._user_frame
            self._log(f"Using configured camera TF frame: {self._user_frame}")
            return self._resolved_frame

        # Auto-detect: first available TF frame from DEFAULT_CAMERA_FRAMES
        if self._resolved_frame:
            return self._resolved_frame
        for name in DEFAULT_CAMERA_FRAMES:
            try:
                self._tf_buffer.can_transform(
                    "map", name, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
                if self._tf_buffer.can_transform(
                    "map", name, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2)):
                    self._resolved_frame = name
                    self._log(f"Found camera TF frame: {name}")
                    return name
            except Exception:
                continue
        # All failed — use base_footprint as last resort
        self._resolved_frame = "base_footprint"
        self._log("No camera TF frame found, using base_footprint as fallback")
        return self._resolved_frame

    def point_camera_to_map(
        self, x: float, y: float, z: float, timeout: float = 0.5
    ) -> tuple:
        """
        Transform a 3D point from the detected camera frame → map.

        Args:
            x, y, z: Position in computer-vision camera frame computed from
                      pinhole projection: x=right, y=forward/depth, z=up.
            timeout: Max seconds to wait for transform

        Returns:
            (map_x, map_y, map_z) in map frame, or (None, None, None) if transform fails.
        """
        src_frame = self._find_camera_frame()

        # The computed point (x=right, y=forward, z=up) needs to be converted
        # to match the convention of whatever TF frame we're using.
        #
        # Convention by frame type:
        #   *camera_link / *camera_rgb_frame (ROS REP-103):
        #       x=forward, y=left, z=up
        #     → code_y (forward) = ROS x,  -code_x (-right) = ROS y,  code_z = ROS z
        #
        #   *optical_frame (OpenCV convention):
        #       x=right, y=down, z=forward
        #     → code_x = optical x,  -code_z = optical y,  code_y = optical z
        #
        # We check the frame name suffix to decide.
        is_optical = "_optical_frame" in src_frame or src_frame.endswith("_optical")

        if is_optical:
            ros_x = float(x)       # right → optical x
            ros_y = -float(z)      # up → -optical y (down)
            ros_z = float(y)       # depth/forward → optical z
        else:
            ros_x = float(y)       # depth/forward → ROS x
            ros_y = -float(x)      # right → ROS -left
            ros_z = float(z)       # up → ROS z

        try:
            p = PointStamped()
            p.header.frame_id = src_frame
            p.header.stamp = rclpy.time.Time().to_msg()  # latest TF available
            p.point.x = ros_x
            p.point.y = ros_y
            p.point.z = ros_z

            transformed = self._tf_buffer.transform(
                p, "map", timeout=rclpy.duration.Duration(seconds=timeout))
            return (
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            )

        except Exception as e:
            self._node.get_logger().warn(
                f"TF transform failed ({x:.2f},{y:.2f},{z:.2f}): {e}")
            return (None, None, None)

    def transform_available(self) -> bool:
        """Check if the camera→map transform chain is ready."""
        src = self._find_camera_frame()
        try:
            self._tf_buffer.can_transform(
                "map", src, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            return True
        except Exception:
            return False

    def get_camera_pose(self) -> tuple:
        """Get the camera's approximate (x, y, theta) in map frame.

        Uses base_footprint→map as a proxy since the camera is fixed
        on the robot body. For drift monitoring, the relative offset
        between camera and base doesn't matter — both drift together.
        """
        import math
        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_footprint",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            t = transform.transform.translation
            r = transform.transform.rotation
            from tf_transformations import euler_from_quaternion
            _, _, theta = euler_from_quaternion([r.w, r.x, r.y, r.z])
            return (t.x, t.y, theta)
        except Exception:
            return None
