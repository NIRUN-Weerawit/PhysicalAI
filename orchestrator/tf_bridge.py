"""
tf_bridge.py — Transform detections from camera frame to map frame.

Uses tf2_ros.Buffer + TransformListener to look up the transform chain:
    camera_rgb_optical_frame → base_link → odom → map

and applies it to a 3D point to get a map-frame position.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, TransformStamped
import time


class TFBridge(Node):
    """Look up TF transforms to convert points between frames."""

    def __init__(self, node: Node):
        self._node = node
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._log = node.get_logger().info

    def point_camera_to_map(
        self, x: float, y: float, z: float, timeout: float = 0.5
    ) -> tuple:
        """
        Transform a 3D point from camera_rgb_optical_frame → map.

        Args:
            x, y, z: Position in camera frame (OpenCV convention:
                      x=right, y=forward/depth, z=up)
            timeout: Max seconds to wait for transform

        Returns:
            (map_x, map_y, map_z) in map frame, or (None, None, None) if transform fails.
        """
        try:
            # Create a point stamped in camera frame
            p = PointStamped()
            p.header.frame_id = "camera_rgb_optical_frame"
            p.header.stamp = self._node.get_clock().now().to_msg()
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = float(z)

            # Look up transform: camera → map
            transform: TransformStamped = self._tf_buffer.lookup_transform(
                "map",
                "camera_rgb_optical_frame",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout),
            )

            # Apply
            transformed = do_transform_point(p, transform)
            return (
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            )

        except Exception as e:
            self._node.get_logger().warn(
                f"TF transform failed ({x:.2f},{y:.2f},{z:.2f}): {e}"
            )
            return (None, None, None)

    def transform_available(self) -> bool:
        """Check if the camera→map transform chain is ready."""
        try:
            self._tf_buffer.can_transform(
                "map",
                "camera_rgb_optical_frame",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return True
        except Exception:
            return False
