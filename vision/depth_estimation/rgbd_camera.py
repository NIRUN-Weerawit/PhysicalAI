"""
RGB-D camera wrapper for hardware-backed depth sensing (RealSense, etc.)
Supports fallback to simulated depth for testing without hardware.
"""
import numpy as np
import cv2

from .base import DepthEstimator


class RGBDCamera(DepthEstimator):
    """Depth from physical RGB-D camera (Intel RealSense, etc.).

    Modes:
      - hardware=True: connects to a RealSense camera via pyrealsense2
      - hardware=False: simulates depth (plane + noise) for testing

    Hardware mode requires pyrealsense2. Falls back gracefully.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth: bool = True,
        use_simulated: bool = False,
        sim_fx: float = 525.0,
        sim_fy: float = 525.0,
        sim_cx: float = 320.0,
        sim_cy: float = 240.0,
        sim_depth_range: tuple = (0.2, 10.0),
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.align_depth = align_depth
        self.use_simulated = use_simulated
        self._sim_fx, self._sim_fy, self._sim_cx, self._sim_cy = sim_fx, sim_fy, sim_cx, sim_cy
        self._sim_depth_range = sim_depth_range
        self._pipeline = None

        if not use_simulated:
            self._try_connect()

    def _try_connect(self):
        """Attempt to open RealSense camera."""
        try:
            import pyrealsense2 as rs
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            profile = self._pipeline.start(config)

            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

            # Get intrinsics
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            intr = color_profile.get_intrinsics()
            self._fx, self._fy = intr.fx, intr.fy
            self._cx, self._cy = intr.cx, intr.cy

            # Alignment
            if self.align_depth:
                align_to = rs.stream.color
                self._align = rs.align(align_to)
            else:
                self._align = None

            print(f"[RGBDCamera] Connected: camera {self.camera_id}")
            print(f"  Intrinsics: fx={self._fx:.1f}, fy={self._fy:.1f}, "
                  f"cx={self._cx:.1f}, cy={self._cy:.1f}")
            print(f"  Depth scale: {self._depth_scale}")
            self._connected = True

        except ImportError:
            print("[RGBDCamera] pyrealsense2 not installed; using simulated depth")
            self.use_simulated = True
            self._connected = False
        except Exception as e:
            print(f"[RGBDCamera] Failed to connect: {e}")
            print("  Using simulated depth fallback")
            self.use_simulated = True
            self._connected = False

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return depth map. In hardware mode, fetches latest aligned frame."""
        h, w = image_bgr.shape[:2]

        if self.use_simulated:
            return self._simulate_depth(h, w)

        if not self._connected or self._pipeline is None:
            return self._simulate_depth(h, w)

        try:
            frames = self._pipeline.wait_for_frames()
            if self._align is not None:
                frames = self._align.process(frames)

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return self._simulate_depth(h, w)

            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_image *= self._depth_scale  # convert to meters

            # Resize to match input if needed
            if depth_image.shape[:2] != (h, w):
                depth_image = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST)

            return depth_image
        except Exception as e:
            print(f"[RGBDCamera] Frame read error: {e}")
            return self._simulate_depth(h, w)

    def _simulate_depth(self, h: int, w: int) -> np.ndarray:
        """Generate a synthetic depth map with a slanted plane plus noise."""
        y, x = np.ogrid[:h, :w]
        # Slanted plane: depth increases from top-left to bottom-right
        depth = 0.5 + 8.0 * (x / w + y / h) / 2.0
        depth = np.clip(depth, self._sim_depth_range[0], self._sim_depth_range[1])
        # Add noise
        depth += np.random.randn(h, w) * 0.05
        return depth.astype(np.float32)

    def get_intrinsics(self) -> dict:
        if self.use_simulated or not self._connected:
            return {
                "fx": self._sim_fx, "fy": self._sim_fy,
                "cx": self._sim_cx, "cy": self._sim_cy,
            }
        return {
            "fx": self._fx, "fy": self._fy,
            "cx": self._cx, "cy": self._cy,
        }

    def stop(self):
        """Release camera pipeline."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None

    @property
    def name(self) -> str:
        return "rgbd" if not self.use_simulated else "rgbd_simulated"
