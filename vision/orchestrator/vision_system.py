"""
Unified VisionSystem: Grounded SAM 2 detection + selectable depth source + 3D projection.

Usage:
    from vision.orchestrator.vision_system import VisionSystem

    vs = VisionSystem(depth_source="depth_anything")
    results = vs.process_frame("car. person. box.", image_bgr)
    for obj in results:
        print(obj["class_name"], obj["position_3d"])  # (x, y, z) in camera frame
"""
import os
import numpy as np
from pathlib import Path

from vision.configs.config import VisionConfig, DepthSource
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper
from vision.depth_estimation.midas_wrapper import MiDaSWrapper
from vision.depth_estimation.rgbd_camera import RGBDCamera
from vision.depth_estimation.base import DepthEstimator


class VisionSystem:
    """Top-level vision system orchestrator.

    Manages:
      - 2D detection (Grounded SAM 2)
      - Depth estimation (runtime-selectable source)
      - 2D-to-3D projection via camera intrinsics
    """

    def __init__(
        self,
        config: VisionConfig = None,
        depth_source: DepthSource = None,
    ):
        self.cfg = config or VisionConfig()
        if depth_source:
            self.cfg.depth_source = depth_source

        print(f"[VisionSystem] Initializing with depth_source={self.cfg.depth_source}")
        self._init_detection()
        self._init_depth()

    def _init_detection(self):
        print("[VisionSystem] Loading Grounded SAM 2...")
        self.detector = GroundedSAM2Wrapper(self.cfg)
        print("[VisionSystem] Grounded SAM 2 ready.")

    def _init_depth(self):
        src = self.cfg.depth_source
        print(f"[VisionSystem] Loading depth estimator: {src}")

        if src == "rgbd":
            self.depth_estimator = RGBDCamera(
                camera_id=self.cfg.rgbd_camera_id,
                width=self.cfg.rgbd_width,
                height=self.cfg.rgbd_height,
                fps=self.cfg.rgbd_fps,
                align_depth=self.cfg.rgbd_align_depth,
                use_simulated=self.cfg.rgbd_use_simulated,
                sim_fx=self.cfg.rgbd_sim_fx,
                sim_fy=self.cfg.rgbd_sim_fy,
                sim_cx=self.cfg.rgbd_sim_cx,
                sim_cy=self.cfg.rgbd_sim_cy,
            )
        elif src == "depth_anything":
            self.depth_estimator = DepthAnythingWrapper(
                encoder=self.cfg.depth_anything_encoder,
                checkpoint_path=self.cfg.depth_anything_checkpoint,
                device=self.cfg.device,
                grayscale=self.cfg.depth_anything_grayscale,
                fx=self.cfg.fx, fy=self.cfg.fy,
                cx=self.cfg.cx, cy=self.cfg.cy,
            )
        elif src == "midas":
            self.depth_estimator = MiDaSWrapper(
                model_type=self.cfg.midas_model_type,
                device=self.cfg.device,
                grayscale=self.cfg.midas_grayscale,
                fx=self.cfg.fx, fy=self.cfg.fy,
                cx=self.cfg.cx, cy=self.cfg.cy,
            )
        else:
            raise ValueError(f"Unknown depth_source: {src}")

        print(f"[VisionSystem] Depth estimator ready: {self.depth_estimator.name}")

    def process_frame(
        self,
        text_prompt: str,
        image_bgr: np.ndarray,
    ) -> list:
        """Full pipeline: detect + depth estimate + 3D project.

        Args:
            text_prompt: e.g. "car. person. box."
            image_bgr: HxWx3 BGR numpy array

        Returns:
            List of dicts:
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "centroid_2d": (u_px, v_px),
                    "depth_at_centroid": float,   # meters
                    "position_3d": (x, y, z),     # meters in camera frame
                    "has_valid_depth": bool,
                }
        """
        # Step 1: 2D detection
        detections = self.detector.detect(text_prompt, image_bgr)

        if not detections:
            return []

        # Step 2: Depth estimation
        depth_map = self.depth_estimator.estimate(image_bgr)
        intrinsics = self.depth_estimator.get_intrinsics()
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

        # Step 3: 3D projection for each detection
        results = []
        for det in detections:
            u, v = det["centroid_2d"]
            d = self._sample_depth(depth_map, u, v)

            if d > 0.001:
                # Camera frame → world frame:
                #   x = right, y = forward (depth), z = up
                x = float((u - cx) * d / fx)
                y = float(d)
                z = float(-(v - cy) * d / fy)
                has_depth = True
            else:
                x, y, z = 0.0, 0.0, 0.0
                has_depth = False

            results.append({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_xyxy": det["bbox_xyxy"],
                "centroid_2d": det["centroid_2d"],
                "depth_at_centroid": float(d),
                "position_3d": (x, y, z),
                "has_valid_depth": has_depth,
            })

        return results

    def _sample_depth(self, depth_map: np.ndarray, u: int, v: int, radius: int = 3) -> float:
        """Sample depth at (u,v) with a small local median for robustness."""
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return 0.0

        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        patch = depth_map[y0:y1, x0:x1]

        valid = patch[patch > 0.001]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    def process_video(
        self,
        text_prompt: str,
        video_path: str,
        output_dir: str = "./output",
        max_frames: int = -1,
    ) -> list:
        """Process video frame-by-frame, returning per-frame detections."""
        import cv2
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total = min(total, max_frames)

        print(f"[VisionSystem] Processing video: {total} frames @ {fps:.1f} fps")
        all_results = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_idx >= max_frames):
                break

            results = self.process_frame(text_prompt, frame)
            all_results.append({
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "detections": results,
            })

            if self.cfg.save_visualizations and results:
                vis_path = str(Path(output_dir) / f"frame_{frame_idx:06d}.jpg")
                self.detector.visualize(frame, results, vis_path)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total} frames...")

        cap.release()
        print(f"[VisionSystem] Video done: {frame_idx} frames, "
              f"{sum(len(r['detections']) for r in all_results)} total detections")

        if self.cfg.dump_json_results:
            import json
            json_path = output_dir / "detections.json"
            # Make JSON-serializable
            serializable = []
            for fr in all_results:
                fr_copy = dict(fr)
                fr_copy["detections"] = [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in d.items()}
                    for d in fr["detections"]
                ]
                serializable.append(fr_copy)
            with open(json_path, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"[VisionSystem] Results saved to {json_path}")

        return all_results

    def switch_depth_source(self, new_source: DepthSource):
        """Hot-switch depth estimator at runtime."""
        if new_source == self.cfg.depth_source:
            return
        print(f"[VisionSystem] Switching depth: {self.cfg.depth_source} -> {new_source}")
        self.cfg.depth_source = new_source
        self._init_depth()

    def release(self):
        """Release any camera resources."""
        if isinstance(self.depth_estimator, RGBDCamera):
            self.depth_estimator.stop()
