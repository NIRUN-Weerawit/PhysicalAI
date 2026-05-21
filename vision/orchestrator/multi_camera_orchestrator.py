"""
Multi-Camera Vision Orchestrator for PhysicalAI.

Ties together:
  - Multiple cameras, each running Grounded SAM 2 detection
  - Per-camera depth estimation (RGB-D or monocular)
  - Cross-camera object matching (same physical object → same ID)
  - Transform tree (world ↔ cameras ↔ robot arm)
  - Object database (persistent world model)

The orchestrator is what the robot control layer talks to.
"""
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

from vision.configs.config import VisionConfig
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper
from vision.depth_estimation.midas_wrapper import MiDaSWrapper
from vision.depth_estimation.rgbd_camera import RGBDCamera
from vision.cross_camera.matcher import CrossCameraMatcher, CameraCalibration
from vision.tf_tree.transform_tree import TransformTree
from vision.world_model.object_db import ObjectDB, ObjectRecord


class CameraConfig:
    """Per-camera configuration."""
    def __init__(self, name: str, depth_source: str = "depth_anything",
                 camera_id: int = 0, use_simulated_depth: bool = True):
        self.name = name
        self.depth_source = depth_source
        self.camera_id = camera_id
        self.use_simulated_depth = use_simulated_depth


class MultiCameraOrchestrator:
    """High-level orchestrator that:

    1 Takes a text prompt and images from N cameras
    2. Runs detection + depth + 3D projection per camera
    3. Matches across cameras → unified object IDs
    4. Updates the world model
    5. Answers spatial queries from the robot controller

    Typical robot workflow:

        orch = MultiCameraOrchestrator(cameras=[...], tf_config=[...])

        # Every cycle (e.g. 1-5 Hz):
        images = {c.config.name: camera_grab() for c in orch.cameras}
        orch.run_cycle("box. can. bottle.", images)

        # Orchestrator queries:
        nearest_can_to_arm = orch.find_nearest_to("arm/tcp", "can", radius=1.0)
        print(f"Grasp at: {nearest_can_to_arm.position_world}")
    """

    def __init__(
        self,
        cameras: list,       # list of CameraConfig
        tf_config: list = None,  # list of frame config dicts
        config: VisionConfig = None,
        db_path: str = ":memory:",
        spatial_threshold: float = 0.05,
    ):
        self.cfg = config or VisionConfig()
        self._cameras = cameras
        self._detectors: dict = {}
        self._depth_estimators: dict = {}
        self._calibrations: dict = {}
        self.tf = TransformTree()
        self.db = ObjectDB(db_path=db_path)

        # TF tree setup
        self._setup_tf(tf_config)

        # Build per-camera pipelines
        for cam in cameras:
            self._build_camera(cam)

        calibrations = list(self._calibrations.values())
        self.matcher = CrossCameraMatcher(calibrations, spatial_threshold=spatial_threshold)

        print(f"[MultiCameraOrchestrator] Ready: {len(cameras)} camera(s), TF tree:")
        self.tf.print_tree()

    # ---- Setup ----

    def _setup_tf(self, tf_config):
        """Register all coordinate frames in the TF tree."""
        if tf_config is None:
            tf_config = self._default_tf_config()
        for tc in tf_config:
            if tc["type"] == "camera":
                self.tf.set_static(
                    f"world/{tc['name']}",
                    tc["translation"],
                    tc.get("rotation"),
                    parent=tc.get("parent", "world"),
                )
            elif tc["type"] == "arm":
                self.tf.set_static(
                    f"world/{tc['name']}_base",
                    tc["base_translation"],
                    tc.get("base_rotation"),
                    parent=tc.get("parent", "world"),
                )
                self.tf.set_static(
                    f"{tc['name']}_base/tcp",
                    tc.get("tcp_offset", [0, 0, 0]),
                    tc.get("tcp_rotation"),
                    frame_parent=f"{tc['name']}_base",
                )

    def _default_tf_config(self) -> list:
        """Default workspace: two cameras, one arm."""
        return [
            {
                "type": "camera",
                "name": "camera_top",
                "translation": [0.0, 0.0, 1.2],   # 1.2m above table
                "rotation": [0, 0, 0, 1],         # looking down (negative Z down)
            },
            {
                "type": "camera",
                "name": "camera_front",
                "translation": [0.0, 0.8, 1.0],   # 80cm in front
                "rotation": [1, 0, 0, 0],         # looking straight
            },
            {
                "type": "arm",
                "name": "arm",
                "base_translation": [0.5, 0.0, 0.0],
                "base_rotation": [1, 0, 0, 0],
                "tcp_offset": [0.3, 0.0, 0.5],    # 30cm forward, 50cm up from base
                "tcp_rotation": [1, 0, 0, 0],
            },
        ]

    def _build_camera(self, cam_config: CameraConfig):
        """Initialize detection + depth for one camera."""
        name = cam_config.name
        print(f"  [Orchestrator] Building camera: {name} (depth={cam_config.depth_source})")

        # Each camera shares the SAME Grounded SAM 2 model (it's stateless wrt images
        # after set_image() is called per frame, so we just use one shared instance)
        if name not in self._detectors:
            cam_spec = VisionConfig()
            cam_spec.device = self.cfg.device
            cam_spec.sam2_checkpoint = self.cfg.sam2_checkpoint
            cam_spec.sam2_model_config = self.cfg.sam2_model_config
            cam_spec.box_threshold = self.cfg.box_threshold
            cam_spec.text_threshold = self.cfg.text_threshold
            cam_spec.multimask_output = self.cfg.multimask_output

            # Create detectors per camera (SAM 2 model is loaded once per instance,
            # but each needs its own for separate set_image calls)
            self._detectors[name] = GroundedSAM2Wrapper(cam_spec)

        # Depth estimator per camera (can differ: one RGB-D, one monocular)
        if cam_config.depth_source == "rgbd":
            self._depth_estimators[name] = RGBDCamera(
                camera_id=cam_config.camera_id,
                use_simulated=cam_config.use_simulated_depth,
            )
        elif cam_config.depth_source == "depth_anything":
            self._depth_estimators[name] = DepthAnythingWrapper(
                encoder=self.cfg.depth_anything_encoder,
                device=self.cfg.device,
                grayscale=self.cfg.depth_anything_grayscale,
            )
        elif cam_config.depth_source == "midas":
            self._depth_estimators[name] = MiDaSWrapper(
                model_type=self.cfg.midas_model_type,
                device=self.cfg.device,
            )

        # Camera calibration (intrinsics + extrinsics in TF tree)
        calib_intrinsics = {
            "fx": self.cfg.fx,
            "fy": self.cfg.fy,
            "cx": self.cfg.cx,
            "cy": self.cfg.cy,
        }
        # Extrinsics come from TF tree → extract T_world→camera
        try:
            T_wc = self.tf.lookup_latest("world", f"world/{name}")
            extr = {
                "translation": T_wc[:3, 3].tolist(),
                "rotation": [1, 0, 0, 0],  # identity quaternion
            }
        except KeyError:
            extr = None

        self._calibrations[name] = CameraCalibration(
            name=name,
            intrinsics=calib_intrinsics,
            extrinsics=extr,
        )

    # ---- Main processing cycle ----

    def run_cycle(
        self,
        text_prompt: str,
        images: dict,
        timestamp: float = None,
    ) -> list:
        """Full cycle: detect → depth → 3D → match → update DB.

        Args:
            text_prompt: e.g. "box. can. bottle."
            images: {"camera_top": image_bgr, "camera_front": image_bgr_2}
            timestamp: optional monotonic time; defaults to now

        Returns:
            List of MatchedObject (from CrossCameraMatcher)
        """
        if timestamp is None:
            timestamp = time.monotonic()

        detections_by_cam = {}
        color_hists_by_cam = {}

        for cam_name, image_bgr in images.items():
            if cam_name not in self._detectors:
                continue

            # 1. Detect
            detector = self._detectors[cam_name]
            dets = detector.detect(text_prompt, image_bgr)

            # 2. Depth estimate
            depth_estimator = self._depth_estimators[cam_name]
            depth_map = depth_estimator.estimate(image_bgr)

            # 3. 3D project each detection
            intrinsics = depth_estimator.get_intrinsics()
            fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

            enriched = []
            for det in dets:
                u, v = det["centroid_2d"]
                h, w = image_bgr.shape[:2]
                d = self._sample_depth(depth_map, u, v)

                # Camera frame → world frame: x=right, y=depth(forward), z=up
                x_w = (u - cx) * d / fx if d > 0.001 else 0.0
                y_w = d if d > 0.001 else 0.0
                z_w = -(v - cy) * d / fy if d > 0.001 else 0.0

                det["depth_at_centroid"] = float(d)
                det["position_3d_cam"] = (float(x_w), float(y_w), float(z_w))
                det["has_valid_depth"] = d > 0.001

                # Extract color histogram for ReID
                if det.get("mask") is not None:
                    mask = det["mask"]
                    rgb = image_bgr if image_bgr.shape[2] == 3 else image_bgr
                    # Convert BGR→RGB for color features
                    if cam_name in ["camera_top", "camera_front"]:
                        import cv2
                        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_img = image_bgr
                    det["color_hist"] = self._extract_color_hist(rgb_img, mask)

                enriched.append(det)

            detections_by_cam[cam_name] = enriched
            color_hists_by_cam[cam_name] = [
                d.get("color_hist", np.zeros(48)) for d in enriched
            ]

            print(f"  [{cam_name}] {len(enriched)} detections. Depth source: "
                  f"{depth_estimator.name}")

        # 4. Cross-camera matching
        match_results = self.matcher.match(detections_by_cam, color_hists_by_cam)

        # 5. Update object database
        for mo in match_results:
            self._update_object_db(mo, timestamp)

        print(f"  [Orchestrator] Matched {len(match_results)} unique objects in world model "
              f"(DB has {len(self.db)} total)")

        return match_results

    def _update_object_db(self, matched_obj, timestamp: float):
        """Upsert a matched object into the world model."""
        # If object already known, update it.  Otherwise, create.
        existing = self.db.get(matched_obj.object_id)
        if existing:
            # Append new observations
            for det in matched_obj.detections:
                existing.observations.append({
                    "camera": det["camera"],
                    "confidence": det["confidence"],
                    "timestamp": timestamp,
                    "centroid_2d": det["centroid_2d"],
                    "depth": det.get("depth", 0.0),
                })
            existing.position_world = matched_obj.world_position
            existing.confidence = max(existing.confidence, matched_obj.confidence)
            existing.timestamp = timestamp
        else:
            obs = [
                {
                    "camera": d["camera"],
                    "confidence": d["confidence"],
                    "timestamp": timestamp,
                    "centroid_2d": d["centroid_2d"],
                    "depth": d.get("depth", 0.0),
                }
                for d in matched_obj.detections
            ]
            new_obj = ObjectRecord(
                object_id=matched_obj.object_id,
                class_name=matched_obj.class_name,
                position_world=matched_obj.world_position,
                confidence=matched_obj.confidence,
                timestamp=timestamp,
                first_seen=timestamp,
                observations=obs,
            )
            self.db.add(new_obj)

    # ---- Orchestrator query interface ----

    def find_objects(self, class_filter: str = None) -> list:
        """All known objects (optionally filtered by class)."""
        if class_filter:
            return self.db.query_by_class(class_filter)
        return self.db.get_all()

    def find_nearest_to(self, frame_name: str,
                        class_filter: str = None,
                        max_results: int = 1) -> list:
        """Find objects nearest to a named frame (e.g. arm/tcp).

        This is the KEY method the robot controller uses:
          "Give me the coordinates of the nearest can to my end-effector."

        Returns objects with their position in world frame AND
        position relative to the queried frame.
        """
        try:
            # Get frame origin in world coords
            T_wf = self.tf.lookup_latest("world", frame_name)
            frame_origin_world = tuple(T_wf[:3, 3])
        except KeyError:
            raise KeyError(f"Frame '{frame_name}' not in TF tree. Known frames: "
                          + ", ".join(self.tf._children.keys()))

        objects = self.db.get_all()
        if class_filter:
            objects = [o for o in objects
                      if class_filter.lower() in o.class_name.lower()]

        # Sort by distance to frame
        import numpy as np
        frame_np = np.array(frame_origin_world)
        objects.sort(key=lambda o: np.linalg.norm(
            np.array(o.position_world) - frame_np))

        # Augment with position relative to the queried frame
        for obj in objects[:max_results]:
            pos_world = np.array(obj.position_world)
            pos_in_frame = (T_wf[:3, :3].T @ (pos_world - T_wf[:3, 3]))
            obj._position_relative_to_frame = tuple(pos_in_frame.tolist())

        return objects[:max_results]

    def grasp_point(self, class_filter: str, arm_frame: str = "arm/tcp") -> Optional[dict]:
        """Return a grasp target for the arm.

        Finds the nearest object of the given class to the arm TCP and
        returns it in the TCP's coordinate frame.
        """
        nearest = self.find_nearest_to(arm_frame, class_filter, max_results=1)
        if not nearest:
            return None
        obj = nearest[0]
        return {
            "object_id": obj.object_id,
            "class_name": obj.class_name,
            "position_world": obj.position_world,
            "position_in_tcp": getattr(obj, "_position_relative_to_frame", obj.position_world),
            "confidence": obj.confidence,
            "observed_by": [d["camera"] for d in obj.observations],
        }

    def scene_state(self) -> dict:
        """Current snapshot of the world model as JSON-serializable dict."""
        return self.db.snapshot()

    # ---- Helpers ----

    @staticmethod
    def _sample_depth(depth_map, u, v, radius=3):
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return 0.0
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0.001]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    @staticmethod
    def _extract_color_hist(image_rgb, mask, bins=16):
        """Simple histogram of the masked region for appearance matching."""
        import cv2
        if not np.any(mask):
            return np.zeros(bins * 3)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        a_vals = lab[mask, 1].astype(int)
        b_vals = lab[mask, 2].astype(int)
        l_vals = lab[mask, 0].astype(int)
        hist = []
        for vals in (l_vals, a_vals, b_vals):
            pt = np.ptp(vals)
            if pt < 1:
                hist.extend([1.0 / bins] * bins)
                continue
            norm = np.clip((vals - vals.min()) / pt * bins, 0, bins - 1).astype(int)
            h = np.bincount(norm, minlength=bins).astype(np.float32)
            n = np.linalg.norm(h)
            if n > 0:
                h /= n
            hist.extend(h)
        return np.array(hist, dtype=np.float32)
