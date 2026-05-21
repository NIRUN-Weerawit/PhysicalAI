"""
Cross-camera object matcher: determines whether detections from
Camera A and Camera B refer to the same physical object.

Two-stage approach:
  Stage 1 — Spatial: transform both detections into a shared world frame
          and check if their 3D positions overlap.
  Stage 2 — Appearance: compare color histograms extracted from the
          segmentation masks to resolve ambiguous spatial matches.
"""
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class CameraCalibration:
    """Full camera calibration (intrinsics + extrinsics)."""
    name: str                          # e.g. "camera_top", "camera_front"
    intrinsics: dict                    # fx, fy, cx, cy
    extrinsics: dict | None = None     # translation [tx,ty,tz] + rotation (qw,qx,qy,qz)
                                       # Rotation is in world-from-camera convention.
                                       # If None, camera IS the world frame.

    def project_to_world(self, point_px: tuple, depth: float) -> tuple:
        """Pixel + depth → 3D in this camera's frame → 3D in world frame."""
        fx, fy, cx, cy = (
            self.intrinsics["fx"],
            self.intrinsics["fy"],
            self.intrinsics["cx"],
            self.intrinsics["cy"],
        )
        u, v = point_px
        # Pinhole unproject to camera frame (forward = +Z)
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        if self.extrinsics is None:
            return (x_cam, y_cam, z_cam)  # camera IS world

        return _apply_rigid_transform(
            np.array([x_cam, y_cam, z_cam]),
            self.extrinsics.get("translation", [0, 0, 0]),
            self.extrinsics.get("rotation", [1, 0, 0, 0]),  # wxyz quaternion
        )


# ---------------------------------------------------------------------------
# Rigid-body helpers (no external deps besides numpy + scipy)
# ---------------------------------------------------------------------------
def _quaternion_to_rotation_matrix(q: list) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _apply_rigid_transform(
    point: np.ndarray,
    translation: list,
    rotation: list,
) -> tuple:
    """R * p + t"""
    R = _quaternion_to_rotation_matrix(rotation)
    t = np.array(translation)
    result = R.dot(point) + t
    return (float(result[0]), float(result[1]), float(result[2]))


# ---------------------------------------------------------------------------
# Colour-feature extraction from mask
# ---------------------------------------------------------------------------
def _extract_color_histogram(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bins: int = 16,
) -> np.ndarray:
    """Extract a hue-based histogram for appearance matching."""
    import cv2
    if not np.any(mask):
        return np.zeros(bins * 3)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    # Flatten to masked pixels only
    a_channel = lab[mask, 1].astype(int)  # a channel: green–red
    b_channel = lab[mask, 2].astype(int)  # b channel: blue–yellow
    # Normalize to 0..bins-1
    a_bins = np.clip((a_channel - np.min(a_channel)) / (np.ptp(a_channel) or 1) * bins, 0, bins - 1).astype(int)
    b_bins = np.clip((b_channel - np.min(b_channel)) / (np.ptp(b_channel) or 1) * bins, 0, bins - 1).astype(int)
    hist_a = np.bincount(a_bins, minlength=bins).astype(np.float32)
    hist_b = np.bincount(b_bins, minlength=bins).astype(np.float32)
    # Luminance
    l_channel = lab[mask, 0].astype(int)
    l_bins = np.clip((l_channel - np.min(l_channel)) / (np.ptp(l_channel) or 1) * bins, 0, bins - 1).astype(int)
    hist_l = np.bincount(l_bins, minlength=bins).astype(np.float32)
    # Normalize
    for h in (hist_a, hist_b, hist_l):
        norm = np.linalg.norm(h)
        if norm > 0:
            h /= norm
    return np.concatenate([hist_l, hist_a, hist_b])


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection distance (0=identical, 1=opposite)."""
    return float(np.sum(np.minimum(a, b)))


# ---------------------------------------------------------------------------
# CrossCameraMatcher
# ---------------------------------------------------------------------------
@dataclass
class MatchedObject:
    """A physical object observed by one or more cameras."""
    object_id: str
    class_name: str
    world_position: tuple  # (x, y, z) in world frame
    detections: list = field(default_factory=list)
        # [{camera_name, centroid_2d, depth, color_hist, ...}]
    confidence: float = 1.0
    timestamp: float = 0.0


class CrossCameraMatcher:
    """Matches detections across multiple calibrated cameras.

    Usage:
        calibrations = [
            CameraCalibration(name="camera_top", intrinsics=..., extrinsics=...),
            CameraCalibration(name="camera_front", intrinsics=..., extrinsics=...),
        ]
        matcher = CrossCameraMatcher(calibrations, spatial_threshold=0.05)

        detections_A = detector.detect("box.", image_a)
        detections_B = detector.detect("box.", image_b)

        matches = matcher.match(
            {"camera_top": detections_A},
            {"camera_front": detections_B},
            color_histograms={"camera_top": ..., "camera_front": ...},
        )
        # matches = list of MatchedObject
    """

    def __init__(
        self,
        calibrations: list,
        spatial_threshold_m: float = 0.05,
        appearance_threshold: float = 0.5,
    ):
        self.calibrations: dict[str, CameraCalibration] = {c.name: c for c in calibrations}
        self.spatial_threshold = spatial_threshold_m
        self.appearance_threshold = appearance_threshold

    def match(
        self,
        detections_by_cam: dict[str, list],
        color_histograms: dict[str, np.ndarray] | None = None,
    ) -> list:
        """Take detections from each camera, merge into unified object list.

        Args:
            detections_by_cam: {"camera_name": [det_dict, ...], ...}
                Each det_dict has: centroid_2d, depth_at_centroid,
                    class_name, bbox_xyxy, mask, confidence
            color_histograms: optional dict {camera_name: np.ndarray per detection}
        """
        # --- Stage 1: unproject every detection into world frame ---
        unprojected: list[dict] = []  # (camera, det, world_xyz, hist)
        for cam_name, dets in detections_by_cam.items():
            calib = self.calibrations[cam_name]
            for det in dets:
                pt2d = det["centroid_2d"]
                depth = det.get("depth_at_centroid", 0.0)
                world_xyz = calib.project_to_world(pt2d, depth)

                # Color histogram for appearance matching
                hist = np.zeros(48)  # placeholder
                if color_histograms and cam_name in color_histograms:
                    det_idx = dets.index(det)
                    if det_idx < len(color_histograms[cam_name]):
                        hist = color_histograms[cam_name][det_idx]

                unprojected.append({
                    "camera": cam_name,
                    "det": det,
                    "world_xyz": np.array(world_xyz),
                    "hist": hist,
                })

        if len(unprojected) == 0:
            return []

        # --- Stage 2: cluster by spatial proximity ---
        clusters = self._spatial_cluster(unprojected)

        # --- Stage 3: resolve with appearance similarity ---
        clusters = self._resolve_by_appearance(clusters)

        # --- Stage 4: build MatchedObject list ---
        results = []
        for cluster in clusters:
            # Use the highest-confidence detection for class_name
            cluster.sort(key=lambda x: x["det"]["confidence"], reverse=True)
            best = cluster[0]["det"]

            # Centroid of all world positions
            avg_world = np.mean([c["world_xyz"] for c in cluster], axis=0)
            # Confidence: geometric mean of per-det confidences
            confs = [c["det"]["confidence"] for c in cluster]
            avg_conf = float(np.exp(np.mean(np.log([max(c, 0.001) for c in confs]))))

            results.append(MatchedObject(
                object_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{best['class_name']}_{avg_world}")),
                class_name=best["class_name"],
                world_position=tuple(avg_world.tolist()),
                detections=[
                    {
                        "camera": c["camera"],
                        "confidence": c["det"]["confidence"],
                        "centroid_2d": c["det"]["centroid_2d"],
                        "depth": c["det"].get("depth_at_centroid", 0.0),
                    }
                    for c in cluster
                ],
                confidence=avg_conf,
            ))
        return results

    # ------------------------------------------------------------------
    # Spatial clustering: detections within spatial_threshold form a group
    # ------------------------------------------------------------------
    def _spatial_cluster(self, items: list) -> list[list]:
        if not items:
            return []
        n = len(items)
        assigned = [False] * n
        clusters: list[list] = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [items[i]]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                dist = np.linalg.norm(items[i]["world_xyz"] - items[j]["world_xyz"])
                if dist < self.spatial_threshold:
                    # Also check class name overlap
                    if items[i]["det"]["class_name"] == items[j]["det"]["class_name"]:
                        cluster.append(items[j])
                        assigned[j] = True
            clusters.append(cluster)
        return clusters

    # ------------------------------------------------------------------
    # Appearance-based resolution: split clusters where color differs
    # ------------------------------------------------------------------
    def _resolve_by_appearance(self, clusters: list[list]) -> list[list]:
        refined: list[list] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                refined.append(cluster)
                continue
            # Simple heuristic: if average pairwise color similarity is
            # below the threshold, the cluster may contain two different
            # objects of the same class.  For a first version, we keep
            # the spatial cluster intact and flag low-confidence merges.
            if not self._group_color_similar(cluster):
                # Mark detections with a warning for downstream filtering
                for c in cluster:
                    c["det"]["appearance_mismatch"] = True
            refined.append(cluster)
        return refined

    @staticmethod
    def _group_color_similar(group: list) -> bool:
        """Return True if all items in group have similar color histograms."""
        hists = [g["hist"] for g in group if np.any(g["hist"])]
        if len(hists) < 2:
            return True
        # Average pairwise intersection
        avg_sim = 0.0
        count = 0
        for i in range(len(hists)):
            for j in range(i + 1, len(hists)):
                avg_sim += _histogram_similarity(hists[i], hists[j])
                count += 1
        if count == 0:
            return True
        return (avg_sim / count) > 0.6  # reasonable threshold
