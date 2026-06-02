"""
drift_monitor.py — Semantic loop closure for localization drift correction.

Uses detected objects as landmarks. When the same object class is detected
from a different pose, checks for systematic position error that indicates
SLAM drift. Reports drift ONCE per landmark, then suppresses until reset.
Compares latest observation against running median of stored positions.
"""
import time
import numpy as np
from collections import defaultdict


class DriftMonitor:
    """Detect and report SLAM drift using object landmarks.

    Fires at most ONCE per landmark class. After detection, subsequent
    calls for that class return empty dict until reset_landmark() is called.
    """

    def __init__(self, drift_threshold: float = 0.3,
                 min_observations: int = 10):
        self._landmarks = defaultdict(list)
        self._drift_threshold = drift_threshold
        self._min_observations = min_observations
        self._reported = set()  # class names that already fired
        self._corrections_applied = 0

    def observe(self, class_name: str, camera_pose: tuple,
                observed_position: tuple) -> dict:
        """Record a landmark observation and check for drift.

        Returns drift info dict ONLY on the first detection, empty dict on all
        subsequent calls for that class until reset_landmark() is called.
        """
        # Already reported for this class — suppress
        if class_name in self._reported:
            return {}

        self._landmarks[class_name].append({
            "camera_pose": camera_pose,
            "observed_position": observed_position,
            "timestamp": time.monotonic(),
        })

        return self._check_drift(class_name)

    def _check_drift(self, class_name: str) -> dict:
        obs = self._landmarks[class_name]
        if len(obs) < self._min_observations:
            return {}

        # Get all stored positions
        positions = np.array([o["observed_position"][:2] for o in obs])

        # Running median as the "true" position estimate
        median_pos = np.median(positions, axis=0)

        # Compute drift: median absolute deviation from median
        deltas = np.linalg.norm(positions - median_pos, axis=1)
        mean_drift = float(np.mean(deltas))

        if mean_drift > self._drift_threshold:
            self._reported.add(class_name)
            return {
                "drift_detected": True,
                "mean_error_m": round(mean_drift, 3),
                "observations": len(obs),
                "source_landmark": class_name,
            }

        return {}

    def apply_correction(self, drift_info: dict):
        self._corrections_applied += 1
        print(f"[DriftMonitor] Correction #{self._corrections_applied}: "
              f"{drift_info['mean_error_m']}m drift via '{drift_info['source_landmark']}'")

    def corrections_count(self) -> int:
        return self._corrections_applied

    def reset_landmark(self, class_name: str):
        """Clear stored observations and re-enable drift reporting for a class."""
        self._landmarks[class_name] = []
        self._reported.discard(class_name)