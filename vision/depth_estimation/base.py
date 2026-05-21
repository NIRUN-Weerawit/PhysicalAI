"""
Abstract base class for all depth estimators.
"""
from abc import ABC, abstractmethod
import numpy as np


class DepthEstimator(ABC):
    """Produces a depth map from an RGB image (or raw depth for RGB-D)."""

    @abstractmethod
    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return depth map: HxW float32, meters, same aspect ratio as input.

        Depth convention: smaller values = closer to camera.
        Returns None if depth is unavailable.
        """
        ...

    @abstractmethod
    def get_intrinsics(self) -> dict:
        """Return camera intrinsics dict with keys: fx, fy, cx, cy."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
