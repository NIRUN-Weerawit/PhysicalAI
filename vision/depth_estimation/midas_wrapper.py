"""
MiDaS wrapper for monocular depth estimation from single RGB.
Uses PyTorch Hub (intel-isl/MiDaS). Falls back gracefully.
"""
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from .base import DepthEstimator


class MiDaSWrapper(DepthEstimator):
    """Monocular depth estimation via MiDaS models (DPT / MiDaS family).

    Supports: DPT_BEiT_L_384 (best), DPT_Large, DPT_Hybrid, MiDaS_small (fastest)
    """

    def __init__(
        self,
        model_type: str = "DPT_BEiT_L_384",
        device: str = "cuda",
        grayscale: bool = True,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ):
        self.model_type = model_type
        self.device = device
        self.grayscale = grayscale
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._model = None
        self._transform = None

    def _lazy_init(self):
        if self._model is not None:
            return
        print(f"[MiDaSWrapper] Loading MiDaS model: {self.model_type}")
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self._model = self._model.to(self.device).eval()

            # Get the appropriate transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_type in ["DPT_BEiT_L_384", "DPT_Large", "DPT_Hybrid"]:
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform
        except Exception as e:
            print(f"[MiDaSWrapper] Failed to load MiDaS: {e}")
            print("  Falling back to zero depth (placeholder)")
            self._model = None

    @torch.inference_mode()
    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        self._lazy_init()
        h, w = image_bgr.shape[:2]

        if self._model is None:
            return np.zeros((h, w), dtype=np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        return depth

    def get_intrinsics(self) -> dict:
        return {"fx": self._fx, "fy": self._fy, "cx": self._cx, "cy": self._cy}

    @property
    def name(self) -> str:
        return f"midas_{self.model_type}"
