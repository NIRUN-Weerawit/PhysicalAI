"""
Depth Anything V2 wrapper for monocular depth estimation from single RGB.
"""
import numpy as np
import cv2
import json
import torch
import os
from pathlib import Path

from .base import DepthEstimator


class DepthAnythingWrapper(DepthEstimator):
    """Monocular depth estimation via Depth Anything V2.

    Uses the depth_anything_v2 package cloned alongside Grounded SAM 2.
    Falls back to MiDaS-style inference if the DA2 checkpoint isn't found.
    """

    MODEL_URLS = {
        "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    }

    def __init__(
        self,
        encoder: str = "vitl",
        checkpoint_path: str = "",
        device: str = "cuda",
        grayscale: bool = True,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ):
        self.encoder = encoder
        self.grayscale = grayscale
        self.device = device
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._model = None

        # Load depth_scale from config.json if available (ChArUco-calibrated)
        self._depth_scale = None
        try:
            _config_path = str(Path(__file__).resolve().parent.parent.parent / "config.json")
            with open(_config_path) as _f:
                _cfg = json.load(_f)
            if "depth_scale" in _cfg:
                self._depth_scale = float(_cfg["depth_scale"])
                print(f"[DepthAnythingWrapper] Using calibrated depth_scale={self._depth_scale:.4f}")
        except Exception:
            pass

    def _lazy_init(self):
        if self._model is not None:
            return True

        # Try loading the depth_anything_v2 package from local repo
        import sys as _sys
        _da2_path = str(Path(__file__).resolve().parent.parent.parent / "depth_anything_v2")
        if _da2_path not in _sys.path:
            _sys.path.insert(0, _da2_path)

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            print(f"[DepthAnythingWrapper] depth_anything_v2 not importable ({e}), "
                  f"falling back to MiDaS-based inference")
            self._use_fallback = True
            return True

        model_cfgs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        cfg = model_cfgs.get(self.encoder, model_cfgs["vitl"])
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(
            self._resolve_checkpoint(), map_location="cpu"
        ))
        model = model.to(self.device).eval()
        self._model = model
        self._use_fallback = False
        return True

    def _resolve_checkpoint(self) -> str:
        """Find checkpoint or download."""
        candidates = [
            Path(f"./depth_anything_v2/checkpoints/depth_anything_v2_{self.encoder}.pth"),
            Path(f"~/PhysicalAI/depth_anything_v2/checkpoints/depth_anything_v2_{self.encoder}.pth").expanduser(),
            Path(f"checkpoints/depth_anything_v2_{self.encoder}.pth"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        print(f"[DepthAnythingWrapper] No checkpoint found at {candidates[0]}")
        print(f"  Download from: {self.MODEL_URLS.get(self.encoder, 'unknown')}")
        print(f"  Place at: {candidates[0].resolve()}")
        raise FileNotFoundError(
            f"Depth Anything V2 {self.encoder} checkpoint not found. "
            f"Download from {self.MODEL_URLS.get(self.encoder)}"
        )

    @torch.inference_mode()
    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        self._lazy_init()
        h, w = image_bgr.shape[:2]

        if self._use_fallback:
            return self._midas_fallback(image_bgr)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        disp = self._model.infer_image(image_rgb)  # HxW float32 — relative disparity

        # Resize to match original resolution
        if disp.shape[:2] != (h, w):
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)

        # Depth Anything outputs RAW inverse-depth values (near=low, far=high).
        inv = 1.0 / np.maximum(disp, 0.1)

        if self._depth_scale is not None:
            depth = inv * self._depth_scale
        else:
            p95 = float(np.percentile(inv, 95))
            if p95 < 1e-6:
                return np.zeros((h, w), dtype=np.float32)
            depth = inv * (3.0 / p95)

        # DEBUG: print depth stats every 30th frame (disabled by default)
        if not hasattr(self, '_dbg_ctr'):
            self._dbg_ctr = 0
        self._dbg_ctr += 1
        if self._dbg_ctr % 30 == 0 and os.environ.get("DEPTH_DEBUG"):
            p5, p50, p95_d = [float(np.percentile(depth, p)) for p in [5, 50, 95]]
            inv_p5, inv_p50, inv_p95 = [float(np.percentile(inv, p)) for p in [5, 50, 95]]
            import sys as _sys
            _sys.stderr.write(
                f"[DAPTH_WRAPPER] inv: p5={inv_p5:.6f} p50={inv_p50:.6f} p95={inv_p95:.6f} | "
                f"depth: p5={p5:.4f} p50={p50:.4f} p95={p95_d:.4f} | "
                f"scale={'calibrated' if self._depth_scale else f'heuristic(p95={p95:.6f})'}\n"
            )
            _sys.stderr.flush()

        depth = np.clip(depth, 0.0, 30.0).astype(np.float32)
        return depth

    def estimate_inverse(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return raw unscaled inverse depth (1/raw model output).
        This is what depth_scale multiplies against.
        """
        self._lazy_init()
        if self._use_fallback:
            return self._midas_fallback(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        disp = self._model.infer_image(image_rgb)
        h, w = image_bgr.shape[:2]
        if disp.shape[:2] != (h, w):
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        inv = 1.0 / np.maximum(disp, 0.1)
        return inv.astype(np.float32)

    def _midas_fallback(self, image_bgr: np.ndarray) -> np.ndarray:
        """Minimal MiDaS fallback for when DA2 isn't available."""
        try:
            import torchvision.transforms as T
            from torch.hub import load as torch_load

            model = torch_load("intel-isl/MiDaS", "DPT_BEiT_L_384")
            model = model.to(self.device).eval()

            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                depth = model(input_tensor)

            depth = depth.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            return depth.astype(np.float32)
        except Exception as e:
            print(f"[DepthAnythingWrapper] MiDaS fallback failed: {e}")
            return np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.float32)

    def get_intrinsics(self) -> dict:
        return {"fx": self._fx, "fy": self._fy, "cx": self._cx, "cy": self._cy}

    @property
    def name(self) -> str:
        return f"depth_anything_{self.encoder}"
