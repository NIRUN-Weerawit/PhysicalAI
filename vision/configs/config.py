"""
Configuration system for the PhysicalAI vision pipeline.
Loads YAML config or environment variables to select depth source, model paths, etc.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Literal, Optional


DepthSource = Literal["rgbd", "depth_anything", "midas"]


@dataclass
class VisionConfig:
    """Immutable vision pipeline configuration."""

    # ---- Detection (Grounded SAM 2) ----
    sam2_checkpoint: str = ""
    sam2_model_config: str = ""
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda"
    multimask_output: bool = False

    # ---- Depth Source ----
    depth_source: DepthSource = "depth_anything"
    # RGB-D camera params
    rgbd_camera_id: int = 0
    rgbd_width: int = 640
    rgbd_height: int = 480
    rgbd_fps: int = 30
    rgbd_align_depth: bool = True
    # Simulated RGB-D (no hardware)
    rgbd_use_simulated: bool = False
    rgbd_sim_fx: float = 525.0
    rgbd_sim_fy: float = 525.0
    rgbd_sim_cx: float = 320.0
    rgbd_sim_cy: float = 240.0
    rgbd_sim_depth_range: tuple = (0.2, 10.0)

    # ---- Depth Anything ----
    depth_anything_encoder: str = "vitl"  # vits, vitb, vitl
    depth_anything_checkpoint: str = ""
    depth_anything_grayscale: bool = True

    # ---- MiDaS ----
    midas_model_type: str = "DPT_BEiT_L_384"
    midas_grayscale: bool = True

    # ---- Output ----
    output_dir: str = "./output"
    save_visualizations: bool = True
    dump_json_results: bool = True

    # ---- Camera intrinsics (for 3D projection) ----
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        return cls(
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_CONFIG = VisionConfig(
    sam2_checkpoint=os.path.expanduser(
        "~/PhysicalAI/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    ),
    sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
)


def load_vision_config(
    path: Optional[str] = None,
    depth_source: Optional[DepthSource] = None,
) -> VisionConfig:
    """Load config from JSON file, overriding depth source if provided."""
    config = DEFAULT_CONFIG
    if path and os.path.exists(path):
        with open(path) as f:
            config = VisionConfig.from_dict(json.load(f))
    if depth_source:
        config.depth_source = depth_source
    return config
