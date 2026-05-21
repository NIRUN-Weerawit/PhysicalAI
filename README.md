# PhysicalAI — Multi-Camera Vision System for Robot Manipulation

A modular vision pipeline for robotic grasping: open-vocabulary detection (Grounded SAM 2), monocular depth estimation (Depth Anything V2), and 3D projection with camera calibration.

## Quick Start

```bash
# Intrinsic calibration (print a ChArUco board first)
python3 scripts/calibrate_intrinsics.py --mode charuco --generate-board
python3 scripts/calibrate_intrinsics.py --mode charuco --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Extrinsic calibration (ChArUco at world origin)
python3 scripts/calibrate_extrinsics.py --method charuco --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Depth scale calibration
python3 scripts/calibrate_depth_scale.py --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Live detection + 3D view
python3 scripts/live_detection.py --width 640 --height 480
```

## Pipeline

`calibrate_intrinsics.py` → `config.json` (fx, fy, cx, cy, distortion)
`calibrate_extrinsics.py` → `camera_extrinsics.json` (camera pose in world)
`calibrate_depth_scale.py` → `depth_scale` in config.json
`live_detection.py` → detection → depth → 3D → ObjectDB

## Requirements

- Python 3.11+
- PyTorch + CUDA
- Grounded SAM 2 (HF Transformers)
- Depth Anything V2
- OpenCV 4.13+ (contrib for ArUco)
