# PhysicalAI — Multi-Camera Vision System for Robot Manipulation

A modular vision pipeline for robotic grasping: open-vocabulary detection (Grounded SAM 2), monocular depth estimation (Depth Anything V2), and 3D projection with camera calibration. Runs on a real webcam or inside an NVIDIA Isaac Sim simulation.

## Quick Start (Real Camera)

```bash
# Intrinsic calibration (print a ChArUco board first)
python3 scripts/calibrate_intrinsics.py --mode charuco --generate-board
python3 scripts/calibrate_intrinsics.py --mode charuco --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Extrinsic calibration (ChArUco at world origin)
python3 scripts/calibrate_extrinsics.py --method charuco --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Depth scale calibration (physical ChArUco board)
python3 scripts/calibrate_depth_scale.py --camera-id 0 \
    --squares-x 7 --squares-y 5 --square-mm <measured> --marker-mm 25

# Live detection + 3D view
python3 scripts/live_detection.py --width 640 --height 480
```

## Quick Start (Isaac Sim)

In your Isaac Sim Python environment (`./python.sh`), first install missing deps
without breaking numpy:

```bash
# numpy must stay at 1.x (Isaac Sim's bundled OpenCV was compiled against it)
./python.sh -m pip install 'numpy<2,>=1.21.2' --force-reinstall

# pycocotools + supervision without pulling numpy 2.x
./python.sh -m pip install pycocotools supervision --no-deps

# SAM 2 backbone dependency
./python.sh -m pip install iopath
```

Then:

```bash
# Live detection inside the simulation
./python.sh ~/PhysicalAI/scripts/isaacsim_live_detection.py

# Depth scale calibration using ground-truth object positions
./python.sh ~/PhysicalAI/scripts/isaacsim_calibrate_depth_scale.py
```

**Note:** Isaac Sim's bundled OpenCV has `GUI: NONE` — `cv2.imshow()` won't work.
The sim scripts save annotated frames to `~/PhysicalAI/output/` instead.

### Isaac Sim OpenCV caveat

The Isaac Sim Python environment ships `opencv-python-headless 4.9.0` (no GUI).
`./python.sh` uses this headless version. Your regular `python3` has the GTK
version. This is intentional — the Isaac Sim scripts save debug frames to disk
instead of opening windows.

## Pipeline

`calibrate_intrinsics.py` → `config.json` (fx, fy, cx, cy, distortion)
`calibrate_extrinsics.py` → `camera_extrinsics.json` (camera pose in world)
`calibrate_depth_scale.py` → `depth_scale` in config.json
`live_detection.py` → detection → depth → 3D → ObjectDB

## Model Downloads

This repo does **not** include model checkpoints (gitignored). After cloning,
download them manually:

```bash
# ── Grounded SAM 2 ──
git clone https://github.com/facebookresearch/sam2.git Grounded-SAM-2
cd Grounded-SAM-2
pip install -e .
cd checkpoints
./download_ckpts.sh      # ~900MB (sam2.1_hiera_large.pt)
cd ../..

# ── Depth Anything V2 ──
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2
mkdir -p depth_anything_v2/checkpoints
# vits encoder (small, ~92MB — matches default config)
wget -P depth_anything_v2/checkpoints/ \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# Or vitl for higher accuracy (~1.3GB):
# wget -P depth_anything_v2/checkpoints/ \
#   https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

# ── Verify ──
ls -lh Grounded-SAM-2/checkpoints/*.pt
ls -lh depth_anything_v2/checkpoints/*.pth
```

### If running inside Isaac Sim's Docker / ./python.sh

Replace `pip install -e .` with:

```bash
cd Grounded-SAM-2
python3 -m pip install -e .
# or if only ./python.sh is the available Python:
./path/to/isaacsim/python.sh -m pip install -e .
```

### Path fixes for Docker / non-ucluser environments

`config.json` has hardcoded paths like `/home/ucluser/PhysicalAI/...`. Adjust
them to match your container:

```bash
sed -i 's|/home/ucluser/PhysicalAI/|/your/actual/path/|g' config.json
```

Also fix the `PHYSICALAI_ROOT` variable in the Isaac Sim scripts:

```bash
sed -i 's|os.path.expanduser("~/PhysicalAI")|"/your/actual/path/PhysicalAI"|' \
  scripts/isaacsim_live_detection.py scripts/isaacsim_calibrate_depth_scale.py
```

## Requirements

- Python 3.11+
- PyTorch + CUDA
- Grounded SAM 2 (HF Transformers) — see model downloads above
- Depth Anything V2 — see model downloads above
- OpenCV 4.13+ (contrib for ArUco)
- Isaac Sim 2025+ for sim scripts
