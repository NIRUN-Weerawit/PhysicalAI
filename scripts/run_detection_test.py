#!/usr/bin/env python3
"""
Quick one-shot test: detect objects, estimate depth, project to 3D.
Accepts an image file or webcam frame.

Usage:
    # Webcam:
    python3 run_detection_test.py

    # Image file:
    python3 run_detection_test.py --image /path/to/image.jpg
"""
import sys, os, json, cv2, torch, numpy as np
sys.path.insert(0, os.path.expanduser("~/PhysicalAI"))
sys.path.insert(0, os.path.expanduser("~/PhysicalAI/Grounded-SAM-2"))
from pathlib import Path
import argparse

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper

# Parse args
p = argparse.ArgumentParser()
p.add_argument("--image", type=str, default=None, help="Image file path instead of webcam")
p.add_argument("--prompt", type=str,
               default="cup. bottle. book. phone. box. pen. can. remote. mouse. keyboard. person. chair.",
               help="Text prompt (dot-separated)")
p.add_argument("--output", type=str, default="/tmp/detection_vis.jpg")
args = p.parse_args()

# 1. Load config (has your calibrated intrinsics)
cfg = load_vision_config(path="/home/ucluser/PhysicalAI/config.json", depth_source="depth_anything")
print(f"Config loaded. Intrinsics: fx={cfg.fx:.3f}, fy={cfg.fy:.3f}, cx={cfg.cx:.3f}, cy={cfg.cy:.3f}")

# 2. Init detection
print("Loading Grounded SAM 2...")
detector = GroundedSAM2Wrapper(cfg)
print("Detection model ready.")

# 3. Init depth
print("Loading Depth Anything V2...")
depth_estimator = DepthAnythingWrapper(
    encoder=cfg.depth_anything_encoder,
    checkpoint_path=cfg.depth_anything_checkpoint,
    device=cfg.device,
    grayscale=cfg.depth_anything_grayscale,
    fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy,
)
print("Depth estimator ready.")

# 4. Get input frame
if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image {args.image}")
        exit(1)
    print(f"Loaded image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        exit(1)
    print(f"Captured webcam frame: {frame.shape[1]}x{frame.shape[0]}")
    cv2.imwrite("/tmp/detection_raw.jpg", frame)

# 5. Detect
print(f"\nDetecting: '{args.prompt}'")
results = detector.detect(args.prompt, frame)
print(f"Found {len(results)} objects:")

for r in results:
    print(f"  {r['class_name']:15s}  conf={r['confidence']:.3f}  "
          f"bbox={[int(v) for v in r['bbox_xyxy']]}")

if not results:
    print("  (none)")
    sys.exit(0)

# 6. Depth estimate
print("\nEstimating depth...")
depth_map = depth_estimator.estimate(frame)
intrinsics = depth_estimator.get_intrinsics()

# 7. 3D projection
fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
h, w = frame.shape[:2]
depth_valid = np.count_nonzero(depth_map > 0.01)
print(f"Depth map: {depth_map.shape}, valid pixels: {depth_valid}/{h*w}")

print(f"\n{'Object':15s} {'Centroid':18s} {'Depth(m)':10s} {'3D Position (x=right, y=forward, z=up)':45s}")
print("-" * 90)

output_results = []
for r in results:
    u, v = r["centroid_2d"]
    y0, y1 = max(0, v-3), min(h, v+4)
    x0, x1 = max(0, u-3), min(w, u+4)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.001]
    d = float(np.median(valid)) if len(valid) > 0 else 0.0

    if d > 0.001:
        x3d = (u - cx) * d / fx
        y3d = d
        z3d = -(v - cy) * d / fy
    else:
        x3d = y3d = z3d = 0.0

    depth_str = f"{d:.3f}"
    pos_str = f"({x3d:7.3f}, {y3d:7.3f}, {z3d:7.3f})"
    print(f"  {r['class_name']:15s} ({u:4d}, {v:4d})        {depth_str:>8s}     {pos_str}")
    output_results.append({
        "class_name": r["class_name"],
        "confidence": r["confidence"],
        "bbox_xyxy": r["bbox_xyxy"],
        "centroid_2d": (u, v),
        "depth_at_centroid": round(d, 3),
        "position_3d_camera": [round(x3d, 3), round(y3d, 3), round(z3d, 3)],
    })

out_json = "/tmp/detection_results.json"
with open(out_json, "w") as f:
    json.dump(output_results, f, indent=2)
print(f"\nResults saved to {out_json}")

detector.visualize(frame, results, args.output)
print(f"Visualization saved to {args.output}")
