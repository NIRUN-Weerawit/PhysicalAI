#!/usr/bin/env python3
"""
isaacsim_calibrate_depth_scale.py
===================================

Calibrate Depth Anything's relative depth to metric meters using ground-truth
object positions from an Isaac Sim simulation.

HOW IT WORKS
------------
1.  Opens the same Franka Panda scene as isaacsim_live_detection.py.
2.  Randomises the positions of the 5 manipulatable objects.
3.  Runs Grounded SAM 2 to detect each object and get its pixel centroid (u, v).
4.  Reads Depth Anything's raw inverse depth at that centroid.
5.  Queries the object's ground-truth world position AND the camera's ground-truth
    world pose from the simulation.
6.  Transforms the object into the camera frame → extracts Z (forward) = true depth.
7.  depth_scale = true_Z / inverse_depth   (one measurement per detection).
8.  Repeats for multiple randomisations (default 10 trials, 5 objects each).
9.  Computes median ± robust std and saves to config.json.

USAGE
-----
    cd ~/isaacsim
    ./python.sh ~/PhysicalAI/scripts/isaacsim_calibrate_depth_scale.py

OUTPUT
------
    - Median depth_scale saved to ~/PhysicalAI/config.json
    - Per-trial and per-object statistics printed to console.
    - Combined display frames saved to ~/PhysicalAI/output/isaac_sim_depth_cal/

REQUIREMENTS
------------
    Same as isaacsim_live_detection.py (pycocotools, supervision installed
    with --no-deps and numpy < 2).
"""

import sys
import os
import json
import time
import numpy as np
import cv2
import torch
import warnings
from collections import defaultdict

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)
sys.path.insert(0, os.path.join(PHYSICALAI_ROOT, "Grounded-SAM-2"))

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper

warnings.filterwarnings("ignore", message=".*has been deprecated.*")

# ── Isaac Sim ──────────────────────────────────────────────────────────────
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import UsdPhysics
from scipy.spatial.transform import Rotation as R

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ═════════════════════════════════════════════════════════════════════════════
# Scene setup  (same as isaacsim_live_detection.py)
# ═════════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_simple_1.usd"
open_stage(USD_PATH)
sim = SimulationContext()
sim.reset()
sim.play()
set_camera_view(eye=[2.0, 0.0, 4.0], target=[0.0, 0.0, 2.5])

WIDTH  = 1280
HEIGHT = 720
FREQ   = 20

robot = SingleArticulation("/World/franka")
robot.initialize()

base         = SingleRigidPrim("/World/franka/panda_link0")
franka_hand  = SingleRigidPrim("/World/franka/panda_hand")

# Manipulatable objects
obj_prims = {
    "dex_cube":    SingleXFormPrim("/World/Xform_dex_cube"),
    "dex_cube_1":  SingleXFormPrim("/World/Xform_dex_cube_01"),
    "rubik":       SingleXFormPrim("/World/Xform_rubik"),
    "rubik_1":     SingleXFormPrim("/World/Xform_rubik_01"),
    "nvidia_cube": SingleXFormPrim("/World/Xform_nvidia_cube"),
}

# Cameras
body_rgb_cam  = Camera(
    prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=FREQ, resolution=(WIDTH, HEIGHT))
mid_rgb_cam   = Camera(
    prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=FREQ, resolution=(WIDTH, HEIGHT))
left_rgb_cam  = Camera(
    prim_path="/World/Realsense_left/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=FREQ, resolution=(WIDTH, HEIGHT))
right_rgb_cam = Camera(
    prim_path="/World/Realsense_right/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=FREQ, resolution=(WIDTH, HEIGHT))

PRIMARY_CAM = mid_rgb_cam
CAM_NAME    = "mid"

base.initialize()
franka_hand.initialize()
body_rgb_cam.initialize()
mid_rgb_cam.initialize()
left_rgb_cam.initialize()
right_rgb_cam.initialize()

# Joint stiffness
robot_prim = get_prim_at_path("/World/franka")
stage = robot_prim.GetStage()
for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)

# Neutral robot pose
robot.set_joint_positions(np.array([
    0.0, -0.93, 0.0, -2.43, 0.0, 2.25, 0.86, 0.0, 0.0,
]))
for _ in range(5):
    sim.step(render=True)

# ═════════════════════════════════════════════════════════════════════════════
# Helper: sample random poses for objects
# ═════════════════════════════════════════════════════════════════════════════
def random_pose():
    """Sample random position for a scene object."""
    z = 2.45
    regions = [
        ((0.0, 0.5), (-1.0, -0.3)),
        ((0.0, 0.5), (0.3, 1.0)),
        ((0.5, 1.15), (-1.0, 1.0)),
    ]
    areas = [(x1 - x0) * (y1 - y0) for (x0, x1), (y0, y1) in regions]
    p = np.array(areas) / sum(areas)
    region = regions[np.random.choice(len(regions), p=p)]
    (x0, x1), (y0, y1) = region
    x = np.random.uniform(x0, x1)
    y = np.random.uniform(y0, y1)
    return np.array([x, y, z]), [0, 0, 0, 1]

# ═════════════════════════════════════════════════════════════════════════════
# Detection pipeline
# ═════════════════════════════════════════════════════════════════════════════
cfg = load_vision_config(
    path=os.path.join(PHYSICALAI_ROOT, "config.json"),
    depth_source="depth_anything",
)
cfg.fx = 805.859
cfg.fy = 782.398
cfg.cx = 657.854
cfg.cy = 362.74

TEXT_PROMPT = "cube. box. rubik's cube. nvidia cube."

print("Loading detection models...")
detector = GroundedSAM2Wrapper(cfg)

# IMPORTANT: construct the depth wrapper but DON'T let it auto-load a scale
depth_estimator = DepthAnythingWrapper(
    encoder=cfg.depth_anything_encoder,
    checkpoint_path=cfg.depth_anything_checkpoint,
    device=cfg.device,
    grayscale=cfg.depth_anything_grayscale,
    fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy,
)
# Force lazy init so model is loaded
depth_estimator._lazy_init()
# We'll use estimate_inverse() which returns raw unscaled values
print("Models loaded.\n")

fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy

# ═════════════════════════════════════════════════════════════════════════════
# Coordinate transforms
# ═════════════════════════════════════════════════════════════════════════════

def camera_frame_to_opencv(point_cam_usd):
    """Transform a point from USD camera frame to OpenCV camera frame.

    USD camera convention (Isaac Sim):
        x = right, y = up, z = backward (looking down -Z)

    OpenCV camera convention (used in 3D projection):
        x = right, y = down, z = forward (looking down +Z)

    Returns (X, Y, Z) in OpenCV frame where Z is forward depth.
    """
    x_usd, y_usd, z_usd = point_cam_usd
    return np.array([
        x_usd,       # right → right (same)
        -y_usd,      # up → down (negate)
        -z_usd,      # backward → forward (negate)
    ])


def compute_true_depth_in_camera_frame(cam_prim, obj_prim):
    """Transform an object's world position into the OpenCV camera frame
    and return the forward (Z) component — the true metric depth.

    Returns:
        (Z_camera, X_camera, Y_camera) in meters in OpenCV camera frame.
    """
    # Get camera world pose: (pos, quat [w,x,y,z])
    cam_pos, cam_quat = cam_prim.get_world_pose()

    # Get object world position
    obj_pos, _ = obj_prim.get_world_pose()

    # Camera → world rotation matrix
    # Isaac Sim returns quat as [w, x, y, z]; scipy expects [x, y, z, w]
    rot = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
    R_cw = rot.as_matrix()  # camera → world

    # World → camera transform
    # point_cam = R_cw^T * (point_world - pos_cam)
    point_world = np.array(obj_pos)
    point_cam_usd = R_cw.T @ (point_world - np.array(cam_pos))

    # Convert from USD camera frame to OpenCV camera frame
    point_cv = camera_frame_to_opencv(point_cam_usd)
    return float(point_cv[2]), float(point_cv[0]), float(point_cv[1])


# ═════════════════════════════════════════════════════════════════════════════
# Calibration loop
# ═════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = os.path.join(PHYSICALAI_ROOT, "output", "isaac_sim_depth_cal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_TRIALS = 10       # number of random object configurations
SAVE_FRAMES = True    # save annotated frames for inspection

all_measurements = []         # list of (trial, obj_name, true_depth, inv_raw, computed_scale)
per_trial = defaultdict(list)  # trial_id → [scale, ...]

print("=" * 60)
print("DEPTH SCALE CALIBRATION via Isaac Sim Ground Truth")
print("=" * 60)
print(f"Camera: {CAM_NAME} ({WIDTH}x{HEIGHT})")
print(f"Objects: {list(obj_prims.keys())}")
print(f"Trials: {NUM_TRIALS}")
print(f"Prompt: {TEXT_PROMPT}")
print()

for trial in range(NUM_TRIALS):
    print(f"--- Trial {trial + 1}/{NUM_TRIALS} ---")

    # 1. Randomise object positions
    for name, prim in obj_prims.items():
        pos, q = random_pose()
        prim.set_world_pose(position=pos, orientation=q)
    for _ in range(10):
        sim.step(render=True)

    # 2. Capture frame
    rgba = PRIMARY_CAM.get_rgba()
    if len(rgba) == 0:
        print("  [WARN] No camera data, skipping trial")
        continue
    color_image = rgba.copy().reshape((HEIGHT, WIDTH, 4))
    frame_bgr = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    # 3. Run Depth Anything (raw inverse)
    depth_inv = depth_estimator.estimate_inverse(frame_bgr)

    # 4. Run Grounded SAM 2 detection
    detections = detector.detect(TEXT_PROMPT, frame_bgr)

    if len(detections) == 0:
        print("  No detections, skipping trial")
        continue

    # 5. For each detected object, find the best-matching ground-truth prim
    display = frame_bgr.copy()
    trial_measurements = []

    for r in detections:
        u, v = r["centroid_2d"]
        class_name = r["class_name"]

        # Map detection class to ground-truth object prim
        # Try matching: "cube" → dex_cube or nvidia_cube or rubik
        candidate_prims = []
        for prim_name, prim in obj_prims.items():
            # Simple heuristic: match detection class to prim name
            if class_name.lower() in prim_name.lower() or prim_name.lower() in class_name.lower():
                candidate_prims.append((prim_name, prim))
            # "cube" matches "dex_cube", "nvidia_cube" etc.
            if "cube" in class_name.lower() and "cube" in prim_name.lower():
                if (prim_name, prim) not in candidate_prims:
                    candidate_prims.append((prim_name, prim))
            if "rubik" in class_name.lower() and "rubik" in prim_name.lower():
                if (prim_name, prim) not in candidate_prims:
                    candidate_prims.append((prim_name, prim))
            if "nvidia" in class_name.lower() or "nvidia" in prim_name.lower():
                if (prim_name, prim) not in candidate_prims:
                    candidate_prims.append((prim_name, prim))

        if not candidate_prims:
            # Fallback: find closest prim by position
            # (this works when detections match well but class names differ)
            # Get camera-to-world transform
            cam_pos, cam_quat = PRIMARY_CAM.get_world_pose()
            rot = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
            R_cw = rot.as_matrix()
            cam_pos_np = np.array(cam_pos)

            # Compute the ray direction for this centroid
            # In OpenCV: X = (u-cx)*Z/fx, Y = -(v-cy)... no, we need to project
            # Just use all prims and find closest by 2D projection
            best_dist = float('inf')
            best_prim = None
            best_prim_name = None
            for pname, pprim in obj_prims.items():
                ppos, _ = pprim.get_world_pose()
                ppos_np = np.array(ppos)
                # Compute in camera frame (OpenCV)
                point_cam_usd = R_cw.T @ (ppos_np - cam_pos_np)
                point_cv = camera_frame_to_opencv(point_cam_usd)
                Z_cv = float(point_cv[2])
                if Z_cv < 0.05:
                    continue
                # Project back to pixels
                up = int(fx * point_cv[0] / Z_cv + cx)
                vp = int(-fy * point_cv[1] / Z_cv + cy)  # y is inverted
                dist = np.hypot(up - u, vp - v)
                if dist < best_dist:
                    best_dist = dist
                    best_prim = pprim
                    best_prim_name = pname

            if best_prim is None or best_dist > 100:
                print(f"  [SKIP] {class_name} @ ({u},{v}) — no matching prim found")
                continue
            candidate_prims = [(best_prim_name, best_prim)]

        # Use the first matching prim
        prim_name, prim = candidate_prims[0]

        # Get raw inverse depth at centroid
        patch = depth_inv[max(0,v-3):min(h,v+4), max(0,u-3):min(w,u+4)]
        valid = patch[patch > 0.001]
        if len(valid) == 0:
            print(f"  [SKIP] {class_name}/{prim_name} — invalid depth at ({u},{v})")
            continue
        inv_raw = float(np.median(valid))

        # Get ground-truth depth from sim
        true_z, _, _ = compute_true_depth_in_camera_frame(PRIMARY_CAM, prim)
        if true_z < 0.05:
            print(f"  [SKIP] {class_name}/{prim_name} — Z={true_z:.3f}m too close")
            continue

        # Compute depth scale
        scale = true_z / inv_raw
        trial_measurements.append(scale)

        all_measurements.append({
            "trial": trial,
            "object": prim_name,
            "class": class_name,
            "true_z": true_z,
            "inv_raw": inv_raw,
            "scale": scale,
        })
        per_trial[trial].append(scale)

        # Draw on display
        x1, y1, x2, y2 = [int(x) for x in r["bbox_xyxy"]]
        label = f"{prim_name}: Z={true_z:.3f}m  inv={inv_raw:.4f}  s={scale:.4f}"
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display, (u, v), 4, (0, 255, 255), -1)
        cv2.putText(display, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        print(f"  {prim_name:12s}  Z={true_z:.3f}m  inv_raw={inv_raw:.4f}  → scale={scale:.4f}")

    if SAVE_FRAMES and trial_measurements:
        out_path = os.path.join(OUTPUT_DIR, f"trial_{trial+1:02d}.jpg")
        cv2.imwrite(out_path, display)
        print(f"  Saved: {out_path}")

    if not trial_measurements:
        print("  No valid measurements this trial")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# Compute final result
# ═════════════════════════════════════════════════════════════════════════════
if len(all_measurements) < 3:
    print(f"\nERROR: Only {len(all_measurements)} measurements collected. Need at least 3.")
    print("Check:")
    print("  - Are objects visible to the mid camera?")
    print("  - Is the detection prompt matching the objects?")
    print("  - Try reducing the scene range or adjusting TEXT_PROMPT.")
    simulation_app.close()
    sys.exit(1)

scales = np.array([m["scale"] for m in all_measurements])

# Remove obvious outliers (beyond 3σ from median)
median_s = np.median(scales)
mad = np.median(np.abs(scales - median_s))  # median absolute deviation
inlier_mask = np.abs(scales - median_s) < 3 * max(mad, 0.01)
scales_clean = scales[inlier_mask]

if len(scales_clean) < 3:
    scales_clean = scales  # fallback to all
    print("[WARN] Outlier rejection removed too many points, using all measurements.")

final_scale = float(np.median(scales_clean))
robust_std = 1.4826 * float(np.median(np.abs(scales_clean - np.median(scales_clean))))
cv_pct = robust_std / final_scale * 100

# Also compute from trial medians (per-trial, then median across trials)
trial_medians = []
for trial_id, trial_scales in per_trial.items():
    if len(trial_scales) >= 2:
        trial_medians.append(np.median(trial_scales))
final_trial_scale = float(np.median(trial_medians)) if trial_medians else final_scale

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total measurements: {len(all_measurements)}")
print(f"Trials: {len(per_trial)}")
print(f"After outlier rejection: {len(scales_clean)} / {len(scales)}")
print()
print(f"Per-measurement median:    {final_scale:.4f}")
print(f"Robust std (MAD):          {robust_std:.4f}  ({cv_pct:.1f}% CV)")
print(f"Per-trial median:          {final_trial_scale:.4f}")
print()
print("All measurements:")
for i, m in enumerate(all_measurements):
    flag = " " if inlier_mask[i] else " ✗"
    print(f"  [{i+1}] trial={m['trial']:2d}  obj={m['object']:12s}  "
          f"Z={m['true_z']:.3f}m  inv={m['inv_raw']:.4f}  "
          f"scale={m['scale']:.4f}{flag}")

print()
print(f"Recommended depth_scale: {final_scale:.4f}")
print(f"  (or from trial medians: {final_trial_scale:.4f})")

# Save to config.json
config_path = os.path.expanduser("~/PhysicalAI/config.json")
with open(config_path) as f:
    config = json.load(f)
config["depth_scale"] = round(final_scale, 4)
config["depth_scale_robust_std"] = round(robust_std, 4)
config["depth_scale_num_measurements"] = len(scales_clean)
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"\nSaved depth_scale={final_scale:.4f} to {config_path}")
print()

simulation_app.close()
