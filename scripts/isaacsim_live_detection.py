#!/usr/bin/env python3
"""
isaacsim_live_detection.py — Grounded SAM 2 + Depth Anything in Isaac Sim
=============================================================================

Runs the PhysicalAI detection pipeline (Grounded SAM 2 → Depth Anything V2 → 3D
projection → ObjectDB) inside an NVIDIA Isaac Sim simulation using the same
Franka Panda scene from rdt_panda_single.py.

Camera images are captured from Isaac Sim's synthetic RGB cameras (not a
webcam).  Detection outputs are overlaid on the simulation view, displayed in
OpenCV windows alongside top-down / side 3D projections, and logged to an
ObjectDB.

USAGE
-----
Must be run inside Isaac Sim's Python environment::

    cd ~/isaacsim
    ./python.sh ~/PhysicalAI/scripts/isaacsim_live_detection.py

Isaac Sim uses Python 3.10 + its own bundled numpy/opencv.  The PhysicalAI
dependencies (pycocotools, supervision) need to be pip-installed WITHOUT
upgrading numpy.  If you broke numpy already::

    ./python.sh -m pip install 'numpy<2' --force-reinstall
    ./python.sh -m pip install pycocotools supervision --no-deps

Then re-run.

KEY DIFFERENCES vs live_detection.py (real webcam)
---------------------------------------------------
- Camera source: Isaac Sim Camera sensor → RGBA tensor, not OpenCV VideoCapture.
- No real-time FPS constraint: simulation is stepped manually.
- Depth comes from Depth Anything V2 (same monocular estimator), NOT sim depth.
- Robot is present in the scene but NOT controlled — it stays at the initial
  pose so you can detect objects around it.
- Scene objects (cubes, rubik's, etc.) are randomised each run.

KEY CONFIGURATION
-----------------
  TEXT_PROMPT     : comma/period-separated classes to detect
  DETECT_INTERVAL : run Grounding DINO every N frames (5 = ~every 5th sim step)
  camera          : primary detection camera (mid_rgb_cam by default)
  camera_freq     : capture frequency in simulation Hz
  resolution      : 720 x 480 (must match calibration in config.json)

  Scene cameras:
    - mid_rgb_cam   (/World/Realsense_mid/)   — overhead view (default)
    - body_rgb_cam  (/World/franka/panda_hand/) — wrist-mounted
    - left_rgb_cam  (/World/Realsense_left/)
    - right_rgb_cam (/World/Realsense_right/)
"""

import sys
import os
import json
import time
import numpy as np
import cv2
import torch
import warnings

# ── PhysicalAI pipeline modules ────────────────────────────────────────────
PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)
sys.path.insert(0, os.path.join(PHYSICALAI_ROOT, "Grounded-SAM-2"))

# supervision's annotation tools need numpy < 2 — try our best
try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False
    print("[WARN] supervision not available — install via `./python.sh -m pip install supervision --no-deps`")

from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper
from vision.world_model.object_db import ObjectDB, ObjectRecord

warnings.filterwarnings("ignore", message=".*has been deprecated.*")

# ═══════════════════════════════════════════════════════════════════════════
# Simulation setup
# ═══════════════════════════════════════════════════════════════════════════
from omni.isaac.kit import SimulationApp

HEADLESS = False
simulation_app = SimulationApp({"headless": HEADLESS})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.extensions import enable_extension
from pxr import UsdPhysics

# simulation_app.set_setting("/app/window/drawMouse", True)
# enable_extension("omni.kit.livestream.webrtc")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
warnings.filterwarnings(
    "ignore",
    message=".*has been deprecated.*",
)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Scene ──────────────────────────────────────────────────────────────────
USD_PATH = "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/piper_description/franka_simple_1.usd"
open_stage(USD_PATH)
sim = SimulationContext()
sim.reset()
sim.play()

set_camera_view(eye=[2.0, 0.0, 4.0], target=[0.0, 0.0, 2.5])

# ── Resolution & frequency ────────────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
FREQ   = 20  # capture Hz

# ── Robot (loaded but not controlled) ──────────────────────────────────────
robot = SingleArticulation("/World/franka")
robot.initialize()
print(f"Robot DOFs: {robot.dof_names}")

base         = SingleRigidPrim("/World/franka/panda_link0")
table        = SingleXFormPrim("/World/Xform_table")
franka_hand  = SingleRigidPrim("/World/franka/panda_hand")

# Manipulatable objects
dex_cube_xform    = SingleXFormPrim("/World/Xform_dex_cube")
dex_cube_1_xform  = SingleXFormPrim("/World/Xform_dex_cube_01")
rubik_xform       = SingleXFormPrim("/World/Xform_rubik")
rubik_1_xform     = SingleXFormPrim("/World/Xform_rubik_01")
nvidia_cube_xform = SingleXFormPrim("/World/Xform_nvidia_cube")

objects = {
    "dex_cube":    dex_cube_xform,
    "dex_cube_1":  dex_cube_1_xform,
    "rubik":       rubik_xform,
    "rubik_1":     rubik_1_xform,
    "nvidia_cube": nvidia_cube_xform,
}

# ── Cameras ────────────────────────────────────────────────────────────────
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

# Primary detection camera — change here to switch views
PRIMARY_CAM = mid_rgb_cam
CAM_NAME    = "mid"

base.initialize()
franka_hand.initialize()
# body_rgb_cam.initialize()
mid_rgb_cam.initialize()
# left_rgb_cam.initialize()
# right_rgb_cam.initialize()

# ── Joint stiffness ────────────────────────────────────────────────────────
robot_prim = get_prim_at_path("/World/franka")
stage = robot_prim.GetStage()
for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
        continue
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(1e4)
        drive.GetDampingAttr().Set(1e2)

# ── Neutral pose ───────────────────────────────────────────────────────────
robot.set_joint_positions(np.array([
    0.0, -0.93, 0.0, -2.43, 0.0, 2.25, 0.86,
    0.0, 0.0,
]))
sim.step(render=True)
print("Robot initial pose set")

# ── Randomise object positions ─────────────────────────────────────────────
def random_pose():
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

for obj_name, obj_prim in objects.items():
    pos, q = random_pose()
    obj_prim.set_world_pose(position=pos, orientation=q)
print("Object positions randomised")

for _ in range(10):
    sim.step(render=True)

# ═══════════════════════════════════════════════════════════════════════════
# Detection pipeline setup
# ═══════════════════════════════════════════════════════════════════════════
cfg = load_vision_config(
    path=os.path.join(PHYSICALAI_ROOT, "config.json"),
    depth_source="depth_anything",
)
# Calibrated intrinsics — update after ChArUco calibration inside sim
cfg.fx = 805.859
cfg.fy = 782.398
cfg.cx = 657.854
cfg.cy = 362.74

TEXT_PROMPT = "black cube. purple box. rubik's cube. nvidia cube. robotic arm. gripper. table. shelf. cart."
DETECT_INTERVAL = 5

print("Loading detection models...")
detector = GroundedSAM2Wrapper(cfg)
depth_estimator = DepthAnythingWrapper(
    encoder=cfg.depth_anything_encoder,
    checkpoint_path=cfg.depth_anything_checkpoint,
    device=cfg.device,
    grayscale=cfg.depth_anything_grayscale,
    fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy,
)
print("Models loaded.")

# ── ObjectDB ───────────────────────────────────────────────────────────────
db = ObjectDB(stale_timeout=3.0)
seen_this_cycle = set()
fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy

# ── Display ────────────────────────────────────────────────────────────────
VIEW_RANGE = 3.0
# 3D view panel: half width for each panel, total width matches camera frame
VIEW_SIZE_H = min(WIDTH // 2, 640)
PPM         = VIEW_SIZE_H / VIEW_RANGE

frame_count = 0
detections  = []
depth_map   = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

FPS_WINDOW = 1.0
fps_start  = time.perf_counter()
fps_count  = 0
fps_display = 0.0

# ── Output directory for debug frames ──
OUTPUT_DIR = os.path.join(PHYSICALAI_ROOT, "output", f"isaac_sim_{CAM_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_EVERY = 10  # save a display frame every N sim steps
LOG_EVERY = 30   # print console status every N sim steps

# ── Run for this many steps (Ctrl+C to quit early) ──
MAX_STEPS = 5000

print("\n=== ISAAC SIM LIVE DETECTION ===")
print(f"Camera: {CAM_NAME}  Resolution: {WIDTH}x{HEIGHT}")
print(f"Prompt: {TEXT_PROMPT}")
print(f"Output frames: {OUTPUT_DIR}/frame_*.jpg (every {SAVE_EVERY} steps)")
print(f"Max steps: {MAX_STEPS}  (Ctrl+C to exit early)\n")

# ═══════════════════════════════════════════════════════════════════════════
# Main simulation + detection loop
# ═══════════════════════════════════════════════════════════════════════════
try:
    while simulation_app.is_running() and frame_count < MAX_STEPS:
        sim.step(render=True)
        frame_count += 1
        is_detect = (frame_count % DETECT_INTERVAL == 0)
        now = time.monotonic()

        # ── Capture frame from Isaac Sim camera ──
        rgba = PRIMARY_CAM.get_rgba()
        if len(rgba) == 0:
            continue
        # Isaac Sim returns flat RGBA, reshape to HxWx4
        color_image = rgba.copy().reshape((HEIGHT, WIDTH, 4))
        frame_bgr = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        # ── Depth estimation ──
        depth_map = depth_estimator.estimate(frame_bgr)

        # ── Detection ──
        if is_detect:
            detections = detector.detect(TEXT_PROMPT, frame_bgr)
            seen_this_cycle = set()

            for r in detections:
                u, v = r["centroid_2d"]
                try:
                    patch = depth_map[max(0,v-3):min(h,v+4), max(0,u-3):min(w,u+4)]
                    valid = patch[patch > 0.001]
                    d = float(np.median(valid)) if len(valid) > 0 else 0.0
                except Exception:
                    d = 0.0

                if d > 0.001:
                    x_w = (u - cx) * d / fx
                    y_w = d
                    z_w = -(v - cy) * d / fy
                    r["_px"], r["_py"], r["_pz"] = x_w, y_w, z_w
                else:
                    r["_px"] = r["_py"] = r["_pz"] = 0.0

                id_key = f"{r['class_name']}_{round(r['_px'],2)}_{round(r['_py'],2)}_{round(r['_pz'],2)}"
                seen_this_cycle.add(id_key)

                obs = {
                    "camera": f"sim_{CAM_NAME}",
                    "confidence": r["confidence"],
                    "timestamp": now,
                    "centroid_2d": (u, v),
                    "depth": d,
                    "bbox_xyxy": r["bbox_xyxy"],
                    "frame": frame_count,
                }

                existing = db.get(id_key)
                if existing:
                    db.update(id_key, (r["_px"], r["_py"], r["_pz"]),
                              timestamp=now, observation=obs,
                              confidence=r["confidence"])
                else:
                    new_obj = ObjectRecord(
                        object_id=id_key,
                        class_name=r["class_name"],
                        position_world=(r["_px"], r["_py"], r["_pz"]),
                        confidence=r["confidence"],
                        timestamp=now,
                        first_seen=now,
                        observations=[obs],
                        metadata={"source": f"isaac_sim_{CAM_NAME}"},
                    )
                    db.add(new_obj)

            for obj in list(db._objects.values()):
                if obj.object_id not in seen_this_cycle:
                    db.remove(obj.object_id)

        # ═══════════════════════════════════════════════════════════════════
        # OpenCV display
        # ═══════════════════════════════════════════════════════════════════
        display = frame_bgr.copy()

        for r in detections:
            u, v = r["centroid_2d"]
            x1, y1, x2, y2 = [int(x) for x in r["bbox_xyxy"]]
            x_w, y_w, z_w = r.get("_px", 0.0), r.get("_py", 0.0), r.get("_pz", 0.0)

            if x_w != 0.0 or y_w != 0.0:
                label = f"{r['class_name']} (x={x_w:.2f}, y={y_w:.2f}, z={z_w:.2f})m"
            else:
                label = f"{r['class_name']} (no depth)"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # ── 3D Dual-view panel ──
        v3d = np.zeros((VIEW_SIZE_H, VIEW_SIZE_H * 2, 3), dtype=np.uint8)
        cv2.rectangle(v3d, (0, 0), (VIEW_SIZE_H*2-1, VIEW_SIZE_H-1), (25, 25, 35), -1)
        cv2.putText(v3d, f"TOP-DOWN (x->  y^)  Camera: {CAM_NAME}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)
        cv2.putText(v3d, f"SIDE (y->  z^)  R={VIEW_RANGE}m", (VIEW_SIZE_H + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)

        for panel_x, label_y, coord_fn in [
            (0, "y", lambda p: (p["_px"], p["_py"])),
            (VIEW_SIZE_H, "z", lambda p: (p["_py"], p["_pz"])),
        ]:
            for dist_m in np.arange(0.5, VIEW_RANGE + 0.5, 0.5):
                px = int(dist_m * PPM)
                color = (50, 50, 60)
                if label_y == "y":
                    cv2.circle(v3d, (panel_x + VIEW_SIZE_H // 2, VIEW_SIZE_H - 1), px, color, 1)
                else:
                    cv2.line(v3d, (panel_x, VIEW_SIZE_H - 1 - px),
                             (panel_x + VIEW_SIZE_H - 1, VIEW_SIZE_H - 1 - px), color, 1)
                cv2.putText(v3d, f"{dist_m:.1f}", (panel_x + 5, VIEW_SIZE_H - px - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 100), 1)

            cam_x = panel_x + VIEW_SIZE_H // 2
            cam_y = VIEW_SIZE_H - 1
            cv2.circle(v3d, (cam_x, cam_y), 4, (0, 255, 255), -1)
            cv2.putText(v3d, "CAM", (cam_x - 15, cam_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

            for r in detections:
                xv, yv = coord_fn(r)
                if abs(xv) > VIEW_RANGE or yv > VIEW_RANGE or yv < 0:
                    continue
                px = int(cam_x + xv * PPM)
                py = int(cam_y - yv * PPM)
                px = np.clip(px, panel_x + 5, panel_x + VIEW_SIZE_H - 5)
                py = np.clip(py, 10, VIEW_SIZE_H - 10)
                cv2.circle(v3d, (px, py), 6, (0, 200, 255), -1)
                cv2.putText(v3d, r["class_name"], (px + 8, py + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # ── ObjectDB overlay ──
        db_objects = db.get_all()
        y_off = 45
        cv2.putText(display, f"ObjectDB: {len(db_objects)} objects", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y_off += 18
        for obj in sorted(db_objects, key=lambda o: o.position_world[1]):
            x, y, z = obj.position_world
            cv2.putText(display,
                        f"  {obj.class_name:10s} x={x:.2f} y={y:.2f} z={z:.2f}  "
                        f"c={obj.confidence:.2f}",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)
            y_off += 16
            if y_off > h - 80:
                break

        # ── FPS ──
        fps_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= FPS_WINDOW:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_start = time.perf_counter()

        tag = " [DETECT]" if is_detect else ""
        cv2.putText(display, f"FPS: {fps_display:.1f} | Objects: {len(detections)}{tag}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Isaac Sim | Camera: {CAM_NAME} | Step: {frame_count}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── Save or log — no cv2.imshow (Isaac Sim's OpenCV has no GUI) ──
        if frame_count % SAVE_EVERY == 0:
            combined = np.vstack([display, v3d])
            out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(out_path, combined)

        if frame_count % LOG_EVERY == 0:
            n_objects = len(detections)
            n_db = len(db.get_all())
            print(f"[step {frame_count}] FPS={fps_display:.1f}  "
                  f"detected={n_objects}  db_objects={n_db}  "
                  f"prompt={TEXT_PROMPT[:30]}...")

except KeyboardInterrupt:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Final report
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Simulation complete ===")
print(f"Total steps: {frame_count}")
print(f"Output frames saved to: {OUTPUT_DIR}/")

snap = db.snapshot()
if snap:
    print(f"\nFinal ObjectDB: {len(snap)} objects tracked:")
    for oid, info in snap.items():
        print(f"  {info['class_name']:12s}  pos={info['position_world']}  "
              f"conf={info['confidence']:.2f}  seen={info['observation_count']}x  "
              f"age={info['age_seconds']:.1f}s")
else:
    print("  (empty)")
print(f"Total: {len(snap)} objects")

simulation_app.close()
