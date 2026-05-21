#!/usr/bin/env python3
"""
Live detection + depth + 3D projection with dual-view 3D display + ObjectDB.

The ObjectDB tracks all objects currently visible in the frame. Objects that
leave the frame are evicted after 3 seconds.

Each object stores: class_name, 3D position (x,y,z), confidence,
depth, centroid_2d, bounding box, timestamp.

Coordinates: x=right, y=forward, z=up (meters).
"""
import sys, os, json, cv2, torch, numpy as np, time, json
sys.path.insert(0, os.path.expanduser("~/PhysicalAI"))
sys.path.insert(0, os.path.expanduser("~/PhysicalAI/Grounded-SAM-2"))
from vision.configs.config import load_vision_config
from vision.detection.grounded_sam2_wrapper import GroundedSAM2Wrapper
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper
from vision.world_model.object_db import ObjectDB, ObjectRecord

import argparse
_parser = argparse.ArgumentParser(description="Live detection + depth + 3D view")
_parser.add_argument("--width", type=int, default=640, help="Camera capture width")
_parser.add_argument("--height", type=int, default=480, help="Camera capture height")
_parser.add_argument("--camera-id", type=int, default=0)
_parser.add_argument("--depth-every", type=int, default=1,
                     help="Run depth every N frames. 1=every frame (default).")
_cam_args = _parser.parse_args()

cfg = load_vision_config(path="/home/ucluser/PhysicalAI/config.json", depth_source="depth_anything")
TEXT_PROMPT = "cup. bottle. book. phone. box. pen. can. remote. mouse. keyboard. person. chair."
DETECT_INTERVAL = 5

# ── ObjectDB ──
# Short stale timeout: objects not seen for 3 seconds are removed
db = ObjectDB(stale_timeout=3.0)

# For tracking which objects were seen in the latest detection cycle
# (used to avoid evicting objects that are still there)
seen_this_cycle = set()

print("Loading models...")
detector = GroundedSAM2Wrapper(cfg)
depth_estimator = DepthAnythingWrapper(
    encoder=cfg.depth_anything_encoder,
    checkpoint_path=cfg.depth_anything_checkpoint,
    device=cfg.device,
    grayscale=cfg.depth_anything_grayscale,
    fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy,
)
print("Models loaded. Opening camera...")

cap = cv2.VideoCapture(_cam_args.camera_id)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {_cam_args.camera_id}")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, _cam_args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _cam_args.height)
print(f"Camera: id={_cam_args.camera_id}  resolution={_cam_args.width}x{_cam_args.height}")

fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
frame_count = 0
detections = []
depth_map = np.zeros((_cam_args.height, _cam_args.width), dtype=np.float32)
FPS_WINDOW = 1.0
fps_start = time.perf_counter()
fps_count = 0
fps_display = 0.0

VIEW_SIZE = 500
VIEW_RANGE = 3.0
PPM = VIEW_SIZE / VIEW_RANGE

print("\n=== LIVE DETECTION ===")
print("ESC or 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_count += 1
    is_detect = (frame_count % DETECT_INTERVAL == 0)
    do_depth = (frame_count % _cam_args.depth_every == 0)
    now = time.monotonic()

    if do_depth:
        depth_map = depth_estimator.estimate(frame)

    if is_detect:
        detections = detector.detect(TEXT_PROMPT, frame)
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
                d = 0.0

            # ── Update ObjectDB ──
            # Use a consistent ID per object: class_name + position hash
            # (simple approach: ID = class_name + rounded coordinates)
            id_key = f"{r['class_name']}_{round(r['_px'], 2)}_{round(r['_py'], 2)}_{round(r['_pz'], 2)}"
            seen_this_cycle.add(id_key)

            obs = {
                "camera": f"cam_{_cam_args.camera_id}",
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
                    metadata={"source": "live_detection"},
                )
                db.add(new_obj)

        # Evict objects not seen this cycle
        for obj in list(db._objects.values()):
            if obj.object_id not in seen_this_cycle:
                db.remove(obj.object_id)

    # Build display
    display = frame.copy()
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

    # ── 3D DUAL-VIEW ──
    v3d = np.zeros((VIEW_SIZE, VIEW_SIZE * 2, 3), dtype=np.uint8)
    cv2.rectangle(v3d, (0, 0), (VIEW_SIZE*2-1, VIEW_SIZE-1), (25, 25, 35), -1)
    cv2.putText(v3d, "TOP-DOWN (x→  y↑)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)
    cv2.putText(v3d, f"SIDE (y→  z↑)  R={VIEW_RANGE}m", (VIEW_SIZE + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)

    for panel_x, label_y, coord_fn in [
        (0, "y", lambda p: (p["_px"], p["_py"])),
        (VIEW_SIZE, "z", lambda p: (p["_py"], p["_pz"])),
    ]:
        for dist_m in np.arange(0.5, VIEW_RANGE + 0.5, 0.5):
            px = int(dist_m * PPM)
            color = (50, 50, 60)
            if label_y == "y":
                cv2.circle(v3d, (panel_x + VIEW_SIZE//2, VIEW_SIZE-1), px, color, 1)
            else:
                cv2.line(v3d, (panel_x, VIEW_SIZE - 1 - px),
                         (panel_x + VIEW_SIZE - 1, VIEW_SIZE - 1 - px), color, 1)
            cv2.putText(v3d, f"{dist_m:.1f}", (panel_x + 5, VIEW_SIZE - px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 100), 1)

        cam_x = panel_x + VIEW_SIZE // 2
        cam_y = VIEW_SIZE - 1
        cv2.circle(v3d, (cam_x, cam_y), 4, (0, 255, 255), -1)
        cv2.putText(v3d, "CAM", (cam_x - 15, cam_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        for r in detections:
            xv, yv = coord_fn(r)
            if abs(xv) > VIEW_RANGE or yv > VIEW_RANGE or yv < 0:
                continue
            px = int(cam_x + xv * PPM)
            py = int(cam_y - yv * PPM)
            px = np.clip(px, panel_x + 5, panel_x + VIEW_SIZE - 5)
            py = np.clip(py, 10, VIEW_SIZE - 10)
            cv2.circle(v3d, (px, py), 6, (0, 200, 255), -1)
            cv2.putText(v3d, r["class_name"], (px + 8, py + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # ── ObjectDB overlay on camera view ──
    db_objects = db.get_all()
    y_off = 45
    cv2.putText(display, f"ObjectDB: {len(db_objects)} objects", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    y_off += 18
    for obj in sorted(db_objects, key=lambda o: o.position_world[1]):  # sorted by depth
        x, y, z = obj.position_world
        cv2.putText(display,
                    f"  {obj.class_name:10s} x={x:.2f} y={y:.2f} z={z:.2f}  c={obj.confidence:.2f}",
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
    cv2.putText(display, f"FPS: {fps_display:.1f} | Objects: {len(detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "ESC to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("PhysicalAI - Live Detection", display)
    cv2.imshow("3D View (left=top-down, right=side)", v3d)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── Print final DB snapshot ──
print("\n=== Final ObjectDB Snapshot ===")
snap = db.snapshot()
if snap:
    for oid, info in snap.items():
        print(f"  {info['class_name']:12s}  pos={info['position_world']}  "
              f"conf={info['confidence']:.2f}  seen={info['observation_count']}x  "
              f"age={info['age_seconds']:.1f}s")
else:
    print("  (empty)")
print(f"Total: {len(snap)} objects")
