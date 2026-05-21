#!/usr/bin/env python3
"""
Calibrate Depth Anything's relative depth to metric meters using a ChArUco board.

Holds the printed ChArUco board at different distances. For each frame where the
board is detected, solvePnP gives the true camera-to-board distance.
The scale factor = true_distance / DepthAnything_relative_output is averaged across frames.

Usage:
    python3 calibrate_depth_scale.py --camera-id 0 \
        --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 25

    Hold the board at 3-5 different distances (0.3m to 2m). Press SPACE/ENTER
    to capture. Press ESC when done. The scale factor is saved to config.json.

    After calibration, run live_detection.py for accurate metric 3D positions.
"""
import sys, os, json, cv2, torch, numpy as np, time
sys.path.insert(0, os.path.expanduser("~/PhysicalAI"))
sys.path.insert(0, os.path.expanduser("~/PhysicalAI/Grounded-SAM-2"))
from pathlib import Path
import argparse

from vision.configs.config import load_vision_config
from vision.depth_estimation.depth_anything_wrapper import DepthAnythingWrapper


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--width", type=int, default=None, help="Camera capture width (e.g. 1280)")
    p.add_argument("--height", type=int, default=None, help="Camera capture height (e.g. 720)")
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=5)
    p.add_argument("--square-mm", type=float, default=35.0)
    p.add_argument("--marker-mm", type=float, default=25.0)
    return p.parse_args()


def create_charuco(args):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts = board.getChessboardCorners()  # (N, 3) in meters
    return board, dictionary, detector, obj_pts


def main():
    args = parse_args()

    # Load config for intrinsics
    cfg = load_vision_config(path=os.path.expanduser("~/PhysicalAI/config.json"),
                              depth_source="depth_anything")
    fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Init Depth Anything (no config scale — we compute it here)
    print("Loading Depth Anything V2...")
    # Temporarily remove depth_scale so estimate() uses heuristic
    # (we sample from estimate_inverse() for calibration)
    da = DepthAnythingWrapper(
        encoder=cfg.depth_anything_encoder,
        checkpoint_path=cfg.depth_anything_checkpoint,
        device=cfg.device,
        grayscale=cfg.depth_anything_grayscale,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )
    da._lazy_init()

    # Create ChArUco board
    board, dictionary, charuco_detector, obj_pts = create_charuco(args)

    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    res_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    res_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"  Camera resolution: {res_w:.0f}x{res_h:.0f}")

    print(f"\n=== Depth Scale Calibration ===")
    print(f"Board: {args.squares_x}x{args.squares_y} squares, "
          f"{args.square_mm}mm squares, {args.marker_mm}mm markers")
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"\nHold the ChArUco board at various distances (0.3m-2m).")
    print(f"SPACE/ENTER → capture this distance")
    print(f"ESC         → finish and compute scale")
    print(f"Try 3-5 different distances for best results.\n")

    scales = []
    distances = []
    last_valid = None  # (charuco_corners, charuco_ids) for delayed capture

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Run Depth Anything — sample RAW inverse depth (what scale multiplies)
        depth_inv = da.estimate_inverse(frame)

        # Detect ChArUco
        cc, ci, mc, mi = charuco_detector.detectBoard(gray)
        display = frame.copy()
        found = ci is not None and len(ci) >= 4

        if found:
            last_valid = (cc, ci)
            # Draw detected corners
            cv2.aruco.drawDetectedCornersCharuco(display, cc, ci)
            if mc is not None:
                cv2.aruco.drawDetectedMarkers(display, mc, mi)

            # solvePnP: use all detected ChArUco corners
            # cc shape: (N, 1, 2) — 2D image points
            # ci shape: (N, 1) — corner IDs (indices into obj_pts)
            corners_2d = cc.reshape(-1, 2).astype(np.float64)
            ids = ci.flatten().astype(np.int32)
            corners_3d = obj_pts[ids].astype(np.float64)

            if len(corners_2d) >= 6:
                success, rvec, tvec = cv2.solvePnP(
                    corners_3d, corners_2d, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE  # IPPE works with 4+ coplanar points
                )
                if success:
                    # Distance from camera to board center = norm of translation
                    true_dist = float(np.linalg.norm(tvec))

                    # Sample Depth Anything output at the same corner pixels
                    disp_at_corners = []
                    for (pu, pv) in corners_2d:
                        pu_i, pv_i = int(round(pu)), int(round(pv))
                        if 0 <= pu_i < w and 0 <= pv_i < h:
                            disp_at_corners.append(depth_inv[pv_i, pu_i])
                    if disp_at_corners:
                        mean_disp = float(np.median(disp_at_corners))
                        status = f"Board: {len(corners_2d)} corners | True dist: {true_dist:.3f}m | DA median: {mean_disp:.4f}"
                    else:
                        status = "Board detected but no depth samples"
                else:
                    status = "solvePnP failed"
            else:
                status = "Too few corners"
            color = (0, 255, 0)
        else:
            status = "No ChArUco board detected"
            color = (0, 0, 255)

        # Overlay info
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(display, f"Captured: {len(scales)} | "
                    f"Scale: {np.mean(scales):.4f}" if scales else "Captured: 0",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "SPACE/ENTER=capture  ESC=finish",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Depth Scale Calibration", display)
        key = cv2.waitKey(10) & 0xFF  # ~100fps max, smoother video

        if key == 27:  # ESC
            break
        elif key in (32, 13):  # SPACE or ENTER
            # Try current frame first, then fallback to last valid
            source_depth = depth_inv
            source_cc = cc
            source_ci = ci
            source_obj = obj_pts
            source_found = found

            if not source_found and last_valid is not None:
                source_cc, source_ci = last_valid
                # Need to re-run depth on the last valid frame... skip fallback for simplicity
                print("  No board in current frame, skipping capture")
                continue

            if not source_found or source_ci is None or len(source_ci) < 6:
                print("  No board detected — hold still and try again")
                continue

            # Compute scale from this capture
            c2d = source_cc.reshape(-1, 2).astype(np.float64)
            ids = source_ci.flatten().astype(np.int32)
            c3d = obj_pts[ids].astype(np.float64)

            success, rvec, tvec = cv2.solvePnP(
                c3d, c2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )
            if not success:
                print("  solvePnP failed, skipping")
                continue

            true_dist = float(np.linalg.norm(tvec))

            # Sample depth at corners
            disp_vals = []
            for (pu, pv) in c2d:
                pu_i, pv_i = int(round(pu)), int(round(pv))
                if 0 <= pu_i < w and 0 <= pv_i < h:
                    disp_vals.append(source_depth[pv_i, pu_i])
            if not disp_vals:
                print("  No depth samples, skipping")
                continue

            mean_disp = float(np.median(disp_vals))
            if mean_disp < 0.0001:
                print(f"  Depth too small ({mean_disp:.6f}), skipping")
                continue

            scale = true_dist / mean_disp
            scales.append(scale)
            distances.append(true_dist)
            print(f"  Captured {len(scales)}: true={true_dist:.3f}m  "
                  f"DA_disp={mean_disp:.4f}  scale={scale:.4f}")

    cap.release()
    cv2.destroyAllWindows()

    if len(scales) < 3:
        print(f"\nERROR: Only {len(scales)} valid captures. Need at least 3.")
        exit(1)

    # Compute final scale
    mean_scale = float(np.mean(scales))
    std_scale = float(np.std(scales))
    print(f"\n=== Results ===")
    print(f"Captures: {len(scales)}")
    print(f"Scale factor: {mean_scale:.4f} ± {std_scale:.4f} ({std_scale/mean_scale*100:.1f}% CV)")
    print(f"Distance range: {min(distances):.3f}m – {max(distances):.3f}m")
    for i, (s, d) in enumerate(zip(scales, distances)):
        print(f"  [{i+1}] dist={d:.3f}m  scale={s:.4f}")

    # Save to config.json
    config_path = os.path.expanduser("~/PhysicalAI/config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["depth_scale"] = round(mean_scale, 4)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved depth_scale={mean_scale:.4f} to {config_path}")

    # Update depth_anything_wrapper to use this scale
    print(f"\nThe scale factor is now in config.json under 'depth_scale'.")
    print(f"The depth wrapper has also been updated to read it automatically.\n")
    print(f"Run live_detection.py to verify accurate 3D positions.")


if __name__ == "__main__":
    main()
