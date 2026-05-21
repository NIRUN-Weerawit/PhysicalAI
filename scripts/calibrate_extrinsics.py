#!/usr/bin/env python3
"""
Extrinsic camera calibration — finds each camera's pose in the world frame.

Three methods:

Method A — ChArUco board (recommended, most accurate):
  1. Place the printed ChArUco board at the world origin (robot base, table corner).
  2. Detect it. solvePnP gives board-pose-in-camera; invert → camera-pose-in-world.
  3. Camera-in-world pose is saved to camera_extrinsics.json (for TF tree).

Method B — Single ArUco marker (quick & practical):
  1. Place a known ArUco marker at world origin.
  2. Same principle: marker pose → invert → camera pose in world.

Method C — Manual measurement:
  1. Measure translation + rotation with a ruler/protractor.
  2. Good for quick prototyping.

Usage:
    # ChArUco (recommended — uses your printed board):
    python3 calibrate_extrinsics.py --method charuco --camera-id 0 \
        --squares-x 7 --squares-y 5 --square-mm 36 --marker-mm 25

    # ArUco marker:
    python3 calibrate_extrinsics.py --method aruco --camera-id 0 \
        --marker-size 0.05 --marker-id 0

    # Manual:
    python3 calibrate_extrinsics.py --method manual \
        --tx 0.0 --ty 0.0 --tz 1.0 --rx 0 --ry 0 --rz 0

Output: camera_extrinsics.json with translation + quaternion for TF tree.

NOTE: Camera-in-world frame uses the same convention as the pipeline:
  x = right, y = forward, z = up
"""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--width", type=int, default=None, help="Camera capture width (e.g. 1280)")
    p.add_argument("--height", type=int, default=None, help="Camera capture height (e.g. 720)")
    p.add_argument("--method", choices=["charuco", "aruco", "manual"], default="charuco")

    # ChArUco params
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=5)
    p.add_argument("--square-mm", type=float, default=35.0)
    p.add_argument("--marker-mm", type=float, default=25.0)

    # ArUco marker params
    p.add_argument("--marker-size", type=float, default=0.05,
                   help="ArUco marker side length in meters")
    p.add_argument("--marker-id", type=int, default=0)

    # Intrinsics
    p.add_argument("--intrinsics", type=str, default="camera_intrinsics.json",
                   help="Intrinsic calibration JSON file")

    # Manual extrinsics
    p.add_argument("--tx", type=float, default=0.0)
    p.add_argument("--ty", type=float, default=0.0)
    p.add_argument("--tz", type=float, default=1.0)
    p.add_argument("--rx", type=float, default=0.0, help="Rotation X in degrees")
    p.add_argument("--ry", type=float, default=0.0)
    p.add_argument("--rz", type=float, default=0.0)

    # Output
    p.add_argument("--camera-name", type=str, default="camera_0")
    p.add_argument("--parent-frame", type=str, default="world")
    p.add_argument("--output", type=str, default="camera_extrinsics.json")

    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────


def load_intrinsics(path):
    """Load camera matrix and distortion from JSON."""
    with open(path) as f:
        calib = json.load(f)
    mtx = np.array([
        [calib["camera_matrix"]["fx"], 0, calib["camera_matrix"]["cx"]],
        [0, calib["camera_matrix"]["fy"], calib["camera_matrix"]["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.array(calib.get("distortion_coefficients", [0, 0, 0, 0, 0]),
                    dtype=np.float64).ravel()
    return mtx, dist


def camera_pose_from_board(R_board_in_cam, t_board_in_cam):
    """Given board-pose-in-camera, compute camera-pose-in-world.
    camera_in_world = inverse(board_in_camera):
      R_cam = R_board^T
      t_cam = -R_cam @ t_board
    """
    R_cam = R_board_in_cam.T
    t_cam = (-R_cam @ t_board_in_cam).flatten()
    return t_cam, R_cam


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion."""
    from scipy.spatial.transform import Rotation as Rot
    q_xyzw = Rot.from_matrix(R).as_quat()  # [x, y, z, w]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def _set_resolution(cap, args):
    """Set camera resolution if --width/--height specified."""
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        w = args.width
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        h = args.height
    print(f"  Camera resolution: {w:.0f}x{h:.0f}")
    return int(w), int(h)


def save_extrinsics(translation, rotation, output_path, camera_name, parent_frame):
    """Save extrinsics as JSON for the pipeline."""
    t_list = translation.tolist() if hasattr(translation, 'tolist') else list(translation)
    quat = rotation_matrix_to_quaternion(rotation)

    data = {
        "camera_name": camera_name,
        "parent_frame": parent_frame,
        "extrinsics": {
            "translation": t_list,
            "rotation": quat,  # [w, x, y, z]
        },
        "rotation_matrix": rotation.tolist() if hasattr(rotation, 'tolist') else rotation,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nExtrinsics saved to {output_path}")
    print(f"\nAdd to TF tree config:")
    print(f"""  {{
    "type": "camera",
    "name": "{camera_name}",
    "translation": {t_list},
    "rotation (qw qx qy qz)": {quat},
    "parent": "{parent_frame}"
  }}""")
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Method A: ChArUco board
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_charuco(args):
    """Interactive ChArUco-based extrinsic calibration.

    Place the printed ChArUco board at the world origin with its surface
    aligned to the world coordinate frame (e.g. flat on the table with
    the marker grid aligned to x/y axes). Press SPACE to capture from
    multiple angles and average the computed camera pose.
    """
    mtx, dist = load_intrinsics(args.intrinsics)

    # Build ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts = board.getChessboardCorners()  # (N, 3) in meters

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return None
    res_w, res_h = _set_resolution(cap, args)

    collected_t = []
    collected_R = []

    print(f"\n=== EXTRINSIC CALIBRATION (ChArUco method) ===")
    print(f"Board: {args.squares_x}x{args.squares_y} squares, "
          f"{args.square_mm}mm squares, {args.marker_mm}mm markers")
    print(f"Place the ChArUco board at the world origin (robot base / table corner).")
    print(f"Hold it flat with the grid facing the camera.")
    print(f"SPACE / ENTER → capture this pose")
    print(f"ESC            → finish and save")
    print(f"Capture from 3+ different angles/positions for averaging.")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect ChArUco
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(gray)
        display = frame.copy()
        found = charuco_ids is not None and len(charuco_ids) >= 6

        if found:
            # Draw detections
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            if marker_corners is not None:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

            # solvePnP: ChArUco corners → board pose in camera frame
            c2d = charuco_corners.reshape(-1, 2).astype(np.float64)
            ids = charuco_ids.flatten().astype(np.int32)
            c3d = obj_pts[ids].astype(np.float64)

            success, rvec, tvec = cv2.solvePnP(
                c3d, c2d, mtx, dist, flags=cv2.SOLVEPNP_IPPE
            )

            if success:
                R_board_in_cam, _ = cv2.Rodrigues(rvec)

                # Draw axes on the board
                cv2.drawFrameAxes(display, mtx, dist, rvec, tvec, 0.1)

                # Convert: camera pose = inverse of board pose
                t_cam, R_cam = camera_pose_from_board(R_board_in_cam, tvec)

                info = f"Captured: {len(collected_t)}  |  "
                info += f"Cam: x={t_cam[0]:.3f} y={t_cam[1]:.3f} z={t_cam[2]:.3f}m"
                cv2.putText(display, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(display, "solvePnP failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"No ChArUco board detected ({len(charuco_ids) if charuco_ids is not None else 0} corners)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(display, "SPACE/ENTER=capture  ESC=finish",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Extrinsic Calibration (ChArUco)", display)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            break
        elif key in (32, 13):
            if not found or not success:
                print("  No board detected — hold still and try again")
                continue

            collected_t.append(t_cam)
            collected_R.append(R_cam)
            print(f"  Capture {len(collected_t)}: cam=({t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f})")

    cap.release()
    cv2.destroyAllWindows()

    if len(collected_t) < 3:
        print(f"ERROR: Only {len(collected_t)} valid captures. Need at least 3.")
        return None

    # Average translation
    avg_t = np.mean(collected_t, axis=0)

    # Average rotation: convert each to quaternion, average, renormalize
    from scipy.spatial.transform import Rotation as Rot
    quats = [Rot.from_matrix(R).as_quat() for R in collected_R]  # [x,y,z,w]
    avg_q = np.mean(quats, axis=0)
    avg_q /= np.linalg.norm(avg_q)
    avg_R = Rot.from_quat(avg_q).as_matrix()

    print(f"\nAverage camera pose (world frame):")
    print(f"  Translation: ({avg_t[0]:.4f}, {avg_t[1]:.4f}, {avg_t[2]:.4f})")
    q_wxyz = rotation_matrix_to_quaternion(avg_R)
    print(f"  Rotation quat [w x y z]: ({q_wxyz[0]:.4f}, {q_wxyz[1]:.4f}, {q_wxyz[2]:.4f}, {q_wxyz[3]:.4f})")

    return avg_t, avg_R


# ═══════════════════════════════════════════════════════════════════════════
#  Method B: Single ArUco marker
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_aruco(args):
    """Interactive ArUco-based extrinsic calibration.

    Place a known ArUco marker at world origin. The camera pose is the
    inverse of the detected marker pose.
    """
    mtx, dist = load_intrinsics(args.intrinsics)

    # OpenCV 4.13+ ArucoDetector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return None
    res_w, res_h = _set_resolution(cap, args)

    collected_t = []
    collected_R = []

    print(f"\n=== EXTRINSIC CALIBRATION (ArUco method) ===")
    print(f"Marker ID: {args.marker_id}, size: {args.marker_size}m")
    print(f"Place marker {args.marker_id} at world origin.")
    print("SPACE / ENTER → capture pose")
    print("ESC            → finish")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        corners, ids, _ = detector.detectMarkers(gray)
        display = frame.copy()

        found = ids is not None and args.marker_id in ids.flatten()

        if found:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            idx = list(ids.flatten()).index(args.marker_id)
            marker_corners = corners[idx]

            # estimatePoseSingleMarkers still works in 4.13
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners, args.marker_size, mtx, dist
            )
            rvec, tvec = rvec[0], tvec[0]

            cv2.drawFrameAxes(display, mtx, dist, rvec, tvec, args.marker_size * 2)

            R_marker, _ = cv2.Rodrigues(rvec)
            t_cam, R_cam = camera_pose_from_board(R_marker, tvec)

            info = f"Captured: {len(collected_t)}  |  "
            info += f"Cam: x={t_cam[0]:.3f} y={t_cam[1]:.3f} z={t_cam[2]:.3f}m"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Marker {args.marker_id} not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(display, "SPACE/ENTER=capture  ESC=finish",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Extrinsic Calibration (ArUco)", display)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            break
        elif key in (32, 13):
            if not found:
                print("  Marker not found, skipping")
                continue
            collected_t.append(t_cam)
            collected_R.append(R_cam)
            print(f"  Capture {len(collected_t)}: cam=({t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f})")

    cap.release()
    cv2.destroyAllWindows()

    if len(collected_t) < 3:
        print(f"ERROR: Only {len(collected_t)} valid captures. Need at least 3.")
        return None

    # Average
    avg_t = np.mean(collected_t, axis=0)
    from scipy.spatial.transform import Rotation as Rot
    quats = [Rot.from_matrix(R).as_quat() for R in collected_R]
    avg_q = np.mean(quats, axis=0)
    avg_q /= np.linalg.norm(avg_q)
    avg_R = Rot.from_quat(avg_q).as_matrix()

    print(f"\nAverage camera position: ({avg_t[0]:.4f}, {avg_t[1]:.4f}, {avg_t[2]:.4f})")
    return avg_t, avg_R


# ═══════════════════════════════════════════════════════════════════════════
#  Method C: Manual
# ═══════════════════════════════════════════════════════════════════════════


def manual_calibration(tx, ty, tz, rx_deg, ry_deg, rz_deg):
    """Compute extrinsics from manual measurements."""
    from scipy.spatial.transform import Rotation as R
    t = np.array([tx, ty, tz])
    rot = R.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True)
    R_mat = rot.as_matrix()
    print(f"\nManual extrinsics:")
    print(f"  Translation: ({tx}, {ty}, {tz})")
    print(f"  Rotation (deg): ({rx_deg}, {ry_deg}, {rz_deg})")
    return t, R_mat


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    args = parse_args()

    # Resolve intrinsics path
    if not Path(args.intrinsics).exists():
        # Try looking in ~/PhysicalAI/
        alt = Path.home() / "PhysicalAI" / args.intrinsics
        if alt.exists():
            args.intrinsics = str(alt)
        else:
            print(f"Intrinsics file not found: {args.intrinsics}")
            exit(1)

    if args.method == "charuco":
        result = calibrate_charuco(args)
    elif args.method == "aruco":
        result = calibrate_aruco(args)
    elif args.method == "manual":
        result = manual_calibration(args.tx, args.ty, args.tz,
                                    args.rx, args.ry, args.rz)
    else:
        print("Unknown method")
        exit(1)

    if result is None:
        exit(1)

    t, R = result
    save_extrinsics(t, R, args.output, args.camera_name, args.parent_frame)
