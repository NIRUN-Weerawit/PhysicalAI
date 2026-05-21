#!/usr/bin/env python3
"""
Intrinsic camera calibration — supports ChArUco board or plain checkerboard.

CHESSBOARD MODE (--mode checkerboard):
    Use a printed checkerboard (alternating black/white squares).
    python3 calibrate_intrinsics.py --mode checkerboard --camera-id 0 \
        --inner-rows 6 --inner-cols 9 --square-mm 25

    --inner-rows / --inner-cols = number of INNER corners (where 4 squares meet).
    For a 7×10 square board, you'd use --inner-rows 6 --inner-cols 9.
    --square-mm = printed square side in mm (measure with a ruler).

ChARUCO MODE (--mode charuco, default):
    Uses a checkerboard grid with ArUco markers in the white squares.
    More robust than plain checkerboard (handles partial occlusion).

    # Generate a board to print:
    python3 calibrate_intrinsics.py --mode charuco --generate-board \
        --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 25

    # Live calibration:
    python3 calibrate_intrinsics.py --mode charuco --camera-id 0 \
        --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 25

    # Batch from images:
    python3 calibrate_intrinsics.py --mode charuco --image-path ./calib_images \
        --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 25

Output: camera_intrinsics.json
"""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)


# ── Argument parsing ──────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)

    p.add_argument("--mode", choices=["charuco", "checkerboard"], default="charuco",
                   help="Calibration target type (default: charuco)")

    # Shared
    p.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    p.add_argument("--width", type=int, default=None, help="Camera capture width (e.g. 1280)")
    p.add_argument("--height", type=int, default=None, help="Camera capture height (e.g. 720)")
    p.add_argument("--output", type=str, default="camera_intrinsics.json")
    p.add_argument("--image-path", type=str, default=None,
                   help="Calibrate from images in a folder instead of live camera")
    p.add_argument("--generate-board", action="store_true",
                   help="Generate a printable board image and exit")

    # Checkerboard-mode params
    p.add_argument("--inner-rows", type=int, default=6,
                   help="Checkerboard inner corners (rows). E.g. 6 for a 7-wide board")
    p.add_argument("--inner-cols", type=int, default=9,
                   help="Checkerboard inner corners (cols). E.g. 9 for a 10-wide board")

    # ChArUco-mode params
    p.add_argument("--squares-x", type=int, default=7, help="ChArUco squares horizontally")
    p.add_argument("--squares-y", type=int, default=5, help="ChArUco squares vertically")
    p.add_argument("--marker-mm", type=float, default=25.0, help="ChArUco marker side in mm")

    # Shared unit
    p.add_argument("--square-mm", type=float, default=35.0, help="Square side in mm")

    return p.parse_args()


# ── Shared helpers ────────────────────────────────────────────────────────


def save_calibration(mtx, dist, output_path, image_width=None, image_height=None):
    """Save calibration to JSON."""
    data = {
        "camera_matrix": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
        },
        "distortion_coefficients": dist.ravel().tolist(),
    }
    if image_width and image_height:
        data["image_size"] = {"width": image_width, "height": image_height}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {output_path}")


def print_results(ret, mtx, dist, img_w, img_h):
    """Print calibration results in human and config-friendly formats."""
    print(f"\n=== RESULTS ===")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion:\n{dist}")
    print(f"Image size: {img_w}x{img_h}")
    print(f"\nConfig-friendly:")
    print(f'  "fx": {mtx[0,0]:.3f},')
    print(f'  "fy": {mtx[1,1]:.3f},')
    print(f'  "cx": {mtx[0,2]:.3f},')
    print(f'  "cy": {mtx[1,2]:.3f}')


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


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKERBOARD MODE
# ═══════════════════════════════════════════════════════════════════════════


def _create_object_points(inner_cols, inner_rows, square_size):
    """Create the 3D object points for a checkerboard."""
    objp = np.zeros((inner_cols * inner_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2) * square_size
    return objp


def _find_checkerboard_corners(gray, inner_cols, inner_rows):
    """Find checkerboard inner corners."""
    ret, corners = cv2.findChessboardCorners(gray, (inner_cols, inner_rows), None)
    return ret, corners


def generate_checkerboard_image(args):
    """Generate and save a printable plain checkerboard."""
    inner_rows, inner_cols = args.inner_rows, args.inner_cols
    sq = args.square_mm / 1000.0  # meters
    ppi = 300

    # Total board squares = (inner_cols+1) x (inner_rows+1)
    total_w = (inner_cols + 1) * sq
    total_h = (inner_rows + 1) * sq
    px_w = int(total_w * ppi / 0.0254)  # ppi → pixel at meters
    px_h = int(total_h * ppi / 0.0254)

    # Draw checkerboard manually
    step_x = px_w // (inner_cols + 1)
    step_y = px_h // (inner_rows + 1)
    img = np.ones((px_h, px_w), dtype=np.uint8) * 255

    for r in range(inner_rows + 1):
        for c in range(inner_cols + 1):
            if (r + c) % 2 == 0:
                y0, y1 = r * step_y, (r + 1) * step_y
                x0, x1 = c * step_x, (c + 1) * step_x
                img[y0:y1, x0:x1] = 0

    out_path = f"checkerboard_{inner_cols+1}x{inner_rows+1}.png"
    cv2.imwrite(out_path, img)
    print(f"Checkerboard saved to {out_path}")
    print(f"  Grid: {inner_cols+1}×{inner_rows+1} squares ({inner_cols}×{inner_rows} inner corners)")
    print(f"  Square size: {args.square_mm} mm")
    print(f"  Image: {img.shape[1]}×{img.shape[0]} px (300 DPI)")
    print(f"  Print at 100% scale, measure square with a ruler")
    cv2.imshow("Checkerboard (press any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_checkerboard_from_images(args):
    """Calibrate checkerboard from a folder of images."""
    inner_rows, inner_cols = args.inner_rows, args.inner_cols
    objp = _create_object_points(inner_cols, inner_rows, args.square_mm / 1000.0)

    path = Path(args.image_path)
    images = sorted(set(path.glob("*.jpg")) | set(path.glob("*.png")))
    if not images:
        print(f"No images found in {args.image_path}")
        return None

    objpoints = []
    imgpoints = []
    img_shape = None

    print(f"Processing {len(images)} images...")
    for fname in images:
        img = cv2.imread(str(fname))
        if img is None:
            print(f"  Skipping {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if img_shape is None:
            img_shape = (w, h)
        elif (w, h) != img_shape:
            print(f"  {fname}: diff res {(w,h)} vs {img_shape} — skip")
            continue

        ret, corners = _find_checkerboard_corners(gray, inner_cols, inner_rows)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"  {fname.name}: found corners ✓")
        else:
            print(f"  {fname.name}: no corners found ✗")

    if len(objpoints) < 10:
        print(f"ERROR: Only {len(objpoints)} valid images. Need at least 10.")
        return None

    print(f"\nCalibrating from {len(objpoints)} views...")
    h_w = img_shape[0] if isinstance(img_shape, tuple) else 640
    init_mtx = np.array([[h_w * 1.5, 0, h_w / 2],
                         [0, h_w * 1.5, img_shape[1] / 2],
                         [0, 0, 1]], dtype=np.float64)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, init_mtx, None, criteria=CRITERIA
    )
    return ret, mtx, dist, img_shape[0], img_shape[1]


def live_checkerboard_calibration(args):
    """Interactive live checkerboard calibration."""
    inner_rows, inner_cols = args.inner_rows, args.inner_cols
    square_size = args.square_mm / 1000.0
    objp = _create_object_points(inner_cols, inner_rows, square_size)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return None
    res_w, res_h = _set_resolution(cap, args)

    objpoints = []
    imgpoints = []
    img_shape = None
    captured = 0
    last_valid_corners = None
    last_valid_gray = None

    print("\n=== CHECKERBOARD INTRINSIC CALIBRATION ===")
    print(f"Inner corners: {inner_cols}×{inner_rows}, Square: {args.square_mm}mm")
    print(f"Camera: {args.camera_id}")
    print("\nHold checkerboard at various angles.")
    print("SPACE / ENTER → capture frame")
    print("ESC            → finish")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if img_shape is None:
            img_shape = (w, h)

        ret_c, corners = _find_checkerboard_corners(gray, inner_cols, inner_rows)
        display = frame.copy()

        if ret_c:
            last_valid_corners = corners
            last_valid_gray = gray
            cv2.drawChessboardCorners(display, (inner_cols, inner_rows), corners, ret_c)
            status = f"Found | Captured: {captured}"
            color = (0, 255, 0)
        else:
            status = f"Not detected | Captured: {captured}"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "SPACE/ENTER=capture  ESC=finish", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Checkerboard Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (32, 13):  # SPACE or ENTER
            if ret_c:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                objpoints.append(objp)
                imgpoints.append(corners2)
                captured += 1
                print(f"  Captured {captured}")
            elif last_valid_corners is not None and last_valid_gray is not None:
                corners2 = cv2.cornerSubPix(last_valid_gray, last_valid_corners,
                                            (11, 11), (-1, -1), CRITERIA)
                objpoints.append(objp)
                imgpoints.append(corners2)
                captured += 1
                print(f"  Captured {captured} (delayed)")
            else:
                print("  No checkerboard detected — hold still and try again")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print(f"ERROR: Only {len(objpoints)} valid captures. Need at least 10.")
        return None

    print(f"\nCalibrating from {len(objpoints)} views...")
    hw = img_shape[0]
    init_mtx = np.array([[hw * 1.5, 0, hw / 2],
                         [0, hw * 1.5, img_shape[1] / 2],
                         [0, 0, 1]], dtype=np.float64)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, init_mtx, None, criteria=CRITERIA
    )
    return ret, mtx, dist, img_shape[0], img_shape[1]


# ═══════════════════════════════════════════════════════════════════════════
#  ChArUco MODE
# ═══════════════════════════════════════════════════════════════════════════


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def create_charuco_board(args):
    dictionary = get_dictionary()
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, dictionary, detector


def detect_charuco(gray, board, dictionary, detector):
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 4:
        return False, None, None
    return True, charuco_corners, charuco_ids


def generate_charuco_board_image(args):
    board, _, _ = create_charuco_board(args)
    ppi = 300
    w_inches = args.squares_x * args.square_mm / 25.4
    h_inches = args.squares_y * args.square_mm / 25.4
    img = board.generateImage((int(w_inches * ppi), int(h_inches * ppi)))
    out_path = f"charuco_board_{args.squares_x}x{args.squares_y}.png"
    cv2.imwrite(out_path, img)
    print(f"ChArUco board saved to {out_path}")
    print(f"  Grid: {args.squares_x}×{args.squares_y} squares")
    print(f"  Square: {args.square_mm}mm, Marker: {args.marker_mm}mm")
    print(f"  Image: {img.shape[1]}×{img.shape[0]} px (300 DPI)")
    print(f"  Print at 100% scale, measure square with a ruler")
    cv2.imshow("ChArUco Board (press any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_charuco_from_images(args):
    board, dictionary, detector = create_charuco_board(args)
    obj_pts = board.getChessboardCorners()

    path = Path(args.image_path)
    images = sorted(set(path.glob("*.jpg")) | set(path.glob("*.png")))
    if not images:
        print(f"No images found in {args.image_path}")
        return None

    all_objpoints = []
    all_imgpoints = []
    img_shape = None

    print(f"Processing {len(images)} images...")
    for fname in images:
        img = cv2.imread(str(fname))
        if img is None:
            print(f"  Skipping {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if img_shape is None:
            img_shape = (w, h)
        elif (w, h) != img_shape:
            print(f"  {fname}: diff res {(w,h)} vs {img_shape} — skip")
            continue

        found, charuco_corners, charuco_ids = detect_charuco(gray, board, dictionary, detector)
        if found:
            corners_2d = charuco_corners.reshape(-1, 2).astype(np.float32)
            idx = charuco_ids.flatten().astype(np.int32)
            corners_3d = obj_pts[idx].astype(np.float32)
            corners_2d = cv2.cornerSubPix(gray, corners_2d, (5, 5), (-1, -1), CRITERIA)
            all_objpoints.append(corners_3d)
            all_imgpoints.append(corners_2d)
            print(f"  {fname.name}: {len(corners_3d)} corners ✓")
        else:
            print(f"  {fname.name}: not detected ✗")

    if len(all_objpoints) < 5:
        print(f"ERROR: Only {len(all_objpoints)} valid images. Need at least 5.")
        return None

    print(f"\nCalibrating from {len(all_objpoints)} views...")
    hw = img_shape[0]
    init_mtx = np.array([[hw * 1.5, 0, hw / 2],
                         [0, hw * 1.5, img_shape[1] / 2],
                         [0, 0, 1]], dtype=np.float64)
    imgpoints_shaped = [p.reshape(-1, 1, 2) for p in all_imgpoints]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_objpoints, imgpoints_shaped, img_shape, init_mtx, None, criteria=CRITERIA
    )
    return ret, mtx, dist, img_shape[0], img_shape[1]


def live_charuco_calibration(args):
    board, dictionary, detector = create_charuco_board(args)
    obj_pts = board.getChessboardCorners()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        return None
    res_w, res_h = _set_resolution(cap, args)

    all_objpoints = []
    all_imgpoints = []
    img_shape = None
    captured = 0
    last_valid_2d = None
    last_valid_3d = None

    print("\n=== ChArUco INTRINSIC CALIBRATION ===")
    print(f"Board: {args.squares_x}×{args.squares_y} squares")
    print(f"Square: {args.square_mm}mm, Marker: {args.marker_mm}mm")
    print(f"Camera: {args.camera_id}")
    print("\nHold board at various angles.")
    print("SPACE / ENTER → capture frame")
    print("ESC            → finish")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if img_shape is None:
            img_shape = (w, h)

        found, charuco_corners, charuco_ids = detect_charuco(gray, board, dictionary, detector)
        display = frame.copy()

        _, _, marker_corners, marker_ids = detector.detectBoard(gray)
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        if found and charuco_corners is not None:
            corners_2d = charuco_corners.reshape(-1, 2).astype(np.float32)
            idx = charuco_ids.flatten().astype(np.int32)
            corners_3d = obj_pts[idx].astype(np.float32)
            corners_2d = cv2.cornerSubPix(gray, corners_2d, (5, 5), (-1, -1), CRITERIA)
            last_valid_2d = corners_2d
            last_valid_3d = corners_3d
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            status = f"{len(corners_2d)} corners | Captured: {captured}"
            color = (0, 255, 0)
        else:
            status = f"Not detected | Captured: {captured}"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "SPACE/ENTER=capture  ESC=finish", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("ChArUco Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break
        elif key in (32, 13):
            if found and charuco_ids is not None and len(charuco_ids) >= 4:
                all_objpoints.append(corners_3d)
                all_imgpoints.append(corners_2d)
                captured += 1
                print(f"  Captured {captured} ({len(corners_3d)} corners)")
            elif last_valid_2d is not None:
                all_objpoints.append(last_valid_3d)
                all_imgpoints.append(last_valid_2d)
                captured += 1
                print(f"  Captured {captured} (delayed, {len(last_valid_3d)} corners)")
            else:
                print("  No board detected — hold still and try again")

    cap.release()
    cv2.destroyAllWindows()

    if len(all_objpoints) < 5:
        print(f"ERROR: Only {len(all_objpoints)} valid captures. Need at least 5.")
        return None

    print(f"\nCalibrating from {len(all_objpoints)} views...")

    # Reshape image points back to (N, 1, 2) — calibrateCamera format
    imgpoints_shaped = [p.reshape(-1, 1, 2) for p in all_imgpoints]

    # Provide initial guess for camera matrix (focal length ≈ image width)
    fx_guess = img_shape[0] * 1.5
    fy_guess = img_shape[0] * 1.5
    cx_guess = img_shape[0] / 2.0
    cy_guess = img_shape[1] / 2.0
    init_mtx = np.array([[fx_guess, 0, cx_guess],
                         [0, fy_guess, cy_guess],
                         [0, 0, 1]], dtype=np.float64)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_objpoints, imgpoints_shaped, img_shape, init_mtx, None, criteria=CRITERIA
    )
    return ret, mtx, dist, img_shape[0], img_shape[1]


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "checkerboard":
        if args.generate_board:
            generate_checkerboard_image(args)
            exit(0)
        if args.image_path:
            result = calibrate_checkerboard_from_images(args)
        else:
            result = live_checkerboard_calibration(args)
    else:
        if args.generate_board:
            generate_charuco_board_image(args)
            exit(0)
        if args.image_path:
            result = calibrate_charuco_from_images(args)
        else:
            result = live_charuco_calibration(args)

    if result is None:
        exit(1)

    ret, mtx, dist, img_w, img_h = result
    print_results(ret, mtx, dist, img_w, img_h)
    save_calibration(mtx, dist, args.output, img_w, img_h)
