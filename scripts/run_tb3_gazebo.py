#!/usr/bin/env python3
"""
run_tb3_gazebo.py — Launch Ignition Fortress + TB3 in small_house
=======================================================================

Starts Ignition Gazebo (gz sim) with the small_house environment,
spawns the TurtleBot3 Waffle Pi, bridges ALL sensor topics to ROS 2,
then runs Grounded SAM 2 detection.

USAGE
-----
    python3 ~/PhysicalAI/scripts/run_tb3_gazebo.py

WHAT YOU SEE
------------
  - Gazebo GUI window (house environment + TB3 robot)
  - OpenCV window showing detection results
  - Console output with FPS and detection updates

ROS 2 TOPICS PUBLISHED
----------------------
  /camera/rgb/image_raw     sensor_msgs/Image
  /camera/rgb/camera_info   sensor_msgs/CameraInfo
  /scan                     sensor_msgs/LaserScan
  /imu                      sensor_msgs/Imu
  /odom                     nav_msgs/Odometry
  /joint_states             sensor_msgs/JointState
  /tf                       tf2_msgs/TFMessage
  /clock                    rosgraph_msgs/Clock

DEPENDENCIES
------------
  sudo apt install ros-humble-turtlebot3-description
  sudo apt install ros-humble-ros-gz-bridge
  # Ignition Fortress should already be installed
"""

import subprocess
import os
import sys
import signal
import time
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────
PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
SCRIPT_DIR = os.path.join(PHYSICALAI_ROOT, "scripts")

# TB3 model from official package
TB3_GAZEBO = "/opt/ros/humble/share/turtlebot3_gazebo"
TB3_MODEL_SDF = os.path.join(TB3_GAZEBO, "models", "turtlebot3_waffle_pi", "model.sdf")
TB3_MESH_DIR = os.path.join(TB3_GAZEBO, "models")

# Navigation2 project paths (contains small_house world + AWS models)
NAV2_DIR = os.path.expanduser("~/navigation2_ignition_gazebo_turtlebot3/turtlebot3")
DATASET_DIR = os.path.join(NAV2_DIR, "maps", "Dataset-of-Gazebo-Worlds-Models-and-Maps")
SMALL_HOUSE_DIR = os.path.join(DATASET_DIR, "worlds", "small_house")
AWS_MODELS_DIR = os.path.join(SMALL_HOUSE_DIR, "models")
SMALL_HOUSE_WORLD_SDF = os.path.join(SMALL_HOUSE_DIR, "small_house_ignition.sdf")

processes = []


def log(msg):
    print(f"[run_tb3] {msg}", flush=True)


def cleanup(signum=None, frame=None):
    log("Shutting down...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    for p in processes:
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    log("All processes stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def check_deps():
    """Verify all required files and commands exist."""
    if not shutil.which("gz"):
        log("ERROR: 'gz' command not found. Install Ignition Gazebo.")
        sys.exit(1)
    if not os.path.exists(TB3_MODEL_SDF):
        log(f"ERROR: TB3 model SDF not found at {TB3_MODEL_SDF}")
        log("Install: sudo apt install ros-humble-turtlebot3-gazebo")
        sys.exit(1)
    if not os.path.exists(SMALL_HOUSE_WORLD_SDF):
        log(f"ERROR: small_house_ignition.sdf not found at {SMALL_HOUSE_WORLD_SDF}")
        log("Make sure the navigation2_ignition_gazebo_turtlebot3 repo is cloned.")
        sys.exit(1)
    log("All dependencies OK.")


def start_gz_sim():
    """Start Ignition Gazebo with the small_house world (contains all furniture + walls)."""
    env = os.environ.copy()
    # Models are symlinked into ~/.ignition/models/ — Ignition finds them automatically
    # Add TB3 meshes to resource path so the TB3 model can find its meshes
    env["GZ_SIM_RESOURCE_PATH"] = os.path.join(TB3_GAZEBO, "models")

    log("Starting gz sim (Ignition Fortress) with small_house world...")
    log(f"  World file: {SMALL_HOUSE_WORLD_SDF}")

    gz_cmd = [
        "gz", "sim",
        "-v", "4",
        "--render-engine", "ogre2",
        SMALL_HOUSE_WORLD_SDF,
    ]

    proc = subprocess.Popen(gz_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    processes.append(proc)
    log(f"  gz sim PID={proc.pid}")
    time.sleep(10)  # wait for init (more models = longer load)


def spawn_tb3_model():
    """Spawn the TurtleBot3 Waffle Pi model into the world."""
    log("Spawning TurtleBot3 Waffle Pi...")
    if not os.path.exists(TB3_MODEL_SDF):
        log(f"ERROR: TB3 model SDF not found at {TB3_MODEL_SDF}")
        return

    spawn_cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-world", "default",
        "-file", TB3_MODEL_SDF,
        "-name", "turtlebot3_waffle_pi",
        "-x", "-3.5",
        "-y", "-4.5",
        "-z", "0.01",
        "-Y", "1.58",
    ]
    proc = subprocess.Popen(
        spawn_cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    processes.append(proc)
    log(f"  spawn TB3 PID={proc.pid}")
    try:
        proc.wait(timeout=15)
        log("  TB3 spawned successfully.")
    except subprocess.TimeoutExpired:
        log("  WARNING: TB3 spawn timed out, continuing anyway...")
    time.sleep(2)


def start_bridge():
    """Bridge ALL topics from gz to ROS 2 using ros_gz_bridge parameter_bridge."""
    yaml_cfg = """
# ── Camera ──────────────────────────────────────────────────────────
- ros_topic_name: "/camera/rgb/image_raw"
  gz_topic_name: "/camera/rgb/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/rgb/camera_info"
  gz_topic_name: "/camera/rgb/camera_info"
  ros_type_name: "sensor_msgs/msg/CameraInfo"
  gz_type_name: "gz.msgs.CameraInfo"
  direction: GZ_TO_ROS

# ── LaserScan ───────────────────────────────────────────────────────
- ros_topic_name: "/scan"
  gz_topic_name: "/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS

# ── IMU ─────────────────────────────────────────────────────────────
- ros_topic_name: "/imu"
  gz_topic_name: "/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

# ── Odometry ────────────────────────────────────────────────────────
- ros_topic_name: "/odom"
  gz_topic_name: "/odom"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS

# ── Joint States ────────────────────────────────────────────────────
- ros_topic_name: "/joint_states"
  gz_topic_name: "/joint_states"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS

# ── TF (transform tree) ─────────────────────────────────────────────
- ros_topic_name: "/tf"
  gz_topic_name: "/tf"
  ros_type_name: "tf2_msgs/msg/TFMessage"
  gz_type_name: "gz.msgs.Pose_V"
  direction: GZ_TO_ROS

# ── Clock ───────────────────────────────────────────────────────────
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
"""
    with open("/tmp/tb3_bridge.yaml", "w") as f:
        f.write(yaml_cfg)

    log("Starting ROS 2 ↔ Gz bridge for all topics...")
    proc = subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
         "--ros-args", "-p", f"config_file:=/tmp/tb3_bridge.yaml"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    processes.append(proc)
    log(f"  bridge PID={proc.pid}")
    time.sleep(2)
    return proc


def check_ros2_topics():
    """Wait for expected ROS 2 topics to appear."""
    log("Waiting for ROS 2 topics...")
    expected = ["camera/rgb/image_raw", "scan", "odom", "imu"]
    for attempt in range(30):
        result = subprocess.run(
            ["timeout", "3", "ros2", "topic", "list"],
            capture_output=True, text=True, timeout=5, env=os.environ
        )
        topics = result.stdout.strip().split("\n")
        found = [t for t in topics if any(e in t for e in expected)]
        if found:
            log(f"  Found topics: {found}")
            return True
        if attempt % 10 == 0:
            log(f"  +{attempt}s: {len(topics)} topics, expected not yet found")
        time.sleep(1)
    log(f"  WARNING: Expected topics not found after 30s.")
    return False


def start_detection():
    """Start tb3_detection.py."""
    script = os.path.join(SCRIPT_DIR, "tb3_detection.py")
    if not os.path.exists(script):
        log(f"ERROR: {script} not found")
        return None

    log("Starting Grounded SAM 2 detection...")
    env = os.environ.copy()
    env["TURTLEBOT3_MODEL"] = "waffle_pi"

    python_bin = shutil.which("python3") or sys.executable
    proc = subprocess.Popen(
        [python_bin, script],
        stdout=sys.stdout, stderr=sys.stderr, env=env
    )
    processes.append(proc)
    log(f"  detection PID={proc.pid}  python={python_bin}")
    return proc


# ═══════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 50)
    log("TB3 + Ignition Gazebo + small_house + Detection")
    log("=" * 50)

    check_deps()
    start_gz_sim()
    spawn_tb3_model()
    bridge = start_bridge()
    check_ros2_topics()
    det_proc = start_detection()

    log("\n--- All systems running ---")
    log("Press Ctrl+C to stop everything.\n")

    # Wait for detection to finish
    if det_proc:
        try:
            det_proc.wait()
        except KeyboardInterrupt:
            pass

    cleanup()


if __name__ == "__main__":
    main()
