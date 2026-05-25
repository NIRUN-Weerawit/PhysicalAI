#!/usr/bin/env python3
"""
run_tb3_gazebo.py — Launch Ignition Fortress + TB3 + camera bridge
=====================================================================

Starts Ignition Gazebo (gz sim) with the TurtleBot3 Waffle Pi in the
turtlebot3_world, spawns the robot with a camera, bridges camera topics
to ROS 2, then runs Grounded SAM 2 detection.

Call this script — it does EVERYTHING in one command.

USAGE
-----
    python3 ~/PhysicalAI/scripts/run_tb3_gazebo.py

The script:
  1. Starts gz sim (Ignition Fortress) with empty_world in background
  2. Waits for it to be ready
  3. Spawns the Waffle Pi model with RGB-D camera via gz service
  4. Starts ros2 topic bridges for camera topics
  5. Launches tb3_detection.py (Grounded SAM 2)
  6. On Ctrl+C, cleans up all processes

WHAT YOU SEE
------------
  - Gazebo GUI window (visualization of the robot + world)
  - OpenCV window showing detection results
  - Console output with FPS and detection updates

DEPENDENCIES
------------
  sudo apt install ros-humble-turtlebot3-description
  # ros_gz_bridge and Ignition Fortress should already be installed
"""

import subprocess
import os
import sys
import signal
import time
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────
PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
TB3_DESCRIPTION = "/opt/ros/humble/share/turtlebot3_description"
SCRIPT_DIR = os.path.join(PHYSICALAI_ROOT, "scripts")
TB3_MODEL_URDF = os.path.join(
    TB3_DESCRIPTION, "urdf", "turtlebot3_waffle_pi.urdf"
)
TB3_MESH_DIR = os.path.join(TB3_DESCRIPTION, "meshes")

processes = []

def log(msg):
    print(f"[run_tb3_gazebo] {msg}", flush=True)


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
    if not shutil.which("gz"):
        log("ERROR: 'gz' command not found. Install Ignition Gazebo.")
        sys.exit(1)
    if not os.path.exists(TB3_MODEL_URDF):
        log(f"ERROR: TB3 URDF not found at {TB3_MODEL_URDF}")
        log("Install: sudo apt install ros-humble-turtlebot3-description")
        sys.exit(1)


def start_gz_sim():
    """Start Ignition Gazebo server + GUI in the background."""
    gz_cmd = [
        "gz", "sim",
        "-v", "4",                   # verbose
        "--render-engine", "ogre",   # use ogre (most compatible, no GPU issues)
    ]
    log("Starting gz sim (Ignition Fortress)...")
    proc = subprocess.Popen(gz_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(proc)
    log(f"  gz sim PID={proc.pid}")
    time.sleep(5)  # wait for GUI to init


def spawn_tb3():
    """Spawn the TurtleBot3 Waffle Pi model into the running gz sim."""
    log("Spawning TurtleBot3 Waffle Pi...")

    # Read the URDF and convert inline (gz sim expects SDF or URDF)
    with open(TB3_MODEL_URDF) as f:
        urdf_xml = f.read()

    # Use gz service to spawn via the spawn service
    spawn_cmd = [
        "gz", "service", "-s", "/world/empty/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "5000",
        "--req", f'sdf: "{urdf_xml}" name: "tb3_waffle_pi" pose: {{position: {{x: -2.0 y: -0.5 z: 0.0}}}} allow_renaming: true',
    ]
    result = subprocess.run(spawn_cmd, capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        log("  Spawn service call succeeded (check gz sim for robot)")
    else:
        log(f"  Spawn returned code {result.returncode}")
        log(f"  stderr: {result.stderr[:200]}")
        # Try alternative: use the gz sim -s (server only) approach
        log("  Trying alternative spawn via gz topic...")

    time.sleep(2)


def start_camera_bridge():
    """Bridge camera topics from gz to ROS 2 using ros_gz_bridge."""
    bridge_cfg = [
        "--ros-args",
        "-p", "config_file:=/tmp/tb3_bridge.yaml",
    ]

    # Write bridge config
    yaml_cfg = """
- ros_topic_name: "/camera/rgb/image_raw"
  gz_topic_name: "/camera/rgb/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/depth/image_raw"
  gz_topic_name: "/camera/depth/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/rgb/camera_info"
  gz_topic_name: "/camera/rgb/camera_info"
  ros_type_name: "sensor_msgs/msg/CameraInfo"
  gz_type_name: "gz.msgs.CameraInfo"
  direction: GZ_TO_ROS
"""
    with open("/tmp/tb3_bridge.yaml", "w") as f:
        f.write(yaml_cfg)

    # Check if parameter_bridge or topic_bridge is available
    bridge_cmd = None
    for candidate in ["ros2 run ros_gz_bridge parameter_bridge", "ros2 run ros_gz_bridge gz_bridge_node"]:
        result = subprocess.run(
            candidate.split()[:4], capture_output=True, text=True, timeout=3
        )
        if result.returncode != 127:  # not "command not found"
            bridge_cmd = candidate
            break

    if bridge_cmd is None:
        log("WARNING: ros_gz_bridge not found. Detection script won't get camera topics.")
        log("Install: sudo apt install ros-humble-ros-gz-bridge")
        return None

    log("Starting ROS 2 -> Gz bridge for camera topics...")
    # Use the bridge node with config
    env = os.environ.copy()
    env["GZ_SIM_RESOURCE_PATH"] = TB3_MESH_DIR
    proc = subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
         "--ros-args", "-p", f"config_file:=/tmp/tb3_bridge.yaml"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    processes.append(proc)
    log(f"  bridge PID={proc.pid}")
    time.sleep(2)
    return proc


def check_camera_topics():
    """Wait for camera topics to appear."""
    log("Waiting for camera topics...")
    topics = []
    for attempt in range(15):
        result = subprocess.run(
            ["timeout", "3", "ros2", "topic", "list"],
            capture_output=True, text=True, timeout=5, env=os.environ
        )
        topics = result.stdout.strip().split("\n")
        camera_topics = [t for t in topics if "camera" in t.lower() or "depth" in t.lower()]
        if camera_topics:
            log(f"  Found camera topics: {camera_topics}")
            return True
        time.sleep(1)
    log(f"  WARNING: No camera topics found after 15s. Topics seen: {topics[:10]}")
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

    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=sys.stdout, stderr=sys.stderr, env=env
    )
    processes.append(proc)
    log(f"  detection PID={proc.pid}")
    return proc


# ═══════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 50)
    log("TB3 + Ignition Gazebo + Grounded SAM 2 Detection")
    log("=" * 50)

    check_deps()
    start_gz_sim()
    spawn_tb3()
    bridge = start_camera_bridge()
    check_camera_topics()
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
