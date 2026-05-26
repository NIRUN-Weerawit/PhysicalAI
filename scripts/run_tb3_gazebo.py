#!/usr/bin/env python3
"""
run_tb3_gazebo.py — Launch TB3 in Gazebo Classic + run detection
================================================================
Starts the full turtlebot3_gazebo launch, ignores the gzclient crash,
monitors for camera topics, then runs Grounded SAM 2 detection.
"""
import subprocess, os, sys, time, signal, threading

PROCS = []

def log(msg):
    print(f"[run_tb3] {msg}", flush=True)

def cleanup(*_):
    for p in list(PROCS):
        try: p.terminate()
        except: pass
    time.sleep(0.5)
    for p in list(PROCS):
        try: p.kill()
        except: pass
    log("Stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

subprocess.run("pkill -9 -f gzserver 2>/dev/null; pkill -9 -f gzclient 2>/dev/null; pkill -9 -f robot_state 2>/dev/null; pkill -9 -f spawn_entity 2>/dev/null; sleep 1", shell=True)

env = os.environ.copy()
env["TURTLEBOT3_MODEL"] = "waffle_pi"

log("Launching TB3 Waffle Pi in Gazebo (turtlebot3_world)...")
log("  gzclient may crash — that's expected, ignoring.")

launch = subprocess.Popen(
    ["ros2", "launch", "turtlebot3_gazebo", "turtlebot3_world.launch.py"],
    env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
PROCS.append(launch)

log("Waiting for gzserver and TB3 spawn...")

def print_launch_output():
    for line in launch.stdout:
        line = line.strip()
        if not line:
            continue
        if "gzclient" in line and ("died" in line or "Assertion" in line or "px != 0" in line):
            continue
        if "gzclient" in line and "process has started" in line:
            continue
        if "spawn" in line.lower() or "successfully" in line.lower():
            log(f"  spawn: {line}")
            continue
        if "joint" in line or "diff" in line or "odom" in line:
            log(f"  robot: {line}")
            continue
        if "ERROR" in line.upper() or "error" in line.lower():
            log(f"  [stderr] {line}")

t = threading.Thread(target=print_launch_output, daemon=True)
t.start()

# Wait for camera topics
found = False
for i in range(45):
    result = subprocess.run(["timeout", "1", "ros2", "topic", "list"], env=env,
                          capture_output=True, text=True)
    topics = result.stdout.strip().split("\n")
    cam = [t for t in topics if "camera" in t]
    has_image = any("image" in t or "depth" in t for t in topics)
    if cam and has_image:
        if not found:
            log(f"=== Camera topics found at +{i}s: {cam} ===")
            found = True
            break
    if i % 10 == 0:
        log(f"  +{i}s: {len(topics)} topics, cameras: {cam or 'none'}")

result = subprocess.run(["timeout", "1", "ros2", "topic", "list"], env=env,
                      capture_output=True, text=True)
all_topics = result.stdout.strip().split("\n")
log(f"Final topics ({len(all_topics)}):")
for t in all_topics:
    log(f"  {t}")

cam_topics = [t for t in all_topics if "camera" in t or "depth" in t or "image" in t]
if cam_topics:
    log("\n=== Starting Grounded SAM 2 Detection ===")
    det = subprocess.run(["python3", os.path.expanduser("~/PhysicalAI/scripts/tb3_detection.py")], env=env)
    log(f"Detection exit code: {det.returncode}")
else:
    log("[FAIL] No camera topics appeared. 45s timeout.")
    log("Check: TURTLEBOT3_MODEL=waffle_pi, gzserver running, TB3 spawned.")

cleanup()
