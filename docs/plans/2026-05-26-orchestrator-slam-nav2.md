# SLAM + Nav2 + Object Detection Orchestrator Plan

> **Goal:** Build a ROS 2 node that ties together live SLAM mapping, Nav2 localization/navigation, and Grounded SAM 2 object detection — so the robot can explore an unknown environment, build the map live, detect objects in map frame, and navigate to them autonomously.

**Architecture:** A single `MapOrchestrator` ROS 2 node that launches `slam_toolbox` (online_async) and Nav2 as subprocesses, runs the existing Grounded SAM 2 detection pipeline, transforms detections from camera frame → map frame via TF, and provides `/navigate_to_object` and `/start_exploration` action servers.

**Tech Stack:** ROS 2 Humble, Nav2, slam_toolbox, TF2, Grounded SAM 2, Depth Anything V2

---

### Task 1: Create orchestrator directory structure and params

**Objective:** Set up the folder structure and Nav2 params for the orchestrator.

**Files:**
- Create: `~/PhysicalAI/orchestrator/__init__.py`
- Create: `~/PhysicalAI/orchestrator/nav2_params_tb3.yaml` (copy from nav2_bringup, adjust for TB3 + live SLAM)

**Step 1: Create directories**

```bash
mkdir -p ~/PhysicalAI/orchestrator
touch ~/PhysicalAI/orchestrator/__init__.py
```

**Step 2: Copy and adjust Nav2 params**

Copy `/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml` to `~/PhysicalAI/orchestrator/nav2_params_tb3.yaml`

Key changes from default:
- Set `use_sim_time: True`
- Set `map_server.yaml_filename: ""` (no pre-built map — we use live SLAM)
- Set `global_costmap.static_layer.plugin: "nav2_costmap_2d::StaticLayer"` but with `map_subscribe_transient_local: True` — it will subscribe to `/map` from slam_toolbox
- Ensure `planner_server.GridBased.allow_unknown: true` (critical for navigation on partial map)

---

### Task 2: Create SLAM + Nav2 launcher module

**Objective:** Python module that starts/stops slam_toolbox online_async and Nav2 as subprocesses.

**Files:**
- Create: `~/PhysicalAI/orchestrator/launcher.py`

**Step 1: Write the module**

```python
"""
launcher.py — Launch and manage SLAM + Nav2 subprocesses.
"""
import subprocess, os, signal, time
from typing import Optional

SLAM_TOOLBOX_CONFIG = "/opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml"
NAV2_PARAMS = "..."  # path to our params file


class NavLauncher:
    """Manages slam_toolbox + Nav2 lifecycle."""

    def __init__(self):
        self._slam_proc: Optional[subprocess.Popen] = None
        self._nav2_proc: Optional[subprocess.Popen] = None

    def start_slam(self, env: dict) -> bool:
        """Launch slam_toolbox online_async."""
        cmd = [
            "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
            "--ros-args", "--params-file", SLAM_TOOLBOX_CONFIG,
        ]
        self._slam_proc = subprocess.Popen(cmd, env=env,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
        return True

    def start_nav2(self, env: dict) -> bool:
        """Launch Nav2 bringup."""
        cmd = [
            "ros2", "launch", "nav2_bringup", "navigation_launch.py",
            "use_sim_time:=True",
            f"params_file:={NAV2_PARAMS}",
        ]
        self._nav2_proc = subprocess.Popen(cmd, env=env,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
        return True

    def stop(self):
        for name, proc in [("slam", self._slam_proc), ("nav2", self._nav2_proc)]:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
```

---

### Task 3: Create TF transform utility for detections → map frame

**Objective:** Given a detection centroid + depth in camera frame, lookup TF transform to map frame, return map-frame position.

**Files:**
- Create: `~/PhysicalAI/orchestrator/tf_bridge.py`

**Step 1: Write the module**

Uses `tf2_ros.Buffer` and `TransformListener`.
- `transform_to_map(tf_buffer, point_camera_frame)` → `(x_map, y_map, z_map)`
- Gets latest transform from `camera_rgb_optical_frame` → `map`
- Transforms the 3D point through the chain

---

### Task 4: Create the MapOrchestrator node (main orchestrator)

**Objective:** ROS 2 node that coordinates everything.

**Files:**
- Create: `~/PhysicalAI/orchestrator/map_orchestrator.py`

**Step 1: Write the node**

The node:
- On init: launches slam_toolbox + Nav2 via NavLauncher
- Runs detection pipeline (reuse GroundedSAM2Wrapper + DepthAnythingWrapper)
- Every DETECT_INTERVAL frames: run detection, get centroid + depth, transform via TF to map frame
- Stores detections in ObjectDB with map-frame coordinates
- Provides service: `/find_object` → searches ObjectDB for nearest instance
- Provides service: `/explore` → starts frontier exploration (drive toward unknown areas)
- On Ctrl+C: saves map + stops all processes

**Step 2: Frontier exploration logic**

Simple approach (no complex frontier detection):
1. Check `/map` topic for current occupancy grid
2. Find "unknown" cells (value -1) adjacent to "free" cells (value 0)
3. Cluster them, send nearest cluster centroid as Nav2 goal
4. Repeat until no frontiers remain or object is found

---

### Task 5: Create the entry point script

**Objective:** Single `python3 scripts/run_orchestrator.py` that launches everything.

**Files:**
- Create: `~/PhysicalAI/scripts/run_orchestrator.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
run_orchestrator.py — Full pipeline: Gazebo + SLAM + Nav2 + Detection
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/PhysicalAI"))

from orchestrator.map_orchestrator import main
main()
```

---

### Task 6: Test end-to-end

**Objective:** Run the full pipeline and verify.

**Step 1: Launch Gazebo + TB3**

```bash
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

**Step 2: Run orchestrator**

```bash
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/humble/setup.bash
cd ~/PhysicalAI
python3 scripts/run_orchestrator.py
```

**Step 3: Verify**

Check:
- `/map` topic appears (slam_toolbox is building map)
- `/tf` shows `map → odom → base_footprint → camera_rgb_optical_frame`
- ObjectDB has detections in map frame
- `/find_object` service responds with nearest object position
- Robot can navigate to an object position
