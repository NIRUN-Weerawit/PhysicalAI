#!/usr/bin/env python3
"""
run_orchestrator.py — Full pipeline: SLAM + Nav2 + Object Detection

Launches the MapOrchestrator node which:
1. Starts slam_toolbox (maps environment live)
2. Starts Nav2 (localization + path planning)
3. Runs Grounded SAM 2 + Depth Anything detection
4. Transforms detections from camera frame → map frame via TF
5. Stores objects in ObjectDB with persistent map-frame positions
6. Provides frontier exploration + object-driven navigation

USAGE
-----
    # Terminal 1 — Gazebo
    export TURTLEBOT3_MODEL=waffle_pi
    source /opt/ros/humble/setup.bash
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

    # Terminal 2 — Orchestrator
    source /opt/ros/humble/setup.bash
    cd ~/PhysicalAI
    python3 scripts/run_orchestrator.py
"""
import sys, os

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)

from orchestrator.map_orchestrator import main

if __name__ == "__main__":
    main()
