#!/usr/bin/env python3
"""
run_mcp_orchestrator.py — ROS 2 orchestrator with MCP server.

Launches the MapOrchestrator ROS 2 node and RobotInterface, then starts
the MCP server on stdin/stdout.  Connect any MCP client (Hermes, Claude
Desktop, etc.) to control the robot.

No TUI, no Telegram, no LLM bridge — just the robot + MCP transport.

USAGE
-----
    # Terminal 1 — Gazebo
    export TURTLEBOT3_MODEL=waffle_pi
    source /opt/ros/humble/setup.bash
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

    # Terminal 2 — Orchestrator + MCP server
    source /opt/ros/humble/setup.bash
    cd ~/PhysicalAI
    python3 scripts/run_mcp_orchestrator.py

    # Terminal 3 (or Hermes) — Send commands via MCP
    # The MCP server speaks JSON-RPC 2.0 over stdio.
    # Connect Hermes via config.yaml:
    #   mcp_servers:
    #     physicalai:
    #       command: python3
    #       args: ["~/PhysicalAI/scripts/run_mcp_orchestrator.py"]
"""
import sys, os, threading, time

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)

from orchestrator.map_orchestrator import MapOrchestrator
from orchestrator.robot_interface import RobotInterface

# Default detection prompt
TEXT_PROMPT = "sphere. shelf. table. chair. human. fire hydrant. stop sign. box. cup. book. bottle. pot. trash can. furniture. sofa. desk. door. plant."


def main():
    import rclpy
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=True"])

    # ── Load config ──
    config_path = os.path.join(PHYSICALAI_ROOT, "physicalai_config.yaml")
    detection_prompt = TEXT_PROMPT

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                det = cfg.get("detection", {})
                detection_prompt = det.get("prompt", detection_prompt)
        except Exception:
            pass

    # 1. Create ROS 2 node
    node = MapOrchestrator()

    # 2. Create RobotInterface and wire it up
    robot = RobotInterface(node)
    robot.bind_tf_bridge(node.tf_bridge)
    robot.bind_perception(
        detector=node.detector,
        depth_estimator=node.depth_estimator,
        bridge=node.bridge,
        object_db=node.db,
        text_prompt=detection_prompt,
        output_dir=getattr(node, 'output_dir', None),
    )
    robot.bind_exploration(node.find_exploration_goals)

    # 3. Periodic map sync
    def _sync_map():
        if node.current_map is not None:
            robot._map = node.current_map
    node.create_timer(3.0, _sync_map)

    # 4. Prompt sync
    def _sync_prompt():
        node._text_prompt = robot._text_prompt
    node.create_timer(2.0, _sync_prompt)

    # 5. Start MCP server in background thread
    from orchestrator.robot_mcp_server import set_robot, main as mcp_main

    set_robot(robot)

    def run_mcp():
        try:
            mcp_main()
        except Exception as e:
            print(f"  [MCP] Server error: {e}", file=sys.stderr)

    mcp_thread = threading.Thread(target=run_mcp, daemon=True)
    mcp_thread.start()

    print(f"\n  [MCP] PhysicalAI robot server active on stdio.")
    print(f"  [MCP] Connect via Hermes config.yaml:")
    print(f"  [MCP]   mcp_servers:")
    print(f"  [MCP]     physicalai:")
    print(f"  [MCP]       command: python3")
    print(f"  [MCP]       args: [\"{os.path.abspath(__file__)}\"]")
    print(f"  [MCP] 24 tools ready. Ctrl+C to stop.\n")

    # 6. Spin ROS 2 (main thread)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.quit_callback()


if __name__ == "__main__":
    main()
