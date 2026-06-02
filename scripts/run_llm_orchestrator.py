#!/usr/bin/env python3
"""
run_llm_orchestrator.py — Full pipeline: SLAM + Nav2 + LLM orchestrator.

Launches the MapOrchestrator ROS 2 node, registers all RobotInterface tools
with the LLM bridge, and starts the conversation loop.

USAGE
-----
    # Terminal 1 — Gazebo
    export TURTLEBOT3_MODEL=waffle_pi
    source /opt/ros/humble/setup.bash
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

    # Terminal 2 — Orchestrator
    source /opt/ros/humble/setup.bash
    cd ~/PhysicalAI
    python3 scripts/run_llm_orchestrator.py
"""
import sys, os, threading, time

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
sys.path.insert(0, PHYSICALAI_ROOT)

from orchestrator.map_orchestrator import MapOrchestrator
from orchestrator.robot_interface import RobotInterface, ToolResult
from orchestrator.llm_tools import registry
from orchestrator.llm_bridge import LLMBridge

# Default detection prompt (matched to map_orchestrator.py)
TEXT_PROMPT = "sphere. shelf. table. chair. human. fire hydrant. stop sign. box. cup. book. bottle. pot. trash can. furniture. sofa. desk. door. plant."


def register_core_tools(registry):
    """Register all RobotInterface methods as tools for the LLM."""

    # Will be populated after the node is created
    _robot_ref = [None]  # mutable container for closure

    def _with_robot():
        return _robot_ref[0]

    # Motion
    registry.register(
        "navigate_to",
        "Drive the robot to any specific (x, y) coordinate on the map. Use this for ALL targeted navigation — even if the target is in an unmapped or unreachable area, the system will automatically snap to the nearest reachable position. x, y in meters, theta in radians (0 = forward). PREFER THIS OVER explore() when you know where you want to go.",
        lambda x, y, theta=0.0: _with_robot().navigate_to(x, y, theta),
        estimated_duration="~4s per meter + 3s overhead",
        failure_codes=["nav_busy", "nav_unavailable"])

    registry.register(
        "navigate_to_object",
        "Navigate to a tracked object by its class name or specific ID (e.g. 'chair_1', 'table'). If multiple objects match, you'll be asked to choose. This internally calls navigate_to() which auto-snaps to reachable positions. PREFER THIS OVER explore() for going to a known object.",
        lambda class_name: _with_robot().navigate_to_object(class_name, _with_robot()._object_db),
        estimated_duration="~4s per meter + 3s overhead",
        failure_codes=["detect_empty", "nav_busy"])

    registry.register(
        "stop",
        "Cancel all navigation goals immediately. Robot stops where it is.",
        lambda: _with_robot().stop(),
        estimated_duration="1s",
        failure_codes=[])

    registry.register(
        "go_home",
        "Return the robot to the map origin (0, 0, 0).",
        lambda: _with_robot().go_home(),
        estimated_duration="~4s per meter from current position")

    registry.register(
        "rotate",
        "Spin the robot in place by N degrees. Positive = counter-clockwise, negative = clockwise.",
        lambda angle_deg: _with_robot().rotate(angle_deg),
        estimated_duration="~1s per 60 degrees",
        failure_codes=["tf_timeout"])

    registry.register(
        "wait",
        "Pause for N seconds. No robot motion during this time.",
        lambda seconds: _with_robot().wait(seconds),
        estimated_duration="same as seconds parameter")

    registry.register(
        "drive",
        "Drive the robot forward or backward a relative distance in meters. Positive = forward, negative = backward. Use for small precise movements or when you don't know the absolute coordinates.",
        lambda distance_m, speed=None: _with_robot().drive(distance_m, speed),
        estimated_duration="~2s per meter",
        failure_codes=[])

    registry.register(
        "explore",
        "Drive toward UNKNOWN (unmapped) frontiers to expand the map. ONLY use this when you want to discover new areas — it picks its OWN destination (nearest unknown frontier). Do NOT use explore() to go to a known coordinate or object — use navigate_to() instead, which accepts specific x,y and snaps to reachable positions.",
        lambda: _with_robot().explore(),
        estimated_duration="~10-60s per frontier",
        failure_codes=["nav_blocked", "nav_busy"])

    # Perception
    registry.register(
        "detect_now",
        "Run object detection on the latest camera frame. Returns all currently visible objects.",
        lambda: _with_robot().detect_now(),
        estimated_duration="~3s (includes detection inference)",
        failure_codes=["internal_error"])

    registry.register(
        "scan_surroundings",
        "Perform a 360° scan: rotate while detecting. Finds objects in all directions.",
        lambda: _with_robot().scan_surroundings(),
        estimated_duration="~25s for full 360° scan",
        failure_codes=[])

    registry.register(
        "search",
        "Search for a specific object class by rotating 360°. If found, reports its position.",
        lambda class_name: _with_robot().search(class_name),
        estimated_duration="~30s for full scan",
        failure_codes=["detect_empty"])

    registry.register(
        "can_see",
        "Check if a specific object class is visible in the latest camera frame.",
        lambda class_name: _with_robot().can_see(class_name))

    # Status
    registry.register(
        "get_status",
        "Full status report: robot pose, map coverage %, number of objects tracked, navigation state.",
        lambda: _with_robot().get_status(_with_robot()._object_db, _with_robot()._map),
        estimated_duration="<1s")

    registry.register(
        "get_pose",
        "Get the robot's current position and orientation on the map.",
        lambda: _with_robot().get_pose(),
        failure_codes=["tf_timeout"])

    registry.register(
        "list_objects",
        "List all objects in the database with their class, position, and confidence.",
        lambda: _with_robot().list_objects())

    registry.register(
        "forget_object",
        "Remove a specific tracked object from memory by its unique ID (e.g. 'chair_1', 'cup_3'). Use list_objects first to see available object IDs.",
        lambda object_id: _with_robot().forget_object(object_id),
        failure_codes=["detect_empty"])

    registry.register(
        "forget_class",
        "Remove all objects of a given class from memory (e.g. 'chair', 'box', 'table'). Useful for cleaning up many objects at once.",
        lambda class_name: _with_robot().forget_class(class_name))

    registry.register(
        "forget_all",
        "Remove ALL tracked objects from memory. Irreversible. Use with caution.",
        lambda: _with_robot().forget_all())

    # Memory
    registry.register(
        "remember_place",
        "Save the robot's current position as a named place (e.g. 'home_base').",
        lambda name: _with_robot().get_pose()  # placeholder — handled by llm_bridge
    )

    registry.register(
        "go_to_place",
        "Navigate to a previously saved named place.",
        lambda name: _with_robot().go_home()  # placeholder — handled by llm_bridge
    )

    registry.register(
        "list_places",
        "List all saved named places.",
        lambda: _with_robot().list_places({})  # placeholder — handled by llm_bridge
    )

    # Introspection
    registry.register(
        "ros2_introspect",
        "Run a read-only ROS 2 CLI command to inspect the system state. Allowed: topic list, topic echo --once, topic info, node list, service list, action list.",
        lambda query: _with_robot().ros2_introspect(query))

    # Special
    registry.register(
        "run_python",
        "Execute arbitrary Python code with read-only access to robot state (pose, objects, map, np, math, json). Assign result to 'result' variable. Use for ad-hoc computations no single tool covers.",
        lambda code: _with_robot().run_python(code),
        estimated_duration="<1s for simple code",
        failure_codes=["internal_error"])

    registry.register(
        "get_detection_prompt",
        "Return the current object detection text prompt (list of classes Grounding DINO searches for).",
        lambda: _with_robot().get_detection_prompt())

    registry.register(
        "set_detection_prompt",
        "Change what objects the camera looks for. Pass a prompt like 'bottle. cup. book. chair. person.' Only use when you need to find objects not in the current prompt.",
        lambda prompt: _with_robot().set_detection_prompt(prompt))

    # Validation
    registry.register(
        "validate_path",
        "Check if a map coordinate is reachable BEFORE attempting navigation. Returns PATH CLEAR or BLOCKED with reason. Use before navigate_to to avoid wasting time on unreachable goals.",
        lambda x, y: _with_robot().validate_path(x, y),
        failure_codes=["goal_out_of_map", "nav_blocked", "map_stale"])

    registry.register(
        "refine_object",
        "Drive closer to a detected object and re-detect it multiple times to improve position accuracy. Use after detecting an object from far away to get a better fix on its exact location.",
        lambda class_name, repetitions: _with_robot().refine_object(class_name, repetitions),
        estimated_duration="~20-60s",
        failure_codes=["detect_empty", "tf_timeout"])

    registry.register(
        "query_graph",
        "Query the spatial knowledge graph for relationships between objects. Examples: 'nearest object to robot', 'objects within 2m of table', 'what is near the sofa'.",
        lambda query: _with_robot().query_graph(query),
        failure_codes=["internal_error"])

    registry.register(
        "create_skill",
        "Write and register a new Python tool at runtime. Pass 'name', 'description', and 'code'. Only use when no existing tool fits.",
        lambda name, description, code: ToolResult(False, "Handled by llm_bridge"),
        estimated_duration="~2s to compile and register")

    registry.register(
        "unregister_skill",
        "Remove a dynamically created skill from the tool registry. Does not affect core tools.",
        lambda name: ToolResult(False, "Handled by llm_bridge"))

    # ── Phase 5: Temporal + Reliability ──
    registry.register(
        "query_history",
        "Query the position history of a detected object over time. Use to answer 'has the chair moved?' or 'where was the sofa 5 minutes ago?'. Returns list of {timestamp, position, confidence} dicts.",
        lambda class_name, minutes_ago: _with_robot().query_history(class_name, minutes_ago),
        failure_codes=["internal_error"])

    registry.register(
        "has_object_moved",
        "Check if an object has physically moved since it was first detected. Returns moved=True/False with the delta in meters. Threshold: 0.2m.",
        lambda class_name: _with_robot().has_object_moved(class_name),
        failure_codes=["internal_error"])

    registry.register(
        "get_battery",
        "Get current battery status: percentage, estimated time remaining, time needed to return home, and whether the robot should head back. Critical if < 10%.",
        lambda: _with_robot().get_battery(),
        failure_codes=["internal_error"])

    registry.register(
        "discover_action",
        "Introspect an unknown ROS 2 action server. Returns the action type and goal/result/feedback fields. Use to discover new robot capabilities at runtime without restarting.",
        lambda action_name: _with_robot().discover_action(action_name),
        failure_codes=["internal_error"])

    registry.register(
        "calibrate_depth",
        "Calibrate depth sensing using a user-provided ground truth position of a tracked object. Call list_objects first to see available IDs. Provide the object_id and the object's actual (real/measured) map-frame x, y coordinates. The system derives a depth scaling correction from the discrepancy, improving all future detections.",
        lambda object_id, ground_truth_x, ground_truth_y: _with_robot().calibrate_depth(object_id, ground_truth_x, ground_truth_y),
        failure_codes=["internal_error"])

    registry.register(
        "get_depth_calibration",
        "Read the current depth scale factor. The system multiplies raw depth readings by this value before computing 3D positions. 1.0 = no correction. Returns the scale, number of calibration samples collected, and the robot's current pose. Use after calibrate_depth() to verify the correction took effect.",
        lambda: _with_robot().get_depth_calibration(),
        failure_codes=["internal_error"])

    registry.register(
        "read_file",
        "Read a text file from the PhysicalAI project directory. Use for inspecting config files, logs, and source code. Path is relative to ~/PhysicalAI/. Only .py, .yaml, .json, .txt, .md, .cfg, .toml, and .log files are allowed for safety.",
        lambda path: _with_robot().read_project_file(path),
        failure_codes=["internal_error"])

    return _robot_ref


def main():
    """Main entry point: starts MapOrchestrator, then LLMBridge in a thread."""
    import rclpy
    rclpy.init()

    # ── Load config from physicalai_config.yaml ──
    config_path = os.path.join(PHYSICALAI_ROOT, "physicalai_config.yaml")
    llm_model = "deepseek-v4-flash:cloud"
    llm_provider = "ollama"
    llm_api_base = "http://localhost:11434"
    llm_api_key = ""
    telegram_token = ""
    telegram_chat = ""
    detection_prompt = TEXT_PROMPT

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                llm_cfg = cfg.get("llm", {})
                llm_provider = llm_cfg.get("provider", "ollama")
                llm_model = llm_cfg.get("model", llm_model)
                llm_api_base = llm_cfg.get("api_base", llm_api_base)
                llm_api_key = llm_cfg.get("api_key", llm_api_key) or ""
                llm_temp = llm_cfg.get("temperature", 0.3)

                tg = cfg.get("telegram", {})
                telegram_token = tg.get("bot_token", "") or ""
                telegram_chat = tg.get("chat_id", "") or ""

                det = cfg.get("detection", {})
                detection_prompt = det.get("prompt", detection_prompt)
                hf_token = det.get("hf_token", "") or ""
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
                    os.environ["HUGGINGFACE_TOKEN"] = hf_token

                rob = cfg.get("robot", {})
                camera_frame = rob.get("camera_frame", "") or ""
                if camera_frame:
                    os.environ["PHYSICALAI_CAMERA_FRAME"] = camera_frame
                persist_db = rob.get("persist_db", "") or ""
                if persist_db:
                    os.environ["PHYSICALAI_PERSIST_DB"] = persist_db

                print(f"  [Config] Loaded from {config_path}")
                print(f"  [Config] LLM: {llm_provider}/{llm_model}")
                print(f"  [Config] Telegram: {'enabled' if telegram_token and telegram_chat else 'disabled'}")
                print(f"  [Config] Object persistence: {'on (SQLite)' if persist_db else 'off (in-memory)'}")
        except ImportError:
            print("  [Config] yaml not installed, using defaults. Install with: pip3 install pyyaml")
        except Exception as e:
            print(f"  [Config] Error reading {config_path}: {e}")

    # 1. Create ROS 2 node
    node = MapOrchestrator()

    # 2. Create RobotInterface and wire it to the node's capabilities
    robot = RobotInterface(node)
    robot.bind_tf_bridge(node.tf_bridge)

    # Bind perception pipeline from the node
    robot.bind_perception(
        detector=node.detector,
        depth_estimator=node.depth_estimator,
        bridge=node.bridge,
        object_db=node.db,
        text_prompt=detection_prompt,
        output_dir=getattr(node, 'output_dir', None),
    )

    # Bind frontier exploration from MapOrchestrator
    robot.bind_exploration(node.find_exploration_goals)

    # Wire orchestrator node reference for calibration etc.
    robot.bind_orchestrator_node(node)

    # 3. Register tools with LLM
    _robot_ref = register_core_tools(registry)
    _robot_ref[0] = robot  # set the mutable ref

    # 4. Create LLM bridge (needed first for the message queue)
    bridge = LLMBridge(robot, registry,
                       model_name=llm_model,
                       api_base=llm_api_base if llm_provider == "ollama" else "",
                       api_key=llm_api_key if llm_provider == "openrouter" else "")

    # 5. Start Telegram gateway (optional) — injects messages into bridge's queue
    telegram_gw = None
    if telegram_token and telegram_chat:
        from orchestrator.telegram_gateway import TelegramGateway
        telegram_gw = TelegramGateway(telegram_token, telegram_chat, bridge._cmd_queue)
        if telegram_gw.test_connection():
            telegram_gw.start_polling()
            print("  [Telegram] Gateway active — you can now command the robot via Telegram.")
        else:
            telegram_gw = None

    # Wire Telegram into bridge
    bridge._telegram = telegram_gw

    # Periodic Nav2 goal checker — injects result into LLM conversation
    def _check_nav_goal():
        result = robot.check_goal_result()
        if result is not None:
            bridge._messages.append({
                "role": "user",
                "content": f"[tool_result] Navigation status: {result.message}",
            })
            if result.success:
                bridge.say(f"Navigation completed: {result.message}")
            else:
                bridge.say(f"Navigation issue: {result.message}")

    node.create_timer(2.0, _check_nav_goal)

    # Keep robot._map in sync with node.current_map
    def _sync_map():
        if node.current_map is not None:
            robot._map = node.current_map

    node.create_timer(3.0, _sync_map)

    # Sync detection prompt: robot._text_prompt → node._text_prompt
    # so MapOrchestrator's detection loop uses the live prompt
    def _sync_prompt():
        node._text_prompt = robot._text_prompt

    node.create_timer(2.0, _sync_prompt)

    def run_llm():
        bridge.run()

    llm_thread = threading.Thread(target=run_llm, daemon=True)
    llm_thread.start()

    # 5b. Start web dashboard (background thread)
    try:
        from orchestrator.dashboard_server import set_shared_state, push_chat_message, set_processing, start_server
        set_shared_state(robot, node, bridge)
        dashboard_thread = start_server(host="0.0.0.0", port=8080)
        bridge._dashboard_push = push_chat_message
        print("  [Dashboard] Web dashboard at http://localhost:8080")
    except ImportError as e:
        print(f"  [Dashboard] fastapi/uvicorn not installed ({e}) — skipping web dashboard")
    except Exception as e:
        print(f"  [Dashboard] Failed to start: {type(e).__name__}: {e} — skipping web dashboard")
        dashboard_thread = None

    # 6. Spin ROS 2 (main thread)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n[FATAL] Orchestrator crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Shutdown] Cleaning up...")
        try:
            bridge.stop()
        except Exception:
            pass
        try:
            node.quit_callback()
        except Exception as e:
            print(f"[Shutdown] quit_callback error: {e}", flush=True)
        # Final sweep: kill any orphaned subprocesses
        try:
            import subprocess
            subprocess.run(["pkill", "-f", "slam_toolbox"],
                           capture_output=True, timeout=3)
            subprocess.run(["pkill", "-f", "nav2_bringup"],
                           capture_output=True, timeout=3)
        except Exception:
            pass
        print("[Shutdown] Done.")


if __name__ == "__main__":
    main()
