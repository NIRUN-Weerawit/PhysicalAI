"""
robot_mcp_server.py — MCP server for PhysicalAI robot control.

Exposes all RobotInterface capabilities as MCP tools over stdin/stdout.
Connect via any MCP client (Hermes, Claude Desktop, etc.).

To use with Hermes, add to ~/.hermes/config.yaml:

    mcp_servers:
      physicalai:
        command: python3
        args: ["~/PhysicalAI/orchestrator/robot_mcp_server.py"]

Protocol: JSON-RPC 2.0 over stdin/stdout (one JSON object per line).
"""
import json, sys, os, importlib.util, traceback

from orchestrator.robot_interface import ToolResult

# ── Tool definitions ─────────────────────────────────────────────────────

TOOLS = [
    # Motion
    {
        "name": "navigate_to",
        "description": "Drive the robot to a coordinate on the map. x, y in meters, theta in radians (0 = forward).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate in map frame (meters)"},
                "y": {"type": "number", "description": "Y coordinate in map frame (meters)"},
                "theta": {"type": "number", "description": "Final orientation in radians (default: 0)"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "navigate_to_object",
        "description": "Find a detected object by class name and navigate to it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Object class to search for (e.g. 'sofa', 'table')"},
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "stop",
        "description": "Cancel all navigation goals immediately. Robot stops where it is. Use this to halt any active motion.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "go_home",
        "description": "Return the robot to the map origin (0, 0, 0).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rotate",
        "description": "Spin the robot in place by N degrees. Positive = CCW, negative = CW.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "angle_deg": {"type": "number", "description": "Degrees to rotate (e.g. 90, -45, 360)"},
            },
            "required": ["angle_deg"],
        },
    },
    {
        "name": "drive",
        "description": "Drive the robot forward or backward a relative distance. Positive = forward, negative = backward.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number", "description": "Distance in meters (positive = forward, negative = backward)"},
                "speed": {"type": "number", "description": "Optional speed in m/s (default: half of max speed)"},
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "wait",
        "description": "Pause for N seconds. No robot motion during this time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to pause"},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "explore",
        "description": "Drive the robot toward unmapped (unknown) areas. BLOCKS until the goal is reached or fails. Use this when you need to expand the map or search for objects. Do NOT call get_status after explore — it returns the outcome directly.",
        "inputSchema": {"type": "object", "properties": {}},
    },

    # Perception
    {
        "name": "detect_now",
        "description": "Run object detection on the latest camera frame. Returns all currently visible objects with class, position, and confidence.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "scan_surroundings",
        "description": "Perform a 360° scan: rotate while detecting. Finds objects in all directions. Returns all detected objects.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "search",
        "description": "Search for a specific object class by rotating 360°. If found, reports its position.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Object class to search for (e.g. 'bottle', 'sofa')"},
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "can_see",
        "description": "Check if a specific object class is visible in the latest camera frame.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Object class to check for"},
            },
            "required": ["class_name"],
        },
    },

    # Status
    {
        "name": "get_status",
        "description": "Full status report: robot pose, map coverage %, number of objects tracked, navigation state, uptime.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_pose",
        "description": "Get the robot's current position and orientation on the map.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_objects",
        "description": "List all objects in the database with their class, position, confidence, and dimensions.",
        "inputSchema": {"type": "object", "properties": {}},
    },

    # Memory & Knowledge
    {
        "name": "remember_place",
        "description": "Save the robot's current position as a named place (e.g. 'home_base', 'charging_station').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for this place"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "go_to_place",
        "description": "Navigate to a previously saved named place.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the place to navigate to"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_places",
        "description": "List all saved named places with their positions.",
        "inputSchema": {"type": "object", "properties": {}},
    },

    # Introspection (read-only ROS 2)
    {
        "name": "ros2_introspect",
        "description": "Run a read-only ROS 2 CLI command. Allowed: topic list, topic echo --once, topic info, node list, service list, action list. Blocked: topic pub, run, lifecycle, bag, launch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "ROS 2 CLI command (e.g. 'topic echo /scan --once', 'node list', 'topic info /camera/rgb/image_raw')"},
            },
            "required": ["query"],
        },
    },

    # Ad-hoc & Detection management
    {
        "name": "run_python",
        "description": "Execute arbitrary Python code with access to robot state (pose, objects, map, np, math, json, math). Assign result to a variable named 'result'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. Must set a variable named 'result'."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_detection_prompt",
        "description": "Return the current object detection text prompt (list of classes Grounding DINO searches for).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_detection_prompt",
        "description": "Change what objects the camera looks for. Pass a prompt like 'bottle. cup. book. chair. person.'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "New detection prompt string (space-separated class names)"},
            },
            "required": ["prompt"],
        },
    },
]

# Will be set by the entry point when the robot is initialized
_robot = None
_places = {}
_latest_status = {}
_db_ref = None
_map_ref = None


def _get_robot():
    """Lazy getter — returns the RobotInterface instance set externally."""
    global _robot
    if _robot is None:
        raise RuntimeError("RobotInterface not initialized. Run robot_mcp_server.py via the orchestrator entry point.")
    return _robot


# ── Tool handler dispatcher ──────────────────────────────────────────────

HANDLERS = {}


def resolve(name: str, args: dict) -> dict:
    """Call a tool by name with args. Returns dict with 'content' and optional 'isError'."""
    handler = HANDLERS.get(name)
    if handler is None:
        return _error_result(f"Unknown tool: {name}")

    try:
        return handler(args)
    except Exception as e:
        traceback.print_exc()
        return _error_result(f"Tool '{name}' failed: {e}")


def _ok_result(result) -> dict:
    """Wrap a successful ToolResult or message for MCP response."""
    msg = str(result) if not isinstance(result, str) else result
    return {
        "content": [{"type": "text", "text": str(msg)}],
    }


def _error_result(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": str(message)}],
        "isError": True,
    }


def _tool_result(result) -> dict:
    """Wrap a ToolResult -> MCP response with correct isError flag."""
    success = getattr(result, 'success', True)
    msg = str(result) if hasattr(result, '__repr__') else str(result)
    if success:
        return {"content": [{"type": "text", "text": msg}]}
    else:
        return {"content": [{"type": "text", "text": msg}], "isError": True}


# ── Register all handlers ────────────────────────────────────────────────

def _reg(name, handler_fn):
    HANDLERS[name] = handler_fn


# Motion
_reg("navigate_to", lambda a: _tool_result(_get_robot().navigate_to(
    float(a.get("x", 0)), float(a.get("y", 0)), float(a.get("theta", 0.0)))))

_reg("navigate_to_object", lambda a: _tool_result(_get_robot().navigate_to_object(
    a.get("class_name", ""), _get_robot()._object_db)))

_reg("stop", lambda a: _tool_result(_get_robot().stop()))

_reg("go_home", lambda a: _tool_result(_get_robot().go_home()))

_reg("rotate", lambda a: _tool_result(_get_robot().rotate(
    a.get("angle_deg", 0))))

_reg("drive", lambda a: _tool_result(_get_robot().drive(
    a.get("distance_m", 0), a.get("speed"))))

_reg("wait", lambda a: _tool_result(_get_robot().wait(
    a.get("seconds", 1))))

_reg("explore", lambda a: _tool_result(_get_robot().explore()))

# Perception
_reg("detect_now", lambda a: _tool_result(_get_robot().detect_now()))

_reg("scan_surroundings", lambda a: _tool_result(_get_robot().scan_surroundings()))

_reg("search", lambda a: _tool_result(_get_robot().search(
    a.get("class_name", ""))))

_reg("can_see", lambda a: _tool_result(_get_robot().can_see(
    a.get("class_name", ""))))

# Status
_reg("get_status", lambda a: _tool_result(_get_robot().get_status(
    _get_robot()._object_db, _get_robot()._map)))

_reg("get_pose", lambda a: _tool_result(_get_robot().get_pose()))

_reg("list_objects", lambda a: _tool_result(_get_robot().list_objects()))

# Memory
_reg("remember_place", lambda a: _handle_remember_place(a))
_reg("go_to_place", lambda a: _handle_go_to_place(a))
_reg("list_places", lambda a: _handle_list_places(a))

# Introspection
_reg("ros2_introspect", lambda a: _tool_result(_get_robot().ros2_introspect(
    a.get("query", ""))))

# Ad-hoc
_reg("run_python", lambda a: _tool_result(_get_robot().run_python(
    a.get("code", ""))))

_reg("get_detection_prompt", lambda a: _tool_result(_get_robot().get_detection_prompt()))

_reg("set_detection_prompt", lambda a: _tool_result(_get_robot().set_detection_prompt(
    a.get("prompt", ""))))


# ── Memory handlers need extra state ─────────────────────────────────────

def _handle_remember_place(args: dict) -> dict:
    name = args.get("name", "unnamed")
    pose_res = _get_robot().get_pose()
    if not pose_res.success:
        return _tool_result(pose_res)
    pose = pose_res.data.get("pose", (0, 0, 0))
    _places[name] = (pose[0], pose[1])
    return _tool_result(ToolResult(True, f"Saved '{name}' at ({pose[0]:.2f}, {pose[1]:.2f})."))


def _handle_go_to_place(args: dict) -> dict:
    name = args.get("name", "")
    if name not in _places:
        return _error_result(f"Place '{name}' not found.")
    x, y = _places[name]
    return _tool_result(_get_robot().navigate_to(x, y))


def _handle_list_places(_args: dict) -> dict:
    if not _places:
        return _tool_result(ToolResult(True, "No places saved."))
    lines = [f"{len(_places)} places:"]
    for name, (x, y) in sorted(_places.items()):
        lines.append(f"  {name}: ({x:.2f}, {y:.2f})")
    return _tool_result(ToolResult(True, "\n".join(lines)))


# ── MCP Protocol (JSON-RPC 2.0 over stdin/stdout) ────────────────────────

def _send(obj: dict):
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    """Read JSON-RPC messages from stdin, dispatch, write to stdout."""
    # Signal ready
    _send({"jsonrpc": "2.0", "method": "log",
           "params": {"message": "PhysicalAI MCP server ready. 24 tools available."}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            continue

        req_id = req.get("id", None)
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "initialize":
            _send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "physicalai-robot", "version": "1.0.0"},
                },
            })

        elif method == "tools/list":
            _send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {"tools": TOOLS},
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            result = resolve(tool_name, tool_args)
            _send({
                "jsonrpc": "2.0", "id": req_id,
                "result": result,
            })

        elif method == "notifications/initialized":
            pass  # No action needed

        else:
            _send({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


def set_robot(robot_instance):
    """Call this from the orchestrator entry point before main() starts."""
    global _robot
    _robot = robot_instance


if __name__ == "__main__":
    main()
