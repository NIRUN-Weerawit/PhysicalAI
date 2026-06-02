"""
llm_tools.py — Tool registry: name → description, JSON schema, handler.

Each tool is a callable function on RobotInterface, wrapped with
metadata for the LLM's system prompt.  The registry is used both to
generate the system prompt and to dispatch tool calls.
"""
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

# ── Core tools that cannot be unregistered ──────────────────────────────────
PROTECTED_TOOLS = {"stop", "emergency_stop", "navigate_to", "goto_home",
                   "create_skill", "unregister_skill"}


@dataclass
class ToolDef:
    """Definition of a callable tool for LLM consumption."""
    name: str
    description: str
    parameters: dict  # JSON schema
    handler: Callable
    estimated_duration: str = ""
    failure_codes: list = field(default_factory=list)


# ── Registry ────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Manage the set of tools available to the LLM."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, name: str, description: str, handler: Callable,
                 estimated_duration: str = "",
                 failure_codes: list = None):
        """Register a new tool."""
        sig = inspect.signature(handler)
        params = {}
        for pname, param in sig.parameters.items():
            if pname in ("self", "args", "kwargs"):
                continue
            ptype = "string"
            if param.annotation in (float, int):
                ptype = "number"
            elif param.annotation is str:
                ptype = "string"
            elif param.annotation is bool:
                ptype = "boolean"
            elif param.annotation is dict:
                ptype = "object"
            params[pname] = {"type": ptype}

        self._tools[name] = ToolDef(
            name=name,
            description=description,
            parameters=params,
            handler=handler,
            estimated_duration=estimated_duration,
            failure_codes=failure_codes or [],
        )

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def all(self) -> dict[str, ToolDef]:
        return dict(self._tools)

    def unregister(self, name: str) -> bool:
        if name in PROTECTED_TOOLS:
            return False
        return self._tools.pop(name, None) is not None

    def generate_system_prompt(self) -> str:
        """Generate the tool descriptions section for the LLM's system prompt.

        Returns a string the LLM can read to understand what tools exist,
        what they do, and how to call them.
        """
        lines = [
            "## Available Tools",
            "",
            "You have the following tools at your disposal.",
            "",
            "**IMPORTANT RULE:** When calling a tool, respond with ONLY valid JSON.",
            "Do NOT add text before or after the JSON object.",
            "",
            "Correct:",
            '  {"tool": "explore", "args": {}}',
            "",
            "Incorrect:",
            '  "Let me explore. {"tool": "explore", "args": {}}"',
            "",
            "Your response MUST be one of these three formats:",
            "  1. Tool call:  {\"tool\": \"<name>\", \"args\": {<parameters>}}",
            '  2. Direct reply:  {"reply": "<your message to the user>"}',
            '  3. Confirmation:  {"confirm": "<question for user>"}',
            "",
        ]  # end format instructions

        # ── Failure recovery catalogue ──
        lines += [
            "",
            "## Failure Recovery",
            "",
            "When a tool fails, check the `failure_type` and follow this recovery guide:",
            "",
            "| failure_type       | Meaning                    | Recovery Strategy |",
            "|---------------------|----------------------------|-------------------|",
            "| `nav_blocked`       | Nav2 can't find a path     | Rotate 45°, retry. If still blocked: back up 1m, explore alternate route |",
            "| `nav_timeout`       | Goal didn't complete in time| Cancel goal, try intermediate waypoint at half distance |",
            "| `nav_unavailable`   | Nav2 server not ready      | Wait 5s, retry. If persistent: ask user to check Nav2 status |",
            "| `detect_empty`      | Detection returned 0 results| Rotate 45°, retry up to 8 times. If still empty: move to nearest frontier with explore() |",
            "| `depth_nan`         | Depth at centroid is invalid| Expand sampling radius, retry with 3×3 or 7×7 patch median |",
            "| `tf_timeout`        | Camera→map transform unavailable| Wait 2s, retry. If persistent: check if SLAM is running via ros2_introspect() |",
            "| `map_stale`         | Map hasn't updated recently | SLAM may have stopped. Check node list via ros2_introspect(). Restart if needed |",
            "| `goal_out_of_map`   | Goal outside known map area| Explore() first to expand the map, then retry navigation |",
            "| `nav_busy`          | Robot already has a goal   | Wait for current goal to finish, or call stop() first |",
            "| `internal_error`    | Unexpected failure inside tool | Retry once. If persistent: report to user with the exact error message |",
            "",
            "**DO NOT** retry the same tool more than 3 times with the same parameters. After 2 failures, switch to a different strategy.",
            "",
        ]

        for name, tool in sorted(self._tools.items()):
            lines.append(f"### {name}")
            lines.append(f"  {tool.description}")
            if tool.parameters:
                lines.append(f"  Parameters: {tool.parameters}")
            if tool.estimated_duration:
                lines.append(f"  Estimated time: {tool.estimated_duration}")
            if tool.failure_codes:
                codes = ", ".join(tool.failure_codes)
                lines.append(f"  Possible failures: {codes}")
            lines.append("")

        lines.extend([
            "",
            "## Failure Recovery Guide",
            "",
            "When a tool returns failure_type, use the corresponding recovery:",
            "",
            "| Failure type | Recovery strategy |",
            "|---|---|",
            "| nav_blocked | Rotate 90°, retry. If still blocked: back up. |",
            "| nav_timeout | Cancel goal, try intermediate waypoint at half distance. |",
            "| detect_empty | Rotate 45°, retry. Repeat up to 360°. |",
            "| tf_timeout | Wait 1s, check if SLAM is running with ros2_introspect. |",
            "| map_stale | Restart slam_toolbox. |",
            "| nav_busy | Call stop() first, then retry. |",
            "| nav_unavailable | Wait a few seconds, check if Nav2 is running. |",
            "| internal_error | Something unexpected. Report to user via say(). |",
            "",
            "## Safety Rules",
            "",
            "1. Before any motion command: call validate_path(x, y) if available.",
            "2. Before any risky action (new skill, fast movement): call confirm()",
            "   to ask the user for permission.",
            "3. If you don't know how to do something: use confirm() and",
            "   describe what you CAN do as an alternative.",
            "4. If get_status().battery_should_return=True: stop all tasks,",
            "   go_home(), and notify user.",
        ])
        return "\n".join(lines)


# ── Global registry instance ────────────────────────────────────────────────
registry = ToolRegistry()
