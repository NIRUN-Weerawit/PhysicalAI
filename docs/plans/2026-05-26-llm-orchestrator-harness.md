# LLM Orchestrator Harness — Plan

> **Goal:** Build an LLM-powered orchestrator that wraps the existing MapOrchestrator capabilities as callable tools, communicates with the user via chat, and makes decisions autonomously.

**Architecture:** A conversation loop that reads user input, passes it to an LLM with a system prompt describing available robot tools, executes the tool call, and reports results back to the user. The LLM is the "brain" — it decides what to do, not a fixed state machine.

**Tech Stack:** Python, the existing MapOrchestrator (ROS 2), Ollama/openrouter API for the LLM, JSON tool-calling format.

## Inventory: what we already have

| File | Status | Purpose |
|---|---|---|
| `orchestrator/map_orchestrator.py` | ✅ Done | Main ROS 2 node: SLAM + Nav2 + detection |
| `orchestrator/launcher.py` | ✅ Done | Start/stop slam_toolbox + Nav2 |
| `orchestrator/tf_bridge.py` | ✅ Done | Camera→map frame transform |
| `orchestrator/nav2_params_tb3.yaml` | ✅ Done | Nav2 params with static layer fix |
| `scripts/run_orchestrator.py` | ✅ Done | Entry point |
| `vision/` detection pipeline | ✅ Done | Grounded SAM 2 + Depth Anything + ObjectDB |

## Tools to expose to the LLM

Each tool is a Python function with a name, description, and typed parameters that the LLM can call via JSON:

1. **`navigate_to(x: float, y: float, theta: float = 0.0)`** — Drive the robot to a map coordinate. Status 6 = reached.

2. **`navigate_to_object(class_name: str)`** — Find an object by class in ObjectDB and navigate to its map-frame position.

3. **`explore()`** — Start frontier-based exploration. Robot drives toward unknown areas.

4. **`stop()`** — Cancel the current Nav2 goal immediately. Robot stops.

5. **`go_home()`** — Return the robot to the map origin (0, 0).

6. **`list_objects()`** — Return a string listing all detected objects with class, map position, and confidence.

7. **`get_status()`** — Return robot pose (x, y, theta), map coverage %, number of objects seen, current exploration state.

8. **`say(message: str)`** — Send a message back to the user (printed to console).

## Missing implementations needed

| Tool | Status | What to add |
|---|---|---|
| `stop()` | ❌ Missing | Cancel Nav2 goal via `_nav_client.async_cancel_goal()` |
| `go_home()` | ❌ Missing | `navigate_to(0.0, 0.0)` as a convenience method |
| `get_status()` | ❌ Missing | Compose robot pose from TF + map coverage + objects |
| `say()` | ❌ Missing | Print to console with [ORCHESTRATOR] prefix |
| Tool registration | ❌ Missing | Dict mapping tool names → (function, description, schema) |

## New files to create

1. **`orchestrator/llm_tools.py`** — Tool definitions + execution registry
2. **`orchestrator/llm_bridge.py`** — LLM client (Ollama/OpenRouter), conversation loop, tool-calling parser
3. **`scripts/run_llm_orchestrator.py`** — Entry point that initializes MapOrchestrator + LLMOrchestrator

## Modified files

1. **`orchestrator/map_orchestrator.py`** — Add `stop()`, `go_home()`, `get_status()` methods, expose ObjectDB + map state to the LLM bridge.

## Task breakdown

### Task 1: Add stop(), go_home(), get_status() to MapOrchestrator

- `cancel_goal()` — uses `_nav_client.async_cancel_all_goals()` on the current goal handle
- `go_home()` — `_send_nav_goal(0.0, 0.0)`
- `get_status()` — reads current pose from `/tf` (base_footprint → map), computes map coverage from `/map` occupancy grid, returns dict

**Files:** Modify `orchestrator/map_orchestrator.py`

### Task 2: Create tool definitions

- `llm_tools.py` defines a `Tool` dataclass: `name`, `description`, `parameters` (JSON schema), `handler` (callable)
- Registry dict: `ALL_TOOLS = {name: Tool(...), ...}`
- Each tool wraps a method on the MapOrchestrator instance

**Files:** Create `orchestrator/llm_tools.py`

### Task 3: Create LLM bridge with conversation loop

- `LLMBridge` class:
  - Stores conversation history
  - Has `system_prompt` that describes all available tools
  - `process_user_message(text)` → sends to LLM → parses tool call → executes → reports
  - Works with Ollama API (localhost:11434) or OpenRouter
- LLM is prompted to respond in JSON: `{"tool": "navigate_to", "args": {"x": 1.5, "y": 2.0}}`
- Or natural language response: `{"reply": "I found the sofa at (3.2, 1.5)."}`

**Files:** Create `orchestrator/llm_bridge.py`

### Task 4: Create entry point

- `run_llm_orchestrator.py` — launches MapOrchestrator + LLM bridge in separate thread, starts chat loop

**Files:** Create `scripts/run_llm_orchestrator.py`
