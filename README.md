# PhysicalAI — Autonomous Mobile Robot Orchestrator with LLM Reasoning

**Repo:** `github.com/NIRUN-Weerawit/PhysicalAI`
**Stack:** ROS 2 Humble | Grounded SAM 2 | Depth Anything V2 | slam_toolbox | Nav2 | LLM (Ollama/OpenRouter) | FastAPI Dashboard | Telegram

An autonomous mobile robot system that combines real-time SLAM mapping, open-vocabulary object detection, and LLM-powered reasoning. The robot can explore unknown environments, detect and track objects, navigate intelligently, and respond to natural language commands through multiple interfaces.

---

## 🏛️ Architecture Overview (4-Layer Design)

```
Layer 4:  User Interface     ─── TUI (rich console) + Telegram bot + Web Dashboard (FastAPI)
Layer 3:  LLM Brain          ─── LLM conversation loop, tool-calling, thinking extraction
Layer 2:  Robot Interface    ─── 28+ tools: motion, perception, memory, introspection
Layer 1:  ROS 2 Core         ─── SLAM + Nav2 + detection + TF transforms
```

Each layer communicates downward via method calls and upward via structured `ToolResult` objects with typed failure codes.

---

## 📁 File Map & Responsibilities

### `orchestrator/` — Core Orchestrator

| File | Lines | Role |
|------|-------|------|
| `map_orchestrator.py` | 745 | Main ROS 2 node: launches SLAM+Nav2, subscribes to camera/map/scan, runs Grounded SAM 2 detection, frontier exploration, depth calibration, map saving |
| `robot_interface.py` | 1407 | Clean API wrapping ALL robot capabilities into ~30 tool methods, each returning `ToolResult(success, message, data, failure_type, estimated_duration_s)` |
| `llm_bridge.py` | 596 | LLM conversation loop: Ollama/OpenRouter client, thinking extraction (7 formats), JSON tool parser, rich TUI panels, Telegram + dashboard push |
| `llm_tools.py` | 170 | Tool registry: `ToolDef(name, description, JSON schema, handler)`, auto-generates LLM system prompt, failure recovery catalog |
| `robot_mcp_server.py` | 446 | MCP (Model Context Protocol) server over stdio: 24 tools exposed as JSON-RPC 2.0, connects Hermes/Claude Desktop |
| `launcher.py` | 131 | Spawns/manages `slam_toolbox` + Nav2 as subprocesses in isolated process groups, killpg cleanup |
| `tf_bridge.py` | 161 | Camera→map frame transform via TF2 buffer, auto-detects camera frame, handles optical vs non-optical conventions |
| `safety_monitor.py` | 123 | Reactive collision avoidance: monitors laser scan forward sector, force-publishes zero-velocity on danger, hysteresis debounce |
| `drift_monitor.py` | 84 | Semantic loop closure: detects SLAM drift by tracking re-observed objects as landmarks |
| `dashboard_server.py` | 614 | FastAPI web dashboard: live camera stream, occupancy grid with robot/objects overlay, velocity graphs, chat panel, system logs |
| `telegram_gateway.py` | 127 | Bidirectional Telegram ↔ LLM bridge: polls for commands, relays messages and photos |
| `nav2_params_tb3.yaml` | 333 | Nav2 parameters for TB3: AMCL, costmaps, planners, BT navigator, recovery behaviors |

### `scripts/` — Entry Points & Utilities

| Script | Purpose |
|--------|---------|
| `run_llm_orchestrator.py` | **Primary entry point** — Full stack: MapOrchestrator + RobotInterface + 28 LLM tools + Telegram + Web dashboard |
| `run_mcp_orchestrator.py` | MCP mode: same stack but connects via stdio JSON-RPC instead of TUI — for Hermes/Claude Desktop |
| `run_orchestrator.py` | Minimal mode: SLAM + Nav2 + detection only (no LLM, no TUI) |
| `run_tb3_gazebo.py` | Launch Ignition Fortress + small_house world + TB3 + topic bridging + detection, automated |
| `tb3_detection.py` | Standalone ROS 2 detection node (Grounding SAM 2 + Depth Anything on camera topics) |
| `live_detection.py` | Local-mode detection pipeline (no ROS) |
| `calibrate_intrinsics.py` | ChArUco board intrinsic calibration |
| `calibrate_extrinsics.py` | Camera extrinsic calibration |
| `calibrate_depth_scale.py` | Monocular depth scale calibration from known distances |
| `isaacsim_live_detection.py` | Grounded SAM 2 + Isaac Sim ground-truth RGB-D |
| `isaacsim_calibrate_depth_scale.py` | Calibrate depth_scale using sim ground-truth object positions |

### `vision/` — Perception Pipeline

| Module | Files | Purpose |
|--------|-------|---------|
| `detection/` | `grounded_sam2_wrapper.py` | Grounding DINO + SAM 2 for open-vocabulary segmentation/detection |
| `depth_estimation/` | `depth_anything_wrapper.py`, `midas_wrapper.py`, `rgbd_camera.py` | Monocular depth (Depth Anything V2 / MiDaS), sim depth fallback |
| `world_model/` | `object_db.py`, `spatial_graph.py` | Object database with CLIP re-identification, NetworkX spatial knowledge graph |
| `reid/` | `embedding_matcher.py` | CLIP-based embedding similarity for re-identifying objects across frames |
| `cross_camera/` | `matcher.py` | Multi-camera cross-identification |
| `tf_tree/` | `transform_tree.py` | Transform tree management |
| `configs/` | `config.py` | Vision configuration loader (intrinsics, model paths, thresholds) |
| `robot_controller.py` | — | Robot controller interface for vision system |

### `docs/plans/` — Development Plans

| File | Purpose |
|------|---------|
| `2026-05-26-master-plan.md` | Comprehensive master development plan across all phases |
| `2026-05-26-orchestrator-slam-nav2.md` | Detailed plan for Phase 2 (SLAM + Nav2 Orchestrator) |
| `2026-05-26-llm-orchestrator-harness.md` | Detailed plan for Phase 3 (LLM Orchestrator) |

---

## 🔧 How It Works — The Full Pipeline

### 1. Startup Sequence

```
1. rclpy.init()
2. Load physicalai_config.yaml (LLM model, Telegram token, detection prompt, persistence)
3. MapOrchestrator() creates ROS 2 node:
   a. NavLauncher.start_slam()     → subprocess: slam_toolbox online_async
   b. NavLauncher.start_nav2()     → subprocess: nav2_bringup navigation_launch.py
   c. Subscribes to: /camera/rgb/image_raw, /camera/depth/image_raw, /camera_info, /map, /scan, costmaps
   d. Loads Grounded SAM 2 + Depth Anything V2 on CUDA
   e. Starts 10Hz process timer, 20Hz safety timer, periodic detection
4. RobotInterface() created, wired to node's TF bridge, perception, exploration
5. 28 tools registered in ToolRegistry
6. LLMBridge() created → starts conversation loop in background thread
7. Web dashboard (FastAPI on port 8080) + Telegram gateway (optional)
8. rclpy.spin() on main thread
```

### 2. Detection Loop (10Hz)

```
Every frame:
  - Receive RGB + depth (sim depth or Depth Anything fallback)
  - Every DETECT_INTERVAL frames: run Grounded SAM 2
    → Open-vocabulary detection from text prompt (e.g. "sphere. shelf. table. chair...")
    → Returns bounding boxes + class names + confidence scores
  For each detection:
    - Sample depth at centroid (3×3 median, calibrated by _depth_scale)
    - Back-project to 3D camera frame: x=(u-cx)*d/fx, y=d, z=-(v-cy)*d/fy
    - TFBridge transforms: camera frame → map frame (handles optical conventions)
    - ObjectDB.add() with CLIP re-identification:
      → Computes embedding of RGB crop
      → Matches against known objects by cosine similarity
      → New object = new ID (e.g. chair_1), matched = position update
    - DriftMonitor checks for systematic position errors (semantic loop closure)
  - Annotate frame with bounding boxes + centroids
  - Save to output/frame_*.jpg every 30 frames
```

### 3. Navigation System

```
navigate_to(x, y, theta):
  1. Validate: map cell reachable? (snap to nearest free cell if needed)
  2. Send goal to Nav2 action server (/navigate_to_pose)
  3. Async callback monitors status
  4. SafetyMonitor runs in parallel (20Hz): checks laser scan forward sector
     → obstacle < 0.35m: publish zero-velocity immediately (hysteresis release at 0.45m)
  5. Goal completion → ToolResult(success, message, failure_type)

explore():
  1. Scan occupancy grid for frontier cells (unknown adjacent to free)
  2. Label frontier clusters via scipy.ndimage.label
  3. For nearest cluster: find closest FREE cell → navigate_to()
  4. Blocks until completion or timeout (120s)
```

### 4. LLM Orchestration

```
Conversation loop:
  1. Read user input (CLI stdin + Telegram + dashboard)
  2. Check for interrupt → cancel active chain
  3. Append to conversation history (max 40 messages)
  4. _process_loop(max_turns=8):
     a. _call_llm() → POST to Ollama API or OpenRouter
     b. Extract thinking from 7 formats (<thinking>, <reasoning>, // comments, etc.)
     c. Parse JSON response
     d. Dispatch:
          {"tool": "...", "args": {...}}   → execute tool → append result → continue
          {"reply": "..."}                 → display to user → end turn
          {"confirm": "..."}               → prompt user yes/no → continue
  5. Update dashboard + Telegram on each step
```

---

## 🛠️ Complete Tool Set

### Motion (7)

| Tool | Signature | Description |
|------|-----------|-------------|
| `navigate_to` | `(x: float, y: float, theta?: float)` | Drive to any map coordinate. Auto-snaps to nearest reachable cell if target is unmapped/blocked. |
| `navigate_to_object` | `(class_name: str)` | Find tracked object by class or ID, navigate to it. Asks user to disambiguate if multiple match. |
| `stop` | `()` | Cancel all Nav2 goals, halt immediately. |
| `go_home` | `()` | Return to map origin (0, 0, 0). |
| `rotate` | `(angle_deg: float)` | Spin in place (CCW positive). Uses direct /cmd_vel publishing. |
| `drive` | `(distance_m: float, speed?: float)` | Move forward/backward relative distance. |
| `wait` | `(seconds: float)` | Pause for N seconds (no motion). |

### Perception (5)

| Tool | Signature | Description |
|------|-----------|-------------|
| `detect_now` | `()` | Run detection on latest camera frame immediately. |
| `scan_surroundings` | `()` | 360° rotate + detect. Finds objects in all directions. |
| `search` | `(class_name: str)` | Rotate 360° searching for a specific object class. |
| `can_see` | `(class_name: str)` | Check if an object class is visible in the latest frame. |
| `refine_object` | `(class_name: str, repetitions?: int)` | Drive closer to an object and re-detect N times for better position accuracy. |

### Memory (7)

| Tool | Signature | Description |
|------|-----------|-------------|
| `list_objects` | `()` | All tracked objects with unique IDs, class, position, confidence. |
| `forget_object` | `(object_id: str)` | Remove a specific object by ID (e.g. 'chair_1'). |
| `forget_class` | `(class_name: str)` | Remove all objects of a given class. |
| `forget_all` | `()` | Wipe the entire object database. |
| `remember_place` | `(name: str)` | Save current robot pose as a named location. |
| `go_to_place` | `(name: str)` | Navigate to a previously saved place. |
| `list_places` | `()` | List all saved named places. |

### Knowledge (1)

| Tool | Signature | Description |
|------|-----------|-------------|
| `query_graph` | `(query: str)` | Query spatial relationships via NetworkX knowledge graph (e.g. "nearest object to robot", "objects within 2m of table"). |

### Introspection (3)

| Tool | Signature | Description |
|------|-----------|-------------|
| `ros2_introspect` | `(query: str)` | Read-only ROS 2 CLI: topic list, echo --once, node list, etc. Blocks: pub, run, lifecycle. |
| `discover_action` | `(action_name: str)` | Introspect an unknown ROS 2 action server at runtime. Returns type + goal/result/feedback fields. |
| `run_python` | `(code: str)` | Execute Python in sandbox with read-only access to robot state (pose, objects, map, np, math). |

### Detection Management (2)

| Tool | Signature | Description |
|------|-----------|-------------|
| `get_detection_prompt` | `()` | Return current Grounding DINO text prompt. |
| `set_detection_prompt` | `(prompt: str)` | Change what objects the camera looks for. |

### Validation (1)

| Tool | Signature | Description |
|------|-----------|-------------|
| `validate_path` | `(x: float, y: float)` | Check if (x, y) is reachable before attempting navigation. Returns CLEAR or BLOCKED with reason. |

### Temporal (2)

| Tool | Signature | Description |
|------|-----------|-------------|
| `query_history` | `(class_name: str, minutes_ago?: float)` | Position history of an object over time. |
| `has_object_moved` | `(class_name: str)` | Check if an object physically moved (delta > 0.2m). |

### Reliability (3)

| Tool | Signature | Description |
|------|-----------|-------------|
| `get_battery` | `()` | Battery status: percentage, estimated minutes, time-to-home, critical flag. |
| `calibrate_depth` | `(object_id: str, ground_truth_x: float, ground_truth_y: float)` | Calibrate depth scale from known object position. |
| `get_depth_calibration` | `()` | Read current depth scale factor and calibration samples. |

### File & Meta (3)

| Tool | Signature | Description |
|------|-----------|-------------|
| `read_project_file` | `(path: str)` | Read text files from ~/PhysicalAI/ (sandboxed, extension-filtered). |
| `create_skill` | `(name: str, description: str, code: str)` | Write and register a new Python tool at runtime. |
| `unregister_skill` | `(name: str)` | Remove a dynamically created skill (core tools protected). |

---

## 🔌 Three Entry Modes

### Mode 1: LLM TUI (Primary)

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
cd ~/PhysicalAI

# Terminal 1 — Gazebo simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2 — Orchestrator with LLM
python3 scripts/run_llm_orchestrator.py
```

→ Rich TUI console (rich panels, colored output, thinking display)
→ Optional Telegram bot for remote control
→ Web dashboard at `http://localhost:8080`

### Mode 2: MCP Server (for Hermes / Claude Desktop)

```yaml
# In ~/.hermes/config.yaml:
mcp_servers:
  physicalai:
    command: python3
    args: ["~/PhysicalAI/scripts/run_mcp_orchestrator.py"]
```

→ 24 tools exposed as JSON-RPC 2.0 over stdio
→ No TUI, no Telegram, no LLM bridge — just tools

### Mode 3: Headless (No LLM)

```bash
# Terminal 1 — Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2 — Headless orchestrator
python3 scripts/run_orchestrator.py
```

→ SLAM + Nav2 + detection only
→ ROS 2 services: `/list_objects`, `/go_to_nearest_object`, `/go_to_object_class`

---

## 📐 Configuration (`physicalai_config.yaml`)

```yaml
llm:
  provider: "ollama"              # "ollama" or "openrouter"
  model: "deepseek-v4-flash:cloud"
  api_base: "http://localhost:1196"
  api_key: ""
  temperature: 0.1

telegram:
  bot_token: ""                   # From @BotFather
  chat_id: ""                     # Your Telegram chat ID

detection:
  prompt: "sphere. shelf. table. chair. bed. human. fire hydrant. stop sign. box. cup. book. bottle. pot. trash can. furniture. sofa. desk. door. plant."
  hf_token: ""                    # HuggingFace token

robot:
  camera_frame: ""                # Auto-detect or specify frame name
  max_linear_speed: 3.0
  max_angular_speed: 2.0
  max_depth: 10.0
  persist_db: ""                  # SQLite path for persistent object DB (empty = in-memory)
```

---

## 🧠 ToolResult Failure Recovery

Every tool returns typed failures so the LLM can recover intelligently:

| Failure Type | Meaning | Recovery Strategy |
|---|---|---|
| `nav_blocked` | Path doesn't exist | Rotate 45°, retry. If still blocked: back up 1m, explore alternate route |
| `nav_timeout` | Goal didn't complete in time | Cancel goal, try intermediate waypoint at half distance |
| `nav_unavailable` | Nav2 server not ready | Wait 5s, retry. If persistent: ask user to check Nav2 status |
| `detect_empty` | Detection returned 0 results | Rotate 45°, retry up to 8 times. If still empty: move to nearest frontier with explore() |
| `depth_nan` | Depth at centroid is invalid | Expand sampling radius, retry with 3×3 or 7×7 patch median |
| `tf_timeout` | Camera→map transform unavailable | Wait 2s, retry. If persistent: check if SLAM is running via ros2_introspect() |
| `map_stale` | Map hasn't updated recently | SLAM may have stopped. Check node list, restart if needed |
| `goal_out_of_map` | Goal outside known map area | Explore() first to expand the map, then retry |
| `nav_busy` | Robot already has a goal | Wait for current goal to finish, or call stop() first |
| `internal_error` | Unexpected failure | Retry once. If persistent: report exact error to user |

**Safety rule:** Do NOT retry the same tool more than 3 times with the same parameters.

---

## 🛡️ Safety System

The `SafetyMonitor` provides hardware-level collision avoidance independent of the LLM:

- **20Hz check loop** scans LaserScan forward sector (±40°)
- **Emergency stop** triggers at 0.35m (force-publishes zero-velocity to `/cmd_vel`)
- **Hysteresis** releases at 0.45m (prevents oscillation)
- **Debounce** 0.5s between repeated stops
- Emergency count and closest-object stats exposed for dashboard

---

## 📦 Key Dependencies

| Component | Technology |
|-----------|-----------|
| ROS 2 Distro | Humble |
| SLAM | slam_toolbox (online_async) |
| Navigation | Nav2 (nav2_bringup) |
| Simulator | Gazebo Classic / Ignition Fortress / Isaac Sim |
| Detection | Grounding DINO + SAM 2 (via Grounded-SAM-2) |
| Depth | Depth Anything V2 / MiDaS |
| Re-ID | CLIP (OpenAI) |
| LLM API | Ollama (local) or OpenRouter (cloud) |
| Dashboard | FastAPI + uvicorn |
| Frontend | HTML/CSS/JS (inline) |
| TUI | rich + prompt_toolkit |
| Telegram | python-telegram-bot API (requests-based) |

---

## 🔮 Next Steps (Planned)

From the master development plan (`docs/plans/2026-05-26-master-plan.md`):

### Phase 4 — Adaptability & Unforeseen Tasks (FUTURE 📋)
- **Code generation** — LLM writes Python snippets to solve novel tasks no single tool covers
- **User-guided fallback** — `confirm()` asks humans when the LLM is stuck on unhandled scenarios
- **Multi-skill composition** — Combine existing tools/skills into complex behaviors (e.g. "follow that person")
- **Map image generation** — Return rendered occupancy grid as image for spatial reasoning

### Phase 5 — Advanced Reliability & Multi-Object Reasoning (ADDITIONAL 📋)
- **Uncertainty quantification** — Propagate depth/pose uncertainty through object positions
- **Multi-object identity tracking** — Handle occlusion, re-identification after lost tracking
- **Real battery monitoring** — Subscribe to /battery_state on physical robots (currently simulated drain at 3%/hour)
- **Dynamic action discovery → auto-registration** — Discover new ROS 2 action servers and register tools automatically
- **OBB-from-mask pipeline** — Extract oriented bounding boxes from SAM 2 segmentation masks by back-projecting to 3D point clouds
- **VLM description generation** — Use a vision-language model to generate semantic descriptions for each object ("red leather sofa")

### Incremental Improvements
- **Persistent skills** — Skills survive restarts (currently temp-files, lost on reboot)
- **Multi-camera system** — Cross-camera object handoff and tracking
- **Map stitching** — Merge multiple exploration sessions into a unified map
- **CI integration** — Automated simulation-based regression tests
- **ROS 2-native action servers** — Replace JSON tool-calling with ROS 2 action servers for production deployments

---

## 🐛 Historical Bugs & Fixes (Archived Knowledge)

| Bug | Cause | Fix |
|-----|-------|-----|
| Robot spins in place, declares success | Global costmap missing `static_layer` → planner sees empty world | Added `static_layer` subscribing to `/map` |
| Nav2 rejects frontier goals | Frontier centroid is in unknown cells (-1) | `find_exploration_goals()` returns nearest FREE cell instead |
| AMCL doesn't know robot position | No `/initialpose` published | `_publish_initial_pose_once()` at (0,0) after 5s |
| Explosion never triggers | `explore_next_frontier()` existed but nothing called it | Added `exploration_tick()` timer at 3s |
| Need 360° detected objects after rotation | No scan_surroundings() tool | Added 4-step rotate+detect scan |
| TF transform fails on Isaac Sim | Different camera frame conventions across simulators | TFBridge auto-detects optical vs non-optical frames |
| Exploration never stops | No LLM to decide when to stop | LLM evaluates map coverage + task completion |

---

## 📜 License & Credits

Developed as part of the VRWIT (NIRUN-Weerawit) robotics research project. Built on top of open-source components: ROS 2, Nav2, slam_toolbox, Grounded-SAM-2, Depth Anything V2, CLIP, NetworkX, FastAPI, and more.

See `docs/plans/2026-05-26-master-plan.md` for the full development roadmap and design rationale.