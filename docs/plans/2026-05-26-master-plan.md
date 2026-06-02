# PhysicalAI — Master Development Plan & Roadmap

> **Project:** Multi-camera vision + robotic orchestrator for autonomous mobile robots
> **Repo:** https://github.com/NIRUN-Weerawit/PhysicalAI
> **Last updated:** 2026-05-26

---

## How to read this document

Each phase represents a major milestone. ✅ = done, 🔧 = in progress, 📋 = planned.
The "Stage indicator" at the top of each phase shows current status: **CURRENT** (we're here), **NEXT** (next to start), or **FUTURE** (later).

---

## PHASE 0 — Foundation: Vision Pipeline (DONE ✅)

**Stage indicator:** ✅ COMPLETED

**Goal:** Single-camera live detection with 3D projection and object tracking.

### What was built

| File | Purpose |
|------|---------|
| `vision/detection/grounded_sam2_wrapper.py` | Grounding DINO + SAM 2 open-vocabulary detection |
| `vision/depth_estimation/depth_anything_wrapper.py` | Monocular depth estimation via Depth Anything V2 |
| `vision/world_model/object_db.py` | Temporal object tracking with 3s stale eviction |
| `vision/configs/config.py` | Configuration loader (intrinsics, depth scale, model paths) |
| `scripts/calibrate_intrinsics.py` | ChArUco / checkerboard intrinsic calibration |
| `scripts/calibrate_extrinsics.py` | ChArUco / ArUco extrinsic calibration |
| `scripts/calibrate_depth_scale.py` | Monocular depth scale calibration from known distances |
| `scripts/live_detection.py` | Live detection pipeline: Grounded SAM 2 → Depth Anything → 3D → ObjectDB |

### Key decisions
- Coordinate convention: x=right, y=forward (depth), z=up
- Depth scale calibrated via ChArUco board + known distances
- OpenCV 4.13+ new ArUco API (ArucoDetector / CharucoDetector)
- Grounding DINO + SAM 2 for open-vocabulary (not fixed classes)

### Known issues
- Depth Anything V2 at close range (<0.5m) gives noisy results
- Grounding DINO recall varies by prompt wording

---

## PHASE 1 — Simulation Backends (DONE ✅)

**Stage indicator:** ✅ COMPLETED

**Goal:** Run the vision pipeline inside Isaac Sim and Gazebo simulations.

### What was built

| File | Purpose |
|------|---------|
| `scripts/isaacsim_live_detection.py` | Grounded SAM 2 + Isaac Sim ground-truth RGB-D (no Depth Anything needed) |
| `scripts/isaacsim_calibrate_depth_scale.py` | Calibrate depth_scale using sim ground-truth object positions |
| `scripts/tb3_detection.py` | ROS 2 node: subscribe to TB3 camera + depth topics, run detection |
| `scripts/run_tb3_gazebo.py` | Launch Gazebo Classic + TB3 + spawn + detection (automated) |

### Key fixes discovered
- Isaac Sim `./python.sh` uses Python 3.10 with headless OpenCV → `cv2.imshow()` crashes, save frames to disk instead
- `~/` in Isaac Sim resolves to `/root/` → hardcode `/home/ucluser/PhysicalAI`
- numpy 2.x breaks Isaac Sim's bundled OpenCV → `pip install 'numpy<2' --force-reinstall`
- TB3 Waffle Pi has no depth publisher → fall back to Depth Anything V2 automatically

### Architecture lesson
Three sim backends discovered: Isaac Sim, Gazebo Classic, Ignition Fortress. Each has different:
- Camera API (direct `get_rgba()` vs ROS 2 topics)
- GUI support (`GUI: NONE` in Isaac Sim vs functional in plain Gazebo)
- Depth availability (Isaac Sim: always, Gazebo: depends on model SDF)

---

## PHASE 2 — SLAM + Nav2 Orchestrator (DONE ✅)

**Stage indicator:** ✅ COMPLETED

**Goal:** Tie together live SLAM mapping, Nav2 localization/navigation, and object detection so the robot explores unknown environments autonomously.

### What was built

```
orchestrator/
├── __init__.py                        # Package marker
├── launcher.py                        # Start/stop slam_toolbox + Nav2 subprocesses
├── tf_bridge.py                       # Camera frame → map frame via TF2
├── map_orchestrator.py                # Main ROS 2 node (~530 lines)
├── nav2_params_tb3.yaml               # Nav2 params (static layer fix applied)
scripts/
└── run_orchestrator.py                # Entry point
```

### MapOrchestrator capabilities

| Capability | Implemented | How |
|------------|-------------|-----|
| Live SLAM (no pre-built map) | ✅ | slam_toolbox online_async |
| Localization on live map | ✅ | Nav2 AMCL + static layer from /map |
| Frontier-based exploration | ✅ | 3s timer drives toward unknown space |
| Object detection (Grounded SAM 2) | ✅ | Every 3 frames |
| Depth estimation (sim or monocular) | ✅ | Sim depth > Depth Anything fallback |
| Camera→map frame transform | ✅ | TF lookup via tf_bridge.py |
| ObjectDB in map frame | ✅ | Persistent world coordinates |
| RViz goal visualization | ✅ | `/exploration_goal` topic |
| Manual goal services | ✅ | `/list_objects`, `/go_to_nearest_object` |

### Key bugs found and fixed

| Bug | Cause | Fix |
|-----|-------|-----|
| Robot spins in place, declares success | Global costmap missing `static_layer` → planner sees empty world | Added `static_layer` subscribing to `/map` |
| Nav2 rejects frontier goals | Frontier centroid is in unknown cells (-1) | `find_exploration_goals()` returns nearest FREE cell instead |
| AMCL doesn't know robot position | No `/initialpose` published | `_publish_initial_pose_once()` at (0,0) after 5s |
| Exploration never triggers | `explore_next_frontier()` existed but nothing called it | Added `exploration_tick()` timer at 3s |

### Current limitations
- Monolithic design (~530 lines, 12 responsibilities in one class)
- No explicit state machine (implicit flags: `_goal_active`, `_exploring_enabled`)
- No thread safety on shared state (ObjectDB + detections accessed from multiple timers)
- No health monitoring (if SLAM or Nav2 crash, robot doesn't know)
- Exploration stops when ANY object is in ObjectDB — cannot distinguish "found the target" vs "found something random"

---

## PHASE 3 — LLM Orchestrator Harness & Semantic Awareness (NEXT 📋)

**Stage indicator:** 🔜 NEXT — planned, not yet built

**Goal:** Replace the hardcoded state machine with an LLM that decides what to do, communicates with the user via chat, and calls robot capabilities as tools. Add semantic understanding: oriented bounding boxes for objects, spatial knowledge graph (graphification), and localization drift correction.

---

### 3A. Robustness Suite (Reinforcement against edge cases)

These features protect the system against common failure modes and make it resilient:

#### 3A.1 — `run_python(code)` tool (ad-hoc computing)

Allows the LLM to write arbitrary Python to solve unforeseen tasks:
```
User: "Is there 1m of space between the sofa and the wall?"

LLM:
  1. run_python("objects = robot.get_all_objects(); sofa = [o for o in objects if o.class_name=='sofa'][0]; ...")
  2. Returns: "The nearest wall is 0.8m north of the sofa — not enough space"
```

The sandbox has access to read-only robot state (pose, map, ObjectDB, spatial graph). No network, no file writes, no ros2 commands.

#### 3A.2 — Progressive Refinement Loop

Single-view detection is unreliable. The system re-detects from closer range:
```
detect("sofa") → confidence=0.35, pos=(3.2, 1.5) from 3m away
navigate_to(3.0, 1.3)  ← approach it
detect("sofa") → confidence=0.82, pos=(3.4, 1.7) from 1m away  ← better!
update_objectdb("sofa", pos=(3.4, 1.7), conf=0.82)
```

Implementation: `refine_object(class_name)` — drives closer, re-detects N times, averages positions weighted by confidence, updates ObjectDB. Returns refined position + confidence delta.

#### 3A.3 — Structured Failure Catalog

Tools return typed failure codes, not just "failed". The LLM's system prompt includes a recovery table:

| Failure type | Meaning | Recovery strategy |
|---|---|---|
| `nav_blocked` | Nav2 can't find path | Rotate 90°, retry. If still blocked: back up, explore alternate route |
| `nav_timeout` | Goal didn't complete in N seconds | Cancel goal, try intermediate waypoint at half distance |
| `detect_empty` | Detection returned 0 results | Rotate 45°, retry. Repeat up to 360°. If still empty: move to nearest frontier |
| `depth_nan` | Depth at centroid is invalid (edge of object) | Expand sampling radius, retry with 7×7 patch median |
| `tf_timeout` | Camera→map transform unavailable | Wait 1s, retry. If persistent: check if SLAM is still running |
| `map_stale` | /map hasn't updated in >30s | SLAM may have crashed → restart slam_toolbox process |
| `goal_out_of_map` | Goal coordinate is outside known map bounds | Explore towards goal direction first, then retry |

#### 3A.4 — Simulation-First Validation

Before executing any motion, the system checks feasibility:
```
LLM says: navigate_to(3.2, 1.5)
System checks:
  1. Is (3.2, 1.5) inside the known map?
  2. Is that cell "free" (value 0) in the global costmap?
  3. Does Nav2 produce a non-empty /plan?
If any check fails → return failure + reason, ask LLM to adjust
```

Implementation: `validate_path(x, y)` — queries costmap and Nav2 planner, returns "PATH CLEAR" or "PATH BLOCKED AT (x, y) — try alternate approach".

#### 3A.5 — Duration Estimation

Every tool returns an estimated time so the LLM manages user expectations:
- `explore()` → "Estimated 2-5 minutes (depends on environment size)"
- `navigate_to(10.0, 5.0)` → "Approximately 40 seconds at 0.26 m/s"
- `search("bottle")` → "Up to 30 seconds for 360° rotation"
- `scan_surroundings()` → "Approximately 25 seconds for full 360° scan"

---

### 3B. Localization Drift Correction — Semantic Loop Closure

**Problem:** Live SLAM drifts over time. Odometry accumulates error. After 5 minutes, the map may have a 0.3-0.5m offset. Objects detected early are at "wrong" map coordinates relative to later detections.

**Solution:** Use detected objects as **semantic landmarks** for loop closure.

#### How it works

```
Step 1 — First detection:
  detect("sofa") → stores "sofa_001" at map position (3.2, 1.5)
  camera position was (2.8, 1.2, θ=30°)
  record observation: {object_id, map_pos, camera_pose, timestamp}

Step 2 — Robot explores, drifts ~0.3m

Step 3 — Re-detection from a different angle:
  detect("sofa") → detects what Grounding DINO thinks is a sofa
  current camera pose (4.1, 2.8, θ=60°) → position: (3.5, 1.8)
  But we already have "sofa_001" at (3.2, 1.5)

Step 4 — Discrepancy detected:
  Two sofa observations are 0.3m apart → could be SLAM drift
  Compute: expected position = re-project sofa_001 using current camera pose
  Observed position = what detection says
  Error = 0.3m → likely drift

Step 5 — Trigger correction:
  Call slam_toolbox's /deserialize_map or /save_map + /load_map
  Or publish a corrected /initialpose to re-anchor AMCL
  Flag all ObjectDB entries as "drift_corrected_timestamp"
```

#### Implementation

```python
class DriftMonitor:
    def __init__(self):
        self.landmark_observations = {}  # object_id → [(camera_pose, detected_pos), ...]

    def check_landmark(self, class_name: str, new_pos: tuple, camera_pose: tuple):
        """Check if this object was seen before from a different angle."""
        existing = self.db.find_by_class(class_name)  # fuzzy match
        if not existing:
            return  # new object, nothing to compare

        expected = self.reproject(existing.position_world, camera_pose)
        error = euclidean(new_pos, expected)

        if error > 0.3:  # meters — significant drift
            self.trigger_correction(source_pos=existing.position_world,
                                    observed_pos=new_pos,
                                    error=error)
```

A ROS 2 `DriftMonitor` timer runs every 5 seconds. After 3 consistent observations of the same object at different positions with the same drift direction, it publishes a corrected pose to `/initialpose` or triggers slam_toolbox's serialization.

---

### 3C. Semantic Object Modeling — Beyond Centroid

**Problem:** The current detection only stores the 2D bounding box centroid as the object's 3D position. A sofa that's 2m wide gets reduced to a single point at its center. The LLM has no idea how big objects are, where their edges are, or what direction they face.

**Solution:** Oriented Bounding Boxes (OBB) from SAM 2 segmentation masks.

#### Pipeline

```
1. Grounding DINO → bounding box + SAM 2 prompt
2. SAM 2 → per-pixel segmentation mask (H×W boolean)
3. For each "object" pixel in the mask:
     - Get (u, v) pixel coordinates
     - Sample depth at (u, v) from depth map
     - Back-project to 3D: (x, y, z) = project_to_3D(u, v, depth)
     → Object 3D Point Cloud (N×3 array)
4. Fit Oriented Bounding Box:
     - Compute 2D convex hull of point cloud projected to ground plane (x-z)
     - cv2.minAreaRect(hull) → (center_x, center_z, width, depth, angle)
     - height = max(z) - min(z) of raw point cloud
5. ObjectDB stores:
     {
       "centroid": (x, y, z),          # 3D center
       "obb": {
           "center_2d": (x, z),         # ground-plane center
           "width": 1.8,                # meters along longer axis
           "depth": 0.9,                # meters along shorter axis  
           "height": 0.75,              # meters vertically
           "angle": 45.0,               # degrees from map x-axis
       },
       "mask_area": 0.65,              # fraction of image covered by mask
       "point_count": 3421,            # number of valid 3D points
     }
```

#### What the LLM gets from this

```
User: "Is there space for a chair between the sofa and the wall?"

LLM:
  1. query_graph("sofa") → OBB = center(3.2, 1.5), width=1.8m, depth=0.9m, angle=45°
  2. query_graph("nearest wall to sofa") → wall at (3.2, 3.0)
  3. run_python("""
        sofa_center = (3.2, 1.5)
        sofa_width = 1.8
        sofa_angle = 45  # degrees
        wall = (3.2, 3.0)
        # Compute distance from sofa's back edge to wall
        # sofa back is at center + (0.9/2)*cos(45°) in the direction away from wall
        gap = compute_edge_to_wall(sofa, wall)
        return f"Gap is {gap:.2f}m — {'enough' if gap > 0.6 else 'not enough'} for a chair"
     """)
  4. say("There is 0.85m of space between the sofa's back edge and the wall. That's enough for a chair.")
```

#### OBB implementation plan

| Step | Code |
|------|------|
| 1. Get mask | `mask = sam2_predict(image, bbox_xyxy)` — H×W bool |
| 2. Back-project | `for (u,v) in mask_pixels: d = depth_map[v,u]; pts.append( (u-cx)*d/fx, d, -(v-cy)*d/fy )` |
| 3. Ground plane | `pts_2d = pts[:, [0, 2]]` — x (right) and z (forward) axes |
| 4. Convex hull + OBB | `cv2.minAreaRect(np.int0(pts_2d * 100))` — returns center, size, angle |
| 5. Metadata | Store area, point count, timestamp |

---

### 3D. Graphification — Spatial Knowledge Graph

**Problem:** The LLM gets a raw text list of objects. It doesn't understand "the chair is next to the table", "the table is against the wall", "the bottle is on the table". These relationships must be **computed from raw positions**, not typed by the user.

**Solution:** A `SpatialGraph` built from ObjectDB using NetworkX, queried with a graph query language.

#### Graph structure

```
Nodes:
  - Robot (current pose, timestamp)
  - Object_001 {class: "sofa", obb: {...}, last_seen: timestamp}
  - Object_002 {class: "table", obb: {...}, last_seen: timestamp}
  - Object_003 {class: "door", position: (4.0, 0.0), last_seen: timestamp}
  - Place_001  {name: "home_base", position: (0.0, 0.0)}
  - Region_001 {type: "room", bounds: polygon}

Edges (computed dynamically, refreshed every time graph is queried):
  - spatial_near(A, B) — distance between centroids < threshold
  - spatial_adjacent(A, B) — A's OBB edge touches or overlaps B's OBB
  - spatial_contains(A, B) — B is inside A's OBB (e.g., bottle on table)
  - spatial_left_of, right_of, in_front_of — relative to robot's current heading
  - spatial_co_visible(A, B) — were A and B seen in the same camera frame?
  - temporal_seen_together(A, B) — were A and B detected in the same detect() call?
```

#### Query interface

```python
class SpatialGraph:
    def add_object(self, obj: ObjectRecord) -> None: ...
    def update_robot_pose(self, pose: tuple) -> None: ...
    def add_place(self, name: str, position: tuple) -> None: ...

    def query(self, expression: str) -> list[dict]:
        """
        Examples:
          "nearest object to robot"
          "objects within 2m of 'table'"
          "largest object in room"
          "objects on top of 'table'" (centroid_2d inside table's xy bounding box)
          "what is to the left of the robot"
          "objects seen together with 'door'"
        """
```

The LLM can call `query_graph("nearest objects to the sofa")` and get back structured results:

```json
[
  {"class": "coffee table", "distance": 0.45, "relation": "in_front_of"},
  {"class": "lamp", "distance": 1.20, "relation": "left_of"},
  {"class": "wall", "distance": 0.80, "relation": "behind"}
]
```

#### Implementation approach (NetworkX)

```python
import networkx as nx
import numpy as np

class SpatialGraph:
    def __init__(self):
        self.G = nx.Graph()
        self._last_build_time = 0

    def _rebuild_edges(self):
        """Recompute all spatial edges from current object positions."""
        objects = self.db.get_all()
        self.G.clear_edges()

        for i, a in enumerate(objects):
            for b in objects[i+1:]:
                dist = np.linalg.norm(
                    np.array(a.position_world[:2]) - np.array(b.position_world[:2])
                )
                if dist < 3.0:  # within 3m → near
                    self.G.add_edge(
                        a.object_id, b.object_id,
                        relation="spatial_near", distance=round(dist, 3)
                    )
                # Check adjacency (OBB overlap)
                if self._obbs_overlap(a, b):
                    self.G.add_edge(
                        a.object_id, b.object_id,
                        relation="spatial_adjacent", distance=0.0
                    )
                # Check containment (b is inside a's OBB)
                if self._contains(a, b):
                    self.G.add_edge(
                        a.object_id, b.object_id,
                        relation="spatial_contains"
                    )
```

#### Why this helps the LLM

Without the graph:
```
LLM received: table(1.0,2.3), chair(1.2,2.5), bottle(1.1,2.4)
LLM must GUESS: are these near each other? Is the bottle on the table?
```

With the graph:
```
LLM calls query_graph("what is near the table")
Returns: chair (0.45m right), bottle (0.20m, possibly on top)
LLM: "The chair is 0.45m to the right of the table. The bottle appears to be on the table."
```

This eliminates hallucination about spatial relationships and gives the LLM a factual basis for decisions.

---

### 3E. Full Tool Registry (RobotInterface)

All tools accessible to the LLM:

#### Motion

| Tool | Signature | Description |
|------|-----------|-------------|
| `navigate_to` | `(x: float, y: float, theta?: float)` | Drive to map coordinate. Returns path status. |
| `navigate_to_object` | `(class_name: str)` | Find object in DB, navigate to it. |
| `stop` | `()` | Cancel all Nav2 goals, halt robot immediately. |
| `rotate` | `(angle_deg: float)` | Spin in place N degrees (positive = CCW). |
| `go_home` | `()` | Navigate to map origin (0, 0). |
| `wait` | `(seconds: float)` | Pause for N seconds (no motion). |

#### Perception

| Tool | Signature | Description |
|------|-----------|-------------|
| `detect_now` | `()` | Force Grounded SAM 2 detection immediately, bypassing timer. |
| `search` | `(class_name: str)` | 360° rotate + detect. Stops when object is found. |
| `scan_surroundings` | `()` | 360° rotate + detect everything. Returns all objects. |
| `refine_object` | `(class_name: str, repetitions?: int)` | Drive closer, re-detect N times, update ObjectDB+OBB. |
| `can_see` | `(class_name: str)` | Is this object in the most recent detection results? |

#### Knowledge & Memory

| Tool | Signature | Description |
|------|-----------|-------------|
| `list_objects` | `()` | All ObjectDB entries with class, position, confidence, OBB. |
| `list_places` | `()` | All named places (remembered positions). |
| `query_graph` | `(query: str)` | Natural-language spatial query ("nearest to table", "objects on sofa"). |
| `remember_place` | `(name: str)` | Store current robot pose as a named location. |
| `go_to_place` | `(name: str)` | Navigate to a remembered place. |
| `run_python` | `(code: str)` | Execute Python in sandbox with read-only robot state access. |

#### Status & Interaction

| Tool | Signature | Description |
|------|-----------|-------------|
| `get_status` | `()` | Full health report: pose, map coverage %, objects tracked, SLAM health, Nav2 state. |
| `get_pose` | `()` | Current (x, y, theta) in map frame from TF. |
| `get_map_coverage` | `()` | Percentage of known vs explored area on the map. |
| `say` | `(message: str)` | Print message to user console with `[Orchestrator]` prefix. |
| `confirm` | `(message: str)` | Ask yes/no, **block** until user responds. Used for safety-critical decisions. |

| `ros2_introspect` | `(query: str)` | Execute read-only ROS 2 CLI command. Allowed: `topic list, node list, echo --once, info`. Blocked: `pub, run, lifecycle`. Timeout: 5s. Output cap: 2000 chars. |

#### Validation & Safety

| Tool | Signature | Description |
|------|-----------|-------------|
| `validate_path` | `(x: float, y: float)` | Check if (x,y) is reachable. Returns "CLEAR" or "BLOCKED at (...)". |
| `emergency_stop` | `()` | Hard brake — cancel goals, stop all motors, set state to "paused". |

---

### 3H. Dynamic Skill Registration — On-the-Fly Tool Creation

**Problem:** No matter how many tools we pre-build, the LLM will encounter situations where the needed capability doesn't exist. Example: "speed up to get through the elevator before it closes" — but there's no `set_velocity()` tool. The LLM needs to create that tool at runtime without restarting the system.

**Solution:** The LLM can author new Python skill files using the `create_skill` tool, and the system immediately registers them as callable functions.

#### How it works

```
1. LLM encounters a missing capability:
   "I need set_velocity(speed_ms) but it's not in my tool list"

2. LLM calls create_skill to author a new skill file:
   create_skill(
     name="set_velocity",
     description="Set the robot's maximum linear velocity in m/s (0.05–0.26)",
     code="""
import rclpy
from geometry_msgs.msg import Twist

def set_velocity(speed_ms: float) -> dict:
    \"\"\"Set the robot's maximum linear velocity.\"\"\"
    speed_ms = max(0.05, min(0.26, speed_ms))
    # Write to /cmd_vel with max speed
    node = rclpy.create_node('vel_setter')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    msg = Twist()
    msg.linear.x = speed_ms
    # Store in robot's persistent state
    robot._max_speed = speed_ms
    return {"status": "success", "new_max_speed": speed_ms}
"""
   )

3. System writes /tmp/physicalai_skills/skill_set_velocity.py
   → Imports the file
   → Registers set_velocity in the tool registry
   → LLM can now call: set_velocity(0.5)

4. LLM: "set_velocity(0.5)" → "Max speed set to 0.5 m/s. Getting through the elevator quickly."
```

#### Safety guardrails

| Guardrail | Rule |
|---|---|
| **File destination** | Skills written to `/tmp/physicalai_skills/skill_<name>.py` only — never to the project directory |
| **Code sandbox** | The code runs in a subprocess with timeout (10s) — can't hang the orchestrator |
| **Review before execute** | `confirm()` is called before first execution: "I'm about to create a new skill: set_velocity. It will publish to /cmd_vel. Approve?" |
| **Kill switch** | `unregister_skill(name)` removes a skill from the registry instantly |
| **Session-only** | Skills are in-memory + temp files — lost on restart. To persist, LLM explicitly calls `save_skill(name)` |
| **No overwrite** | Can't unregister or override core tools (`stop`, `navigate_to`, `emergency_stop` — protected) |
| **Max 5 dynamic skills** | Prevents the LLM from flooding the system with hundreds of broken skills |

#### The two new tools

| Tool | Signature | Description |
|---|---|---|
| `create_skill` | `(name: str, description: str, code: str)` | Write a new Python file, import it, register it as a callable tool. Calls `confirm()` before registering. |
| `unregister_skill` | `(name: str)` | Remove a dynamically registered skill from the tool registry. Does NOT affect core tools. |

#### Implementation

```python
# orchestrator/llm_tools.py

SKILL_DIR = "/tmp/physicalai_skills"

def create_skill(name: str, description: str, code: str) -> ToolResult:
    """Create and register a new tool from LLM-authored code."""

    # 1. Validate name (no overwriting core tools)
    if name in PROTECTED_TOOLS:
        return ToolResult(False, f"Cannot override protected tool: {name}")

    # 2. Write to temp file
    os.makedirs(SKILL_DIR, exist_ok=True)
    skill_path = os.path.join(SKILL_DIR, f"skill_{name}.py")
    with open(skill_path, 'w') as f:
        f.write(code)

    # 3. Import and extract the function
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"skill_{name}", skill_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, name, None)
    if fn is None:
        return ToolResult(False, f"Skill file created but no function '{name}' found in it.")

    # 4. Register in the live tool registry
    llm_tools.register(name, description, fn)

    return ToolResult(True, f"Skill '{name}' registered and ready to use.")
```

#### Example lifecycle

```
Session 1: Robot explores a building, encounters an elevator
  LLM: "I don't have a way to set speed. Let me create one."
  → create_skill("set_velocity", "...", "def set_velocity(speed_ms): ...")
  → confirm("Creating a new behavior: set_velocity. Publish to /cmd_vel?")
  → User: "Yes"
  → LLM: set_velocity(0.5)
  → Robot speeds up, enters elevator

Session 2 (next day, robot restarted):
  LLM: "I remember set_velocity from yesterday, but it's not available."
  LLM: "Checking skill history in /tmp/physicalai_skills/..."
  LLM: "Found skill_set_velocity.py from yesterday. Re-registering."
  → load_skill("set_velocity") → ready again
```

#### Why this matters

Without dynamic skill registration, the robot has a fixed set of capabilities. It hits a wall: "I can't do X because nobody programmed it." With it, the LLM can **extend itself** at runtime. This is the difference between a tool-calling agent and a **self-augmenting agent**.

The key insight: the LLM doesn't just call tools — it **writes** them. When it encounters a missing capability, it authors the code, registers it, and uses it — all within the same session.

---

### 3I. Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `orchestrator/robot_interface.py` | **NEW** | Layer 3: Clean command/query interface wrapping all capabilities |
| `orchestrator/llm_tools.py` | **NEW** | Tool registry: name, description, JSON schema, handler for each tool |
| `orchestrator/llm_bridge.py` | **NEW** | LLM client (Ollama/OpenRouter) + conversation loop + tool-calling parser |
| `orchestrator/llm_orchestrator.py` | **NEW** | Entry point: launches MapOrchestrator + LLM Bridge |
| `vision/world_model/spatial_graph.py` | **NEW** | NetworkX graph builder + query engine |
| `orchestrator/drift_monitor.py` | **NEW** | Semantic loop closure: detect + correct SLAM drift |
| `vision/world_model/object_db.py` | **MODIFY** | Add OBB fields (width, depth, height, angle, mask_area, point_count) |
| `orchestrator/map_orchestrator.py` | **MODIFY** | Extract logic to RobotInterface, add stop/get_pose/get_status |

---

### 3G. Phase 3 build order (implementation sequence)

```
Step 1:  Refactor MapOrchestrator → extract RobotInterface (clean separation)
Step 2:  Add stop(), get_pose(), get_status(), go_home() to RobotInterface
Step 3:  Add detect_now(), scan_surroundings(), search() to RobotInterface
Step 4:  Upgrade ObjectDB → store OBB from SAM 2 masks + 3D back-projection
Step 5:  Build DriftMonitor → semantic loop closure for localization
Step 6:  Build SpatialGraph → NetworkX graph + query_graph() tool
Step 7:  Build llm_tools.py → tool registry + failure catalog
Step 8:  Build llm_bridge.py → LLM conversation loop + tool parser
Step 9:  Build llm_orchestrator.py → entry point for full LLM stack
Step 10: Test: manual commands → LLM → tool execution → user report back
```

---

## PHASE 4 — Adaptability & Unforeseen Tasks (FUTURE 📋)

**Stage indicator:** 🔮 FUTURE — design phase

**Goal:** The system should handle tasks it was never explicitly programmed for, by composing existing skills, asking the user for help, or learning from experience.

### Why this matters

No matter how many tools we pre-define, users will ask things like:
- "Is there enough space for a chair between the sofa and the wall?"
- "Follow that person without losing them"
- "Check if the door is open"
- "Count how many boxes are in the room"
- "Map this floor, then come back and show me a top-down view"

### Strategy 1: Code generation

The LLM generates Python snippets to compute things no single tool provides:
```
User: "Is there 1m of space between the sofa and the wall?"

LLM:
  1. get_map_image() → returns occupancy grid as PIL/array  
  2. Generate code: measure distance between sofa OBB and nearest occupied cell
  3. Execute code → return result
```

### Strategy 2: Ask the user

When the LLM genuinely doesn't know, it should say so using `confirm()`:
```
User: "Follow the red ball"

LLM: "I can detect red balls — but I don't have a continuous 'follow' behavior.
      I CAN rotate to keep the ball in frame and drive toward it. Should I try that?"
```

---

## Phase 5 — Advanced Reliability & Multi-Object Reasoning (ADDITIONAL 📋)

**Stage indicator:** 📋 ADDITIONAL — post-core, not yet planned in detail

**Goal:** Address remaining gaps that limit the system's reliability in complex environments: temporal reasoning, uncertainty quantification, multi-object identity tracking, battery awareness, and dynamic ROS 2 introspection.

---

### 5A. Temporal Queries — "Where was it 5 minutes ago?"

**Problem:** ObjectDB only stores the latest state. The LLM can't ask "where was the sofa before the robot moved?" or "has the bottle moved since last observation?"

**Solution:** Time-series in ObjectDB. Every observation is stored with a timestamp. A `query_history(class_name, time_range)` tool returns positions at past points in time.

```python
def query_history(class_name: str, minutes_ago: float) -> list[dict]:
    # Returns [{"timestamp": t1, "position": (x1,y1,z1)}, 
    #          {"timestamp": t2, "position": (x2,y2,z2)}, ...]
```

Use case: "Did the chair move since I entered the room?" → compare last two positions. If delta > 0.2m, the chair was moved.

---

### 5B. Uncertainty Propagation — "How sure are you?"

**Problem:** A detection at confidence=0.55 from 4m away at a steep angle is treated the same as confidence=0.99 from 1m away. The system is overconfident on noisy data.

**Solution:** Every position stored in ObjectDB carries a covariance matrix:

```json
{
  "position": (3.2, 1.5, 0.0),
  "uncertainty": {
    "xy_std": 0.35,    // meters — grows with distance, shrinks with confidence
    "z_std": 0.12,
    "covariance": [[0.12, 0.01], [0.01, 0.08]]
  },
  "detection_quality": {
    "distance_to_object": 3.8,  // meters — closer = better
    "viewing_angle": 60,        // degrees off-axis — lower = better  
    "depth_valid_ratio": 0.72   // fraction of mask pixels with valid depth
  }
}
```

The SpatialGraph uses uncertainty to answer: "the sofa is at (3.2, 1.5) ± 0.35m — it might be up to 0.7m away from that point."

The LLM should avoid making precision-dependent decisions on high-uncertainty objects.

---

### 5C. Multi-Object Re-Identification — "Which chair is which?"

**Problem:** If the room has two identical chairs, ObjectDB creates `chair_001` and `chair_002`. But when re-detecting, it can't tell which is which — the LLM says "go to the chair" and the robot doesn't know which one.

**Solution:** Re-identification via position + OBB + appearance embedding.

When a new detection comes in, the system asks:
1. Is there an existing object of the same class within position uncertainty?
2. Does the new OBB (size, angle) match the existing OBB?
3. Optionally: does a feature vector from SAM 2's image encoder match?

If multiple candidates match → spawn new object (it's a new instance).
If one candidate matches well → update existing object (same instance).

```python
class ObjectDB:
    def reidentify(self, detection: Detection) -> str:
        """
        Match a detection to an existing object or create new.
        Returns the matched/new object_id.
        """
        candidates = self.find_by_class(detection.class_name)
        
        best_match = None
        best_score = 0.0
        
        for obj in candidates:
            pos_dist = detection.position - obj.position_world
            obb_similarity = iou_obb(detection.obb, obj.obb)
            
            score = 1.0 / (1.0 + pos_dist) * 0.6 + obb_similarity * 0.4
            
            if score > best_score and score > MATCH_THRESHOLD:
                best_match = obj
                best_score = score
        
        if best_match:
            return best_match.object_id  # Re-identified
        else:
            return self._create_new_object(detection)  # New instance
```

---

### 5D. Battery & Energy Awareness

**Problem:** No tool answers "how much battery is left?" The robot might be mid-exploration 20m from home with 5% battery.

**Solution:** `get_battery()` tool subscribes to `/battery_state` if available (real robot) or estimates from simulation time.

```python
def get_battery() -> dict:
    return {
        "percentage": 73.2,
        "voltage": 11.8,
        "estimated_minutes_remaining": 18.5,
        "time_to_home": 120,  # seconds — computed from distance to (0,0) / speed
        "critical": False,    # True if < 10%
        "should_return": False  # True if minutes_remaining < time_to_home * 1.5
    }
```

The LLM's system prompt includes a rule: "If `should_return` is True, navigate home immediately and notify the user. Do not start new tasks."

---

### 5E. Dynamic ROS 2 Action Discovery

**Problem:** `ros2_introspect` can list available ROS 2 actions, but the LLM can only call pre-registered ones. If a new action server appears (e.g., `/inspect_area`), the LLM sees it but can't use it.

**Solution:** `discover_action(action_name)` tool introspects an unknown action, creates a dynamic tool wrapper:

```python
def discover_action(action_name: str) -> dict:
    """
    Introspect the action type from the ROS 2 graph.
    Generate and register a new tool that sends goals to this action.
    
    Example: discover_action("/inspect_area")
    → Registers: inspect_area(location_x, location_y, duration_sec)
    → Returns: "Registered /inspect_area as a new tool."
    """
    # ros2 action info /inspect_area → find action type
    # ros2 interface show action_type → get goal fields
    # Generate wrapper function based on fields
    # Register in tool registry
```

Safety: `confirm()` before first use. Only works with action servers that follow standard patterns.

---

### 5F. Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `vision/world_model/object_db.py` | **MODIFY** | Add time-series history, covariance/uncertainty, re-identification logic |
| `orchestrator/llm_tools.py` | **MODIFY** | Add `query_history`, `get_battery`, `discover_action` tools |

---

## Phase 4 — UI Layer: CLI Chat + Telegram Gateway (NEXT 📋)

**Stage indicator:** 🔜 NEXT — after Phase 3 tools are built

**Goal:** Give the user two channels to command the robot and receive updates: terminal CLI and Telegram messenger. The LLM's `say()` output is routed to both. User can send commands from anywhere.

---

### 4A. CLI Chat

The conversation loop in `llm_orchestrator.py` already reads from stdin and writes to stdout. This is the baseline that works immediately — no extra code needed for Phase 3.

### 4B. Telegram Gateway

**How it works:**
1. A Telegram bot is connected to the running orchestrator process
2. User messages to the bot are injected into the LLM conversation loop as if typed in the CLI
3. The LLM's `say()` output is sent back to the Telegram chat
4. Annotated camera frames are sent as Telegram photos

**Implementation:** The gateway is lightweight — we don't build a full Telegram client from scratch. Instead, we use a simple polling-based approach or a minimal `python-telegram-bot` wrapper script that:

```python
# orchestrator/telegram_gateway.py — simplified design

import requests, time, threading

class TelegramGateway:
    """Poll Telegram for user messages, forward to LLM loop."""

    def __init__(self, token: str, chat_id: str, message_queue: Queue):
        self._token = token
        self._chat_id = chat_id
        self._queue = message_queue  # incoming user messages
        self._last_update_id = 0
        self._bot_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, text: str):
        """Called by the LLM's say() — sends to Telegram."""
        requests.post(f"{self._bot_url}/sendMessage", json={
            "chat_id": self._chat_id, "text": text
        })

    def send_photo(self, image_path: str):
        """Send an annotated camera frame."""
        with open(image_path, "rb") as f:
            requests.post(f"{self._bot_url}/sendPhoto",
                          data={"chat_id": self._chat_id},
                          files={"photo": f})

    def poll(self):
        """Background thread: poll for new messages → push to LLM queue."""
        while True:
            resp = requests.get(f"{self._bot_url}/getUpdates",
                                params={"offset": self._last_update_id + 1,
                                        "timeout": 10})
            for update in resp.json().get("result", []):
                text = update.get("message", {}).get("text", "")
                uid = update["update_id"]
                self._last_update_id = uid
                if text:
                    self._queue.put(("[telegram]", text))
            time.sleep(0.5)
```

### 4C. Dual-channel architecture

```
User (Telegram) ──→ TelegramGateway.poll() ──→ message_queue ──→ LLM Loop
User (CLI stdin) ───────────────────────────────────────────→ LLM Loop

LLM Loop → say() → TelegramGateway.send_message() + print(stdout)
                 → TelegramGateway.send_photo()  + save_to_disk
```

Both channels feed into the same LLM conversation loop. The LLM doesn't need to know the source — it just reads from the queue and sends output via `say()`.

### 4D. CLI enhancements for the chat terminal

- **Colored output:** robot messages in green, errors in red, detection info in yellow  
- **History:** up/down arrows recall previous commands (`readline` module)
- **Status bar:** persistent display of robot state (pose, objects, mode) without cluttering the chat

These are purely cosmetic but improve the operator experience significantly.

---

## Phase 5 — Web Dashboard (FUTURE 📋)

**Stage indicator:** 🔮 FUTURE — after Telegram gateway is proven

**Goal:** A browser-based dashboard showing live video feed, map with object overlays, and a chat panel.

**Tech stack:** FastAPI (backend) + vanilla HTML/CSS/JS (frontend). No React, no build step — a single `dashboard.html` served by FastAPI.

### 5A. Live camera feed (MJPEG stream)

```python
@app.get("/video_feed")
def video_feed():
    """MJPEG stream: serve latest annotated frame continuously."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

The frontend: `<img src="/video_feed">` — a plain HTML img tag. Works everywhere, zero JS needed.

### 5B. Map with objects

```python
@app.get("/map")
def get_map():
    """Return occupancy grid + object positions as JSON."""
    return {
        "map_png": map_as_base64,
        "objects": [{"class": o.class_name, "x": o.x, "y": o.y} for o in db.get_all()],
        "robot_pose": (x, y, theta),
    }
```

Frontend renders the map as a `<canvas>` overlay with dots for objects and an arrow for the robot.

### 5C. Chat panel

WebSocket endpoint that relays user ↔ LLM messages. Same queue-based design as Telegram gateway but with a websocket client instead of HTTP polling.

### Files to create

| File | Purpose |
|------|---------|
| `orchestrator/dashboard_server.py` | FastAPI app serving video, map, chat |
| `orchestrator/static/dashboard.html` | Single-page dashboard (inline CSS/JS) |
| `orchestrator/telegram_gateway.py` | Already created in Phase 4 |

---

## Full file inventory

```
PhysicalAI/
├── config.json
├── README.md                            # Setup guide
├── .gitignore                           # Ignored files
│
├── vision/
│   ├── configs/
│   │   └── config.py                    # Configuration loader
│   ├── detection/
│   │   ├── __init__.py
│   │   └── grounded_sam2_wrapper.py     
│   ├── depth_estimation/
│   │   ├── __init__.py
│   │   ├── base.py                      
│   │   └── depth_anything_wrapper.py    
│   └── world_model/
│       ├── __init__.py
│       ├── object_db.py                 # ObjectDB with temporal tracking
│       └── spatial_graph.py             # 🔜 NetworkX graph builder
│
├── orchestrator/
│   ├── __init__.py
│   ├── launcher.py                      # NavLauncher (SLAM + Nav2 lifecycle)
│   ├── tf_bridge.py                     # Camera→map TF transform
│   ├── map_orchestrator.py              
│   ├── nav2_params_tb3.yaml             
│   └── robot_interface.py              # 🔜 Layer 3 (Phase 3)
│   └── llm_tools.py                    # 🔜 Tool registry (Phase 3)
│   └── llm_bridge.py                   # 🔜 LLM conversation loop (Phase 3)
│   └── llm_orchestrator.py             # 🔜 Main LLM entry point (Phase 3)
│
├── scripts/
│   ├── live_detection.py                
│   ├── calibrate_intrinsics.py          
│   ├── calibrate_extrinsics.py          
│   ├── calibrate_depth_scale.py         
│   ├── isaacsim_live_detection.py       
│   ├── isaacsim_calibrate_depth_scale.py
│   ├── tb3_detection.py                 
│   ├── run_tb3_gazebo.py               
│   ├── run_orchestrator.py             
│   └── run_llm_orchestrator.py         # 🔜 Phase 3: LLM entry point
│
└── docs/
    └── plans/
        ├── 2026-05-26-orchestrator-slam-nav2.md
        ├── 2026-05-26-llm-orchestrator-harness.md
        └── 2026-05-26-master-plan.md    
```

## Phase completion checklist

### Phase 0 — Vision Pipeline ✅
- [x] Intrinsic camera calibration
- [x] Extrinsic camera calibration
- [x] Depth scale calibration
- [x] Open-vocabulary detection (Grounded SAM 2)
- [x] Monocular depth (Depth Anything V2)
- [x] 3D projection
- [x] ObjectDB with temporal tracking

### Phase 1 — Simulation Backends ✅
- [x] Isaac Sim integration (RGB-D from sim)
- [x] Isaac Sim depth calibration from ground truth
- [x] TB3 Gazebo detection via ROS 2 topics
- [x] Automated Gazebo launch wrapper
- [x] Headless/docker-safe detection (no cv2.imshow)
- [x] Depth Anything fallback when no depth topic

### Phase 2 — SLAM + Nav2 Orchestrator ✅
- [x] SLAM toolbox integration (live mapping)
- [x] Nav2 bringup (localization + planning)
- [x] Frontier-based exploration
- [x] Object detection in map frame (via TF)
- [x] ObjectDB in world coordinates
- [x] Manual goal services (/list_objects, /go_to_nearest_object)
- [x] RViz exploration goal visualization
- [x] static_layer fix (global costmap was empty)
- [x] Initial pose published for AMCL
- [x] Exploration goals in free space (not unknown cells)

### Phase 3 — LLM Harness + Semantic Awareness ✅ COMPLETE (2026-05-29)
- [x] Refactor: extract RobotInterface from MapOrchestrator
- [x] Add stop(), rotate(), go_home(), drive(), wait() methods
- [x] Add detect_now(), scan_surroundings(), search(), can_see() tools
- [x] Add get_pose(), get_status(), list_places(), remember_place() tools
- [x] Add ros2_introspect() tool (read-only ROS 2 CLI)
- [x] Build llm_tools.py (Tool registry + system prompt + failure catalog)
- [x] Build llm_bridge.py (conversation loop + rich TUI + 5-format thinking extraction + status bar)
- [x] Build run_llm_orchestrator.py (entry point, 27 tools, config.yaml)
- [x] ObjectDB upgrade: OBB fields (width, depth, height, angle)
- [x] Localization: DriftMonitor built + wired into detection loop
- [x] Graphification: SpatialGraph + query_graph() registered as LLM tool
- [x] Ad-hoc Python: run_python() sandbox tool
- [x] Detection prompt: get_detection_prompt() / set_detection_prompt()
- [x] Validation: validate_path() costmap check before navigation
- [x] Progressive refinement: refine_object() approach + multi-detect
- [x] Embedded JSON parsing (text + {json} mixed responses)
- [x] Exploration is LLM-driven tool (not automatic timer)
- [x] Config file (physicalai_config.yaml) instead of env vars

### Phase 4 — UI Layer: CLI Chat + Telegram Gateway ✅ COMPLETE (2026-05-29)
- [x] Build telegram_gateway.py (polling + send_message/send_photo)
- [x] Route say() output to both stdout and Telegram
- [x] Inject Telegram messages into LLM conversation loop
- [x] CLI enhancements: periodic goal-checking, map sync, prompt sync
- [x] Rich TUI panels (colored status, thinking display, tool results)
- [x] Persistent status bar (pose, objects, nav state, model)

### Phase 5 — Web Dashboard + Advanced Reliability ✅ COMPLETE (2026-05-29)
- [x] Build dashboard_server.py (FastAPI + MJPEG stream + map overlay)
- [x] Dashboard HTML (live video + object list + status bar)
- [x] 5A: Temporal queries (query_history + has_object_moved in ObjectDB + tools)
- [x] 5B: Uncertainty propagation (position_uncertainty field in ObjectRecord + OBB)
- [x] 5C: Multi-object re-identification (object_id keying + observations dedup)
- [x] 5D: Battery awareness (get_battery tool with Sim/real estimates)
- [x] 5E: Dynamic action discovery (discover_action tool)
- [ ] 5F: Test dashboard end-to-end (requires running orchestrator)
## How to track progress

The `orchestrator/` directory grows with each phase:
- Phase 2: `orchestrator/` = 5 files (existing)
- Phase 3: `orchestrator/` = 8 files (+3 new)
- Phase 4: no new files, just new tools attached to RobotInterface

Current stage: **START OF PHASE 3** — we're about to refactor and build the LLM harness.
