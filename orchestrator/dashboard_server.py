"""
dashboard_server.py — Modular RViz-like web dashboard for PhysicalAI.
"""
import os, sys, time, json, threading, base64
import numpy as np
import cv2

PHYSICALAI_ROOT = os.path.expanduser("~/PhysicalAI")
if PHYSICALAI_ROOT not in sys.path:
    sys.path.insert(0, PHYSICALAI_ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

app = FastAPI(title="PhysicalAI Dashboard", version="3.0")

_shared = {
    "robot": None, "node": None, "bridge": None,
    "chat_messages": [], "chat_max": 500,
    "system_logs": [], "logs_max": 300,
    "thinking": False,
    "processing": False,  # True while LLM chain is executing
    # Velocity history for graphing
    "vel_history": [],      # list of {ts, vx, wz}
    "vel_max_samples": 300,
    # Pending choices for object disambiguation
    "pending_choices": None,  # {question, choices: [{id, label, desc}], resolved: bool}
}

def set_shared_state(robot, node, bridge=None):
    _shared["robot"] = robot
    _shared["node"] = node
    if bridge: _shared["bridge"] = bridge

def push_chat_message(role: str, text: str):
    _shared["chat_messages"].append({"role": role, "text": str(text)[:800], "timestamp": time.time()})
    if len(_shared["chat_messages"]) > _shared["chat_max"]:
        _shared["chat_messages"] = _shared["chat_messages"][-500:]

def push_log(level: str, message: str):
    _shared["system_logs"].append({"level": level, "message": str(message)[:500], "timestamp": time.time()})
    if len(_shared["system_logs"]) > _shared["logs_max"]:
        _shared["system_logs"] = _shared["system_logs"][-300:]

def set_processing(active: bool):
    _shared["processing"] = active
    _shared["thinking"] = active

def set_pending_choices(question: str, choices: list):
    """Set a pending disambiguation choice that the dashboard will show as a modal.

    Args:
        question: Text to display (e.g. "Which object would you like to navigate to?")
        choices: List of {id, label, desc} dicts. Each entry has:
                 id: unique string (e.g. 'chair_1', 'chair_2')
                 label: short display title (e.g. 'Chair #1')
                 desc: location/description (e.g. 'At (-1.5, 4.2) - Red plastic')
    """
    _shared["pending_choices"] = {
        "question": question,
        "choices": choices,
        "resolved": False,
        "selected": None,
    }

def _grid_to_png(grid, robot_xy=None, robot_theta=None, map_info=None, objects=None,
                 rgb_free=(255,255,255), rgb_occupied=(255,80,80), rgb_unknown=(40,40,40)):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid == -1] = rgb_unknown
    img[grid == 0] = rgb_free
    occ = grid > 0
    for c in range(3): img[occ, c] = rgb_occupied[c]
    if robot_xy and robot_theta is not None and map_info:
        res = map_info["resolution"]
        ox, oy = map_info["origin_x"], map_info["origin_y"]
        px = int((robot_xy[0] - ox) / res)
        py = int((robot_xy[1] - oy) / res)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), 5, (0, 210, 255), -1)
            arrow_len = 18
            dx = int(np.cos(robot_theta) * arrow_len)
            dy = -int(np.sin(robot_theta) * arrow_len)
            cv2.arrowedLine(img, (px, py), (px + dx, py + dy), (255, 255, 0), 2, tipLength=0.35)
    if objects and map_info:
        res = map_info["resolution"]
        ox, oy = map_info["origin_x"], map_info["origin_y"]
        for obj in objects:
            gx = int((obj["x"] - ox) / res)
            gy = int((obj["y"] - oy) / res)
            if 0 <= gx < w and 0 <= gy < h:
                cv2.circle(img, (gx, gy), 3, (255, 50, 50), -1)
    _, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _gen_frames(key: str):
    while True:
        try:
            node = _shared["node"]
            if node is None: time.sleep(0.1); continue
            if key == "rgb":
                f = node.latest_rgb
                if f is None: time.sleep(0.1); continue
                f = f.copy()
                for r in getattr(node, 'detections', []):
                    x1, y1, x2, y2 = map(int, r.get("bbox_xyxy", [0, 0, 0, 0]))
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(f, r.get("class_name", "?"), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            elif key == "depth":
                f = node.latest_depth
                if f is None: time.sleep(0.1); continue
                f = f.copy().astype(np.float32)
                f = np.clip(f, 0.1, 10.0)
                f = cv2.applyColorMap((f / 10.0 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            else:
                time.sleep(0.1); continue
            _, jpeg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 55])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
        except Exception:
            time.sleep(0.5)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(_gen_frames("rgb"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_feed")
async def depth_feed():
    return StreamingResponse(_gen_frames("depth"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/map")
async def get_map():
    robot = _shared["robot"]
    if robot is None or robot._map is None: return JSONResponse({"has_map": False})
    grid = np.array(robot._map.data, dtype=np.int8).reshape(robot._map.info.height, robot._map.info.width)
    info = {"resolution": robot._map.info.resolution, "origin_x": robot._map.info.origin.position.x, "origin_y": robot._map.info.origin.position.y}
    rp = rt = None
    pr = robot.get_pose()
    if pr.success:
        p = pr.data.get("pose", (0, 0, 0))
        rp = (p[0], p[1]); rt = p[2]
    objs = []
    if robot._object_db:
        objs = [{"x": o.position_world[0], "y": o.position_world[1], "class": o.class_name, "conf": o.confidence} for o in robot._object_db.get_all()]
    png = _grid_to_png(grid, robot_xy=rp, robot_theta=rt, map_info=info, objects=objs)
    return JSONResponse({"has_map": True, "png": png, "info": info, "pose": list(rp) if rp else None, "objects": objs})

@app.get("/global_costmap")
async def get_global_costmap():
    n = _shared.get("node")
    if n is None: return JSONResponse({"has_data": False})
    g = getattr(n, 'global_costmap', None)
    if g is None: return JSONResponse({"has_data": False})
    grid = np.array(g.data, dtype=np.int8).reshape(g.info.height, g.info.width)
    return JSONResponse({"has_data": True,
        "png": _grid_to_png(grid, rgb_free=(230,230,240), rgb_occupied=(255,60,60), rgb_unknown=(25,25,45)),
        "info": {"resolution": g.info.resolution, "origin_x": g.info.origin.position.x, "origin_y": g.info.origin.position.y}})

@app.get("/local_costmap")
async def get_local_costmap():
    n = _shared.get("node")
    if n is None: return JSONResponse({"has_data": False})
    l = getattr(n, 'local_costmap', None)
    if l is None: return JSONResponse({"has_data": False})
    grid = np.array(l.data, dtype=np.int8).reshape(l.info.height, l.info.width)
    return JSONResponse({"has_data": True,
        "png": _grid_to_png(grid, rgb_free=(240,240,245), rgb_occupied=(255,100,40), rgb_unknown=(25,25,45)),
        "info": {"resolution": l.info.resolution, "origin_x": l.info.origin.position.x, "origin_y": l.info.origin.position.y}})

@app.get("/scan")
async def get_scan():
    s = getattr(_shared.get("node"), 'latest_scan', None)
    if s is None: return JSONResponse({"has_data": False})
    return JSONResponse({"has_data": True, "angle_min": s.angle_min, "angle_max": s.angle_max,
        "angle_increment": s.angle_increment, "range_min": s.range_min, "range_max": s.range_max,
        "ranges": list(s.ranges)[:360]})

@app.get("/status")
async def get_status():
    r = _shared["robot"]
    if r is None: return JSONResponse({"error": "no robot"}, 503)
    pr = r.get_pose()
    return JSONResponse({"pose": list(pr.data.get("pose", (0,0,0))) if pr.success else None,
        "objects": len(r._object_db.get_all()) if r._object_db else 0,
        "nav": "navigating" if r._goal_active else "idle",
        "thinking": _shared.get("thinking", False), "processing": _shared.get("processing", False),
        "safety": _get_safety_state()})

def _get_safety_state():
    n = _shared.get("node")
    if n is None or not hasattr(n, 'safety'):
        return {"emergency": False}
    s = n.safety
    return {
        "emergency": s.is_in_emergency,
        "closest_m": round(s.closest_object_m, 2) if s.closest_object_m != float('inf') else None,
        "count": s.emergency_count,
    }

@app.get("/context")
async def get_context():
    b = _shared.get("bridge")
    if b is None: return JSONResponse({"model":"?","tools":0,"ctx":"0/0"})
    m = b._model
    return JSONResponse({"model": m.split("/")[-1] if "/" in m else m, "tools": len(b._tools.all()),
        "ctx": f"{len(b._messages)}/{b._max_history}", "tokens": b._last_token_count})

@app.get("/topics")
async def list_topics():
    import subprocess
    try:
        r = subprocess.run(["timeout","3","ros2","topic","list"], capture_output=True, text=True, timeout=4)
        return JSONResponse({"topics": [t.strip() for t in r.stdout.strip().split("\n") if t.strip()]})
    except: return JSONResponse({"topics": []})

def _echo_raw(topic: str):
    import subprocess
    if topic in ("/cmd_vel", "cmd_vel"):
        return {"status": "blocked", "data": "Topic '/cmd_vel' is write-protected. Echo reading blocked."}
    r = subprocess.run(["timeout","3","ros2","topic","echo","--once","--flow-style",topic], capture_output=True, text=True, timeout=4)
    return {"status": "ok", "data": r.stdout.strip()[:2000] or None}

@app.post("/topic_echo")
async def topic_echo(request: Request):
    body = await request.json()
    topic = body.get("topic","").strip()
    if not topic: return JSONResponse({"status":"error"},400)
    return JSONResponse(_echo_raw(topic))

@app.post("/topic_echo_multi")
async def topic_echo_multi(request: Request):
    body = await request.json()
    topics = body.get("topics", [])
    if not topics: return JSONResponse({"results": {}})
    results = {}
    for t in topics:
        results[t] = _echo_raw(t)
    return JSONResponse({"results": results})

@app.get("/chat")
async def get_chat(since: int = 0):
    msgs = _shared["chat_messages"]
    new_msgs = msgs[since:] if since < len(msgs) else []
    return JSONResponse({"messages": new_msgs, "since": since, "total": len(msgs)})

@app.post("/send_command")
async def send_command(request: Request):
    body = await request.json()
    text = body.get("text","").strip()
    if not text: return JSONResponse({"status":"error"},400)
    push_chat_message("user", text)
    b = _shared.get("bridge")
    if b and hasattr(b, 'inject_message'): b.inject_message("[dashboard]", text)
    return JSONResponse({"status":"ok"})

@app.get("/logs")
async def get_logs(since: int = 0):
    logs = _shared["system_logs"]
    new_logs = logs[since:] if since < len(logs) else []
    return JSONResponse({"logs": new_logs, "since": since, "total": len(logs)})

@app.get("/velocity")
async def get_velocity():
    hist = _shared["vel_history"]
    latest = hist[-1] if hist else None
    return JSONResponse({"latest": latest, "history": hist[-100:]})

@app.get("/pending_choices")
async def get_pending_choices():
    pc = _shared.get("pending_choices")
    if pc and not pc.get("resolved"):
        return JSONResponse({"pending": True, "question": pc["question"], "choices": pc["choices"]})
    return JSONResponse({"pending": False})

@app.post("/resolve_choice")
async def resolve_choice(request: Request):
    body = await request.json()
    choice_index = body.get("choice")
    pc = _shared.get("pending_choices")
    if not pc or pc.get("resolved"):
        return JSONResponse({"status": "no_pending"})
    if choice_index is None or choice_index < 0 or choice_index >= len(pc["choices"]):
        return JSONResponse({"status": "invalid_choice"})
    pc["resolved"] = True
    pc["selected"] = choice_index
    choice = pc["choices"][choice_index]
    # Inject the resolved choice into the chat as a user message
    push_chat_message("user", f"I choose: {choice['label']} ({choice['desc']})")
    return JSONResponse({"status": "ok", "choice": choice})

def start_server(host="0.0.0.0", port=8080):
    import uvicorn
    def _run(): uvicorn.run(app, host=host, port=port, log_level="warning")
    t = threading.Thread(target=_run, daemon=True)

    # Start velocity sampler in background
    def _sample_vel():
        import subprocess
        while True:
            try:
                # Use timeout+ros2 topic echo --once (no --field) to check
                # if /cmd_vel has any message at all
                r = subprocess.run(
                    ["timeout","0.5","ros2","topic","echo","--once","--flow-style","/cmd_vel"],
                    capture_output=True, text=True, timeout=1)
                out = r.stdout.strip()
                if out:
                    # Parse linear.x and angular.z from YAML-like output
                    vx, wz = 0.0, 0.0
                    for line in out.split('\n'):
                        ls = line.strip()
                        if ls.startswith('linear.x:'):
                            try: vx = float(ls.split(':')[-1].strip())
                            except: pass
                        elif ls.startswith('angular.z:'):
                            try: wz = float(ls.split(':')[-1].strip())
                            except: pass
                    hist = _shared["vel_history"]
                    hist.append({"ts": time.time(), "vx": vx, "wz": wz})
                    if len(hist) > _shared["vel_max_samples"]:
                        _shared["vel_history"] = hist[-_shared["vel_max_samples"]:]
                else:
                    # No message → robot is stopped, push 0,0
                    hist = _shared["vel_history"]
                    hist.append({"ts": time.time(), "vx": 0.0, "wz": 0.0})
                    if len(hist) > _shared["vel_max_samples"]:
                        _shared["vel_history"] = hist[-_shared["vel_max_samples"]:]
            except Exception:
                pass
            time.sleep(0.25)

    vs = threading.Thread(target=_sample_vel, daemon=True)
    vs.start()
    t.start()
    return t

@app.get("/")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PhysicalAI</title>
<style>
:root{--bg:#0a0a14;--pan:#12122a;--bdr:#2a2a44;--ac:#00d4ff;--ac2:#30e060;--tx:#c0c0d0;--dim:#555;--rd:#e05050;--yel:#d0c030;--f:system-ui,sans-serif;--mo:'Courier New',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:var(--f);font-size:11px;overflow:hidden;height:100vh;display:flex;flex-direction:column}
#tb{display:flex;align-items:center;gap:12px;padding:4px 10px;background:#080818;border-bottom:1px solid var(--bdr);min-height:30px;flex-wrap:wrap}
#tb .br{color:var(--ac);font-weight:700;font-size:14px}
#tb .st{color:var(--dim);font-size:11px}
#tb .st .v{color:var(--ac);font-family:var(--mo)}
#tb .ctx{background:#0d0d1a;padding:2px 6px;border-radius:3px;font-family:var(--mo);color:var(--ac2);font-size:10px}
#tb select,#tb button{background:#0d0d1a;border:1px solid var(--bdr);color:var(--tx);padding:3px 7px;border-radius:3px;font-size:10px;cursor:pointer}
#tb button:hover{background:var(--ac);color:#000}
#main{flex:1;overflow:hidden}
#canvas{display:grid;grid-auto-flow:dense;gap:3px;padding:3px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));height:100%;overflow-y:auto;align-content:start}
.panel{border:1px solid var(--bdr);border-radius:4px;overflow:hidden;display:flex;flex-direction:column;min-height:180px;background:var(--pan)}
.panel.r2{grid-row:span 2}.panel.r3{grid-row:span 3}
.panel-head{display:flex;justify-content:space-between;align-items:center;padding:3px 8px;background:#0d0d1a;border-bottom:1px solid var(--bdr);font-size:11px;color:var(--ac);font-weight:600}
.panel-head .rm{color:var(--rd);cursor:pointer;font-size:15px;line-height:1;opacity:0.6}
.panel-head .rm:hover{opacity:1}
.panel-body{flex:1;overflow:hidden;position:relative;min-height:0;display:flex;flex-direction:column}
.panel-body img,.panel-body canvas{width:100%;height:100%;object-fit:contain;display:block;flex:1}
.panel-body pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mo);font-size:9px;padding:5px;overflow-y:auto;color:var(--ac2);max-height:100%;margin:0}
/* Chat */
#chat-list{flex:1;overflow-y:auto;padding:4px 6px}
.msg{margin:1px 0;padding:3px 6px;border-radius:3px;line-height:1.3;word-break:break-word;font-size:11px}
.msg .ts{color:var(--dim);font-size:9px;margin-right:5px;font-family:var(--mo)}
.msg.user{background:#1a1a3a;color:#a0c0ff}
.msg.robot{background:#0a2a1a;color:var(--ac2)}
.msg.tool{background:#1a1a0a;color:#d0d080;font-family:var(--mo);font-size:10px}
.msg.thinking{color:#888;font-size:10px;border-left:2px solid var(--ac);margin-left:4px;padding-left:6px;font-style:italic;background:transparent}
.chat-inp{display:flex;padding:4px 6px;border-top:1px solid var(--bdr);background:#080818}
.chat-inp input{flex:1;background:#0d0d1a;border:1px solid var(--bdr);color:var(--tx);padding:4px 7px;border-radius:3px;font-size:11px;outline:none}
.chat-inp input:focus{border-color:var(--ac)}
.chat-inp button{margin-left:4px;background:var(--ac);color:#000;border:none;padding:4px 10px;border-radius:3px;cursor:pointer;font-weight:600;font-size:11px}
.log-info{color:#aaa}.log-warn{color:#d0a040}.log-error{color:var(--rd)}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:2px}
</style></head>
<body>
<div id="tb">
  <span class="br">PhysicalAI</span>
  <span class="st">📍<span class="v" id="sp">-</span></span>
  <span class="st">🔍<span class="v" id="so">0</span></span>
  <span class="st">🚗<span class="v" id="ss">idle</span></span>
  <span class="ctx" id="sc">--</span>
  <span class="ctx" id="sctx">0/0</span>
  <span id="safety-indicator" style="display:none;color:var(--rd);font-weight:700;font-size:11px">⚠️ STOP</span>
  <span id="tl" style="display:none;color:var(--yel);font-size:10px">⏳</span>
  <span style="flex:1"></span>
  <select id="psel" onchange="addPanel(this.value);this.value=''"><option value="">+ Add</option><option value="rgb">📷RGB</option><option value="depth">🌊Depth</option><option value="map">🗺Map</option><option value="gcm">🌐GlobalCM</option><option value="lcm">🎯LocalCM</option><option value="scan">📏Scan</option><option value="vel">📊Velocity</option><option value="chat">💬Chat</option><option value="echo">🪵Echo</option><option value="objects">📦Objs</option><option value="logs">📜Logs</option></select>
  <button onclick="reset()">Reset</button>
</div>
<div id="choices-overlay" style="display:none;position:fixed;inset:0;z-index:999;background:rgba(0,0,0,0.7);justify-content:center;align-items:center">
  <div style="background:var(--pan);border:1px solid var(--ac);border-radius:6px;max-width:500px;width:90%;max-height:80vh;overflow-y:auto;padding:16px">
    <div style="color:var(--ac);font-weight:700;font-size:14px;margin-bottom:8px" id="ch-q">Choose:</div>
    <div id="ch-list"></div>
    <div style="margin-top:10px;display:flex;gap:6px;justify-content:flex-end">
      <button onclick="dismissChoices()" style="background:var(--dim);color:#fff;border:none;padding:5px 12px;border-radius:3px;cursor:pointer">Cancel</button>
    </div>
  </div>
</div>
<div id="main"><div id="canvas"></div></div>
<script>
let cc=0,lc=0,pidx=0,ACT={chat:0,logs:0},liveTopics=[];

const P={
  rgb:{t:'📷 RGB',r2:true, html:()=>'<img src="/video_feed">'},
  depth:{t:'🌊 Depth',r2:true, html:()=>'<img src="/depth_feed">'},
  map:{t:'🗺 Map',r2:true,html(id){return`<img id="${id}-img">`},
    async upd(id){try{const r=await fetch('/map');const d=await r.json();
    if(d.has_map)document.getElementById(id+'-img').src='data:image/png;base64,'+d.png;}catch(e){}}},
  gcm:{t:'🌐 Global Costmap',html(id){return`<img id="${id}-img">`},
    async upd(id){try{const r=await fetch('/global_costmap');const d=await r.json();
    if(d.has_data)document.getElementById(id+'-img').src='data:image/png;base64,'+d.png;}catch(e){}}},
  lcm:{t:'🎯 Local Costmap',html(id){return`<img id="${id}-img">`},
    async upd(id){try{const r=await fetch('/local_costmap');const d=await r.json();
    if(d.has_data)document.getElementById(id+'-img').src='data:image/png;base64,'+d.png;}catch(e){}}},
  scan:{t:'📏 Laser Scan',html(id){return`<canvas id="${id}-cv"></canvas>`},
    async upd(id){try{const r=await fetch('/scan');const d=await r.json();
    if(d.has_data){const cv=document.getElementById(id+'-cv');if(!cv)return;
    const W=cv.parentElement.clientWidth,H=cv.parentElement.clientHeight;cv.width=W;cv.height=H;
    const ctx=cv.getContext('2d');ctx.fillStyle='#080808';ctx.fillRect(0,0,W,H);
    const cx=W/2,cy=H/2,sc=Math.min(W,H)/2.2/d.range_max;
    ctx.strokeStyle='#30e06020';for(let r=0.5;r<=d.range_max;r+=0.5){ctx.beginPath();ctx.arc(cx,cy,r*sc,0,2*Math.PI);ctx.stroke()}
    ctx.strokeStyle='#00ff44';ctx.beginPath();
    for(let i=0;i<d.ranges.length;i++){const a=d.angle_min+i*d.angle_increment,r=d.ranges[i];
    if(!r||r<=0.01||r>d.range_max)continue;const px=cx+Math.cos(a)*r*sc,py=cy+Math.sin(a)*r*sc;i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);}
    ctx.stroke();ctx.fillStyle='#00d4ff';ctx.beginPath();ctx.arc(cx,cy,4,0,2*Math.PI);ctx.fill();}}catch(e){}}},
  vel:{t:'📊 Velocity',r2:true,html(id){return`<canvas id="${id}-cv"></canvas>`},
    async upd(id){try{const r=await fetch('/velocity');const d=await r.json();
    if(!d.history||d.history.length<2)return;
    const cv=document.getElementById(id+'-cv');if(!cv)return;
    const W=cv.parentElement.clientWidth,H=cv.parentElement.clientHeight;cv.width=W;cv.height=H;
    const ctx=cv.getContext('2d');
    const pad={t:15,b:12,l:35,r:15};const gw=W-pad.l-pad.r,gh=H-pad.t-pad.b;
    const h=d.history.slice(-150);const n=h.length;
    const maxV=Math.max(0.5,...h.map(p=>Math.abs(p.vx)),...h.map(p=>Math.abs(p.wz)))*1.2;
    ctx.clearRect(0,0,W,H);
    // Grid lines
    ctx.strokeStyle='#2a2a44';ctx.lineWidth=0.5;
    for(let i=-4;i<=4;i++){const y=pad.t+gh/2-i*(gh/2/maxV);if(y<pad.t||y>pad.t+gh)continue;
      ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
      ctx.fillStyle='#555';ctx.font='8px monospace';ctx.textAlign='right';
      ctx.fillText((i/4*maxV).toFixed(1),pad.l-2,y+3);}
    // Zero line
    ctx.strokeStyle='#444';ctx.lineWidth=1;const zy=pad.t+gh/2;
    ctx.beginPath();ctx.moveTo(pad.l,zy);ctx.lineTo(W-pad.r,zy);ctx.stroke();
    // Velocity lines
    const step=gw/(n-1||1);
    function drawLine(key,color){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
      for(let i=0;i<n;i++){const x=pad.l+i*step;const y=zy-h[i][key]/maxV*gh/2;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();}
    drawLine('vx','#00d4ff');drawLine('wz','#ff8040');
    // Labels
    ctx.fillStyle='#00d4ff';ctx.font='bold 9px monospace';ctx.textAlign='left';
    ctx.fillText(`vx ${(d.latest?.vx||0).toFixed(2)}`,pad.l+4,13);
    ctx.fillStyle='#ff8040';ctx.fillText(`wz ${(d.latest?.wz||0).toFixed(2)}`,pad.l+70,13);
    }catch(e){}}},
  chat:{t:'💬 Chat',r3:true,html(id){ACT.chat=1;
    return`<div id="chat-list" style="flex:1;overflow-y:auto;padding:4px 6px"></div><div class="chat-inp"><input id="inp" placeholder="Command..." onkeydown="if(event.key==='Enter')snd()"><button onclick="snd()">Send</button></div>`;},
    async upd(id){try{const r=await fetch(`/chat?since=${cc}`);const d=await r.json();
    if(d.messages&&d.messages.length>0){const b=document.getElementById('chat-list');
    for(const m of d.messages){const dv=document.createElement('div');dv.className='msg '+m.role;
    dv.innerHTML=`<span class="ts">${new Date(m.timestamp*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>`+
    (m.role==='user'?`<b>▸</b> `:m.role==='robot'?'🤖 ':m.role==='thinking'?'💭 ':m.role==='tool'?'🔧 ':'')+e(m.text);
    b.appendChild(dv);}
    b.scrollTop=b.scrollHeight;cc=d.total||d.messages.length;}}catch(e){}}},
  echo:{t:'🪵 Topic Echo',r3:true,html(id){
    liveTopics.push(id);
    return`<div style="display:flex;gap:3px;padding:3px;background:#0d0d1a;border-bottom:1px solid var(--bdr)">
    <input id="${id}-tp" placeholder="topic name" style="flex:1;background:#080818;border:1px solid var(--bdr);color:var(--tx);padding:3px 6px;font-size:10px;border-radius:3px"
    onkeydown="if(event.key==='Enter')startLive('${id}')"><button onclick="startLive('${id}')" style="background:var(--ac);color:#000;border:none;padding:3px 8px;border-radius:3px;cursor:pointer;font-size:10px">Watch</button>
    <button onclick="stopLive('${id}')" style="background:var(--dim);color:#fff;border:none;padding:3px 8px;border-radius:3px;cursor:pointer;font-size:10px">Stop</button>
    </div><pre id="${id}-pre" style="flex:1;overflow-y:auto;font-family:var(--mo);font-size:9px;padding:5px;color:var(--ac2);word-break:break-all;white-space:pre-wrap">Enter a topic name and click Watch.</pre>`;
  },onRemove(id){stopLive(id)}},
  objects:{t:'📦 Objects',html(id){return`<div id="${id}-list" style="overflow-y:auto;padding:4px 6px;font-family:var(--mo);font-size:10px"></div>`},
    async upd(id){try{const r=await fetch('/map');const d=await r.json();let h='';
    for(const o of(d.objects||[]))h+=`<div>${e(o.class)} (${o.x.toFixed(1)},${o.y.toFixed(1)}) <span style="color:var(--dim)">c=${o.conf.toFixed(2)}</span></div>`;
    document.getElementById(id+'-list').innerHTML=h||'No objects';}catch(e){}}},
  logs:{t:'📜 System Logs',r3:true,html(id){ACT.logs=1;return`<div id="log-${id}" style="flex:1;overflow-y:auto;font-family:var(--mo);font-size:9px;padding:3px 6px;background:#080808;color:var(--dim)"></div>`},
    async upd(id){try{const r=await fetch(`/logs?since=${lc}`);const d=await r.json();
    if(d.logs&&d.logs.length>0){const b=document.getElementById('log-'+id);
    for(const l of d.logs){const dv=document.createElement('div');dv.className='log-'+l.level;
    dv.textContent=`[${new Date(l.timestamp*1000).toLocaleTimeString()}] [${l.level.toUpperCase()}] ${l.message}`;
    b.appendChild(dv);}
    b.scrollTop=b.scrollHeight;lc=d.total||d.logs.length;}}catch(e){}}},
};

let panels={};
function addPanel(type){
  if(!P[type])return;
  if(type==='chat'&&ACT.chat)return;
  if(type==='logs'&&ACT.logs)return;
  const id='p'+ ++pidx;
  const p=P[type];
  const cls=p.r2?'r2':p.r3?'r3':'';
  const div=document.createElement('div');
  div.className='panel '+cls;
  div.setAttribute('data-pid',id);
  div.setAttribute('data-ptype',type);
  div.innerHTML=`<div class="panel-head"><span>${p.t}</span><span class="rm" onclick="delPanel('${id}','${type}')">×</span></div><div class="panel-body">${typeof p.html==='function'?p.html(id):p.html}</div>`;
  document.getElementById('canvas').appendChild(div);
  panels[id]={type,el:div};
  if(type==='chat'){setTimeout(()=>{const inp=document.getElementById('inp');if(inp)inp.focus();},200);}
  return id;
}
function delPanel(id,type){
  if(P[type]&&P[type].onRemove)P[type].onRemove(id);
  if(type==='chat')ACT.chat=0;
  if(type==='logs')ACT.logs=0;
  delete panels[id];
  const el=document.querySelector(`[data-pid="${id}"]`);
  if(el)el.remove();
}
function reset(){
  document.getElementById('canvas').innerHTML='';
  panels={};ACT.chat=ACT.logs=0;
  ['rgb','depth','map','objects','chat','gcm','lcm','scan','logs'].forEach(t=>addPanel(t));
}

async function startLive(id){
  const tp=document.getElementById(id+'-tp').value.trim();
  if(!tp)return;
  const pre=document.getElementById(id+'-pre');
  pre.setAttribute('data-live-topic',tp);
  pre.setAttribute('data-live-id',id);
  pre.style.background='#0a1a0a';
}
function stopLive(id){
  const pre=document.getElementById(id+'-pre');
  if(pre)pre.style.background='';
}

async function pollEcho(){
  const active=[];
  for(const id of liveTopics){
    const pre=document.getElementById(id+'-pre');
    if(!pre)continue;
    const tp=pre.getAttribute('data-live-topic');
    if(!tp)continue;
    active.push({id,tp,pre});
  }
  if(!active.length)return;
  const topics=active.map(a=>a.tp);
  try{
    const r=await fetch('/topic_echo_multi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics})});
    const d=await r.json();
    for(const a of active){
      const res=d.results[a.tp];
      if(res&&res.data){
        const ts=new Date().toLocaleTimeString();
        a.pre.textContent=`[${ts}] ${res.data}`;
        a.pre.scrollTop=a.pre.scrollHeight;
      }
    }
  }catch(e){}
}

async function snd(){
  const inp=document.getElementById('inp');if(!inp)return;
  const t=inp.value.trim();if(!t)return;
  inp.value='';inp.disabled=true;
  try{await fetch('/send_command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});}catch(e){}
  setTimeout(()=>{inp.disabled=false;inp.focus();},300);
}
async function makeChoice(idx){
  try{await fetch('/resolve_choice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice:idx})});
  document.getElementById('choices-overlay').style.display='none';}catch(e){}
}
function dismissChoices(){document.getElementById('choices-overlay').style.display='none';}

async function tick(){
  // Status
  try{const r=await fetch('/status');const d=await r.json();
  document.getElementById('sp').textContent=d.pose?`(${d.pose[0].toFixed(1)},${d.pose[1].toFixed(1)})`:'-';
  document.getElementById('so').textContent=d.objects;
  document.getElementById('ss').textContent=d.nav;
  const si=document.getElementById('safety-indicator');
  if(d.safety&&d.safety.emergency){si.style.display='inline';si.textContent='⚠️ STOP '+d.safety.closest_m+'m';}
  else si.style.display='none';
  document.getElementById('tl').style.display=d.thinking?'inline':'none';
  // Processing state — update chat input placeholder
  const inp=document.getElementById('inp');
  if(inp){
    inp.placeholder=d.processing?'⏳ Processing... (type to interrupt)':'Command...';
    inp.style.color=d.processing?'#d0c030':'';
  }}catch(e){}
  try{const r=await fetch('/context');const d=await r.json();
  document.getElementById('sc').textContent=d.model||'?';
  document.getElementById('sctx').textContent=d.ctx||'0/0';}catch(e){}
  // Panel updates
  for(const [id,pi] of Object.entries(panels)){
    const p=P[pi.type];
    if(p&&p.upd)await p.upd(id);
  }
  // Live topic poll
  await pollEcho();
  // Choices poll
  try{const r=await fetch('/pending_choices');const d=await r.json();
  if(d.pending){const ov=document.getElementById('choices-overlay');ov.style.display='flex';
    document.getElementById('ch-q').textContent=d.question;
    document.getElementById('ch-list').innerHTML=d.choices.map((c,i)=>`<div onclick="makeChoice(${i})" style="cursor:pointer;background:#0d0d1a;border:1px solid var(--bdr);border-radius:4px;padding:8px 10px;margin:3px 0;display:flex;gap:6px;align-items:flex-start" onmouseover="this.style.borderColor='var(--ac)'" onmouseout="this.style.borderColor=''">
      <span style="color:var(--ac);font-family:var(--mo);font-size:12px;font-weight:700;min-width:28px">#${i+1}</span>
      <div><div style="color:var(--ac2);font-weight:600;font-size:12px">${e(c.label||c.id)}</div>
      <div style="color:var(--dim);font-size:10px">${e(c.desc||'')}</div></div>
    </div>`).join('');}
  else{document.getElementById('choices-overlay').style.display='none';}}catch(e){}
}
function e(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
reset();
tick();
setInterval(tick,1500);
</script></body></html>"""
