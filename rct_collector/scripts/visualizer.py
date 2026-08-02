#!/usr/bin/env python3
"""
Interactive Trial Motion Visualizer for Online Tuner / RCT Data Collector.

This script launches a lightweight HTTP web server providing a responsive,
interactive web UI for visualising navigation trials recorded by trial_runner.py.

Key Features:
  - Input field & file selector to load any trial JSON file.
  - Interactive HTML5 Canvas showing map, start/goal poses, global plan, and executed path.
  - Smooth pan & zoom controls for map navigation.
  - Time slider & playback controls (Play, Pause, Step, Speed 0.5x-10x).
  - Robot animation: position (x, y), heading angle, footprint polygon ("tucked", "carry", etc.).
  - Ground Truth vs Believed (AMCL) pose comparison.
  - Real-time telemetry dashboard (velocities, clearance, footprint cost, 8-D risk state).
  - Nav2 treatment hyperparameter configuration display.

Usage:
  python3 -m rct_collector.scripts.visualizer --json /path/to/trial.json
  # or after colcon build:
  rct_visualize --json /path/to/trial.json
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import sys
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

# Default Footprint Polygons (PMB2 / TIAGo base)
FOOTPRINTS = {
    "tucked": [
        [-0.275, 0.000], [-0.238, -0.138], [-0.138, -0.238], [-0.000, -0.275],
        [0.138, -0.238], [0.209, -0.181], [0.238, -0.138], [0.275, 0.000],
        [0.252, 0.182], [0.217, 0.242], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]
    ],
    "carry": [
        [-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641],
        [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138],
        [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]
    ]
}

# Default circular footprint fallback if unknown (radius ~0.275 m)
DEFAULT_CIRCULAR_FOOTPRINT = [
    [0.275 * math.cos(a), 0.275 * math.sin(a)]
    for a in [i * (2 * math.pi / 16) for i in range(16)]
]


def resolve_map_path(map_yaml_path: str) -> str:
    """Find map yaml file on local filesystem, with fallbacks if missing."""
    if map_yaml_path and os.path.exists(map_yaml_path):
        return map_yaml_path

    # Fallback search paths in workspace
    workspace_root = Path(__file__).resolve().parents[4]
    candidates = [
        workspace_root / "src" / "gazebo_simulation" / "maps" / "map.yaml",
        workspace_root / "src" / "gazebo_simulation" / "maps" / "house.yaml",
        workspace_root / "src" / "gazebo_simulation" / "maps" / "supermarkt.yaml",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    # Search any map.yaml in src
    found = list(workspace_root.glob("**/maps/*.yaml"))
    if found:
        return str(found[0])

    return ""


def load_map_metadata_and_image(map_yaml_path: str) -> dict:
    """Load map YAML and convert map image to Base64 PNG string."""
    real_path = resolve_map_path(map_yaml_path)
    if not real_path or not os.path.exists(real_path):
        return {"error": f"Map file not found: {map_yaml_path}"}

    try:
        with open(real_path, "r") as f:
            info = yaml.safe_load(f)

        resolution = float(info.get("resolution", 0.05))
        origin = [float(v) for v in info.get("origin", [0.0, 0.0, 0.0])]
        negate = int(info.get("negate", 0))

        img_rel = info.get("image", "map.pgm")
        img_path = Path(real_path).parent / img_rel

        if not img_path.exists():
            return {"error": f"Map image file missing: {img_path}"}

        img = Image.open(img_path)
        img_w, img_h = img.size

        # Convert to PNG Data URI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "yaml_path": real_path,
            "resolution": resolution,
            "origin": origin,
            "negate": negate,
            "width": img_w,
            "height": img_h,
            "image_data_uri": f"data:image/png;base64,{img_b64}"
        }
    except Exception as e:
        return {"error": f"Failed to load map: {str(e)}"}


def parse_footprint(fp_val: Any) -> list:
    """Parse footprint string or list from nav2_config."""
    if isinstance(fp_val, str):
        if fp_val in FOOTPRINTS:
            return FOOTPRINTS[fp_val]
        try:
            parsed = json.loads(fp_val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    elif isinstance(fp_val, list):
        return fp_val
    return FOOTPRINTS.get("carry", DEFAULT_CIRCULAR_FOOTPRINT)


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    initial_json_path = ""
    workspace_dir = ""

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        params = urllib.parse.parse_qs(parsed_url.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = get_visualizer_html(self.initial_json_path)
            self.wfile.write(html.encode("utf-8"))

        elif path == "/api/load_trial":
            trial_path = params.get("path", [""])[0].strip()
            if not trial_path:
                trial_path = self.initial_json_path

            res = self.handle_load_trial(trial_path)
            self.send_json(res)

        elif path == "/api/list_trials":
            res = self.handle_list_trials()
            self.send_json(res)

        else:
            self.send_error(404, "Not Found")

    def handle_load_trial(self, trial_path: str) -> dict:
        if not trial_path or not os.path.exists(trial_path):
            return {"success": False, "error": f"File does not exist: '{trial_path}'"}

        try:
            with open(trial_path, "r") as f:
                data = json.load(f)

            # Resolve map
            map_yaml = data.get("map_yaml", "")
            map_meta = load_map_metadata_and_image(map_yaml)

            # Resolve footprint
            nav2_cfg = data.get("nav2_config", {})
            fp_type = nav2_cfg.get("local_costmap__footprint", "carry")
            footprint_poly = parse_footprint(fp_type)

            return {
                "success": True,
                "file_path": os.path.abspath(trial_path),
                "data": data,
                "map": map_meta,
                "footprint_polygon": footprint_poly,
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to parse trial JSON: {str(e)}"}

    def handle_list_trials(self) -> dict:
        ws = self.workspace_dir or str(Path(__file__).resolve().parents[4])
        trials = []
        try:
            for p in Path(ws).glob("**/*.json"):
                if "trial_" in p.name:
                    trials.append(str(p))
            trials.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return {"success": True, "trials": trials[:50]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def send_json(self, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def get_visualizer_html(initial_path: str = "") -> str:
    """Embedded single-page application source code for the trial visualizer."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RCT Trial Motion Visualizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #0f172a;
      --bg-card: #1e293b;
      --bg-input: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #22c55e;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --accent-amber: #f59e0b;
      --border-color: #334155;
    }}
    
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    /* Top Bar */
    header {{
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 15px;
      z-index: 10;
    }}

    .logo-area {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .logo-area h1 {{
      font-size: 1.1rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
    }}

    .input-bar {{
      flex: 1;
      max-width: 800px;
      display: flex;
      gap: 8px;
    }}

    input[type="text"] {{
      flex: 1;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }}

    input[type="text"]:focus {{
      border-color: var(--accent-blue);
    }}

    button {{
      background: var(--accent-blue);
      color: #0f172a;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    button:hover {{
      opacity: 0.9;
      transform: translateY(-1px);
    }}

    button.btn-secondary {{
      background: var(--bg-input);
      color: var(--text-main);
    }}

    button.btn-secondary:hover {{
      background: #475569;
    }}

    /* Main Container */
    .app-container {{
      flex: 1;
      display: flex;
      overflow: hidden;
    }}

    /* Viewport / Canvas */
    .viewport {{
      flex: 1;
      position: relative;
      background: #020617;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      user-select: none;
    }}

    canvas {{
      display: block;
      cursor: grab;
    }}

    canvas:active {{
      cursor: grabbing;
    }}

    /* Floating View Controls */
    .canvas-overlay-controls {{
      position: absolute;
      top: 15px;
      left: 15px;
      display: flex;
      gap: 8px;
      background: rgba(30, 41, 59, 0.85);
      backdrop-filter: blur(8px);
      padding: 6px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }}

    .status-badge {{
      position: absolute;
      top: 15px;
      right: 15px;
      padding: 6px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 0.8rem;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid var(--border-color);
    }}

    .status-SUCCESS {{ color: var(--accent-green); border-color: var(--accent-green); }}
    .status-COLLISION {{ color: var(--accent-red); border-color: var(--accent-red); }}
    .status-TIMEOUT {{ color: var(--accent-amber); border-color: var(--accent-amber); }}
    .status-PLANNING_FAILED {{ color: var(--accent-purple); border-color: var(--accent-purple); }}

    /* Bottom Timeline Bar */
    .timeline-bar {{
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      width: 90%;
      max-width: 900px;
      background: rgba(30, 41, 59, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 12px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}

    .timeline-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 15px;
    }}

    .btn-group {{
      display: flex;
      gap: 6px;
    }}

    .slider-container {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
    }}

    input[type="range"] {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      accent-color: var(--accent-blue);
      cursor: pointer;
    }}

    .time-readout {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: var(--accent-blue);
      min-width: 80px;
      text-align: right;
    }}

    /* Sidebar Dashboard */
    .sidebar {{
      width: 360px;
      background: var(--bg-card);
      border-left: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }}

    .panel {{
      padding: 16px;
      border-bottom: 1px solid var(--border-color);
    }}

    .panel-title {{
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }}

    .metric-card {{
      background: var(--bg-dark);
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .metric-label {{
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}

    .metric-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-main);
    }}

    .config-table {{
      width: 100%;
      font-size: 0.78rem;
      border-collapse: collapse;
    }}

    .config-table td {{
      padding: 6px 4px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}

    .config-table td.key {{
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }}

    .config-table td.val {{
      text-align: right;
      font-weight: 600;
      color: var(--accent-blue);
      font-family: 'JetBrains Mono', monospace;
    }}

    .toggle-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 0.85rem;
    }}

    .toggle-switch {{
      position: relative;
      display: inline-block;
      width: 36px;
      height: 20px;
    }}

    .toggle-switch input {{
      opacity: 0;
      width: 0;
      height: 0;
    }}

    .slider-round {{
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      transition: .2s;
      border-radius: 20px;
    }}

    .slider-round:before {{
      position: absolute;
      content: "";
      height: 14px;
      width: 14px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .2s;
      border-radius: 50%;
    }}

    input:checked + .slider-round {{
      background-color: var(--accent-blue);
    }}

    input:checked + .slider-round:before {{
      transform: translateX(16px);
    }}

    /* Dropdown helper */
    .dropdown-container {{
      position: relative;
    }}

    .dropdown-menu {{
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      max-height: 200px;
      overflow-y: auto;
      z-index: 100;
      display: none;
      margin-top: 4px;
    }}

    .dropdown-item {{
      padding: 8px 12px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      cursor: pointer;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}

    .dropdown-item:hover {{
      background: var(--bg-input);
      color: var(--accent-blue);
    }}
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="logo-area">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
      <h1>RCT Trial Visualizer</h1>
    </div>

    <div class="input-bar">
      <input type="text" id="jsonPathInput" value="{initial_path}" placeholder="Enter trial JSON path (e.g. rct_data/trials/trial_00001.json)">
      <button onclick="loadTrialFromInput()">Load Trial</button>
      <button class="btn-secondary" onclick="browseTrialFiles()">Browse</button>
      <input type="file" id="filePicker" accept=".json" style="display:none" onchange="onFilePicked(event)">
    </div>
  </header>

  <!-- Main Workspace -->
  <div class="app-container">
    
    <!-- Map Viewport -->
    <div class="viewport" id="viewport">
      <canvas id="mapCanvas"></canvas>

      <!-- View Overlay Controls -->
      <div class="canvas-overlay-controls">
        <button class="btn-secondary" onclick="resetCamera()">Reset View</button>
        <button class="btn-secondary" id="btnToggleAMCL" onclick="toggleAMCL()">Believed Pose: OFF</button>
      </div>

      <!-- Status Badge -->
      <div class="status-badge" id="statusBadge">NO TRIAL LOADED</div>

      <!-- Bottom Timeline Overlay -->
      <div class="timeline-bar">
        <div class="timeline-controls">
          <div class="btn-group">
            <button id="btnPlay" onclick="togglePlay()">Play</button>
            <button class="btn-secondary" onclick="stepFrame(-1)">&#9664;</button>
            <button class="btn-secondary" onclick="stepFrame(1)">&#9654;</button>
          </div>

          <div class="slider-container">
            <input type="range" id="timeSlider" min="0" max="0" value="0" oninput="onSliderChange(this.value)">
            <div class="time-readout" id="timeReadout">0.00s</div>
          </div>

          <select id="speedSelect" style="background:var(--bg-input); color:var(--text-main); border:1px solid var(--border-color); padding:6px; border-radius:6px;" onchange="setSpeed(this.value)">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1.0x</option>
            <option value="2">2.0x</option>
            <option value="5">5.0x</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Sidebar Dashboard -->
    <div class="sidebar">
      
      <!-- Trial Summary -->
      <div class="panel">
        <div class="panel-title">Trial Metadata</div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Trial ID</div>
            <div class="metric-value" id="valTrialId">-</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value" id="valStatus">-</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Travel Time</div>
            <div class="metric-value" id="valTravelTime">-</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Exec Path Len</div>
            <div class="metric-value" id="valPathLen">-</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Final Error</div>
            <div class="metric-value" id="valFinalErr">-</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Footprint</div>
            <div class="metric-value" id="valFootprintType">-</div>
          </div>
        </div>
      </div>

      <!-- Live Robot State -->
      <div class="panel">
        <div class="panel-title">Live Telemetry (Step <span id="valStepIdx">0</span>/<span id="valTotalSteps">0</span>)</div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Pose X, Y (m)</div>
            <div class="metric-value" id="valPoseXY">0.00, 0.00</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Yaw Heading</div>
            <div class="metric-value" id="valYaw">0.0°</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Lin Speed (m/s)</div>
            <div class="metric-value" id="valLinVel">0.00</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Ang Speed (rad/s)</div>
            <div class="metric-value" id="valAngVel">0.00</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Footprint Cost</div>
            <div class="metric-value" id="valFootprintCost">0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Min Scan Dist</div>
            <div class="metric-value" id="valMinScan">-</div>
          </div>
        </div>
      </div>

      <!-- Display Options -->
      <div class="panel">
        <div class="panel-title">Layer Toggles</div>
        <div class="toggle-row">
          <span>Global Planner Path</span>
          <label class="toggle-switch">
            <input type="checkbox" id="chkGlobalPath" checked onchange="requestRender()">
            <span class="slider-round"></span>
          </label>
        </div>
        <div class="toggle-row">
          <span>Executed Controller Path</span>
          <label class="toggle-switch">
            <input type="checkbox" id="chkControllerPath" checked onchange="requestRender()">
            <span class="slider-round"></span>
          </label>
        </div>
        <div class="toggle-row">
          <span>Robot Footprint Polygon</span>
          <label class="toggle-switch">
            <input type="checkbox" id="chkFootprint" checked onchange="requestRender()">
            <span class="slider-round"></span>
          </label>
        </div>
      </div>

      <!-- Nav2 Config Parameters -->
      <div class="panel">
        <div class="panel-title">Nav2 Hyperparameters</div>
        <table class="config-table" id="configTable">
          <tbody>
            <tr><td colspan="2" style="color:var(--text-muted)">No parameters loaded</td></tr>
          </tbody>
        </table>
      </div>

    </div>

  </div>

  <script>
    // State Variables
    let trialData = null;
    let mapMeta = null;
    let mapImage = new Image();
    let isMapLoaded = false;
    let footprintPoly = [];

    let currentStep = 0;
    let isPlaying = false;
    let playbackSpeed = 1.0;
    let playTimer = null;
    let showAMCL = false;

    // Viewport Transform (Pan & Zoom)
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const viewport = document.getElementById('viewport');

    // Initialize Canvas Size
    function resizeCanvas() {{
      canvas.width = viewport.clientWidth;
      canvas.height = viewport.clientHeight;
      requestRender();
    }}
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Mouse Controls (Pan & Zoom)
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      
      const mouseX = e.clientX - viewport.offsetLeft;
      const mouseY = e.clientY - viewport.offsetTop;

      panX = mouseX - (mouseX - panX) * zoomFactor;
      panY = mouseY - (mouseY - panY) * zoomFactor;
      zoom *= zoomFactor;

      requestRender();
    }});

    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      dragStartX = e.clientX - panX;
      dragStartY = e.clientY - panY;
    }});

    window.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        panX = e.clientX - dragStartX;
        panY = e.clientY - dragStartY;
        requestRender();
      }}
    }});

    window.addEventListener('mouseup', () => {{
      isDragging = false;
    }});

    // Load Trial Data
    async function loadTrialFromInput() {{
      const path = document.getElementById('jsonPathInput').value.trim();
      if (!path) return;

      try {{
        const res = await fetch(`/api/load_trial?path=${{encodeURIComponent(path)}}`);
        const json = await res.json();
        
        if (!json.success) {{
          alert(`Error loading trial: ${{json.error}}`);
          return;
        }}

        processLoadedTrial(json);
      }} catch (err) {{
        alert(`Failed to fetch trial: ${{err}}`);
      }}
    }}

    function browseTrialFiles() {{
      document.getElementById('filePicker').click();
    }}

    function onFilePicked(e) {{
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {{
        try {{
          const data = JSON.parse(event.target.result);
          // If loaded locally via browser file API
          processLoadedTrial({{
            success: true,
            file_path: file.name,
            data: data,
            map: null,
            footprint_polygon: []
          }});
        }} catch(ex) {{
          alert("Failed to parse JSON file");
        }}
      }};
      reader.readAsText(file);
    }}

    function processLoadedTrial(payload) {{
      trialData = payload.data;
      footprintPoly = payload.footprint_polygon || [];

      // Update Trial Info Summary
      document.getElementById('valTrialId').innerText = trialData.trial_id ?? '-';
      
      const status = trialData.status || 'UNKNOWN';
      const badge = document.getElementById('statusBadge');
      badge.innerText = status;
      badge.className = `status-badge status-${{status}}`;
      document.getElementById('valStatus').innerText = status;

      document.getElementById('valTravelTime').innerText = trialData.travel_time_sec ? `${{trialData.travel_time_sec.toFixed(2)}}s` : '-';
      document.getElementById('valPathLen').innerText = trialData.local_path_length ? `${{trialData.local_path_length.toFixed(2)}}m` : '-';
      document.getElementById('valFinalErr').innerText = trialData.final_xy_error ? `${{trialData.final_xy_error.toFixed(2)}}m` : '-';
      
      const fpType = trialData.nav2_config?.['local_costmap__footprint'] || 'carry';
      document.getElementById('valFootprintType').innerText = fpType;

      // Update Nav2 Hyperparameter Table
      const cfgTable = document.getElementById('configTable');
      cfgTable.innerHTML = '';
      if (trialData.nav2_config) {{
        for (const [k, v] of Object.entries(trialData.nav2_config)) {{
          const tr = document.createElement('tr');
          const shortKey = k.replace('controller_server__FollowPath.', '').replace('local_costmap__', '');
          const formattedVal = typeof v === 'number' ? v.toFixed(3) : String(v);
          tr.innerHTML = `<td class="key">${{shortKey}}</td><td class="val">${{formattedVal}}</td>`;
          cfgTable.appendChild(tr);
        }}
      }}

      // Setup Time Slider
      const samples = trialData.path_with_controller || [];
      const slider = document.getElementById('timeSlider');
      slider.max = Math.max(0, samples.length - 1);
      slider.value = 0;
      currentStep = 0;
      document.getElementById('valTotalSteps').innerText = samples.length;

      // Load Map Image
      if (payload.map && payload.map.image_data_uri) {{
        mapMeta = payload.map;
        mapImage = new Image();
        mapImage.onload = () => {{
          isMapLoaded = true;
          resetCamera();
        }};
        mapImage.src = payload.map.image_data_uri;
      }} else {{
        isMapLoaded = false;
        resetCamera();
      }}

      updateTelemetry(0);
      requestRender();
    }}

    // World (m) -> Map Pixel Coordinates
    function worldToPixel(x, y) {{
      if (!mapMeta) {{
        // Fallback default scaling if no map image loaded
        return {{ px: canvas.width / 2 + x * 40, py: canvas.height / 2 - y * 40 }};
      }}
      const res = mapMeta.resolution;
      const ox = mapMeta.origin[0];
      const oy = mapMeta.origin[1];
      const px = (x - ox) / res;
      const py = mapMeta.height - (y - oy) / res;
      return {{ px, py }};
    }}

    function resetCamera() {{
      if (mapMeta && mapImage.complete) {{
        const scaleX = canvas.width / mapMeta.width;
        const scaleY = canvas.height / mapMeta.height;
        zoom = Math.min(scaleX, scaleY) * 0.9;
        panX = (canvas.width - mapMeta.width * zoom) / 2;
        panY = (canvas.height - mapMeta.height * zoom) / 2;
      }} else {{
        zoom = 1.0;
        panX = 0;
        panY = 0;
      }}
      requestRender();
    }}

    // Render Canvas
    function requestRender() {{
      requestAnimationFrame(render);
    }}

    // Helper to draw a robot pose (center dot, heading arrow, and rotated footprint)
    function drawRobotPose(x, y, yaw, opts = {{}}) {{
      const p = worldToPixel(x, y);
      ctx.save();
      ctx.translate(p.px, p.py);
      ctx.rotate(-yaw);

      // Footprint Polygon
      if (document.getElementById('chkFootprint').checked && footprintPoly && footprintPoly.length > 0) {{
        const res = mapMeta ? mapMeta.resolution : 0.05;
        ctx.beginPath();
        ctx.fillStyle = opts.fillColor || 'rgba(56, 189, 248, 0.25)';
        ctx.strokeStyle = opts.strokeColor || '#38bdf8';
        ctx.lineWidth = (opts.lineWidth || 2) / zoom;
        if (opts.isDashed) ctx.setLineDash([4 / zoom, 4 / zoom]);
        else ctx.setLineDash([]);

        footprintPoly.forEach((pt, idx) => {{
          const fx = pt[0] / res;
          const fy = -pt[1] / res;
          if (idx === 0) ctx.moveTo(fx, fy);
          else ctx.lineTo(fx, fy);
        }});
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // Robot Center Dot
      ctx.fillStyle = opts.centerColor || '#a855f7';
      ctx.beginPath();
      ctx.arc(0, 0, (opts.radius || 4) / zoom, 0, Math.PI * 2);
      ctx.fill();

      // Heading Arrow (pointing in orientation direction)
      const arrowLen = (opts.arrowLength || 22) / zoom;
      const headLen = 6 / zoom;
      const arrowColor = opts.arrowColor || opts.strokeColor || '#f59e0b';

      ctx.strokeStyle = arrowColor;
      ctx.lineWidth = (opts.arrowWidth || 2.5) / zoom;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(arrowLen, 0);
      ctx.stroke();

      // Arrowhead Tip
      ctx.fillStyle = arrowColor;
      ctx.beginPath();
      ctx.moveTo(arrowLen + 2 / zoom, 0);
      ctx.lineTo(arrowLen - headLen, -headLen * 0.5);
      ctx.lineTo(arrowLen - headLen, headLen * 0.5);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }}

    function render() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.translate(panX, panY);
      ctx.scale(zoom, zoom);

      // 1. Draw Map Image
      if (isMapLoaded && mapImage) {{
        ctx.drawImage(mapImage, 0, 0);
      }} else {{
        // Draw grid placeholder if no map
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        for (let x = -1000; x < 2000; x += 50) {{
          ctx.beginPath(); ctx.moveTo(x, -1000); ctx.lineTo(x, 2000); ctx.stroke();
        }}
        for (let y = -1000; y < 2000; y += 50) {{
          ctx.beginPath(); ctx.moveTo(-1000, y); ctx.lineTo(2000, y); ctx.stroke();
        }}
      }}

      if (!trialData) {{
        ctx.restore();
        return;
      }}

      // 2. Draw Global Path (Planner)
      if (document.getElementById('chkGlobalPath').checked && trialData.path_global_planner) {{
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2 / zoom;

        trialData.path_global_planner.forEach((pt, idx) => {{
          const p = worldToPixel(pt.pose[0], pt.pose[1]);
          if (idx === 0) ctx.moveTo(p.px, p.py);
          else ctx.lineTo(p.px, p.py);
        }});
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // 3. Draw Executed Path (Controller)
      const samples = trialData.path_with_controller || [];
      if (document.getElementById('chkControllerPath').checked && samples.length > 0) {{
        ctx.beginPath();
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2.5 / zoom;

        samples.forEach((sample, idx) => {{
          const pose = (showAMCL && sample.pose_believed) ? sample.pose_believed : sample.pose;
          const p = worldToPixel(pose[0], pose[1]);
          if (idx === 0) ctx.moveTo(p.px, p.py);
          else ctx.lineTo(p.px, p.py);
        }});
        ctx.stroke();
      }}

      // 4. Draw Start & Goal Poses (with footprint and orientation arrows for all trials)
      if (trialData.initial_pose) {{
        drawRobotPose(trialData.initial_pose.x, trialData.initial_pose.y, trialData.initial_pose.yaw, {{
          fillColor: 'rgba(34, 197, 94, 0.25)',
          strokeColor: '#22c55e',
          arrowColor: '#22c55e',
          centerColor: '#22c55e',
          radius: 5,
          arrowLength: 26,
          arrowWidth: 3
        }});
      }}

      if (trialData.goal_pose) {{
        drawRobotPose(trialData.goal_pose.x, trialData.goal_pose.y, trialData.goal_pose.yaw, {{
          fillColor: 'rgba(239, 68, 68, 0.20)',
          strokeColor: '#ef4444',
          arrowColor: '#ef4444',
          centerColor: '#ef4444',
          radius: 5,
          arrowLength: 26,
          arrowWidth: 3,
          isDashed: true
        }});
      }}

      // 5. Draw Active Animated Robot at Current Step
      if (samples.length > 0 && currentStep < samples.length) {{
        const curr = samples[currentStep];
        const pose = (showAMCL && curr.pose_believed) ? curr.pose_believed : curr.pose;
        drawRobotPose(pose[0], pose[1], pose[2], {{
          fillColor: 'rgba(56, 189, 248, 0.40)',
          strokeColor: '#38bdf8',
          arrowColor: '#f59e0b',
          centerColor: '#a855f7',
          radius: 5,
          arrowLength: 24,
          arrowWidth: 3
        }});
      }}

      ctx.restore();
    }}

    // Telemetry Update
    function updateTelemetry(stepIdx) {{
      const samples = trialData?.path_with_controller || [];
      if (samples.length === 0) {{
        document.getElementById('valStepIdx').innerText = '0 (N/A)';
        if (trialData?.initial_pose) {{
          const ip = trialData.initial_pose;
          document.getElementById('valPoseXY').innerText = `${{ip.x.toFixed(2)}}, ${{ip.y.toFixed(2)}}`;
          document.getElementById('valYaw').innerText = `${{(ip.yaw * 180 / Math.PI).toFixed(1)}}°`;
        }} else {{
          document.getElementById('valPoseXY').innerText = '-';
          document.getElementById('valYaw').innerText = '-';
        }}
        document.getElementById('valLinVel').innerText = '0.00';
        document.getElementById('valAngVel').innerText = '0.00';
        document.getElementById('valFootprintCost').innerText = 'N/A';
        document.getElementById('valMinScan').innerText = 'N/A';
        document.getElementById('timeReadout').innerText = '0.00s';
        return;
      }}

      if (stepIdx >= samples.length) return;

      const sample = samples[stepIdx];
      const pose = (showAMCL && sample.pose_believed) ? sample.pose_believed : sample.pose;

      document.getElementById('valStepIdx').innerText = stepIdx;
      document.getElementById('valPoseXY').innerText = `${{pose[0].toFixed(2)}}, ${{pose[1].toFixed(2)}}`;
      document.getElementById('valYaw').innerText = `${{(pose[2] * 180 / Math.PI).toFixed(1)}}°`;

      const lin = sample.linear_velocity || [0, 0, 0];
      const speed = Math.hypot(lin[0], lin[1]);
      document.getElementById('valLinVel').innerText = speed.toFixed(2);

      const ang = sample.angular_velocity || [0, 0, 0];
      document.getElementById('valAngVel').innerText = ang[2].toFixed(2);

      document.getElementById('valFootprintCost').innerText = sample.footprint_cost ?? 0;
      document.getElementById('valMinScan').innerText = sample.min_scan_value ? `${{sample.min_scan_value.toFixed(2)}}m` : '-';

      // Readout
      if (samples.length > 0 && sample.timestamp && samples[0].timestamp) {{
        const dt = sample.timestamp - samples[0].timestamp;
        document.getElementById('timeReadout').innerText = `${{dt.toFixed(2)}}s`;
      }}
    }}

    // Animation & Playback Controls
    function onSliderChange(val) {{
      currentStep = parseInt(val, 10);
      updateTelemetry(currentStep);
      requestRender();
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      const btn = document.getElementById('btnPlay');
      if (isPlaying) {{
        btn.innerText = 'Pause';
        btn.style.background = 'var(--accent-amber)';
        playTimer = setInterval(advanceFrame, 100 / playbackSpeed);
      }} else {{
        btn.innerText = 'Play';
        btn.style.background = 'var(--accent-blue)';
        clearInterval(playTimer);
      }}
    }}

    function advanceFrame() {{
      const samples = trialData?.path_with_controller || [];
      if (currentStep < samples.length - 1) {{
        currentStep++;
        document.getElementById('timeSlider').value = currentStep;
        updateTelemetry(currentStep);
        requestRender();
      }} else {{
        togglePlay(); // Reached end
      }}
    }}

    function stepFrame(dir) {{
      const samples = trialData?.path_with_controller || [];
      currentStep = Math.max(0, Math.min(samples.length - 1, currentStep + dir));
      document.getElementById('timeSlider').value = currentStep;
      updateTelemetry(currentStep);
      requestRender();
    }}

    function setSpeed(spd) {{
      playbackSpeed = parseFloat(spd);
      if (isPlaying) {{
        clearInterval(playTimer);
        playTimer = setInterval(advanceFrame, 100 / playbackSpeed);
      }}
    }}

    function toggleAMCL() {{
      showAMCL = !showAMCL;
      const btn = document.getElementById('btnToggleAMCL');
      btn.innerText = `Believed Pose: ${{showAMCL ? 'ON' : 'OFF'}}`;
      btn.style.borderColor = showAMCL ? 'var(--accent-purple)' : 'var(--border-color)';
      updateTelemetry(currentStep);
      requestRender();
    }}

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {{
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') {{
        e.preventDefault();
        togglePlay();
      }} else if (e.code === 'ArrowLeft') {{
        stepFrame(-1);
      }} else if (e.code === 'ArrowRight') {{
        stepFrame(1);
      }}
    }});

    // Auto-load on init if initial_path is present
    if (document.getElementById('jsonPathInput').value.trim() !== '') {{
      loadTrialFromInput();
    }}
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="RCT Trial Motion Visualizer Server")
    parser.add_argument("--json", type=str, default="", help="Path to trial JSON file to load initially")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the HTTP server on (default 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    VisualizerRequestHandler.initial_json_path = os.path.abspath(args.json) if args.json else ""
    VisualizerRequestHandler.workspace_dir = str(Path(__file__).resolve().parents[4])

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, VisualizerRequestHandler)

    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"  RCT Trial Motion Visualizer running at: {url}")
    print(f"=======================================================\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down visualizer server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
