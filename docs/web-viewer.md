# Local browser preview (feature/web-viewer)

This branch adds MuJoCo 4.0 Studio WebViewer while keeping the desktop preview.
The backend runs the same RGB ArUco detection, 3D LiDAR fusion and AutoAim code.
A browser renders the 3D scene and receives the camera panel over Studio's
HTTP/WebSocket protocol. Everything uses **one Python process and one port**.
No Docker, separate frontend server or desktop display is required.

## Start locally

From the repository root:

```bash
python -m pip install 'mujoco==4.0.0' numpy opencv-python-headless
python scripts/web_viewer.py --port 8080
```

Open `http://localhost:8080` in Chrome/Edge or another WebGL2-capable desktop
browser on the same computer. Use a different `--port` if 8080 is occupied.
Ctrl+C stops the service. `--duration 30` provides a bounded smoke-test run.
The original `python scripts/preview_mujoco.py` desktop workflow still works.

On Linux, offscreen RGB sensing defaults to EGL. Mesa software EGL is
sufficient; no NVIDIA GPU is required. Debian/Ubuntu installations may need:

```bash
sudo apt-get install libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglib2.0-0
```

For macOS/Windows, or a Linux system using desktop OpenGL, set `MUJOCO_GL=glfw`
before launching. Those platforms were not tested in this cloud environment.
Do not set EGL on a system without EGL support.

## Experience

The Studio viewport supports orbiting, object selection and mouse perturbation.
The dockable `Gimbal camera and fusion` panel shows RGB with a laser overlay, ArUco pixel XY,
fused Z/range and aiming status. Drag the panel's title bar to reposition it.
Double-click an object to select it; Ctrl+right-drag applies a translation force,
and Ctrl+left-drag applies a torque. Watch the fixed camera and automatic aiming
follow the marker. Pause/Run and Reset are available in Studio's left sidebar.

Sensing/control targets update at 15 Hz; the camera image updates at 5 Hz to
limit texture-stream bandwidth. Physics runs at 500 Hz and scene snapshots target
60 Hz. Actual frame rate depends on software-rendering and client WebGL speed.

All tabs share **one simulation**. Studio grants a controller role to one browser,
with up to eight spectators by default. The controller sees the custom camera/UI
panel and can manipulate objects; spectators receive only the 3D scene. This is
not an isolated simulation per visitor. Close or release the controlling tab to
transfer control to another tab.

## Network scope and later deployment

This preview is intended for local use. Upstream Studio WebViewer binds all
interfaces and has no authentication; keep the port private while previewing.
A remote plain-HTTP address is also insufficient for its multithreaded WASM:
use SSH local port forwarding for a remote workstation, or HTTPS/access control
when a public server is configured later. That later setup has not been deployed.
No Docker or additional service is part of this branch.

MuJoCo Live (`https://live.mujoco.org`) is a separate model-only browser option.
It does not execute this repository's Python sensing, fusion or AutoAim logic.

## Verification

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

A real headless Chromium/WebGL client was used locally to verify the scene,
point cloud, camera texture, ArUco coordinates, fused depth and tracking status.
The test suite also checks pause/reset and sensor/overlay message handling.
