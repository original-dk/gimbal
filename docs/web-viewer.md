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

## Windows troubleshooting: cannot import `headless_ui`

`headless_ui` and `state_payload` are compiled submodules exported by
`mujoco.studio._studio`, then registered by `mujoco.studio.web.__init__`.
A missing import can mean these bindings were omitted from the installed build,
or that the native extension failed to load. A MuJoCo 4.0 version number does
not establish that its particular wheel has working WebViewer support.
Do not install a third-party package named `headless_ui` or copy a Linux `.so`
into Windows. Diagnose the actual native extension in the Windows venv:

```powershell
python -c "from mujoco.studio import _studio as s; print(s.__file__); print('headless_ui:', hasattr(s, 'headless_ui')); print('state_payload:', hasattr(s, 'state_payload'))"
python -m pip show mujoco mujoco-core mujoco-studio
```

If `_studio` imports but reports either binding as false, reinstalling the same
wheel cannot add those compiled capabilities. If importing `_studio` itself
fails, retain the specific DLL/import error to diagnose its runtime dependencies.
The entry point now checks these capabilities before starting a viewer thread
and gives a direct diagnostic instead of the late thread traceback.

The verified alternative is **one Python service in WSL2/Linux**, not Docker.
If WSL2/Ubuntu is already installed, open its shell and run:

```bash
cd /mnt/d/pycharm_project/simple_pan_tilt_gimbal
sudo apt-get update
sudo apt-get install -y python3-venv libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglib2.0-0
python3 -m venv ~/.venvs/gimbal-web
source ~/.venvs/gimbal-web/bin/activate
python -m pip install 'mujoco==4.0.0' numpy opencv-python-headless
python scripts/web_viewer.py --port 8080
```

Use the Linux venv, never the existing Windows `.venv`. Open
`http://localhost:8080` in the Windows browser (standard WSL2 localhost
forwarding). If WSL2 is not installed, installing/enabling it is a Windows
system setup step; this repository does not perform it automatically.
Alternatively, keep using the native desktop preview on Windows:
`python scripts/preview_mujoco.py`.
