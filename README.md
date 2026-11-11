# Simple MuJoCo Pan-Tilt Gimbal

A compact MJCF model for a two-axis gimbal with a laser pointer.

## Coordinate convention

- X: forward
- Y: left
- Z: up
- `yaw_joint`: hinge around +Z
- `pitch_joint`: hinge around local -Y
- Positive pitch raises the laser direction.
- Laser optical direction: local +X of `pitch_stage` / `laser_origin`.

## Bodies and sensors

- `base`: fixed base plate and pedestal
- `yaw_stage`: yaw rotating platform and pitch supports
- `pitch_stage`: pitch shaft and laser housing
- `front_sensors`: front-mounted camera and LiDAR module, fixed to the base
- `fixed_camera`: RGB camera facing +X; image up is +Z, image right is -Y
- `lidar_origin`: origin of the simulated 3D LiDAR
- `laser_origin`: laser emission site

The model includes two position actuators:
- `yaw_position`
- `pitch_position`

## Placeholder dimensions

This is intentionally a simple starter model, not a measured CAD model.
All dimensions are in meters:
- Base plate: 240 x 200 x 30 mm (half-sizes are specified in MJCF)
- Yaw axis height: 105 mm above the base body origin
- Pitch axis: 160 mm above the yaw-stage origin
- Laser aperture: 172 mm forward of the pitch axis
- Yaw range: +/-180 degrees
- Pitch range: +/-45 degrees

Change the body positions, geom sizes, joint axes/ranges, camera position and lidar site position to match the real device.

## Install and run

```bash
pip install mujoco numpy opencv-python-headless
python scripts/preview_mujoco.py --yaw 20 --pitch 10
```

Run from this project directory so the default XML path resolves.

## Notes

- Device geometry remains visual-only. The seven movable targets collide with each other and the ground.
- The preview casts a laser ray from `laser_origin` along its moving +X axis. The red beam stops at the nearest environment surface and shows a red hit spot.
- The preview implements a 3D LiDAR using batched `mj_multiRay`. It models geometric ranges, not radar Doppler, reflectivity, noise or material effects.
- The 16x16 grid is best drawn as an image overlay initially, rather than as a collidable transparent object.
- This model has not been calibrated against the physical gimbal. Replace all placeholder dimensions before using it for real-world angle prediction.

## Fixed vision and LiDAR

Both sensors are mounted at the front of the base and do not rotate with yaw or
pitch. Camera vertical FOV is 60 degrees. LiDAR defaults to a 120-degree
horizontal FOV, 60-degree vertical FOV, 3 m maximum range and 41 layers × 121
beams (4961 rays). Cyan wireframes show the 3D sensing volume and green points
show returns (display is subsampled, measurements are not). Seven movable colored boxes, spheres and cylinders at different distances demonstrate
depth, occlusion and ranging. A checkerboard floor provides perspective cues.

```bash
# Overview with live LiDAR visualization
python scripts/preview_mujoco.py --lidar-fov 120 --lidar-range 3 --lidar-rays 121
# Look through the fixed front camera (use the viewer camera selector to switch)
python scripts/preview_mujoco.py --camera
# No display required: RGB image, overview with scan, and numeric scan data
MUJOCO_GL=egl python scripts/preview_mujoco.py --headless /tmp/gimbal-sensors
python -m unittest discover -s tests -v
```

In this cloud instance, use `/workspace/.venvs/gimbal/bin/python` for `python`.
Exports are `camera.ppm`, `lidar.ppm` and `scan.npz`. PPM is a standard RGB image
format. NPZ contains world-space `origin`, unit `directions`, meter `ranges` and
boolean `hits`. No-return rays use the maximum range with `hits=False`.
Sensor scans include static and moving geometry in geom group 0. Device geometry
uses group 1 to exclude self-returns; put new detectable obstacles in group 0.
Visualization is a scene overlay, so it does not affect measurements or physics.
The interactive camera mode shows this overlay; the exported RGB image includes the laser beam and spot, but omits the LiDAR overlay.

## Laser and smooth motion

Automatic aiming is enabled by default; manual yaw/pitch targets default to
0/0 degrees when using `--manual`. `--laser-range 10` controls maximum beam length in meters. No hit means
a full-length beam without a spot. The laser shares LiDAR's environment group
filter and excludes the device itself. Beam and spot are visual overlays, not
physical lighting or material effects.

```bash
python scripts/preview_mujoco.py --camera --yaw 0 --pitch 0
python scripts/preview_mujoco.py --manual --yaw 20 --pitch 10 --move-duration 1.5
```

Manual startup commands use a smooth quintic trajectory; position actuators have damping
ratio 1 to reduce ringing. Physics retains its 2 ms step, scans and RGB fusion update at 15 Hz, and drawing
updates at approximately 60 Hz. This avoids performing all ray casts and viewer
synchronization at every physics step. Long stalls are limited to 100 ms of
catch-up to keep the UI responsive; sustained overload can still slow simulation.
In manual mode, viewer actuator sliders remain available after the startup trajectory completes;
manual slider jumps are direct targets and do not use the startup easing.

## Separate live RGB window

Running the ordinary command now opens **two windows** by default:

```bash
python scripts/preview_mujoco.py --yaw 20 --pitch 10
```

- MuJoCo viewer: orbit, inspect and control the simulation as before.
- `Fixed camera RGB (live)`: an independent 640×480 window (framebuffer may scale
  with OS DPI), continuously rendered from `fixed_camera` using the same live
  simulation state. Main-view navigation does not move this sensor camera.
  Laser beam and hit spot are included; LiDAR visualization is omitted.

The sensor window refreshes with the main display loop, targeting 60 Hz subject
to rendering performance. Press Esc while this window has focus, or close it,
to dismiss only the RGB window. Closing the main viewer exits the program and
cleans up both windows. `--no-camera-window` disables the extra window;
`--camera` still switches the main viewer to the sensor perspective.
Both windows use GLFW. RGB icon detection additionally requires
`opencv-python-headless` (no OpenCV GUI backend is needed). A working desktop/OpenGL display is required for interactive mode.
`MUJOCO_GL=egl --headless ...` remains the separate export workflow.

To verify the RGB rendering and scene updates without a desktop:

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

## Select and move objects

The seven front objects are independent 1 kg bodies with free joints and ground
collision. They can translate/rotate, fall, roll and tip under mouse forces;
they are no longer static world geometry. The base and sensor module stay fixed.

Use the **main MuJoCo viewer**, not the RGB window:

1. Double-click the left mouse button on an object to select its body.
2. Hold Ctrl and drag with the right mouse button to translate/apply a force.
3. Hold Ctrl and drag with the left mouse button to rotate/apply a torque.
4. Add Shift to switch the movement/rotation plane.

With simulation running, manipulation applies forces; heavy objects may resist
movement because of ground friction. The RGB window is display-only and updates
as objects move. Orbiting the main view does not change the fixed sensor pose.

## Attached icon and live pixel coordinates

The front (-X) face of the central red box has a white square/black-cross icon.
Its geometry and `target_icon_center` site belong to the movable body's local
frame, so both follow translation and rotation. Run the usual preview command:

```bash
python scripts/preview_mujoco.py --yaw 20 --pitch 10
```

The independent RGB window overlays a green crosshair on the icon center and
shows `Marker: x=..., y=... px`. Coordinates use the rendered framebuffer:
origin at the top-left, X increases right, Y increases down. Typical resolution
is 640×480; OS DPI can scale the framebuffer. Move the central red box in the
main viewer to watch its coordinates change. The icon itself appears in both
windows; the coordinate HUD is shown only in the independent RGB window.

The live center is detected from RGB pixels using the white square and black
cross pattern, rather than reading the marker's simulation site. Missing or
ambiguous detections show `not visible`. Detection is a simple pattern heuristic,
not a general marker tracker: small, severely rotated, occluded or poorly lit
icons may not be detected. The `marker_pixel` projection helper remains for
simulation verification only; it is not used for live detection or depth.
Icon geometry is visual-only and excluded from laser/LiDAR ranging.
Headless exports additionally include `marker.json` with visibility, XY,
resolution and coordinate convention; the exported RGB image has no HUD.

## RGB + LiDAR fused depth

Install dependencies with `pip install -e .` or the installation command above.
The normal two-window preview now shows `Fusion Z=... m | range=... m | N=...`
in the RGB window. Z is camera optical-axis depth; range is straight-line distance
from the camera to the inferred label center. `N` is the fitted LiDAR hit count.

The camera FOV/resolution define pinhole intrinsics. The fixed camera and LiDAR
installation poses define ideal simulation extrinsics; these are known model
calibration, not a calibration procedure estimated from measurements. RGB pattern
detection supplies the label pixel. LiDAR hits are transformed into optical camera
coordinates, associated in both image X and Y near the detected icon, split at
depth discontinuities, and fitted to a local 3D plane using RANSAC and SVD. The camera ray through the detected label
intersects this plane to estimate depth. No object identity, tag world position,
true depth, or true object pose is used to fill the estimate.

The plane can tilt in any direction: there is no vertical-surface assumption.
At least six inliers with spread in two dimensions must surround the icon ray
in image space. A single scan line, sparse patch, multiple surfaces, strong
curvature, or large residual causes an unavailable result rather than a fallback
to true depth. Icon recognition can still fail under severe rotation/occlusion;
small curved patches or incorrect associations can still produce biased estimates.
The estimate measures the supporting surface, approximately 4 mm behind the
front of the decorative icon in this model.

Headless runs export `fusion.json` with detection, validity, metric estimates,
fit count/residual or failure reason. `marker.json` now also uses RGB detection.
To test actual rendering, icon detection and depth changes:

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

## 3D scan configuration and performance

Requires MuJoCo 4.0+ for the batch ray API used here.

```bash
python scripts/preview_mujoco.py --lidar-fov 120 --lidar-vfov 60 \
  --lidar-rays 121 --lidar-layers 41 --lidar-range 3 --lidar-hz 15
```

Directions are cached, ray intersections run in one native batch, and the viewer
shows at most 500 return points plus FOV wireframes. The full scan is retained for
fusion/export. Scan and RGB detection/fusion refresh together at `--lidar-hz`;
HUD coordinates/depth remain at the last measurement between refreshes. Display
and physics run independently. Setting `--lidar-layers 1` is a diagnostic planar
scan; depth fusion rejects its insufficient 3D support.

`scan.npz` includes `shape=[layers, rays]`, flattened directions/ranges/hits in
layer-major order and world-space hit `points`. Range misses retain max range
and `hits=False`. Live scan frequency is a target, not a real-time guarantee;
fast object motion can move between measurements. This models ideal geometric
returns, not spinning-scan distortion, intensity, multipath or sensor noise.

## Convert fused pixel/depth to 3D coordinates

`pixel_to_3d(x_px, y_px, estimated_z_m, intrinsics, *, frame='camera',
camera_to_world=None)` returns a NumPy `[X, Y, Z]` vector in meters. Pass
`fusion['depth_z_m']` from a valid fusion result, not straight-line range and not
simulation ground truth. K must correspond to the input image resolution.

```python
from scripts.preview_mujoco import camera_calibration, pixel_to_3d

# model/data provide only the calibrated camera parameters and pose here.
K, T_world_camera = camera_calibration(model, data, width=640, height=480)
if fusion['valid'] and detection is not None:
    x, y = detection['xy']
    z = fusion['depth_z_m']
    camera_xyz = pixel_to_3d(x, y, z, K)
    world_xyz = pixel_to_3d(x, y, z, K, frame='world',
                            camera_to_world=T_world_camera)
```

Camera optical axes are X right, Y down, Z forward; world axes are X forward,
Y left, Z up. The pinhole inverse is `P_camera = Z * inv(K) @ [x, y, 1]`, and
`P_world = R_world_camera @ P_camera + t_world_camera`. Use the camera-to-world
transform, not its inverse. For a moving camera use its pose at measurement time.
Pixels must already be undistorted for a real camera; this simulated camera has
no distortion. No object/tag pose is queried by this conversion method.

Valid live fusion results and `fusion.json` now include `point_camera_m` and
`point_world_m`; the RGB HUD also shows world XYZ. Invalid fusion results do not
produce coordinates. Measurement, association and calibration errors propagate
into XYZ: these are estimated coordinates, not corrected true positions.

## Automatic laser aiming (default)

```bash
# Detect the icon, fuse depth, reconstruct the target and smoothly aim the laser
python scripts/preview_mujoco.py
# Original explicit angle behavior
python scripts/preview_mujoco.py --manual --yaw 20 --pitch 10
```

Automatic mode consumes only valid fusion `point_world_m` measurements. The
`--yaw`, `--pitch` and `--move-duration` options apply to manual mode. Numerical
inverse kinematics uses the emitter's local +X axis and full model kinematics,
including yaw/pitch pivot and aperture offsets; it does not query the true icon
pose. The calculation/controller lives in `scripts/aiming.py`. Commands are
limited to 60 degrees/s and 180 degrees/s² with joint/actuator limits, servo
damping and gravity compensation. The independent RGB HUD shows aiming status.

Targets refresh with sensing at 15 Hz; motor commands update every physics step.
Invalid, lost, unreachable or stale (>0.25 s simulation time) measurements stop
target pursuit and hold the current commanded angle. Valid measurements reacquire
automatically. Fusion/recognition errors propagate into the aim; this is not a
path planner and an intervening object can intercept the laser. Manual control
sliders should be used only with `--manual`, since auto mode owns motor commands.

Sensing uses a dedicated 640×480 offscreen camera renderer, so closing/disabling
the RGB display does not stop automatic aiming. The display and sensor capture
share the fixed camera calibration; display DPI does not affect the control
measurement. The sensing image excludes visualization overlays. Headless mode
runs the same RGB + 3D LiDAR + control loop and exports final aiming status.
