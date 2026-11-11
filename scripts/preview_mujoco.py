#!/usr/bin/env python3
"""Fixed RGB camera and planar LiDAR preview for the pan/tilt gimbal."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import time

import mujoco
import numpy as np


def scan_lidar(model, data, fov=120.0, max_range=3.0, rays=121):
    """Return world origin, directions, ranges and hit flags (no hit = max range).

    Group 0 contains the environment. Group 1 contains the complete device,
    excluded from scans to prevent the housing from occluding its own sensor.
    """
    site = model.site('lidar_origin').id
    origin = data.site_xpos[site].copy()
    angles = np.linspace(-np.deg2rad(fov) / 2, np.deg2rad(fov) / 2, rays)
    local = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(rays)))
    directions = local @ data.site_xmat[site].reshape(3, 3).T
    ranges = np.full(rays, max_range, dtype=float)
    hits = np.zeros(rays, dtype=bool)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    for i, direction in enumerate(directions):
        distance = mujoco.mj_ray(model, data, origin, direction, groups, 1, -1, geom_id)
        if 0 <= distance <= max_range:
            ranges[i] = distance
            hits[i] = True
    return origin, directions, ranges, hits


def draw_scan(scene, scan, max_range):
    """Draw full range fan in cyan, measured rays green and returns red."""
    origin, directions, ranges, hits = scan

    def line(a, b, color, width=1):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('LiDAR visualization exceeds scene capacity')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
        scene.ngeom += 1

    edge = origin + directions * max_range
    line(origin, edge[0], [0, 0.8, 1, 1], 2)
    line(origin, edge[-1], [0, 0.8, 1, 1], 2)
    for a, b in zip(edge[:-1], edge[1:]):
        line(a, b, [0, 0.8, 1, 1], 2)
    for direction, distance, hit in zip(directions, ranges, hits):
        endpoint = origin + direction * distance
        line(origin, endpoint, [0.1, 0.9, 0.4, 0.45] if hit else [0, 0.7, 1, 0.12])
        if hit:
            line(endpoint - [0, 0, 0.008], endpoint + [0, 0, 0.008], [1, 0.1, 0.1, 1], 4)


def cast_laser(model, data, max_range=10.0):
    """Cast local +X from the moving aperture to the nearest environment surface."""
    site = model.site('laser_origin').id
    origin = data.site_xpos[site].copy()
    direction = data.site_xmat[site].reshape(3, 3)[:, 0].copy()
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction,
                             np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                             1, -1, geom_id)
    hit = 0 <= distance <= max_range
    endpoint = origin + direction * (distance if hit else max_range)
    return origin, endpoint, hit


def draw_laser(scene, laser):
    origin, endpoint, hit = laser
    if scene.ngeom + 2 > scene.maxgeom:
        raise RuntimeError('Laser visualization exceeds scene capacity')
    beam = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(beam, mujoco.mjtGeom.mjGEOM_LINE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), [1, 0.02, 0.02, 1])
    mujoco.mjv_connector(beam, mujoco.mjtGeom.mjGEOM_LINE, 2, origin, endpoint)
    beam.emission = 1
    scene.ngeom += 1
    if hit:
        # Offset toward the emitter to keep the spot outside the surface.
        toward_emitter = origin - endpoint
        toward_emitter /= np.linalg.norm(toward_emitter)
        spot = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(spot, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.009, 0, 0]), endpoint + toward_emitter * .002,
                           np.eye(3).ravel(), [1, 0, 0, 1])
        spot.emission = 1
        scene.ngeom += 1


def startup_control(data, target, duration):
    """Ease into the startup target with zero initial/final velocity and acceleration."""
    t = np.clip(data.time / duration, 0, 1)
    data.ctrl[:] = target * (t**3 * (10 - 15*t + 6*t*t))


def update_camera_scene(model, data, scene, camera, option, laser_range):
    """Build only the fixed camera image, independent of the interactive view."""
    mujoco.mjv_updateScene(model, data, option, None, camera,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    draw_laser(scene, cast_laser(model, data, laser_range))


def marker_pixel(model, data, width=640, height=480):
    """Project the attached marker center; return None if outside/occluded/back-facing.

    Uses simulation ground truth, not image recognition. Pixel origin: top-left.
    """
    camera = model.camera('fixed_camera').id
    site = model.site('target_icon_center').id
    point = data.site_xpos[site]
    origin = data.cam_xpos[camera]
    body = model.site_bodyid[site]
    front_normal = -data.xmat[body].reshape(3, 3)[:, 0]
    if np.dot(front_normal, origin - point) <= 0:
        return None
    local = data.cam_xmat[camera].reshape(3, 3).T @ (point - origin)
    depth = -local[2]
    if not model.vis.map.znear * model.stat.extent < depth < model.vis.map.zfar * model.stat.extent:
        return None
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
    x = width / 2 + focal * local[0] / depth
    y = height / 2 - focal * local[1] / depth
    if not (0 <= x < width and 0 <= y < height):
        return None
    direction = point - origin
    distance = np.linalg.norm(direction)
    geom_id = np.array([-1], dtype=np.int32)
    nearest = mujoco.mj_ray(model, data, origin, direction / distance,
                            np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8), 1, -1, geom_id)
    if 0 <= nearest < distance - .002:
        return None
    return float(x), float(y)


def detect_icon(rgb):
    """Find the white square/black cross using RGB pixels only."""
    import cv2
    values = rgb.astype(np.int16)
    mask = ((values.min(axis=2) > 130) & (np.ptp(values, axis=2) < 35)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (9 <= w <= rgb.shape[1] / 6 and 9 <= h <= rgb.shape[0] / 6 and .65 < w / h < 1.5):
            continue
        crop = cv2.resize(rgb[y:y+h, x:x+w], (30, 30)).mean(axis=2)
        # Cross arms must be dark; four surrounding quadrants must be bright.
        dark = np.mean([crop[15, 15], crop[15, 7], crop[15, 22], crop[7, 15], crop[22, 15]])
        white = np.mean([crop[7, 7], crop[7, 22], crop[22, 7], crop[22, 22]])
        if white > 130 and dark < white * .65:
            candidates.append({'xy': [x + (w-1)/2, y + (h-1)/2], 'bbox': [x, y, w, h]})
    return candidates[0] if len(candidates) == 1 else None


def fuse_depth(detection, camera_points, width, height, fovy):
    """Intersect the RGB ray with a vertical plane fitted to associated LiDAR hits.

    Inputs contain measurements and calibration only, no object poses or IDs.
    Optical camera coordinates: +X right, +Y down, +Z forward.
    """
    if detection is None:
        return {'valid': False, 'reason': 'icon not detected or ambiguous'}
    u, v = detection['xy']
    f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    points = np.asarray(camera_points)
    if len(points) < 3:
        return {'valid': False, 'reason': 'insufficient LiDAR returns'}
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .05)]
    projected_u = width / 2 + f * points[:, 0] / points[:, 2]
    gate = max(12, detection['bbox'][2] * .65)
    points = points[np.abs(projected_u - u) < gate]
    if len(points) < 3:
        return {'valid': False, 'reason': 'no matching LiDAR surface'}
    # Split depth discontinuities to avoid fitting through different objects.
    cuts = np.flatnonzero(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1) > .12) + 1
    groups = np.split(points, cuts)
    fits = []
    ray = np.array([(u-width/2)/f, (v-height/2)/f, 1.0])
    for group in groups:
        if len(group) < 3:
            continue
        horizontal = group[:, [0, 2]]
        center = horizontal.mean(axis=0)
        _, singular, vh = np.linalg.svd(horizontal-center, full_matrices=False)
        if singular[0] < .025 or singular[1] / singular[0] > .15:
            continue
        normal = vh[-1]
        residual = float(np.sqrt(np.mean(((horizontal-center) @ normal)**2)))
        denominator = normal @ ray[[0, 2]]
        if residual > .008 or abs(denominator) < .15:
            continue
        z = float((normal @ center) / denominator)
        hit = ray * z
        if z <= 0 or abs(hit[1] - group[:, 1].mean()) > .35:
            continue
        # Restrict intersection to the measured horizontal extent.
        if not horizontal[:, 0].min()-.015 <= hit[0] <= horizontal[:, 0].max()+.015:
            continue
        fits.append({'valid': True, 'depth_z_m': z, 'distance_m': float(np.linalg.norm(hit)),
                     'points': len(group), 'residual_m': residual,
                     'assumption': 'locally planar vertical surface'})
    if len(fits) != 1:
        return {'valid': False, 'reason': 'ambiguous or nonplanar LiDAR surface'}
    return fits[0]


def estimate_fusion(model, data, rgb, scan):
    """Use calibrated sensor poses to transform LiDAR measurements into RGB frame."""
    detection = detect_icon(rgb)
    origin, directions, ranges, hits = scan
    world_points = origin + directions[hits] * ranges[hits, None]
    camera = model.camera('fixed_camera').id
    optical_rotation = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    points = (world_points - data.cam_xpos[camera]) @ optical_rotation
    height, width = rgb.shape[:2]
    result = fuse_depth(detection, points, width, height, model.cam_fovy[camera])
    return detection, result


def draw_marker_hud(model, data, width, height, context, detection=None, fusion=None):
    pixel = detection['xy'] if detection is not None else None
    text = 'Marker: not visible' if pixel is None else f'Marker: x={pixel[0]:.1f}, y={pixel[1]:.1f} px'
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                       mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, text,
                       'Origin: top-left | RGB detection', context)
    if fusion is not None:
        depth_text = (f"Fusion Z={fusion['depth_z_m']:.3f} m | range={fusion['distance_m']:.3f} m | N={fusion['points']}"
                      if fusion['valid'] else 'Fusion unavailable: ' + fusion['reason'])
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, depth_text, 'Assumption: vertical planar surface', context)
    if pixel is not None:
        x, y = int(round(pixel[0])), height - 1 - int(round(pixel[1]))
        for dx, dy, w, h in [(-8, 0, 17, 1), (0, -8, 1, 17)]:
            mujoco.mjr_rectangle(mujoco.MjrRect(x + dx, y + dy, w, h), 0, 1, 0, 1)


class CameraWindow:
    """A separate GLFW window rendering the actual fixed-camera perspective."""
    def __init__(self, model, width=640, height=480):
        import glfw
        self.glfw = glfw
        self.window = None
        self.context = None
        self.width, self.height = width, height
        previous = glfw.get_current_context()
        if not glfw.init():
            raise RuntimeError('Cannot initialize camera display. Start a graphical desktop and check DISPLAY.')
        glfw.default_window_hints()
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, 'Fixed camera RGB (live) - Esc to close', None, None)
        if not self.window:
            raise RuntimeError('Cannot create live camera window. Check desktop/OpenGL availability.')
        try:
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            self.scene = mujoco.MjvScene(model, maxgeom=5000)
            self.option = mujoco.MjvOption()
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = model.camera('fixed_camera').id
        except BaseException:
            self.close()
            raise
        finally:
            glfw.make_context_current(previous)

    def update(self, model, data, laser_range, scan):
        glfw = self.glfw
        glfw.poll_events()
        if glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            return False
        previous = glfw.get_current_context()
        try:
            glfw.make_context_current(self.window)
            update_camera_scene(model, data, self.scene, self.camera, self.option, laser_range)
            width, height = glfw.get_framebuffer_size(self.window)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.context)
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), self.context)
            detection, fusion = estimate_fusion(model, data, rgb[::-1].copy(), scan)
            draw_marker_hud(model, data, width, height, self.context, detection, fusion)
            glfw.swap_buffers(self.window)
        finally:
            glfw.make_context_current(previous)
        return True

    def close(self):
        if self.window:
            previous = self.glfw.get_current_context()
            self.glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
            self.glfw.make_context_current(previous)
            self.glfw.destroy_window(self.window)
            self.window = None
        # Do not terminate GLFW: the main MuJoCo viewer still owns a window.


def save_frame(path, pixels):
    """Write portable RGB image without additional image dependencies."""
    height, width, _ = pixels.shape
    with open(path, 'wb') as stream:
        stream.write(f'P6\n{width} {height}\n255\n'.encode())
        stream.write(pixels.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', default='assets/simple_pan_tilt_gimbal.xml')
    parser.add_argument('--yaw', type=float, default=0)
    parser.add_argument('--pitch', type=float, default=0)
    parser.add_argument('--lidar-fov', type=float, default=120, help='Horizontal degrees')
    parser.add_argument('--lidar-range', type=float, default=3, help='Maximum meters')
    parser.add_argument('--lidar-rays', type=int, default=121)
    parser.add_argument('--laser-range', type=float, default=10, help='Laser maximum meters')
    parser.add_argument('--move-duration', type=float, default=1.5, help='Startup motion duration in seconds')
    parser.add_argument('--no-camera-window', action='store_true', help='Disable the separate live RGB window')
    parser.add_argument('--camera', action='store_true', help='View through fixed front camera')
    parser.add_argument('--headless', type=Path, help='Write camera.ppm, lidar.ppm and scan.npz to directory; use MUJOCO_GL=egl')
    args = parser.parse_args()
    if not (0 < args.lidar_fov <= 360 and np.isfinite(args.lidar_range) and args.lidar_range > 0 and 2 <= args.lidar_rays <= 1000):
        parser.error('Require 0 < FOV <= 360, positive finite range, and 2..1000 rays')
    if not (np.isfinite([args.yaw, args.pitch, args.laser_range, args.move_duration]).all() and args.laser_range > 0 and args.move_duration > 0):
        parser.error('Angles must be finite; laser range and motion duration must be positive and finite')
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    target = np.clip(np.deg2rad([args.yaw, args.pitch]), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    def scan():
        return scan_lidar(model, data, args.lidar_fov, args.lidar_range, args.lidar_rays)

    if args.headless:
        args.headless.mkdir(parents=True, exist_ok=True)
        for _ in range(int((args.move_duration + 3.5) / model.opt.timestep)):
            startup_control(data, target, args.move_duration)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        result = scan()
        with mujoco.Renderer(model, height=480, width=640, max_geom=5000) as renderer:
            renderer.update_scene(data, camera='fixed_camera')
            draw_laser(renderer.scene, cast_laser(model, data, args.laser_range))
            rgb = renderer.render().copy()
            detection, fusion = estimate_fusion(model, data, rgb, result)
            (args.headless / 'marker.json').write_text(json.dumps({'visible': detection is not None, 'xy': detection['xy'] if detection else None, 'resolution': [640, 480], 'origin': 'top-left', 'source': 'RGB detection'}, indent=2))
            (args.headless / 'fusion.json').write_text(json.dumps({'detection': detection, 'fusion': fusion}, indent=2))
            save_frame(args.headless / 'camera.ppm', rgb)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [1.1, 0, 0.1]
            camera.distance, camera.azimuth, camera.elevation = 5, 140, -65
            renderer.update_scene(data, camera=camera)
            draw_scan(renderer.scene, result, args.lidar_range)
            draw_laser(renderer.scene, cast_laser(model, data, args.laser_range))
            save_frame(args.headless / 'lidar.ppm', renderer.render())
        np.savez(args.headless / 'scan.npz', origin=result[0], directions=result[1], ranges=result[2], hits=result[3])
        print(f'Exported RGB camera, LiDAR visualization and scan: {result[3].sum()}/{args.lidar_rays} hits')
        return

    from mujoco import viewer as mj_viewer
    with ExitStack() as stack:
        viewer = stack.enter_context(mj_viewer.launch_passive(model, data))
        camera_window = None
        if not args.no_camera_window:
            camera_window = CameraWindow(model)
            stack.callback(camera_window.close)
            print('Live RGB: separate fixed-camera window. Esc closes it; main viewer stays open.')
        if args.camera:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera('fixed_camera').id
        # Physics runs at 500 Hz; expensive ray casting and viewer sync run at 60 Hz.
        # Accumulate wall time so drawing cost does not slow the simulated motors.
        previous = time.monotonic()
        accumulator = 0.0
        while viewer.is_running():
            start = time.monotonic()
            accumulator += min(start - previous, 0.1)
            previous = start
            while accumulator >= model.opt.timestep:
                if data.time <= args.move_duration + model.opt.timestep:
                    startup_control(data, target, args.move_duration)
                mujoco.mj_step(model, data)
                accumulator -= model.opt.timestep
            mujoco.mj_forward(model, data)
            scan_result = scan()
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_scan(viewer.user_scn, scan_result, args.lidar_range)
                draw_laser(viewer.user_scn, cast_laser(model, data, args.laser_range))
            viewer.sync()
            if camera_window is not None and not camera_window.update(model, data, args.laser_range, scan_result):
                camera_window.close()
                camera_window = None
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
