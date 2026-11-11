#!/usr/bin/env python3
"""Fixed RGB camera and 3D LiDAR preview for the pan/tilt gimbal."""
import argparse
import json
from functools import lru_cache
from contextlib import ExitStack
from pathlib import Path
import time

import mujoco
import numpy as np


@lru_cache(maxsize=16)
def lidar_directions(fov, rays, vertical_fov, layers):
    yaw, pitch = np.meshgrid(np.linspace(-np.deg2rad(fov)/2, np.deg2rad(fov)/2, rays),
                             np.linspace(-np.deg2rad(vertical_fov)/2, np.deg2rad(vertical_fov)/2, layers)
                             if layers > 1 else np.array([0.0]))
    directions = np.column_stack((np.cos(pitch).ravel()*np.cos(yaw).ravel(),
                                  np.cos(pitch).ravel()*np.sin(yaw).ravel(), np.sin(pitch).ravel()))
    directions.setflags(write=False)
    return directions


def scan_lidar(model, data, fov=120.0, max_range=3.0, rays=121, vertical_fov=60.0, layers=41):
    """Batched 3D scan. Flattening order: vertical layer, then horizontal beam."""
    site = model.site('lidar_origin').id
    origin = data.site_xpos[site].copy()
    directions = np.ascontiguousarray(lidar_directions(fov, rays, vertical_fov, layers)
                                      @ data.site_xmat[site].reshape(3, 3).T)
    count = len(directions)
    distances = np.full(count, -1.0)
    geom_ids = np.full(count, -1, dtype=np.int32)
    mujoco.mj_multiRay(model, data, origin, directions.ravel(),
                       np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1,
                       geom_ids, distances, None, count, max_range)
    hits = (distances >= 0) & (distances <= max_range)
    return origin, directions, np.where(hits, distances, float(max_range)), hits


def draw_scan(scene, scan, max_range, layers=41):
    """Bounded overlay: cyan 3D FOV wireframe, green returns and sparse beams."""
    origin, directions, ranges, hits = scan
    def line(a, b, color):
        if scene.ngeom >= scene.maxgeom - 8:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1, a, b)
        scene.ngeom += 1
    if len(directions) % layers:
        layers = 1
    grid = (origin + directions * max_range).reshape(layers, -1, 3)
    for row in sorted(set([0, layers//2, layers-1])):
        for a, b in zip(grid[row, :-1], grid[row, 1:]):
            line(a, b, [0, .8, 1, .5])
    for col in [0, grid.shape[1]-1]:
        for a, b in zip(grid[:-1, col], grid[1:, col]):
            line(a, b, [0, .8, 1, .5])
        line(origin, grid[0, col], [0, .8, 1, .5])
        line(origin, grid[-1, col], [0, .8, 1, .5])
    indices = np.flatnonzero(hits)
    # Reserve room for the laser and cap visual points, independent of scan density.
    budget = max(0, min(500, scene.maxgeom - scene.ngeom - 8))
    if len(indices) > budget:
        indices = indices[np.linspace(0, len(indices)-1, budget, dtype=int)] if budget else []
    for index in indices:
        point = origin + directions[index]*ranges[index]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([.006, 0, 0]), point, np.eye(3).ravel(), [.1, 1, .4, 1])
        scene.ngeom += 1


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
    mask = ((values.min(axis=2) > 85) & (np.ptp(values, axis=2) < 35)).astype(np.uint8) * 255
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
        if white > 90 and dark < white * .65:
            candidates.append({'xy': [x + (w-1)/2, y + (h-1)/2], 'bbox': [x, y, w, h]})
    return candidates[0] if len(candidates) == 1 else None


def fuse_depth(detection, camera_points, width, height, fovy):
    """Fit a local 3D plane from RGB-associated LiDAR points; intersect icon ray."""
    import cv2
    fail = lambda reason: {'valid': False, 'reason': reason}
    if detection is None:
        return fail('icon not detected or ambiguous')
    u, v = detection['xy']
    focal = height / (2*np.tan(np.deg2rad(fovy)/2))
    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .05)]
    if len(points) < 6:
        return fail('insufficient 3D LiDAR returns')
    pixels = np.column_stack((width/2+focal*points[:, 0]/points[:, 2],
                              height/2+focal*points[:, 1]/points[:, 2]))
    gates = np.maximum(12, np.array(detection['bbox'][2:])*0.85)
    selected = (np.abs(pixels-[u, v]) < gates).all(axis=1)
    points, pixels = points[selected], pixels[selected]
    if len(points) < 6:
        return fail('no matching 3D LiDAR patch')
    # Separate foreground/background before robust local fitting.
    order = np.argsort(points[:, 2])
    points, pixels = points[order], pixels[order]
    groups = np.split(np.arange(len(points)), np.flatnonzero(np.diff(points[:, 2]) > .08)+1)
    ray = np.array([(u-width/2)/focal, (v-height/2)/focal, 1.0])
    fits = []
    rng = np.random.default_rng(0)
    for group in groups:
        patch, uv = points[group], pixels[group]
        if len(patch) < 6:
            continue
        triples = np.array([rng.choice(len(patch), 3, replace=False) for _ in range(64)])
        a, b, c = (patch[triples[:, i]] for i in range(3))
        normals = np.cross(b-a, c-a)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        if not valid.any():
            continue
        normals, a = normals[valid]/lengths[valid, None], a[valid]
        errors = np.abs(patch @ normals.T - np.sum(a*normals, axis=1))
        mask = errors[:, np.argmax((errors < .006).sum(axis=0))] < .006
        if mask.sum() < 6 or mask.mean() < .7:
            continue
        inliers = patch[mask]
        center = inliers.mean(axis=0)
        _, singular, vh = np.linalg.svd(inliers-center, full_matrices=False)
        # A single scan line cannot constrain a 3D plane.
        if singular[1] < .012 or singular[2]/singular[1] > .15:
            continue
        normal = vh[-1]
        residual = float(np.sqrt(np.mean(((inliers-center) @ normal)**2)))
        denominator = normal @ ray
        if residual > .006 or abs(denominator) < .15:
            continue
        z = float((normal @ center)/denominator)
        hull = cv2.convexHull(uv[mask].astype(np.float32))
        if z <= 0 or cv2.pointPolygonTest(hull, (float(u), float(v)), False) < 0:
            continue
        fits.append({'valid': True, 'depth_z_m': z, 'distance_m': float(z*np.linalg.norm(ray)),
                     'points': int(mask.sum()), 'residual_m': residual,
                     'assumption': 'local 3D plane', 'normal_camera': normal.tolist()})
    return fits[0] if len(fits) == 1 else fail('ambiguous, sparse or nonplanar 3D surface')


def camera_calibration(model, data, width=640, height=480):
    """Return pinhole K and optical-camera-to-world rigid transform from calibration."""
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    camera = model.camera('fixed_camera').id
    focal = height / (2*np.tan(np.deg2rad(model.cam_fovy[camera])/2))
    intrinsics = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    transform[:3, 3] = data.cam_xpos[camera]
    return intrinsics, transform


def pixel_to_3d(x_px, y_px, estimated_z_m, intrinsics, *, frame='camera', camera_to_world=None):
    """Back-project a pixel using *estimated optical-axis depth* in meters.

    Camera axes: X right, Y down, Z forward. K must match the actual image
    resolution. World output requires the calibrated 4x4 camera-to-world pose.
    This method accepts measured depth only; it has no access to scene objects.
    """
    values = np.asarray([x_px, y_px, estimated_z_m], dtype=float)
    if not np.isfinite(values).all() or estimated_z_m <= 0:
        raise ValueError('Pixel coordinates must be finite and estimated Z must be positive and finite')
    k = np.asarray(intrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Expected a finite 3x3 pinhole intrinsic matrix with positive focal lengths')
    try:
        ray = np.linalg.solve(k, [x_px, y_px, 1.])
    except np.linalg.LinAlgError as error:
        raise ValueError('Intrinsic matrix must be invertible') from error
    point = ray * (estimated_z_m / ray[2])
    if frame == 'camera':
        return point
    if frame != 'world':
        raise ValueError("frame must be 'camera' or 'world'")
    transform = np.asarray(camera_to_world, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('World output requires a finite calibrated 4x4 camera-to-world transform')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Camera-to-world rotation must be a proper rigid rotation')
    return rotation @ point + transform[:3, 3]


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
    if result['valid']:
        intrinsics, transform = camera_calibration(model, data, width, height)
        x, y = detection['xy']
        estimated_z = result['depth_z_m']
        result['point_camera_m'] = pixel_to_3d(x, y, estimated_z, intrinsics).tolist()
        result['point_world_m'] = pixel_to_3d(x, y, estimated_z, intrinsics,
                                            frame='world', camera_to_world=transform).tolist()
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
                           viewport, depth_text, ('World XYZ: ' + ', '.join(f'{v:.3f}' for v in fusion['point_world_m']) + ' m'
                                                 if fusion['valid'] and 'point_world_m' in fusion
                                                 else 'Local 3D plane | RGB + 3D LiDAR'), context)
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
        self.measurement = (None, None)
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

    def update(self, model, data, laser_range, scan, fresh_scan=True):
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
            if fresh_scan:
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), self.context)
                self.measurement = estimate_fusion(model, data, rgb[::-1].copy(), scan)
            detection, fusion = self.measurement
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
    parser.add_argument('--lidar-rays', type=int, default=121, help='Beams per vertical layer')
    parser.add_argument('--lidar-layers', type=int, default=41)
    parser.add_argument('--lidar-vfov', type=float, default=60, help='Vertical degrees')
    parser.add_argument('--lidar-hz', type=float, default=15, help='Scan and fusion refresh rate')
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
    if not (1 <= args.lidar_layers <= 128 and 0 < args.lidar_vfov < 180 and 0 < args.lidar_hz <= 60 and args.lidar_rays*args.lidar_layers <= 50000):
        parser.error('Require 1..128 layers, 0 < vertical FOV < 180, 0 < scan Hz <= 60, at most 50000 beams')
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    target = np.clip(np.deg2rad([args.yaw, args.pitch]), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    def scan():
        return scan_lidar(model, data, args.lidar_fov, args.lidar_range, args.lidar_rays, args.lidar_vfov, args.lidar_layers)

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
            camera.distance, camera.azimuth, camera.elevation = 6.5, 140, -35
            renderer.update_scene(data, camera=camera)
            draw_scan(renderer.scene, result, args.lidar_range, args.lidar_layers)
            draw_laser(renderer.scene, cast_laser(model, data, args.laser_range))
            save_frame(args.headless / 'lidar.ppm', renderer.render())
        np.savez(args.headless / 'scan.npz', origin=result[0], directions=result[1], ranges=result[2], hits=result[3], shape=[args.lidar_layers, args.lidar_rays], points=result[0]+result[1][result[3]]*result[2][result[3], None])
        print(f'Exported RGB camera, LiDAR visualization and scan: {result[3].sum()}/{args.lidar_rays*args.lidar_layers} hits')
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
        scan_result = scan()
        next_scan = 0.0
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
            fresh_scan = start >= next_scan
            if fresh_scan:
                scan_result = scan()
                next_scan = start + 1 / args.lidar_hz
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_scan(viewer.user_scn, scan_result, args.lidar_range, args.lidar_layers)
                draw_laser(viewer.user_scn, cast_laser(model, data, args.laser_range))
            viewer.sync()
            if camera_window is not None and not camera_window.update(model, data, args.laser_range, scan_result, fresh_scan):
                camera_window.close()
                camera_window = None
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
