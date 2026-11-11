#!/usr/bin/env python3
"""Fixed RGB camera and planar LiDAR preview for the pan/tilt gimbal."""
import argparse
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
    ranges = np.full(rays, max_range)
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
            save_frame(args.headless / 'camera.ppm', renderer.render())
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
    with mj_viewer.launch_passive(model, data) as viewer:
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
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_scan(viewer.user_scn, scan(), args.lidar_range)
                draw_laser(viewer.user_scn, cast_laser(model, data, args.laser_range))
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
