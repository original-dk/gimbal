# Simple MuJoCo Pan-Tilt Gimbal

A compact MJCF model for a two-axis gimbal with a laser pointer.

## Coordinate convention

- X: forward
- Y: left
- Z: up
- `yaw_joint`: hinge around +Z
- `pitch_joint`: hinge around local -Y
- Positive pitch rais[simple_pan_tilt_gimbal.xml](assets/simple_pan_tilt_gimbal.xml)es the laser direction.
- Laser optical direction: local +X of `pitch_stage` / `laser_origin`.

## Bodies and sensors

- `base`: fixed base plate and pedestal
- `yaw_stage`: yaw rotating platform and pitch supports
- `pitch_stage`: pitch shaft and laser housing
- `fixed_camera`: fixed camera attached to the base
- `lidar_origin`: fixed LiDAR reference site
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
pip install mujoco
python scripts/preview_mujoco.py --yaw 20 --pitch 10
```

Run from this project directory so the default XML path resolves.

## Notes

- Geometries are currently visual-only (`contype=0`, `conaffinity=0`) to keep this first model simple and avoid self-collision complications.
- The laser housing is visual geometry. The laser ray itself should be calculated separately using MuJoCo ray casting (`mj_ray`) from `laser_origin` along the site's +X direction.
- `fixed_camera` is a MuJoCo camera. `lidar_origin` is only a reference site; a LiDAR scanner still needs to be implemented using ray casting or an external sensor plugin.
- The 16x16 grid is best drawn as an image overlay initially, rather than as a collidable transparent object.
- This model has not been calibrated against the physical gimbal. Replace all placeholder dimensions before using it for real-world angle prediction.
