import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preview', ROOT / 'scripts/preview_mujoco.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_camera_forward_and_fixed_mounts(self):
        camera = self.model.camera('fixed_camera').id
        site = self.model.site('lidar_origin').id
        position = self.data.cam_xpos[camera].copy()
        rotation = self.data.cam_xmat[camera].copy()
        lidar = self.data.site_xpos[site].copy()
        np.testing.assert_allclose(-rotation.reshape(3, 3)[:, 2], [1, 0, 0], atol=1e-12)
        self.data.joint('yaw_joint').qpos[0] = 1
        self.data.joint('pitch_joint').qpos[0] = .4
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_allclose(self.data.cam_xpos[camera], position)
        np.testing.assert_allclose(self.data.cam_xmat[camera], rotation)
        np.testing.assert_allclose(self.data.site_xpos[site], lidar)

    def test_ranges_hits_and_maximum_range(self):
        origin, directions, ranges, hits = preview.scan_lidar(self.model, self.data, layers=1)
        self.assertTrue(hits[60])
        self.assertAlmostEqual(ranges[60], 1.42 - origin[0], places=8)
        np.testing.assert_allclose(directions[60], [1, 0, 0], atol=1e-12)
        self.assertTrue((~hits).any())
        np.testing.assert_allclose(ranges[~hits], 3)
        _, _, ranges, hits = preview.scan_lidar(self.model, self.data, max_range=.5, layers=1)
        self.assertFalse(hits.any())
        np.testing.assert_allclose(ranges, .5)

    def test_laser_nearest_surface_and_moving_direction(self):
        origin, endpoint, hit = preview.cast_laser(self.model, self.data)
        self.assertTrue(hit)
        self.assertAlmostEqual(endpoint[0], 1.42, places=8)
        self.assertAlmostEqual(endpoint[1], 0, places=8)
        _, short_endpoint, hit = preview.cast_laser(self.model, self.data, .1)
        self.assertFalse(hit)
        self.assertAlmostEqual(np.linalg.norm(short_endpoint - origin), .1)
        self.data.joint('yaw_joint').qpos[0] = np.pi / 2
        self.data.joint('pitch_joint').qpos[0] = 0
        mujoco.mj_forward(self.model, self.data)
        origin, endpoint, hit = preview.cast_laser(self.model, self.data)
        self.assertFalse(hit)
        np.testing.assert_allclose((endpoint - origin) / 10, [0, 1, 0], atol=1e-12)

    def test_smooth_startup_and_motor_settling(self):
        target = np.deg2rad([20, 10])
        samples = []
        for _ in range(2500):
            preview.startup_control(self.data, target, 1.5)
            mujoco.mj_step(self.model, self.data)
            samples.append([self.data.joint('yaw_joint').qpos[0], self.data.joint('pitch_joint').qpos[0]])
        samples = np.array(samples)
        self.assertTrue(np.isfinite(samples).all())
        self.assertLess(np.max(np.diff(samples[:, 0])), .002)
        self.assertLess(samples[:, 0].max(), target[0] + .001)
        np.testing.assert_allclose(samples[-1], target, atol=.01)

    def test_objects_have_free_joints_and_respond_to_forces(self):
        body = self.model.body('scan_target_center_body').id
        joint = self.model.joint('scan_target_center_joint').id
        self.assertEqual(self.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
        start_y = self.data.joint('scan_target_center_joint').qpos[1]
        self.data.xfrc_applied[body, 1] = 15
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.assertGreater(self.data.joint('scan_target_center_joint').qpos[1], start_y + .05)
        self.data.xfrc_applied[:] = 0
        for _ in range(750):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        for name in ['scan_target_center', 'scan_target_left', 'near_sphere',
                     'middle_cylinder', 'far_box', 'far_sphere', 'far_column']:
            self.assertGreater(self.data.joint(name + '_joint').qpos[2], .05)

    def test_marker_pixels_follow_object_and_hide_when_invisible(self):
        xy = preview.marker_pixel(self.model, self.data)
        self.assertIsNotNone(xy)
        self.assertAlmostEqual(xy[0], 320)
        joint = self.data.joint('scan_target_center_joint')
        joint.qpos[1] += .2
        mujoco.mj_forward(self.model, self.data)
        moved = preview.marker_pixel(self.model, self.data)
        self.assertIsNotNone(moved)
        self.assertLess(moved[0], xy[0])
        joint.qpos[2] += .1
        mujoco.mj_forward(self.model, self.data)
        self.assertLess(preview.marker_pixel(self.model, self.data)[1], moved[1])
        joint.qpos[1] = 10
        mujoco.mj_forward(self.model, self.data)
        self.assertIsNone(preview.marker_pixel(self.model, self.data))
        joint.qpos[1] = 0
        joint.qpos[3:7] = [0, 0, 0, 1]
        mujoco.mj_forward(self.model, self.data)
        self.assertIsNone(preview.marker_pixel(self.model, self.data))

    def test_marker_occlusion(self):
        self.assertIsNotNone(preview.marker_pixel(self.model, self.data))
        # Place another body between camera and marker center.
        self.data.joint('scan_target_left_joint').qpos[:3] = [1, 0, .25]
        mujoco.mj_forward(self.model, self.data)
        self.assertIsNone(preview.marker_pixel(self.model, self.data))

    def test_integer_max_range_preserves_fractional_returns(self):
        origin, _, ranges, hits = preview.scan_lidar(self.model, self.data, max_range=3, layers=1)
        self.assertTrue(hits[60])
        self.assertAlmostEqual(ranges[60], 1.42 - origin[0], places=8)
        self.assertEqual(ranges.dtype.kind, 'f')

    def test_fusion_rejects_missing_and_nonplanar_data(self):
        detection = {'xy': [320, 168], 'bbox': [308, 156, 24, 24]}
        self.assertFalse(preview.fuse_depth(None, np.empty((0, 3)), 640, 480, 60)['valid'])
        self.assertFalse(preview.fuse_depth(detection, np.empty((0, 3)), 640, 480, 60)['valid'])
        # Degenerate coincident returns cannot define a plane.
        points = np.tile([0, -.055, 1.2], (5, 1))
        self.assertFalse(preview.fuse_depth(detection, points, 640, 480, 60)['valid'])

    def test_3d_scan_layers_and_batch_matches_individual_rays(self):
        origin, directions, ranges, hits = preview.scan_lidar(self.model, self.data)
        self.assertEqual(directions.shape, (41*121, 3))
        self.assertGreater(np.ptp(directions[:, 2]), .9)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        gid = np.zeros(1, dtype=np.int32)
        for index in [0, 60, 20*121+60, 40*121+60, 4000]:
            distance = mujoco.mj_ray(self.model, self.data, origin, directions[index], groups, 1, -1, gid)
            hit = 0 <= distance <= 3
            self.assertEqual(hits[index], hit)
            self.assertAlmostEqual(ranges[index], distance if hit else 3)

    def test_full_3d_plane_with_tilt_and_outliers(self):
        detection = {'xy': [320, 240], 'bbox': [300, 220, 40, 40]}
        x, y = np.meshgrid(np.linspace(-.07, .07, 9), np.linspace(-.07, .07, 9))
        # Tilt around both horizontal axes: impossible to infer from one scan line.
        points = np.column_stack((x.ravel(), y.ravel(), 1.4+.4*x.ravel()+.6*y.ravel()))
        outliers = points[::8].copy()
        outliers[:, 2] += .025
        result = preview.fuse_depth(detection, np.vstack([points, outliers]), 640, 480, 60)
        self.assertTrue(result['valid'], result)
        self.assertAlmostEqual(result['depth_z_m'], 1.4, places=5)
        # Remove vertical diversity: the estimator must reject a single line.
        line = points[36:45]
        self.assertFalse(preview.fuse_depth(detection, line, 640, 480, 60)['valid'])

    def test_dense_visualization_preserves_laser_capacity(self):
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        scan = preview.scan_lidar(self.model, self.data, rays=401, layers=41)
        preview.draw_scan(scene, scan, 3)
        preview.draw_laser(scene, preview.cast_laser(self.model, self.data))
        self.assertLessEqual(scene.ngeom, 1000)

    def test_visualization_contains_range_and_returns(self):
        scene = mujoco.MjvScene(self.model, maxgeom=5000)
        preview.draw_scan(scene, preview.scan_lidar(self.model, self.data), 3)
        self.assertGreater(scene.ngeom, 243)


if __name__ == '__main__':
    unittest.main()
