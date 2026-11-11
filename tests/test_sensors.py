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
        self.data.qpos[:] = [1, .4]
        mujoco.mj_forward(self.model, self.data)
        np.testing.assert_allclose(self.data.cam_xpos[camera], position)
        np.testing.assert_allclose(self.data.cam_xmat[camera], rotation)
        np.testing.assert_allclose(self.data.site_xpos[site], lidar)

    def test_ranges_hits_and_maximum_range(self):
        origin, directions, ranges, hits = preview.scan_lidar(self.model, self.data)
        self.assertTrue(hits[60])
        self.assertAlmostEqual(ranges[60], 1.42 - origin[0], places=8)
        np.testing.assert_allclose(directions[60], [1, 0, 0], atol=1e-12)
        self.assertTrue((~hits).any())
        np.testing.assert_allclose(ranges[~hits], 3)
        _, _, ranges, hits = preview.scan_lidar(self.model, self.data, max_range=.5)
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
        self.data.qpos[:] = [np.pi / 2, 0]
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
            samples.append(self.data.qpos.copy())
        samples = np.array(samples)
        self.assertTrue(np.isfinite(samples).all())
        self.assertLess(np.max(np.diff(samples[:, 0])), .002)
        self.assertLess(samples[:, 0].max(), target[0] + .001)
        np.testing.assert_allclose(samples[-1], target, atol=.01)

    def test_visualization_contains_range_and_returns(self):
        scene = mujoco.MjvScene(self.model, maxgeom=5000)
        preview.draw_scan(scene, preview.scan_lidar(self.model, self.data), 3)
        self.assertGreater(scene.ngeom, 243)


if __name__ == '__main__':
    unittest.main()
