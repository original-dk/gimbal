"""Run with MUJOCO_GL=egl for actual offscreen RGB rendering checks."""
import os
import unittest

import mujoco
import numpy as np

from test_sensors import ROOT, preview


@unittest.skipUnless(os.environ.get('MUJOCO_GL') == 'egl', 'Requires EGL rendering')
class CameraRenderTests(unittest.TestCase):
    def test_live_scene_reflects_object_changes(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = model.camera('fixed_camera').id
        option = mujoco.MjvOption()
        with mujoco.Renderer(model, height=480, width=640, max_geom=5000) as renderer:
            def frame():
                preview.update_camera_scene(model, data, renderer.scene, camera, option, 10)
                return renderer.render().copy()
            first = frame()
            # Exercise the native HUD drawing calls in a real EGL context.
            preview.draw_marker_hud(model, data, 640, 480, renderer._mjr_context)
            # A change to the main observer's camera cannot affect the sensor camera.
            observer = mujoco.MjvCamera()
            observer.azimuth = 180
            observer.distance = 10
            np.testing.assert_array_equal(first, frame())
            # Move the visible center obstacle, mimicking a changed simulation scene.
            data.joint('scan_target_center_joint').qpos[1] += .4
            mujoco.mj_forward(model, data)
            second = frame()
            self.assertEqual(second.shape, (480, 640, 3))
            self.assertGreater(np.count_nonzero(first != second), 1000)


if __name__ == '__main__':
    unittest.main()
