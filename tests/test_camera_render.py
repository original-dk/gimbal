"""Run with MUJOCO_GL=egl for actual offscreen RGB rendering checks."""
import os
import unittest

import mujoco
import numpy as np

from test_sensors import ROOT, preview


@unittest.skipUnless(os.environ.get('MUJOCO_GL') == 'egl', 'Requires EGL rendering')
class CameraRenderTests(unittest.TestCase):
    def test_rgb_lidar_fusion_tracks_depth_without_object_pose_input(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, height=480, width=640, max_geom=5000) as renderer:
            for x in [1.5, 1.9]:
                data.joint('scan_target_center_joint').qpos[0] = x
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='fixed_camera')
                # Match production sensing: acquire RGB before visualization overlays.
                rgb = renderer.render().copy()
                # No marker site or object pose is passed into the fusion estimator.
                detection, result = preview.estimate_fusion(model, data, rgb, preview.scan_lidar(model, data, max_range=3))
                self.assertIsNotNone(detection)
                self.assertTrue(result['valid'], result)
                self.assertAlmostEqual(result['depth_z_m'], x-.08-.175, delta=.015)
                preview.draw_marker_hud(model, data, 640, 480, renderer._mjr_context, detection, result)
                scan = list(preview.scan_lidar(model, data))
                scan[3] = np.zeros_like(scan[3])
                _, missing = preview.estimate_fusion(model, data, rgb, scan)
                self.assertFalse(missing['valid'])
            self.assertIsNone(preview.detect_icon(np.zeros((480, 640, 3), dtype=np.uint8)))

    def test_rendered_tilted_marker_uses_3d_patch(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        angle = np.deg2rad(20)
        data.joint('scan_target_center_joint').qpos[3:7] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera='fixed_camera')
            detection, result = preview.estimate_fusion(model, data, renderer.render().copy(),
                                                        preview.scan_lidar(model, data))
            self.assertIsNotNone(detection)
            self.assertTrue(result['valid'], result)
            # Truth is used only here as an independent verification reference.
            truth = data.site_xpos[model.site('target_icon_center').id][0] - data.cam_xpos[0][0]
            self.assertAlmostEqual(result['depth_z_m'], truth, delta=.015)

    def test_auto_aim_from_rgb_and_lidar_hits_marker(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        data.joint('yaw_joint').qpos[0] = np.deg2rad(20)
        data.joint('pitch_joint').qpos[0] = np.deg2rad(10)
        mujoco.mj_forward(model, data)
        aim = preview.AutoAim(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for index in range(2000):
                if index == 1000:
                    data.joint('scan_target_center_joint').qpos[:2] += [.2, .15]
                    data.joint('scan_target_center_joint').qvel[:] = 0
                if index % 34 == 0:
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera='fixed_camera')
                    _, fusion = preview.estimate_fusion(model, data, renderer.render().copy(), preview.scan_lidar(model, data))
                    aim.observe(data, fusion)
                aim.step(data)
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        _, endpoint, hit = preview.cast_laser(model, data)
        self.assertTrue(hit)
        marker = data.site_xpos[model.site('target_icon_center').id]
        self.assertLess(np.linalg.norm(endpoint-marker), .012)
        self.assertEqual(aim.status, 'tracking')

    def test_aruco_large_tilts_and_in_plane_rotation(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for axis, degrees in [(2, 50), (1, 45), (0, 60)]:
                angle = np.deg2rad(degrees)
                quat = np.zeros(4)
                quat[0], quat[axis+1] = np.cos(angle/2), np.sin(angle/2)
                data.joint('scan_target_center_joint').qpos[3:7] = quat
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='fixed_camera')
                detection, fusion = preview.estimate_fusion(model, data, renderer.render().copy(), preview.scan_lidar(model, data))
                self.assertIsNotNone(detection, (axis, degrees))
                self.assertEqual(detection['id'], 7)
                self.assertEqual(len(detection['corners']), 4)
                np.testing.assert_allclose(detection['xy'], preview.marker_pixel(model, data), atol=2)
                self.assertTrue(fusion['valid'], (axis, degrees, fusion))

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
