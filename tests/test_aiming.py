import unittest

import mujoco
import numpy as np

from test_sensors import ROOT, preview


class AimingTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/simple_pan_tilt_gimbal.xml'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.aim = preview.AutoAim(self.model, self.data)

    def test_ik_accounts_for_emitter_and_pivot_offsets(self):
        for target in [[1.42, .2, .33], [1, -.4, .45], [2, .1, .12]]:
            angles = self.aim.solve(self.data, target)
            self.assertIsNotNone(angles)
            self.data.qpos[self.aim.qpos] = angles
            mujoco.mj_forward(self.model, self.data)
            origin, endpoint, _ = preview.cast_laser(self.model, self.data)
            direction = (endpoint-origin)/np.linalg.norm(endpoint-origin)
            np.testing.assert_allclose(direction, (target-origin)/np.linalg.norm(target-origin), atol=1e-6)

    def test_unreachable_target_and_loss_hold_command(self):
        self.assertIsNone(self.aim.solve(self.data, [0, 0, 10]))
        self.aim.observe(self.data, {'valid': True, 'point_world_m': [1.4, .2, .33]})
        for _ in range(20):
            self.aim.step(self.data)
        command = self.aim.command.copy()
        self.aim.observe(self.data, {'valid': False})
        for _ in range(20):
            self.aim.step(self.data)
        np.testing.assert_allclose(self.aim.command, command)
        self.assertEqual(self.aim.status, 'fusion unavailable')
        self.aim.observe(self.data, {'valid': True, 'point_world_m': [1.4, .2, .33]})
        self.data.time += .3
        self.aim.step(self.data)
        self.assertEqual(self.aim.status, 'measurement stale')

    def test_servo_tracks_fused_target_with_gravity_compensation(self):
        target = np.array([1.42, .10, .33])
        self.aim.observe(self.data, {'valid': True, 'point_world_m': target})
        previous = self.aim.command.copy()
        for index in range(2500):
            if index % 30 == 0:
                self.aim.observe(self.data, {'valid': True, 'point_world_m': target})
            self.aim.step(self.data)
            self.assertLessEqual(np.max(np.abs(self.aim.command-previous)), self.aim.speed*.002+1e-9)
            previous = self.aim.command.copy()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        origin, endpoint, _ = preview.cast_laser(self.model, self.data)
        direction = (endpoint-origin)/np.linalg.norm(endpoint-origin)
        self.assertLess(np.linalg.norm(np.cross(target-origin, direction)), .002)
