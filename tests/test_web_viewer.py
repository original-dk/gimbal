"""Web simulation plugin behavior; run with MUJOCO_GL=egl."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco

from test_sensors import ROOT
HAS_EGL = os.environ.get('MUJOCO_GL') == 'egl'
if HAS_EGL:
    import web_viewer as web


@unittest.skipUnless(HAS_EGL, 'Requires EGL')
class WebViewerTests(unittest.TestCase):
    def test_web_native_bindings_are_checked_before_launch(self):
        web.validate_web_backend()
        with patch.object(web.importlib, 'import_module', return_value=SimpleNamespace()):
            with self.assertRaisesRegex(web.WebBackendUnavailable, 'missing native bindings'):
                web.validate_web_backend()
        with patch.object(web.importlib, 'import_module', side_effect=ImportError('DLL load failed')):
            with self.assertRaisesRegex(web.WebBackendUnavailable, 'DLL load failed'):
                web.validate_web_backend()

    def test_sensor_messages_pause_reset_and_overlays(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/simple_pan_tilt_gimbal.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sent = []
        simulation = web.GimbalSimulation()
        simulation.handle = SimpleNamespace(model=model, data=data, send_to_viewer=sent.append)
        try:
            simulation.on_model(web.messages.PostModelEvent(model=model, data=data))
            simulation.previous -= .02
            simulation.on_step(web.messages.StepEvent(model=model, data=data))
            self.assertGreater(data.time, 0)
            frame = sent[-1]
            self.assertIsInstance(frame, web.SensorSnapshot)
            self.assertTrue(frame.fusion['valid'])
            self.assertEqual(frame.status, 'tracking')
            self.assertEqual(len(frame.rgb), 640*480*3)
            panel = web.SensorPanel()
            panel.viewer = SimpleNamespace(model=model, extra_geoms=[],
                                           upload_image=lambda *args: 123)
            panel.on_sensor(frame)
            panel.on_update(web.messages.UpdateEvent())
            self.assertEqual(panel.texture, 123)
            self.assertGreater(len(panel.viewer.extra_geoms), 100)
            simulation.on_controls(web.messages.StepControlSnapshot(pause_state=web.sim.PauseState.NORMAL_PAUSED))
            before = data.time
            simulation.previous -= .02
            simulation.on_step(web.messages.StepEvent(model=model, data=data))
            self.assertEqual(data.time, before)
            simulation.on_single_step(web.messages.SingleStepEvent())
            self.assertGreater(data.time, before)
            simulation.on_controls(web.messages.StepControlSnapshot(pause_state=web.sim.PauseState.UNPAUSED, speed=50))
            self.assertFalse(simulation.paused)
            self.assertEqual(simulation.speed, .5)
            simulation.on_reset(web.messages.ResetEvent())
            self.assertEqual(simulation.next_scan, 0)
        finally:
            simulation.close()
