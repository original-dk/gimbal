#!/usr/bin/env python3
"""MuJoCo Studio WebViewer with live camera, 3D LiDAR and automatic aiming."""
import argparse
import math
import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass

# Sensor observations need offscreen rendering; the browser renders the 3D view.
os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform.startswith('linux') else 'glfw')

import mujoco
from mujoco.studio import app, launch_web, messages, plugin, sim, viewer_protocol
from mujoco.studio.dear_imgui import dear_imgui as imgui

if __package__:
    from . import preview_mujoco as gimbal
else:
    import preview_mujoco as gimbal


@dataclass(frozen=True)
class SensorSnapshot(messages.Snapshot):
    scan: tuple
    laser: tuple
    detection: dict | None
    fusion: dict
    status: str
    rgb: bytes | None


class GimbalSimulation(plugin.SimPlugin):
    """Simulation and sensing stay on the calling thread, never the UI thread."""

    def __init__(self):
        self.renderer = None
        self.paused = False
        self.speed = 1.0
        self.previous = time.monotonic()
        self.accumulator = 0.0
        self.next_scan = self.next_image = 0.0

    @messages.handler
    def on_model(self, event: messages.PostModelEvent):
        self.close()
        self.aim = gimbal.AutoAim(event.model, event.data)
        self.renderer = mujoco.Renderer(event.model, height=480, width=640)
        self.previous = time.monotonic()
        self.accumulator = 0.0
        self.next_scan = self.next_image = 0.0

    @messages.handler
    def on_reset(self, _: messages.ResetEvent):
        if self.handle.model is not None:
            self.aim = gimbal.AutoAim(self.handle.model, self.handle.data)
            self.next_scan = self.next_image = 0.0

    @messages.handler
    def on_pause(self, event: messages.RequestPauseEvent):
        self.paused = event.pause_state != sim.PauseState.UNPAUSED
        self.handle.send_to_viewer(event)

    @messages.handler
    def on_controls(self, controls: messages.StepControlSnapshot):
        self.paused = controls.pause_state != sim.PauseState.UNPAUSED
        if math.isfinite(controls.speed):
            self.speed = max(0, min(1000, controls.speed))/100
        self.handle.send_to_viewer(messages.StepControlSnapshot(
            pause_state=controls.pause_state, speed=self.speed*100))

    @messages.handler
    def on_single_step(self, _: messages.SingleStepEvent):
        if self.paused and self.handle.model is not None:
            self.aim.step(self.handle.data)
            mujoco.mj_step(self.handle.model, self.handle.data)
            self.next_scan = 0.0

    @messages.handler
    def on_step(self, event: messages.StepEvent):
        model, data = event.model, event.data
        now = time.monotonic()
        elapsed = min(now-self.previous, .1)
        self.previous = now
        self.accumulator = 0 if self.paused else self.accumulator+elapsed*self.speed
        while self.accumulator >= model.opt.timestep:
            self.aim.step(data)
            mujoco.mj_step(model, data)
            self.accumulator -= model.opt.timestep
        mujoco.mj_forward(model, data)
        if now < self.next_scan:
            return
        scan = gimbal.scan_lidar(model, data)
        self.renderer.update_scene(data, camera='fixed_camera')
        rgb = self.renderer.render().copy()
        detection, fusion = gimbal.estimate_fusion(model, data, rgb, scan)
        self.aim.observe(data, fusion)
        image = None
        if now >= self.next_image:
            gimbal.draw_laser(self.renderer.scene, gimbal.cast_laser(model, data))
            image = self.renderer.render().tobytes()
            self.next_image = now+1/5
        self.handle.send_to_viewer(SensorSnapshot(scan, gimbal.cast_laser(model, data),
                                                detection, fusion, self.aim.status, image))
        self.next_scan = now+1/15

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class SensorPanel(plugin.ViewerPlugin):
    """Dockable browser panel and transient scene overlays using Studio APIs."""

    def __init__(self):
        self.frame = None
        self.pending_rgb = None
        self.texture = 0
        self.scene = None

    @messages.handler
    def on_sensor(self, frame: SensorSnapshot):
        self.frame = frame
        if frame.rgb is not None:
            self.pending_rgb = frame.rgb

    @messages.handler
    def on_update(self, _: messages.UpdateEvent):
        if self.frame is None or self.viewer.model is None:
            return
        if self.pending_rgb is not None:
            self.texture = self.viewer.upload_image(self.texture, self.pending_rgb, 640, 480, 3)
            self.pending_rgb = None
        if self.scene is None:
            self.scene = mujoco.MjvScene(self.viewer.model, maxgeom=1500)
        self.scene.ngeom = 0
        gimbal.draw_scan(self.scene, self.frame.scan, 3)
        gimbal.draw_laser(self.scene, self.frame.laser)
        self.viewer.extra_geoms.clear()
        self.viewer.extra_geoms.extend(self.scene.geoms[:self.scene.ngeom])

    @messages.handler
    def on_gui(self, _: messages.BuildGuiEvent):
        imgui.SetNextWindowSize(imgui.Vec2(680, 640), imgui.Cond.FirstUseEver)
        imgui.SetNextWindowPos(imgui.Vec2(750, 60), imgui.Cond.FirstUseEver)
        visible, _ = imgui.Begin('Gimbal camera and fusion')
        if visible:
            imgui.TextUnformatted('Drag objects in the 3D viewport. The camera remains fixed.')
            if self.frame is not None:
                frame = self.frame
                imgui.TextUnformatted('Auto aim: '+frame.status)
                if frame.detection:
                    x, y = frame.detection['xy']
                    imgui.TextUnformatted(f'ArUco 7: x={x:.1f}, y={y:.1f} px')
                if frame.fusion['valid']:
                    imgui.TextUnformatted(f"Depth Z={frame.fusion['depth_z_m']:.3f} m; range={frame.fusion['distance_m']:.3f} m")
                else:
                    imgui.TextUnformatted('Fusion unavailable: '+frame.fusion['reason'])
            if self.texture:
                width = max(64, min(640, imgui.GetContentRegionAvail().x))
                imgui.Image(self.texture, imgui.Vec2(width, width*.75))
        imgui.End()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--duration', type=float, help='Optional bounded smoke-test duration in seconds')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.duration is not None and args.duration <= 0):
        parser.error('Require a valid port and positive duration')
    path = Path(__file__).resolve().parents[1]/'assets/simple_pan_tilt_gimbal.xml'
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    simulation = GimbalSimulation()
    config = viewer_protocol.ViewerConfig(title='Gimbal: RGB + 3D LiDAR auto aim',
                                          gfx='web', http_port=args.port)
    try:
        with launch_web.launch(config, plugins=[app.App(), SensorPanel(), simulation]) as handle:
            handle.set_model(model, path=str(path), data=data)
            deadline = time.monotonic()+args.duration if args.duration else float('inf')
            while handle.is_running() and time.monotonic() < deadline:
                started = time.monotonic()
                handle.sync()
                time.sleep(max(0, 1/60-(time.monotonic()-started)))
    finally:
        simulation.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
