"""Laser inverse kinematics and rate-limited control from estimated world targets."""
import mujoco
import numpy as np


class AutoAim:
    """Consume fusion measurements; never read the marker's true position."""

    def __init__(self, model, data, speed=np.deg2rad(60), acceleration=np.deg2rad(180)):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.joints = [model.joint(name).id for name in ('yaw_joint', 'pitch_joint')]
        self.qpos = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        self.actuators = [model.actuator(name).id for name in ('yaw_position', 'pitch_position')]
        self.limits = model.jnt_range[self.joints]
        self.site = model.site('laser_origin').id
        self.command = data.qpos[self.qpos].copy()
        self.goal = self.command.copy()
        self.velocity = np.zeros(2)
        self.speed, self.acceleration = speed, acceleration
        self.last_measurement = -np.inf
        self.status = 'waiting for fusion'

    def solve(self, data, target):
        """Solve against the moving emitter frame, including all model offsets."""
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            return None
        self.scratch.qpos[:] = data.qpos
        anchor = data.xanchor[self.joints[0]]
        delta = target - anchor
        seed = np.array([np.arctan2(delta[1], delta[0]),
                         np.arctan2(delta[2], np.linalg.norm(delta[:2]))])

        def residual(angles):
            self.scratch.qpos[self.qpos] = angles
            mujoco.mj_kinematics(self.model, self.scratch)
            origin = self.scratch.site_xpos[self.site]
            direction = self.scratch.site_xmat[self.site].reshape(3, 3)[:, 0]
            vector = target-origin
            distance = np.linalg.norm(vector)
            if distance < 1e-6:
                return np.ones(3)*10
            return vector/distance-direction

        angles = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
        for _ in range(25):
            error = residual(angles)
            if np.linalg.norm(error) < 1e-7:
                return angles
            jacobian = np.column_stack([(residual(angles+np.eye(2)[i]*1e-5)-error)/1e-5
                                        for i in range(2)])
            change = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
            angles = np.clip(angles+np.clip(change, -.2, .2), self.limits[:, 0], self.limits[:, 1])
        return angles if np.linalg.norm(residual(angles)) < 1e-4 else None

    def observe(self, data, fusion):
        if not fusion or not fusion.get('valid') or 'point_world_m' not in fusion:
            self.hold('fusion unavailable')
            return
        angles = self.solve(data, fusion['point_world_m'])
        if angles is None:
            self.hold('target unreachable')
            return
        self.goal = angles
        self.last_measurement = data.time
        self.status = 'tracking'

    def hold(self, reason):
        self.goal = self.command.copy()
        self.velocity[:] = 0
        self.status = reason

    def step(self, data):
        if self.status == 'tracking' and data.time-self.last_measurement > .25:
            self.hold('measurement stale')
        dt = self.model.opt.timestep
        error = self.goal-self.command
        desired = np.sign(error)*np.minimum(self.speed, np.sqrt(2*self.acceleration*np.abs(error)))
        self.velocity += np.clip(desired-self.velocity, -self.acceleration*dt, self.acceleration*dt)
        movement = self.velocity*dt
        arrived = (np.abs(movement) >= np.abs(error)) & (movement*error >= 0)
        self.command += np.where(arrived, error, movement)
        self.velocity[arrived] = 0
        # Position servos need gravity compensation to avoid a steady pitch error.
        gains = self.model.actuator_gainprm[self.actuators, 0]
        controls = self.command + data.qfrc_bias[self.dofs]/gains
        limits = self.model.actuator_ctrlrange[self.actuators]
        data.ctrl[self.actuators] = np.clip(controls, limits[:, 0], limits[:, 1])
