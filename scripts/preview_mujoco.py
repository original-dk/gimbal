#!/usr/bin/env python3
"""Simple MuJoCo viewer for the pan/tilt laser gimbal."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="assets/simple_pan_tilt_gimbal.xml",
        help="Path to MuJoCo MJCF XML",
    )
    parser.add_argument("--yaw", type=float, default=20.0,
                        help="Yaw target in degrees")
    parser.add_argument("--pitch", type=float, default=10.0,
                        help="Pitch target in degrees")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    yaw_target = np.deg2rad(args.yaw)
    pitch_target = np.deg2rad(args.pitch)

    print(f"Target: yaw={args.yaw:.2f} deg, pitch={args.pitch:.2f} deg")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()

            data.ctrl[0] = yaw_target
            data.ctrl[1] = pitch_target

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
