"""Render the README images from the full MuJoCo scene.

Usage::

    .venv/bin/python docs/render_images.py

The images are written to ``docs/images`` by default.  They are ordinary
MuJoCo renders (no post-processing) of the 11-object scene at 1600x900.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mujoco_sim import SuctionEnv
from mujoco_sim.config import OBJ_SPECS

#: free-camera poses used for the two README images
POSES: dict[str, dict] = {
    "cell": dict(
        lookat=(0.34, 0.0, 0.88), distance=1.35, azimuth=120.0, elevation=-25.0
    ),
    "payloads": dict(
        lookat=(0.36, 0.0, 0.86), distance=1.25, azimuth=125.0, elevation=-24.0
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/images"),
        help="output directory (default: docs/images)",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    env = SuctionEnv(objects=list(OBJ_SPECS))
    try:
        env.hold(0.5)  # let the payloads settle on the table
        env.model.vis.global_.offwidth = args.width
        env.model.vis.global_.offheight = args.height
        renderer = mujoco.Renderer(env.model, args.height, args.width)
        try:
            for name, pose in POSES.items():
                camera = mujoco.MjvCamera()
                camera.lookat[:] = np.asarray(pose["lookat"], dtype=float)
                camera.distance = float(pose["distance"])
                camera.azimuth = float(pose["azimuth"])
                camera.elevation = float(pose["elevation"])
                renderer.update_scene(env.data, camera=camera)
                path = args.out / f"{name}.png"
                Image.fromarray(renderer.render()).save(path)
                print(path)
        finally:
            renderer.close()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
