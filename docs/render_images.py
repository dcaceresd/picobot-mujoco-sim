"""Render the README images from the full MuJoCo scene.

Usage::

    .venv/bin/python docs/render_images.py

``payloads.png`` is the settled 11-object scene.  ``cell.png`` shows the
UR5e + piCOBOT holding the ceramic mug after a real scripted pick (approach,
descend, 15 kPa engage, lift).  Images are written to ``docs/images`` by
default at 1600x900.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

# allow ``python docs/render_images.py`` straight from a checkout
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mujoco_sim import SuctionEnv  # noqa: E402
from mujoco_sim.config import OBJ_SPECS  # noqa: E402

#: free-camera poses used for the two README images
POSES: dict[str, dict] = {
    "payloads": dict(
        lookat=(0.36, 0.0, 0.86), distance=1.25, azimuth=125.0, elevation=-24.0
    ),
    "cell": dict(
        lookat=(0.38, 0.0, 0.95), distance=1.45, azimuth=125.0, elevation=-24.0
    ),
}

#: object and lift target for the hero image (cup-tip target, world metres)
PICK_OBJECT = "ceramic_mug"
PICK_LIFT_TARGET = (0.48, -0.02, 0.92)


def _render(renderer, env, pose: dict, path: Path) -> None:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.asarray(pose["lookat"], dtype=float)
    camera.distance = float(pose["distance"])
    camera.azimuth = float(pose["azimuth"])
    camera.elevation = float(pose["elevation"])
    renderer.update_scene(env.data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    print(path)


def _pick_object(env) -> None:
    top = env.object_top_center(PICK_OBJECT)
    env.move_tip_to(top + np.array([0, 0, 0.09]), duration_s=1.0, check=False)
    env.move_tip_to(top + np.array([0, 0, -0.0002]), duration_s=1.0, check=False)
    env.hold(0.5)
    env.suction.set_cmd(15.0)
    env.hold(1.2)
    if not env.suction.part_present:
        raise RuntimeError(f"{PICK_OBJECT} pick did not seal")
    env.move_tip_to(np.asarray(PICK_LIFT_TARGET), duration_s=1.5, check=False)
    env.hold(0.5)
    if not env.suction.part_present:
        raise RuntimeError(f"{PICK_OBJECT} was lost during the lift")
    print(
        "%s held at %.3f m clearance, vacuum %.1f kPa"
        % (PICK_OBJECT, env.object_clearance_m(PICK_OBJECT),
           env.suction.vacuum_kpa)
    )


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
            # payload close-up first, while every object is still on the table
            _render(renderer, env, POSES["payloads"], args.out / "payloads.png")
            _pick_object(env)
            _render(renderer, env, POSES["cell"], args.out / "cell.png")
        finally:
            renderer.close()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
