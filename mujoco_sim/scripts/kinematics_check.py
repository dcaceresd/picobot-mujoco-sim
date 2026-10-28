"""Validate the simulated cell against the real UR5e + piCOBOT geometry.

Checks:
- ``tool0 -> suction_cup_tip`` is exactly 0.370 m (real URDF offset)
- the cup axis points out of the flange (down at home) and the tip is the
  lowest point of the tool
- the tool clears the table in the home pose
- IK coverage over the table workspace
- position servo hold stability (gravity compensation sanity)
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import SimConfig
from ..env import TOP_DOWN_R, SuctionEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="white_foam")
    parser.add_argument("--grid-step", type=float, default=0.08)
    args = parser.parse_args(argv)

    env = SuctionEnv(objects=[args.object])
    mujoco = env._mj
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} - {detail}")

    # --- tool chain ---------------------------------------------------------
    tip, tip_R = env.cup_tip_pose()
    tool, _ = env.tool0_pose()
    offset = float(np.linalg.norm(tip - tool))
    check(
        "tool0 -> suction_cup_tip = 0.370 m",
        abs(offset - 0.370) < 1e-6,
        f"{offset:.6f} m",
    )

    wrist3 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    flange_dir = (tool - env.data.xpos[wrist3])
    flange_dir /= np.linalg.norm(flange_dir)
    z_axis = tip_R[:, 2]
    check(
        "cup +z axis points out of the flange",
        float(np.dot(z_axis, flange_dir)) > 0.99,
        f"dot={float(np.dot(z_axis, flange_dir)):.4f}",
    )
    check(
        "cup approaches table vertically at home",
        z_axis[2] < -0.99,
        f"tool z = {z_axis.round(3)}",
    )
    clear = tip[2] - env.cfg.table_height_m
    check(
        "home pose clears the table",
        0.05 < clear < 0.30,
        f"{clear * 100:.1f} cm above table",
    )

    # --- workspace IK coverage ---------------------------------------------
    base = np.array(env.cfg.robot_mount_xyz)
    heights = env.cfg.table_height_m + 0.12
    xs = np.arange(0.30, 0.68 + 1e-9, args.grid_step)
    ys = np.arange(-0.32, 0.32 + 1e-9, args.grid_step)
    ok_count = total = 0
    reach = []
    for x in xs:
        for y in ys:
            total += 1
            target = np.array([x, y, heights])
            _, pe, re = env.ik(target, TOP_DOWN_R)
            if pe < 1e-3 and re < 0.03:
                ok_count += 1
                reach.append(float(np.linalg.norm(target[:2] - base[:2])))
    coverage = ok_count / max(total, 1)
    check(
        "IK coverage over table > 85%",
        coverage > 0.85,
        f"{100 * coverage:.1f}% of {total} targets, "
        f"reach {min(reach):.2f}-{max(reach):.2f} m" if reach else "no solutions",
    )

    # --- servo stability ----------------------------------------------------
    env.reset()
    q0 = env.q.copy()
    env.set_ctrl(q0)
    env.hold(1.0)
    drift = float(np.max(np.abs(env.q - q0)))
    check("servo hold drift < 2 mrad", drift < 2e-3, f"{drift * 1e3:.2f} mrad")
    env.close()

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
