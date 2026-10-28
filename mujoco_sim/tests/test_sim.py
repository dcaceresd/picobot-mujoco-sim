"""Tests for the MuJoCo suction simulation.

Run with the dedicated venv::

    .venv/bin/python -m pytest mujoco_sim/tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mujoco_sim import (  # noqa: E402
    PicobotParams,
    PumpModel,
    SuctionEnv,
)
from mujoco_sim.config import object_spec  # noqa: E402


# ---------------------------------------------------------------------------
# pure pump model (no physics)
# ---------------------------------------------------------------------------


def test_pump_engage_transient_and_steady_level():
    pump = PumpModel(PicobotParams())
    assert pump.set_cmd(12.0) == 12.0
    for _ in range(2000):  # 4 s at 2 ms
        pump.step(0.002, sealed=True, seal_quality=1.0)
    assert pump.peak_kpa >= 45.0          # engage overshoot ~40+0.7*12
    assert abs(pump.vacuum_kpa - 13.0) < 1.0  # settles to cmd + 1
    assert pump.part_present              # PP latched by the overshoot
    pump.release()
    for _ in range(500):
        pump.step(0.002, sealed=True, seal_quality=1.0)
    assert pump.vacuum_kpa < 0.5
    assert not pump.part_present


def test_pump_cannot_build_vacuum_unsealed():
    pump = PumpModel(PicobotParams())
    pump.set_cmd(20.0)
    for _ in range(1000):
        pump.step(0.002, sealed=False)
    assert pump.vacuum_kpa < 0.5
    assert not pump.part_present


def test_pump_clamps_command():
    pump = PumpModel(PicobotParams())
    assert pump.set_cmd(100.0) == 65.0
    assert pump.set_cmd(1.0) == 5.0
    pump.release()
    assert pump.set_cmd(0.0) == 0.0 and pump.cmd_kpa is None


# ---------------------------------------------------------------------------
# model / kinematics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    e = SuctionEnv(objects=["white_foam", "heavy_cube"])
    yield e
    e.close()


def test_tool_geometry(env):
    tip, R = env.cup_tip_pose()
    tool, _ = env.tool0_pose()
    assert math.isclose(np.linalg.norm(tip - tool), 0.370, abs_tol=1e-6)
    # cup looks down at home
    assert R[2, 2] < -0.99
    # home pose clears the table
    assert tip[2] > env.cfg.table_height_m + 0.05


def test_real_picobot_hardware(env):
    """Official piCOBOT Electric / B30-2 dimensions and the calibrated camera."""
    import mujoco

    m = env.model
    d = env.data
    cfg = env.cfg
    # piCOBOT body envelope + extension derived from the measured 0.370 m tip
    assert math.isclose(cfg.tool_body_bottom_m, 0.1222, abs_tol=1e-9)
    assert math.isclose(cfg.cup_top_z, 0.3438, abs_tol=1e-9)
    assert math.isclose(cfg.extension_length_m, 0.2216, abs_tol=1e-9)

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "picobot")
    cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cup_seal")
    # 1.43 kg pump + adapter + tube/collar + wrist camera mount
    assert 1.40 < float(m.body_mass[bid]) < 1.55
    assert math.isclose(float(m.body_mass[cup]), 0.017, abs_tol=1e-6)

    # wrist camera is exactly at the hand-eye calibration pose, looking down
    tool_pos, tool_R = env.tool0_pose()
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    expected = tool_pos + tool_R @ np.asarray(cfg.wrist_cam_pos_tool)
    assert np.linalg.norm(d.cam_xpos[cid] - expected) < 1e-6
    cam_z = d.cam_xmat[cid].reshape(3, 3)[:, 2]
    assert float(np.dot(cam_z, tool_R[:, 2])) < -0.999  # camera z is up

    # Orbbec stream resolution used by the recordings
    assert (env.cameras.intrinsics.width, env.cameras.intrinsics.height) == (1280, 720)



def test_ik_reaches_workspace(env):
    env.reset()
    top = env.object_top_center("white_foam")
    q, pos_err, rot_err = env.ik(top + np.array([0, 0, 0.12]))
    assert pos_err < 1e-3
    assert rot_err < 0.03
    assert q.shape == (6,)


# ---------------------------------------------------------------------------
# suction physics end-to-end
# ---------------------------------------------------------------------------


def test_pick_and_release(env):
    env.reset()
    top = env.object_top_center("white_foam")
    env.move_tip_to(top + np.array([0, 0, 0.09]), duration_s=1.0, check=False)
    env.move_tip_to(top + np.array([0, 0, -0.0002]), duration_s=0.8, check=False)
    env.suction.set_cmd(15.0)
    env.hold(1.0)
    assert env.suction.vacuum_kpa > 20.0
    assert env.suction.part_present

    env.move_tip_to(top + np.array([0, 0, 0.05]), duration_s=1.2, check=False)
    env.hold(1.0)
    clearance = env.object_clearance_m("white_foam")
    lin, ang = env.object_speed("white_foam")
    assert clearance > 0.03
    assert lin < 0.3 and ang < 1.5
    assert env.suction.part_present
    assert env.suction.pop_off_count == 0

    env.suction.release()
    env.hold(1.2)
    assert env.object_clearance_m("white_foam") < 0.03
    assert not env.suction.part_present


def test_heavy_cube_breaks_away_at_low_pressure(env):
    env.reset()
    top = env.object_top_center("heavy_cube")
    env.move_tip_to(top + np.array([0, 0, 0.09]), duration_s=1.0, check=False)
    env.move_tip_to(top + np.array([0, 0, -0.0002]), duration_s=0.8, check=False)
    env.suction.set_cmd(5.0)
    env.hold(1.0)
    env.move_tip_to(top + np.array([0, 0, 0.05]), duration_s=1.2, check=False)
    env.hold(1.5)
    assert env.object_clearance_m("heavy_cube") < 0.03  # never lifted
    assert not env.suction.part_present


def test_payload_visuals_and_spawn_clearance():
    """Every payload has a mesh asset, sits on the table, and has a valid pick."""
    from mujoco_sim.config import ASSETS_DIR, OBJ_SPECS

    for spec in OBJ_SPECS.values():
        assert spec.visual_meshes, f"{spec.name} has no visual mesh"
        for vm in spec.visual_meshes:
            path = ASSETS_DIR / "objects" / f"{vm.file}.obj"
            assert path.is_file(), f"missing payload mesh {path}"

    with SuctionEnv(objects=list(OBJ_SPECS)) as env:
        for name, spec in OBJ_SPECS.items():
            assert env.object_clearance_m(name) > -1e-3
            top = env.object_top_center(name)
            lo, hi = env.object_aabb(name)
            assert lo[0] - 1e-3 <= top[0] <= hi[0] + 1e-3
            assert lo[1] - 1e-3 <= top[1] <= hi[1] + 1e-3
            assert (
                lo[2] - 1e-3
                <= top[2]
                <= hi[2] + spec.soft_skin_height_m + 1e-3
            )


def test_bowl_interior_pick():
    """The bowl is hollow; the cup must reach the interior floor, not the rim."""
    with SuctionEnv(objects=["white_bowl"]) as env:
        spec = object_spec("white_bowl")
        top = env.object_top_center("white_bowl")
        floor = env.cfg.table_height_m + 0.008
        assert abs(float(top[2]) - floor) < 2e-3
        assert top[2] < env.cfg.table_height_m + spec.top_offset - 0.03

        env.move_tip_to(top + np.array([0, 0, 0.09]), duration_s=1.0, check=False)
        env.move_tip_to(top + np.array([0, 0, -0.0002]), duration_s=1.0, check=False)
        env.suction.set_cmd(15.0)
        env.hold(1.0)
        assert env.suction.part_present
        env.move_tip_to(top + np.array([0, 0, 0.05]), duration_s=1.2, check=False)
        env.hold(0.8)
        assert env.object_clearance_m("white_bowl") > 0.03
        assert env.suction.part_present


# ---------------------------------------------------------------------------
# interactive viewer logic (headless; no GUI needed)
# ---------------------------------------------------------------------------


def test_viewer_command_parsing():
    from mujoco_sim.scripts.view_sim import (
        keycode_to_command,
        parse_command,
    )

    assert parse_command("1") == ("select", 0)
    assert parse_command("D") == ("descend", None)  # GLFW sends uppercase
    assert parse_command("V") == ("vacuum_toggle", None)
    assert parse_command("v") == ("vacuum_toggle", None)
    assert parse_command("+") == ("setpoint", 1.0)
    assert parse_command("-") == ("setpoint", -1.0)
    assert parse_command("set 20") == ("set_kpa", 20.0)
    assert parse_command("vac 15") == ("vac_kpa", 15.0)
    assert parse_command("move 0.4 -0.1 0.8")[0] == "move"
    assert parse_command("q") == ("quit", None)
    assert parse_command("nonsense")[0] == "unknown"

    # raw GLFW key codes from the viewer window
    assert keycode_to_command(ord("D")) == "d"
    assert keycode_to_command(ord("V")) == "v"
    assert keycode_to_command(ord("1")) == "1"
    assert keycode_to_command(61) == "+"      # '='
    assert keycode_to_command(45) == "-"      # '-'
    assert keycode_to_command(333) == "-"     # keypad '-'
    assert keycode_to_command(256) == "q"     # escape
    assert keycode_to_command(294) is None    # F-key etc.


def _run_motion(control, max_ticks: int = 400) -> None:
    for _ in range(max_ticks):
        if not control.moving():
            return
        control.update_motion()
    raise AssertionError("motion did not finish")


def test_viewer_control_flow(env):
    from mujoco_sim.scripts.view_sim import ViewerControl

    control = ViewerControl(env)
    control.setpoint = 15.0
    assert control.apply("reset", None)
    assert control.apply("select", 0)  # white_foam
    _run_motion(control)
    assert control.apply("descend", None)
    _run_motion(control)
    assert control.apply("vac_kpa", 15.0)
    for _ in range(100):
        control.update_motion()
    assert control.apply("lift", None)
    _run_motion(control)
    for _ in range(50):
        control.update_motion()
    assert env.suction.part_present
    assert env.object_clearance_m("white_foam") > 0.02
    assert control.apply("release", None)
    for _ in range(120):
        control.update_motion()
    assert not env.suction.part_present
    assert env.object_clearance_m("white_foam") < 0.03
    assert control.apply("quit", None) is False

