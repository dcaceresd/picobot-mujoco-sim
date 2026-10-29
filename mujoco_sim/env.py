"""SuctionEnv - a small gym-style API around the UR5e + piCOBOT scene.

Typical use::

    from mujoco_sim import SuctionEnv
    env = SuctionEnv(objects=["white_foam"])
    env.move_tip_to(env.object_top_center("white_foam") + [0, 0, 0.08])
    env.move_tip_to(env.object_top_center("white_foam") + [0, 0, 0.0015])
    env.suction.set_cmd(12)
    env.hold(1.0)
    env.move_tip_to(env.object_top_center("white_foam") + [0, 0, 0.06], duration_s=1.0)
    print(env.suction.part_present, env.suction.vacuum_kpa)

All positions are world metres; the table top is ``config.table_height_m``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cameras import CameraSuite, intrinsics_from_file
from .config import SimConfig, object_spec
from .ik import IkError, solve_ik_multiseed
from .model import ARM_HOME_CTRL, ARM_HOME_QPOS, ARM_JOINT_NAMES, build_scene
from .suction import SuctionController
from .compliant import CompliantCupSuction

log = logging.getLogger(__name__)


def _yaw_quat(deg: float):
    half = np.radians(deg) * 0.5
    return (np.cos(half), 0.0, 0.0, np.sin(half))

#: tool z-axis pointing straight down at the table
TOP_DOWN_R = np.diag([1.0, -1.0, -1.0])

ARM_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


@dataclass
class MoveResult:
    ok: bool
    target: np.ndarray
    position: np.ndarray
    pos_err_m: float
    rot_err_rad: float

    def __repr__(self) -> str:
        return (
            f"MoveResult(ok={self.ok}, pos_err={self.pos_err_m * 1e3:.2f} mm, "
            f"rot_err={np.degrees(self.rot_err_rad):.2f} deg)"
        )


class SuctionEnv:
    """Deterministic simulation of the UR5e + piCOBOT pick cell."""

    def __init__(
        self,
        config: SimConfig | None = None,
        objects: list[str] | tuple[str, ...] = ("white_foam",),
        *,
        seed: int = 0,
        scene_path: Path | str | None = None,
    ):
        import mujoco

        self._mj = mujoco
        self.cfg = config or SimConfig()
        self.object_names = tuple(objects)
        for name in self.object_names:
            object_spec(name)  # validate

        self._scene_is_temp = scene_path is None
        if scene_path is not None:
            self.scene_path = Path(scene_path)
        else:
            # unique per process: parallel envs with different object sets must
            # not race on one generated file
            fd, tmp = tempfile.mkstemp(prefix="suction_sim_", suffix=".xml")
            os.close(fd)
            self.scene_path = build_scene(self.cfg, self.object_names, Path(tmp))
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # --- ids -----------------------------------------------------------
        self._tip_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "suction_cup_tip")
        self._tool0_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "tool0")

        self._arm_qadr = np.array([
            int(self.model.jnt_qposadr[
                self._id(mujoco.mjtObj.mjOBJ_JOINT, j)
            ])
            for j in ARM_JOINT_NAMES
        ])
        self._arm_dadr = np.array([
            int(self.model.jnt_dofadr[
                self._id(mujoco.mjtObj.mjOBJ_JOINT, j)
            ])
            for j in ARM_JOINT_NAMES
        ])
        self._arm_act = np.array([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATOR_NAMES
        ])

        self._object_bodies: dict[str, int] = {}
        seal_quality: dict[int, float] = {}
        specs_by_body: dict[int, object] = {}
        skin_objects: dict[int, int] = {}
        for name in self.object_names:
            spec = object_spec(name)
            bid = self._id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")
            self._object_bodies[name] = bid
            seal_quality[bid] = spec.seal_quality
            specs_by_body[bid] = spec

        # soft payload skin pads -> their parent object body
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid
            ) or ""
            for name in self.object_names:
                if bname.startswith(f"obj_{name}_skin_"):
                    skin_objects[bid] = self._object_bodies[name]
                    break

        if self.cfg.cup_model == "compliant":
            self.suction = CompliantCupSuction(
                self.model, self.data,
                pump_params=self.cfg.pump,
                vacuum_params=self.cfg.vacuum,
                cfg=self.cfg,
                object_specs_by_body=specs_by_body,
                skin_objects=skin_objects,
            )
        else:
            self.suction = SuctionController(
                self.model, self.data, params=self.cfg.pump,
                seal_quality_by_body=seal_quality,
            )
        self._cams: CameraSuite | None = None
        #: feed-forward gravity compensation on the arm (UR5e-like tracking)
        self.gravity_compensation: bool = True
        #: optional ``callback(env)`` called after every control tick
        self.telemetry_callback: Callable[["SuctionEnv"], None] | None = None

        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> dict:
        """Reset the arm to home, respawn objects, and vent the cup."""
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qadr] = ARM_HOME_QPOS
        self.data.ctrl[self._arm_act] = ARM_HOME_CTRL
        self._place_objects()
        if self.cfg.randomize_object_yaw_deg > 0:
            self._randomize_object_yaws()
        self.suction.reset()
        self._mj.mj_forward(self.model, self.data)
        return self.obs()

    def _place_objects(self) -> None:
        for name in self.object_names:
            bid = self._object_bodies[name]
            jid = int(self.model.body_jntadr[bid])
            if jid < 0:
                continue
            adr = int(self.model.jnt_qposadr[jid])
            x, y = self.cfg.placement(name)
            yaw = self.cfg.object_yaw_deg.get(name, 0.0)
            quat = _yaw_quat(yaw)
            self.data.qpos[adr:adr + 3] = (x, y, self.cfg.table_height_m)
            self.data.qpos[adr + 3:adr + 7] = quat

    def close(self) -> None:
        if self._cams is not None:
            self._cams.close()
            self._cams = None
        if self._scene_is_temp:
            try:
                self.scene_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self) -> "SuctionEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _randomize_object_yaws(self) -> None:
        limit = float(self.cfg.randomize_object_yaw_deg)
        for name in self.object_names:
            bid = self._object_bodies[name]
            jid = int(self.model.body_jntadr[bid])
            adr = int(self.model.jnt_qposadr[jid])
            base_yaw = float(self.cfg.object_yaw_deg.get(name, 0.0))
            yaw = np.radians(base_yaw + self.rng.uniform(-limit, limit))
            half = 0.5 * yaw
            self.data.qpos[adr + 3: adr + 7] = (
                np.cos(half), 0.0, 0.0, np.sin(half)
            )

    # -- control -------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self._arm_qadr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self._arm_dadr].copy()

    def set_ctrl(self, q_target) -> None:
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (len(self._arm_act),):
            raise ValueError(f"expected {len(self._arm_act)} joint targets")
        self.data.ctrl[self._arm_act] = q_target

    def hold_ctrl(self) -> np.ndarray:
        return self.data.ctrl[self._arm_act].copy()

    def step(self, n_steps: int = 1) -> None:
        """Advance ``n_steps`` control ticks (physics runs at ``timestep_s``).

        If :attr:`telemetry_callback` is set, it is called with this env after
        every control tick (useful for logging pressure/force traces).
        """
        substeps = self.cfg.substeps_per_control
        for _ in range(max(1, int(n_steps))):
            for _ in range(substeps):
                self.data.qfrc_applied[:] = 0.0
                if self.gravity_compensation:
                    # feed-forward gravity/Coriolis compensation on the arm -
                    # the real UR5e controller does this; without it the
                    # position servos sag ~10 mm under the tool payload.
                    self.data.qfrc_applied[self._arm_dadr] = (
                        self.data.qfrc_bias[self._arm_dadr]
                    )
                self.suction.update()
                self._mj.mj_step(self.model, self.data)
            if self.telemetry_callback is not None:
                self.telemetry_callback(self)

    def hold(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.cfg.control_dt_s))))

    # -- kinematics ------------------------------------------------------------
    def cup_tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._tip_site].copy(),
            self.data.site_xmat[self._tip_site].reshape(3, 3).copy(),
        )

    def tool0_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._tool0_site].copy(),
            self.data.site_xmat[self._tool0_site].reshape(3, 3).copy(),
        )

    def ik(
        self,
        target_pos,
        target_R=TOP_DOWN_R,
        *,
        seed_value: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """IK for the cup tip; returns ``(q, pos_err, rot_err)``.

        Non-destructive: the solver works on ``data.qpos`` internally, but the
        actual state is restored before returning so callers can stream the
        solution without teleporting the robot.
        """
        seed = seed_value if seed_value is not None else int(self.data.time * 1e6)
        q_before = self.data.qpos[self._arm_qadr].copy()
        q, pe, re = solve_ik_multiseed(
            self.model, self.data, "suction_cup_tip",
            np.asarray(target_pos, dtype=float), target_R,
            seed_value=seed,
        )
        self.data.qpos[self._arm_qadr] = q_before
        self._mj.mj_forward(self.model, self.data)
        return q, pe, re

    def move_tip_to(
        self,
        target_pos,
        target_R=TOP_DOWN_R,
        *,
        duration_s: float = 1.2,
        settle_s: float = 0.5,
        tol_m: float = 0.004,
        tol_rot_rad: float = 0.08,
        check: bool = True,
    ) -> MoveResult:
        """Smooth PTP move of the cup tip (MoveIt-style quintic profile).

        An instantaneous setpoint jump would rip the suction cup off a light
        object (the contact constraint cannot accelerate the payload that
        fast), so the joint targets follow a quintic with zero velocity and
        acceleration at both ends - just like a UR PTP move.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        q_target, pos_err, rot_err = self.ik(target_pos, target_R)
        if check and (pos_err > tol_m or rot_err > tol_rot_rad):
            raise IkError(
                f"IK failed for target {target_pos.round(4)}: "
                f"pos_err={pos_err * 1e3:.2f} mm rot_err={np.degrees(rot_err):.2f} deg"
            )
        q0 = self.q
        n = max(2, int(round(duration_s / self.cfg.control_dt_s)))
        for i in range(1, n + 1):
            s = i / n
            w = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            self.set_ctrl(q0 + (q_target - q0) * w)
            self.step(1)
        if settle_s > 0:
            self.set_ctrl(q_target)
            self.step(max(1, int(round(settle_s / self.cfg.control_dt_s))))

        pos, R = self.cup_tip_pose()
        err = float(np.linalg.norm(pos - target_pos))
        rot = 0.0
        if target_R is not None:
            Rerr = target_R @ R.T
            rot = float(np.linalg.norm(
                0.5 * np.array([
                    Rerr[2, 1] - Rerr[1, 2],
                    Rerr[0, 2] - Rerr[2, 0],
                    Rerr[1, 0] - Rerr[0, 1],
                ])
            ))
        result = MoveResult(ok=err <= tol_m, target=target_pos, position=pos,
                            pos_err_m=err, rot_err_rad=rot)
        if check and not result.ok:
            raise IkError(f"move did not converge: {result}")
        return result

    # -- objects ---------------------------------------------------------------
    def object_body_id(self, name: str) -> int:
        try:
            return self._object_bodies[name]
        except KeyError:
            raise KeyError(f"object {name!r} not in this scene") from None

    def object_pose(self, name: str) -> dict:
        bid = self.object_body_id(name)
        pos = self.data.xpos[bid].copy()
        R = self.data.xmat[bid].reshape(3, 3).copy()
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, R.reshape(9))
        return {"position": pos, "rotation": R, "quaternion": quat}

    def object_top_center(self, name: str) -> np.ndarray:
        spec = object_spec(name)
        pose = self.object_pose(name)
        top = spec.top_offset + spec.pick_offset_z
        if spec.soft:
            top += spec.soft_skin_height_m
        return pose["position"] + pose["rotation"] @ np.array(
            [spec.pick_offset_xy[0], spec.pick_offset_xy[1], top]
        )

    def object_bottom_z(self, name: str) -> float:
        return float(self.object_pose(name)["position"][2] + object_spec(name).bottom_offset)

    def object_aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB of all colliders of an object (robust to tilting)."""
        import mujoco

        bid = self.object_body_id(name)
        mj = self._mj
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) != bid:
                continue
            if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0:
                continue  # render-only mesh
            pos = self.data.geom_xpos[gid]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            size = self.model.geom_size[gid]
            gtype = self.model.geom_type[gid]
            if gtype == mj.mjtGeom.mjGEOM_BOX:
                half = np.abs(R) @ size[:3]
            elif gtype in (mj.mjtGeom.mjGEOM_SPHERE,):
                half = np.full(3, size[0])
            elif gtype == mj.mjtGeom.mjGEOM_ELLIPSOID:
                half = np.abs(R) @ size[:3]
            else:  # cylinder / capsule / cone: axis = local z
                axis = np.abs(R[:, 2])
                # radius projects onto the plane perpendicular to the axis
                half = axis * size[1] + size[0] * np.sqrt(
                    np.maximum(0.0, 1.0 - axis**2)
                )
            lo = np.minimum(lo, pos - half)
            hi = np.maximum(hi, pos + half)
        return lo, hi

    def object_clearance_m(self, name: str) -> float:
        """Lowest point of the object above the table top (negative = pressed in)."""
        return float(self.object_aabb(name)[0][2] - self.cfg.table_height_m)

    def object_speed(self, name: str) -> tuple[float, float]:
        """(linear speed m/s, angular speed rad/s) of the object body."""
        bid = self.object_body_id(name)
        jid = int(self.model.body_jntadr[bid])
        adr = int(self.model.jnt_dofadr[jid])
        v = self.data.qvel[adr: adr + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    # -- observation / rendering ------------------------------------------------
    def obs(self) -> dict:
        tip_pos, tip_R = self.cup_tip_pose()
        tool_pos, tool_R = self.tool0_pose()
        sealed_name = None
        if self.suction.sealed_body is not None:
            for name, bid in self._object_bodies.items():
                if bid == self.suction.sealed_body:
                    sealed_name = name
                    break
        return {
            "time": float(self.data.time),
            "arm_q": self.q,
            "arm_qd": self.qd,
            "cup_tip_pos": tip_pos,
            "cup_tip_R": tip_R,
            "tool0_pos": tool_pos,
            "tool0_R": tool_R,
            "vacuum_kpa": self.suction.vacuum_kpa,
            "vacuum_peak_kpa": self.suction.peak_kpa,
            "cmd_kpa": self.suction.cmd_kpa,
            "suction_force_n": self.suction.suction_force_n,
            "sealed_object": sealed_name,
            "part_present": self.suction.part_present,
            "objects": {
                name: self.object_pose(name)["position"]
                for name in self.object_names
            },
        }

    @property
    def cameras(self) -> CameraSuite:
        if self._cams is None:
            intr = (
                intrinsics_from_file(self.cfg.intrinsics_path)
                if self.cfg.intrinsics_path else None
            )
            self._cams = CameraSuite(self.model, self.data, intr)
        return self._cams

    def render_camera(self, name: str = "external") -> dict:
        """RGB + depth in the calibrated camera frame."""
        rgb, depth = self.cameras.render_rgbd(name)
        return {"rgb": rgb, "depth": depth, "K": self.cameras.intrinsics.K}

    def capture(self, out_dir: Path | str, cams: tuple[str, ...] | None = None) -> dict:
        """Save RGB-D frames + metadata; returns the metadata dict."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        suite = self.cameras
        meta = {
            "time_s": float(self.data.time),
            "intrinsics": suite.intrinsics.to_dict(),
            "cup_tip_pos": self.cup_tip_pose()[0].tolist(),
            "vacuum_kpa": self.suction.vacuum_kpa,
            "cmd_kpa": self.suction.cmd_kpa,
            "part_present": self.suction.part_present,
            "objects": {
                name: {
                    "position": self.object_pose(name)["position"].tolist(),
                    "quaternion": self.object_pose(name)["quaternion"].tolist(),
                }
                for name in self.object_names
            },
            "cameras": {},
        }
        for name in cams or suite.names:
            rgb, depth = suite.render_rgbd(name)
            np.savez_compressed(
                out / f"{name}_rgbd.npz", rgb=rgb, depth=depth,
                K=suite.intrinsics.K,
            )
            meta["cameras"][name] = {
                "position": suite.camera_pose(name)[0].tolist(),
                "rgbd_file": f"{name}_rgbd.npz",
            }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    # -- internals ---------------------------------------------------------------
    def _id(self, obj_type, name: str) -> int:
        oid = self._mj.mj_name2id(self.model, obj_type, name)
        if oid < 0:
            raise ValueError(f"model has no {obj_type.name} named {name!r}")
        return oid
