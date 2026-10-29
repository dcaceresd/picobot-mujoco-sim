"""Deformable-cup suction controller.

Physics decomposition
---------------------
* The cup is a **compliant mechanism**: a B30 body on a spring-damped axial
  slide + two tilt hinges, with 12 individually sprung rubber lip pads.  The
  pads conform to the object and transmit the vacuum reaction; the bellows
  visibly compresses/tilts and can peel.
* The **pneumatics** are a finite-volume model (``pneumatics.LumpedVacuum``):
  measured pump source + datasheet flow limit + seal leakage + vent + cup
  volume change.
* The **payload attachment** is a kinematic seal while the vacuum force can
  support it: the part follows the captured pose relative to the cup, and is
  released as soon as ``m*g`` exceeds the available ``dP*A`` pull-off force.
  This avoids force-feedback instabilities while keeping the low-pressure
  P_min behaviour physical (the pull-off criterion itself is physical).
* Seal state comes from the pad contacts: coverage, compression, and
  alignment of the contact patch normal with the cup axis.

This is a reduced-order deformable/contact model (not continuum FEM),
calibrated to the real B30 mechanism and pneumatic behaviour.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .pneumatics import KPA_TO_PA, LumpedVacuum, PicobotParams, VacuumParams
from .suction import PAD_BODY_NAME, PAD_GEOM_NAME

log = logging.getLogger(__name__)


@dataclass
class PadContact:
    contact_index: int
    pad_geom: int
    pad_body: int
    position: np.ndarray
    normal: np.ndarray
    dist: float
    compression: float
    seated: bool


class CompliantCupSuction:
    """Glue between the compliant cup, the pneumatic model and MuJoCo."""

    def __init__(
        self,
        model,
        data,
        *,
        pump_params: PicobotParams,
        vacuum_params: VacuumParams,
        cfg,
        object_specs_by_body: Mapping[int, object],
        skin_objects: Mapping[int, int] | None = None,
        cup_body: str = PAD_BODY_NAME,
        pad_geom_prefix: str = PAD_GEOM_NAME,
    ):
        import mujoco

        self._mj = mujoco
        self.model = model
        self.data = data
        self.cfg = cfg
        self.pump_params = pump_params
        self.vacuum = LumpedVacuum(vacuum_params, pump_params)
        self._object_specs = dict(object_specs_by_body)
        self._skin_objects = dict(skin_objects or {})

        self._cup_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, cup_body)
        # reaction is carried by the arm, not the compliant cup (applying it
        # to the cup would press the lip into the payload and cancel the lift)
        self._tool_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "picobot")
        self._tip_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "suction_cup_tip")
        self._pad_prefix = pad_geom_prefix
        self._pad_geoms: list[int] = []
        self._pad_bodies: list[int] = []
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(pad_geom_prefix):
                self._pad_geoms.append(gid)
                self._pad_bodies.append(int(model.geom_bodyid[gid]))
        if not self._pad_geoms:
            raise ValueError(f"no lip pads named {pad_geom_prefix}_*")
        self._pad_geom_set = set(self._pad_geoms)
        self._n_pads = len(self._pad_geoms)

        # each pad has one axial slide joint
        self._pad_qadr = []
        for bid in self._pad_bodies:
            jid = int(model.body_jntadr[bid])
            if jid < 0:
                raise ValueError("lip pad has no slide joint")
            self._pad_qadr.append(int(model.jnt_qposadr[jid]))

        self._pad_area = math.pi * cfg.cup_seal_radius_m**2
        self._geom_friction_default = np.array([1.0, 0.01, 0.001])
        self._contact_gap = float(model.geom_gap[self._pad_geoms[0]])

        self.sealed_body: int | None = None
        self.contact_count: int = 0
        self.coverage: float = 0.0
        self.seal_quality: float = 0.0
        self.suction_force_n: float = 0.0
        self.pop_off_count: int = 0
        self._holding = False
        self._pad_contacts: list[PadContact] = []
        self._coverage_f = 0.0
        self._coverage_tau_s = 0.03
        self._last_contact_time = -1e9
        self._grasp: dict | None = None
        self._grasp_cooldown = 0.0
        self._applied_force = np.zeros(3)
        self._force_req = 0.0
        self._force_budget = 0.0

    # -- commands ----------------------------------------------------------
    def set_cmd(self, kpa: float) -> float:
        return self.vacuum.set_cmd(kpa)

    def release(self) -> None:
        self.vacuum.release()
        self.suction_force_n = 0.0

    def reset(self) -> None:
        self.vacuum.reset()
        self.sealed_body = None
        self.contact_count = 0
        self.coverage = 0.0
        self.seal_quality = 0.0
        self.suction_force_n = 0.0
        self._holding = False
        self.pop_off_count = 0
        self._pad_contacts = []
        self._coverage_f = 0.0
        self._last_contact_time = -1e9
        self._grasp = None
        self._grasp_cooldown = 0.0
        self._applied_force = np.zeros(3)
        self._force_req = 0.0
        self._force_budget = 0.0

    # -- properties --------------------------------------------------------
    @property
    def vacuum_kpa(self) -> float:
        return self.vacuum.vac_kpa

    @property
    def peak_kpa(self) -> float:
        return self.vacuum.peak_kpa

    @property
    def cmd_kpa(self) -> float | None:
        return self.vacuum.cmd_kpa

    @property
    def part_present(self) -> bool:
        return self._grasp is not None and self.vacuum.part_present

    @property
    def force_budget_n(self) -> float:
        """Vacuum pull-off force available for the current seal."""
        return self._force_budget

    @property
    def coverage_filtered(self) -> float:
        """Low-pass filtered lip coverage (0..1) used by the seal logic."""
        return self._coverage_f

    @property
    def holding_body(self) -> int | None:
        return self.sealed_body if self.part_present else None

    def is_holding(self, body_name: str) -> bool:
        bid = _id(self.model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        return self.part_present and self.sealed_body == bid

    # -- physics update ----------------------------------------------------
    def update(self, dt: float | None = None) -> None:
        mj = self._mj
        d = self.data
        m = self.model
        dt = float(m.opt.timestep if dt is None else dt)

        tip = d.site_xpos[self._tip_site].copy()
        pad_contacts, scan_target = self._scan_contacts(tip)
        now = float(d.time)

        # ---- seal metrics ---------------------------------------------------
        best_by_pad: dict[int, PadContact] = {}
        for c in pad_contacts:
            if not c.seated:
                continue
            prev = best_by_pad.get(c.pad_geom)
            if prev is None or c.compression > prev.compression:
                best_by_pad[c.pad_geom] = c
        seated = list(best_by_pad.values())
        count = len(seated)
        coverage = count / self._n_pads
        if seated:
            mean_comp = float(np.mean([c.compression for c in seated]))
            mean_normal = np.mean([c.normal for c in seated], axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm > 1e-9:
                mean_normal /= norm
            compq = float(np.clip(mean_comp / (self.cfg.lip_pad_radius_m * 0.8), 0.0, 1.0))
        else:
            mean_normal = np.zeros(3)
            compq = 0.0
        cup_axis = d.xmat[self._cup_body].reshape(3, 3)[:, 2].copy()
        suck_dir = -cup_axis
        align = float(np.clip(np.dot(mean_normal, suck_dir), 0.0, 1.0)) if count else 0.0

        a_cov = 1.0 - math.exp(-max(dt, 0.0) / self._coverage_tau_s)
        self._coverage_f += (coverage - self._coverage_f) * a_cov
        if scan_target is not None:
            self._last_contact_time = now

        grasp = self._grasp
        target = grasp["body"] if grasp is not None else scan_target
        # a real bellows keeps its seal for a short time if the lip contact
        # flickers; the virtual grasp keeps the payload attached meanwhile
        sealed = (
            target is not None
            and (
                self._coverage_f >= self.cfg.vacuum.min_coverage * 0.5
                or (grasp is not None and now - self._last_contact_time < 0.20)
            )
        )
        spec = self._object_specs.get(target, None)
        material_leak = float(getattr(spec, "leak_m3_s_pa", 2e-11))

        self.vacuum.update(
            dt,
            sealed=sealed,
            coverage=max(self._coverage_f, 0.2 if grasp else 0.0),
            compression_quality=compq,
            tilt_quality=align,
            material_leak_m3_s_pa=material_leak,
            cup_volume_m3=None,  # pad-volume feedback is numerically stiff
        )

        # ---- vacuum force budget -------------------------------------------
        force_budget = 0.0
        if sealed and self.vacuum.cmd_kpa is not None and spec is not None:
            cov_for_area = grasp["coverage"] if grasp is not None else self._coverage_f
            area = (
                self._pad_area
                * max(cov_for_area, 0.05)
                * self.pump_params.seal_area_efficiency
                * float(spec.seal_quality)
            )
            force_budget = (
                max(self.vacuum.vac_kpa, 0.0)
                * KPA_TO_PA
                * area
                * self.cfg.grasp_force_safety
            )

        # ---- capture / hold / break ----------------------------------------
        if grasp is None:
            if (
                sealed
                and scan_target is not None
                and self.vacuum.cmd_kpa is not None
                and force_budget > 0.05
                and now >= self._grasp_cooldown
            ):
                self._capture_grasp(scan_target, coverage)
                grasp = self._grasp
        if grasp is not None:
            keep = self._update_grasp(grasp, force_budget, dt)
            if not keep:
                self._drop_grasp(now)
                grasp = None

        # ---- telemetry -----------------------------------------------------
        self._pad_contacts = pad_contacts
        self.sealed_body = grasp["body"] if grasp is not None else (
            scan_target if sealed else None
        )
        self.contact_count = count
        self.coverage = coverage
        self.seal_quality = (
            coverage * (0.4 + 0.6 * compq) * (0.4 + 0.6 * align)
            if target is not None else 0.0
        )
        self._force_budget = force_budget
        if grasp is not None:
            self.suction_force_n = float(np.linalg.norm(self._applied_force))
        else:
            self.suction_force_n = 0.0

        holding = bool(grasp is not None and self.vacuum.part_present)
        if self._holding and not holding:
            self.pop_off_count += 1
        self._holding = holding

    # -- virtual suction grasp ---------------------------------------------
    def _capture_grasp(self, body: int, coverage: float) -> None:
        d = self.data
        p_cup = d.xpos[self._cup_body].copy()
        R_cup = d.xmat[self._cup_body].reshape(3, 3).copy()
        p_obj = d.xpos[body].copy()
        R_obj = d.xmat[body].reshape(3, 3).copy()
        self._grasp = {
            "body": int(body),
            "rel_p": R_cup.T @ (p_obj - p_cup),
            "rel_R": R_cup.T @ R_obj,
            "coverage": float(np.clip(coverage, 0.05, 1.0)),
            "time": float(d.time),
        }

    def _drop_grasp(self, now: float) -> None:
        if self._grasp is not None:
            self._grasp = None
            self._grasp_cooldown = now + 0.4
            self._applied_force = np.zeros(3)
            self._force_req = 0.0

    def _update_grasp(
        self, grasp: dict, force_budget: float, dt: float
    ) -> bool:
        """Attach the payload kinematically while the vacuum can support it.

        This is the standard reduced-order suction coupling: the sealed part
        is held at its captured pose relative to the cup, and the vacuum can
        only support ``m*g`` (plus the configured safety margin).  When the
        required force exceeds ``dP*A`` the seal pulls off and the payload is
        released to the rigid-body solver.  All pneumatic/seal/deformation
        physics is upstream of this force limit.
        """
        mj = self._mj
        d = self.data
        body = grasp["body"]
        p_cup = d.xpos[self._cup_body]
        R_cup = d.xmat[self._cup_body].reshape(3, 3)
        p_des = p_cup + R_cup @ grasp["rel_p"]
        R_des = R_cup @ grasp["rel_R"]

        mass = float(self.model.body_subtreemass[body])
        req_force = mass * 9.81
        self._force_req = req_force
        self._applied_force = np.array([0.0, 0.0, req_force])

        if req_force > force_budget + 1e-6:
            self._break_reason = (
                f"weight {req_force:.2f}N > vacuum {force_budget:.2f}N"
            )
            return False

        # kinematic attachment: set the free-joint pose to the cup-relative
        # pose; zero velocity keeps the contacts/pads quiet
        jid = int(self.model.body_jntadr[body])
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr:adr + 3] = p_des
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, R_des.reshape(9))
        self.data.qpos[adr + 3:adr + 7] = quat
        vadr = int(self.model.jnt_dofadr[jid])
        self.data.qvel[vadr:vadr + 6] = 0.0

        # the arm supports the payload: reaction of the suction force
        zero = np.zeros(3)
        mj.mj_applyFT(
            self.model, d, np.array([0.0, 0.0, -req_force]), zero,
            p_des, self._tool_body, d.qfrc_applied,
        )
        return True

    # -- helpers -----------------------------------------------------------
    # -- helpers -----------------------------------------------------------
    def _scan_contacts(self, tip: np.ndarray):
        d = self.data
        m = self.model
        pad_set = self._pad_geom_set
        by_target: dict[int, list[PadContact]] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 in pad_set:
                pad_geom, other = g1, g2
            elif g2 in pad_set:
                pad_geom, other = g2, g1
            else:
                continue
            if other < 0:
                continue
            bid = int(m.geom_bodyid[other])
            bid = self._skin_objects.get(bid, bid)
            if bid <= 0 or bid == self._cup_body or bid in self._pad_bodies:
                continue
            if bid not in self._object_specs:
                continue
            n = np.asarray(c.frame[:3], dtype=float).copy()
            pos = np.asarray(c.pos, dtype=float).copy()
            if np.dot(n, tip - pos) < 0.0:
                n = -n
            dist = float(c.dist)
            pc = PadContact(
                contact_index=i,
                pad_geom=pad_geom,
                pad_body=int(m.geom_bodyid[pad_geom]),
                position=pos,
                normal=n,
                dist=dist,
                compression=max(0.0, -dist),
                seated=dist <= self._contact_gap + 1e-9,
            )
            by_target.setdefault(bid, []).append(pc)
        if not by_target:
            return [], None
        target = max(by_target, key=lambda b: len(by_target[b]))
        return by_target[target], target

    def _apply_force(
        self,
        target_body: int,
        normal: np.ndarray,
        force_n: float,
    ) -> None:
        mj = self._mj
        d = self.data
        if force_n <= 0.0 or len(self._pad_contacts) == 0:
            return
        seated = [c for c in self._pad_contacts if c.seated]
        if not seated:
            return
        centroid = np.mean([c.position for c in seated], axis=0)
        force = normal * force_n
        zero = np.zeros(3)
        mj.mj_applyFT(
            self.model, d, force, zero, centroid, target_body,
            d.qfrc_applied,
        )
        # reaction on the tool flange / arm
        mj.mj_applyFT(
            self.model, d, -force, zero, centroid, self._tool_body,
            d.qfrc_applied,
        )


def _id(model, obj_type, name: str) -> int:
    import mujoco

    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid < 0:
        raise ValueError(f"MuJoCo model has no {obj_type.name} named {name!r}")
    return oid
