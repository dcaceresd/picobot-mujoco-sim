"""Piab piCOBOT Electric model and MuJoCo suction coupling.

Why this looks the way it does
------------------------------
``study/device_config.md`` documents the *measured* pump behaviour on the real
cell.  The simulation deliberately reproduces it so that sweeps and the
online P_min study can be exercised without hardware:

======================  =====================================================
engage peak             ``~40 + 0.7 * cmd`` kPa (irreversible pump overshoot)
steady level            ``cmd + ~1`` kPa (ES-type = Setpoint, ES setpoint 50)
ceiling                 ``~67`` kPa
part-present (``get_pp``)  vacuum comparator at 40 kPa
release                 active vent, vacuum collapses in ~0.15 s
======================  =====================================================

This module implements the **fast legacy rigid-cup mode**.  The default scene
uses the deformable cup + physical vacuum in ``compliant.py`` /
``pneumatics.py``.  In rigid mode the suction force is applied through MuJoCo's
per-geom contact adhesion, which remains useful for quick kinematic tests but
does not model the bellows deformation or the finite-volume pump.

Layers
------
``pneumatics.PumpModel``     measured pump source trajectory (pure numpy).
``pneumatics.LumpedVacuum``  finite-volume pump/leak/vent model.
``SuctionController``        legacy rigid-cup contact adhesion (fast mode).
``compliant.CompliantCupSuction``  deformable cup + physical vacuum (default).
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

log = logging.getLogger(__name__)

from .pneumatics import KPA_TO_PA, PicobotParams, PumpModel

PAD_BODY_NAME = "cup_seal"
PAD_GEOM_NAME = "cup_seal_geom"


class SuctionController:
    """Legacy fast mode: couples the pump model to per-geom contact adhesion.

    The default deformable-cup + physical-vacuum controller lives in
    ``compliant.CompliantCupSuction``.

    Every control update:

    1. scan ``data.contact`` for contacts of the cup pad and classify them
       into *object* vs. *world* (table/floor) contacts,
    2. advance the vacuum state (only an object seal builds vacuum),
    3. write ``F = ΔP · A · seal_quality`` (N) into the pad's
       ``geom_adhesion``, split over the current number of object contacts.

    MuJoCo treats adhesion as a tensile branch of the contact cone: the cup
    can *accelerate* the object (not merely pull with a constant force) and
    the bond breaks once the load exceeds the vacuum force - the same
    pull-off criterion as the real cup.  Table/floor geoms have priority 2 in
    the scene, so the pad's adhesion is ignored for cup-table contacts.
    """

    def __init__(
        self,
        model,
        data,
        *,
        params: PicobotParams | None = None,
        seal_quality_by_body: Mapping[int, float] | None = None,
        pad_body: str = PAD_BODY_NAME,
    ):
        import mujoco  # local import keeps the pure model importable anywhere

        self.model = model
        self.data = data
        self.params = params or PicobotParams()
        self.pump = PumpModel(self.params)

        self._mj = mujoco
        self._pad_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, pad_body)
        # every colliding geom of the cup body: the annular rubber lip geoms
        self._pad_geoms = [
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == self._pad_body
            and (model.geom_contype[gid] or model.geom_conaffinity[gid])
        ]
        if not self._pad_geoms:
            raise ValueError(f"body {pad_body!r} has no colliding geoms")
        self._pad_geom_set = set(self._pad_geoms)
        self._seal_quality = dict(seal_quality_by_body or {})

        if not hasattr(model, "geom_adhesion") or model.geom_adhesion.size == 0:
            raise RuntimeError(
                "this MuJoCo build has no geom adhesion; need mujoco >= 3.2"
            )
        self._adhesion = model.geom_adhesion  # writable numpy view
        self._adhesion[self._pad_geoms] = 0.0

        self.sealed_body: int | None = None
        self.contact_count: int = 0
        self.suction_force_n: float = 0.0
        self.pop_off_count: int = 0
        self._was_holding: bool = False
        self._hold_body: int | None = None

    # -- commands ----------------------------------------------------------
    def set_cmd(self, kpa: float) -> float:
        applied = self.pump.set_cmd(kpa)
        if applied == 0.0:
            self._set_adhesion_n(0.0)
        return applied

    def release(self) -> None:
        self.pump.release()
        self._set_adhesion_n(0.0)

    def reset(self) -> None:
        self.release()
        self.pump.vacuum_kpa = 0.0
        self.pump.peak_kpa = 0.0
        self.sealed_body = None
        self.contact_count = 0
        self._was_holding = False
        self._hold_body = None
        self.pop_off_count = 0

    # -- physics step ------------------------------------------------------
    def update(self) -> None:
        """Call once per physics substep *before* ``mj_step``."""
        self.sealed_body, self.contact_count = self._seal_contacts()
        sealed = self.sealed_body is not None
        quality = self._seal_quality.get(self.sealed_body, 1.0) if sealed else 1.0
        self.pump.step(
            float(self.model.opt.timestep), sealed=sealed, seal_quality=quality
        )

        if self.pump.cmd_kpa is None or not sealed:
            self._set_adhesion_n(0.0)
        else:
            self._set_adhesion_n(self.pump.force_n(quality))

        # holding telemetry / pop-off bookkeeping
        holding = bool(
            sealed and self.pump.cmd_kpa is not None
            and self.pump.part_present
        )
        if self._was_holding and not holding:
            self.pop_off_count += 1
        self._was_holding = holding
        self._hold_body = self.sealed_body

    # -- helpers -----------------------------------------------------------
    @property
    def vacuum_kpa(self) -> float:
        return self.pump.vacuum_kpa

    @property
    def peak_kpa(self) -> float:
        return self.pump.peak_kpa

    @property
    def cmd_kpa(self) -> float | None:
        return self.pump.cmd_kpa

    @property
    def part_present(self) -> bool:
        return self.sealed_body is not None and self.pump.part_present

    @property
    def holding_body(self) -> int | None:
        return self._hold_body

    def is_holding(self, body_name: str) -> bool:
        bid = _id(self.model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        return self.part_present and self._hold_body == bid

    def _seal_contacts(self) -> tuple[int | None, int]:
        """Return (first object body touching the cup, number of contacts)."""
        d = self.data
        pad_body = self._pad_body
        pad_geoms = self._pad_geom_set
        first: int | None = None
        count = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in pad_geoms:
                other = c.geom2
            elif c.geom2 in pad_geoms:
                other = c.geom1
            else:
                continue
            bid = self.model.geom_bodyid[other]
            if bid == pad_body or bid == 0:
                continue  # own body or world (table/floor)
            count += 1
            if first is None:
                first = int(bid)
        return first, count

    def _set_adhesion_n(self, force_n: float) -> None:
        """Distribute the total vacuum force over the current contacts."""
        n = max(1, self.contact_count)
        per_contact = max(force_n, 0.0) / n
        self._adhesion[self._pad_geoms] = per_contact
        self.suction_force_n = max(force_n, 0.0)


def _id(model, obj_type, name: str) -> int:
    import mujoco

    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid < 0:
        raise ValueError(f"MuJoCo model has no {obj_type.name} named {name!r}")
    return oid
