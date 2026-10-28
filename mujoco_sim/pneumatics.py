"""Pneumatic model of the piCOBOT Electric suction system.

Two layers:

``PicobotParams`` / ``PumpModel``
    The measured *source* behaviour of the real pump (engage overshoot,
    regulated steady level, one-way valves).  This is the empirical part
    identified from the corpus in ``study/device_config.md``.

``VacuumParams`` / ``LumpedVacuum``
    A physical chamber model around that source: finite volume, pump flow
    limit (Piab datasheet curve), seal leakage, venting, cup-volume change,
    and pressure sensors.  This lets leaks, porous objects and partial seals
    produce realistic pressure droop and drop-outs.

The gauge pressure convention is vacuum-positive: ``vac_kpa = (Patm - P)/1000``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

KPA_TO_PA = 1000.0
P_ATM = 101325.0


# ---------------------------------------------------------------------------
# measured pump source behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PicobotParams:
    """Pump + cup parameters (defaults from the measured piCOBOT Electric)."""

    min_cmd_kpa: float = 5.0
    max_cmd_kpa: float = 65.0
    pump_ceiling_kpa: float = 67.0

    engage_peak_intercept_kpa: float = 40.0
    engage_peak_slope: float = 0.7
    steady_offset_kpa: float = 1.0

    spin_up_tau_s: float = 0.25
    settle_tau_s: float = 0.6
    open_leak_tau_s: float = 0.20
    release_tau_s: float = 0.15

    pp_threshold_kpa: float = 40.0

    cup_seal_radius_m: float = 0.0170
    seal_area_efficiency: float = 0.8313

    @property
    def cup_area_m2(self) -> float:
        return math.pi * self.cup_seal_radius_m**2

    def peak_for(self, cmd_kpa: float) -> float:
        return min(
            self.engage_peak_intercept_kpa + self.engage_peak_slope * cmd_kpa,
            self.pump_ceiling_kpa,
        )

    def steady_for(self, cmd_kpa: float) -> float:
        return min(cmd_kpa + self.steady_offset_kpa, self.pump_ceiling_kpa)


class PumpModel:
    """Empirical pump trajectory: ``set_esLvl`` / ``set_vac`` analogue."""

    def __init__(self, params: PicobotParams = PicobotParams()):
        self.params = params
        self.cmd_kpa: float | None = None
        self.vacuum_kpa: float = 0.0
        self.peak_kpa: float = 0.0
        self._peak_reached = False
        self._pp_latched = False

    def set_cmd(self, kpa: float) -> float:
        p = self.params
        if kpa < 0:
            raise ValueError(f"vacuum command must be >= 0 kPa, got {kpa}")
        if kpa == 0:
            self.release()
            return 0.0
        applied = float(np.clip(kpa, p.min_cmd_kpa, p.max_cmd_kpa))
        if not math.isclose(applied, kpa):
            log.warning(
                "vacuum command %.2f kPa outside [%.1f, %.1f]; clamped to %.2f",
                kpa, p.min_cmd_kpa, p.max_cmd_kpa, applied,
            )
        self.cmd_kpa = applied
        self._peak_reached = False
        self.peak_kpa = 0.0
        return applied

    def release(self) -> None:
        self.cmd_kpa = None
        self._peak_reached = False
        self._pp_latched = False

    def step(self, dt: float, *, sealed: bool, seal_quality: float = 1.0) -> float:
        p = self.params

        def approach(current, target, tau):
            return current + (target - current) * (
                1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-6))
            )

        if self.cmd_kpa is None:
            self.vacuum_kpa = approach(self.vacuum_kpa, 0.0, p.release_tau_s)
            return self.vacuum_kpa
        if not sealed:
            self._pp_latched = False
            self.vacuum_kpa = approach(self.vacuum_kpa, 0.0, p.open_leak_tau_s)
            return self.vacuum_kpa
        quality = float(np.clip(seal_quality, 0.0, 1.0))
        if quality <= 0.0:
            self._pp_latched = False
            self.vacuum_kpa = approach(self.vacuum_kpa, 0.0, p.open_leak_tau_s)
            return self.vacuum_kpa
        peak = p.peak_for(self.cmd_kpa)
        steady = p.steady_for(self.cmd_kpa)
        if not self._peak_reached:
            self.vacuum_kpa = approach(self.vacuum_kpa, peak, p.spin_up_tau_s)
            if self.vacuum_kpa >= peak - 0.5:
                self._peak_reached = True
        else:
            self.vacuum_kpa = approach(self.vacuum_kpa, steady, p.settle_tau_s)
        self.vacuum_kpa = min(self.vacuum_kpa, p.pump_ceiling_kpa)
        self.peak_kpa = max(self.peak_kpa, self.vacuum_kpa)
        if self.vacuum_kpa >= p.pp_threshold_kpa:
            self._pp_latched = True
        return self.vacuum_kpa

    @property
    def part_present(self) -> bool:
        return self.cmd_kpa is not None and self._pp_latched

    def force_n(self, seal_quality: float = 1.0) -> float:
        quality = float(np.clip(seal_quality, 0.0, 1.0))
        return (
            max(self.vacuum_kpa, 0.0)
            * KPA_TO_PA
            * self.params.cup_area_m2
            * self.params.seal_area_efficiency
            * quality
        )


# ---------------------------------------------------------------------------
# physical vacuum system
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VacuumParams:
    """Chamber, pump-flow and leak parameters (SI units)."""

    atmospheric_pa: float = P_ATM
    #: pump + tube + cup internal volume at rest
    chamber_volume_m3: float = 4.0e-5
    cup_volume_m3: float = 1.0e-5

    #: low-pressure pump conductance (m^3/s/Pa) - chamber follows the source
    pump_gain_m3_s_pa: float = 6.0e-7
    #: Piab datasheet flow curve: (vacuum kPa, normal litres/s at max current)
    pump_flow_curve_nl_s: tuple[tuple[float, float], ...] = (
        (0.0, 0.79), (10.0, 0.60), (20.0, 0.44), (30.0, 0.31),
        (40.0, 0.20), (50.0, 0.13), (60.0, 0.07), (67.0, 0.02),
    )
    #: active vent (release valve), tuned to the measured ~0.15 s release
    vent_conductance_m3_s_pa: float = 2.7e-9
    #: no-seal/open-cup leak: large enough that the pump cannot evacuate
    #: open-cup leak: finite (not instantaneous) so a one-step contact
    #: flicker does not collapse the chamber; a sustained open cup still
    #: out-leaks the pump (tau ~50 ms at 50 kPa)
    open_leak_conductance_m3_s_pa: float = 8.0e-9
    #: perfect-seal leak (edges, fittings)
    #: calibrated to the measured engage overshoot decay (~35 kPa in ~0.7 s)
    seal_leak_base_m3_s_pa: float = 1.0e-10

    #: seal coverage below which the cup is considered open
    min_coverage: float = 0.30
    #: sensor model
    sensor_noise_kpa: float = 0.0
    sensor_filter_tau_s: float = 0.02
    pp_threshold_kpa: float = 40.0
    ps_threshold_kpa: float = 45.0
    pp_hysteresis_kpa: float = 3.0
    #: PP is not cleared until the seal has been absent this long
    pp_seal_grace_s: float = 0.20

    def flow_nl_s(self, vacuum_kpa: float) -> float:
        curve = self.pump_flow_curve_nl_s
        if vacuum_kpa <= curve[0][0]:
            return curve[0][1]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if vacuum_kpa <= x1:
                t = (vacuum_kpa - x0) / max(x1 - x0, 1e-9)
                return y0 + t * (y1 - y0)
        return curve[-1][1]


class LumpedVacuum:
    """Finite-volume pneumatic model driven by the measured pump trajectory."""

    def __init__(
        self,
        params: VacuumParams = VacuumParams(),
        pump_params: PicobotParams = PicobotParams(),
    ):
        self.params = params
        self.pump = PumpModel(pump_params)
        self.p_pa: float = params.atmospheric_pa
        self.vac_sensor_kpa: float = 0.0
        self._pp_latched = False
        self._released = True
        self._unsealed_time = 0.0
        self._rng = np.random.default_rng(0)
        self._v_cup_prev = params.cup_volume_m3

    # -- commands ----------------------------------------------------------
    def set_cmd(self, kpa: float) -> float:
        applied = self.pump.set_cmd(kpa)
        self._released = applied == 0.0
        return applied

    def release(self) -> None:
        self.pump.release()
        self._released = True

    def reset(self) -> None:
        self.release()
        self.p_pa = self.params.atmospheric_pa
        self.pump.vacuum_kpa = 0.0
        self.pump.peak_kpa = 0.0
        self.vac_sensor_kpa = 0.0
        self._pp_latched = False
        self._v_cup_prev = self.params.cup_volume_m3

    # -- state -------------------------------------------------------------
    @property
    def cmd_kpa(self) -> float | None:
        return self.pump.cmd_kpa

    @property
    def vac_kpa(self) -> float:
        """True gauge vacuum (positive), in kPa."""
        return max(0.0, (self.params.atmospheric_pa - self.p_pa) / KPA_TO_PA)

    @property
    def peak_kpa(self) -> float:
        return self.pump.peak_kpa

    @property
    def part_present(self) -> bool:
        return self.cmd_kpa is not None and self._pp_latched

    @property
    def part_secured(self) -> bool:
        return self.vac_sensor_kpa >= self.params.ps_threshold_kpa

    def update(
        self,
        dt: float,
        *,
        sealed: bool,
        coverage: float = 1.0,
        compression_quality: float = 1.0,
        tilt_quality: float = 1.0,
        material_leak_m3_s_pa: float = 0.0,
        cup_volume_m3: float | None = None,
        vent: bool | None = None,
    ) -> float:
        p = self.params
        atm = p.atmospheric_pa
        # source trajectory (measured pump behaviour)
        self.pump.step(dt, sealed=True, seal_quality=1.0)
        p_src = atm - max(self.pump.vacuum_kpa, 0.0) * KPA_TO_PA

        # pump flow (one-way, limited by the datasheet curve)
        q_max = p.flow_nl_s(self.vac_kpa) * 1e-3  # Nl/s -> m3/s (at 1 atm)
        if self.p_pa > p_src:
            q_out = min(q_max, p.pump_gain_m3_s_pa * (self.p_pa - p_src))
        else:
            q_out = 0.0

        # leak + vent
        if sealed:
            cov = float(np.clip(coverage, 0.0, 1.0))
            comp = float(np.clip(compression_quality, 0.0, 1.0))
            tilt = float(np.clip(tilt_quality, 0.0, 1.0))
            seal_quality = max(1e-6, cov * (0.5 + 0.5 * comp) * (0.5 + 0.5 * tilt))
            c_leak = (
                max(material_leak_m3_s_pa, 0.0)
                + p.seal_leak_base_m3_s_pa
                * (1.0 / (seal_quality * seal_quality) - 1.0)
            )
        else:
            c_leak = p.open_leak_conductance_m3_s_pa
        q_in = c_leak * max(atm - self.p_pa, 0.0)
        venting = self._released if vent is None else vent
        if venting:
            q_in += p.vent_conductance_m3_s_pa * max(atm - self.p_pa, 0.0)

        # cup volume change (pads/bellows compression)
        if cup_volume_m3 is None:
            v_cup = p.cup_volume_m3
        else:
            v_cup = cup_volume_m3
        dv_dt = (v_cup - self._v_cup_prev) / max(dt, 1e-6)
        self._v_cup_prev = v_cup
        volume = max(p.chamber_volume_m3 + v_cup, 1e-7)

        dp = (atm * q_in - self.p_pa * q_out) / volume - (
            self.p_pa / volume
        ) * dv_dt
        self.p_pa += dp * dt
        # physical bounds
        vmin = atm - self.pump.params.pump_ceiling_kpa * KPA_TO_PA
        self.p_pa = float(np.clip(self.p_pa, vmin, atm))

        # PP comparator on the filtered sensor
        target = self.vac_kpa
        self.vac_sensor_kpa += (target - self.vac_sensor_kpa) * (
            1.0 - math.exp(-max(dt, 0.0) / max(p.sensor_filter_tau_s, 1e-6))
        )
        if p.sensor_noise_kpa > 0:
            self.vac_sensor_kpa += float(
                self._rng.normal(0.0, p.sensor_noise_kpa)
            )
        # PP is a comparator that latches while the part remains sealed
        # (the engage overshoot trips it; it clears on release/seal loss)
        if not sealed:
            self._unsealed_time += dt
        else:
            self._unsealed_time = 0.0
        if self._released or self._unsealed_time > p.pp_seal_grace_s:
            self._pp_latched = False
        elif self.vac_sensor_kpa >= p.pp_threshold_kpa:
            self._pp_latched = True
        return self.vac_kpa
