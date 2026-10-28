"""Run one scripted suction pick in simulation, mirroring the hardware cycle.

Phases (same order as the real recorder):

    approach -> descend -> set_esLvl -> engage dwell -> lift -> hold ->
    release -> drop confirm

Prints a timeline summary and can write a JSON outcome with ``--json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SimConfig
from ..env import SuctionEnv


def run_attempt(
    env: SuctionEnv,
    name: str,
    kpa: float,
    *,
    approach_m: float = 0.09,
    descend_press_m: float = -0.0002,
    engage_s: float = 1.0,
    lift_m: float = 0.05,
    hold_s: float = 2.0,
    release_s: float = 1.0,
    capture_dir: Path | None = None,
) -> dict[str, Any]:
    """One attempt; returns an outcome dict (JSON-serialisable)."""
    env.reset()
    tip0 = env.cup_tip_pose()[0].copy()
    top = env.object_top_center(name)
    obj_start = env.object_pose(name)["position"].copy()

    timeline: list[dict[str, float | str | bool]] = []

    def sample(phase: str) -> None:
        timeline.append(
            {
                "phase": phase,
                "t_s": float(env.data.time),
                "vacuum_kpa": float(env.suction.vacuum_kpa),
                "part_present": bool(env.suction.part_present),
                "sealed": env.suction.sealed_body is not None,
                "cup_tip_z": float(env.cup_tip_pose()[0][2]),
                "object_z": float(env.object_pose(name)["position"][2]),
            }
        )

    approach = env.move_tip_to(top + np.array([0.0, 0.0, approach_m]),
                               duration_s=1.2, check=False)
    sample("approach")
    captures: dict[str, str] = {}
    if capture_dir is not None:
        cap = Path(capture_dir) / "perception"
        env.capture(cap)
        captures["perception"] = str(cap)
    descend = env.move_tip_to(top + np.array([0.0, 0.0, descend_press_m]),
                              duration_s=0.9, check=False)
    sample("descend")

    env.suction.set_cmd(kpa)
    n_engage = max(1, int(round(engage_s / env.cfg.control_dt_s / 10)))
    for i in range(n_engage):
        env.step(10)
        if i % 3 == 0:
            sample("engage")
    peak = env.suction.peak_kpa
    sealed_at_engage = env.suction.sealed_body is not None

    lift = env.move_tip_to(top + np.array([0.0, 0.0, lift_m]),
                           duration_s=1.2, check=False)
    sample("lift")
    n_hold = max(1, int(round(hold_s / env.cfg.control_dt_s / 10)))
    for i in range(n_hold):
        env.step(10)
        if i % 5 == 0:
            sample("hold")

    z_held = float(env.object_pose(name)["position"][2])
    clearance = env.object_clearance_m(name)
    lin_speed, ang_speed = env.object_speed(name)
    pp_held = env.suction.part_present
    vacuum_hold = env.suction.vacuum_kpa
    force_hold = env.suction.suction_force_n
    force_budget_hold = float(getattr(env.suction, "force_budget_n", 0.0))
    coverage_hold = float(getattr(env.suction, "coverage", 0.0))
    seal_quality_hold = float(getattr(env.suction, "seal_quality", 0.0))
    object_mass = float(env.model.body_subtreemass[env.object_body_id(name)])
    # a real pick: still sealed, object clear of the table, and not tumbling
    picked = bool(
        pp_held
        and clearance > 0.02
        and lin_speed < 0.30
        and ang_speed < 1.5
    )

    capture_dir_out = None
    if capture_dir is not None:
        cap = Path(capture_dir) / "hold"
        env.capture(cap)
        captures["hold"] = str(cap)
        capture_dir_out = str(capture_dir)

    env.suction.release()
    env.hold(release_s)
    sample("released")
    z_after = float(env.object_pose(name)["position"][2])
    clearance_after = env.object_clearance_m(name)
    on_table = clearance_after < 0.03
    stuck = clearance_after > 0.02 and not env.suction.part_present

    outcome = {
        "object": name,
        "setpoint_kpa": float(kpa),
        "picked": picked,
        "held": bool(pp_held),
        "object_stuck_on_cup": stuck,
        "object_on_table_after_release": bool(on_table),
        "sealed_at_engage": sealed_at_engage,
        "vacuum_peak_kpa": float(peak),
        "vacuum_hold_kpa": float(vacuum_hold),
        "hold_force_n": float(force_hold),
        "force_budget_n": force_budget_hold,
        "seal_coverage": coverage_hold,
        "seal_quality": seal_quality_hold,
        "object_mass_kg": object_mass,
        "object_start_z": float(obj_start[2]),
        "object_lift_z": z_held,
        "object_lift_clearance_m": clearance,
        "object_linear_speed": lin_speed,
        "object_angular_speed": ang_speed,
        "object_after_release_z": z_after,
        "object_after_release_clearance_m": clearance_after,
        "cup_tip_start": tip0.tolist(),
        "approach": approach.__dict__ | {"target": approach.target.tolist(),
                                         "position": approach.position.tolist()},
        "descend": descend.__dict__ | {"target": descend.target.tolist(),
                                       "position": descend.position.tolist()},
        "lift": lift.__dict__ | {"target": lift.target.tolist(),
                                 "position": lift.position.tolist()},
        "timeline": timeline,
        "capture_dir": capture_dir_out,
        "captures": captures,
    }
    return outcome


def _print_summary(out: dict[str, Any]) -> None:
    verdict = "PICKED" if out["picked"] else "FAILED"
    print(f"\n{out['object']} @ {out['setpoint_kpa']:.0f} kPa -> {verdict}")
    print(f"  mass            : {out['object_mass_kg'] * 1e3:.1f} g")
    print(f"  vacuum peak     : {out['vacuum_peak_kpa']:.1f} kPa")
    print(f"  vacuum at hold  : {out['vacuum_hold_kpa']:.1f} kPa")
    print(f"  hold force      : {out['hold_force_n']:.2f} N "
          f"(weight {out['object_mass_kg'] * 9.81:.2f} N)")
    print(f"  vacuum budget   : {out['force_budget_n']:.2f} N "
          f"(coverage {out['seal_coverage'] * 100:.0f}%)")
    print(f"  object z        : {out['object_start_z']:.3f} -> "
          f"{out['object_lift_z']:.3f} -> {out['object_after_release_z']:.3f} m")
    print(f"  clearance       : {out['object_lift_clearance_m'] * 100:.1f} cm above table, "
          f"lin {out['object_linear_speed']:.2f} m/s, "
          f"ang {out['object_angular_speed']:.2f} rad/s")
    print(f"  stuck on cup    : {out['object_stuck_on_cup']}")
    phases = {}
    for row in out["timeline"]:
        phases.setdefault(row["phase"], []).append(row)
    print("  phase timeline  : " + " | ".join(
        f"{phase}: vac {rows[-1]['vacuum_kpa']:.1f} kPa"
        for phase, rows in phases.items()
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="white_foam",
                        help="object from the registry (see model.py --list-objects)")
    parser.add_argument("--setpoint", type=float, default=12.0,
                        help="vacuum command in kPa (set_esLvl analogue)")
    parser.add_argument("--lift-mm", type=float, default=50.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the outcome dict to this file")
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="save RGB-D captures here (external + wrist)")
    args = parser.parse_args(argv)

    with SuctionEnv(SimConfig(), objects=[args.object]) as env:
        outcome = run_attempt(
            env, args.object, args.setpoint,
            lift_m=args.lift_mm / 1000.0, hold_s=args.hold_s,
            capture_dir=args.capture_dir,
        )
    _print_summary(outcome)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(outcome, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if outcome["picked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
