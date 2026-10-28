"""Pressure sweep in simulation (small analogue of the adaptive P_min study).

Runs one pick attempt per pressure level and reports the lowest command that
still holds.  Useful to sanity-check the pump model and per-object seal
calibration without touching hardware.

Example::

    python -m mujoco_sim.scripts.sweep_test \
        --object white_bowl --levels 5 8 11 15 21
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import SimConfig
from ..env import SuctionEnv
from .pick_test import run_attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="white_bowl")
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[5, 8, 11, 15, 21, 27, 33, 40])
    parser.add_argument("--repeats", type=int, default=1,
                        help="attempts per level (majority vote)")
    parser.add_argument("--lift-mm", type=float, default=50.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = []
    with SuctionEnv(SimConfig(), objects=[args.object]) as env:
        for kpa in sorted(args.levels):
            outcomes = [
                run_attempt(env, args.object, kpa,
                            lift_m=args.lift_mm / 1000.0, hold_s=args.hold_s)
                for _ in range(max(1, args.repeats))
            ]
            successes = sum(1 for o in outcomes if o["picked"])
            held = successes * 2 > len(outcomes)
            rows.append({
                "setpoint_kpa": kpa,
                "successes": successes,
                "attempts": len(outcomes),
                "held": held,
                "vacuum_peak_kpa": outcomes[-1]["vacuum_peak_kpa"],
                "vacuum_hold_kpa": outcomes[-1]["vacuum_hold_kpa"],
                "hold_force_n": outcomes[-1]["hold_force_n"],
            })
            print(f"{kpa:5.1f} kPa: {'HOLD' if held else 'drop'}"
                  f"  ({successes}/{len(outcomes)})")

    held_levels = [r["setpoint_kpa"] for r in rows if r["held"]]
    summary = {
        "object": args.object,
        "levels": rows,
        "p_min_hold_kpa": min(held_levels) if held_levels else None,
        "holds_at_all_levels": len(held_levels) == len(rows),
        "never_holds": not held_levels,
    }
    if summary["p_min_hold_kpa"] is not None:
        print(f"\nlowest holding command: {summary['p_min_hold_kpa']:.1f} kPa")
    else:
        print("\nobject never held in this range")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    return 0 if summary["p_min_hold_kpa"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
