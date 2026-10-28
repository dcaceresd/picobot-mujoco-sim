"""Generate a synthetic suction dataset (RGB-D + outcomes) in simulation.

For every (object, pressure setpoint, attempt) it runs the same scripted
pipeline as ``pick_test`` and saves:

    <out>/objects/<object>/set_<kpa>/attempt_<i>/
        outcome.json                  # labels: picked, vacuum/force, timeline
        perception/{external,wrist}_rgbd.npz + meta.json
        hold/{external,wrist}_rgbd.npz + meta.json

The layout intentionally mirrors the hardware recorder so downstream scripts
can consume either source.

Example::

    python -m mujoco_sim.scripts.generate_dataset \
        --objects white_foam white_bowl soda_can \
        --levels 5 8 11 15 21 --attempts 2 \
        --out /tmp/opencode/sim_suction_dataset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import SimConfig
from ..env import SuctionEnv
from .pick_test import run_attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="+",
                        default=["white_foam", "white_bowl", "ceramic_mug",
                                 "soda_can", "magazine", "racket"])
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[5, 8, 11, 15, 21])
    parser.add_argument("--attempts", type=int, default=1,
                        help="attempts per (object, level)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-yaw-deg", type=float, default=15.0,
                        help="random object yaw about z per attempt")
    parser.add_argument("--no-capture", action="store_true",
                        help="skip RGB-D rendering (faster, labels only)")
    parser.add_argument("--hold-s", type=float, default=2.0)
    args = parser.parse_args(argv)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig()
    cfg.randomize_object_yaw_deg = float(args.randomize_yaw_deg)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sim": "ur5e_picobot",
        "objects": args.objects,
        "levels_kpa": sorted(args.levels),
        "attempts_per_level": args.attempts,
        "seed": args.seed,
        "hold_s": args.hold_s,
        "randomize_yaw_deg": args.randomize_yaw_deg,
        "capture": not args.no_capture,
        "attempts": [],
    }

    total = len(args.objects) * len(args.levels) * args.attempts
    done = 0
    for object_name in args.objects:
        with SuctionEnv(cfg, objects=[object_name], seed=args.seed) as env:
            for kpa in sorted(args.levels):
                for attempt in range(args.attempts):
                    attempt_dir = (
                        out_root / "objects" / object_name
                        / f"set_{kpa:g}" / f"attempt_{attempt:03d}"
                    )
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    outcome = run_attempt(
                        env, object_name, kpa,
                        hold_s=args.hold_s,
                        capture_dir=None if args.no_capture else attempt_dir,
                    )
                    (attempt_dir / "outcome.json").write_text(
                        json.dumps(outcome, indent=2)
                    )
                    manifest["attempts"].append({
                        "object": object_name,
                        "setpoint_kpa": kpa,
                        "attempt": attempt,
                        "picked": outcome["picked"],
                        "vacuum_peak_kpa": outcome["vacuum_peak_kpa"],
                        "vacuum_hold_kpa": outcome["vacuum_hold_kpa"],
                        "path": str(attempt_dir),
                    })
                    done += 1
                    print(
                        f"[{done}/{total}] {object_name} @ {kpa:g} kPa "
                        f"attempt {attempt}: {'HOLD' if outcome['picked'] else 'drop'}"
                    )

    held = sum(1 for a in manifest["attempts"] if a["picked"])
    manifest["summary"] = {
        "attempts": len(manifest["attempts"]),
        "picked": held,
        "success_rate": held / max(len(manifest["attempts"]), 1),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_root}/manifest.json "
          f"({held}/{len(manifest['attempts'])} picks held)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
