"""Write the generated MJCF scene to disk for inspection / editing.

``SuctionEnv`` regenerates the scene on every construction; this CLI just
builds the current configuration so you can open it in MuJoCo's ``simulate``
or check the XML.

    python -m mujoco_sim.scripts.build_scene --list-objects
    python -m mujoco_sim.scripts.build_scene \
        --objects white_foam white_bowl --out /tmp/scene.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import OBJ_SPECS, SimConfig
from ..model import build_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="+",
                        default=["white_foam", "white_bowl", "soda_can"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--list-objects", action="store_true")
    args = parser.parse_args(argv)

    if args.list_objects:
        print(f"{'object':14s} {'mass':>8s} {'seal':>6s} {'height':>8s}")
        for name, spec in OBJ_SPECS.items():
            print(f"{name:14s} {spec.mass_kg:6.2f}kg {spec.seal_quality:6.2f} "
                  f"{spec.height * 1e3:6.0f}mm")
        return 0

    for name in args.objects:
        if name not in OBJ_SPECS:
            parser.error(f"unknown object {name!r}; see --list-objects")
    path = build_scene(SimConfig(), args.objects, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
