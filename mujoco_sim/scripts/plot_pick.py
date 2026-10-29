"""Run one suction pick and plot pressure, force and seal evolution.

The default is the ceramic mug at 15 kPa, matching the README hero image::

    .venv/bin/python -m mujoco_sim.scripts.plot_pick
    .venv/bin/python -m mujoco_sim.scripts.plot_pick \
        --object white_foam --setpoint 12 --out /tmp/foam.png

The trace is sampled after every control tick (100 Hz), so the plot shows the
real engage overshoot, settle level, PP latch, lift and release vent.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..env import SuctionEnv  # noqa: E402

PHASE_ORDER = ("approach", "descend", "engage", "lift", "hold", "release")


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def run_traced(
    env: SuctionEnv,
    name: str,
    kpa: float,
    *,
    approach_m: float = 0.09,
    lift_m: float = 0.05,
    engage_s: float = 1.0,
    hold_s: float = 2.0,
    release_s: float = 1.0,
) -> tuple[list[dict[str, Any]], bool, float]:
    """Run the standard pick sequence and return ``(trace, held, clearance)``."""
    trace: list[dict[str, Any]] = []
    phase = {"name": "approach"}

    def record(e: SuctionEnv) -> None:
        s = e.suction
        mass = float(e.model.body_subtreemass[e.object_body_id(name)])
        trace.append(
            {
                "t_s": float(e.data.time),
                "phase": phase["name"],
                "cmd_kpa": float(s.cmd_kpa) if s.cmd_kpa is not None else 0.0,
                "source_kpa": float(s.vacuum.pump.vacuum_kpa),
                "chamber_kpa": float(s.vacuum_kpa),
                "sensor_kpa": float(s.vacuum.vac_sensor_kpa),
                "pp": bool(s.part_present),
                "coverage": float(getattr(s, "coverage_filtered", s.coverage)),
                "seal_quality": float(s.seal_quality),
                "clearance_m": float(e.object_clearance_m(name)),
                "force_n": float(s.suction_force_n),
                "budget_n": float(getattr(s, "force_budget_n", 0.0)),
                "weight_n": mass * 9.81,
            }
        )

    env.reset()
    env.telemetry_callback = record
    record(env)

    top = env.object_top_center(name)
    phase["name"] = "approach"
    env.move_tip_to(
        top + np.array([0.0, 0.0, approach_m]), duration_s=1.2, check=False
    )
    phase["name"] = "descend"
    env.move_tip_to(
        top + np.array([0.0, 0.0, -0.0002]), duration_s=0.9, check=False
    )
    phase["name"] = "engage"
    env.suction.set_cmd(kpa)
    env.hold(engage_s)
    phase["name"] = "lift"
    env.move_tip_to(
        top + np.array([0.0, 0.0, lift_m]), duration_s=1.2, check=False
    )
    phase["name"] = "hold"
    env.hold(hold_s)

    held = bool(env.suction.part_present)
    clearance = float(env.object_clearance_m(name))

    phase["name"] = "release"
    env.suction.release()
    env.hold(release_s)
    env.telemetry_callback = None
    return trace, held, clearance


def _phase_intervals(trace: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    starts: list[tuple[str, float]] = []
    for row in trace:
        if not starts or row["phase"] != starts[-1][0]:
            starts.append((row["phase"], row["t_s"]))
    end = trace[-1]["t_s"]
    out = []
    for i, (name, t0) in enumerate(starts):
        t1 = starts[i + 1][1] if i + 1 < len(starts) else end
        out.append((name, t0, t1))
    return out


def plot_trace(
    trace: list[dict[str, Any]],
    name: str,
    kpa: float,
    out: Path,
    *,
    held: bool,
    clearance: float,
) -> None:
    t = np.array([r["t_s"] for r in trace])
    cmd = np.array([r["cmd_kpa"] for r in trace])
    source = np.array([r["source_kpa"] for r in trace])
    chamber = np.array([r["chamber_kpa"] for r in trace])
    sensor = np.array([r["sensor_kpa"] for r in trace])
    pp = np.array([r["pp"] for r in trace], dtype=float)
    coverage = np.array([r["coverage"] for r in trace]) * 100.0
    quality = np.array([r["seal_quality"] for r in trace]) * 100.0
    clearance_cm = np.array([r["clearance_m"] for r in trace]) * 100.0
    force = np.array([r["force_n"] for r in trace])
    budget = np.array([r["budget_n"] for r in trace])
    weight = np.array([r["weight_n"] for r in trace])
    hold_rows = [i for i, r in enumerate(trace) if r["phase"] == "hold"]
    hold_kpa = sensor[hold_rows[-1]] if hold_rows else sensor[-1]

    fig = plt.figure(figsize=(14.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.20, 1.05, 0.90])
    phase_ax = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :], sharex=phase_ax)
    ax2 = fig.add_subplot(gs[2, 0], sharex=phase_ax)
    ax3 = fig.add_subplot(gs[2, 1], sharex=phase_ax)
    ax4 = fig.add_subplot(gs[2, 2], sharex=phase_ax)

    phases = _phase_intervals(trace)
    cmap = plt.get_cmap("Pastel1")
    for i, (pname, t0, t1) in enumerate(phases):
        color = cmap(i % cmap.N)
        phase_ax.axvspan(t0, t1, color=color, lw=0)
        phase_ax.text(
            (t0 + t1) * 0.5, 0.5, pname, ha="center", va="center", fontsize=8,
            color="0.25",
        )
        if i:
            for ax in (ax1, ax2, ax3, ax4):
                ax.axvline(t0, color="0.55", lw=0.7, ls=":")
    phase_ax.set_ylim(0.0, 1.0)
    phase_ax.set_yticks([])
    phase_ax.tick_params(axis="x", labelbottom=False)
    phase_ax.set_title("Phase", loc="left", fontsize=8, color="0.35")

    ax1.step(t, cmd, where="post", color="0.40", ls="--", lw=1.1,
             label="Command (set_esLvl)")
    ax1.plot(t, source, color="tab:orange", lw=1.3, label="Pump source")
    ax1.plot(t, chamber, color="tab:blue", lw=1.8, label="Chamber (true)")
    ax1.plot(t, sensor, color="tab:cyan", lw=1.0, label="Sensor (filtered)")
    ax1.set_ylabel("vacuum (kPa)")
    ax1.set_title(
        f"{_pretty(name)} pick at {kpa:.0f} kPa — "
        f"peak {sensor.max():.1f} kPa, hold {hold_kpa:.1f} kPa, "
        f"lift {clearance * 100:.1f} cm {'HOLD' if held else 'DROP'}",
        loc="left", fontsize=10,
    )
    ax1.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
    ax1.set_ylim(bottom=0.0)
    ax1.tick_params(axis="x", labelbottom=False)
    peak_i = int(np.argmax(sensor))
    ax1.annotate(
        f"{sensor[peak_i]:.1f} kPa", xy=(t[peak_i], sensor[peak_i]),
        xytext=(10, -16), textcoords="offset points", fontsize=8,
        color="tab:orange",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="tab:orange"),
    )

    ax2.plot(t, clearance_cm, color="tab:green", lw=1.7,
             label="Clearance")
    ax2.axhline(0.0, color="0.55", lw=0.8)
    ax2.set_ylabel("clearance (cm)")
    ax2.set_title("Object clearance / PP", loc="left", fontsize=9)
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2b = ax2.twinx()
    ax2b.step(t, pp, where="post", color="tab:purple", lw=1.0, alpha=0.75)
    ax2b.set_ylim(-0.1, 1.1)
    ax2b.set_yticks([0.0, 1.0])
    ax2b.set_yticklabels(["0", "1"])
    ax2b.set_ylabel("PP", rotation=0, labelpad=10, fontsize=8)

    ax3.plot(t, force, color="tab:red", lw=1.7, label="Applied force")
    ax3.plot(t, budget, color="tab:blue", lw=1.4, ls="--",
             label="Pull-off budget")
    ax3.axhline(weight[0], color="0.35", lw=0.9, ls=":",
                label=f"Weight ({weight[0]:.2f} N)")
    ax3.set_ylabel("force (N)")
    ax3.set_title("Force margin", loc="left", fontsize=9)
    ax3.legend(loc="upper left", fontsize=7.5)

    ax4.plot(t, coverage, color="tab:green", lw=1.7, label="Coverage")
    ax4.plot(t, quality, color="tab:olive", lw=1.2, label="Contact quality")
    ax4.set_ylim(0.0, 106.0)
    ax4.set_ylabel("seal (%)")
    ax4.set_title("Lip seal", loc="left", fontsize=9)
    ax4.legend(loc="upper left", fontsize=7.5)

    for ax in (ax2, ax3, ax4):
        ax.set_xlabel("time (s)")
    ax4.set_xlim(t[0], t[-1])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def write_csv(trace: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", default="ceramic_mug",
        help="object from the registry (default: ceramic_mug)",
    )
    parser.add_argument("--setpoint", type=float, default=15.0,
                        help="vacuum command in kPa (default: 15)")
    parser.add_argument(
        "--out", type=Path, default=Path("docs/images/pick_pressure.png"),
        help="output PNG path",
    )
    parser.add_argument("--csv", type=Path, default=None,
                        help="optional CSV trace path")
    parser.add_argument("--hold-s", type=float, default=2.0,
                        help="hold dwell in seconds (default: 2)")
    args = parser.parse_args(argv)

    with SuctionEnv(objects=[args.object]) as env:
        trace, held, clearance = run_traced(
            env, args.object, args.setpoint, hold_s=args.hold_s
        )
    plot_trace(trace, args.object, args.setpoint, args.out,
               held=held, clearance=clearance)
    if args.csv:
        write_csv(trace, args.csv)
    print(
        f"{_pretty(args.object)}: {'HOLD' if held else 'DROP'}, "
        f"{clearance * 100:.1f} cm clearance, "
        f"peak {max(r['sensor_kpa'] for r in trace):.1f} kPa"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
