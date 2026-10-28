"""Interactive MuJoCo viewer for the UR5e + piCOBOT cell.

Controls work in the viewer window (click it first) **and** in the terminal
where you launched the script - whichever has focus:

    1..9            select object N and move the cup 10 cm above it
    d               descend onto the selected object (seal position)
    v               toggle vacuum at the current setpoint
    set <kPa>       change setpoint (5..65); re-applies if vacuum is on
    vac <kPa>       vacuum on at <kPa>
    + / -           setpoint +/- 1 kPa
    l               lift 6 cm
    r               release vacuum
    h / reset       reset the scene
    m / move x y z  move the cup tip to world coordinates (metres)
    p / status      print state
    q / quit        exit
    help            show commands

Examples in the terminal::

    1                # select first object, move above it
    d                # descend
    v                # vacuum on at the setpoint
    +                # raise setpoint by 1 kPa
    l                # lift
    r                # release
    move 0.42 -0.10 0.82
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from ..config import OBJ_SPECS, SimConfig
from ..env import TOP_DOWN_R, SuctionEnv

HELP = (
    "1-9 object | d descend | v vacuum | +/- setpoint | l lift | "
    "r release | h reset | move x y z | p status | q quit  (terminal works too)"
)

KEY_ESCAPE = 256
KEY_KP_ADD = 334
KEY_KP_SUBTRACT = 333
KEY_EQUAL = 61
KEY_MINUS = 45
QUINTIC = (10.0, -15.0, 6.0)  # smoothstep weights: 10s^3 - 15s^4 + 6s^5


def keycode_to_command(keycode: int) -> str | None:
    """Translate a raw GLFW key code into a terminal-style command.

    MuJoCo passes GLFW codes to ``key_callback`` - letters arrive as
    *uppercase* ASCII ('D' == 68), which was why lowercase comparisons never
    fired.  Normalise here so window keys and typed commands share one path.
    """
    if keycode == KEY_ESCAPE:
        return "q"
    if keycode in (KEY_EQUAL, KEY_KP_ADD):
        return "+"
    if keycode in (KEY_MINUS, KEY_KP_SUBTRACT):
        return "-"
    if 32 <= keycode < 127:
        return chr(keycode).lower()
    return None


# ---------------------------------------------------------------------------
# command parsing (pure, testable without a GUI)
# ---------------------------------------------------------------------------


def parse_command(text: str) -> tuple[str, Any]:
    """Turn a key/terminal command into ``(action, argument)``."""
    text = text.strip()
    low = text.lower()
    if not text:
        return "noop", None
    if low in ("q", "quit", "exit"):
        return "quit", None
    if low in ("help", "?"):
        return "help", None
    if low in ("p", "status"):
        return "status", None
    if low in ("h", "reset", "home"):
        return "reset", None
    if low in ("r", "release"):
        return "release", None
    if low in ("v", "vacuum", "vac"):
        return "vacuum_toggle", None
    if low in ("l", "lift"):
        return "lift", None
    if low in ("d", "descend"):
        return "descend", None
    if low in ("+", "=", "setpoint +"):
        return "setpoint", +1.0
    if low in ("-", "_", "setpoint -"):
        return "setpoint", -1.0

    parts = low.split()
    if len(parts) == 1 and len(parts[0]) == 1 and parts[0] in "123456789":
        return "select", int(parts[0]) - 1
    if parts[0] in ("set", "setpoint") and len(parts) == 2:
        try:
            return "set_kpa", float(parts[1])
        except ValueError:
            return "unknown", text
    if parts[0] in ("vac", "vacuum") and len(parts) == 2:
        try:
            return "vac_kpa", float(parts[1])
        except ValueError:
            return "unknown", text
    if parts[0] in ("m", "move") and len(parts) == 4:
        try:
            return "move", tuple(float(v) for v in parts[1:])
        except ValueError:
            return "unknown", text
    if parts[0] in ("obj", "object") and len(parts) == 2:
        try:
            idx = int(parts[1]) - 1
        except ValueError:
            return "unknown", text
        return "select", idx
    return "unknown", text


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------


class ViewerControl:
    """Non-blocking motion + suction actions shared by keyboard and terminal."""

    def __init__(self, env: SuctionEnv):
        self.env = env
        self.setpoint = 12.0
        self.selected = 0
        self._motion: dict[str, Any] | None = None

    # -- motions -----------------------------------------------------------
    def start_move(
        self, target_pos, *, duration_s: float = 1.1, label: str = ""
    ) -> bool:
        target = np.asarray(target_pos, dtype=float)
        q1, pos_err, rot_err = self.env.ik(target, TOP_DOWN_R)
        if pos_err > 5e-3 or rot_err > 0.1:
            print(f"  unreachable target {target.round(3)} "
                  f"(err {pos_err * 1e3:.1f} mm / {np.degrees(rot_err):.1f} deg)")
            return False
        self._motion = {
            "q0": self.env.q.copy(),
            "q1": q1,
            "t": 0.0,
            "duration": max(duration_s, self.env.cfg.control_dt_s * 2),
            "label": label,
        }
        return True

    def moving(self) -> bool:
        return self._motion is not None

    def update_motion(self) -> None:
        """Advance one control tick of the active quintic move (if any)."""
        m = self._motion
        if m is None:
            self.env.step(1)
            return
        m["t"] += self.env.cfg.control_dt_s
        s = min(1.0, m["t"] / m["duration"])
        w = QUINTIC[0] * s**3 + QUINTIC[1] * s**4 + QUINTIC[2] * s**5
        self.env.set_ctrl(m["q0"] + (m["q1"] - m["q0"]) * w)
        self.env.step(1)
        if s >= 1.0:
            label = m["label"] or "move"
            print(f"  {label} done")
            self._motion = None

    # -- object helpers ------------------------------------------------------
    def object_name(self, index: int | None = None) -> str:
        idx = self.selected if index is None else index
        idx = int(np.clip(idx, 0, len(self.env.object_names) - 1))
        self.selected = idx
        return self.env.object_names[idx]

    def above_object(self, name: str) -> np.ndarray:
        return self.env.object_top_center(name) + np.array([0.0, 0.0, 0.10])

    def seal_position(self, name: str) -> np.ndarray:
        return self.env.object_top_center(name) + np.array([0.0, 0.0, -0.0002])

    # -- actions -------------------------------------------------------------
    def apply(self, action: str, arg: Any) -> bool:
        """Apply one command; returns ``False`` when the viewer should exit."""
        env = self.env
        if action == "quit":
            return False
        if action == "noop":
            return True
        if action == "help":
            print(HELP)
        elif action == "status":
            print(self.status_line())
        elif action == "reset":
            env.reset()
            env.suction.release()
            self._motion = None
            print("  scene reset")
        elif action == "release":
            env.suction.release()
            print("  vacuum OFF")
        elif action == "vacuum_toggle":
            if env.suction.cmd_kpa is None:
                env.suction.set_cmd(self.setpoint)
                print(f"  vacuum ON @ {self.setpoint:.0f} kPa")
            else:
                env.suction.release()
                print("  vacuum OFF")
        elif action == "setpoint":
            self.setpoint = float(np.clip(self.setpoint + arg, 5.0, 65.0))
            if env.suction.cmd_kpa is not None:
                env.suction.set_cmd(self.setpoint)
            print(f"  setpoint {self.setpoint:.0f} kPa")
        elif action == "set_kpa":
            self.setpoint = float(np.clip(arg, 5.0, 65.0))
            if env.suction.cmd_kpa is not None:
                env.suction.set_cmd(self.setpoint)
            print(f"  setpoint {self.setpoint:.0f} kPa")
        elif action == "vac_kpa":
            self.setpoint = float(np.clip(arg, 5.0, 65.0))
            env.suction.set_cmd(self.setpoint)
            print(f"  vacuum ON @ {self.setpoint:.0f} kPa")
        elif action == "select":
            name = self.object_name(arg)
            env.suction.release()
            self.start_move(self.above_object(name), label=f"{name}: approach")
            print(f"  selected {name}")
        elif action == "descend":
            name = self.object_name()
            env.suction.release()
            self.start_move(self.seal_position(name), label=f"{name}: descend")
            print(f"  descending onto {name}")
        elif action == "lift":
            if not self.moving():
                target = env.cup_tip_pose()[0] + np.array([0.0, 0.0, 0.06])
                self.start_move(target, label="lift")
                print("  lifting 6 cm")
        elif action == "move":
            print(f"  moving to {np.round(arg, 3)}")
            self.start_move(np.asarray(arg, dtype=float), label="move")
        else:
            print(f"  unknown command: {arg!r} (type 'help')")
        return True

    def status_line(self) -> str:
        name = self.object_name()
        env = self.env
        return (
            f"{name} | setpoint {self.setpoint:.0f} kPa | "
            f"vac {env.suction.vacuum_kpa:5.1f} kPa | "
            f"pp {env.suction.part_present} | "
            f"tip z {env.cup_tip_pose()[0][2]:.3f} m | "
            f"{'MOVING ' if self.moving() else ''}"
            f"objects: {', '.join(env.object_names)}"
        )


def _stdin_reader(commands: "queue.Queue[str]") -> None:
    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                commands.put(line)
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects", nargs="+",
        default=["white_foam", "red_foam", "white_bowl", "ceramic_mug",
                 "soda_can", "magazine", "test_cube"],
        help="objects to spawn (max 9 selectable with number keys)",
    )
    parser.add_argument("--setpoint", type=float, default=12.0,
                        help="initial vacuum setpoint in kPa")
    parser.add_argument("--debug-keys", action="store_true",
                        help="print raw GLFW key codes (keyboard troubleshooting)")
    args = parser.parse_args(argv)

    for name in args.objects:
        if name not in OBJ_SPECS:
            parser.error(f"unknown object {name!r}; available: {sorted(OBJ_SPECS)}")

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display found: view_sim needs a graphical session.")
        print("Use pick_test/sweep_test/generate_dataset for headless runs; "
              "for offscreen rendering export MUJOCO_GL=egl.")
        return 1

    env = SuctionEnv(SimConfig(), objects=args.objects)
    control = ViewerControl(env)
    control.setpoint = float(np.clip(args.setpoint, 5.0, 65.0))
    commands: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()

    import mujoco
    import mujoco.viewer

    def key_callback(keycode: int) -> None:
        if args.debug_keys:
            print(f"  [key] {keycode} ({chr(keycode) if 32 <= keycode < 127 else '?'})")
        command = keycode_to_command(keycode)
        if command is not None:
            commands.put(command)

    before_threads = set(threading.enumerate())
    try:
        viewer = mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=key_callback
        )
    except Exception as exc:  # pragma: no cover - depends on the machine
        print(f"could not open the viewer ({exc}).")
        print("For headless rendering export MUJOCO_GL=egl.")
        env.close()
        return 1
    viewer_threads = [t for t in threading.enumerate() if t not in before_threads]

    viewer.cam.lookat[:] = (0.40, 0.0, 0.82)
    viewer.cam.distance = 1.35
    viewer.cam.elevation = -28.0
    viewer.cam.azimuth = 135.0
    viewer.set_texts([(None, None, "piCOBOT sim - keyboard AND terminal work",
                       HELP)])

    print("Interactive UR5e + piCOBOT simulation")
    print(HELP)
    print("The terminal accepts the same commands; type 'help' or 'quit'.")
    print(f"spawned: {', '.join(env.object_names)}")

    last_status = 0.0
    next_tick = time.monotonic()
    try:
        while viewer.is_running():
            try:
                while True:
                    raw = commands.get_nowait()
                    action, arg = parse_command(raw)
                    if not control.apply(action, arg):
                        return 0
            except queue.Empty:
                pass

            control.update_motion()
            viewer.sync()

            now = time.monotonic()
            if now - last_status > 1.0:
                viewer.set_texts([(None, None,
                                   "piCOBOT sim - keyboard AND terminal work",
                                   control.status_line())])
                last_status = now
            next_tick += env.cfg.control_dt_s
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        # The render thread needs to destroy the GLFW window itself; exiting
        # the process earlier segfaults inside GLX teardown.
        deadline = time.monotonic() + 3.0
        for thread in viewer_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
