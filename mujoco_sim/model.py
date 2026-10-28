"""Build the MuJoCo scene: world + table + UR5e (Menagerie) + piCOBOT tool.

The UR5e kinematics/meshes come from DeepMind's ``mujoco_menagerie``
(``universal_robots_ur5e``, copied into ``assets/ur5e``).  We import its arm
subtree, then bolt on the picobot tool exactly like the real robot:

``tool0 -> suction_cup_tip = (0, 0, 0.370) m`` in the tool frame.

The resulting MJCF is written to ``assets/generated_scene.xml`` and loaded with
``mujoco.MjModel.from_xml_path`` so mesh paths resolve deterministically.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .cameras import Intrinsics, intrinsics_from_file
from .config import (
    ASSETS_DIR,
    GENERATED_SCENE_PATH,
    UR5E_DIR,
    GeomSpec,
    ObjectSpec,
    SimConfig,
    object_spec,
)
from .suction import PAD_BODY_NAME, PAD_GEOM_NAME

#: Menagerie joint order == qpos order of the arm (all revolute, 1 dof each).
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

#: Menagerie ``home`` keyframe (also the ROS ``home`` in the SRDF)
ARM_HOME_QPOS = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
ARM_HOME_CTRL = ARM_HOME_QPOS


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _fmt(values) -> str:
    """Format numbers for MJCF floats."""
    return " ".join(f"{float(v):.6g}" for v in values)


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    """WXYZ quaternion for a rotation about world z."""
    half = math.radians(deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _camera_xyaxes(pos, target) -> tuple[float, float, float, float, float, float]:
    """xyaxes for an MJCF camera at ``pos`` looking at ``target``.

    MuJoCo cameras look along their **-z** axis (+x right, +y up), so the
    camera z-axis points from target back to the camera.
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    z = pos - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return (*x, *y)


def _geom_extent_axis(g: GeomSpec) -> str:
    """MJCF geom attributes for a primitive spec."""
    attrs = {
        "type": g.type,
        "size": _fmt(g.size),
        "pos": _fmt(g.pos),
        "rgba": _fmt(g.rgba),
    }
    if any(abs(v) > 1e-9 for v in g.euler_deg):
        # compiler angle="radian"; GeomSpec stores degrees
        attrs["euler"] = _fmt(np.radians(g.euler_deg))
    return attrs


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_scene(
    config: SimConfig | None = None,
    objects: list[str] | tuple[str, ...] = ("white_foam",),
    out_path: Path | None = None,
) -> Path:
    """Generate the MJCF scene file and return its path.

    Parameters
    ----------
    config:
        cell layout; defaults are calibrated to the real workspace
    objects:
        names from :data:`mujoco_sim.config.OBJ_SPECS`
    out_path:
        where to write the XML (defaults to ``assets/generated_scene.xml``)
    """
    cfg = config or SimConfig()
    specs = [object_spec(name) for name in objects]
    out_path = Path(out_path or GENERATED_SCENE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    intr = intrinsics_from_file(cfg.intrinsics_path) if cfg.intrinsics_path else None
    intr = intr or Intrinsics()
    fovy = math.degrees(2.0 * math.atan(intr.height / (2.0 * intr.fy)))

    ref_root = ET.parse(UR5E_DIR / "ur5e_reference.xml").getroot()
    ref_default = ref_root.find("default")
    ref_asset = ref_root.find("asset")
    ref_arm_body = ref_root.find("worldbody/body")
    ref_actuator = ref_root.find("actuator")
    if ref_arm_body is None or ref_default is None or ref_actuator is None:
        raise RuntimeError(
            "unexpected Menagerie ur5e_reference.xml structure - "
            "re-download universal_robots_ur5e"
        )
    arm_home_qpos = list(ARM_HOME_QPOS)

    # --- root --------------------------------------------------------------
    root = ET.Element("mujoco", model="ur5e_picobot")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir=str(ASSETS_DIR.resolve()),
        autolimits="true",
        balanceinertia="true",
    )
    ET.SubElement(
        root,
        "option",
        timestep=_fmt([cfg.timestep_s]),
        integrator="implicitfast",
        cone=cfg.contact_cone,
        gravity="0 0 -9.81",
        impratio="10",
    )
    ET.SubElement(root, "size", njmax="600", nconmax="250")
    ET.SubElement(
        root, "visual",
    )
    visual = root.find("visual")
    ET.SubElement(
        visual, "global", offwidth=str(intr.width), offheight=str(intr.height),
    )
    ET.SubElement(
        visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5",
        specular="0.1 0.1 0.1",
    )

    # --- defaults -----------------------------------------------------------
    defaults = _copy(ref_default)
    _add_picobot_defaults(defaults)
    _add_payload_defaults(defaults)
    root.append(defaults)

    # --- assets -------------------------------------------------------------
    asset = _copy(ref_asset)
    # Menagerie meshes now live one directory down (meshdir = assets root)
    for mesh in asset.findall("mesh"):
        filename = mesh.get("file", "")
        mesh.set("file", f"ur5e/{Path(filename).name}")
    for name in ("picobot_body", "b30_cup", "yellow_collar", "hex_nut"):
        ET.SubElement(asset, "mesh", name=name, file=f"tool/{name}.obj")
    seen_meshes: set[str] = set()
    for spec in specs:
        for vm in spec.visual_meshes:
            if vm.file not in seen_meshes:
                seen_meshes.add(vm.file)
                ET.SubElement(
                    asset, "mesh", name=f"obj_mesh_{vm.file}",
                    file=f"objects/{vm.file}.obj",
                )
    ET.SubElement(
        asset, "texture", name="floor_tex", type="2d", builtin="checker",
        rgb1="0.30 0.31 0.32", rgb2="0.34 0.35 0.36", width="512", height="512",
    )
    ET.SubElement(
        asset, "material", name="floor_mat", texture="floor_tex",
        texrepeat="6 6", reflectance="0.05",
    )
    root.append(asset)

    # --- worldbody ------------------------------------------------------------
    world = ET.SubElement(root, "worldbody")

    ET.SubElement(
        world, "light", name="key", pos="0.2 -0.6 2.3", dir="0.15 0.25 -1",
        directional="true", diffuse="0.75 0.75 0.75", specular="0.25 0.25 0.25",
    )
    ET.SubElement(
        world, "light", name="fill", pos="1.6 1.2 1.8", dir="-0.5 -0.4 -1",
        directional="true", diffuse="0.35 0.35 0.4", specular="0.1 0.1 0.1",
    )

    if cfg.include_floor:
        ET.SubElement(
            world, "geom", name="floor", type="plane", size="8 8 0.1",
            rgba="0.32 0.33 0.34 1", material="floor_mat",
            friction="1.0 0.005 0.0001", priority="2",
        )

    _build_table(world, cfg)
    _build_robot(world, cfg, ref_arm_body, fovy)
    _build_objects(world, cfg, specs)

    # external RGB-D camera, calibrated to the configured mount
    ET.SubElement(
        world, "camera", name="external",
        pos=_fmt(cfg.external_cam_xyz),
        xyaxes=_fmt(_camera_xyaxes(cfg.external_cam_xyz, cfg.external_cam_target)),
        fovy=_fmt([fovy]),
    )

    # --- actuators ------------------------------------------------------------
    actuator = _copy(ref_actuator)
    root.append(actuator)

    ET.SubElement(root, "statistic", center="0.30 0 0.82", extent="1.15")

    # No MJCF keyframe: compliant cup/pad and soft-body DOFs are respawned by
    # SuctionEnv.reset() (mj_resetData + explicit object poses).

    # --- serialize -------------------------------------------------------------
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    tmp_path.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _copy(elem: ET.Element) -> ET.Element:
    import copy

    return copy.deepcopy(elem)


def _add_picobot_defaults(defaults: ET.Element) -> None:
    picobot = ET.SubElement(defaults, "default", **{"class": "picobot"})
    ET.SubElement(
        picobot, "geom", type="cylinder", contype="1", conaffinity="1",
        condim="3", friction="0.9 0.005 0.0001",
        solref="0.006 1", solimp="0.92 0.97 0.001 0.5 2",
        rgba="0.16 0.17 0.19 1",
    )
    ET.SubElement(
        picobot, "site", size="0.003 0.003 0.003", rgba="0.9 0.2 0.2 0.6",
        group="4",
    )
    cup = ET.SubElement(picobot, "default", **{"class": "picobot_cup"})
    # NB: adhesion must be nonzero at *compile* time to set mjModel.flg_adhesion
    # (engine_setconst.c).  The controller overwrites this tiny value every
    # step with the real vacuum force, so 1e-9 N is just a fuse.
    ET.SubElement(
        cup, "geom", type="cylinder", contype="1", conaffinity="1",
        condim="3", friction="0.8 0.005 0.0001", priority="1",
        solref="0.003 1", solimp="0.92 0.97 0.001 0.5 2",
        adhesion="1e-9", rgba="0.07 0.07 0.08 1",
    )


def _add_payload_defaults(defaults: ET.Element) -> None:
    payload = ET.SubElement(defaults, "default", **{"class": "payload"})
    ET.SubElement(
        payload, "geom", type="box", contype="1", conaffinity="1", condim="3",
        friction="1.0 0.005 0.0001", solref="0.004 1",
        solimp="0.9 0.95 0.001 0.5 2",
    )


def _build_table(world: ET.Element, cfg: SimConfig) -> None:
    cx, cy = cfg.table_center_xy
    hx, hy = cfg.table_half_xy
    thickness = cfg.table_thickness_m
    top_z = cfg.table_height_m
    ET.SubElement(
        world, "geom", name="table_top", type="box",
        pos=_fmt([cx, cy, top_z - thickness * 0.5]),
        size=_fmt([hx, hy, thickness * 0.5]),
        rgba="0.46 0.33 0.22 1", friction="0.9 0.005 0.0001",
        priority="2",
    )
    leg_z = (top_z - thickness) * 0.5
    ET.SubElement(
        world, "geom", name="table_base", type="box",
        pos=_fmt([cx, cy, leg_z]),
        size=_fmt([hx * 0.88, hy * 0.85, leg_z]),
        rgba="0.33 0.24 0.17 1", friction="0.9 0.005 0.0001",
        priority="2",
    )


def _build_robot(world: ET.Element, cfg: SimConfig, ref_arm_body: ET.Element,
                 fovy_deg: float) -> None:
    mount = ET.SubElement(
        world, "body", name="robot_mount",
        pos=_fmt(cfg.robot_mount_xyz),
        quat=_fmt(_yaw_quat(cfg.robot_yaw_deg)),
    )
    arm = _copy(ref_arm_body)
    mount.append(arm)

    # find wrist_3 / attachment site in the copied Menagerie subtree
    wrist3 = None
    for body in arm.iter("body"):
        if body.get("name") == "wrist_3_link":
            wrist3 = body
            break
    if wrist3 is None:
        raise RuntimeError("wrist_3_link not found in Menagerie UR5e model")
    site = None
    for s in wrist3.iter("site"):
        if s.get("name") == "attachment_site":
            site = s
            break
    tool_pos = site.get("pos", "0 0 0")
    tool_quat = site.get("quat", "1 0 0 0")
    _build_picobot(wrist3, cfg, tool_pos, tool_quat, fovy_deg)


def _build_picobot(
    wrist3: ET.Element, cfg: SimConfig, tool_pos: str, tool_quat: str,
    fovy_deg: float,
) -> None:
    """piCOBOT Electric + black extension + B30-2 silicone cup.

    Stack (tool0 z=0, +z towards the cup)::

        z=0.000   UR/ISO adapter plate (12 mm)
        z=0.012   piCOBOT Electric housing (110.2 mm, 120 x 124.4 mm)
        z=0.1222  40x30 tool plate, black extension tube (Ø12.5 mm)
        z=0.3265  hex fitting
        z=0.337   yellow collar (Ø39 x 9 mm)
        z=0.3438  B30-2 silicone bellows cup (26.2 mm, Ø34/36)
        z=0.3700  cup lip = suction_cup_tip
    """
    adapter = cfg.ur_adapter_thickness_m
    body_h = cfg.picobot_body_height_m
    body_w = cfg.picobot_body_width_m
    body_d = cfg.picobot_body_depth_m
    body_bottom = cfg.tool_body_bottom_m
    cup_top = cfg.cup_top_z
    ext_len = cfg.extension_length_m
    tube_r = cfg.extension_tube_od_m * 0.5

    tool = ET.SubElement(
        wrist3, "body", name="picobot", pos=tool_pos, quat=tool_quat,
        childclass="picobot",
    )
    ET.SubElement(tool, "site", name="tool0", pos="0 0 0",
                  size="0.004 0.004 0.004")

    # ---- combined inertial (pump + extension + collar + camera) ------------
    cam = np.asarray(cfg.wrist_cam_pos_tool, dtype=float)
    parts = [
        (cfg.picobot_mass_kg, np.array([0.0, 0.0, adapter + 0.5 * body_h])),
        (cfg.extension_mass_kg, np.array([0.0, 0.0, body_bottom + 0.5 * ext_len])),
    ]
    total_mass = sum(m for m, _ in parts)
    cg = sum(m * p for m, p in parts) / total_mass

    def box_inertia(m, sx, sy, sz):
        return np.array([
            m / 12.0 * (sy * sy + sz * sz),
            m / 12.0 * (sx * sx + sz * sz),
            m / 12.0 * (sx * sx + sy * sy),
        ])

    inertia = np.zeros(3)
    for m, p in parts:
        d = p - cg
        inertia += m * np.array([d[1] ** 2 + d[2] ** 2,
                                 d[0] ** 2 + d[2] ** 2,
                                 d[0] ** 2 + d[1] ** 2])
    inertia += box_inertia(cfg.picobot_mass_kg, body_w, body_d, body_h)
    inertia += np.array([
        cfg.extension_mass_kg * (3 * tube_r ** 2 + ext_len ** 2) / 12.0,
        cfg.extension_mass_kg * (3 * tube_r ** 2 + ext_len ** 2) / 12.0,
        0.5 * cfg.extension_mass_kg * tube_r ** 2,
    ])
    ET.SubElement(
        tool, "inertial", mass=_fmt([total_mass]), pos=_fmt(cg),
        diaginertia=_fmt(np.maximum(inertia, 1e-6)),
    )

    # ---- robot adapter + pump housing ---------------------------------------
    ET.SubElement(
        tool, "geom", name="ur_adapter", type="cylinder",
        size=_fmt([0.029, adapter * 0.5]), pos=_fmt([0, 0, adapter * 0.5]),
        rgba="0.26 0.27 0.29 1",
    )
    ET.SubElement(
        tool, "geom", name="picobot_shell", type="mesh", mesh="picobot_body",
        pos=_fmt([0, 0, adapter]), rgba="0.075 0.08 0.09 1",
    )
    # OLED + LED strip on the operator side (-Y)
    y_face = -body_d * 0.5 - 0.0006
    ET.SubElement(
        tool, "geom", name="display_bezel", type="box",
        size=_fmt([0.026, 0.0008, 0.014]),
        pos=_fmt([-0.004, y_face, adapter + 0.072]),
        rgba="0.02 0.02 0.025 1", contype="0", conaffinity="0",
    )
    ET.SubElement(
        tool, "geom", name="display_screen", type="box",
        size=_fmt([0.021, 0.0006, 0.010]),
        pos=_fmt([-0.004, y_face - 0.0004, adapter + 0.072]),
        rgba="0.015 0.02 0.03 1", contype="0", conaffinity="0",
    )
    ET.SubElement(
        tool, "geom", name="display_led", type="box",
        size=_fmt([0.030, 0.0006, 0.0020]),
        pos=_fmt([-0.004, y_face - 0.0004, adapter + 0.048]),
        rgba="0.55 0.95 0.15 1", contype="0", conaffinity="0",
    )
    for i, dx in enumerate((-0.016, -0.004, 0.008)):
        ET.SubElement(
            tool, "geom", name=f"display_digits_{i}", type="box",
            size=_fmt([0.006, 0.0004, 0.0015]),
            pos=_fmt([dx, y_face - 0.0006, adapter + 0.069]),
            rgba="0.15 0.85 0.95 1", contype="0", conaffinity="0",
        )
    # tool-side plate + black extension
    ET.SubElement(
        tool, "geom", name="tool_plate", type="box",
        size=_fmt([0.020, 0.015, 0.002]),
        pos=_fmt([0, 0, body_bottom - 0.002]),
        rgba="0.32 0.33 0.35 1",
    )
    ET.SubElement(
        tool, "geom", name="extension_tube", type="cylinder",
        size=_fmt([tube_r, ext_len * 0.5]),
        pos=_fmt([0, 0, body_bottom + ext_len * 0.5]),
        rgba="0.045 0.045 0.05 1",
    )
    ET.SubElement(
        tool, "geom", name="hex_nut", type="mesh", mesh="hex_nut",
        pos=_fmt([0, 0, cup_top - 0.0173]), rgba="0.11 0.11 0.12 1",
    )

    # Sensor only: the physical camera/bracket is on the robot wrist, not on
    # the piCOBOT body (removed from the tool visuals by request).
    ET.SubElement(
        tool, "camera", name="wrist", pos=_fmt(cam),
        xyaxes=_fmt(cfg.wrist_cam_xyaxes_tool), fovy=_fmt([fovy_deg]),
    )

    # ---- cup: compliant bellows + sprung lip (or fast rigid fallback) ------
    if cfg.cup_model == "compliant":
        _build_compliant_cup(tool, cfg, cup_top)
    else:
        _build_rigid_cup(tool, cfg, cup_top)


def _build_cup_visuals(cup: ET.Element, cfg: SimConfig) -> None:
    ET.SubElement(
        cup, "geom", name="b30_cup", type="mesh", mesh="b30_cup",
        pos="0 0 0", rgba="0.32 0.72 0.52 0.90",
        contype="0", conaffinity="0",
    )
    ET.SubElement(
        cup, "geom", name="yellow_collar", type="mesh", mesh="yellow_collar",
        pos=_fmt([0, 0, -0.0068]), rgba="0.95 0.80 0.10 1",
        contype="0", conaffinity="0",
    )
    ET.SubElement(
        cup, "site", name="suction_cup_tip",
        pos=_fmt([0, 0, cfg.cup_height_m]),
        size="0.004 0.004 0.004", rgba="0.1 0.9 0.1 0.8",
    )


def _build_compliant_cup(tool: ET.Element, cfg: SimConfig, cup_top: float) -> None:
    """B30 cup on a compliant bellows mount with individually sprung lip pads."""
    mount = ET.SubElement(tool, "body", name="cup_mount",
                          pos=_fmt([0, 0, cup_top]))
    cup = ET.SubElement(
        mount, "body", name=PAD_BODY_NAME, childclass="picobot_cup",
    )
    ET.SubElement(
        cup, "joint", name="cup_axial", type="slide", axis="0 0 1",
        range=_fmt([-cfg.cup_axial_travel_m, 0.002]),
        stiffness=_fmt([cfg.cup_axial_stiffness_n_m]),
        damping=_fmt([cfg.cup_axial_damping_ns_m]), springref="0",
    )
    ET.SubElement(
        cup, "joint", name="cup_tilt_x", type="hinge", axis="1 0 0",
        range=_fmt([-cfg.cup_tilt_limit_rad, cfg.cup_tilt_limit_rad]),
        stiffness=_fmt([cfg.cup_tilt_stiffness_nm_rad]),
        damping=_fmt([cfg.cup_tilt_damping_nms_rad]), springref="0",
    )
    ET.SubElement(
        cup, "joint", name="cup_tilt_y", type="hinge", axis="0 1 0",
        range=_fmt([-cfg.cup_tilt_limit_rad, cfg.cup_tilt_limit_rad]),
        stiffness=_fmt([cfg.cup_tilt_stiffness_nm_rad]),
        damping=_fmt([cfg.cup_tilt_damping_nms_rad]), springref="0",
    )
    ET.SubElement(
        cup, "inertial", mass=_fmt([cfg.cup_mass_kg + cfg.collar_mass_kg]),
        pos="0 0 0.012", diaginertia="4e-6 4e-6 3e-6",
    )
    _build_cup_visuals(cup, cfg)

    # lip pads: each is a short tangential rubber capsule on its own spring
    ring = cfg.cup_seal_radius_m - cfg.lip_pad_radius_m
    circumference = 2.0 * math.pi * ring
    half_len = max(0.0004, circumference / cfg.lip_pad_count * 0.5
                   - cfg.lip_pad_radius_m)
    for i in range(cfg.lip_pad_count):
        ang = 2.0 * math.pi * i / cfg.lip_pad_count
        # capsule axis tangent to the ring
        qw = math.sqrt(0.5)
        quat = (qw, -qw * math.cos(ang), -qw * math.sin(ang), 0.0)
        pad = ET.SubElement(
            cup, "body", name=f"lip_pad_{i}",
            pos=_fmt([ring * math.cos(ang), ring * math.sin(ang),
                      cfg.cup_height_m - cfg.lip_pad_radius_m]),
            quat=_fmt(quat),
        )
        ET.SubElement(
            pad, "joint", name=f"lip_pad_joint_{i}", type="slide",
            axis="0 0 1",
            range=_fmt([-cfg.lip_pad_travel_m, 0.0015]),
            stiffness=_fmt([cfg.lip_pad_stiffness_n_m]),
            damping=_fmt([cfg.lip_pad_damping_ns_m]), springref="0",
        )
        ET.SubElement(
            pad, "inertial", mass=_fmt([cfg.lip_pad_mass_kg]), pos="0 0 0",
            diaginertia="1e-7 1e-7 1e-7",
        )
        ET.SubElement(
            pad, "geom", name=f"{PAD_GEOM_NAME}_{i}", type="capsule",
            size=_fmt([cfg.lip_pad_radius_m, half_len]),
            rgba="0.24 0.55 0.42 1", group="2",
            contype="2", conaffinity="1",
            gap=_fmt([cfg.cup_gap_m]), priority="1", condim="3",
            friction="1.0 0.01 0.001", solref="0.004 1",
            solimp="0.9 0.95 0.001 0.5 2",
        )


def _build_rigid_cup(tool: ET.Element, cfg: SimConfig, cup_top: float) -> None:
    """Fast legacy cup: rigid B30 body with eight contact points."""
    cup = ET.SubElement(
        tool, "body", name=PAD_BODY_NAME, pos=_fmt([0, 0, cup_top]),
        childclass="picobot_cup",
    )
    ET.SubElement(
        cup, "inertial", mass=_fmt([cfg.cup_mass_kg + cfg.collar_mass_kg]),
        pos="0 0 0.010", diaginertia="2e-6 2e-6 1.5e-6",
    )
    _build_cup_visuals(cup, cfg)
    n_lip = 8
    sphere_r = 0.0025
    ring_radius = cfg.cup_seal_radius_m - sphere_r
    for i in range(n_lip):
        ang = 2.0 * math.pi * i / n_lip
        pos = (ring_radius * math.cos(ang), ring_radius * math.sin(ang),
               cfg.cup_height_m - sphere_r)
        ET.SubElement(
            cup, "geom", name=f"{PAD_GEOM_NAME}_{i}", type="sphere",
            size=_fmt([sphere_r]), pos=_fmt(pos),
            gap=_fmt([cfg.cup_gap_m]), group="3",
        )


def _build_objects(world: ET.Element, cfg: SimConfig, specs: list[ObjectSpec]) -> None:
    for spec in specs:
        x, y = cfg.placement(spec.name)
        yaw = cfg.object_yaw_deg.get(spec.name, 0.0)
        body = ET.SubElement(
            world, "body", name=f"obj_{spec.name}",
            pos=_fmt([x, y, cfg.table_height_m]),
            quat=_fmt(_yaw_quat(yaw)),
            childclass="payload",
        )
        lo, hi = spec.aabb()
        ixx, iyy, izz = spec.inertia()
        skin_count = spec.soft_skin_counts[0] * spec.soft_skin_counts[1]
        skin_mass = skin_count * spec.soft_skin_mass_kg if spec.soft else 0.0
        body_mass = max(spec.mass_kg - skin_mass, 0.01) if spec.soft else spec.mass_kg
        ET.SubElement(
            body, "inertial", mass=_fmt([body_mass]),
            pos=_fmt([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]),
                      0.5 * (lo[2] + hi[2])]),
            diaginertia=_fmt([ixx, iyy, izz]),
        )
        ET.SubElement(body, "freejoint", name=f"{spec.name}_free")
        for i, g in enumerate(spec.geoms):
            attrs = _geom_extent_axis(g)
            attrs["name"] = f"obj_{spec.name}_geom_{i}"
            if spec.visual_meshes:
                # collision proxy: keep it out of the rendered scene
                attrs["group"] = "3"
            ET.SubElement(body, "geom", **attrs)

        for vm in spec.visual_meshes:
            attrs = {
                "name": f"obj_{spec.name}_vis_{vm.file}",
                "type": "mesh",
                "mesh": f"obj_mesh_{vm.file}",
                "pos": _fmt(vm.pos),
                "rgba": _fmt(vm.rgba),
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            }
            if any(abs(v) > 1e-9 for v in vm.euler_deg):
                attrs["euler"] = _fmt(np.radians(vm.euler_deg))
            ET.SubElement(body, "geom", **attrs)

        if spec.soft:
            # deformable top skin: a grid of sprung contact pads on the top
            # surface.  The payload remains a stable rigid free body, while
            # the surface compresses locally under the cup.
            nx, ny = spec.soft_skin_counts
            pr = spec.soft_skin_radius_m
            ph = 0.5 * spec.soft_skin_height_m
            top = hi[2]
            for iy in range(ny):
                for ix in range(nx):
                    x = lo[0] + pr + (hi[0] - lo[0] - 2 * pr) * ix / max(nx - 1, 1)
                    y = lo[1] + pr + (hi[1] - lo[1] - 2 * pr) * iy / max(ny - 1, 1)
                    pad = ET.SubElement(
                        body, "body", name=f"obj_{spec.name}_skin_{ix}_{iy}",
                        pos=_fmt([x, y, top + ph]),
                    )
                    ET.SubElement(
                        pad, "joint", name=f"obj_{spec.name}_skin_j_{ix}_{iy}",
                        type="slide", axis="0 0 1",
                        range=_fmt([-0.001, spec.soft_skin_travel_m]),
                        stiffness=_fmt([spec.soft_skin_stiffness_n_m]),
                        damping=_fmt([spec.soft_skin_damping_ns_m]),
                        springref="0",
                    )
                    ET.SubElement(
                        pad, "inertial", mass=_fmt([spec.soft_skin_mass_kg]),
                        pos="0 0 0", diaginertia="2e-8 2e-8 2e-8",
                    )
                    ET.SubElement(
                        pad, "geom", name=f"obj_{spec.name}_skin_geom_{ix}_{iy}",
                        type="cylinder", size=_fmt([pr, ph]), pos="0 0 0",
                        rgba=_fmt(spec.geoms[0].rgba),
                        group="2" if spec.soft_skin_visible else "3",
                        contype="1", conaffinity="1",
                        friction="1.0 0.01 0.001", condim="3",
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import OBJ_SPECS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=["white_foam"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--list-objects", action="store_true")
    args = parser.parse_args(argv)

    if args.list_objects:
        for name, spec in OBJ_SPECS.items():
            print(f"{name:14s} mass={spec.mass_kg:5.2f} kg  "
                  f"seal={spec.seal_quality:4.2f}  height={spec.height * 1e3:5.1f} mm")
        return 0

    path = build_scene(SimConfig(), args.objects, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
