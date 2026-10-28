"""Configuration and object registry for the UR5e + piCOBOT MuJoCo scene.

The defaults mirror the real cell documented in ``study/device_config.md``:

- UR5e mounted on the work table, ``tool0 -> suction_cup_tip`` = 0.370 m
- Piab piCOBOT Electric, single ~30 mm bellows cup
- table-top workspace in front of the arm

Everything spatial is in *world* coordinates.  The robot base is placed by
``SimConfig.robot_mount_*``; objects by ``SimConfig.object_placements``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pneumatics import PicobotParams, VacuumParams

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
UR5E_DIR = ASSETS_DIR / "ur5e"
GENERATED_SCENE_PATH = ASSETS_DIR / "generated_scene.xml"

#: world-frame gravity used throughout the package
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Object primitives (approximations of the study corpus)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeomSpec:
    """One primitive collision/visual geom of an object.

    The object body origin sits on the table surface (z = 0 of the body), so
    ``pos`` values are naturally "height above the table".
    """

    type: str  # box | cylinder | sphere | capsule | ellipsoid
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)


@dataclass(frozen=True)
class VisualMesh:
    """One render-only mesh attached to an object body."""

    file: str  # stem under ``assets/objects``
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)


@dataclass(frozen=True)
class ObjectSpec:
    """A pickable object: primitive geoms, mass, and sealing behaviour."""

    name: str
    mass_kg: float
    #: fraction of the ideal vacuum force the cup keeps on this object's
    #: surface (1.0 = perfect seal, curved/porous surfaces seal worse)
    seal_quality: float
    geoms: tuple[GeomSpec, ...]
    #: render-only meshes (collision remains the primitive geoms above)
    visual_meshes: tuple[VisualMesh, ...] = ()
    #: pick point in the object frame relative to the collision AABB top
    pick_offset_xy: tuple[float, float] = (0.0, 0.0)
    #: vertical pick-surface offset from the AABB top (negative = inside a
    #: vessel, e.g. the interior floor of a bowl or mug)
    pick_offset_z: float = 0.0
    #: porosity/leak of the object's surface, m^3/s/Pa (paper/foam leak more)
    leak_m3_s_pa: float = 2.0e-11
    #: deformable contact skin used for soft payloads (foam/paper)
    soft: bool = False
    soft_skin_counts: tuple[int, int] = (6, 6)
    soft_skin_radius_m: float = 0.004
    soft_skin_stiffness_n_m: float = 700.0
    soft_skin_damping_ns_m: float = 1.5
    soft_skin_travel_m: float = 0.006
    soft_skin_mass_kg: float = 0.0002
    #: full thickness of each flat skin pad (visual + contact height)
    soft_skin_height_m: float = 0.0016
    #: draw the skin pads (off by default: they look like studs on hard shells)
    soft_skin_visible: bool = False

    def _rotation(self, euler_deg: tuple[float, float, float]) -> np.ndarray:
        a, b, c = (math.radians(v) for v in euler_deg)
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        cc, sc = math.cos(c), math.sin(c)
        # intrinsic z-y-x (same convention as MuJoCo ``euler`` with xyz axes)
        return np.array(
            [
                [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
                [-sb, cb * sa, cb * ca],
            ]
        )

    @staticmethod
    def _primitive_half_extents(geom: GeomSpec) -> np.ndarray:
        t, s = geom.type, geom.size
        if t == "box":
            return np.array(s[:3], dtype=float)
        if t == "sphere":
            return np.array([s[0], s[0], s[0]])
        if t == "ellipsoid":
            return np.array(s[:3], dtype=float)
        if t in ("cylinder", "capsule"):
            # MJCF: cylinder/capsule size = (radius, half_length)
            return np.array([s[0], s[0], s[1]])
        raise ValueError(f"unknown geom type {t!r}")

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Object AABB in its local (table-aligned) frame."""
        lo = np.full(3, math.inf)
        hi = np.full(3, -math.inf)
        for g in self.geoms:
            R = self._rotation(g.euler_deg)
            half = np.abs(R) @ self._primitive_half_extents(g)
            lo = np.minimum(lo, np.asarray(g.pos) - half)
            hi = np.maximum(hi, np.asarray(g.pos) + half)
        return lo, hi

    def z_bounds(self) -> tuple[float, float]:
        """Lowest / highest point of the object in its local frame."""
        lo, hi = self.aabb()
        return float(lo[2]), float(hi[2])

    @property
    def height(self) -> float:
        lo, hi = self.z_bounds()
        return hi - lo

    @property
    def top_offset(self) -> float:
        """Height of the highest point above the body origin."""
        return self.z_bounds()[1]

    @property
    def bottom_offset(self) -> float:
        return self.z_bounds()[0]

    def inertia(self) -> tuple[float, float, float]:
        """Inertia of a box with the object's AABB extents (udf-free bodies)."""
        lo, hi = self.aabb()
        dx, dy, dz = hi - lo
        m = self.mass_kg
        return (
            m / 12.0 * (dy * dy + dz * dz),
            m / 12.0 * (dx * dx + dz * dz),
            m / 12.0 * (dx * dx + dy * dy),
        )


# --- geometry helpers -------------------------------------------------------

_WHITE = (0.92, 0.92, 0.90, 1.0)
_RED = (0.75, 0.16, 0.14, 1.0)
_DARK = (0.16, 0.17, 0.19, 1.0)
_CERAMIC = (0.93, 0.93, 0.95, 1.0)
_BLUE = (0.20, 0.34, 0.60, 1.0)
_GREY = (0.55, 0.57, 0.60, 1.0)
_RUBBER = (0.62, 0.10, 0.10, 1.0)
_CAP = (0.12, 0.28, 0.58, 1.0)


def _wall_ring(
    radius: float,
    thickness: float,
    z0: float,
    z1: float,
    count: int,
    rgba: tuple[float, float, float, float],
) -> tuple[GeomSpec, ...]:
    """Ring of box walls (approximates a hollow circular vessel)."""
    zc = 0.5 * (z0 + z1)
    hz = 0.5 * (z1 - z0)
    chord = 2.0 * radius * math.sin(math.pi / count)
    out = []
    for i in range(count):
        ang = 2.0 * math.pi * i / count
        out.append(
            GeomSpec(
                "box",
                (0.5 * thickness, 0.55 * chord, hz),
                (radius * math.cos(ang), radius * math.sin(ang), zc),
                euler_deg=(0.0, 0.0, math.degrees(ang)),
                rgba=rgba,
            )
        )
    return tuple(out)


OBJ_SPECS: dict[str, ObjectSpec] = {
    # Seal qualities are calibrated to the real study:
    #   q = 1.55 * m*g / ((P_hold+1kPa)*A_cup)
    # with measured_hold_kpa from study/online/ledger.json and measured masses.
    # The 1.55 factor accounts for the peel margin of the rubber lip (the
    # solver load is uneven across the ring), so the sim breaks away near the
    # measured pressure.  See README "Seal calibration" for the table.
    "white_foam": ObjectSpec(
        "white_foam", mass_kg=0.05, seal_quality=0.099,
        geoms=(GeomSpec("box", (0.075, 0.075, 0.010), (0, 0, 0.010), rgba=_WHITE),),
        visual_meshes=(VisualMesh("white_foam", rgba=_WHITE),),
        leak_m3_s_pa=3.0e-10, soft=True,
        soft_skin_counts=(6, 6), soft_skin_radius_m=0.005,
        soft_skin_stiffness_n_m=500.0, soft_skin_travel_m=0.006,
    ),
    "red_foam": ObjectSpec(
        "red_foam", mass_kg=0.05, seal_quality=0.099,
        geoms=(GeomSpec("box", (0.070, 0.070, 0.010), (0, 0, 0.010), rgba=_RED),),
        visual_meshes=(VisualMesh("red_foam", rgba=_RED),),
        leak_m3_s_pa=3.0e-10, soft=True,
        soft_skin_counts=(6, 6), soft_skin_radius_m=0.005,
        soft_skin_stiffness_n_m=500.0, soft_skin_travel_m=0.006,
    ),
    # Real white ceramic bowl: 130 x 130 mm, 52 mm high, open top.  The cup
    # descends into it and seals on the interior floor (the study picks there
    # too), so collision is a floor slab plus four walls, not a solid cylinder.
    "white_bowl": ObjectSpec(
        "white_bowl", mass_kg=0.219, seal_quality=0.552,
        geoms=(
            GeomSpec("box", (0.060, 0.060, 0.004), (0, 0, 0.004), rgba=_WHITE),
            GeomSpec("box", (0.0025, 0.065, 0.022), (0.0625, 0, 0.030), rgba=_WHITE),
            GeomSpec("box", (0.0025, 0.065, 0.022), (-0.0625, 0, 0.030), rgba=_WHITE),
            GeomSpec("box", (0.060, 0.0025, 0.022), (0, 0.0625, 0.030), rgba=_WHITE),
            GeomSpec("box", (0.060, 0.0025, 0.022), (0, -0.0625, 0.030), rgba=_WHITE),
        ),
        visual_meshes=(VisualMesh("white_bowl", rgba=_WHITE),),
        pick_offset_z=-0.044,  # interior floor at z=0.008, rim top at 0.052
    ),
    # Ceramic mug: Ø84 mm, 90 mm high, hollow, handle on -x.
    "ceramic_mug": ObjectSpec(
        "ceramic_mug", mass_kg=0.248, seal_quality=0.832,
        geoms=(
            GeomSpec("cylinder", (0.042, 0.003), (0, 0, 0.003), rgba=_CERAMIC),
            *_wall_ring(0.040, 0.004, 0.006, 0.090, 12, _CERAMIC),
            GeomSpec("capsule", (0.006, 0.030), (-0.072, 0, 0.045), rgba=_CERAMIC),
            GeomSpec("capsule", (0.006, 0.012), (-0.054, 0, 0.075),
                     euler_deg=(0, 90, 0), rgba=_CERAMIC),
            GeomSpec("capsule", (0.006, 0.012), (-0.054, 0, 0.015),
                     euler_deg=(0, 90, 0), rgba=_CERAMIC),
        ),
        visual_meshes=(
            VisualMesh("mug_body", rgba=_CERAMIC),
            VisualMesh("mug_handle", (-0.042, 0, 0.045), rgba=_CERAMIC),
        ),
        pick_offset_z=-0.084,  # interior floor at z=0.006
    ),
    "bottle": ObjectSpec(
        "bottle", mass_kg=0.60, seal_quality=0.741,
        geoms=(
            GeomSpec("cylinder", (0.033, 0.064), (0, 0, 0.064), rgba=_WHITE),
            GeomSpec("cylinder", (0.022, 0.034), (0, 0, 0.162), rgba=_WHITE),
            GeomSpec("cylinder", (0.019, 0.014), (0, 0, 0.210), rgba=_CAP),
        ),
        visual_meshes=(
            VisualMesh("bottle", rgba=_WHITE),
            VisualMesh("bottle_label", (0, 0, 0.055), rgba=_CAP),
            VisualMesh("bottle_cap", (0, 0, 0.206), rgba=_CAP),
        ),
    ),
    "soda_can": ObjectSpec(
        "soda_can", mass_kg=0.37, seal_quality=0.622,
        geoms=(GeomSpec("cylinder", (0.033, 0.0575), (0, 0, 0.0575), rgba=_RED),),
        visual_meshes=(
            VisualMesh("soda_can", rgba=_RED),
            VisualMesh("can_lid", (0, 0, 0.113), rgba=(0.72, 0.73, 0.75, 1.0)),
        ),
    ),
    "magazine": ObjectSpec(
        "magazine", mass_kg=0.045, seal_quality=0.113,
        geoms=(GeomSpec("box", (0.105, 0.140, 0.004), (0, 0, 0.004), rgba=_GREY),),
        visual_meshes=(
            VisualMesh("magazine", rgba=_GREY),
            VisualMesh("magazine_cover", (0, 0, 0.0075), rgba=_WHITE),
            VisualMesh("magazine_band", (-0.060, 0, 0.0090), rgba=_RED),
        ),
        leak_m3_s_pa=6.0e-10, soft=True,
        soft_skin_counts=(8, 10), soft_skin_radius_m=0.0025,
        soft_skin_stiffness_n_m=1500.0, soft_skin_travel_m=0.003,
    ),
    # Table-tennis racket: Ø160 mm blade ahead of the handle, red rubber face.
    "racket": ObjectSpec(
        "racket", mass_kg=0.168, seal_quality=0.677,
        geoms=(
            GeomSpec("cylinder", (0.080, 0.0075), (0, 0.085, 0.0075), rgba=_DARK),
            GeomSpec("box", (0.012, 0.055, 0.006), (0, -0.060, 0.006), rgba=_DARK),
        ),
        visual_meshes=(
            VisualMesh("racket_blade", (0, 0.085, 0), rgba=_DARK),
            VisualMesh("racket_rubber", (0, 0.085, 0.0140), rgba=_RUBBER),
            VisualMesh("racket_handle", (0, -0.060, 0), rgba=_DARK),
        ),
        pick_offset_xy=(0.0, 0.085),
    ),
    # Toy rocket lying on its side: body along y, nose at -y.
    "rocket": ObjectSpec(
        "rocket", mass_kg=0.09, seal_quality=0.2020,
        geoms=(
            GeomSpec("cylinder", (0.028, 0.120), (0, 0, 0.028),
                     euler_deg=(90, 0, 0), rgba=_WHITE),
            GeomSpec("cylinder", (0.022, 0.030), (0, -0.150, 0.028),
                     euler_deg=(90, 0, 0), rgba=_RED),
            GeomSpec("sphere", (0.006,), (0, -0.180, 0.028), rgba=_RED),
        ),
        visual_meshes=(
            VisualMesh("rocket_body", (0, 0.120, 0.028), (90, 0, 0), _WHITE),
            VisualMesh("rocket_nose", (0, -0.120, 0.028), (90, 0, 0), _RED),
        ),
    ),
    "test_cube": ObjectSpec(
        "test_cube", mass_kg=0.20, seal_quality=0.950,
        geoms=(GeomSpec("box", (0.025, 0.025, 0.025), (0, 0, 0.025), rgba=_BLUE),),
        visual_meshes=(VisualMesh("test_cube", rgba=_BLUE),),
    ),
    "heavy_cube": ObjectSpec(
        "heavy_cube", mass_kg=1.50, seal_quality=0.95,
        geoms=(GeomSpec("box", (0.030, 0.030, 0.030), (0, 0, 0.030), rgba=_DARK),),
        visual_meshes=(VisualMesh("heavy_cube", rgba=_DARK),),
    ),
}

#: object placement in world coordinates (x, y) for the default scene
#: (spaced so the default multi-object scene has no initial overlaps)
DEFAULT_PLACEMENTS: dict[str, tuple[float, float]] = {
    "white_foam": (0.34, -0.30),
    "red_foam": (0.34, 0.08),
    "white_bowl": (0.52, -0.32),
    "ceramic_mug": (0.43, 0.24),
    "bottle": (0.66, -0.30),
    "soda_can": (0.64, -0.10),
    "magazine": (0.16, 0.27),
    "racket": (0.68, 0.20),
    "rocket": (0.72, -0.03),
    "test_cube": (0.24, -0.06),
    "heavy_cube": (0.46, 0.02),
}

#: spawn yaw per object (the mug handle faces +x, the racket lies along x)
DEFAULT_YAW_DEG: dict[str, float] = {
    "ceramic_mug": 180.0,
    "racket": 90.0,
}


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Everything spatial/parametric about the simulated cell."""

    # --- timing ------------------------------------------------------------
    timestep_s: float = 0.002
    control_dt_s: float = 0.01  # 100 Hz servo / suction update

    # --- table -------------------------------------------------------------
    table_height_m: float = 0.72
    table_center_xy: tuple[float, float] = (0.22, 0.0)
    table_half_xy: tuple[float, float] = (0.58, 0.42)
    table_thickness_m: float = 0.05

    # --- robot mount ---------------------------------------------------------
    #: base_link position; the arm is bolted to the table top
    robot_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.72)
    #: yaw about world z; -90 deg puts the Menagerie "home" pose over the
    #: table in front of the arm, which is where objects live
    robot_yaw_deg: float = -90.0

    # --- real piCOBOT Electric stack (Piab manual rev.04 / datasheets) ------
    #: tool0 -> suction_cup_tip, measured on the real UR5e cell
    tool_length_m: float = 0.370
    #: piCOBOT Electric incl. UR adapter plate (manual section 8)
    picobot_mass_kg: float = 1.43
    ur_adapter_thickness_m: float = 0.012
    picobot_body_width_m: float = 0.120     # drawing 9. (X)
    picobot_body_depth_m: float = 0.1244    # drawing 9. (Y, core body)
    picobot_body_height_m: float = 0.1102   # drawing 9. (Z below flange)

    #: Piab B30-2 silicone bellows cup, item 0101431 (B30-2.21 datasheet)
    cup_bellows_od_m: float = 0.034
    cup_flange_od_m: float = 0.036
    cup_height_m: float = 0.0262
    cup_mass_kg: float = 0.007
    #: effective sealing-lip radius used for visual + contact ring
    cup_seal_radius_m: float = 0.0170
    #: attraction range of the soft lip (bellows compliance)
    cup_gap_m: float = 0.006  # effective bellows seal band (datasheet travel 14 mm)
    #: "compliant" (deformable bellows + sprung lip) or "rigid" (fast legacy)
    cup_model: str = "compliant"
    #: bellows axial/tilt compliance
    cup_axial_stiffness_n_m: float = 1500.0
    cup_axial_damping_ns_m: float = 15.0
    cup_axial_travel_m: float = 0.012
    cup_tilt_stiffness_nm_rad: float = 4.0
    cup_tilt_damping_nms_rad: float = 0.15
    cup_tilt_limit_rad: float = 0.25
    #: individually sprung rubber lip pads
    lip_pad_count: int = 12
    lip_pad_radius_m: float = 0.0028
    lip_pad_stiffness_n_m: float = 1200.0
    lip_pad_damping_ns_m: float = 3.0
    lip_pad_travel_m: float = 0.004
    lip_pad_mass_kg: float = 0.0002
    #: safety factor on the vacuum pull-off force (seal edges/peel margin)
    grasp_force_safety: float = 0.85

    #: custom installed extension (length derived from the 0.370 m tip)
    extension_tube_od_m: float = 0.0125
    extension_mass_kg: float = 0.030
    collar_od_m: float = 0.039
    collar_height_m: float = 0.009
    collar_mass_kg: float = 0.010

    # --- wrist camera (hand-eye result in launch/publish_wrist_calibration) --
    #: camera origin in the tool0 frame (metres)
    wrist_cam_pos_tool: tuple[float, float, float] = (
        -0.038591, 0.119463, 0.064162,
    )
    #: MuJoCo camera frame in tool0 coordinates (x right, y up, looks -z)
    wrist_cam_xyaxes_tool: tuple[float, ...] = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # --- objects -------------------------------------------------------------
    object_placements: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PLACEMENTS)
    )
    object_yaw_deg: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_YAW_DEG)
    )
    randomize_object_yaw_deg: float = 0.0

    # --- pump / vacuum ---------------------------------------------------
    pump: PicobotParams = field(default_factory=PicobotParams)
    vacuum: VacuumParams = field(default_factory=VacuumParams)

    # --- cameras ------------------------------------------------------------
    external_cam_xyz: tuple[float, float, float] = (-0.30, -0.78, 1.32)
    external_cam_target: tuple[float, float, float] = (0.42, 0.0, 0.72)
    #: optional Orbbec intrinsics JSON (fx/fy/cx/cy/width/height); defaults
    #: match the recorded 1280x720 Orbbec stream
    intrinsics_path: Path | None = None

    # --- scene extras --------------------------------------------------------
    include_floor: bool = True
    contact_cone: str = "elliptic"

    def __post_init__(self) -> None:
        if self.timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        self.control_dt_s = max(self.control_dt_s, self.timestep_s)

    @property
    def substeps_per_control(self) -> int:
        return max(1, int(round(self.control_dt_s / self.timestep_s)))

    @property
    def tool_body_bottom_m(self) -> float:
        """Tool0 -> piCOBOT tool-side face."""
        return self.ur_adapter_thickness_m + self.picobot_body_height_m

    @property
    def cup_top_z(self) -> float:
        """Tool0 -> top of the B30-2 cup (where the extension ends)."""
        return self.tool_length_m - self.cup_height_m

    @property
    def extension_length_m(self) -> float:
        """Length of the installed black extension tube."""
        return self.cup_top_z - self.tool_body_bottom_m

    def placement(self, name: str) -> tuple[float, float]:
        try:
            return self.object_placements[name]
        except KeyError:
            raise KeyError(
                f"no placement for object {name!r}; "
                f"known: {sorted(self.object_placements)}"
            ) from None


def object_spec(name: str) -> ObjectSpec:
    try:
        return OBJ_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown object {name!r}; available: {sorted(OBJ_SPECS)}"
        ) from None
