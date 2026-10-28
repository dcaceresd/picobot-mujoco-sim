"""Procedural meshes for the real piCOBOT Electric tool stack and payloads.

Everything is authored in metres.  The meshes are generated once and committed
under ``assets/tool`` and ``assets/objects``; this module is only needed to
regenerate them::

    .venv/bin/python -m mujoco_sim.meshgen

The tool shapes follow the official Piab dimensions:

- piCOBOT Electric housing: 120 x 124.4 x 110.2 mm body envelope
  (drawing 9. / technical data 120 x 110.2 x 139.4 mm incl. connectors)
- B30-2 silicone bellows cup: Ø34 mm bellows, Ø36 mm flange, 26.2 mm high
- custom yellow collar: Ø39 x 9 mm (as installed on the real robot)

Payload meshes are render-only: their collision proxies live in
``config.OBJ_SPECS`` and stay primitive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import ASSETS_DIR

TOOL_MESH_DIR = ASSETS_DIR / "tool"
OBJECT_MESH_DIR = ASSETS_DIR / "objects"


def _write_obj(path: Path, vertices: np.ndarray, faces: list[tuple[int, int, int]], name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {name} (metres)", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def _finish(path: Path) -> None:
    """Re-load and fix normals if trimesh is available (generation only)."""
    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh")
        # fix_normals needs scipy in some trimesh versions; the explicitly
        # wound OBJ is still valid without it
        try:
            mesh.fix_normals()
        except Exception:
            pass
        try:
            if mesh.volume < 0:
                mesh.invert()
        except Exception:
            pass
        mesh.export(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# primitives written as OBJ
# ---------------------------------------------------------------------------


def rounded_rect_outline(hx: float, hy: float, radius: float, segments: int = 8):
    """CCW outline of a rounded rectangle in the XY plane."""
    radius = min(radius, hx, hy)
    cx, cy = hx - radius, hy - radius
    corners = (
        (cx, cy, 0.0),
        (-cx, cy, 90.0),
        (-cx, -cy, 180.0),
        (cx, -cy, 270.0),
    )
    points = []
    for x0, y0, start in corners:
        for i in range(segments + 1):
            a = math.radians(start + 90.0 * i / segments)
            points.append((x0 + radius * math.cos(a), y0 + radius * math.sin(a)))
    # drop duplicated corner point at the end
    return points[:-1]


def extrude_outline(
    path: Path, outline: list[tuple[float, float]], height: float, name: str
) -> None:
    """Extrude a convex CCW outline from z=0 to z=height, with caps."""
    n = len(outline)
    verts = [(x, y, 0.0) for x, y in outline]
    verts += [(x, y, height) for x, y in outline]
    cb = len(verts)
    verts.append((0.0, 0.0, 0.0))
    ct = len(verts)
    verts.append((0.0, 0.0, height))
    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((cb, j, i))                       # bottom (normal -z)
        faces.append((ct, n + i, n + j))               # top (normal +z)
        faces.append((i, j, n + j))                    # side
        faces.append((i, n + j, n + i))
    _write_obj(path, np.asarray(verts), faces, name)
    _finish(path)


def hex_prism(path: Path, across_flats: float, height: float, name: str) -> None:
    r = 0.5 * across_flats / math.cos(math.radians(30.0))
    outline = [
        (r * math.cos(math.radians(30 + 60 * i)),
         r * math.sin(math.radians(30 + 60 * i)))
        for i in range(6)
    ]
    extrude_outline(path, outline, height, name)


def annulus(path: Path, r_in: float, r_out: float, height: float, segments: int = 64, name: str = "annulus") -> None:
    """Closed ring (pipe) solid from z=0 to z=height."""
    rings = []
    verts = []
    for r, z in ((r_in, 0.0), (r_out, 0.0), (r_out, height), (r_in, height)):
        start = len(verts)
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(list(range(start, start + segments)))
    faces: list[tuple[int, int, int]] = []
    for k in range(4):
        a = rings[k]
        b = rings[(k + 1) % 4]
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((a[i], a[j], b[j]))
            faces.append((a[i], b[j], b[i]))
    _write_obj(path, np.asarray(verts), faces, name)
    _finish(path)


def revolve_profile(
    path: Path, profile: list[tuple[float, float]], segments: int = 72, name: str = "revolve"
) -> None:
    """Revolve an (r, z) polyline around z.

    ``r == 0`` points become single poles; endpoints at r=0 close the solid.
    """
    rings: list[list[int]] = []
    verts: list[tuple[float, float, float]] = []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            start = len(verts)
            for i in range(segments):
                a = 2.0 * math.pi * i / segments
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(list(range(start, start + segments)))
    faces: list[tuple[int, int, int]] = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        if len(a) == 1 and len(b) > 1:
            for i in range(segments):
                faces.append((a[0], b[i], b[(i + 1) % segments]))
        elif len(a) > 1 and len(b) == 1:
            for i in range(segments):
                faces.append((a[i], b[0], a[(i + 1) % segments]))
        elif len(a) > 1 and len(b) > 1:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append((a[i], a[j], b[j]))
                faces.append((a[i], b[j], b[i]))
    _write_obj(path, np.asarray(verts), faces, name)
    _finish(path)


# ---------------------------------------------------------------------------
# the actual tool parts
# ---------------------------------------------------------------------------


def make_picobot_body() -> Path:
    """Rounded piCOBOT Electric housing, origin at its top (robot) face."""
    path = TOOL_MESH_DIR / "picobot_body.obj"
    outline = rounded_rect_outline(0.060, 0.0622, 0.012, segments=8)
    extrude_outline(path, outline, 0.1102, "picobot_body")
    return path


def make_b30_cup() -> Path:
    """B30-2 silicone bellows cup (solid outer silhouette), origin at its top."""
    # z measured from the fitting top (0.349) down to the lip (0.370):
    # profile travels from the top pole, down the outside, around the lip,
    # and back up to the bottom pole.
    profile = [
        (0.0, 0.0),          # top centre (under the fitting)
        (0.0095, 0.0),       # fitting shoulder
        (0.0095, 0.008),
        (0.0120, 0.010),
        (0.0180, 0.0125),    # Ø36 top flange
        (0.0180, 0.0150),
        (0.0140, 0.0175),    # bellows shoulder
        (0.0125, 0.0200),    # waist Ø25
        (0.0155, 0.0225),
        (0.0170, 0.0240),    # lip roll Ø34
        (0.0170, 0.0262),    # lip plane (tip)
        (0.0, 0.0262),       # bottom centre (closed mouth visual)
    ]
    revolve_profile(TOOL_MESH_DIR / "b30_cup.obj", profile, segments=72, name="b30_cup")
    return TOOL_MESH_DIR / "b30_cup.obj"


def make_collar() -> Path:
    path = TOOL_MESH_DIR / "yellow_collar.obj"
    annulus(path, r_in=0.0170, r_out=0.0195, height=0.009, segments=64,
            name="yellow_collar")
    return path


def make_hex_nut() -> Path:
    path = TOOL_MESH_DIR / "hex_nut.obj"
    hex_prism(path, across_flats=0.019, height=0.009, name="hex_nut")
    return path


# ---------------------------------------------------------------------------
# realistic payload meshes
# ---------------------------------------------------------------------------


def _rounded_box(path: Path, sx: float, sy: float, sz: float, radius: float,
                 name: str) -> Path:
    outline = rounded_rect_outline(sx * 0.5, sy * 0.5, radius, segments=8)
    extrude_outline(path, outline, sz, name)
    return path


def _lathe(path: Path, profile, name: str, segments: int = 80) -> Path:
    revolve_profile(path, profile, segments=segments, name=name)
    return path


def _loft(
    path: Path,
    rings: list[tuple[list[tuple[float, float]], float]],
    name: str,
    cap_bottom: bool = True,
    cap_top: bool = True,
) -> Path:
    """Loft equal-length 2-D outlines at increasing heights, with caps."""
    n = len(rings[0][0])
    verts: list[tuple[float, float, float]] = []
    ids: list[list[int]] = []
    for outline, z in rings:
        if len(outline) != n:
            raise ValueError("all loft outlines must have the same length")
        ring = []
        for x, y in outline:
            ring.append(len(verts))
            verts.append((x, y, z))
        ids.append(ring)
    faces: list[tuple[int, int, int]] = []
    for k in range(len(ids) - 1):
        a, b = ids[k], ids[k + 1]
        for i in range(n):
            j = (i + 1) % n
            faces.append((a[i], a[j], b[j]))
            faces.append((a[i], b[j], b[i]))
    if cap_bottom:
        c = len(verts)
        verts.append((0.0, 0.0, rings[0][1]))
        a = ids[0]
        for i in range(n):
            faces.append((c, a[(i + 1) % n], a[i]))
    if cap_top:
        c = len(verts)
        verts.append((0.0, 0.0, rings[-1][1]))
        b = ids[-1]
        for i in range(n):
            faces.append((c, b[i], b[(i + 1) % n]))
    _write_obj(path, np.asarray(verts), faces, name)
    _finish(path)
    return path


def _hollow_bowl(path: Path, name: str) -> Path:
    """Rounded-square ceramic bowl, open top, ~6 mm wall (visual only)."""
    outline = lambda hx, r: rounded_rect_outline(hx, hx, r, segments=10)
    rings = [
        (outline(0.060, 0.018), 0.000),   # outer bottom edge
        (outline(0.064, 0.021), 0.007),   # outer foot
        (outline(0.065, 0.022), 0.048),   # outer wall / rim
        (outline(0.065, 0.022), 0.052),   # flat rim top
        (outline(0.060, 0.020), 0.052),   # rim inner edge
        (outline(0.059, 0.0195), 0.048),  # inner wall top
        (outline(0.052, 0.017), 0.012),   # inner wall bottom
        (outline(0.050, 0.016), 0.008),   # inner floor edge
    ]
    return _loft(path, rings, name, cap_bottom=True, cap_top=True)


def _torus_handle(path: Path, ring_radius: float = 0.030,
                  tube_radius: float = 0.006, segments: int = 40,
                  tube_segments: int = 12) -> Path:
    """Half-torus handle in the local x-z plane, centered at the origin."""
    verts = []
    rings = []
    for i in range(segments + 1):
        phi = math.pi * 0.5 + math.pi * i / segments
        px, pz = ring_radius * math.cos(phi), ring_radius * math.sin(phi)
        nx, nz = math.cos(phi), math.sin(phi)
        ids = []
        for j in range(tube_segments):
            psi = 2.0 * math.pi * j / tube_segments
            x = px + tube_radius * math.cos(psi) * nx
            y = tube_radius * math.sin(psi)
            z = pz + tube_radius * math.cos(psi) * nz
            ids.append(len(verts))
            verts.append((x, y, z))
        rings.append(ids)
    faces = []
    for k in range(segments):
        a, b = rings[k], rings[k + 1]
        for j in range(tube_segments):
            jn = (j + 1) % tube_segments
            faces.append((a[j], a[jn], b[jn]))
            faces.append((a[j], b[jn], b[j]))
    for ring, center in ((rings[0], (0.0, 0.0, ring_radius)),
                         (rings[-1], (0.0, 0.0, -ring_radius))):
        c = len(verts)
        verts.append(center)
        for j in range(tube_segments):
            jn = (j + 1) % tube_segments
            faces.append((ring[j], ring[jn], c))
    _write_obj(path, np.asarray(verts), faces, path.stem)
    _finish(path)
    return path


def make_object_meshes() -> dict[str, Path]:
    """Generate the visual payload meshes (metres, object-local frames)."""
    OBJECT_MESH_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    out["white_foam"] = _rounded_box(
        OBJECT_MESH_DIR / "white_foam.obj", 0.150, 0.150, 0.020, 0.008,
        "white_foam")
    out["red_foam"] = _rounded_box(
        OBJECT_MESH_DIR / "red_foam.obj", 0.140, 0.140, 0.020, 0.008,
        "red_foam")
    out["magazine"] = _rounded_box(
        OBJECT_MESH_DIR / "magazine.obj", 0.210, 0.280, 0.008, 0.003,
        "magazine")
    out["magazine_cover"] = _rounded_box(
        OBJECT_MESH_DIR / "magazine_cover.obj", 0.194, 0.264, 0.0015, 0.004,
        "magazine_cover")
    out["magazine_band"] = _rounded_box(
        OBJECT_MESH_DIR / "magazine_band.obj", 0.040, 0.264, 0.0008, 0.002,
        "magazine_band")
    out["test_cube"] = _rounded_box(
        OBJECT_MESH_DIR / "test_cube.obj", 0.050, 0.050, 0.050, 0.004,
        "test_cube")
    out["heavy_cube"] = _rounded_box(
        OBJECT_MESH_DIR / "heavy_cube.obj", 0.060, 0.060, 0.060, 0.005,
        "heavy_cube")

    out["white_bowl"] = _hollow_bowl(
        OBJECT_MESH_DIR / "white_bowl.obj", "white_bowl")

    mug_profile = [
        (0.0, 0.000), (0.042, 0.000), (0.042, 0.088), (0.040, 0.090),
        (0.038, 0.090), (0.038, 0.008), (0.034, 0.006), (0.0, 0.006),
    ]
    out["mug_body"] = _lathe(
        OBJECT_MESH_DIR / "mug_body.obj", mug_profile, "mug_body")
    out["mug_handle"] = _torus_handle(
        OBJECT_MESH_DIR / "mug_handle.obj")

    bottle_profile = [
        (0.0, 0.0), (0.033, 0.0), (0.033, 0.128), (0.031, 0.142),
        (0.026, 0.160), (0.020, 0.180), (0.016, 0.196), (0.016, 0.208),
        (0.0, 0.208),
    ]
    out["bottle"] = _lathe(
        OBJECT_MESH_DIR / "bottle.obj", bottle_profile, "bottle")
    bottle_label_profile = [
        (0.0330, 0.0), (0.0334, 0.0), (0.0334, 0.050), (0.0330, 0.050),
        (0.0330, 0.0),
    ]
    out["bottle_label"] = _lathe(
        OBJECT_MESH_DIR / "bottle_label.obj", bottle_label_profile,
        "bottle_label")
    cap_profile = [
        (0.0, 0.000), (0.019, 0.000), (0.019, 0.003), (0.0185, 0.004),
        (0.0185, 0.016), (0.019, 0.017), (0.019, 0.018), (0.0, 0.018),
    ]
    out["bottle_cap"] = _lathe(
        OBJECT_MESH_DIR / "bottle_cap.obj", cap_profile, "bottle_cap")

    can_profile = [
        (0.0, 0.0), (0.033, 0.0), (0.033, 0.108), (0.032, 0.111),
        (0.029, 0.113), (0.0, 0.113),
    ]
    out["soda_can"] = _lathe(
        OBJECT_MESH_DIR / "soda_can.obj", can_profile, "soda_can")
    can_lid_profile = [
        (0.0, 0.0), (0.030, 0.0), (0.030, 0.0012), (0.0285, 0.002),
        (0.0, 0.002),
    ]
    out["can_lid"] = _lathe(
        OBJECT_MESH_DIR / "can_lid.obj", can_lid_profile, "can_lid")

    blade_profile = [
        (0.0, 0.0), (0.080, 0.002), (0.080, 0.013), (0.078, 0.015),
        (0.0, 0.015),
    ]
    out["racket_blade"] = _lathe(
        OBJECT_MESH_DIR / "racket_blade.obj", blade_profile, "racket_blade")
    out["racket_handle"] = _rounded_box(
        OBJECT_MESH_DIR / "racket_handle.obj", 0.024, 0.110, 0.012, 0.006,
        "racket_handle")
    rubber_profile = [(0.0, 0.0), (0.078, 0.0), (0.078, 0.0015), (0.0, 0.0015)]
    out["racket_rubber"] = _lathe(
        OBJECT_MESH_DIR / "racket_rubber.obj", rubber_profile, "racket_rubber")

    rocket_profile = [
        (0.0, 0.000), (0.028, 0.000), (0.028, 0.240), (0.0, 0.240),
    ]
    out["rocket_body"] = _lathe(
        OBJECT_MESH_DIR / "rocket_body.obj", rocket_profile, "rocket_body")
    nose_profile = [(0.0, 0.0), (0.028, 0.0), (0.0, 0.060)]
    out["rocket_nose"] = _lathe(
        OBJECT_MESH_DIR / "rocket_nose.obj", nose_profile, "rocket_nose")
    return out


def build_all() -> dict[str, Path]:
    object_meshes = make_object_meshes()
    return {
        **object_meshes,
        "picobot_body": make_picobot_body(),
        "b30_cup": make_b30_cup(),
        "yellow_collar": make_collar(),
        "hex_nut": make_hex_nut(),
    }


if __name__ == "__main__":
    for name, path in build_all().items():
        print(f"{name}: {path}")
