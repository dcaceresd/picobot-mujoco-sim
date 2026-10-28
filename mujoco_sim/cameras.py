"""Calibrated RGB-D rendering for the simulation.

The simulation exposes two cameras that mirror the real cell:

- ``external`` - fixed RGB-D camera looking at the workspace
- ``wrist``    - camera mounted on the piCOBOT body, looking down the tool axis

Both use the Orbbec intrinsics from ``intrinsics.json`` when available
(``SimConfig.intrinsics_path``).  MuJoCo cameras look along **-z** (+x right,
+y up); back-projection below converts to the usual optical convention
(+z forward, +y down).

Rendering needs a GL context.  On a workstation this just works; on a headless
box export ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    #: defaults match the Orbbec 1280x720 stream recorded on the real cell
    #: (perception/depth_intrinsics.json and workspace/intrinsics.json)
    width: int = 1280
    height: int = 720
    fx: float = 752.209716796875
    fy: float = 751.9856567382812
    cx: float = 644.6649169921875
    cy: float = 365.3125915527344

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
        }


# ---------------------------------------------------------------------------
# intrinsics file loading
# ---------------------------------------------------------------------------


def _find_camera_matrix(obj) -> dict | None:
    """Recursively search an arbitrary intrinsics JSON for camera parameters."""
    if isinstance(obj, dict):
        if "fx" in obj and "fy" in obj:
            return obj
        if "K" in obj and isinstance(obj["K"], list) and len(obj["K"]) == 3:
            K = obj["K"]
            merged = {
                "fx": K[0][0], "fy": K[1][1],
                "cx": K[0][2], "cy": K[1][2],
            }
            for key in ("width", "height", "image_width", "image_height"):
                if key in obj:
                    merged[key] = obj[key]
            if "width" not in merged and "image_width" in merged:
                merged["width"] = merged["image_width"]
            if "height" not in merged and "image_height" in merged:
                merged["height"] = merged["image_height"]
            return merged
        if "camera_matrix" in obj and isinstance(obj["camera_matrix"], list):
            K = obj["camera_matrix"]
            if len(K) == 3:
                merged = {
                    "fx": K[0][0], "fy": K[1][1],
                    "cx": K[0][2], "cy": K[1][2],
                }
                if "image_width" in obj:
                    merged["width"] = obj["image_width"]
                if "image_height" in obj:
                    merged["height"] = obj["image_height"]
                return merged
        for value in obj.values():
            found = _find_camera_matrix(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_camera_matrix(value)
            if found is not None:
                return found
    return None


def intrinsics_from_file(path: str | Path | None) -> Intrinsics | None:
    """Best-effort loader for the various intrinsics files in the workspace."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        warnings.warn(f"intrinsics file not found: {path}")
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"could not parse intrinsics {path}: {exc}")
        return None
    found = _find_camera_matrix(data)
    if found is None:
        warnings.warn(f"no camera matrix found in {path}")
        return None
    return Intrinsics(
        width=int(found.get("width", found.get("image_width", 640))),
        height=int(found.get("height", found.get("image_height", 480))),
        fx=float(found["fx"]),
        fy=float(found["fy"]),
        cx=float(found.get("cx", found.get("width", 640) / 2)),
        cy=float(found.get("cy", found.get("height", 480) / 2)),
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


class CameraSuite:
    """Lazily created MuJoCo renderers for the scene cameras."""

    def __init__(self, model, data, intrinsics: Intrinsics | None = None):
        self.model = model
        self.data = data
        self.intrinsics = intrinsics or Intrinsics()
        self._renderers: dict[str, object] = {}
        self._cam_ids: dict[str, int] = {}
        import mujoco

        for name in ("external", "wrist"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid >= 0:
                self._cam_ids[name] = cid
        if not self._cam_ids:
            raise RuntimeError("scene has no 'external'/'wrist' camera")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._cam_ids)

    def _renderer(self, name: str):
        if name not in self._cam_ids:
            raise KeyError(f"unknown camera {name!r}; have {self.names}")
        if name not in self._renderers:
            import mujoco

            self._renderers[name] = mujoco.Renderer(
                self.model,
                height=self.intrinsics.height,
                width=self.intrinsics.width,
            )
        return self._renderers[name]

    def _update(self, name: str):
        renderer = self._renderer(name)
        renderer.update_scene(self.data, camera=self._cam_ids[name])
        return renderer

    def render_rgb(self, name: str = "external") -> np.ndarray:
        """RGB uint8 image, shape (H, W, 3)."""
        return self._update(name).render()

    def render_depth(self, name: str = "external") -> np.ndarray:
        """Depth in metres along the optical axis, shape (H, W)."""
        renderer = self._update(name)
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render()
        finally:
            renderer.disable_depth_rendering()
        return depth

    def render_rgbd(self, name: str = "external") -> tuple[np.ndarray, np.ndarray]:
        rgb = self.render_rgb(name)
        depth = self.render_depth(name)
        return rgb, depth

    def camera_pose(self, name: str = "external") -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix of the camera frame."""
        cid = self._cam_ids[name]
        pos = self.data.cam_xpos[cid].copy()
        R = self.data.cam_xmat[cid].reshape(3, 3).copy()
        return pos, R

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        name: str = "external",
        *,
        max_depth_m: float = 5.0,
        world: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Back-project a depth image; returns (points Nx3, colors Nx3 if rgb).

        Uses the pinhole model with the suite intrinsics.  Points are in world
        coordinates when ``world=True``.
        """
        intr = self.intrinsics
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float64)
        mask = (z > 1e-4) & (z < max_depth_m) & np.isfinite(z)

        # MuJoCo camera frame (+x right, +y up, -z forward) -> optical (+z fwd)
        x_mj = (us - intr.cx) / intr.fx * z
        y_mj = -(vs - intr.cy) / intr.fy * z
        z_mj = -z
        pts_cam = np.stack([x_mj[mask], y_mj[mask], z_mj[mask]], axis=1)

        if world:
            pos, R = self.camera_pose(name)
            pts = pts_cam @ R.T + pos
        else:
            pts = pts_cam
        return pts, mask

    def pointcloud_with_colors(
        self, name: str = "external", *, max_depth_m: float = 5.0
    ) -> np.ndarray:
        """Nx6 point cloud (xyz + rgb 0..1) in world coordinates."""
        rgb, depth = self.render_rgbd(name)
        pts, mask = self.depth_to_pointcloud(depth, name, max_depth_m=max_depth_m)
        colors = rgb[mask].astype(np.float64) / 255.0
        return np.concatenate([pts, colors], axis=1)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
