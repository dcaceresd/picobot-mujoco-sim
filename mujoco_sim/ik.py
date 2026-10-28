"""Minimal 6-DoF damped-least-squares IK for the cup tip.

The real pipeline plans with MoveIt; in simulation we need the same Cartesian
waypoints (approach above the object, descend, lift, ...) without a ROS stack.
This solver is deterministic, uses MuJoCo's site Jacobians, and only touches
the six arm joints - object free joints stay untouched.
"""

from __future__ import annotations

import numpy as np

from .model import ARM_JOINT_NAMES


class IkError(RuntimeError):
    """Raised when a Cartesian target is not reachable."""


def _joint_indices(model, joint_names):
    import mujoco

    qadr, dadr, jids = [], [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not in model")
        jids.append(jid)
        qadr.append(int(model.jnt_qposadr[jid]))
        dadr.append(int(model.jnt_dofadr[jid]))
    return np.array(qadr), np.array(dadr), jids


def solve_ik(
    model,
    data,
    site_name: str,
    target_pos,
    target_R=None,
    *,
    seed: np.ndarray | None = None,
    joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    max_iters: int = 200,
    tol_pos_m: float = 5e-4,
    tol_rot_rad: float = 5e-3,
    damping: float = 1e-4,
    max_step_rad: float = 0.35,
    rot_weight: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Solve IK for ``site_name`` to reach ``target_pos`` (and optionally R).

    Returns ``(q_arm, pos_err_m, rot_err_rad)``.  Mutates ``data.qpos`` while
    iterating; the caller decides whether to keep or retarget the state.
    """
    import mujoco

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"site {site_name!r} not in model")
    qadr, dadr, jids = _joint_indices(model, joint_names)

    q = (
        np.asarray(seed, dtype=float).copy()
        if seed is not None
        else data.qpos[qadr].copy()
    )
    target_pos = np.asarray(target_pos, dtype=float)
    target_R = None if target_R is None else np.asarray(target_R, dtype=float)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    dof_cols = dadr

    def kinematics():
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    def errors() -> tuple[np.ndarray, float, float]:
        pos = data.site_xpos[sid]
        e_pos = target_pos - pos
        if target_R is None:
            return e_pos, float(np.linalg.norm(e_pos)), 0.0
        Rc = data.site_xmat[sid].reshape(3, 3)
        Rerr = target_R @ Rc.T
        e_rot = 0.5 * np.array(
            [Rerr[2, 1] - Rerr[1, 2],
             Rerr[0, 2] - Rerr[2, 0],
             Rerr[1, 0] - Rerr[0, 1]]
        )
        return (
            np.concatenate([e_pos, rot_weight * e_rot]),
            float(np.linalg.norm(e_pos)),
            float(np.linalg.norm(e_rot)),
        )

    best_q = q.copy()
    best_score = np.inf
    for _ in range(max_iters):
        kinematics()
        e, pos_err, rot_err = errors()
        score = float(np.linalg.norm(e))
        if score < best_score:
            best_score = score
            best_q = q.copy()
        if pos_err < tol_pos_m and (target_R is None or rot_err < tol_rot_rad):
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, dof_cols], rot_weight * jacr[:, dof_cols]])
        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        A = J @ J.T + (damping**2) * np.eye(6)
        try:
            dq = J.T @ np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            dq = J.T @ np.linalg.lstsq(A, e, rcond=None)[0]
        step = float(np.max(np.abs(dq)))
        if step > max_step_rad:
            dq *= max_step_rad / step
        q = q + dq
        for k, jid in enumerate(jids):
            lo, hi = model.jnt_range[jid]
            q[k] = float(np.clip(q[k], lo, hi))

    # leave the model at the best solution found
    q = best_q
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    _, pos_err, rot_err = errors()
    return q, pos_err, rot_err


def solve_ik_multiseed(
    model,
    data,
    site_name: str,
    target_pos,
    target_R=None,
    *,
    n_seeds: int = 12,
    seed_value: int = 0,
    seed: np.ndarray | None = None,
    joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    tol_pos_m: float = 5e-4,
    tol_rot_rad: float = 5e-3,
):
    """Try the current config first, then random restarts; return the best."""
    q0, pe, re = solve_ik(
        model, data, site_name, target_pos, target_R, seed=seed,
        joint_names=joint_names, tol_pos_m=tol_pos_m, tol_rot_rad=tol_rot_rad,
    )
    best = (q0, pe, re)
    if pe < 5.0 * tol_pos_m and (target_R is None or re < 5.0 * tol_rot_rad):
        return best

    rng = np.random.default_rng(seed_value)
    _, _, jids = _joint_indices(model, joint_names)
    ranges = np.array([model.jnt_range[j] for j in jids])
    current = data.qpos[_joint_indices(model, joint_names)[0]].copy()
    for _ in range(n_seeds):
        trial = current + rng.normal(0.0, 1.0, size=len(jids))
        trial = np.clip(trial, ranges[:, 0], ranges[:, 1])
        q, pe, re = solve_ik(
            model, data, site_name, target_pos, target_R, seed=trial,
            joint_names=joint_names, tol_pos_m=tol_pos_m,
            tol_rot_rad=tol_rot_rad,
        )
        if pe < best[1] and (target_R is None or re < best[2]):
            best = (q, pe, re)
        if pe < tol_pos_m and (target_R is None or re < tol_rot_rad):
            break
    return best
