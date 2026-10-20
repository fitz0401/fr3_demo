"""Force input of the pi05_bc2f_* checkpoints, computed from the raw ~1 kHz arm stream (pure numpy).

Training (openpi branch fr3/drop-v5, scripts/force_sidecar.py + policies/force_table.py) turns every raw arm sample
(controller time, q, tau_J) into

    tau_ext = tau_J - g(q)              g: FR3 + Franka Hand gravity model, per-robot first-moment correction folded in
    wrench  = -pinv(J^T) tau_ext        flange (fr3_link8) wrench in base axes, force BY the environment ON the robot

and, for a frame with controller time t, averages the wrench over 16 equal bins of (t - 64 ms, t] (a bin without a
raw sample takes the linear interpolation at its centre), then pools adjacent bins down to 4. The model sees those
4 x 6 = 24 numbers, oldest bin first. This module reproduces that exactly; the robot-specific model (joint
placements, masses and first moments after the correction) comes from `force_model.json`, exported on Leonardo by
openpi `scripts/export_force_model.py` from the same URDF and correction the training sidecar used, and checked
there against pinocchio and against the training table.

Raw payload layout (what the runner sends as `observation/raw_force`): float64 [N, 15] =
    [controller time (s, bamboo `time_sec`), q (7, rad), tau_J (7, Nm)]
covering at least the 64 ms before `observation/robot_timestamp` (the controller time of the state sent as the
current joint position). Samples after that time may be included; they only matter where a bin is empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WINDOW_S = 0.064
NATIVE_BINS = 16
BINS = 4
RAW_COLUMNS = 15
# The window is rejected when fewer bins than this hold a raw sample (training: >99 % of bins were filled).
MIN_VALID_FRACTION = 0.75
# ... or when the newest raw sample is older than the observation time by more than this.
MAX_STALENESS_S = 0.010


def bin_means(t_raw: np.ndarray, x_raw: np.ndarray, t_end: np.ndarray, window_s: float, bins: int):
    """Copy of openpi `policies/force_window.bin_means`: mean of x_raw over `bins` equal half-open bins of
    (t_end - window_s, t_end]; empty bins fall back to linear interpolation at their centre."""
    t_raw = np.asarray(t_raw, np.float64)
    x_raw = np.asarray(x_raw, np.float64)
    t_end = np.asarray(t_end, np.float64)
    if t_raw.ndim != 1 or x_raw.shape[0] != t_raw.shape[0]:
        raise ValueError(f"t_raw {t_raw.shape} and x_raw {x_raw.shape} disagree")
    if np.any(np.diff(t_raw) <= 0):
        raise ValueError("raw timestamps must be strictly increasing")
    width = window_s / bins
    edges = t_end[:, None] - window_s + width * np.arange(bins + 1)[None, :]
    idx = np.searchsorted(t_raw, edges, side="right")
    csum = np.concatenate([np.zeros((1, x_raw.shape[1])), np.cumsum(x_raw, axis=0)], axis=0)
    counts = idx[:, 1:] - idx[:, :-1]
    sums = csum[idx[:, 1:]] - csum[idx[:, :-1]]
    valid = counts > 0
    means = np.where(valid[..., None], sums / np.maximum(counts, 1)[..., None], 0.0)
    if not valid.all():
        centres = 0.5 * (edges[:, 1:] + edges[:, :-1])
        miss = ~valid
        for d in range(x_raw.shape[1]):
            means[..., d][miss] = np.interp(centres[miss], t_raw, x_raw[:, d])
    return means.astype(np.float32), valid


def pool_bins(window: np.ndarray, bins: int) -> np.ndarray:
    """Copy of openpi `policies/force_window.pool_bins`: average adjacent bins of [..., B, D] down to [..., bins, D]."""
    b = window.shape[-2]
    if b % bins:
        raise ValueError(f"cannot pool {b} bins into {bins}")
    return window.reshape(*window.shape[:-2], bins, b // bins, window.shape[-1]).mean(axis=-2)


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues: [K] angles about one unit axis -> [K, 3, 3]."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3)[None] + s * k[None] + (1.0 - c) * (k @ k)[None]


class ForceModel:
    """Serial-chain gravity torque and flange Jacobian of the FR3 + hand, as pinocchio computes them.

    force_model.json: {"gravity": [0, 0, -9.81], "rcond": 1e-3,
                       "joints": [{"placement": 4x4 (joint frame relative to the parent joint frame at q = 0),
                                   "axis": [x, y, z] (revolute axis in the joint frame),
                                   "mass": kg, "first_moment": [m*c] (kg m, in the joint frame)}, ...] (7),
                       "flange": 4x4 (fr3_link8 relative to joint 7)}
    """

    def __init__(self, spec: dict):
        joints = spec["joints"]
        if len(joints) != 7:
            raise ValueError(f"expected 7 joints, got {len(joints)}")
        self.placements = np.asarray([j["placement"] for j in joints], np.float64)
        self.axes = np.asarray([j["axis"] for j in joints], np.float64)
        self.masses = np.asarray([j["mass"] for j in joints], np.float64)
        self.first_moments = np.asarray([j["first_moment"] for j in joints], np.float64)
        self.flange = np.asarray(spec["flange"], np.float64)
        self.gravity = np.asarray(spec.get("gravity", [0.0, 0.0, -9.81]), np.float64)
        self.rcond = float(spec.get("rcond", 1e-3))
        self.spec = spec

    @classmethod
    def from_json(cls, path: str | Path) -> ForceModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def kinematics(self, q: np.ndarray):
        """World rotation [K, 7, 3, 3] and origin [K, 7, 3] of every joint frame, flange origin [K, 3]."""
        q = np.asarray(q, np.float64).reshape(-1, 7)
        n = len(q)
        rot = np.empty((n, 7, 3, 3))
        pos = np.empty((n, 7, 3))
        r_prev, p_prev = np.broadcast_to(np.eye(3), (n, 3, 3)), np.zeros((n, 3))
        for i in range(7):
            r_fix, p_fix = self.placements[i, :3, :3], self.placements[i, :3, 3]
            p_i = p_prev + r_prev @ p_fix
            r_i = (r_prev @ r_fix) @ _axis_rotations(self.axes[i], q[:, i])
            rot[:, i], pos[:, i] = r_i, p_i
            r_prev, p_prev = r_i, p_i
        flange = pos[:, 6] + rot[:, 6] @ self.flange[:3, 3]
        return rot, pos, flange

    def gravity_and_jacobian(self, q: np.ndarray):
        """g(q) [K, 7] (torque that holds the arm against gravity) and the flange Jacobian [K, 6, 7]
        (linear rows first, base axes, reference point at the flange origin = pinocchio LOCAL_WORLD_ALIGNED)."""
        rot, pos, flange = self.kinematics(q)
        axes = np.einsum("kiab,ib->kia", rot, self.axes)  # world joint axes [K, 7, 3]
        support = -self.gravity  # force that holds a unit mass against gravity
        # Body k: m_k (p_k - p_i) + R_k h_k is its first moment about joint i's origin.
        moment_k = self.masses[None, :, None] * pos + np.einsum("kiab,ib->kia", rot, self.first_moments)
        grav = np.empty(pos.shape[:2])
        for i in range(7):
            first = moment_k[:, i:].sum(axis=1) - self.masses[i:].sum() * pos[:, i]
            grav[:, i] = np.einsum("ka,ka->k", axes[:, i], np.cross(first, support[None]))
        jac = np.empty((len(pos), 6, 7))
        jac[:, :3] = np.cross(axes, flange[:, None, :] - pos).transpose(0, 2, 1)
        jac[:, 3:] = axes.transpose(0, 2, 1)
        return grav, jac

    def compensate(self, q: np.ndarray, tau: np.ndarray):
        """tau_ext [K, 7] and the flange wrench [K, 6] (fx fy fz mx my mz) for raw samples q, tau_J."""
        grav, jac = self.gravity_and_jacobian(q)
        tau_ext = np.asarray(tau, np.float64).reshape(-1, 7) - grav
        wrench = -np.einsum("kij,kj->ki", np.linalg.pinv(np.transpose(jac, (0, 2, 1)), rcond=self.rcond), tau_ext)
        return tau_ext, wrench


def clean_raw(raw: np.ndarray) -> np.ndarray:
    """Validate the [N, 15] payload, sort by controller time and drop repeated controller states (the poller can
    read the same state twice), as the training sidecar did."""
    raw = np.asarray(raw, np.float64)
    if raw.ndim != 2 or raw.shape[1] != RAW_COLUMNS:
        raise ValueError(f"raw force must be [N, {RAW_COLUMNS}] = [time, q(7), tau_J(7)], got {raw.shape}")
    if len(raw) < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("raw force needs at least two finite samples")
    raw = raw[np.argsort(raw[:, 0], kind="stable")]
    keep = np.r_[True, np.diff(raw[:, 0]) > 0]
    return raw[keep]


def window_features(model: ForceModel, raw: np.ndarray, t_end: float) -> dict:
    """The 24 force inputs for one observation, plus diagnostics.

    Returns {"force": [24] float32 (4 bins x fx fy fz mx my mz, oldest first), "window": [4, 6], "valid_fraction",
    "samples_in_window", "staleness_s"}. Raises ValueError when the payload does not cover the window."""
    raw = clean_raw(raw)
    t, q, tau = raw[:, 0], raw[:, 1:8], raw[:, 8:15]
    t_end = float(t_end)
    if t[0] > t_end - WINDOW_S:
        raise ValueError(f"raw force starts {1000 * (t[0] - (t_end - WINDOW_S)):.1f} ms after the window start")
    staleness = t_end - t[t <= t_end].max() if np.any(t <= t_end) else np.inf
    if staleness > MAX_STALENESS_S:
        raise ValueError(f"newest raw sample is {1000 * staleness:.1f} ms older than robot_timestamp")
    _, wrench = model.compensate(q, tau)
    means, valid = bin_means(t, wrench, np.array([t_end]), WINDOW_S, NATIVE_BINS)
    fraction = float(valid.mean())
    if fraction < MIN_VALID_FRACTION:
        raise ValueError(f"only {fraction:.0%} of the {NATIVE_BINS} force bins hold a raw sample")
    window = pool_bins(means, BINS)[0]
    in_window = int(np.sum((t > t_end - WINDOW_S) & (t <= t_end)))
    return {"force": window.reshape(-1).astype(np.float32), "window": window, "valid_fraction": fraction,
            "samples_in_window": in_window, "staleness_s": float(staleness)}
