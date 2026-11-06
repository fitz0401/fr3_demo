"""Automatic scoring of a blue-cup pour rollout from what the robot itself measures.

The operator only resets the scene and types the scale readings at ``fr3-pi05-trials record``; whether the
robot grasped, lifted, poured and put the bottle back, and how long each part took, is read from the rollout
recording with fixed rules. Two pieces:

* ``AutoStop`` ends a rollout online (no operator judgment of "done"): ``completed`` once the gripper has been
  open for ``release_s`` after a pour, ``stall`` when the arm has not moved ``stall_rad`` for ``stall_s``,
  ``timeout`` after ``max_duration_s``. The demonstrations end right after the release (they never return
  home), so nothing after it is part of the task.

* ``score_steps`` turns the per-control-step recording (and, when available, the flange force rebuilt from the
  raw ~1 kHz arm stream) into stage events and metrics.

Stages, in order (``completion`` = how many were reached, 0-4). Thresholds come from the 22 blue_cup_2
demonstrations (fr3_train/bc2f/validate_autoscore.py prints them per episode):

    grasp  gripper closed on something, GRASP_MIN <= g < HOLD_BELOW, for 1 s       demos: g 0.780-0.802
    lift   holding, end effector >= grasp height + 3 cm and F_z <= home F_z - 4 N    demos: F_z -6.8..-9.4 N
           for 0.5 s (height only when no force is available)                      (home about -0.5 N)
    pour   PourDetector pouring (tilt >= 60 deg while holding), after the lift     demos: max 87-104 deg
    place  first release after the pour (g >= HOLD_BELOW for 0.5 s) at grasp height demos: -5.1..+0.2 cm,
           -8..+3 cm and <= 12 cm (xy) from where the bottle was grasped             0.4-8.1 cm

A drop is a lifted, non-pouring hold whose F_z comes back to the empty-hand level (> home - 2 N) for 0.5 s.
The poured amount is not readable from the force (held-bottle F_z before vs after the pour correlates 0.23 with
the scale in the demonstrations, ~50 g scatter), so it stays a scale reading.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fr3_demo.kinematics import forward_kinematics

HOLD_BELOW = 0.9  # gripper reading below this = closed (PourDetector.closed_below); 1.0 = open
GRASP_MIN = 0.5  # closed below this = closed on nothing (the bottle stops the fingers at ~0.79)
GRASP_S = 1.0
LIFT_DZ_M = 0.03
LIFT_DFZ_N = -4.0
DROP_DFZ_N = -2.0
SUSTAIN_S = 0.5
PLACE_DZ_M = (-0.08, 0.03)
PLACE_XY_M = 0.12
MOVE_RAD = 0.02  # start of motion: any joint this far from the initial pose
IDLE_RAD_S = 0.01  # arm idle: fastest joint slower than this
FORCE_WINDOW_S = 0.1  # per-step force = mean flange wrench over (t - 0.1 s, t]
HOME_FORCE_S = (0.3, 1.5)  # home baseline = median F_z over this span of the rollout (arm still, hand empty)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FORCE_MODEL = REPO / "fr3_train" / "bc2f" / "force_model.json"


@dataclass
class AutoStop:
    """Decide online when a rollout is over. Call ``update`` once per control step; a non-None return is the
    end reason."""

    release_s: float = 2.0
    stall_s: float = 30.0
    stall_rad: float = MOVE_RAD
    max_duration_s: float = 240.0
    _open_since: float | None = None
    _anchor: tuple[float, np.ndarray] | None = field(default=None, repr=False)

    def update(self, t: float, q: np.ndarray, gripper: float, pour_started: bool) -> str | None:
        q = np.asarray(q, dtype=float)
        if self._anchor is None or np.max(np.abs(q - self._anchor[1])) > self.stall_rad:
            self._anchor = (t, q.copy())
        if pour_started and gripper >= HOLD_BELOW:
            self._open_since = t if self._open_since is None else self._open_since
            if t - self._open_since >= self.release_s:
                return "completed"
        else:
            self._open_since = None
        if t - self._anchor[0] >= self.stall_s:
            return "stall"
        if t >= self.max_duration_s:
            return "timeout"
        return None


def load_force_model(path: str | Path | None = None):
    """fr3_train.force_features.ForceModel from the repository checkout (fr3_train is not an installed package)."""
    module_path = REPO / "fr3_train" / "force_features.py"
    model_path = Path(path) if path else DEFAULT_FORCE_MODEL
    if not module_path.exists() or not model_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("fr3_force_features", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.ForceModel.from_json(model_path)


def step_wrench(raw: np.ndarray, time_sec: np.ndarray, force_model=None) -> np.ndarray | None:
    """Flange wrench [K, 6] (base axes, force BY the environment ON the robot) over FORCE_WINDOW_S before each
    step's controller time, from the raw [N, 15] arm stream. None when raw data or the model is missing.

    q and tau_J are averaged over the window first and compensated once per step (the arm moves <= 0.035 rad in
    0.1 s, so this differs from averaging per-sample wrenches by far less than the newton-level thresholds here,
    and it is ~60x cheaper than compensating every 1 kHz sample)."""
    loaded = force_model or load_force_model()
    if loaded is None or raw is None or len(raw) < 2:
        return None
    module, model = loaded
    raw = module.clean_raw(raw)
    means, _ = module.bin_means(raw[:, 0], raw[:, 1:15], np.asarray(time_sec, float), FORCE_WINDOW_S, 1)
    means = means[:, 0, :].astype(float)
    _, wrench = model.compensate(means[:, :7], means[:, 7:])
    return wrench


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs in a boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def _first_sustained(mask: np.ndarray, t: np.ndarray, seconds: float, start: int = 0) -> int | None:
    """Index where the first True run starting at or after `start` and lasting >= `seconds` begins."""
    for a, b in _runs(mask):
        if b <= start:
            continue
        a = max(a, start)
        if t[b - 1] - t[a] >= seconds - 1e-9:
            return int(a)
    return None


def _r(value: Any, digits: int = 3) -> Any:
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def score_steps(t: np.ndarray, q: np.ndarray, gripper: np.ndarray, pouring: np.ndarray, tilt: np.ndarray,
                wrench: np.ndarray | None = None, eef: np.ndarray | None = None) -> dict[str, Any]:
    """Stage events and metrics of one rollout. ``t`` in seconds from the rollout start, one row per control step;
    ``eef`` [K, 3] defaults to the forward kinematics of ``q``."""
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    g = np.asarray(gripper, float)
    pouring = np.asarray(pouring, bool)
    tilt = np.asarray(tilt, float)
    k = len(t)
    if k < 2 or q.shape != (k, 7) or g.shape != (k,) or pouring.shape != (k,) or tilt.shape != (k,):
        raise ValueError("score_steps needs at least two steps of t, q [K, 7], gripper, pouring and tilt")
    if eef is None:
        eef = np.stack([forward_kinematics(x)[:3, 3] for x in q])
    eef = np.asarray(eef, float)
    z, xy = eef[:, 2], eef[:, :2]
    force = None if wrench is None else np.asarray(wrench, float)[:, :3]
    fz = None if force is None else force[:, 2]
    if force is not None:
        home = (t - t[0] >= HOME_FORCE_S[0]) & (t - t[0] <= HOME_FORCE_S[1]) & (g >= HOLD_BELOW)
        force_home = np.median(force[home] if home.any() else force[: max(2, k // 20)], axis=0)
        fz_home = float(force_home[2])
        dfz = fz - fz_home

    held = g < HOLD_BELOW
    on_object = held & (g >= GRASP_MIN)
    events: dict[str, int | None] = {"grasp": None, "lift": None, "pour": None, "release": None, "drop": None}
    events["grasp"] = _first_sustained(on_object, t, GRASP_S)
    if events["grasp"] is not None:
        gi = events["grasp"]
        lifted = on_object & (z >= z[gi] + LIFT_DZ_M)
        if fz is not None:
            lifted &= dfz <= LIFT_DFZ_N
        events["lift"] = _first_sustained(lifted, t, SUSTAIN_S, gi)
    if events["lift"] is not None:
        li = events["lift"]
        if fz is not None:
            dropped = on_object & (z >= z[events["grasp"]] + LIFT_DZ_M) & ~pouring & (dfz > DROP_DFZ_N)
            events["drop"] = _first_sustained(dropped, t, SUSTAIN_S, li)
        pour_idx = np.flatnonzero(pouring[li:])
        events["pour"] = int(li + pour_idx[0]) if len(pour_idx) else None
    if events["pour"] is not None:
        events["release"] = _first_sustained(~held, t, SUSTAIN_S, events["pour"])

    place = {"dz_cm": None, "xy_cm": None, "ok": False, "fz_after_N": None}
    if events["release"] is not None:
        last_held = max(events["release"] - 1, 0)
        gi = events["grasp"]
        dz, dxy = z[last_held] - z[gi], float(np.linalg.norm(xy[last_held] - xy[gi]))
        place.update(dz_cm=_r(100 * dz, 1), xy_cm=_r(100 * dxy, 1),
                     ok=bool(PLACE_DZ_M[0] <= dz <= PLACE_DZ_M[1] and dxy <= PLACE_XY_M))
        if fz is not None:
            after = (t >= t[events["release"]] + 0.3) & (t <= t[events["release"]] + 1.5)
            place["fz_after_N"] = _r(np.median(dfz[after]), 2) if after.any() else None

    dropped_before_pour = events["drop"] is not None and (events["pour"] is None or events["drop"] < events["pour"])
    reached = [events["grasp"] is not None, events["lift"] is not None,
               events["pour"] is not None and not dropped_before_pour, place["ok"] and events["drop"] is None]
    completion = 0
    for ok in reached:
        if not ok:
            break
        completion += 1

    def at(name: str) -> float | None:
        return None if events[name] is None else _r(t[events[name]] - t[0], 2)

    dt = np.diff(t)
    speed = np.max(np.abs(np.diff(q, axis=0)), axis=1) / np.maximum(dt, 1e-6)
    idle = _runs(speed < IDLE_RAD_S)
    moved = np.flatnonzero(np.max(np.abs(q - q[0]), axis=1) > MOVE_RAD)
    closes = _runs(held)
    first_lift = events["lift"] if events["lift"] is not None else k
    metrics: dict[str, Any] = {
        "completion": completion,
        "stages": dict(zip(("grasp", "lift", "pour", "place"), reached)),
        "t_grasp_s": at("grasp"), "t_lift_s": at("lift"), "t_pour_s": at("pour"), "t_release_s": at("release"),
        "drop": events["drop"] is not None, "t_drop_s": at("drop"),
        "duration_s": _r(t[-1] - t[0], 2),
        "start_delay_s": _r(t[moved[0]] - t[0], 2) if len(moved) else None,
        "longest_idle_s": _r(max((t[b] - t[a] for a, b in idle), default=0.0), 2),
        "path_rad": _r(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)), 2),
        "grasp_attempts": sum(1 for a, _ in closes if a < first_lift),
        "empty_closes": sum(1 for a, b in closes if np.min(g[a:b]) < GRASP_MIN),
        "pour_s": _r(np.sum(dt[pouring[:-1]]), 2),
        "pour_bouts": len(_runs(pouring)),
        "max_tilt_deg": _r(np.max(tilt), 1),
        "place": place,
        "force_available": fz is not None,
    }
    if force is not None:
        held_mass = None
        if events["lift"] is not None:
            lifted_pre = on_object & (z >= z[events["grasp"]] + LIFT_DZ_M) & ~pouring
            lifted_pre[: events["lift"]] = False
            if events["pour"] is not None:
                lifted_pre[events["pour"]:] = False
            if lifted_pre.any():
                held_mass = -1000 * np.median(dfz[lifted_pre]) / 9.81
        metrics.update(
            fz_home_N=_r(fz_home, 2),
            # Held bottle, -dF_z / g before the pour: a check of the grasp against the bottle on the scale, NOT the
            # poured amount.
            held_mass_g=_r(held_mass, 0),
            # Largest upward push on the hand while holding (bottle pressed onto the table or into the cup).
            max_hold_fz_N=_r(np.max(dfz[held]), 1) if held.any() else None,
            # Largest force change from home anywhere in the rollout (collisions).
            max_force_N=_r(np.max(np.linalg.norm(force - force_home, axis=1)), 1),
        )
    return metrics


def score_recording(arrays: dict[str, np.ndarray], force_model=None) -> dict[str, Any]:
    """Score the arrays a RolloutLogger wrote (``steps_*`` and, for bc2f_force, ``raw_force``)."""
    wrench = None
    if "raw_force" in arrays and len(arrays["raw_force"]) > 1 and "steps_time_sec" in arrays:
        wrench = step_wrench(arrays["raw_force"], arrays["steps_time_sec"], force_model)
    return score_steps(arrays["steps_t"], arrays["steps_q"], arrays["steps_gripper"], arrays["steps_pouring"],
                       arrays["steps_tilt"], wrench)


def outcome(score: dict[str, Any] | None, poured: float | None, spilled: float | None, end_reason: str | None,
            minimum: float | None, spill_tolerance: float) -> str:
    """Trial outcome = the first thing that went wrong, in task order; weights decide only once the bottle is back."""
    if end_reason == "error":
        return "error"
    if not score:
        return "other"
    stages = score["stages"]
    if not stages["grasp"]:
        return "grasp_fail"
    if score["drop"]:
        return "drop"
    if not stages["lift"]:
        return "lift_fail"
    if not stages["pour"]:
        return "pour_fail"
    if spilled is not None and spilled > spill_tolerance:
        return "spill"
    if not stages["place"]:
        return "place_fail"
    if poured is None:
        return "other"
    if minimum is not None and poured < minimum:
        return "under"
    return "success"
