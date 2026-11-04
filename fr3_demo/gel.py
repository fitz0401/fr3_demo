"""GelSight gel guard: the gel's own deformation -> a level, and which way it is loaded.

The marker tracker (tactile_shear) measures how far the gel surface has
shifted, twisted and been pressed since an unloaded reference. That
deformation is what tears the gel, so the level is it over configured limits,
whatever holds the tool: guide mode stiffens against further loading as the
level rises, and streams yield at level 1.

Which way the arm is loading the gel (so guide mode can block only that way)
comes from a fit by fr3-gel-calibrate against Franka's force estimate. With
a hard finger opposite the gel, the gel feels only part of the load, a part
that changes with every grasp, so only the direction of that fit is used,
never its size. Without one, guide mode holds at the limit instead.

The gel is pressed whenever the gripper closes, so the reference is retaken
(re-zeroed) after each gripper command, once the image has settled. Everything
fails closed: no reference, lost tracking, stale frames or a processing error
all give an invalid reading, never a small number.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fr3_demo.tactile import TactileFrame
    from fr3_demo.tactile_shear import MarkerShearTracker, ShearEstimate


@dataclass(frozen=True)
class GelLimits:
    """The gel's deformation at level 1: what tears it, whatever holds the tool."""

    shift_px: float = 15.0  # marker shift
    twist_rad: float = math.radians(6.0)
    press: float = 0.02  # spread (scale - 1) of the marker field when pressed

    def __post_init__(self) -> None:
        values = (self.shift_px, self.twist_rad, self.press)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("gel limits must be positive")

    def level(self, estimate: ShearEstimate) -> float:
        shift = float(np.hypot(*estimate.shear_px))
        return max(
            shift / self.shift_px,
            abs(float(estimate.twist_rad)) / self.twist_rad,
            max(0.0, float(estimate.spread)) / self.press,
        )


@dataclass(frozen=True)
class GelDirections:
    """Which way the arm is loading the gel, from a fr3-gel-calibrate fit.

    The fit maps the gel's shift, spread and twist to Franka's force estimate
    (robot on environment, end-effector frame), so the result points the way
    the arm pushes; motion along it loads the gel further. Its size is not
    used: with a hard finger opposite the gel it changes with every grasp.
    """

    shear_to_force: np.ndarray  # (3, 2): per px of marker shift along image x and y
    spread_to_force: np.ndarray  # (3,)
    twist_to_force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)

    def __post_init__(self) -> None:
        shear = np.asarray(self.shear_to_force, dtype=float)
        spread = np.asarray(self.spread_to_force, dtype=float)
        twist = np.asarray(self.twist_to_force, dtype=float)
        if (
            shear.shape != (3, 2)
            or spread.shape != (3,)
            or twist.shape != (3,)
            or not (np.all(np.isfinite(shear)) and np.all(np.isfinite(spread)) and np.all(np.isfinite(twist)))
        ):
            raise ValueError(
                "gel directions need a finite 3x2 shear_to_force and 3-vectors spread_to_force and twist_to_force"
            )
        if not (np.any(shear) or np.any(twist)):
            raise ValueError("gel directions map every shift and twist to nothing")
        object.__setattr__(self, "shear_to_force", shear)
        object.__setattr__(self, "spread_to_force", spread)
        object.__setattr__(self, "twist_to_force", twist)

    def load(self, estimate: ShearEstimate) -> np.ndarray:
        shear = np.asarray(estimate.shear_px, dtype=float)
        return (
            self.shear_to_force @ shear
            + self.spread_to_force * float(estimate.spread)
            + self.twist_to_force * float(estimate.twist_rad)
        )

    def direction(self, estimate: ShearEstimate) -> np.ndarray | None:
        load = self.load(estimate)
        size = float(np.linalg.norm(load))
        return load / size if size > 0 and math.isfinite(size) else None


def gel_guard_settings(gel: dict[str, Any]) -> tuple[GelLimits, GelDirections | None]:
    """Limits and, when [gel] has a fr3-gel-calibrate fit, directions from the config section."""

    limits = GelLimits(
        shift_px=float(gel.get("shift_limit_px", 15.0)),
        twist_rad=math.radians(float(gel.get("twist_limit_deg", 6.0))),
        press=float(gel.get("press_limit_percent", 2.0)) / 100.0,
    )
    if "shear_to_force" not in gel:
        return limits, None
    directions = GelDirections(
        shear_to_force=np.asarray(gel["shear_to_force"], dtype=float),
        spread_to_force=np.asarray(gel.get("spread_to_force", [0.0, 0.0, 0.0]), dtype=float),
        twist_to_force=np.asarray(gel.get("twist_to_force", [0.0, 0.0, 0.0]), dtype=float),
    )
    return limits, directions


def describe_gel_guard(limits: GelLimits, directions: GelDirections | None) -> str:
    text = (
        f"GelSight guard on: level 1 is a marker shift of {limits.shift_px:.0f} px, a twist of "
        f"{math.degrees(limits.twist_rad):.0f} deg or {100 * limits.press:.1f}% pressing."
    )
    if directions is None:
        text += (
            " No loading direction is calibrated, so guide mode holds the arm at the limit; "
            "open the gripper (B) to unload the gel."
        )
    return text


@dataclass(frozen=True)
class GelReading:
    """The latest gel state, as guide mode and the stream guards consume it."""

    seq: int  # increases with every processed frame
    timestamp: float  # host monotonic time of the frame
    valid: bool
    reason: str  # why it is invalid, or "" when valid
    level: float
    direction_ee: tuple[float, float, float] | None  # unit; arm motion along it loads the gel more
    zeroing: bool = False
    shift_px: tuple[float, float] = (0.0, 0.0)  # the gel's deformation behind the level
    twist_rad: float = 0.0
    spread: float = 0.0


def _median_motion(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Median distance from each marker to the nearest marker of the previous frame."""

    if previous is None or len(previous) < 4 or len(current) < 4:
        return float("inf")
    distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2).min(axis=1)
    return float(np.median(distances))


class GelMonitor:
    """Runs the marker tracker on every GelSight frame and keeps the latest reading.

    ``on_frame`` is a GelSightCamera observer, so it runs on the capture thread;
    it never raises, because a gel problem must not stop capture or recording.
    ``latest`` may be called from any thread.
    """

    def __init__(
        self,
        limits: GelLimits,
        directions: GelDirections | None = None,
        tracker: MarkerShearTracker | None = None,
        settle_frames: int = 8,
        settle_px: float = 0.25,
        direction_level: float = 0.1,
    ) -> None:
        if tracker is None:
            from fr3_demo.tactile_shear import MarkerShearTracker

            tracker = MarkerShearTracker()
        self.limits = limits
        self.directions = directions
        self._tracker = tracker
        self.settle_frames = int(settle_frames)
        self.settle_px = float(settle_px)
        # Below this level the gel is barely loaded and its direction is noise.
        self.direction_level = float(direction_level)
        self._lock = threading.Lock()
        self._seq = 0
        self._suspended: str | None = None
        self._zero_requested = True
        self._previous_markers: np.ndarray | None = None
        self._settled = 0
        self._reading = GelReading(0, 0.0, False, "not zeroed yet", 0.0, None, zeroing=True)

    def suspend(self, reason: str) -> None:
        """Report invalid until rezero(), e.g. while the gripper moves and loads the gel."""

        with self._lock:
            self._suspended = reason
            self._reading = replace(self._reading, valid=False, reason=reason, zeroing=True)

    def rezero(self) -> None:
        """Retake the unloaded reference once the gel image has settled (after a grasp)."""

        with self._lock:
            self._suspended = None
            self._zero_requested = True
            self._previous_markers = None
            self._settled = 0
            self._reading = replace(self._reading, valid=False, reason="re-zeroing", zeroing=True)

    @property
    def zeroed(self) -> bool:
        with self._lock:
            return not self._zero_requested

    def wait_zeroed(self, timeout: float) -> GelReading:
        """Block until a reference is taken; raise with the last reason if it is not."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.zeroed:
                return self.latest()
            time.sleep(0.02)
        reason = self.latest().reason
        raise RuntimeError(
            f"The GelSight could not be zeroed within {timeout:.0f} s ({reason}). "
            "Is the marker gel fitted, and is nothing touching it?"
        )

    def on_frame(self, frame: TactileFrame) -> None:
        try:
            self._process(frame)
        except Exception as error:  # noqa: BLE001 - reported as an invalid reading, never raised into capture
            self._publish(frame.captured_monotonic, False, f"gel processing failed: {error}")

    def _process(self, frame: TactileFrame) -> None:
        with self._lock:
            suspended = self._suspended
            zeroing = self._zero_requested
        if suspended is not None:
            self._publish(frame.captured_monotonic, False, suspended)
            return
        if zeroing:
            markers = self._tracker.detect(frame.image)
            moved = _median_motion(self._previous_markers, markers)
            self._previous_markers = markers
            self._settled = self._settled + 1 if moved < self.settle_px else 0
            if self._settled < self.settle_frames:
                self._publish(frame.captured_monotonic, False, "re-zeroing: waiting for the gel to settle")
                return
            self._tracker.set_reference(frame.image)
            with self._lock:
                self._zero_requested = False
            self._publish(frame.captured_monotonic, True, "")
            return
        estimate = self._tracker.update(frame.image, frame.captured_monotonic, frame.frame_number)
        if not estimate.valid:
            self._publish(frame.captured_monotonic, False, estimate.reason or "marker tracking failed")
            return
        level = self.limits.level(estimate)
        if not math.isfinite(level):
            self._publish(frame.captured_monotonic, False, "gel level is not finite")
            return
        direction = None
        if self.directions is not None and level >= self.direction_level:
            direction = self.directions.direction(estimate)
        self._publish(frame.captured_monotonic, True, "", level, direction, estimate)

    def _publish(
        self,
        timestamp: float,
        valid: bool,
        reason: str,
        level: float = 0.0,
        direction: np.ndarray | None = None,
        estimate: ShearEstimate | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            self._reading = GelReading(
                seq=self._seq,
                timestamp=timestamp,
                valid=valid,
                reason=reason,
                level=float(level),
                direction_ee=None if direction is None else tuple(float(value) for value in direction),
                zeroing=self._zero_requested or self._suspended is not None,
                shift_px=(0.0, 0.0) if estimate is None else tuple(float(value) for value in estimate.shear_px),
                twist_rad=0.0 if estimate is None else float(estimate.twist_rad),
                spread=0.0 if estimate is None else float(estimate.spread),
            )

    def latest(self, max_age: float = 0.2) -> GelReading:
        """The newest reading; invalid once frames are older than ``max_age`` s."""

        with self._lock:
            reading = self._reading
        if reading.valid and time.monotonic() - reading.timestamp > max_age:
            return replace(reading, valid=False, reason="GelSight frames stopped arriving")
        return reading


class GelStreamGuard:
    """When a stream (teleop or a policy) must yield because of the gel.

    It must yield at ``stop_level``, on an invalid reading, or when a re-zero
    after a gripper command takes longer than ``zero_grace_s``; during that
    grace the stream is protected by Bamboo's contact limits alone.
    """

    def __init__(self, monitor: GelMonitor, stop_level: float = 1.0, zero_grace_s: float = 2.5) -> None:
        self.monitor = monitor
        self.stop_level = float(stop_level)
        self.zero_grace_s = float(zero_grace_s)
        self._zero_started: float | None = None

    def check(self) -> str | None:
        """A reason to yield, or None."""

        reading = self.monitor.latest()
        now = time.monotonic()
        if reading.zeroing:
            if self._zero_started is None:
                self._zero_started = now
            if now - self._zero_started > self.zero_grace_s:
                return f"the GelSight did not re-zero within {self.zero_grace_s:.1f} s ({reading.reason})"
            return None
        self._zero_started = None
        if not reading.valid:
            return f"the GelSight signal is not trustworthy ({reading.reason})"
        if reading.level >= self.stop_level:
            return (
                f"the gel reached level {reading.level:.2f} (shift {np.hypot(*reading.shift_px):.1f} px, "
                f"twist {math.degrees(reading.twist_rad):.1f} deg, pressed {100 * reading.spread:.1f}%)"
            )
        return None


def limit_twist(twist: np.ndarray, reading: GelReading, rotation: np.ndarray, slow_level: float = 0.5) -> np.ndarray:
    """Remove the part of a commanded flange twist that would load the gel further.

    From ``slow_level`` to level 1 the motion along the gel's load direction
    (and all rotation, which swings the tool tip) fades to zero; backing off
    and sliding sideways stay free. While the gel is re-zeroing the command is
    zero; an invalid reading otherwise slows everything to a quarter, so the
    operator can still back off while Bamboo's contact limits protect.
    """

    command = np.asarray(twist, dtype=float).copy()
    if reading.zeroing:
        return np.zeros_like(command)
    if not reading.valid:
        return 0.25 * command
    if reading.direction_ee is None or reading.level < slow_level:
        return command
    keep = float(np.clip((1.0 - reading.level) / (1.0 - slow_level), 0.0, 1.0))
    direction = np.asarray(rotation, dtype=float) @ np.asarray(reading.direction_ee, dtype=float)
    toward = float(command[:3] @ direction)
    if toward > 0.0:
        command[:3] -= (1.0 - keep) * toward * direction
    command[3:] *= keep
    return command


AXES = ("x", "y", "z")
# Ridge weight on standardised features in the calibration fit.
RIDGE = 1e-3
# A push axis that, pushed as hard as in the run, deformed the gel less than
# this share of its limit is carried by something else (a hard opposite finger).
AROUND_THE_GEL_LEVEL = 0.2


def fit_calibration(
    shear_px: np.ndarray, spread: np.ndarray, force_ee: np.ndarray, twist: np.ndarray | None = None
) -> dict[str, Any]:
    """Least-squares map from marker shift (px), spread and twist to the force through the gel.

    ``force_ee`` is Franka's external force estimate (robot on environment)
    in the end-effector frame, tared, while only the tool is touched, so it
    is the force through the gel. A constant offset is fitted too, so drift
    in the estimate or a load left at zeroing does not spoil the fit; it is
    reported, not kept. Returns the maps and how well they fit, overall and
    per end-effector axis.
    """

    shear = np.asarray(shear_px, dtype=float).reshape(-1, 2)
    press = np.asarray(spread, dtype=float).reshape(-1, 1)
    turn = np.zeros((len(shear), 1)) if twist is None else np.asarray(twist, dtype=float).reshape(-1, 1)
    force = np.asarray(force_ee, dtype=float).reshape(-1, 3)
    if not (len(shear) == len(press) == len(turn) == len(force)) or len(force) < 20:
        raise ValueError("calibration needs at least 20 matched samples")
    features = np.hstack((shear, press, turn, np.ones((len(force), 1))))
    # A light ridge on the scaled features: a sideways push shows as both
    # shift and twist, and without it the fit splits that between them
    # arbitrarily, leaking force into the other axes. The offset is free.
    scale = np.std(features[:, :4], axis=0)
    penalty = np.zeros((4, 5))
    penalty[:, :4] = np.diag(np.sqrt(RIDGE * len(force)) * np.where(scale > 0, scale, 1.0))
    solution, *_ = np.linalg.lstsq(np.vstack((features, penalty)), np.vstack((force, np.zeros((4, 3)))), rcond=None)
    residual = force - features @ solution
    centred = force - force.mean(axis=0)
    total = float(np.sum(centred**2))
    axis_total = np.sum(centred**2, axis=0)
    axis_error = np.sum(residual**2, axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return {
        "shear_to_force": solution[:2].T,  # (3, 2)
        "spread_to_force": solution[2],  # (3,)
        "twist_to_force": solution[3],  # (3,)
        "offset_n": solution[4],  # (3,), reported only
        "r_squared": 1.0 - float(np.sum(axis_error)) / total if total > 0 else 0.0,
        "axis_r_squared": np.where(axis_total > 0, 1.0 - axis_error / np.maximum(axis_total, 1e-12), 0.0),
        "axis_range_n": np.percentile(force, 98, axis=0) - np.percentile(force, 2, axis=0),
        "rms_error_n": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "peak_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        # Spread of the pushes: the second singular value says whether more
        # than one direction was pushed.
        "push_directions": singular / np.sqrt(len(force)),
    }


def align_calibration(run: dict[str, np.ndarray], max_lag_s: float = 0.2) -> tuple[float, dict[str, Any]]:
    """Fit a recorded run at the gel-to-Franka delay that fits best; return the delay and the fit."""

    valid = np.asarray(run["valid"], dtype=bool)
    frame_time = np.asarray(run["frame_time"], dtype=float)[valid]
    force_time = np.asarray(run["force_time"], dtype=float)
    force = np.asarray(run["force_ee"], dtype=float)
    best: tuple[float, dict[str, Any]] | None = None
    for lag in np.arange(-max_lag_s, max_lag_s + 1e-9, 0.01):
        sampled = frame_time + lag
        inside = (sampled >= force_time[0]) & (sampled <= force_time[-1])
        if inside.sum() < 20:
            continue
        matched = np.stack([np.interp(sampled[inside], force_time, force[:, axis]) for axis in range(3)], axis=1)
        fit = fit_calibration(
            np.asarray(run["shear_px"])[valid][inside],
            np.asarray(run["spread"])[valid][inside],
            matched,
            np.asarray(run["twist_rad"])[valid][inside],
        )
        if best is None or fit["r_squared"] > best[1]["r_squared"]:
            best = (float(lag), fit)
    if best is None:
        raise ValueError("calibration needs at least 20 tracked frames inside the force recording")
    return best


def axis_sensitivity(run: dict[str, np.ndarray], limits: GelLimits, lag: float = 0.0) -> np.ndarray:
    """How much one newton along each EE axis moves the gel, in level units (deformation over limits)."""

    valid = np.asarray(run["valid"], dtype=bool)
    frame_time = np.asarray(run["frame_time"], dtype=float)[valid] + lag
    force_time = np.asarray(run["force_time"], dtype=float)
    force = np.stack([np.interp(frame_time, force_time, run["force_ee"][:, axis]) for axis in range(3)], axis=1)
    deformation = np.column_stack(
        (
            np.asarray(run["shear_px"], dtype=float)[valid] / limits.shift_px,
            np.asarray(run["twist_rad"], dtype=float)[valid] / limits.twist_rad,
            np.asarray(run["spread"], dtype=float)[valid] / limits.press,
        )
    )
    response, *_ = np.linalg.lstsq(np.column_stack((force, np.ones(len(force)))), deformation, rcond=None)
    return np.linalg.norm(response[:3], axis=1)


def report_calibration(run: dict[str, np.ndarray], limits: GelLimits | None = None) -> tuple[bool, list[str]]:
    """Print how a recorded run fits and why it does not, if it does not; return (usable, lines to paste).

    Each end-effector axis that was pushed is either seen by the gel, and then
    must fit, or barely moves the gel at all: then its force goes around the
    gel (through a hard opposite finger, say), which the gel can neither
    measure nor be torn by, and it is reported, not refused.
    """

    limits = limits or GelLimits()
    valid = np.asarray(run["valid"], dtype=bool)
    reasons = [str(reason) for reason, ok in zip(run["reason"], valid) if not ok]
    lost = [reason for reason in reasons if reason.startswith("lost track")]
    print(f"\n{len(valid)} gel frames, {valid.mean():.0%} tracked.", end=" ")
    if reasons:
        common = max(set(reasons), key=reasons.count)
        print(f"Most common reason when not: {common!r}.")
    else:
        print()
    lag, fit = align_calibration(run)
    # How far pushing each axis as hard as it was pushed deformed the gel, in level units.
    produced = axis_sensitivity(run, limits, lag) * fit["axis_range_n"]
    shift = np.hypot(*np.asarray(run["shear_px"], dtype=float)[valid].T)
    twist = np.abs(np.asarray(run["twist_rad"], dtype=float)[valid])
    press = np.asarray(run["spread"], dtype=float)[valid]
    print(
        f"Fit: R^2 {fit['r_squared']:.2f}, pushes up to {fit['peak_force_n']:.1f} N; gel lags Franka by "
        f"{-lag * 1e3:.0f} ms; offset {np.linalg.norm(fit['offset_n']):.2f} N."
    )
    print(
        f"At those pushes the gel shifted up to {shift.max():.1f} px, twisted up to {np.degrees(twist.max()):.1f} deg "
        f"and pressed up to {100 * max(0.0, press.max()):.1f}% (limits: {limits.shift_px:.0f} px, "
        f"{np.degrees(limits.twist_rad):.0f} deg, {100 * limits.press:.1f}%)."
    )
    unseen, inconsistent, unpushed = [], [], []
    for axis, name in enumerate(AXES):
        span, r2, moved = fit["axis_range_n"][axis], fit["axis_r_squared"][axis], produced[axis]
        if span < 3.0:
            verdict = "barely pushed"
            unpushed.append(name)
        elif moved < AROUND_THE_GEL_LEVEL:
            verdict = "goes around the gel"
            unseen.append(name)
        elif r2 < 0.6:
            verdict = "seen, but inconsistently"
            inconsistent.append(name)
        else:
            verdict = "seen by the gel"
        print(
            f"  EE {name}: force range {span:4.1f} N, moved the gel {moved:4.0%} of its limit, fit R^2 {r2:.2f}: "
            f"{verdict}"
        )
    if unseen:
        print(
            f"  Force along EE {', '.join(unseen)} hardly moves the gel: it goes through the other finger. The gel "
            "is not strained by it, so the guard does not need it, but it cannot measure it either."
        )
    problems = []
    if valid.mean() < 0.5:
        problems.append(
            "the markers were lost for much of the run"
            + (" (push more slowly, and if the tool pivoted or slipped in the grip, re-grasp it)" if lost else "")
        )
    if fit["peak_force_n"] < 3.0:
        problems.append("push harder (at least 3 N)")
    if unpushed:
        problems.append(f"push along EE {', '.join(unpushed)} too, a few newtons each")
    if inconsistent:
        problems.append(
            f"EE {', '.join(inconsistent)} pushes move the gel inconsistently: push slowly, hold each push, "
            "touch only the tool, and if it pivots in the grip, push closer to the grip"
        )
    if not any(name not in unseen + unpushed + inconsistent for name in AXES):
        problems.append("no direction moved the gel consistently")
    if problems:
        print("Not good enough to use: " + "; ".join(problems) + ".")
        return False, []

    def row(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{value:.5f}" for value in values) + "]"

    lines = [
        "shear_to_force = [" + ", ".join(row(values) for values in fit["shear_to_force"]) + "]",
        "spread_to_force = " + row(fit["spread_to_force"]),
        "twist_to_force = " + row(fit["twist_to_force"]),
    ]
    return True, lines


def calibrate_main(argv: list[str] | None = None) -> int:
    """Fit the GelSight's force calibration against Franka's own force estimate.

    The arm must stand still in Execution mode with no control loop running,
    and only the tool may be touched: then Franka's external force estimate is
    exactly the force through the gel. Every run is saved, and --analyze
    re-fits a saved run without recording again.
    """

    import argparse
    import queue
    from pathlib import Path

    from fr3_demo.settings import default_config_path, load_config

    parser = argparse.ArgumentParser(description=calibrate_main.__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--seconds", type=float, default=40.0, help="how long to record pushes")
    parser.add_argument("--server-ip")
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--analyze", type=Path, help="re-fit a saved run instead of recording one")
    args = parser.parse_args(argv)
    if args.analyze is not None:
        with np.load(args.analyze) as saved:
            run = {key: saved[key] for key in saved.files}
        usable, lines = report_calibration(run, gel_guard_settings(load_config(args.config).get("gel", {}))[0])
    else:
        config = load_config(args.config)
        run = _record_calibration(args, config, queue)
        folder = args.config.resolve().parent / "data" / "gel_calibration"
        folder.mkdir(parents=True, exist_ok=True)
        saved_path = folder / f"gel_calibration_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(saved_path, **run)
        print(f"Saved this run to {saved_path} (re-fit it with: fr3-gel-calibrate --analyze {saved_path})")
        usable, lines = report_calibration(run, gel_guard_settings(config.get("gel", {}))[0])
    if not usable:
        print("Run it again.")
        return 1
    print("\nCopy into [gel] in config.toml, for the direction guide mode blocks at the limit:")
    for line in lines:
        print(line)
    return 0


def _record_calibration(args: Any, config: dict[str, Any], queue: Any) -> dict[str, np.ndarray]:
    from fr3_demo.tactile import GelSightCamera
    from fr3_demo.tactile_shear import MarkerShearTracker

    robot_config = config.get("robot", {})
    tactile_config = config.get("tactile", {})
    from bamboo import BambooFrankaClient

    def connect() -> Any:
        return BambooFrankaClient(
            server_ip=args.server_ip or robot_config.get("server_ip", "172.16.0.20"),
            control_port=args.control_port or int(robot_config.get("control_port", 5555)),
            enable_gripper=False,
        )

    def force_ee(sample: dict[str, Any]) -> np.ndarray:
        rotation = np.asarray(sample["ee_pose"], dtype=float)[:3, :3]
        return rotation.T @ np.asarray(sample["ext_wrench"][:3], dtype=float)

    client = connect()
    camera = None
    sampling = threading.Event()
    force_samples: list[tuple[float, np.ndarray]] = []
    sampler: threading.Thread | None = None
    try:
        state = client.get_joint_states()
        if "ext_wrench" not in state:
            raise RuntimeError("This Bamboo controller does not report Franka's force estimate; update Bamboo")
        if state.get("loop_mode", "idle") != "idle":
            raise RuntimeError("Stop teleop, guide mode or the policy first: the arm must stand still")
        frames: queue.Queue[TactileFrame] = queue.Queue(maxsize=64)
        camera = GelSightCamera(
            device=tactile_config.get("device") or None,
            width=int(tactile_config.get("width", 320)),
            height=int(tactile_config.get("height", 240)),
            rotate_180=bool(tactile_config.get("rotate_180", False)),
        )
        camera.add_frame_observer(lambda frame: frames.put_nowait(frame) if not frames.full() else None)
        camera.start()
        tracker = MarkerShearTracker()

        print("Zeroing: grip the tool as you will use it, and touch nothing for a moment...")
        previous, settled = None, 0
        while settled < 10:
            image = frames.get(timeout=2.0).image
            markers = tracker.detect(image)
            settled = settled + 1 if _median_motion(previous, markers) < 0.25 else 0
            previous = markers
        count = tracker.set_reference(image)
        tare = np.mean([force_ee(client.get_joint_states()) for _ in range(50)], axis=0)
        print(f"Zeroed on {count} markers; Franka's force at rest {np.linalg.norm(tare):.2f} N (tared).")

        def sample_forces() -> None:
            # Franka's estimate at ~200 Hz on its own connection, so each gel
            # frame can be matched to the force at its own time.
            own = connect()
            try:
                while sampling.is_set():
                    before = time.monotonic()
                    sample = own.get_joint_states()
                    force_samples.append((0.5 * (before + time.monotonic()), force_ee(sample) - tare))
                    time.sleep(0.004)
            finally:
                own.close()

        sampling.set()
        sampler = threading.Thread(target=sample_forces, name="gel-calibration-force", daemon=True)
        sampler.start()
        print(
            f"\nFor the next {args.seconds:.0f} s, press the tool where it will meet the fruit, about 3 to 10 N, "
            "slowly, in several directions: along the tool toward the gripper, sideways both ways, and into and "
            "away from its face. Hold each push for a second or two, then let go fully. Touch only the tool, "
            "never the gripper or the arm.\n"
        )
        rows: list[tuple] = []
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            frame = frames.get(timeout=2.0)
            estimate = tracker.update(frame.image, frame.captured_monotonic, frame.frame_number)
            rows.append(
                (
                    frame.captured_monotonic,
                    estimate.valid,
                    estimate.reason,
                    estimate.shear_px,
                    estimate.twist_rad,
                    estimate.spread,
                )
            )
            force = force_samples[-1][1] if force_samples else np.zeros(3)
            shift = np.hypot(*estimate.shear_px) if estimate.valid else float("nan")
            twist = np.degrees(estimate.twist_rad) if estimate.valid else float("nan")
            print(
                f"\r  force {np.linalg.norm(force):5.1f} N | marker shift {shift:5.1f} px | twist {twist:5.1f} deg | "
                f"{'tracking' if estimate.valid else estimate.reason:28.28s}",
                end="",
                flush=True,
            )
        print()
    finally:
        sampling.clear()
        if sampler is not None:
            sampler.join(timeout=2.0)
        if camera is not None:
            camera.close()
        client.close()
    if not force_samples:
        raise RuntimeError("Franka's force estimate was never read")
    return {
        "frame_time": np.array([row[0] for row in rows], dtype=float),
        "valid": np.array([row[1] for row in rows], dtype=bool),
        "reason": np.array([row[2] for row in rows], dtype=str),
        "shear_px": np.array([row[3] for row in rows], dtype=float).reshape(-1, 2),
        "twist_rad": np.array([row[4] for row in rows], dtype=float),
        "spread": np.array([row[5] for row in rows], dtype=float),
        "force_time": np.array([sample[0] for sample in force_samples], dtype=float),
        "force_ee": np.array([sample[1] for sample in force_samples], dtype=float).reshape(-1, 3),
        "tare_n": tare,
    }
