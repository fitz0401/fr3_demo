"""Trial scheduling, automatic rollout logs and result tables for the bluecup_force robot round.

The operator's job is reduced to: reset the scene, start the runner with the command this tool
prints, and type (or let the scale send) the weights after each trial. Everything else -- which
model and fault to run, when the fault was active, why a rollout ended, success/spill/under/over --
is recorded automatically.

    fr3-pi05-trials plan    --session r1 --models flag moddrop alloc_fixed --conditions none cams_absent --reps 10
    fr3-pi05-trials next    --session r1                 # prints the next trial and the runner command
    fr3-pi05-trials record  --session r1 --trial 7 --rollout-log trials/r1/logs/7.json
    fr3-pi05-trials summary --session r1 --target-g 120 --tolerance-g 15

Scope-v5 round (fr3_train/bc2f/DEPLOY_PLAN.md): models bc2f_vanilla / bc2f_moddrop / bc2f_moddrop_prop, condition
"none" only (every input present), no target amount. The runner ends each rollout itself (--auto-stop) and
records joints, gripper and the raw arm stream; ``record`` scores the stages with fr3_pi05.autoscore (grasp,
lift, pour, place: completion 0-4, timings, drops, retries, placement error) and only asks for the weights.
Outcome = the first thing that went wrong in task order (grasp_fail, drop, lift_fail, pour_fail, spill,
place_fail, under), else success: all four stages, at least --min-poured-g in the cup, spill within
--spill-tolerance-g.

Weights (grams): tare the empty cup on the scale before the trial (cup_before = 0) and weigh the
bottle while refilling it to its start level (bottle_before). After the rollout and ~3 s of
settling read the cup (cup_after) and weigh the bottle again (bottle_after). Then
    poured_g  = cup_after - cup_before
    spilled_g = (bottle_before - bottle_after) - poured_g
A scale with a USB/serial output can be read automatically with --scale-port (needs pyserial).

The runner side writes one JSON per rollout through ``RolloutLogger`` (see its docstring and
fr3_train/bluecup/DEPLOY_PLAN.md for the integration points).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEDULE_FIELDS = ("trial_id", "block", "model", "preset", "trigger", "rep")
RESULT_FIELDS = (
    "trial_id", "model", "preset", "trigger", "rep", "recorded_at",
    "end_reason", "duration_s", "pour_started", "pour_steps", "max_tilt_deg", "fault_active_steps",
    "gripper_closed_empty", "server_model", "bundle_step",
    "cup_before_g", "cup_after_g", "bottle_before_g", "bottle_after_g", "poured_g", "spilled_g",
    "outcome", "notes",
    # fr3_pi05.autoscore (empty when the rollout left no per-step recording)
    "completion", "stage_grasp", "stage_lift", "stage_pour", "stage_place", "drop",
    "start_delay_s", "t_grasp_s", "t_lift_s", "t_pour_s", "t_release_s", "pour_s", "pour_bouts",
    "grasp_attempts", "empty_closes", "longest_idle_s", "path_rad", "place_dz_cm", "place_xy_cm",
    "held_mass_g", "max_hold_fz_N", "max_force_N", "force_available", "inference_ms_median",
)
OUTCOMES = ("success", "under", "over", "spill", "grasp_fail", "drop", "lift_fail", "pour_fail", "place_fail",
            "abort", "error", "other")
SCORE_FIELDS = ("start_delay_s", "t_grasp_s", "t_lift_s", "t_pour_s", "t_release_s", "pour_s", "pour_bouts",
                "grasp_attempts", "empty_closes", "longest_idle_s", "path_rad", "held_mass_g", "max_hold_fz_N",
                "max_force_N", "force_available")
MODEL_CHECKPOINTS = {
    "flag": "pi05_bluecup_flag_10500",
    "moddrop": "pi05_bluecup_moddrop_14999",
    "alloc_fixed": "pi05_bluecup_alloc_fixed_14500",
    "bc2f_vanilla": "pi05_bc2f_vanilla_s0_14999",
    "bc2f_moddrop": "pi05_bc2f_moddrop_s0_14999",
    "bc2f_moddrop_prop": "pi05_bc2f_moddrop_prop_s0_14999",
}


def model_profile(model: str) -> str:
    """Runner/server contract profile of a scheduled model."""
    return "bc2f_force" if model.startswith("bc2f_") else "bluecup_force"


def bundle_model_name(model: str) -> str:
    """The `model` field of the bundle's BUNDLE.json, as the server reports it (directory name without the step)."""
    return MODEL_CHECKPOINTS[model].rsplit("_", 1)[0]


def raw_arrays_path(log_path: Path) -> Path:
    """Where a rollout's raw arm stream lives: <log stem>.raw.npz next to the JSON."""
    log_path = Path(log_path)
    return log_path.with_name(log_path.stem + ".raw.npz")


# --------------------------------------------------------------------------------------------- #
# Runner side: one JSON per rollout.
# --------------------------------------------------------------------------------------------- #
@dataclass
class RolloutLogger:
    """Collects what happened in one rollout and writes it as JSON when the rollout ends.

    Runner integration:
        log = RolloutLogger(path, meta={...server metadata subset, fault preset/trigger, pour_deg...})
        each control step:   log.step(detector.state(), fault_active, gripper_position, q=q, time_sec=state["time_sec"])
        each inference:      log.inference(ms)
        notable events:      log.event("safety_violation", detail=str(error))
        always, in finally:  log.finish(end_reason, error=None, arrays={"raw_force": ...})

    Steps given with ``q`` are also kept as arrays and written to ``<path stem>.npz`` (``steps_t`` s since the
    start, ``steps_time_sec`` controller time, ``steps_q``, ``steps_gripper``, ``steps_tilt``, ``steps_pouring``;
    ~0.3 MB, committed with the results); ``arrays`` passed to ``finish`` (the raw arm stream, ~10-20 MB) go to
    ``<path stem>.raw.npz``, which stays on the workstation. fr3_pi05.autoscore scores the rollout from both.
    """

    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    closed_empty_below: float = 0.05  # gripper reading this low while "holding" = closed on nothing
    _start: float = field(default_factory=time.time)
    _steps: int = 0
    _fault_steps: int = 0
    _pour_first_step: int | None = None
    _pour_steps: int = 0
    _max_tilt: float = 0.0
    _gripper_min: float = 1.0
    _inference_ms: list[float] = field(default_factory=list)
    _events: list[dict[str, Any]] = field(default_factory=list)
    _rows: list[tuple] = field(default_factory=list)
    _start_monotonic: float = field(default_factory=time.monotonic)
    _finished: bool = False

    def step(self, detector_state: dict[str, Any] | None, fault_active: bool, gripper_position: float,
             q: Any = None, time_sec: float | None = None) -> None:
        if q is not None:
            state = detector_state or {}
            self._rows.append((time.monotonic() - self._start_monotonic,
                               float("nan") if time_sec is None else float(time_sec),
                               tuple(float(x) for x in q), float(gripper_position),
                               float(state.get("tilt_deg", 0.0)), bool(state.get("pouring", False))))
        self._steps += 1
        self._fault_steps += int(bool(fault_active))
        self._gripper_min = min(self._gripper_min, float(gripper_position))
        if detector_state:
            if detector_state.get("pouring"):
                self._pour_steps += 1
                if self._pour_first_step is None:
                    self._pour_first_step = self._steps
            self._max_tilt = max(self._max_tilt, float(detector_state.get("tilt_deg", 0.0)))

    def inference(self, milliseconds: float) -> None:
        self._inference_ms.append(float(milliseconds))

    def event(self, name: str, **detail: Any) -> None:
        self._events.append({"t": round(time.time() - self._start, 3), "event": name, **detail})

    def write_arrays(self, arrays: dict[str, Any] | None = None) -> Path | None:
        """Write the per-step recording to <stem>.npz and `arrays` to <stem>.raw.npz; returns the steps file."""
        import numpy as np

        def save(target: Path, data: dict[str, Any]) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp.npz")
            np.savez_compressed(tmp, **data)
            os.replace(tmp, target)

        extra = {k: np.asarray(v) for k, v in (arrays or {}).items() if v is not None}
        if extra:
            save(raw_arrays_path(self.path), extra)
        if not self._rows:
            return None
        t, time_sec, q, gripper, tilt, pouring = zip(*self._rows)
        target = self.path.with_suffix(".npz")
        save(target, {"steps_t": np.asarray(t), "steps_time_sec": np.asarray(time_sec), "steps_q": np.asarray(q),
                      "steps_gripper": np.asarray(gripper), "steps_tilt": np.asarray(tilt),
                      "steps_pouring": np.asarray(pouring, dtype=bool)})
        return target

    def finish(self, end_reason: str, error: str | None = None,
               arrays: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._finished:
            return {}
        self._finished = True
        try:
            arrays_path = self.write_arrays(arrays)
        except Exception as write_error:  # noqa: BLE001 - the JSON must still be written
            arrays_path = None
            self.event("arrays_write_failed", detail=f"{type(write_error).__name__}: {write_error}")
        record = {
            **self.meta,
            "start_time": self._start,
            "end_time": time.time(),
            "duration_s": round(time.time() - self._start, 3),
            "steps": self._steps,
            "end_reason": end_reason,
            "error": error,
            "pour": {"started": self._pour_first_step is not None, "first_step": self._pour_first_step,
                     "steps": self._pour_steps, "max_tilt_deg": round(self._max_tilt, 2)},
            "fault_active_steps": self._fault_steps,
            "gripper_min": round(self._gripper_min, 4),
            "gripper_closed_empty": self._gripper_min < self.closed_empty_below,
            "inference_ms_median": round(statistics.median(self._inference_ms), 1) if self._inference_ms else None,
            "inferences": len(self._inference_ms),
            "arrays": arrays_path.name if arrays_path else None,
            "events": self._events,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        return record


# --------------------------------------------------------------------------------------------- #
# Operator side: schedule, record, summary.
# --------------------------------------------------------------------------------------------- #
def session_dir(root: Path, session: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session):
        raise ValueError("Session names may only contain letters, digits, '_', '-' and '.'")
    return root / session


def make_schedule(models: list[str], conditions: list[str], reps: int, block_reps: int, seed: int,
                  trigger: str = "pour") -> list[dict[str, Any]]:
    """Blocks of one model (one server start each) holding `block_reps` repetitions of every
    condition in random order; block order is shuffled per round so no model always goes first."""
    if reps < 1 or block_reps < 1:
        raise ValueError("reps and block_reps must be positive")
    rng = random.Random(seed)
    rows, trial, block = [], 0, 0
    rep_done = 0
    while rep_done < reps:
        n = min(block_reps, reps - rep_done)
        order = list(models)
        rng.shuffle(order)
        for model in order:
            cells = [(cond, rep_done + r) for r in range(n) for cond in conditions]
            rng.shuffle(cells)
            for cond, rep in cells:
                rows.append({"trial_id": trial, "block": block, "model": model, "preset": cond,
                             "trigger": "never" if cond == "none" else trigger, "rep": rep})
                trial += 1
            block += 1
        rep_done += n
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(tmp, path)


def next_trial(schedule: list[dict[str, str]], results: list[dict[str, str]]) -> dict[str, str] | None:
    done = {row["trial_id"] for row in results}
    return next((row for row in schedule if row["trial_id"] not in done), None)


def classify(poured: float | None, spilled: float | None, log: dict[str, Any], target: float | None,
             tolerance: float, spill_tolerance: float, minimum: float | None = None) -> str:
    reason = log.get("end_reason")
    if reason == "error":
        return "error"
    if log.get("gripper_closed_empty"):
        return "grasp_fail"
    if spilled is not None and spilled > spill_tolerance:
        return "spill"
    if poured is not None and target is not None:
        if poured < target - tolerance:
            return "under"
        if poured > target + tolerance:
            return "over"
        return "success" if reason in (None, "completed", "operator_stop") else "abort"
    if poured is not None and minimum is not None:
        if reason not in (None, "completed", "operator_stop"):
            return "abort"
        return "under" if poured < minimum else "success"
    return "abort" if reason not in (None, "completed", "operator_stop") else "other"


def read_scale(port: str, baud: int = 9600, pattern: str = r"(-?\d+(?:\.\d+)?)", settle_s: float = 3.0,
               timeout_s: float = 15.0) -> float:
    """Read a settled weight from a serial scale: the last value after `settle_s` of identical readings."""
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Reading the scale needs pyserial: pip install pyserial") from error
    regex = re.compile(pattern)
    deadline = time.monotonic() + timeout_s
    stable_since, last = None, None
    with serial.Serial(port, baud, timeout=1.0) as device:
        while time.monotonic() < deadline:
            match = regex.search(device.readline().decode("ascii", errors="ignore"))
            if not match:
                continue
            value = float(match.group(1))
            if last is None or abs(value - last) > 1e-9:
                last, stable_since = value, time.monotonic()
            elif time.monotonic() - stable_since >= settle_s:
                return value
    raise RuntimeError(f"Scale on {port} did not settle within {timeout_s:.0f} s")


def _weight(value: float | None, label: str, port: str | None) -> float | None:
    if value is not None:
        return value
    if port:
        input(f"Place the {label} on the scale and press Enter ... ")
        reading = read_scale(port)
        print(f"  {label}: {reading:.1f} g")
        return reading
    if not sys.stdin.isatty():
        return None
    answer = input(f"{label} in grams (blank = unknown): ").strip()
    return float(answer) if answer else None


def score_log(log_path: Path | None, force_model: str | None = None) -> dict[str, Any] | None:
    """fr3_pi05.autoscore score of a rollout from the .npz its RolloutLogger wrote next to the JSON."""
    if log_path is None:
        return None
    arrays_path = Path(log_path).with_suffix(".npz")
    if not arrays_path.exists():
        return None
    import numpy as np

    from fr3_pi05.autoscore import load_force_model, score_recording

    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}
    raw_path = raw_arrays_path(log_path)
    if raw_path.exists():
        with np.load(raw_path) as data:
            arrays.update({key: data[key] for key in data.files})
    if "steps_t" not in arrays or len(arrays["steps_t"]) < 2:
        return None
    return score_recording(arrays, load_force_model(force_model))


def score_columns(score: dict[str, Any] | None) -> dict[str, Any]:
    if not score:
        return {}
    place = score.get("place") or {}
    return {"completion": score["completion"], **{f"stage_{k}": v for k, v in score["stages"].items()},
            "drop": score["drop"], "place_dz_cm": place.get("dz_cm"), "place_xy_cm": place.get("xy_cm"),
            **{k: score.get(k) for k in SCORE_FIELDS}}


def describe_score(score: dict[str, Any]) -> str:
    stages = " ".join(f"{name}:{'OK' if ok else '--'}" for name, ok in score["stages"].items())
    place = score.get("place") or {}
    parts = [f"completion {score['completion']}/4 ({stages})", f"duration {score['duration_s']} s",
             f"start delay {score['start_delay_s']} s", f"pour {score['pour_s']} s in {score['pour_bouts']} bout(s)",
             f"grasp attempts {score['grasp_attempts']}"]
    if score["drop"]:
        parts.append(f"DROP at {score['t_drop_s']} s")
    if place.get("dz_cm") is not None:
        parts.append(f"release {place['dz_cm']} cm from grasp height, {place['xy_cm']} cm from grasp point")
    if score.get("held_mass_g") is not None:
        parts.append(f"held bottle ~{score['held_mass_g']:.0f} g")
    if not score.get("force_available"):
        parts.append("no raw force: lift judged by height only")
    return "; ".join(parts)


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def summarize(results: list[dict[str, str]], target: float | None, tolerance: float) -> str:
    def f(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row[key]) if row.get(key) not in (None, "") else None
        except ValueError:
            return None

    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in results:
        cells.setdefault((row["model"], row["preset"]), []).append(row)
    lines = ["| model | condition | n | success | poured g (mean ± sd) | |poured - target| g | spilled g | outcomes |",
             "|---|---|---|---|---|---|---|---|"]
    for (model, preset), rows in sorted(cells.items()):
        poured = [v for v in (f(r, "poured_g") for r in rows) if v is not None]
        spilled = [v for v in (f(r, "spilled_g") for r in rows) if v is not None]
        success = sum(r.get("outcome") == "success" for r in rows)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("outcome") or "?"] = counts.get(r.get("outcome") or "?", 0) + 1
        mean_sd = (f"{statistics.mean(poured):.1f} ± {statistics.stdev(poured):.1f}" if len(poured) > 1
                   else (f"{poured[0]:.1f}" if poured else "-"))
        err = f"{statistics.mean(abs(p - target) for p in poured):.1f}" if poured and target is not None else "-"
        spill = f"{statistics.mean(spilled):.1f}" if spilled else "-"
        lines.append(f"| {model} | {preset} | {len(rows)} | {success}/{len(rows)} | {mean_sd} | {err} | {spill} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + " |")
    if target is not None:
        lines.append(f"\nsuccess = within {target:.0f} ± {tolerance:.0f} g, no spill above threshold, rollout not aborted")
    scored = [r for r in results if r.get("completion") not in (None, "")]
    if scored:
        lines += ["", "Automatic stage scores (fr3_pi05.autoscore; stage counts = trials that reached it):", "",
                  "| model | condition | n | completion mean | 0/1/2/3/4 | grasp | lift | pour | place | drops | "
                  "start delay s (median) | duration s (median, success) | grasp attempts (mean) | end reasons |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        cells = {}
        for row in scored:
            cells.setdefault((row["model"], row["preset"]), []).append(row)
        for (model, preset), rows in sorted(cells.items()):
            completion = [int(float(r["completion"])) for r in rows]
            histogram = "/".join(str(completion.count(c)) for c in range(5))
            reached = {k: sum(r.get(f"stage_{k}") == "True" for r in rows) for k in ("grasp", "lift", "pour", "place")}
            drops = sum(r.get("drop") == "True" for r in rows)
            delay = [v for v in (f(r, "start_delay_s") for r in rows) if v is not None]
            duration = [v for v in (f(r, "duration_s") for r in rows if r.get("outcome") == "success") if v is not None]
            attempts = [v for v in (f(r, "grasp_attempts") for r in rows) if v is not None]
            reasons: dict[str, int] = {}
            for r in rows:
                reasons[r.get("end_reason") or "?"] = reasons.get(r.get("end_reason") or "?", 0) + 1
            lines.append(f"| {model} | {preset} | {len(rows)} | {statistics.mean(completion):.2f} | {histogram} | "
                         + " | ".join(str(reached[k]) for k in ("grasp", "lift", "pour", "place"))
                         + f" | {drops} | {_median(delay)} | {_median(duration)} | "
                         + (f"{statistics.mean(attempts):.1f}" if attempts else "-") + " | "
                         + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + " |")
    return "\n".join(lines)


def runner_command(row: dict[str, str], prompt: str, log_path: Path, pour_deg: float) -> str:
    if model_profile(row["model"]) == "bc2f_force":
        return f"fr3-pi05-run --checkpoint bc2f_force --prompt '{prompt}' --execute --auto-stop --trial-log {log_path}"
    return (f"fr3-pi05-run --checkpoint bluecup_force --prompt '{prompt}' --execute "
            f"--fault {row['preset']} --fault-trigger {row['trigger']} --pour-deg {pour_deg:g} "
            f"--trial-log {log_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bluecup_force real-robot trials: schedule, record, summarize")
    parser.add_argument("--root", type=Path, default=Path("trials"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--session", required=True)
    p.add_argument("--models", nargs="+", default=list(MODEL_CHECKPOINTS))
    p.add_argument("--conditions", nargs="+", default=["none", "cams_absent", "cams_black"])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--block-reps", type=int, default=5, help="repetitions per model block (one server start)")
    p.add_argument("--trigger", default="pour", choices=("pour", "always"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing schedule")

    p = sub.add_parser("next")
    p.add_argument("--session", required=True)
    p.add_argument("--prompt", default="pour lillet into the blue cup")
    p.add_argument("--pour-deg", type=float, default=60.0)

    p = sub.add_parser("record")
    p.add_argument("--session", required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--rollout-log", type=Path)
    for name in ("cup-before", "cup-after", "bottle-before", "bottle-after"):
        p.add_argument(f"--{name}", type=float)
    p.add_argument("--scale-port")
    p.add_argument("--target-g", type=float)
    p.add_argument("--tolerance-g", type=float, default=15.0)
    p.add_argument("--spill-tolerance-g", type=float, default=5.0)
    p.add_argument("--min-poured-g", type=float,
                   help="without --target-g: success needs at least this much in the cup (scope-v5 round: 20)")
    p.add_argument("--outcome", choices=OUTCOMES, help="override the automatic classification")
    p.add_argument("--notes", default="")
    p.add_argument("--force-model", help="force_model.json for the stage scores (default: fr3_train/bc2f/force_model.json)")

    p = sub.add_parser("score", help="print the automatic stage score of a rollout log (nothing is recorded)")
    p.add_argument("rollout_log", type=Path)
    p.add_argument("--force-model")

    p = sub.add_parser("summary")
    p.add_argument("--session", required=True)
    p.add_argument("--target-g", type=float)
    p.add_argument("--tolerance-g", type=float, default=15.0)
    p.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "score":
        score = score_log(args.rollout_log, args.force_model)
        if score is None:
            print(f"No per-step recording next to {args.rollout_log} (expected {args.rollout_log.with_suffix('.npz')})")
            return 1
        print(describe_score(score))
        print(json.dumps(score, indent=1))
        return 0
    directory = session_dir(args.root, args.session)
    schedule_path, results_path = directory / "schedule.csv", directory / "results.csv"

    if args.command == "plan":
        if schedule_path.exists() and not args.force:
            parser.error(f"{schedule_path} exists; pass --force to overwrite")
        unknown = [m for m in args.models if m not in MODEL_CHECKPOINTS]
        if unknown:
            parser.error(f"unknown model(s) {unknown}; expected some of {list(MODEL_CHECKPOINTS)}")
        from fr3_pi05.faults import FAULT_PRESETS

        bad = [c for c in args.conditions if c not in FAULT_PRESETS]
        if bad:
            parser.error(f"unknown condition(s) {bad}; expected some of {sorted(FAULT_PRESETS)}")
        if any(model_profile(m) == "bc2f_force" for m in args.models) and args.conditions != ["none"]:
            parser.error("bc2f models take every input at deployment: use --conditions none")
        rows = make_schedule(args.models, args.conditions, args.reps, args.block_reps, args.seed, args.trigger)
        _write_csv(schedule_path, SCHEDULE_FIELDS, rows)
        blocks = len({r["block"] for r in rows})
        print(f"{len(rows)} trials in {blocks} model blocks -> {schedule_path}")
        return 0

    schedule, results = _read_csv(schedule_path), _read_csv(results_path)
    if args.command == "next":
        row = next_trial(schedule, results)
        if row is None:
            print("All trials recorded.")
            return 0
        current = next((r for r in schedule if r["trial_id"] == row["trial_id"]), row)
        previous = [r for r in schedule if r["trial_id"] in {x["trial_id"] for x in results}]
        if not previous or previous[-1]["model"] != current["model"]:
            print(f">>> (re)start the GPU server: start_checkpoint.sh {model_profile(current['model'])} "
                  f"/mnt/data/yurui/models/{MODEL_CHECKPOINTS[current['model']]}")
        log_path = directory / "logs" / f"{row['trial_id']}.json"
        print(f"trial {row['trial_id']}: model={row['model']} condition={row['preset']} trigger={row['trigger']} rep={row['rep']}")
        print(runner_command(row, args.prompt, log_path, args.pour_deg))
        thresholds = " --min-poured-g 20 --spill-tolerance-g 10" if model_profile(row["model"]) == "bc2f_force" else ""
        print(f"then: fr3-pi05-trials record --session {args.session} --trial {row['trial_id']} --rollout-log {log_path}"
              f"{thresholds}")
        return 0

    if args.command == "record":
        row = next((r for r in schedule if int(r["trial_id"]) == args.trial), None)
        if row is None:
            parser.error(f"trial {args.trial} is not in {schedule_path}")
        if any(int(r["trial_id"]) == args.trial for r in results):
            parser.error(f"trial {args.trial} is already recorded in {results_path}")
        log: dict[str, Any] = {}
        if args.rollout_log and args.rollout_log.exists():
            log = json.loads(args.rollout_log.read_text(encoding="utf-8"))
        score = score_log(args.rollout_log, args.force_model) if log else None
        if score is not None:
            print(f"trial {args.trial} (end: {log.get('end_reason')}): {describe_score(score)}")
        elif model_profile(row["model"]) == "bc2f_force":
            print("WARNING: no per-step recording for this rollout; stages are not scored")
        cup_before = _weight(args.cup_before, "cup before (tared: 0)", args.scale_port)
        cup_after = _weight(args.cup_after, "cup after", args.scale_port)
        bottle_before = _weight(args.bottle_before, "bottle before", args.scale_port)
        bottle_after = _weight(args.bottle_after, "bottle after", args.scale_port)
        poured = cup_after - cup_before if cup_after is not None and cup_before is not None else None
        spilled = (bottle_before - bottle_after) - poured if None not in (bottle_before, bottle_after, poured) else None
        if args.outcome:
            outcome = args.outcome
        elif score is not None and args.target_g is None:
            from fr3_pi05.autoscore import outcome as staged_outcome

            outcome = staged_outcome(score, poured, spilled, log.get("end_reason"), args.min_poured_g,
                                     args.spill_tolerance_g)
        else:
            outcome = classify(poured, spilled, log, args.target_g, args.tolerance_g, args.spill_tolerance_g,
                               args.min_poured_g)
        pour = log.get("pour", {})
        results.append({
            **{k: row[k] for k in ("trial_id", "model", "preset", "trigger", "rep")},
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_reason": log.get("end_reason", ""), "duration_s": log.get("duration_s", ""),
            "pour_started": pour.get("started", ""), "pour_steps": pour.get("steps", ""),
            "max_tilt_deg": pour.get("max_tilt_deg", ""), "fault_active_steps": log.get("fault_active_steps", ""),
            "gripper_closed_empty": log.get("gripper_closed_empty", ""),
            "server_model": log.get("server_model", ""), "bundle_step": log.get("bundle_step", ""),
            "cup_before_g": cup_before, "cup_after_g": cup_after, "bottle_before_g": bottle_before,
            "bottle_after_g": bottle_after, "poured_g": poured, "spilled_g": spilled,
            "outcome": outcome, "notes": args.notes,
            "inference_ms_median": log.get("inference_ms_median", ""),
            **score_columns(score),
        })
        if log and log.get("fault_preset") not in (None, row["preset"]):
            print(f"WARNING: rollout log says fault {log.get('fault_preset')!r}, schedule says {row['preset']!r}")
        if log and log.get("server_model") and str(log.get("server_model")) != bundle_model_name(row["model"]):
            print(f"WARNING: server reported model {log.get('server_model')!r}, schedule says {row['model']!r} "
                  f"({bundle_model_name(row['model'])})")
        _write_csv(results_path, RESULT_FIELDS, results)
        print(f"trial {args.trial}: outcome={outcome} poured={poured} spilled={spilled} -> {results_path}")
        return 0

    text = summarize(results, args.target_g, args.tolerance_g)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
