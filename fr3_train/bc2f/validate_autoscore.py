"""Check fr3_pi05.autoscore on the blue_cup_2 demonstrations: every demo must score 4/4, and cutting a demo short
must lose exactly the stages after the cut.

Each demo is replayed as the runner would record it: control steps at the 15 Hz frames (joint positions,
gripper, PourDetector replayed causally) plus the raw ~1 kHz arm stream. The demos stop within a second of the
release, so the last frame is held for 1 s more (the runner's auto-stop waits 2 s after the release).

    PYTHONPATH=. python fr3_train/bc2f/validate_autoscore.py --dataset <lerobot>/fitz0401/franka_pour_lillet_blue_cup_2_force
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from fr3_pi05.autoscore import GRASP_S, SUSTAIN_S, load_force_model, score_steps, step_wrench
from fr3_pi05.faults import PourDetector


def demo_arrays(dataset: str, path: str, hold_s: float = 1.0) -> dict[str, np.ndarray]:
    frames = pd.read_parquet(path, columns=["joint_position", "gripper_position", "timestamp", "episode_index"])
    ep = int(frames.episode_index.iloc[0])
    raw = np.load(os.path.join(dataset, "raw_force", f"episode_{ep:06d}.npz"))
    q = np.stack(frames.joint_position.to_numpy()).astype(float)
    g = np.asarray([np.ravel(x)[0] for x in frames.gripper_position], float)
    t = frames.timestamp.to_numpy(float)
    time_sec = raw["robot_timestamp"].astype(float)
    if len(time_sec) != len(t):
        raise ValueError(f"episode {ep}: {len(t)} frames but {len(time_sec)} frame times in raw_force")
    pad = int(round(hold_s * 15))
    t = np.r_[t, t[-1] + np.arange(1, pad + 1) / 15]
    time_sec = np.r_[time_sec, time_sec[-1] + np.arange(1, pad + 1) / 15]
    q = np.vstack([q, np.repeat(q[-1:], pad, axis=0)])
    g = np.r_[g, np.repeat(g[-1], pad)]
    detector = PourDetector()
    pouring, tilt = np.zeros(len(t), bool), np.zeros(len(t))
    for i in range(len(t)):
        pouring[i] = detector.update(q[i], g[i])
        tilt[i] = detector.tilt_deg
    raw_force = np.concatenate([raw["raw_force_robot_timestamp"][:, None].astype(float),
                                raw["raw_force_joint_position"].astype(float),
                                raw["raw_force_joint_torque"].astype(float)], axis=1)
    return {"episode": ep, "steps_t": t, "steps_time_sec": time_sec, "steps_q": q, "steps_gripper": g,
            "steps_pouring": pouring, "steps_tilt": tilt, "raw_force": raw_force}


def score_cut(a: dict, wrench: np.ndarray, stop: int) -> dict:
    return score_steps(a["steps_t"][:stop], a["steps_q"][:stop], a["steps_gripper"][:stop],
                       a["steps_pouring"][:stop], a["steps_tilt"][:stop], wrench[:stop])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force-model", default=None)
    ap.add_argument("--out", default=None, help="write the per-episode table as JSON")
    args = ap.parse_args(argv)
    model = load_force_model(args.force_model)
    if model is None:
        raise FileNotFoundError("force model / fr3_train/force_features.py not found")
    weights = {}
    weights_path = os.path.join(args.dataset, "meta", "episode_weights.jsonl")
    if os.path.exists(weights_path):
        for line in open(weights_path, encoding="utf-8"):
            row = json.loads(line)
            weights[row["episode_index"]] = row
    rows, failures = [], []
    for path in sorted(glob.glob(os.path.join(args.dataset, "data", "*", "episode_*.parquet"))):
        a = demo_arrays(args.dataset, path)
        ep = a["episode"]
        wrench = step_wrench(a["raw_force"], a["steps_time_sec"], model)
        full = score_cut(a, wrench, len(a["steps_t"]))
        if full["completion"] != 4 or full["drop"]:
            failures.append(f"episode {ep}: completion {full['completion']}, stages {full['stages']}, drop {full['drop']}")
        # Cut checks: the stages after the cut must be lost, the ones before kept.
        # A stage counts once its evidence is complete (grasp: 1 s closed; lift and release: 0.5 s), so a cut just
        # before the lift must still leave the grasp its full second; it then ends inside the lift's 0.5 s.
        t = a["steps_t"]
        cuts = {0: full["t_grasp_s"], 1: full["t_lift_s"], 2: full["t_pour_s"], 3: full["t_release_s"]}
        if full["t_lift_s"] is not None:
            cuts[1] = max(full["t_lift_s"], full["t_grasp_s"] + GRASP_S + 1.5 / 15)
            assert cuts[1] < full["t_lift_s"] + SUSTAIN_S
        for expect, at_s in cuts.items():
            if at_s is None:
                continue
            stop = int(np.searchsorted(t, t[0] + at_s - 0.5 / 15))  # steps before the cut (times are rounded)
            got = score_cut(a, wrench, max(stop, 2))["completion"]
            if got != expect:
                failures.append(f"episode {ep}: cut before stage {expect + 1} scores {got}, expected {expect}")
        row = {"episode": ep, "completion": full["completion"], "drop": full["drop"],
               **{k: full[k] for k in ("start_delay_s", "t_grasp_s", "t_lift_s", "t_pour_s", "t_release_s",
                                       "pour_s", "pour_bouts", "max_tilt_deg", "grasp_attempts", "empty_closes",
                                       "longest_idle_s", "path_rad", "held_mass_g", "max_hold_fz_N", "max_force_N")},
               "place_dz_cm": full["place"]["dz_cm"], "place_xy_cm": full["place"]["xy_cm"],
               "fz_after_release_N": full["place"]["fz_after_N"],
               "bottle_g": weights.get(ep, {}).get("bottle_before_g")}
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.to_string(index=False))
    if table["held_mass_g"].notna().any() and table["bottle_g"].notna().any():
        ok = table.dropna(subset=["held_mass_g", "bottle_g"])
        print(f"\nheld mass vs bottle on the scale: corr {np.corrcoef(ok.held_mass_g, ok.bottle_g)[0, 1]:.3f}, "
              f"mean difference {np.mean(ok.held_mass_g - ok.bottle_g):+.0f} g")
    if args.out:
        table.to_json(args.out, orient="records", indent=1)
    if failures:
        print("\nFAIL\n" + "\n".join(failures))
        return 1
    print(f"\nPASS: {len(rows)} demos score 4/4 with no drop; every cut loses exactly the later stages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
