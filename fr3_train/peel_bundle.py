"""Round-2 peel bundle helpers (2026-11-13), numpy + json only so the unit tests run without openpi / jax.

Used by serve_peel.py and the zmq peel loader."""

from __future__ import annotations

import json
import pathlib

import numpy as np

DEFAULT_ASSET_ID = "fitz0401/fr3_peel_mix"
GRIPPER_DIMS = (7, 15, 23)  # gripper entries of the State and History segments


def bundle_asset_id(checkpoint_dir: str) -> str:
    """The norm-stats asset the bundle ships (assets/<org>/<name>/norm_stats.json): round 1 fitz0401/fr3_peel_mix,
    round 2 fitz0401/fr3_peel2_mix. Falls back to DEFAULT_ASSET_ID when there is not exactly one."""
    found = sorted(p.parent.relative_to(pathlib.Path(checkpoint_dir) / "assets").as_posix()
                   for p in (pathlib.Path(checkpoint_dir) / "assets").glob("*/*/norm_stats.json"))
    return found[0] if len(found) == 1 else DEFAULT_ASSET_ID


def pinned_gripper(checkpoint_dir: str, q01: np.ndarray, q99: np.ndarray) -> float:
    """The gripper value fed to the model whatever reading arrives (the peeler stays clamped).

    Round 1 (fr3_peel_mix): every training frame read 0.12334802, so the norm stats hold one constant and that is fed.
    Round 2 (fr3_peel2_mix, new mount): the reading is constant within an episode but one of three quantisation steps
    (0.2863 / 0.2907 / 0.2952) from grasp to grasp, so the bundle names the value (BUNDLE.json `held_gripper`, the most
    frequent training value) and it must lie inside the training range."""
    held = q01[list(GRIPPER_DIMS)]
    if np.all(held == q99[list(GRIPPER_DIMS)]) and np.all(held == held[0]):
        return float(held[0])
    path = pathlib.Path(checkpoint_dir) / "BUNDLE.json"
    value = json.loads(path.read_text()).get("held_gripper") if path.is_file() else None
    if value is None:
        raise ValueError(f"gripper norm stats are not one constant ({q01[list(GRIPPER_DIMS)]} / {q99[list(GRIPPER_DIMS)]}) "
                         f"and {path} names no held_gripper: the server cannot choose the pinned value")
    value = float(value)
    if not np.all((q01[list(GRIPPER_DIMS)] <= value) & (value <= q99[list(GRIPPER_DIMS)])):
        raise ValueError(f"held_gripper {value} lies outside the training range {q01[list(GRIPPER_DIMS)]} .. "
                         f"{q99[list(GRIPPER_DIMS)]}")
    return value
