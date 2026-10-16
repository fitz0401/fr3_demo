"""Sensor-fault injection for the bluecup_force robustness trials.

Two pieces, both pure numpy so they can be unit-tested without the robot:

* ``PourDetector`` decides online whether the robot is pouring, with the SAME rule that labelled the
  training data (openpi scripts/phase_labels.py): while the gripper holds the bottle, the bottle axis
  is the tool-frame axis most aligned with world +z one second after the grasp, and the robot is
  pouring when that axis is tilted past a threshold. Training used max(45 deg, 0.7 x the episode's
  maximum tilt); on the 21 blue-cup demonstrations that came to 59-73 deg (median 66), so the online
  default is 60 deg. The forward kinematics are fr3_demo.kinematics, identical to the labeller's.

* ``FaultSchedule`` turns a preset name ("cams_absent", "force_absent", ...) and a trigger ("pour",
  "always", "never") into the ``fault/...`` fields that fr3_train/serve_bluecup.py applies. The
  presets are the conditions of the offline robustness table, so every real-robot cell has an
  offline counterpart.

Runner integration (see fr3_train/bluecup/DEPLOY_PLAN.md): call ``detector.update(q, gripper)`` once
per control step with the current joint positions and gripper reading, then pass
``schedule.fields(detector)`` as ``fault=`` to ``build_droid_observation``. Log ``detector.state()``
with every inference so each trial records when the fault was actually active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fr3_demo.kinematics import forward_kinematics

CAMERA_KEYS = ("exterior_image_1_left", "wrist_image_left", "exterior_image_2_left")

FAULT_PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    # Declared camera loss: picture black AND image_mask False (what the dropout arms trained on).
    "cams_absent": {"fault/absent_cameras": list(CAMERA_KEYS)},
    # Undeclared blackout: pictures black, the model is not told.
    "cams_black": {"fault/black_cameras": list(CAMERA_KEYS)},
    "exterior1_absent": {"fault/absent_cameras": ["exterior_image_1_left"]},
    "wrist_absent": {"fault/absent_cameras": ["wrist_image_left"]},
    "exterior2_absent": {"fault/absent_cameras": ["exterior_image_2_left"]},
    # Declared force loss: torque zeroed and presence flag 0.
    "force_absent": {"fault/force": "absent"},
    # Sensor died silently: torque zeroed, flag stays 1.
    "force_dead_undeclared": {"fault/force": "dead_undeclared"},
}
TRIGGERS = ("pour", "always", "never")


@dataclass
class PourDetector:
    """Online pour-phase detector; mirrors phase_labels.label_episode causally."""

    closed_below: float = 0.9  # gripper reading below this = holding (blue-cup data: 0.784 closed, 1.0 open)
    grasp_seconds: float = 1.0
    control_hz: float = 15.0
    pour_deg: float = 60.0
    _closed_steps: int = 0
    _axis: np.ndarray | None = None
    tilt_deg: float = 0.0
    holding: bool = False
    pouring: bool = False
    pour_started: bool = False  # latched: a pour has begun in this episode
    pour_steps: int = 0
    max_tilt_deg: float = 0.0
    history: list[tuple[float, bool, bool]] = field(default_factory=list)

    def reset(self) -> None:
        self._closed_steps, self._axis = 0, None
        self.tilt_deg, self.holding, self.pouring = 0.0, False, False
        self.pour_started, self.pour_steps, self.max_tilt_deg = False, 0, 0.0
        self.history.clear()

    def update(self, joint_position: np.ndarray, gripper_position: float) -> bool:
        q = np.asarray(joint_position, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("PourDetector needs seven finite joint positions")
        rotation = forward_kinematics(q)[:3, :3]
        self.holding = float(gripper_position) < self.closed_below
        if not self.holding:
            # Released (or not yet grasped): forget the bottle axis; the next grasp re-learns it.
            self._closed_steps, self._axis = 0, None
            self.tilt_deg, self.pouring = 0.0, False
        else:
            self._closed_steps += 1
            if self._axis is None and self._closed_steps >= max(1, int(round(self.grasp_seconds * self.control_hz))):
                # Tool-frame axis (with sign) most aligned with world +z, one second into the grasp.
                candidates = [(sign * rotation[2, axis], axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
                _, axis, sign = max(candidates)
                self._axis = np.zeros(3)
                self._axis[axis] = sign
            if self._axis is not None:
                up = rotation @ self._axis
                self.tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
                self.max_tilt_deg = max(self.max_tilt_deg, self.tilt_deg)
            self.pouring = self._axis is not None and self.tilt_deg >= self.pour_deg
        if self.pouring:
            self.pour_started = True
            self.pour_steps += 1
        self.history.append((self.tilt_deg, self.holding, self.pouring))
        return self.pouring

    def state(self) -> dict[str, Any]:
        return {
            "tilt_deg": round(self.tilt_deg, 2),
            "holding": self.holding,
            "pouring": self.pouring,
            "pour_started": self.pour_started,
            "pour_steps": self.pour_steps,
            "max_tilt_deg": round(self.max_tilt_deg, 2),
        }


@dataclass(frozen=True)
class FaultSchedule:
    preset: str = "none"
    trigger: str = "pour"

    def __post_init__(self) -> None:
        if self.preset not in FAULT_PRESETS:
            raise ValueError(f"Unknown fault preset {self.preset!r}; expected one of {sorted(FAULT_PRESETS)}")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"Unknown fault trigger {self.trigger!r}; expected one of {TRIGGERS}")

    def active(self, detector: PourDetector | None) -> bool:
        if self.preset == "none" or self.trigger == "never":
            return False
        if self.trigger == "always":
            return True
        return bool(detector is not None and detector.pouring)

    def fields(self, detector: PourDetector | None) -> dict[str, Any]:
        """The fault/... keys for this step; empty when the fault is not active."""
        if not self.active(detector):
            return {}
        return {key: (list(value) if isinstance(value, list) else value) for key, value in FAULT_PRESETS[self.preset].items()}
