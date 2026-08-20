import unittest
from unittest.mock import patch

import numpy as np

from fr3_demo.kinematics import JOINT_UPPER, WorkspaceBounds
from fr3_pi05.policy import (
    SafetyViolation,
    _resize,
    build_droid_observation,
    predict_joint_path,
    safe_joint_velocity,
    validate_action_chunk,
)


class Pi05PolicyTest(unittest.TestCase):
    Q = np.array([-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031])
    WORKSPACE = WorkspaceBounds((0.10, -0.60, 0.05), (0.80, 0.60, 1.00))

    @patch("fr3_pi05.policy._resize", side_effect=lambda image: image[:224, :224])
    def test_observation_matches_openpi_droid_keys(self, _resize) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = build_droid_observation(image, image, self.Q, 0.7, "pick up the cube")

        self.assertEqual(
            set(observation),
            {
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
                "observation/joint_position",
                "observation/gripper_position",
                "prompt",
            },
        )
        self.assertEqual(observation["observation/joint_position"].shape, (7,))
        self.assertEqual(observation["observation/gripper_position"].shape, (1,))

    def test_current_pi05_horizon_is_accepted_and_clipped(self) -> None:
        response = {"actions": np.full((15, 8), 2.0)}
        actions = validate_action_chunk(response, minimum_horizon=8)
        self.assertEqual(actions.shape, (15, 8))
        self.assertTrue(np.all(actions == 1.0))

    def test_droid_aspect_is_padded_without_stretching(self) -> None:
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        resized = _resize(image)

        self.assertEqual(resized.shape, (224, 224, 3))
        active_rows = np.flatnonzero(np.any(resized, axis=(1, 2)))
        self.assertEqual(len(active_rows), 126)
        self.assertEqual(active_rows[0], 49)
        self.assertEqual(active_rows[-1], 174)

    def test_malformed_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[N, 8\]"):
            validate_action_chunk({"actions": np.zeros((15, 7))}, minimum_horizon=8)
        with self.assertRaisesRegex(ValueError, "fewer"):
            validate_action_chunk({"actions": np.zeros((4, 8))}, minimum_horizon=8)

    def test_velocity_is_uniformly_limited(self) -> None:
        velocity = safe_joint_velocity(
            self.Q,
            np.array([1.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0]),
            duration=1 / 15,
            maximum=0.2,
            joint_margin=0.1,
            workspace=self.WORKSPACE,
        )
        self.assertAlmostEqual(float(np.max(np.abs(velocity))), 0.2)
        self.assertAlmostEqual(velocity[1] / velocity[0], -0.5)

    def test_joint_margin_stops_outward_velocity(self) -> None:
        q = self.Q.copy()
        q[0] = JOINT_UPPER[0] - 0.1
        velocity = safe_joint_velocity(
            q,
            np.array([0.2, 0, 0, 0, 0, 0, 0]),
            duration=1 / 15,
            maximum=0.2,
            joint_margin=0.1,
            workspace=WorkspaceBounds((-2, -2, -2), (2, 2, 2)),
        )
        np.testing.assert_array_equal(velocity, np.zeros(7))

    def test_workspace_violation_stops_action(self) -> None:
        with self.assertRaises(SafetyViolation):
            safe_joint_velocity(
                self.Q,
                np.zeros(7),
                duration=1 / 15,
                maximum=0.2,
                joint_margin=0.1,
                workspace=WorkspaceBounds((10, 10, 10), (11, 11, 11)),
            )

    def test_prediction_has_initial_and_horizon_states(self) -> None:
        path = predict_joint_path(
            self.Q,
            np.zeros((15, 8)),
            horizon=8,
            control_hz=15,
            maximum=0.2,
            joint_margin=0.1,
            workspace=self.WORKSPACE,
        )
        self.assertEqual(path.shape, (9, 7))
        np.testing.assert_array_equal(path[0], self.Q)


if __name__ == "__main__":
    unittest.main()
