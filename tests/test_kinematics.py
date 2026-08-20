import unittest

import numpy as np

from fr3_demo.kinematics import (
    JOINT_LOWER,
    JOINT_UPPER,
    WorkspaceBounds,
    forward_kinematics,
    geometric_jacobian,
    link_positions,
    resolved_rate_step,
)


class KinematicsTest(unittest.TestCase):
    Q = np.array([-0.08777529, -0.49271475, -0.07260183, -2.19538373, -0.07789254, 1.83422679, 0.64515432])

    def test_forward_kinematics_matches_bamboo_measurement(self) -> None:
        expected = np.array(
            [
                [0.70024885, -0.70326587, 0.12275448, 0.40936734],
                [-0.70203795, -0.70957202, -0.06041743, -0.07775553],
                [0.12959266, -0.04387107, -0.99059632, 0.58676608],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        actual = forward_kinematics(self.Q)
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=2e-7)
        # The sample was measured under impedance control; allow sub-mm tracking error.
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=2e-4)

    def test_linear_jacobian_matches_finite_difference(self) -> None:
        jacobian = geometric_jacobian(self.Q)
        epsilon = 1e-7
        origin = forward_kinematics(self.Q)[:3, 3]
        numeric = np.empty((3, 7))
        for index in range(7):
            perturbed = self.Q.copy()
            perturbed[index] += epsilon
            numeric[:, index] = (forward_kinematics(perturbed)[:3, 3] - origin) / epsilon
        np.testing.assert_allclose(jacobian[:3], numeric, atol=1e-6)

    def test_resolved_step_obeys_joint_limits_and_velocity_cap(self) -> None:
        target, velocity = resolved_rate_step(
            self.Q,
            np.array([0.05, -0.03, 0.02, 0.1, -0.1, 0.2]),
            0.12,
            max_joint_velocity=0.30,
            workspace=WorkspaceBounds(),
        )
        self.assertLessEqual(float(np.max(np.abs(velocity))), 0.30 + 1e-12)
        self.assertTrue(np.all(target >= JOINT_LOWER + 0.08 - 1e-12))
        self.assertTrue(np.all(target <= JOINT_UPPER - 0.08 + 1e-12))

    def test_workspace_violation_is_rejected(self) -> None:
        pose = forward_kinematics(self.Q)
        tiny = WorkspaceBounds(tuple(pose[:3, 3] - 1e-6), tuple(pose[:3, 3] + 1e-6))
        with self.assertRaisesRegex(ValueError, "workspace"):
            resolved_rate_step(self.Q, np.array([0.1, 0, 0, 0, 0, 0]), 0.2, workspace=tiny)

    def test_link_positions_include_base_and_eef(self) -> None:
        positions = link_positions(self.Q)
        self.assertEqual(positions.shape, (9, 3))
        np.testing.assert_array_equal(positions[0], np.zeros(3))
        np.testing.assert_allclose(positions[-1], forward_kinematics(self.Q)[:3, 3])


if __name__ == "__main__":
    unittest.main()
