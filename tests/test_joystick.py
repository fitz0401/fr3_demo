import unittest

from fr3_demo.joystick import JoystickSnapshot, shaped_axis
from fr3_demo.teleop import Mapping, joystick_tool_y, joystick_tool_z, joystick_twist, lock_dpad_axis


class JoystickMappingTest(unittest.TestCase):
    def test_deadzone(self) -> None:
        self.assertEqual(shaped_axis(0.1, deadzone=0.12), 0.0)
        self.assertEqual(shaped_axis(-0.1, deadzone=0.12), 0.0)
        self.assertAlmostEqual(shaped_axis(1.0), 1.0)
        self.assertAlmostEqual(shaped_axis(-1.0), -1.0)

    @staticmethod
    def snapshot(axes: tuple[float, ...], pressed: tuple[int, ...] = ()) -> JoystickSnapshot:
        buttons = tuple(index in pressed for index in range(12))
        return JoystickSnapshot(
            axes=axes,
            buttons=buttons,
            press_counts=(0,) * 12,
            connected=True,
            name="test",
            timestamp=0.0,
        )

    def test_joy_listener_stick_mapping(self) -> None:
        snapshot = self.snapshot((0.5, -1.0, 0.5, 1.0, 0.0, 0.0))
        twist = joystick_twist(snapshot, Mapping(), 0.1, 0.4, deadzone=0.0)
        self.assertAlmostEqual(twist[0], 0.1)
        self.assertLess(twist[1], 0.0)
        self.assertEqual(twist[2], 0.0)
        self.assertGreater(twist[3], 0.0)
        self.assertLess(twist[4], 0.0)
        self.assertEqual(twist[5], 0.0)

    def test_joy_listener_trigger_and_bumper_mapping(self) -> None:
        triggers = self.snapshot((0.0,) * 6, pressed=(6, 7))
        trigger_twist = joystick_twist(triggers, Mapping(), 0.1, 0.4, deadzone=0.0)
        self.assertAlmostEqual(trigger_twist[2], -0.1)
        self.assertAlmostEqual(trigger_twist[5], -0.4)

        bumpers = self.snapshot((0.0,) * 6, pressed=(4, 5))
        bumper_twist = joystick_twist(bumpers, Mapping(), 0.1, 0.4, deadzone=0.0)
        self.assertAlmostEqual(bumper_twist[2], 0.1)
        self.assertAlmostEqual(bumper_twist[5], 0.4)

    def test_dpad_vertical_maps_to_tool_z(self) -> None:
        dpad_up = self.snapshot((0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
        dpad_down = self.snapshot((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        self.assertAlmostEqual(joystick_tool_z(dpad_up, Mapping(), 0.1, deadzone=0.0), 0.1)
        self.assertAlmostEqual(joystick_tool_z(dpad_down, Mapping(), 0.1, deadzone=0.0), -0.1)

    def test_dpad_horizontal_maps_to_tool_y(self) -> None:
        dpad_left = self.snapshot((0.0, 0.0, 0.0, 0.0, -1.0, 0.0))
        dpad_right = self.snapshot((0.0, 0.0, 0.0, 0.0, 1.0, 0.0))

        self.assertAlmostEqual(joystick_tool_y(dpad_left, Mapping(), 0.1, deadzone=0.0), -0.1)
        self.assertAlmostEqual(joystick_tool_y(dpad_right, Mapping(), 0.1, deadzone=0.0), 0.1)

    def test_dpad_locks_to_first_active_axis(self) -> None:
        tool_y, tool_z, active = lock_dpad_axis(-0.1, 0.0, None)
        self.assertEqual((tool_y, tool_z, active), (-0.1, 0.0, "y"))

        tool_y, tool_z, active = lock_dpad_axis(-0.1, 0.1, active)
        self.assertEqual((tool_y, tool_z, active), (-0.1, 0.0, "y"))

        tool_y, tool_z, active = lock_dpad_axis(0.0, 0.1, active)
        self.assertEqual((tool_y, tool_z, active), (0.0, 0.1, "z"))

    def test_dpad_fresh_diagonal_uses_y_only(self) -> None:
        tool_y, tool_z, active = lock_dpad_axis(-0.1, 0.1, None)

        self.assertEqual((tool_y, tool_z, active), (-0.1, 0.0, "y"))

    def test_gripper_and_quit_buttons_do_not_conflict_with_motion(self) -> None:
        mapping = Mapping()
        self.assertEqual(mapping.close_button, 2)  # A
        self.assertEqual(mapping.open_button, 1)  # B
        self.assertEqual(mapping.home_button, 9)  # Menu
        self.assertEqual(mapping.quit_button, 8)  # Back
        self.assertEqual(mapping.record_button, 3)  # X
        self.assertEqual(mapping.tool_y_axis, 4)  # D-pad horizontal
        self.assertEqual(mapping.tool_z_axis, 5)  # D-pad vertical


if __name__ == "__main__":
    unittest.main()
