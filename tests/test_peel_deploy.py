"""Peeling deployment (2026-11-06): the peel_force profile = bc2f_force + the GelSight marker summary."""

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fr3_pi05.policy import RESPONSE_INFO_KEYS, build_droid_observation
from fr3_pi05.remote.serve_pi05_zmq import warm_up
from fr3_pi05.runner import SafetyViolation, _parse, _tactile_reader, _validate_policy_contract

ROOT = Path(__file__).resolve().parents[1]
START_Q = np.array([-0.0334, -0.7522, -0.0277, -2.2872, -0.0163, 1.5795, 0.04])
PEEL_METADATA = {
    "model": "pi05_peel", "loader": "peel", "action_horizon": 16, "state_history_lags": [45, 75],
    "num_state_frames": 3, "joint_observation_shape": [3, 7], "gripper_observation_shape": [3],
    "image_observation_shape": [180, 320, 3], "proprio_history_offsets": [0, 45, 75],
    "uses_exterior2": True, "requires_raw_force": True, "requires_tactile": True, "tactile_dims": 4,
    "tasks": ["peel the lemon"], "bundle_model": "pi05_peel_splice_s0", "bundle_step": 9999,
    "gripper_input_pinned_to": 0.12334802001714706,
}


def _force_features():
    spec = importlib.util.spec_from_file_location("force_features_peel_test", ROOT / "fr3_train" / "force_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ff = _force_features()


def still_arm(t_end: float, span: float = 0.128, rate: float = 860.0) -> np.ndarray:
    t = t_end - np.arange(int(span * rate))[::-1] / rate
    tau = np.array([-0.298, -5.7, 0.642, 25.275, 1.575, 2.16, 0.173])
    return np.concatenate([t[:, None], np.tile(START_Q, (len(t), 1)), np.tile(tau, (len(t), 1))], axis=1)


@dataclass
class Reading:
    valid: bool
    reason: str = ""
    shift_px: tuple = (0.0, 0.0)
    twist_rad: float = 0.0
    spread: float = 0.0


class Monitor:
    def __init__(self, reading):
        self.reading = reading

    def latest(self):
        return self.reading


class PeelProtocolTest(unittest.TestCase):
    def test_observation_carries_tactile(self) -> None:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        obs = build_droid_observation(image, image, np.tile(START_Q, (3, 1)), np.zeros(3), "peel the lemon",
                                      exterior2_image=image, raw_force=still_arm(50.0), robot_timestamp=50.0,
                                      tactile=np.array([1.2, -0.3, 0.01, 0.0005]))
        np.testing.assert_allclose(obs["observation/tactile"], [1.2, -0.3, 0.01, 0.0005], rtol=1e-6)
        for bad in (np.zeros(2), np.array([np.nan, 0, 0, 0])):
            with self.assertRaisesRegex(ValueError, "tactile must be"):
                build_droid_observation(image, image, np.tile(START_Q, (3, 1)), np.zeros(3), "x", tactile=bad)
        self.assertIn("tactile", RESPONSE_INFO_KEYS)

    def test_runner_profile_holds_the_tool_and_uses_the_history_contract(self) -> None:
        with redirect_stdout(io.StringIO()):
            args = _parse(["--checkpoint", "peel_force", "--open-loop-horizon", "3", "--prefetch-actions", "2"])
        self.assertEqual(args.open_loop_horizon, 16)
        self.assertEqual(args.prefetch_actions, 0)
        self.assertEqual(args.policy_port, 8005)
        self.assertTrue(args.no_gripper)

    def test_runner_requires_exact_peel_metadata(self) -> None:
        with redirect_stdout(io.StringIO()):
            _validate_policy_contract("peel_force", dict(PEEL_METADATA))
        for key, value in (("requires_tactile", False), ("loader", "bc2f"), ("tactile_dims", 2)):
            metadata = dict(PEEL_METADATA, **{key: value})
            with self.assertRaisesRegex(RuntimeError, "peel_force server"):
                _validate_policy_contract("peel_force", metadata)

    def test_server_warm_up_sends_tactile_and_a_consistent_raw_window(self) -> None:
        seen = {}

        class Policy:
            def infer(self, obs):
                seen.update(obs)
                return {"actions": np.zeros((16, 8))}

        warm_up(Policy(), "peel the lemon", (0, 45, 75), 16, True, False, True, True)
        np.testing.assert_array_equal(seen["observation/tactile"], np.zeros(4))
        model = ff.ForceModel.from_json(ROOT / "fr3_train" / "peel" / "force_model.json")
        out = ff.window_features(model, seen["observation/raw_force"], seen["observation/robot_timestamp"])
        self.assertEqual(out["valid_fraction"], 1.0)

    def test_tactile_reader_passes_valid_readings_and_stops_on_invalid_ones(self) -> None:
        read = _tactile_reader(Monitor(Reading(True, shift_px=(1.5, -0.2), twist_rad=0.02, spread=0.001)))
        np.testing.assert_allclose(read(), [1.5, -0.2, 0.02, 0.001], rtol=1e-6)
        read = _tactile_reader(Monitor(Reading(False, reason="marker tracking failed")))
        with self.assertRaisesRegex(SafetyViolation, "marker tracking failed"):
            read()

    def test_peel_force_model_is_the_verified_peeler_model(self) -> None:
        model = ff.ForceModel.from_json(ROOT / "fr3_train" / "peel" / "force_model.json")
        self.assertIsNotNone(model)
        import json

        spec = json.loads((ROOT / "fr3_train" / "peel" / "force_model.json").read_text())
        self.assertEqual(spec["correction"], "artifacts/force/gravity_correction_peel.json")
        self.assertEqual(spec["verification"]["training_table"]["max_abs"], 0.0)


def _peel_bundle():
    spec = importlib.util.spec_from_file_location("peel_bundle_test", ROOT / "fr3_train" / "peel_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PeelBundleRound2Test(unittest.TestCase):
    """Round 2 (fr3_peel2_mix): the gripper reading is one of three steps, so the bundle names the pinned value."""

    def setUp(self) -> None:
        import tempfile

        self.pb = _peel_bundle()
        self.tmp = Path(tempfile.mkdtemp())

    def stats(self, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        q01, q99 = np.zeros(52), np.ones(52)
        q01[list(self.pb.GRIPPER_DIMS)], q99[list(self.pb.GRIPPER_DIMS)] = lo, hi
        return q01, q99

    def test_round1_constant_is_fed(self) -> None:
        self.assertEqual(self.pb.pinned_gripper(str(self.tmp), *self.stats(0.12334802, 0.12334802)), 0.12334802)

    def test_round2_needs_the_bundle_value_inside_the_training_range(self) -> None:
        import json

        q01, q99 = self.stats(0.2863, 0.2952)
        with self.assertRaisesRegex(ValueError, "names no held_gripper"):
            self.pb.pinned_gripper(str(self.tmp), q01, q99)
        (self.tmp / "BUNDLE.json").write_text(json.dumps({"held_gripper": 0.2863}))
        self.assertEqual(self.pb.pinned_gripper(str(self.tmp), q01, q99), 0.2863)
        (self.tmp / "BUNDLE.json").write_text(json.dumps({"held_gripper": 0.0}))
        with self.assertRaisesRegex(ValueError, "outside the training range"):
            self.pb.pinned_gripper(str(self.tmp), q01, q99)

    def test_asset_is_the_one_the_bundle_ships(self) -> None:
        self.assertEqual(self.pb.bundle_asset_id(str(self.tmp)), "fitz0401/fr3_peel_mix")
        (self.tmp / "assets" / "fitz0401" / "fr3_peel2_mix").mkdir(parents=True)
        (self.tmp / "assets" / "fitz0401" / "fr3_peel2_mix" / "norm_stats.json").write_text("{}")
        self.assertEqual(self.pb.bundle_asset_id(str(self.tmp)), "fitz0401/fr3_peel2_mix")


if __name__ == "__main__":
    unittest.main()
