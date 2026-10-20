import importlib.util
import io
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from fr3_pi05.policy import build_droid_observation
from fr3_pi05.raw_force import RawForceSampler
from fr3_pi05.remote.serve_pi05_zmq import warm_up
from fr3_pi05.runner import _parse, _validate_policy_contract
from fr3_pi05.trials import bundle_model_name, classify, main, model_profile, runner_command

ROOT = Path(__file__).resolve().parents[1]
HOME_Q = np.array([-0.047, -0.734, -0.028, -2.279, 0.002, 1.578, 0.033])


def _force_features():
    spec = importlib.util.spec_from_file_location("force_features_test", ROOT / "fr3_train" / "force_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ff = _force_features()
MODEL = ff.ForceModel.from_json(ROOT / "fr3_train" / "bc2f" / "force_model.json")


def still_arm(t_end: float, tau: np.ndarray, span: float = 0.128, rate: float = 860.0) -> np.ndarray:
    t = t_end - np.arange(int(span * rate))[::-1] / rate
    return np.concatenate([t[:, None], np.tile(HOME_Q, (len(t), 1)), np.tile(tau, (len(t), 1))], axis=1)


class ForceFeaturesTest(unittest.TestCase):
    def test_gravity_only_torque_reads_as_zero_wrench(self) -> None:
        grav, _ = MODEL.gravity_and_jacobian(HOME_Q)
        out = ff.window_features(MODEL, still_arm(50.0, grav[0]), 50.0)
        self.assertEqual(out["force"].shape, (24,))
        np.testing.assert_allclose(out["force"], 0.0, atol=1e-5)
        self.assertEqual(out["valid_fraction"], 1.0)

    def test_known_flange_force_is_recovered_in_base_axes(self) -> None:
        # A held 0.5 kg mass: the environment pulls the flange down, F_z = -4.9 N (training sign convention).
        grav, jac = MODEL.gravity_and_jacobian(HOME_Q)
        wrench = np.array([0.0, 0.0, -0.5 * 9.81, 0.0, 0.0, 0.0])
        tau = grav[0] - jac[0].T @ wrench
        out = ff.window_features(MODEL, still_arm(50.0, tau), 50.0)
        np.testing.assert_allclose(out["window"], np.tile(wrench, (4, 1)), atol=1e-4)

    def test_bins_average_only_the_causal_window(self) -> None:
        grav, jac = MODEL.gravity_and_jacobian(HOME_Q)
        raw = still_arm(50.0, grav[0])
        late = raw[:, 0] > 50.0 - 0.016  # last 16 ms = last of the 4 model bins
        raw[late, 8:15] = grav[0] - jac[0].T @ np.array([1.0, 0, 0, 0, 0, 0])
        future = still_arm(50.05, grav[0] - jac[0].T @ np.array([9.0, 0, 0, 0, 0, 0]), span=0.04)
        out = ff.window_features(MODEL, np.concatenate([raw, future]), 50.0)
        np.testing.assert_allclose(out["window"][:3, 0], 0.0, atol=1e-4)
        self.assertAlmostEqual(float(out["window"][3, 0]), 1.0, places=3)

    def test_uncovered_or_stale_windows_are_rejected(self) -> None:
        grav, _ = MODEL.gravity_and_jacobian(HOME_Q)
        with self.assertRaisesRegex(ValueError, "after the window start"):
            ff.window_features(MODEL, still_arm(50.0, grav[0], span=0.03), 50.0)
        with self.assertRaisesRegex(ValueError, "older than robot_timestamp"):
            ff.window_features(MODEL, still_arm(50.0, grav[0]), 50.03)
        sparse = still_arm(50.0, grav[0], rate=100.0)
        with self.assertRaisesRegex(ValueError, "bins hold a raw sample"):
            ff.window_features(MODEL, sparse, 50.0)

    def test_repeated_controller_states_are_dropped(self) -> None:
        grav, _ = MODEL.gravity_and_jacobian(HOME_Q)
        raw = still_arm(50.0, grav[0])
        doubled = np.concatenate([raw, raw[::3]])
        np.testing.assert_array_equal(ff.clean_raw(doubled), raw)


class FakeBamboo:
    def __init__(self, fail_after: int | None = None) -> None:
        self.t = 10.0
        self.calls = 0
        self.fail_after = fail_after

    def get_joint_states(self) -> dict:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError("bamboo went away")
        if self.calls % 5:  # every fifth poll returns the same controller state again
            self.t += 0.001
        return {"qpos": HOME_Q.tolist(), "dq": [0.0] * 7, "tau_J": [1.0] * 7, "time_sec": self.t}


class RawForceSamplerTest(unittest.TestCase):
    def test_window_holds_unique_samples_around_the_stamp(self) -> None:
        fake = FakeBamboo()
        sampler = RawForceSampler(lambda: fake, rate_hz=5000.0).start()
        try:
            deadline = time.monotonic() + 3.0
            while fake.t < 10.3 and time.monotonic() < deadline:
                time.sleep(0.01)
            stamp = fake.t - 0.05
            window = sampler.window(stamp)
            self.assertEqual(window.shape[1], 15)
            self.assertTrue(np.all(np.diff(window[:, 0]) > 0))
            self.assertLessEqual(window[0, 0] - (stamp - 0.128), 0.0011)
            self.assertGreaterEqual(window[-1, 0], stamp)
            np.testing.assert_allclose(window[:, 1:8], np.tile(HOME_Q, (len(window), 1)))
        finally:
            sampler.close()

    def test_poller_failure_reaches_the_control_thread(self) -> None:
        sampler = RawForceSampler(lambda: FakeBamboo(fail_after=20), rate_hz=5000.0).start()
        try:
            deadline = time.monotonic() + 2.0
            while sampler._error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            with self.assertRaisesRegex(RuntimeError, "poller failed"):
                sampler.check()
        finally:
            sampler.close()

    def test_slow_poller_is_refused(self) -> None:
        sampler = RawForceSampler(lambda: FakeBamboo(), rate_hz=100.0).start()
        try:
            with self.assertRaisesRegex(RuntimeError, "reaches only|did not collect|stalled"):
                sampler.wait_ready(timeout=1.5, min_span_s=0.2)
        finally:
            sampler.close()


class Bc2fProtocolTest(unittest.TestCase):
    def test_observation_carries_raw_force_and_stamp_together(self) -> None:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        raw = still_arm(50.0, np.zeros(7))
        obs = build_droid_observation(image, image, np.tile(HOME_Q, (3, 1)), np.ones(3), "pour lillet into the blue cup",
                                      exterior2_image=image, raw_force=raw, robot_timestamp=50.0)
        self.assertEqual(obs["observation/raw_force"].shape, raw.shape)
        self.assertEqual(obs["observation/robot_timestamp"], 50.0)
        with self.assertRaisesRegex(ValueError, "together"):
            build_droid_observation(image, image, np.tile(HOME_Q, (3, 1)), np.ones(3), "x", raw_force=raw)
        with self.assertRaisesRegex(ValueError, "raw_force must be"):
            build_droid_observation(image, image, np.tile(HOME_Q, (3, 1)), np.ones(3), "x",
                                    raw_force=raw[:, :8], robot_timestamp=50.0)

    def test_runner_profile_uses_history_contract(self) -> None:
        args = _parse(["--checkpoint", "bc2f_force", "--open-loop-horizon", "3", "--prefetch-actions", "2"])
        self.assertEqual(args.open_loop_horizon, 16)
        self.assertEqual(args.prefetch_actions, 0)
        self.assertEqual(args.policy_port, 8004)

    def test_runner_requires_exact_bc2f_metadata(self) -> None:
        metadata = {
            "model": "pi05_bc2f", "loader": "bc2f", "action_horizon": 16, "state_history_lags": [45, 75],
            "num_state_frames": 3, "joint_observation_shape": [3, 7], "gripper_observation_shape": [3],
            "image_observation_shape": [180, 320, 3], "proprio_history_offsets": [0, 45, 75],
            "uses_exterior2": True, "requires_raw_force": True, "tasks": ["pour lillet into the blue cup"],
            "bundle_model": "pi05_bc2f_moddrop_s0", "bundle_step": 14999,
        }
        with redirect_stdout(io.StringIO()):
            _validate_policy_contract("bc2f_force", metadata)
        metadata["requires_raw_force"] = False
        with self.assertRaisesRegex(RuntimeError, "bc2f_force server"):
            _validate_policy_contract("bc2f_force", metadata)

    def test_server_warm_up_sends_a_consistent_raw_window(self) -> None:
        seen = {}

        class Policy:
            def infer(self, obs):
                seen.update(obs)
                return {"actions": np.zeros((16, 8))}

        warm_up(Policy(), "pour lillet into the blue cup", (0, 45, 75), 16, True, False, True)
        raw, stamp = seen["observation/raw_force"], seen["observation/robot_timestamp"]
        out = ff.window_features(MODEL, raw, stamp)  # the deployed loader must accept it
        self.assertEqual(out["valid_fraction"], 1.0)
        np.testing.assert_allclose(raw[-1, 1:8], seen["observation/joint_position"][0], atol=1e-6)


class Bc2fTrialsTest(unittest.TestCase):
    def test_models_map_to_profiles_and_exact_bundle_names(self) -> None:
        self.assertEqual(model_profile("bc2f_moddrop"), "bc2f_force")
        self.assertEqual(model_profile("alloc_fixed"), "bluecup_force")
        self.assertEqual(bundle_model_name("bc2f_moddrop"), "pi05_bc2f_moddrop_s0")
        self.assertNotEqual(bundle_model_name("bc2f_moddrop_prop"), bundle_model_name("bc2f_moddrop"))
        cmd = runner_command({"model": "bc2f_vanilla", "preset": "none", "trigger": "never"}, "pour lillet into the blue cup",
                             Path("trials/s/logs/0.json"), 60.0)
        self.assertIn("--checkpoint bc2f_force", cmd)
        self.assertNotIn("--fault", cmd)

    def test_success_without_target_needs_a_minimum_pour(self) -> None:
        log = {"end_reason": "operator_stop"}
        self.assertEqual(classify(45.0, 2.0, log, None, 15.0, 10.0, 20.0), "success")
        self.assertEqual(classify(5.0, 0.0, log, None, 15.0, 10.0, 20.0), "under")
        self.assertEqual(classify(45.0, 12.0, log, None, 15.0, 10.0, 20.0), "spill")
        self.assertEqual(classify(45.0, 2.0, {"end_reason": "safety"}, None, 15.0, 10.0, 20.0), "abort")

    def test_bc2f_plan_refuses_fault_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--root", root, "plan", "--session", "v5", "--models", "bc2f_vanilla", "--conditions",
                      "none", "cams_absent"])
            self.assertEqual(main(["--root", root, "plan", "--session", "v5", "--models", "bc2f_vanilla",
                                   "bc2f_moddrop", "bc2f_moddrop_prop", "--conditions", "none", "--reps", "10"]), 0)
            self.assertEqual(len((Path(root) / "v5" / "schedule.csv").read_text().splitlines()), 31)


if __name__ == "__main__":
    unittest.main()
