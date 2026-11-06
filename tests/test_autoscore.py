import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fr3_pi05.autoscore import AutoStop, outcome, score_steps
from fr3_pi05.raw_force import RawForceSampler
from fr3_pi05.runner import _parse
from fr3_pi05.trials import RolloutLogger, main, runner_command, score_log

HOME_Q = np.array([-0.047, -0.734, -0.028, -2.279, 0.002, 1.578, 0.033])
HZ = 15.0


def pour_rollout(**change):
    """A synthetic successful rollout: home 2 s, close on the bottle 1.5 s, lift 3 s, pour 5 s, carry back 3 s,
    release and stay open 1 s. Segments can be altered through keyword arguments."""
    plan = [  # (seconds, gripper, dz above the grasp, pouring, F_z relative to home)
        (2.0, 1.0, 0.0, False, 0.0),
        (1.5, change.get("grasp_g", 0.79), 0.0, False, change.get("grasp_fz", -1.0)),
        (3.0, change.get("grasp_g", 0.79), 0.10, False, change.get("lift_fz", -8.0)),
        (5.0, change.get("grasp_g", 0.79), 0.10, change.get("pour", True), change.get("pour_fz", -7.5)),
        (3.0, change.get("grasp_g", 0.79), change.get("carry_dz", 0.10), False, change.get("carry_fz", -7.5)),
        (0.4, change.get("grasp_g", 0.79), change.get("release_dz", 0.0), False, 5.0),
        (1.0, 1.0, change.get("release_dz", 0.0) + 0.02, False, 0.5),
    ]
    t, g, z, pouring, fz = [], [], [], [], []
    for seconds, grip, dz, pour, force in plan:
        n = int(round(seconds * HZ))
        start = t[-1] + 1 / HZ if t else 0.0
        t += list(start + np.arange(n) / HZ)
        g += [grip] * n
        z += [0.2 + dz] * n
        pouring += [pour] * n
        fz += [force] * n
    t = np.array(t)
    k = len(t)
    q = np.tile(HOME_Q, (k, 1))
    q[:, 0] += np.linspace(0, 1.0, k) * (t > 2.0)  # the arm moves after the 2 s at home
    eef = np.zeros((k, 3))
    eef[:, 2] = z
    wrench = np.zeros((k, 6))
    wrench[:, 2] = np.array(fz) - 0.5  # home F_z -0.5 N as in the demonstrations
    tilt = np.where(pouring, 95.0, 5.0)
    return dict(t=t, q=q, gripper=np.array(g), pouring=np.array(pouring), tilt=tilt, wrench=wrench, eef=eef)


def score(r, force=True):
    return score_steps(r["t"], r["q"], r["gripper"], r["pouring"], r["tilt"], r["wrench"] if force else None, r["eef"])


class ScoreStepsTest(unittest.TestCase):
    def test_complete_pour_scores_four_stages_with_timings(self) -> None:
        s = score(pour_rollout())
        self.assertEqual(s["completion"], 4)
        self.assertFalse(s["drop"])
        self.assertAlmostEqual(s["t_grasp_s"], 2.0, delta=0.1)
        self.assertAlmostEqual(s["t_lift_s"], 3.5, delta=0.1)
        self.assertAlmostEqual(s["t_pour_s"], 6.5, delta=0.1)
        self.assertAlmostEqual(s["pour_s"], 5.0, delta=0.15)
        self.assertEqual(s["pour_bouts"], 1)
        self.assertEqual(s["grasp_attempts"], 1)
        self.assertAlmostEqual(s["start_delay_s"], 2.0, delta=0.2)
        self.assertAlmostEqual(s["held_mass_g"], 8.0 / 9.81 * 1000, delta=5)
        self.assertEqual(s["place"]["dz_cm"], 0.0)
        self.assertEqual(outcome(s, 45.0, 2.0, "completed", 20.0, 10.0), "success")

    def test_closing_on_nothing_is_a_grasp_failure(self) -> None:
        s = score(pour_rollout(grasp_g=0.02, pour=False))
        self.assertEqual(s["completion"], 0)
        self.assertGreaterEqual(s["empty_closes"], 1)
        self.assertEqual(outcome(s, None, None, "stall", 20.0, 10.0), "grasp_fail")

    def test_bottle_slipping_out_while_lifted_is_a_drop(self) -> None:
        s = score(pour_rollout(carry_fz=0.0))
        self.assertTrue(s["drop"])
        self.assertEqual(s["completion"], 3)  # lifted and poured before it fell, never placed
        self.assertEqual(outcome(s, 40.0, 1.0, "completed", 20.0, 10.0), "drop")

    def test_lift_needs_the_bottle_weight_when_force_is_available(self) -> None:
        s = score(pour_rollout(lift_fz=-1.0, pour_fz=-1.0, carry_fz=-1.0))
        self.assertEqual(s["completion"], 1)
        self.assertEqual(outcome(s, None, None, "stall", 20.0, 10.0), "lift_fail")
        self.assertEqual(score(pour_rollout(lift_fz=-1.0, pour_fz=-1.0, carry_fz=-1.0), force=False)["completion"], 4)

    def test_no_pour_and_bad_placement_stop_the_count(self) -> None:
        self.assertEqual(score(pour_rollout(pour=False))["completion"], 2)
        high = score(pour_rollout(release_dz=0.15))
        self.assertEqual(high["completion"], 3)
        self.assertEqual(outcome(high, 45.0, 1.0, "completed", 20.0, 10.0), "place_fail")

    def test_weights_decide_once_the_bottle_is_back(self) -> None:
        s = score(pour_rollout())
        self.assertEqual(outcome(s, 5.0, 1.0, "completed", 20.0, 10.0), "under")
        self.assertEqual(outcome(s, 45.0, 15.0, "completed", 20.0, 10.0), "spill")
        self.assertEqual(outcome(s, None, None, "completed", 20.0, 10.0), "other")
        self.assertEqual(outcome(s, 45.0, 1.0, "error", 20.0, 10.0), "error")


class AutoStopTest(unittest.TestCase):
    def test_release_after_a_pour_completes(self) -> None:
        stop = AutoStop(release_s=2.0)
        q = HOME_Q.copy()
        for i in range(100):  # moving, holding, no pour yet
            q = q + 0.01
            self.assertIsNone(stop.update(i / HZ, q, 0.79, False))
        self.assertIsNone(stop.update(7.0, q, 1.0, False))  # open without a pour: keep going
        self.assertIsNone(stop.update(7.1, q, 1.0, True))
        self.assertIsNone(stop.update(9.0, q, 1.0, True))
        self.assertEqual(stop.update(9.2, q, 1.0, True), "completed")

    def test_a_still_arm_stalls_and_motion_resets_the_clock(self) -> None:
        stop = AutoStop(stall_s=30.0)
        self.assertIsNone(stop.update(0.0, HOME_Q, 1.0, False))
        self.assertIsNone(stop.update(29.0, HOME_Q, 1.0, False))
        self.assertIsNone(stop.update(29.5, HOME_Q + 0.05, 1.0, False))
        self.assertIsNone(stop.update(59.0, HOME_Q + 0.05, 1.0, False))
        self.assertEqual(stop.update(59.6, HOME_Q + 0.05, 1.0, False), "stall")

    def test_time_limit(self) -> None:
        stop = AutoStop(max_duration_s=240.0)
        q = HOME_Q.copy()
        reason = None
        for i in range(0, 2500):
            q = q + (0.03 if i % 2 else -0.03)
            reason = stop.update(i * 0.1, q, 0.79, False)
            if reason:
                break
        self.assertEqual(reason, "timeout")


class RecordingTest(unittest.TestCase):
    def test_logger_writes_steps_and_raw_stream_that_record_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trials"
            with redirect_stdout(io.StringIO()):
                main(["--root", str(root), "plan", "--session", "s", "--models", "bc2f_vanilla",
                      "--conditions", "none", "--reps", "1"])
            log_path = root / "s" / "logs" / "0.json"
            log = RolloutLogger(log_path, meta={"server_model": "pi05_bc2f_vanilla_s0"})
            raw = []
            for i in range(60):
                stamp = 100.0 + i / HZ
                raw += [[stamp - j / 1000.0, *HOME_Q, *([1.0] * 7)] for j in range(66, -1, -1)]
                log.step({"tilt_deg": 3.0, "pouring": False}, False, 1.0, q=HOME_Q, time_sec=stamp)
            raw = np.unique(np.array(raw), axis=0)
            log.finish("stall", arrays={"raw_force": raw})
            with np.load(log_path.with_suffix(".npz")) as data:
                self.assertEqual(data["steps_q"].shape, (60, 7))
            with np.load(root / "s" / "logs" / "0.raw.npz") as data:
                self.assertEqual(data["raw_force"].shape[1], 15)
            self.assertEqual(score_log(log_path)["completion"], 0)
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--root", str(root), "record", "--session", "s", "--trial", "0", "--rollout-log", str(log_path),
                      "--cup-before", "0", "--cup-after", "0", "--bottle-before", "800", "--bottle-after", "800",
                      "--min-poured-g", "20", "--spill-tolerance-g", "10"])
            self.assertIn("completion 0/4", out.getvalue())
            results = (root / "s" / "results.csv").read_text()
            self.assertIn("grasp_fail", results)
            self.assertIn("completion", results.splitlines()[0])
            summary = io.StringIO()
            with redirect_stdout(summary):
                main(["--root", str(root), "summary", "--session", "s"])
            self.assertIn("Automatic stage scores", summary.getvalue())

    def test_sampler_archive_keeps_every_sample(self) -> None:
        class Fake:
            t = 10.0

            def get_joint_states(self):
                self.t += 0.001
                return {"qpos": HOME_Q.tolist(), "tau_J": [1.0] * 7, "time_sec": self.t}

        sampler = RawForceSampler(lambda: Fake(), rate_hz=5000.0, keep_s=0.2, archive=True).start()
        try:
            time.sleep(0.5)
        finally:
            sampler.close()
        archived = sampler.archived()
        self.assertGreater(len(archived), 300)  # more than the 0.2 s ring buffer holds
        self.assertTrue(np.all(np.diff(archived[:, 0]) > 0))
        self.assertIsNone(RawForceSampler(lambda: Fake()).archived())

    def test_runner_and_trial_command_use_auto_stop(self) -> None:
        args = _parse(["--checkpoint", "bc2f_force", "--auto-stop", "--stall-s", "20"])
        self.assertTrue(args.auto_stop)
        self.assertEqual(args.stall_s, 20.0)
        command = runner_command({"model": "bc2f_moddrop", "preset": "none", "trigger": "never"}, "pour", Path("l.json"), 60)
        self.assertIn("--auto-stop", command)


if __name__ == "__main__":
    unittest.main()
