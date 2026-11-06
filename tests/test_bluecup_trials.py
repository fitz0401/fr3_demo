import csv
import io
import json
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from fr3_pi05.trials import RolloutLogger, classify, main, make_schedule


class ScheduleTest(unittest.TestCase):
    def test_every_cell_appears_reps_times_and_blocks_hold_one_model(self) -> None:
        rows = make_schedule(["flag", "moddrop", "alloc_fixed"], ["none", "cams_absent"], reps=10, block_reps=5, seed=3)
        self.assertEqual(len(rows), 3 * 2 * 10)
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            counts[(row["model"], row["preset"])] = counts.get((row["model"], row["preset"]), 0) + 1
            self.assertEqual(row["trigger"], "never" if row["preset"] == "none" else "pour")
        self.assertEqual(set(counts.values()), {10})
        blocks: dict[int, set[str]] = {}
        for row in rows:
            blocks.setdefault(row["block"], set()).add(row["model"])
        self.assertTrue(all(len(models) == 1 for models in blocks.values()))
        self.assertEqual(len(blocks), 6)  # two rounds of five reps, three models each
        self.assertEqual([r["trial_id"] for r in rows], list(range(len(rows))))

    def test_schedule_is_reproducible(self) -> None:
        a = make_schedule(["flag", "moddrop"], ["none", "cams_black"], 4, 2, seed=7)
        b = make_schedule(["flag", "moddrop"], ["none", "cams_black"], 4, 2, seed=7)
        self.assertEqual(a, b)


class ClassifyTest(unittest.TestCase):
    def test_outcomes(self) -> None:
        ok = {"end_reason": "completed"}
        self.assertEqual(classify(120, 0, ok, 120, 15, 5), "success")
        self.assertEqual(classify(90, 0, ok, 120, 15, 5), "under")
        self.assertEqual(classify(150, 0, ok, 120, 15, 5), "over")
        self.assertEqual(classify(120, 20, ok, 120, 15, 5), "spill")
        self.assertEqual(classify(None, None, {"gripper_closed_empty": True}, 120, 15, 5), "grasp_fail")
        self.assertEqual(classify(120, 0, {"end_reason": "safety"}, 120, 15, 5), "abort")
        self.assertEqual(classify(None, None, {"end_reason": "error"}, 120, 15, 5), "error")


class RolloutLoggerTest(unittest.TestCase):
    def test_writes_a_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "3.json"
            log = RolloutLogger(path, meta={"fault_preset": "cams_absent", "server_model": "pi05_bluecup_alloc_fixed"})
            log.step({"tilt_deg": 10.0, "pouring": False}, False, 1.0)
            log.step({"tilt_deg": 70.0, "pouring": True}, True, 0.78)
            log.step({"tilt_deg": 80.0, "pouring": True}, True, 0.78)
            log.inference(90.0)
            log.inference(110.0)
            log.event("safety_violation", detail="workspace")
            record = log.finish("completed")
            self.assertEqual(log.finish("error"), {})  # second call is a no-op
            saved = json.loads(path.read_text())
        self.assertEqual(saved, record)
        self.assertEqual(saved["steps"], 3)
        self.assertEqual(saved["fault_active_steps"], 2)
        self.assertEqual(saved["pour"], {"started": True, "first_step": 2, "steps": 2, "max_tilt_deg": 80.0})
        self.assertFalse(saved["gripper_closed_empty"])
        self.assertEqual(saved["inference_ms_median"], 100.0)
        self.assertEqual(saved["fault_preset"], "cams_absent")


class CliTest(unittest.TestCase):
    def test_plan_next_record_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--root", str(root), "plan", "--session", "r1", "--models", "alloc_fixed", "moddrop",
                      "--conditions", "none", "cams_absent", "--reps", "2", "--block-reps", "2"])
                main(["--root", str(root), "next", "--session", "r1"])
            text = out.getvalue()
            self.assertIn("8 trials", text)
            self.assertIn("start_checkpoint.sh bluecup_force", text)
            self.assertIn("fr3-pi05-run --checkpoint bluecup_force", text)

            log_path = root / "r1" / "logs" / "0.json"
            RolloutLogger(log_path, meta={"fault_preset": None}).finish("completed")
            with redirect_stdout(io.StringIO()):
                main(["--root", str(root), "record", "--session", "r1", "--trial", "0", "--rollout-log", str(log_path),
                      "--cup-before", "0", "--cup-after", "118", "--bottle-before", "600", "--bottle-after", "480",
                      "--target-g", "120"])
            with open(root / "r1" / "results.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["poured_g"]), 118.0)
            self.assertEqual(float(rows[0]["spilled_g"]), 2.0)
            self.assertEqual(rows[0]["outcome"], "success")

            summary = io.StringIO()
            with redirect_stdout(summary):
                main(["--root", str(root), "summary", "--session", "r1", "--target-g", "120"])
            self.assertIn("| 1/1 |", summary.getvalue())

            # Recording the same trial twice is refused.
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), unittest.mock.patch("sys.stderr"):
                main(["--root", str(root), "record", "--session", "r1", "--trial", "0",
                      "--cup-before", "0", "--cup-after", "1", "--bottle-before", "1", "--bottle-after", "0"])


if __name__ == "__main__":
    unittest.main()
