import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fr3_demo.downsize import downsize_episode, find_episodes


def _episode(root: Path, *, size: tuple[int, int] = (424, 240), complete: bool = True) -> Path:
    episode = root / "session_test" / "episode_000000"
    for camera in ("exterior_image_left", "wrist_image"):
        (episode / "frames" / camera).mkdir(parents=True)
        for index in range(3):
            image = Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
            image.save(episode / "frames" / camera / f"frame_{index:06d}.jpg", quality=92)
    (episode / "metadata.json").write_text(
        json.dumps({"complete": complete, "frame_count": 3, "image_size": None}), encoding="utf-8"
    )
    (episode / "trajectory.npz").write_bytes(b"untouched")
    return episode


class DownsizeTest(unittest.TestCase):
    def test_dry_run_reports_savings_without_touching_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))
            before = {p: p.stat().st_size for p in episode.glob("frames/*/*.jpg")}

            report = downsize_episode(episode, (320, 180), apply=False)

            self.assertEqual(report.resized, 6)
            self.assertGreater(report.reclaimed, 0)
            self.assertEqual({p: p.stat().st_size for p in episode.glob("frames/*/*.jpg")}, before)
            self.assertIsNone(json.loads((episode / "metadata.json").read_text())["image_size"])

    def test_apply_rewrites_frames_and_records_the_new_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))

            report = downsize_episode(episode, (320, 180), apply=True)

            self.assertEqual(report.resized, 6)
            for path in episode.glob("frames/*/*.jpg"):
                with Image.open(path) as stored:
                    self.assertEqual(stored.size, (320, 180))
            metadata = json.loads((episode / "metadata.json").read_text())
            self.assertEqual(metadata["image_size"], {"width": 320, "height": 180, "resample": "bicubic"})
            self.assertTrue(metadata["resized_after_recording"])
            self.assertEqual((episode / "trajectory.npz").read_bytes(), b"untouched")
            self.assertEqual(list(episode.glob("frames/*/*.tmp")), [])

    def test_rerunning_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary))
            downsize_episode(episode, (320, 180), apply=True)
            sizes = {p: p.stat().st_size for p in episode.glob("frames/*/*.jpg")}

            report = downsize_episode(episode, (320, 180), apply=True)

            self.assertEqual(report.resized, 0)
            self.assertEqual({p: p.stat().st_size for p in episode.glob("frames/*/*.jpg")}, sizes)

    def test_incomplete_episodes_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode(Path(temporary), complete=False)

            self.assertEqual(downsize_episode(episode, (320, 180), apply=True).skipped, "incomplete")

    def test_sessions_and_episodes_are_both_valid_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = _episode(root)

            self.assertEqual(find_episodes([root]), [episode])
            self.assertEqual(find_episodes([episode]), [episode])


if __name__ == "__main__":
    unittest.main()
