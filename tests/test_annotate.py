import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fr3_demo.annotate import main


def _episode(root: Path, index: int, language: str | None) -> Path:
    episode = root / f"episode_{index:06d}"
    episode.mkdir(parents=True)
    metadata = {"complete": True, "language_instruction": language}
    path = episode / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


class AnnotateTest(unittest.TestCase):
    def test_all_prompts_once_and_skips_labeled_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _episode(root, 0, None)
            labeled = _episode(root, 1, "existing task")
            third = _episode(root, 2, "   ")

            with patch("builtins.input", return_value="shared task") as prompt:
                result = main(["--data-dir", str(root), "--all"])

            self.assertEqual(result, 0)
            prompt.assert_called_once_with("Language for all unlabeled episodes (blank to skip): ")
            self.assertEqual(json.loads(first.read_text())["language_instruction"], "shared task")
            self.assertEqual(json.loads(labeled.read_text())["language_instruction"], "existing task")
            self.assertEqual(json.loads(third.read_text())["language_instruction"], "shared task")

    def test_all_does_not_prompt_when_every_episode_is_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labeled = _episode(root, 0, "existing task")

            with patch("builtins.input") as prompt:
                result = main(["--data-dir", str(root), "--all"])

            self.assertEqual(result, 0)
            prompt.assert_not_called()
            self.assertEqual(json.loads(labeled.read_text())["language_instruction"], "existing task")


if __name__ == "__main__":
    unittest.main()
