"""Convert raw FR3 demonstrations to OpenPI's pi0.5-DROID LeRobot schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_lerobot_dataset() -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("LeRobot conversion support is not installed; run: pip install -e '.[convert]'") from error
    return LeRobotDataset


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image conversion requires Pillow; run: pip install -e '.[recording]'") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((320, 180), resample=Image.Resampling.BICUBIC))


def find_episodes(data_dir: Path) -> list[Path]:
    episodes = []
    for metadata_path in sorted(data_dir.expanduser().resolve().glob("**/episode_*/metadata.json")):
        episode = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete") and (episode / "trajectory.npz").is_file():
            episodes.append(episode)
    return episodes


def _features() -> dict[str, dict[str, Any]]:
    return {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        },
        # OpenPI's LeRobotDROIDDataConfig repacks this key, although DroidInputs ignores it for pi0.5.
        "exterior_image_2_left": {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
        "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    }


def convert(
    data_dir: Path, repo_id: str, output_root: Path | None = None, push: bool = False, public: bool = False
) -> Path:
    episodes = find_episodes(data_dir)
    if not episodes:
        raise RuntimeError(f"No completed raw episodes found under {data_dir}")
    missing_language = []
    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        if not str(metadata.get("language_instruction") or "").strip():
            missing_language.append(str(episode))
    if missing_language:
        joined = "\n  ".join(missing_language)
        raise RuntimeError(f"Language is missing for:\n  {joined}\nRun fr3-annotate before conversion.")

    LeRobotDataset = _load_lerobot_dataset()
    root = None if output_root is None else output_root.expanduser().resolve() / repo_id
    if root is not None and root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="fr3",
        fps=15,
        root=root,
        features=_features(),
        use_videos=True,
        image_writer_threads=4,
    )

    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        task = str(metadata["language_instruction"]).strip()
        with np.load(episode / "trajectory.npz") as trajectory:
            frame_count = len(trajectory["joint_position"])
            exterior_paths = sorted((episode / "frames" / "exterior_image_left").glob("frame_*.jpg"))
            wrist_paths = sorted((episode / "frames" / "wrist_image").glob("frame_*.jpg"))
            if len(exterior_paths) != frame_count or len(wrist_paths) != frame_count:
                raise RuntimeError(f"Camera/numeric frame count mismatch in {episode}")
            black_exterior = np.zeros((180, 320, 3), dtype=np.uint8)
            for index in range(frame_count):
                exterior = _load_rgb(exterior_paths[index])
                wrist = _load_rgb(wrist_paths[index])
                action = np.concatenate(
                    [trajectory["action_joint_velocity"][index], trajectory["action_gripper_position"][index]],
                    dtype=np.float32,
                )
                dataset.add_frame(
                    {
                        "exterior_image_1_left": exterior,
                        "exterior_image_2_left": black_exterior,
                        "wrist_image_left": wrist,
                        "joint_position": np.asarray(trajectory["joint_position"][index], dtype=np.float32),
                        "gripper_position": np.asarray(trajectory["gripper_position"][index], dtype=np.float32),
                        "actions": action,
                        "task": task,
                    }
                )
        dataset.save_episode()
        print(f"Converted {episode.name}: {frame_count} frames, task={task!r}")

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    if push:
        dataset.push_to_hub(
            tags=["droid", "fr3", "pi05", "lerobot"],
            private=not public,
            push_videos=True,
            license="apache-2.0",
        )
    return Path(dataset.root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert annotated FR3 demonstrations to LeRobot and optionally upload."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset id, e.g. username/fr3_task")
    parser.add_argument("--output-root", type=Path, help="defaults to LeRobot's HF_LEROBOT_HOME")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--public", action="store_true", help="make an uploaded dataset public (default: private)")
    args = parser.parse_args(argv)
    output = convert(args.data_dir, args.repo_id, args.output_root, args.push_to_hub, args.public)
    print(f"LeRobot dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
