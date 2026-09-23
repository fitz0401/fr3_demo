"""Convert raw FR3 demonstrations to OpenPI's pi0.5-DROID LeRobot schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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


def find_episodes(data_dirs: Path | Sequence[Path]) -> list[Path]:
    """Find unique completed episodes beneath one or more input roots."""

    roots = [data_dirs] if isinstance(data_dirs, Path) else list(data_dirs)
    episodes: set[Path] = set()
    for data_dir in roots:
        for metadata_path in data_dir.expanduser().resolve().glob("**/episode_*/metadata.json"):
            episode = metadata_path.parent
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("complete") and (episode / "trajectory.npz").is_file():
                episodes.add(episode)
    return sorted(episodes)


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


def _validate_episode_sync(
    episode: Path,
    metadata: dict[str, Any],
    *,
    allow_legacy_unsynchronized: bool,
) -> dict[str, Any] | None:
    """Reject unverified/index-aligned raw data unless explicitly requested."""

    schema_version = int(metadata.get("schema_version", 1))
    if schema_version < 2:
        if allow_legacy_unsynchronized:
            return None
        raise RuntimeError(
            f"{episode} uses legacy schema v{schema_version}, which only aligns streams by loop index. "
            "Re-record it with the synchronized collector, or use --allow-legacy-unsynchronized "
            "only after auditing the episode."
        )
    if metadata.get("timebase") != "host_monotonic":
        raise RuntimeError(f"Synchronized episode has an unknown timebase: {episode}")
    report_path = episode / "sync_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"Synchronized episode is missing sync_report.json: {episode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("valid") is not True or metadata.get("sync", {}).get("valid") is not True:
        raise RuntimeError(f"Episode failed synchronization validation: {episode}")
    with np.load(episode / "trajectory.npz") as trajectory:
        required = {
            "host_monotonic_timestamp",
            "robot_before_host_timestamp",
            "robot_after_host_timestamp",
            "exterior_camera_hardware_timestamp",
            "exterior_camera_aligned_timestamp",
            "exterior_camera_frame_number",
            "wrist_camera_hardware_timestamp",
            "wrist_camera_aligned_timestamp",
            "wrist_camera_frame_number",
            "action_joint_host_timestamp",
        }
        missing = required - set(trajectory.files)
        if missing:
            raise RuntimeError(f"Synchronized timing fields are missing in {episode}: {', '.join(sorted(missing))}")
        timestamps = np.asarray(trajectory["timestamp"], dtype=np.float64)
        if len(timestamps) > 1:
            expected_period = 1.0 / float(metadata["fps"])
            maximum_error = float(np.max(np.abs(np.diff(timestamps) - expected_period)))
            if maximum_error > 1e-5:
                raise RuntimeError(f"Episode is not on its declared fixed-rate timeline: {episode}")
        for key in ("exterior_camera_frame_number", "wrist_camera_frame_number"):
            numbers = np.asarray(trajectory[key], dtype=np.int64)
            if len(np.unique(numbers)) != len(numbers):
                raise RuntimeError(f"Duplicate camera frame detected in {episode}: {key}")
        if "exterior2_camera_frame_number" in trajectory:
            numbers = np.asarray(trajectory["exterior2_camera_frame_number"], dtype=np.int64)
            if len(np.unique(numbers)) != len(numbers):
                raise RuntimeError(f"Duplicate camera frame detected in {episode}: exterior2_camera_frame_number")
    return report


def convert(
    data_dirs: Path | Sequence[Path],
    repo_id: str,
    output_root: Path | None = None,
    push: bool = False,
    public: bool = False,
    allow_legacy_unsynchronized: bool = False,
) -> Path:
    episodes = find_episodes(data_dirs)
    if not episodes:
        roots = [data_dirs] if isinstance(data_dirs, Path) else list(data_dirs)
        joined = ", ".join(str(path) for path in roots)
        raise RuntimeError(f"No completed raw episodes found under: {joined}")
    missing_language = []
    episode_metadata: dict[Path, dict[str, Any]] = {}
    sync_reports: dict[Path, dict[str, Any] | None] = {}
    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        episode_metadata[episode] = metadata
        if not str(metadata.get("language_instruction") or "").strip():
            missing_language.append(str(episode))
        sync_reports[episode] = _validate_episode_sync(
            episode,
            metadata,
            allow_legacy_unsynchronized=allow_legacy_unsynchronized,
        )
    if missing_language:
        joined = "\n  ".join(missing_language)
        raise RuntimeError(f"Language is missing for:\n  {joined}\nRun fr3-annotate before conversion.")

    LeRobotDataset = _load_lerobot_dataset()
    root = None if output_root is None else output_root.expanduser().resolve() / repo_id
    if root is not None and root.exists():
        raise FileExistsError(f"Output already exists: {root}")
    episode_fps = {float(metadata.get("fps", 15.0)) for metadata in episode_metadata.values()}
    if len(episode_fps) != 1:
        raise RuntimeError(f"Input sessions use different recording rates: {sorted(episode_fps)}")
    dataset_fps = episode_fps.pop()
    if not dataset_fps.is_integer():
        raise RuntimeError(f"LeRobot requires an integer dataset fps, got {dataset_fps}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="fr3",
        fps=int(dataset_fps),
        root=root,
        features=_features(),
        use_videos=True,
        image_writer_threads=4,
    )

    for episode in episodes:
        metadata = episode_metadata[episode]
        task = str(metadata["language_instruction"]).strip()
        with np.load(episode / "trajectory.npz") as trajectory:
            frame_count = len(trajectory["joint_position"])
            exterior_paths = sorted((episode / "frames" / "exterior_image_left").glob("frame_*.jpg"))
            exterior2_paths = sorted((episode / "frames" / "exterior_image_2_left").glob("frame_*.jpg"))
            wrist_paths = sorted((episode / "frames" / "wrist_image").glob("frame_*.jpg"))
            if len(exterior_paths) != frame_count or len(wrist_paths) != frame_count:
                raise RuntimeError(f"Camera/numeric frame count mismatch in {episode}")
            if exterior2_paths and len(exterior2_paths) != frame_count:
                raise RuntimeError(f"Optional exterior camera/frame count mismatch in {episode}")
            black_exterior = np.zeros((180, 320, 3), dtype=np.uint8)
            for index in range(frame_count):
                exterior = _load_rgb(exterior_paths[index])
                exterior2 = _load_rgb(exterior2_paths[index]) if exterior2_paths else black_exterior
                wrist = _load_rgb(wrist_paths[index])
                action = np.concatenate(
                    [trajectory["action_joint_velocity"][index], trajectory["action_gripper_position"][index]],
                    dtype=np.float32,
                )
                dataset.add_frame(
                    {
                        "exterior_image_1_left": exterior,
                        "exterior_image_2_left": exterior2,
                        "wrist_image_left": wrist,
                        "joint_position": np.asarray(trajectory["joint_position"][index], dtype=np.float32),
                        "gripper_position": np.asarray(trajectory["gripper_position"][index], dtype=np.float32),
                        "actions": action,
                        "task": task,
                    }
                )
        dataset.save_episode()
        report = sync_reports[episode]
        sync_status = "legacy/index-aligned"
        if report is not None:
            camera_p95 = report["camera_pair_skew_ms"]["p95"]
            robot_p95 = report["robot_interpolation_gap_ms"]["p95"]
            sync_status = f"camera skew p95={camera_p95:.1f} ms, robot gap p95={robot_p95:.1f} ms"
        print(f"Converted {episode.name}: {frame_count} frames, {sync_status}, task={task!r}")

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
    parser.add_argument(
        "--data-dir",
        dest="data_dirs",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="one or more session/parent directories; repeated --data-dir is also accepted",
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset id, e.g. username/fr3_task")
    parser.add_argument("--output-root", type=Path, help="defaults to LeRobot's HF_LEROBOT_HOME")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--public", action="store_true", help="make an uploaded dataset public (default: private)")
    parser.add_argument(
        "--allow-legacy-unsynchronized",
        action="store_true",
        help="allow schema-v1 episodes that were aligned only by recorder-loop index",
    )
    args = parser.parse_args(argv)
    output = convert(
        args.data_dirs,
        args.repo_id,
        args.output_root,
        args.push_to_hub,
        args.public,
        args.allow_legacy_unsynchronized,
    )
    print(f"LeRobot dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
