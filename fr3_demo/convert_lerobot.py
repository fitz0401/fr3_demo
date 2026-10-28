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


def _load_rgb(path: Path, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    """Load a frame at the given (width, height). Defaults to the DROID size.

    Tactile frames are 4:3 rather than the cameras' 16:9, so the size has to
    follow the stream instead of being fixed.
    """

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Image conversion requires Pillow; run: pip install -e '.[recording]'") from error
    with Image.open(path) as image:
        picture = image.convert("RGB")
        if picture.size != size:
            picture = picture.resize(size, resample=Image.Resampling.BICUBIC)
        return np.asarray(picture)


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


TACTILE_KEY = "tactile_image"


def _tactile_size(episode: Path) -> tuple[int, int]:
    """Stored tactile frame size, from the episode that recorded it."""

    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    signal = metadata.get("tactile_signal") or {}
    return int(signal["height"]), int(signal["width"])


def _episodes_with_tactile(episodes: Sequence[Path]) -> list[Path]:
    """Episodes that have no tactile stream, so the feature can be omitted."""

    missing = []
    for episode in episodes:
        with np.load(episode / "trajectory.npz") as trajectory:
            if "tactile_frame_index" not in trajectory.files:
                missing.append(episode)
    return missing


def _features(
    include_joint_torque: bool = False,
    tactile_shape: tuple[int, int] | None = None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
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
    if include_joint_torque:
        # Measured link-side joint torque (Franka tau_J, Nm). OpenPI's DROID
        # config repacks only the keys above and ignores this one, so carrying
        # it costs nothing today and keeps force available for later training.
        features["joint_torque"] = {"dtype": "float32", "shape": (7,), "names": ["joint_torque"]}
    if tactile_shape is not None:
        # The gel image nearest each grid instant. The sensor runs faster than
        # the 15 Hz grid, so this is a resampling of it: the complete stream is
        # written to tactile/ alongside the dataset.
        features[TACTILE_KEY] = {
            "dtype": "image",
            "shape": (*tactile_shape, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _episodes_with_joint_torque(episodes: Sequence[Path]) -> list[Path]:
    missing = []
    for episode in episodes:
        with np.load(episode / "trajectory.npz") as trajectory:
            if "joint_torque" not in trajectory.files:
                missing.append(episode)
    return missing


def write_tactile_sidecar(root: Path, episode_index: int, episode: Path) -> dict[str, Any] | None:
    """Copy one episode's full-rate tactile stream next to the dataset.

    The parquet carries one gel image per 15 Hz row, which throws away the
    sensor's extra temporal resolution, so the complete stream is kept here.
    The JPEGs are stored byte-for-byte as recorded, concatenated into one blob
    with an offset table: thousands of small files per episode would be
    unusable on the Hub, and an object array would need pickle to read back.
    """

    frames_dir = episode / "frames" / TACTILE_KEY
    with np.load(episode / "trajectory.npz") as trajectory:
        if "tactile_frame_index" not in trajectory.files:
            return None
        timestamps = np.asarray(trajectory["raw_tactile_timestamp"], dtype=np.float64)
        frame_numbers = np.asarray(trajectory["raw_tactile_frame_number"], dtype=np.int64)
        row_index = np.asarray(trajectory["tactile_frame_index"], dtype=np.int64)
        row_aligned = np.asarray(trajectory["tactile_aligned_timestamp"], dtype=np.float64)
        host = np.asarray(trajectory["host_monotonic_timestamp"], dtype=np.float64)

    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(paths) != len(timestamps):
        raise RuntimeError(
            f"Tactile frame count mismatch in {episode}: {len(paths)} JPEGs for {len(timestamps)} timestamps"
        )
    blobs = [path.read_bytes() for path in paths]
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    if blobs:
        offsets[1:] = np.cumsum([len(blob) for blob in blobs])
    payload = {
        "jpeg_blob": np.frombuffer(b"".join(blobs), dtype=np.uint8),
        "jpeg_offsets": offsets,
        "timestamp": timestamps,
        "frame_number": frame_numbers,
        "row_frame_index": row_index,
        "row_aligned_timestamp": row_aligned,
        "row_host_timestamp": host,
        "source_episode": np.asarray(episode.name),
    }
    directory = root / "tactile"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"episode_{episode_index:06d}.npz", **payload)

    span = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    gaps = np.diff(timestamps)
    errors_ms = np.abs(row_aligned - host) * 1000.0
    return {
        "episode_index": episode_index,
        "source_episode": episode.name,
        "tactile_frames": len(timestamps),
        "grid_frames": len(row_index),
        "rate_hz": (len(timestamps) - 1) / span if span > 0 else 0.0,
        "max_gap_ms": float(gaps.max() * 1000.0) if len(gaps) else 0.0,
        "grid_alignment_p95_ms": float(np.percentile(errors_ms, 95)) if len(errors_ms) else 0.0,
        "grid_alignment_max_ms": float(errors_ms.max()) if len(errors_ms) else 0.0,
        "reused_frames": int(len(row_index) - len(np.unique(row_index))),
        "bytes": int(offsets[-1]),
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
    # Force is only carried when every episode has it: filling gaps with zeros
    # would put fabricated readings in the training set.
    without_torque = _episodes_with_joint_torque(episodes)
    include_joint_torque = not without_torque
    # Same rule as force: carry tactile only when every episode has it, so the
    # feature is never padded with fabricated frames.
    without_tactile = _episodes_with_tactile(episodes)
    tactile_shape = None if without_tactile else _tactile_size(episodes[0])
    if without_tactile and len(without_tactile) < len(episodes):
        joined = "\n  ".join(str(path) for path in without_tactile[:5])
        more = f"\n  ... and {len(without_tactile) - 5} more" if len(without_tactile) > 5 else ""
        print(
            f"Omitting {TACTILE_KEY}: {len(without_tactile)} of {len(episodes)} episodes have no "
            f"tactile stream:\n  {joined}{more}"
        )
    elif tactile_shape is not None:
        print(f"Including {TACTILE_KEY} {tactile_shape[1]}x{tactile_shape[0]} from all {len(episodes)} episodes.")
        sizes = {_tactile_size(episode) for episode in episodes}
        if len(sizes) != 1:
            raise RuntimeError(f"Input sessions recorded different tactile sizes: {sorted(sizes)}")
    if without_torque:
        joined = "\n  ".join(str(path) for path in without_torque[:5])
        more = f"\n  ... and {len(without_torque) - 5} more" if len(without_torque) > 5 else ""
        print(
            f"Omitting joint_torque: {len(without_torque)} of {len(episodes)} episodes were recorded "
            f"before torque was captured:\n  {joined}{more}"
        )
    else:
        print(f"Including joint_torque from all {len(episodes)} episodes.")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="fr3",
        fps=int(dataset_fps),
        root=root,
        features=_features(include_joint_torque, tactile_shape),
        use_videos=True,
        image_writer_threads=4,
    )

    tactile_entries: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        metadata = episode_metadata[episode]
        task = str(metadata["language_instruction"]).strip()
        with np.load(episode / "trajectory.npz") as trajectory:
            frame_count = len(trajectory["joint_position"])
            exterior_paths = sorted((episode / "frames" / "exterior_image_left").glob("frame_*.jpg"))
            exterior2_paths = sorted((episode / "frames" / "exterior_image_2_left").glob("frame_*.jpg"))
            wrist_paths = sorted((episode / "frames" / "wrist_image").glob("frame_*.jpg"))
            tactile_paths: list[Path] = []
            tactile_rows = None
            if tactile_shape is not None:
                tactile_paths = sorted((episode / "frames" / TACTILE_KEY).glob("frame_*.jpg"))
                tactile_rows = np.asarray(trajectory["tactile_frame_index"], dtype=np.int64)
                if len(tactile_rows) != frame_count:
                    raise RuntimeError(f"Tactile row index/frame count mismatch in {episode}")
                if tactile_rows.size and int(tactile_rows.max()) >= len(tactile_paths):
                    raise RuntimeError(
                        f"Tactile row index points past the stored frames in {episode}: "
                        f"{int(tactile_rows.max())} >= {len(tactile_paths)}"
                    )
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
                frame = {
                    "exterior_image_1_left": exterior,
                    "exterior_image_2_left": exterior2,
                    "wrist_image_left": wrist,
                    "joint_position": np.asarray(trajectory["joint_position"][index], dtype=np.float32),
                    "gripper_position": np.asarray(trajectory["gripper_position"][index], dtype=np.float32),
                    "actions": action,
                    "task": task,
                }
                if include_joint_torque:
                    frame["joint_torque"] = np.asarray(trajectory["joint_torque"][index], dtype=np.float32)
                if tactile_rows is not None:
                    # Nearest-frame selection was already resolved at recording
                    # time, so this only follows the stored index.
                    frame[TACTILE_KEY] = _load_rgb(
                        tactile_paths[int(tactile_rows[index])],
                        (tactile_shape[1], tactile_shape[0]),
                    )
                dataset.add_frame(frame)
        dataset.save_episode()
        tactile_entry = write_tactile_sidecar(Path(dataset.root), episode_index, episode)
        if tactile_entry is not None:
            tactile_entries.append(tactile_entry)
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

    if tactile_entries:
        root_path = Path(dataset.root)
        meta_directory = root_path / "meta"
        meta_directory.mkdir(parents=True, exist_ok=True)
        rates = [entry["rate_hz"] for entry in tactile_entries]
        (meta_directory / "tactile_report.json").write_text(
            json.dumps(
                {
                    "episodes_total": len(episodes),
                    "episodes_with_tactile": len(tactile_entries),
                    "episodes_without_tactile": [path.name for path in without_tactile],
                    "aligned_feature": TACTILE_KEY if tactile_shape is not None else None,
                    "episodes": tactile_entries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        total = sum(entry["tactile_frames"] for entry in tactile_entries)
        grid = sum(entry["grid_frames"] for entry in tactile_entries)
        print(
            f"Tactile: {len(tactile_entries)} episodes, {total:,} frames at "
            f"{min(rates):.1f}-{max(rates):.1f} Hz ({total / max(1, grid):.2f}x the 15 Hz grid), "
            f"{sum(e['bytes'] for e in tactile_entries) / 1e6:.0f} MB in tactile/."
        )

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
