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


def _features(include_joint_torque: bool = False) -> dict[str, dict[str, Any]]:
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
    return features


def _episodes_with_joint_torque(episodes: Sequence[Path]) -> list[Path]:
    missing = []
    for episode in episodes:
        with np.load(episode / "trajectory.npz") as trajectory:
            if "joint_torque" not in trajectory.files:
                missing.append(episode)
    return missing


# Every arm sample of an episode, on the controller's own timeline, plus the
# two per-frame clocks the training side needs to line them up with the 15 Hz
# frames. Copied out verbatim; gravity compensation, windowing and calibration
# belong to the training stack, not here.
RAW_FORCE_KEYS = (
    "raw_force_joint_torque",
    "raw_force_joint_position",
    "raw_force_robot_timestamp",
    "raw_force_request_send_timestamp",
    "raw_force_response_receive_timestamp",
)
FRAME_CLOCK_KEYS = ("robot_timestamp", "host_monotonic_timestamp")
WEIGHT_FIELDS = (
    "bottle_before_g",
    "cup_before_g",
    "bottle_after_g",
    "cup_after_g",
    "poured_g",
    "spilled_g",
)
# Re-interpolating the raw stream at the frame clock reproduces the frame
# torque column to float32 round-off (measured 1.9e-06 Nm). Pairing one
# episode's raw arrays with another's frames misses by whole Nm, so anything
# above this is a wiring bug rather than noise.
ALIGNMENT_TOLERANCE_NM = 1e-2


def _legacy_feature_tags(node: Any) -> Any:
    """Rename the datasets 4.x "List" feature tag back to "Sequence"."""

    if isinstance(node, dict):
        converted = {key: _legacy_feature_tags(value) for key, value in node.items()}
        if converted.get("_type") == "List":
            converted["_type"] = "Sequence"
            converted.setdefault("length", -1)
        return converted
    if isinstance(node, list):
        return [_legacy_feature_tags(value) for value in node]
    return node


def restore_sequence_feature_tags(root: Path) -> list[Path]:
    """Make the parquet footers readable by every HF datasets version.

    datasets 4.x renamed the fixed-length list feature from "Sequence" to
    "List" and stamps that name into every parquet footer. A reader on
    datasets 3.x refuses the file outright with "Feature type 'List' not
    found", which is why previous FR3 datasets had to be repaired by hand
    after conversion. Both names decode to the same fixed_size_list, and 4.x
    and 5.x still accept "Sequence", so writing the older name makes a single
    file readable on either side. Only schema metadata is rewritten; values
    and encoded image bytes are preserved exactly.
    """

    paths = sorted(root.glob("data/**/*.parquet"))
    if not paths:
        return []
    import pyarrow.parquet as pq

    rewritten: list[Path] = []
    for path in paths:
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        encoded = metadata.get(b"huggingface")
        if encoded is None:
            continue
        info = json.loads(encoded)
        features = info.get("info", {}).get("features")
        if features is None:
            continue
        patched = _legacy_feature_tags(features)
        if patched == features:
            continue
        info["info"]["features"] = patched
        metadata[b"huggingface"] = json.dumps(info).encode("utf-8")
        # Keep the codec the writer chose so only the footer changes. pyarrow
        # reports an uncompressed column as "UNCOMPRESSED" but only accepts
        # "none" when writing one back.
        codec = str(pq.ParquetFile(path).metadata.row_group(0).column(0).compression).lower()
        # Never write over the original in place: pq.write_table truncates the
        # destination before it validates its arguments, so a rejected argument
        # destroys the file it was supposed to rewrite. Write beside it and
        # replace only once the new file is complete.
        staged = path.with_suffix(path.suffix + ".tmp")
        try:
            pq.write_table(
                table.replace_schema_metadata(metadata),
                staged,
                compression="none" if codec == "uncompressed" else codec,
            )
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
        rewritten.append(path)
    return rewritten


def _alignment_error_nm(
    raw_time: np.ndarray,
    raw_torque: np.ndarray,
    frame_time: np.ndarray,
    frame_torque: np.ndarray,
) -> tuple[float, int]:
    """Largest torque mismatch between the re-interpolated raw stream and the frames.

    Returns the error and how many frames had to be skipped because they fall
    outside the raw stream. The recorder can interpolate the first grid frame
    from a poll taken a fraction of a millisecond before the raw buffer starts
    retaining samples, which happens on about half of the episodes here.
    np.interp would clamp to the first sample for those frames, so comparing
    them measures the clamp rather than the data.
    """

    # The sampler can poll the same controller state twice; keep the first
    # occurrence so np.interp sees a strictly increasing x, while the saved
    # arrays stay untouched.
    keep = np.sort(np.unique(raw_time, return_index=True)[1])
    times = raw_time[keep]
    inside = (frame_time >= times[0]) & (frame_time <= times[-1])
    outside = int((~inside).sum())
    if not inside.any():
        raise RuntimeError("Raw force does not overlap the episode's frames")
    worst = 0.0
    for joint in range(raw_torque.shape[1]):
        interpolated = np.interp(frame_time[inside], times, raw_torque[keep, joint].astype(np.float64))
        difference = np.abs(interpolated - frame_torque[inside, joint].astype(np.float64))
        worst = max(worst, float(np.max(difference)))
    return worst, outside


def write_raw_force(
    root: Path,
    episode_index: int,
    episode: Path,
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Copy one episode's raw force stream next to the dataset; None if absent."""

    with np.load(episode / "trajectory.npz") as trajectory:
        if not all(key in trajectory.files for key in RAW_FORCE_KEYS):
            return None
        payload = {key: trajectory[key] for key in RAW_FORCE_KEYS}
        for key in FRAME_CLOCK_KEYS:
            if key not in trajectory.files:
                raise RuntimeError(f"Episode has raw force but no {key}: {episode}")
            payload[key] = trajectory[key]
        frame_torque = np.asarray(trajectory["joint_torque"], dtype=np.float32)

    estimate = ((report or {}).get("clock_estimates") or {}).get("robot_controller")
    if not estimate:
        raise RuntimeError(f"Episode has raw force but no robot clock estimate: {episode}")

    payload["source_episode"] = np.asarray(episode.name)
    payload["robot_clock_scale"] = np.asarray(float(estimate["scale"]))
    payload["robot_clock_offset"] = np.asarray(float(estimate["offset"]))

    raw_time = np.asarray(payload["raw_force_robot_timestamp"], dtype=np.float64)
    frame_time = np.asarray(payload["robot_timestamp"], dtype=np.float64)
    alignment, outside = _alignment_error_nm(
        raw_time, payload["raw_force_joint_torque"], frame_time, frame_torque
    )
    if alignment > ALIGNMENT_TOLERANCE_NM:
        raise RuntimeError(
            f"Raw force does not match the frames of {episode}: max torque difference {alignment:.4g} Nm "
            f"exceeds {ALIGNMENT_TOLERANCE_NM:g} Nm. The arrays most likely belong to a different episode."
        )

    directory = root / "raw_force"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"episode_{episode_index:06d}.npz", **payload)

    gaps = np.diff(raw_time)
    span = float(raw_time[-1] - raw_time[0])
    return {
        "episode_index": episode_index,
        "source_episode": episode.name,
        "raw_samples": len(raw_time),
        "frames": len(frame_time),
        "raw_rate_hz": (len(raw_time) - 1) / span if span > 0 else 0.0,
        "max_gap_ms": float(gaps.max() * 1000.0) if len(gaps) else 0.0,
        "duplicate_controller_timestamps": int(len(raw_time) - len(np.unique(raw_time))),
        "alignment_max_abs_diff_nm": alignment,
        "frames_outside_raw_span": outside,
        "robot_clock_scale": float(estimate["scale"]),
        "robot_clock_offset": float(estimate["offset"]),
    }


def weights_row(episode_index: int, episode: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """One episode_weights.jsonl line; missing readings stay null, never zero."""

    weights = metadata.get("weights") or {}
    row: dict[str, Any] = {"episode_index": episode_index, "source_episode": episode.name}
    for field in WEIGHT_FIELDS:
        value = weights.get(field)
        row[field] = None if value is None else float(value)
    row["unit"] = weights.get("unit")
    row["source"] = weights.get("source")
    # A reviewed episode carries its superseded values and the reason, so the
    # training side can tell a corrected outcome from a measured one.
    for field in ("outcome_source", "outcome_reason"):
        if weights.get(field) is not None:
            row[field] = weights[field]
    for field in ("measured_poured_g", "measured_spilled_g"):
        if field in weights:
            value = weights[field]
            row[field] = None if value is None else float(value)
    return row


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
        features=_features(include_joint_torque),
        use_videos=True,
        image_writer_threads=4,
    )

    raw_force_entries: list[dict[str, Any]] = []
    raw_force_missing: list[Path] = []
    weight_rows: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
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
                dataset.add_frame(frame)
        dataset.save_episode()
        # Written after save_episode so the sidecar index always matches the
        # episode LeRobot just committed. Raw force never gates the parquet:
        # an older episode simply gets no sidecar.
        entry = write_raw_force(Path(dataset.root), episode_index, episode, sync_reports[episode])
        if entry is None:
            raw_force_missing.append(episode)
        else:
            raw_force_entries.append(entry)
        weight_rows.append(weights_row(episode_index, episode, metadata))
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

    root_path = Path(dataset.root)
    meta_directory = root_path / "meta"
    meta_directory.mkdir(parents=True, exist_ok=True)
    with (meta_directory / "episode_weights.jsonl").open("w", encoding="utf-8") as handle:
        for row in weight_rows:
            handle.write(json.dumps(row) + "\n")
    raw_force_report = {
        "episodes_total": len(episodes),
        "episodes_with_raw_force": len(raw_force_entries),
        "episodes_without_raw_force": [path.name for path in raw_force_missing],
        "alignment_tolerance_nm": ALIGNMENT_TOLERANCE_NM,
        "episodes": raw_force_entries,
    }
    (meta_directory / "raw_force_report.json").write_text(
        json.dumps(raw_force_report, indent=2) + "\n", encoding="utf-8"
    )

    # Must run before the upload so the Hub copy carries the portable tag.
    rewritten = restore_sequence_feature_tags(root_path)
    print(f"Rewrote the parquet feature tag to 'Sequence' in {len(rewritten)} files for datasets 3.x readers.")

    with_weights = sum(1 for row in weight_rows if row["poured_g"] is not None)
    print(
        f"Raw force: {len(raw_force_entries)} of {len(episodes)} episodes "
        f"(weights present on {with_weights} of {len(weight_rows)})."
    )
    if raw_force_entries:
        rates = [entry["raw_rate_hz"] for entry in raw_force_entries]
        gaps = [entry["max_gap_ms"] for entry in raw_force_entries]
        worst = max(entry["alignment_max_abs_diff_nm"] for entry in raw_force_entries)
        print(
            f"  rate {min(rates):.1f}-{max(rates):.1f} Hz, largest gap {max(gaps):.1f} ms, "
            f"worst alignment error {worst:.3g} Nm"
        )
    if raw_force_missing:
        joined = "\n  ".join(path.name for path in raw_force_missing)
        print(f"  no raw force (recorded before it was captured):\n  {joined}")

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
