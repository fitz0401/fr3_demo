"""Rewrite recorded frames at the stored size used by newer sessions.

Episodes recorded before the recorder resized on write hold native-resolution
JPEGs, which the LeRobot conversion downsizes to 320x180 anyway.  Rewriting them
reclaims most of that space without changing what training sees.  The rewrite is
lossy and irreversible, so it runs as a dry run unless ``--apply`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fr3_demo.recording import DEFAULT_IMAGE_SIZE


def _load_pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Resizing requires Pillow; run: pip install -e '.[recording]'") from error
    return Image


@dataclass
class EpisodeReport:
    episode: Path
    frames: int
    resized: int
    before_bytes: int
    after_bytes: int
    skipped: str | None = None

    @property
    def reclaimed(self) -> int:
        return self.before_bytes - self.after_bytes


def find_episodes(roots: Sequence[Path]) -> list[Path]:
    episodes: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if (resolved / "metadata.json").is_file():
            episodes.add(resolved)
            continue
        for metadata in resolved.glob("**/episode_*/metadata.json"):
            episodes.add(metadata.parent)
    return sorted(episodes)


def _resize_file(image_class, path: Path, size: tuple[int, int], apply: bool) -> tuple[int, int, bool]:
    """Return (before, after, resized) bytes for one frame."""

    before = path.stat().st_size
    with image_class.open(path) as picture:
        if picture.size == size:
            return before, before, False
        small = picture.convert("RGB").resize(size, image_class.Resampling.BICUBIC)
        if not apply:
            import io

            buffer = io.BytesIO()
            small.save(buffer, format="JPEG", quality=92, subsampling=0)
            return before, buffer.tell(), True
        temporary = path.with_suffix(".jpg.tmp")
        small.save(temporary, format="JPEG", quality=92, subsampling=0)
    # Replace only once the new file is fully written.
    temporary.replace(path)
    return before, path.stat().st_size, True


def downsize_episode(
    episode: Path,
    size: tuple[int, int],
    *,
    apply: bool,
    workers: int = 8,
) -> EpisodeReport:
    image_class = _load_pillow()
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        return EpisodeReport(episode, 0, 0, 0, 0, skipped="incomplete")

    frames = sorted((episode / "frames").glob("*/frame_*.jpg"))
    if not frames:
        return EpisodeReport(episode, 0, 0, 0, 0, skipped="no frames")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda path: _resize_file(image_class, path, size, apply), frames))

    before = sum(item[0] for item in results)
    after = sum(item[1] for item in results)
    resized = sum(1 for item in results if item[2])
    if apply and resized:
        metadata["image_size"] = {"width": size[0], "height": size[1], "resample": "bicubic"}
        metadata["resized_after_recording"] = True
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
    return EpisodeReport(episode, len(frames), resized, before, after)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite recorded JPEGs at the stored dataset size to reclaim disk.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dirs",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="session or episode directories to rewrite",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rewrite the frames; without this the run only reports what would change",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    episodes = find_episodes(args.data_dirs)
    if not episodes:
        parser.error(f"No episodes found under: {', '.join(str(path) for path in args.data_dirs)}")

    size = (args.width, args.height)
    print(f"{'apply' if args.apply else 'DRY RUN'}: {len(episodes)} episodes -> {size[0]}x{size[1]}\n")
    total_before = total_after = total_frames = 0
    for episode in episodes:
        report = downsize_episode(episode, size, apply=args.apply, workers=args.workers)
        if report.skipped:
            print(f"  {episode.parent.name}/{episode.name}: skipped ({report.skipped})")
            continue
        total_before += report.before_bytes
        total_after += report.after_bytes
        total_frames += report.resized
        print(
            f"  {episode.parent.name}/{episode.name}: {report.resized}/{report.frames} frames, "
            f"{report.before_bytes / 1e6:8.1f} -> {report.after_bytes / 1e6:7.1f} MB "
            f"(-{report.reclaimed / 1e6:.1f} MB)"
        )
    reclaimed = total_before - total_after
    print(
        f"\n{'Reclaimed' if args.apply else 'Would reclaim'} {reclaimed / 1e9:.2f} GB "
        f"({total_before / 1e9:.2f} -> {total_after / 1e9:.2f} GB) across {total_frames} frames."
    )
    if not args.apply:
        print("Re-run with --apply to rewrite. The originals are not recoverable afterwards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
