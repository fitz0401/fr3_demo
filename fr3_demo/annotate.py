"""Add language instructions to recorded demonstration episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def episode_metadata_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.expanduser().resolve().glob("**/episode_*/metadata.json")
        if not path.parent.name.endswith(".inprogress")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add language instructions to raw FR3 demonstration episodes.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="session directory or a parent containing sessions"
    )
    instruction_mode = parser.add_mutually_exclusive_group()
    instruction_mode.add_argument(
        "--language", help="apply this instruction to every selected episode instead of prompting"
    )
    instruction_mode.add_argument(
        "--all",
        dest="one_for_all",
        action="store_true",
        help="prompt once and apply the instruction to every unlabeled episode",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing non-empty annotations")
    args = parser.parse_args(argv)
    if args.one_for_all and args.overwrite:
        parser.error("--all always preserves labeled episodes and cannot be combined with --overwrite")

    paths = episode_metadata_paths(args.data_dir)
    if not paths:
        parser.error(f"no completed episodes found under {args.data_dir}")

    shared_instruction: str | None = None
    if args.one_for_all:
        has_unlabeled = any(
            not str(json.loads(path.read_text(encoding="utf-8")).get("language_instruction") or "").strip()
            for path in paths
        )
        if has_unlabeled:
            shared_instruction = input("Language for all unlabeled episodes (blank to skip): ").strip()

    updated = 0
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        existing = str(metadata.get("language_instruction") or "").strip()
        if existing and not args.overwrite:
            print(f"{path.parent}: {existing!r} (kept)")
            continue
        if args.one_for_all:
            instruction = shared_instruction or ""
        elif args.language is not None:
            instruction = args.language.strip()
        else:
            instruction = input(f"Language for {path.parent.name} (blank to skip): ").strip()
        if not instruction:
            print(f"{path.parent}: skipped")
            continue
        metadata["language_instruction"] = instruction
        _write_metadata(path, metadata)
        print(f"{path.parent}: {instruction!r}")
        updated += 1

    print(f"Updated {updated} of {len(paths)} episode annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
