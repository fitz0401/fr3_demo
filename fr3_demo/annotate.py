"""Add language instructions to recorded demonstration episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_language(path: Path, instruction: str) -> None:
    """Write only the instruction, into the metadata as it is on disk now.

    A session can still be recording: a replay that finishes while a prompt
    waits links itself into its demonstration's metadata, and writing back a
    copy read before the prompt would drop that link.
    """

    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["language_instruction"] = instruction
    _write_metadata(path, metadata)


def _demonstration_language(path: Path, metadata: dict) -> str | None:
    """For a replay, its demonstration's instruction ("" when it has none); None for any other episode."""

    name = metadata.get("replay_of")
    if not name:
        return None
    demonstration = path.parent.parent / str(name) / "metadata.json"
    if not demonstration.is_file():
        return ""
    return str(json.loads(demonstration.read_text(encoding="utf-8")).get("language_instruction") or "").strip()


def episode_metadata_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.expanduser().resolve().glob("**/episode_*/metadata.json")
        if not path.parent.name.endswith(".inprogress")
    )


def _annotate_weights(path: Path, metadata: dict, args: argparse.Namespace) -> bool:
    """Ask for the scale readings the operator took around one pour."""

    from fr3_demo.recording import WEIGHT_READINGS, record_episode_weights

    existing = metadata.get("weights")
    if existing and not args.replace:
        print(f"{path.parent}: {existing} (kept)")
        return False
    readings: dict[str, float] = {}
    for key, label in WEIGHT_READINGS:
        answer = input(f"{path.parent.name} - {label} in grams (blank to skip): ").strip()
        if not answer:
            continue
        try:
            readings[key] = float(answer)
        except ValueError:
            print("  not a number; skipped")
    if not readings:
        return False
    weights = record_episode_weights(path.parent, readings)["weights"]
    # Keep the caller's copy in step; it is rewritten when the language is set.
    metadata["weights"] = weights
    print(f"{path.parent}: {weights}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add language instructions to raw FR3 demonstration episodes.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="session directory or a parent containing sessions"
    )
    instruction_mode = parser.add_mutually_exclusive_group()
    instruction_mode.add_argument(
        "--language", help="apply this instruction to every selected episode instead of prompting"
    )
    # Without --language or --all, a robot replay (fr3-collect --replay) is not asked about:
    # it takes its demonstration's instruction, which is asked first (it records first).
    instruction_mode.add_argument(
        "--all",
        dest="one_for_all",
        action="store_true",
        help="prompt once and apply the instruction to every unlabeled episode",
    )
    parser.add_argument(
        "--replace",
        "--overwrite",
        dest="replace",
        action="store_true",
        help="replace existing non-empty annotations; with --all, replace every episode from one prompt",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="also ask for the bottle and cup scale readings in grams for each episode",
    )
    args = parser.parse_args(argv)

    paths = episode_metadata_paths(args.data_dir)
    if not paths:
        parser.error(f"no completed episodes found under {args.data_dir}")

    shared_instruction: str | None = None
    if args.one_for_all:
        has_target = args.replace or any(
            not str(json.loads(path.read_text(encoding="utf-8")).get("language_instruction") or "").strip()
            for path in paths
        )
        if has_target:
            target = "all episodes" if args.replace else "all unlabeled episodes"
            shared_instruction = input(f"Language for {target} (blank to skip): ").strip()

    updated = 0
    weighed = 0
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if args.weights:
            weighed += _annotate_weights(path, metadata, args)
        existing = str(metadata.get("language_instruction") or "").strip()
        inherited = None if args.one_for_all or args.language is not None else _demonstration_language(path, metadata)
        if inherited is not None:
            # A replay does what its demonstration did: same instruction, and
            # it follows when the demonstration's is replaced.
            if existing and (existing == inherited or not args.replace):
                note = "" if existing == inherited or not inherited else f"; its demonstration says {inherited!r}"
                print(f"{path.parent}: {existing!r} (kept{note})")
                continue
            if not inherited:
                print(f"{path.parent}: skipped (its demonstration {metadata['replay_of']} has no instruction yet)")
                continue
            _set_language(path, inherited)
            print(f"{path.parent}: {inherited!r} (from its demonstration {metadata['replay_of']})")
            updated += 1
            continue
        if existing and not args.replace:
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
        _set_language(path, instruction)
        print(f"{path.parent}: {instruction!r}")
        updated += 1

    if args.weights:
        print(f"Recorded weights for {weighed} of {len(paths)} episodes.")
    print(f"Updated {updated} of {len(paths)} episode annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
