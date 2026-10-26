# /// script
# requires-python = ">=3.10"
# dependencies = ["tomlkit>=0.13.3"]
# ///
"""
Copy `[dependency-groups].ml` into `[project.optional-dependencies].ml`.

The dependency group is the source of truth: the `ml` extra published to PyPI is
rewritten to match it exactly, with `{include-group = ...}` entries expanded.

Usage:
    uv run --script .github/scripts/sync_ml_extra.py [--check] [--notes-file PATH]
"""

import argparse
import re
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tomlkit


GROUP = "ml"
EXTRA = "ml"

_NAME_RE = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _package(requirement: str) -> str:
    match = _NAME_RE.match(requirement)
    return _normalize(match.group(1)) if match else requirement


def resolve_group(
    groups: Mapping[str, Sequence[Any]], name: str, stack: tuple[str, ...] = ()
) -> list[str]:
    """Flatten a PEP 735 dependency group, expanding `{include-group = ...}` entries."""
    keys = {_normalize(key): key for key in groups}
    key = keys.get(_normalize(name))
    if key is None:
        raise SystemExit(f"error: dependency group {name!r} is not defined")
    if key in stack:
        raise SystemExit(f"error: dependency group cycle: {' -> '.join((*stack, key))}")

    requirements: list[str] = []
    for entry in groups[key]:
        if isinstance(entry, str):
            requirements.append(str(entry))
        else:
            requirements.extend(
                resolve_group(groups, entry["include-group"], (*stack, key))
            )
    return list(dict.fromkeys(requirements))


def describe_changes(old: list[str], new: list[str]) -> list[str]:
    """Describe how `new` differs from `old` as human-readable lines."""
    removed = [req for req in old if req not in new]
    added = [req for req in new if req not in old]

    changes: list[str] = []
    for req in added:
        previous = [
            old_req for old_req in removed if _package(old_req) == _package(req)
        ]
        if len(previous) == 1:
            removed.remove(previous[0])
            changes.append(f"update `{previous[0]}` -> `{req}`")
        else:
            changes.append(f"add `{req}`")
    changes.extend(f"remove `{req}`" for req in removed)
    return changes or ["reorder entries"]


def main() -> int:
    """Run the sync and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=f"Sync the `{EXTRA}` extra with the `{GROUP}` dependency group."
    )
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report the difference; exit with 1 when out of sync.",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        help="Write a commit message body describing the sync here (only when something changed).",
    )
    args = parser.parse_args()

    doc = tomlkit.parse(args.pyproject.read_text(encoding="utf-8"))
    desired = resolve_group(doc.get("dependency-groups", {}), GROUP)
    extras = doc["project"].setdefault("optional-dependencies", tomlkit.table())
    current = [str(req) for req in extras.get(EXTRA, [])]
    if current == desired:
        print(f"`{EXTRA}` extra is in sync with the `{GROUP}` dependency group")
        return 0

    notes = "\n".join(
        [
            f"- Sync `project.optional-dependencies.{EXTRA}` with `dependency-groups.{GROUP}`:",
            *(f"  - {change}" for change in describe_changes(current, desired)),
        ]
    )
    print(notes)
    if args.check:
        return 1

    array = tomlkit.array()
    array.extend(desired)
    extras[EXTRA] = array.multiline(True)
    args.pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")
    if args.notes_file is not None:
        args.notes_file.write_text(notes + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
