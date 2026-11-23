#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Sync patched source files into every built environment, then refresh fingerprints.

Why this exists
---------------
A venv environment is a *copy* of the project (``uv pip install <path>``, not
``-e``), so editing ``src/`` does not reach an environment that was already
built. Worse, editing ``src/`` changes the source fingerprint, so the next
session decides the environment is stale and rebuilds it: ``setup.sh`` re-runs,
re-clones from GitHub, and on a machine behind a restricted mirror that clone
fails with exit 128 and takes the whole run down with it.

So a source edit made while a run is *not* active must be followed by both
steps, in this order:

1. copy the changed files into each environment's ``site-packages/exgentic``;
2. rewrite the stored fingerprint to match the tree.

Missing step 1 leaves an environment serving old code -- silently, because
nothing compares the copy against the tree at import time.  Missing step 2
triggers a rebuild on the next session.

Usage
-----
    python scripts/sync_envs.py            # sync + refresh, report what changed
    python scripts/sync_envs.py --check    # exit 1 if anything is out of sync
    python scripts/sync_envs.py --only tau2

Environments are discovered by scanning for ``.installed`` markers under
``~/.exgentic`` rather than by listing known names.  That is deliberate: the
set grows with every benchmark added, and a hardcoded list is exactly how the
tau2 environment got missed the first time.  The files to compare are found
the same way -- by walking ``src/exgentic`` -- for the same reason: a file
list cannot see a file that was just added, which is how ``session_refs.py``
reached the tree without reaching any environment.

This does not touch a running process.  Stop the run first: replacing a module
underneath a live interpreter can leave it half-loaded.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "exgentic"
DEFAULT_BASE = Path.home() / ".exgentic"

# Nothing is listed: the set of files to compare is the whole package tree.
#
# This used to be a hand-maintained ``PATCHED_FILES`` list, and that failed
# twice in the same way it failed for the environment list right below it.
# First ``agents/tool_calling`` went stale unseen; then ``session_refs.py`` was
# *added* while the list was in place, so the copy step -- which only iterated
# the listed names -- never created it, and every environment failed to import
# a module the tree had shipped.  Enumerating the source tree costs one rglob
# and cannot drift.
def source_files() -> list[str]:
    """Every ``.py`` under ``src/exgentic``, relative to that package root."""
    return sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def find_environments(base: Path) -> list[tuple[str, Path, Path]]:
    """Return ``(name, package_dir, marker_path)`` for every built venv."""
    found = []
    for marker in sorted(base.rglob(".installed")):
        env_dir = marker.parent
        try:
            data = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "venv" not in data:
            continue
        # site-packages layout is stable for the venv backend on CPython 3.12+.
        candidates = sorted((env_dir / "venv" / "lib").glob("python3.*/site-packages/exgentic"))
        if not candidates:
            continue
        name = str(env_dir.relative_to(base))
        found.append((name, candidates[-1], marker))
    return found


def current_fingerprint() -> str:
    """Compute the fingerprint the same way the runtime does."""
    sys.path.insert(0, str(SRC.parent))
    from exgentic.environment.helpers import get_source_fingerprint

    value = get_source_fingerprint(REPO_ROOT)
    if not value:
        raise SystemExit("could not compute source fingerprint (not a git checkout?)")
    return value


def stale_files(package_dir: Path) -> list[str]:
    """Files in the package copy that differ from the source tree.

    Missing files count: an environment built before a module existed is not
    a subset that happens to work, it is a broken package.
    """
    out = []
    for rel in source_files():
        src = SRC / rel
        dst = package_dir / rel
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            out.append(rel)
    return out


def extra_files(package_dir: Path) -> list[str]:
    """Modules present in the copy but gone from the tree (renames, deletes).

    Left behind, they keep being importable, so a deleted module appears to
    still work in every environment until the next rebuild.
    """
    wanted = set(source_files())
    out = []
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(package_dir))
        if rel not in wanted:
            out.append(rel)
    return sorted(out)


def drop_bytecode(package_dir: Path, rel: str) -> None:
    """Remove the .pyc for *rel* so it cannot shadow the new source."""
    target = package_dir / rel
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)


def apply(package_dir: Path, stale: list[str], extra: list[str]) -> None:
    for rel in stale:
        dst = package_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, dst)
        drop_bytecode(package_dir, rel)
    for rel in extra:
        (package_dir / rel).unlink(missing_ok=True)
        drop_bytecode(package_dir, rel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only, exit 1 if out of sync")
    parser.add_argument("--only", metavar="SUBSTRING", help="restrict to envs whose name contains this")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="environment root")
    args = parser.parse_args()

    envs = find_environments(Path(args.base))
    if args.only:
        envs = [e for e in envs if args.only in e[0]]
    if not envs:
        print("no environments found", file=sys.stderr)
        return 1

    fingerprint = current_fingerprint()
    print(f"source fingerprint: {fingerprint}")

    problems = 0
    for name, package_dir, marker in envs:
        stale = stale_files(package_dir)
        extra = extra_files(package_dir)
        data = json.loads(marker.read_text())
        stored = data.get("venv", {}).get("source_fingerprint")
        fingerprint_ok = stored == fingerprint

        if not stale and not extra and fingerprint_ok:
            print(f"  {name:<24s} up to date")
            continue

        print(f"  {name:<24s} " + ", ".join(filter(None, [
            f"{len(stale)} stale file(s)" if stale else "",
            f"{len(extra)} orphan module(s)" if extra else "",
            "" if fingerprint_ok else "fingerprint stale",
        ])))

        if args.check:
            problems += 1
            continue

        apply(package_dir, stale, extra)
        data.setdefault("venv", {})["source_fingerprint"] = fingerprint
        marker.write_text(json.dumps(data, indent=2))
        print(f"  {'':<24s} -> synced and fingerprint refreshed")

    if args.check and problems:
        print(f"\n{problems} environment(s) out of sync", file=sys.stderr)
        return 1
    print("\nall environments consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
