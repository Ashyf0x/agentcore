#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
#
# Refresh the vendored agentcore tree from a working checkout.
#
# Usage:
#   scripts/sync_agentcore.sh /path/to/agent-runtime
#
# The adapter under src/exgentic/agents/agentcore/ carries a copy of the
# agentcore runtime so a normal install can run it without a separate checkout.
# That copy is a snapshot: run this after changing the runtime, or the numbers a
# run produces stop matching the code they are attributed to.
#
# Only the subpackages the evaluation loop reaches for are copied. The upstream
# `app/` CLI and `tools/mcp.py` are skipped on purpose: nothing on this path
# imports them, and they would pull in `mcp` and `python-dotenv` for nothing.
#
# The copied modules keep their `from agentcore.X import Y` imports transformed
# to relative form, which is what lets the copy live inside Exgentic without
# colliding with an installed `agentcore`.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <path-to-agent-runtime>" >&2
    echo "  e.g. $0 ~/code/agent-runtime" >&2
    exit 2
fi

SOURCE_ROOT="$1"
SOURCE_PKG="$SOURCE_ROOT/agentcore"

if [ ! -f "$SOURCE_PKG/__init__.py" ]; then
    echo "error: $SOURCE_PKG is not an agentcore package" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_PKG="$REPO_ROOT/src/exgentic/agents/agentcore"

SUBPACKAGES=(context core memory observability storage tools)

echo "source: $SOURCE_PKG"
echo "dest:   $DEST_PKG"

for name in "${SUBPACKAGES[@]}"; do
    if [ ! -d "$SOURCE_PKG/$name" ]; then
        echo "error: missing subpackage '$name' in $SOURCE_PKG" >&2
        exit 1
    fi
    rm -rf "${DEST_PKG:?}/$name"
    cp -r "$SOURCE_PKG/$name" "$DEST_PKG/$name"
    find "$DEST_PKG/$name" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
done

# Modules that are not on the evaluation path. Removing them keeps the copy free
# of the `mcp` and `python-dotenv` dependencies the upstream extras need.
rm -f "$DEST_PKG/tools/mcp.py"

# Rewrite the absolute intra-package imports into relative ones. The mapping is
# depth-aware: `agentcore.core.types` from inside `context/` is `..core.types`,
# but from inside `core/` it must collapse to `.types` — a naive
# `".." * depth + module` would produce `..core.types`, which resolves to the
# same object but is not what a hand-written module would say, and it makes the
# synced copy drift from the source every time.
#
# Existing line endings are preserved: the upstream checkout may be CRLF, and
# normalising it here would turn every sync into a whole-file diff.
python - "$DEST_PKG" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = re.compile(r"^(?P<indent>\s*)from agentcore\.(?P<module>[\w.]+) import (?P<rest>.+)$")


def relativise(module: str, depth: int, own_package: str) -> str:
    """Turn `module` into the shortest correct relative import spec."""
    # `agentcore.a.b` seen from depth-1 package `pkg`: drop `agentcore`, then
    # drop a repeated `pkg` prefix so `.b` wins over `..pkg.b`.
    parts = module.split(".")
    if depth == 1 and parts and parts[0] == own_package:
        parts = parts[1:]
        return "." + ".".join(parts) if len(parts) > 1 else "." + parts[0]
    return "." * (depth + 1) + ".".join(parts)


for path in sorted(root.rglob("*.py")):
    if path.parent == root:
        continue  # adapter modules, not vendored ones
    rel = path.relative_to(root)
    if path.name == "__init__.py":
        own_package = rel.parts[0] if len(rel.parts) > 1 else ""
        depth = 1 if len(rel.parts) > 1 else 0
    else:
        own_package = rel.parts[0]
        depth = len(rel.parts) - 1
    if depth < 1:
        continue

    raw = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    changed = False
    lines = []
    for line in raw.decode("utf-8").splitlines():
        match = pattern.match(line)
        if match:
            spec = relativise(match.group("module"), depth, own_package)
            line = f"{match.group('indent')}from {spec} import {match.group('rest')}"
            changed = True
        lines.append(line)
    if changed:
        path.write_bytes(newline.join(line.encode("utf-8") for line in lines) + newline)
        print(f"rewrote imports: {rel}")
PY

echo
count=$(find "$DEST_PKG" -mindepth 2 -name '*.py' | wc -l | tr -d ' ')
echo "vendored $count modules"
echo "next: run 'pytest tests/agents/test_agentcore_integration.py' to confirm the"
echo "      copy still matches the behaviour the adapter expects"
