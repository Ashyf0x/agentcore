#!/usr/bin/env bash
set -euo pipefail

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Error: git-lfs is required but not installed. Install it first: brew install git-lfs (macOS) or apt-get install git-lfs (Linux)" >&2
    exit 1
fi

APPWORLD_ROOT="."
export APPWORLD_ROOT

TMPDIR="$(mktemp -d)"
git lfs install >/dev/null 2>&1 || true
git clone https://github.com/StonyBrookNLP/appworld.git "$TMPDIR/appworld"
cd "$TMPDIR/appworld"
git checkout edc960129fa6889c2b381715ecd108982029f6d1
git lfs pull

uv pip install "."

# `uv pip install .` can re-resolve appworld's unbounded `fastapi>=0.110`
# and pull 0.162+, which no longer accepts `Path(embed=...)`. Re-pin at the
# end so a fresh install always ends up on a version that builds all apps.
uv pip install "fastapi==0.115.6"

python -m appworld.cli install

cd - >/dev/null 2>&1 || true
rm -rf "$TMPDIR"
python -m appworld.cli download data --root "."
