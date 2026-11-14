#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
reference_root="${CP_ANYWHERE_REFERENCE_ROOT:-$HOME/.cache/cp-anywhere/reference}"
source_dir="${CP_ANYWHERE_REFERENCE_SOURCE:-$reference_root/cellprofiler}"
python="$reference_root/venv/bin/python"
expected="2df090ff281cff80943fe679a1c3192fbda9cd6d"

if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected" ]]; then
    echo "CellProfiler source commit does not match reference/source-lock.json" >&2
    exit 1
fi
if [[ ! -x "$python" ]]; then
    echo "Reference Python is missing. Run reference/install.sh first." >&2
    exit 1
fi
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$repo_root/scripts/reference/generate.py" "$@"
