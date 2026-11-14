#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
reference_root="${CP_ANYWHERE_REFERENCE_ROOT:-$HOME/.cache/cp-anywhere/reference}"
source_dir="${CP_ANYWHERE_REFERENCE_SOURCE:-$reference_root/cellprofiler}"
venv_dir="$reference_root/venv"
source_commit="2df090ff281cff80943fe679a1c3192fbda9cd6d"

mkdir -p "$reference_root"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/CellProfiler/CellProfiler.git "$source_dir"
    git -C "$source_dir" checkout --detach "$source_commit"
fi
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$source_commit" ]]; then
    echo "CellProfiler source commit mismatch: $actual_commit" >&2
    exit 1
fi

uv python install 3.9.25
uv venv --clear --python 3.9.25 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" --no-deps \
    'numpy==1.23.5' 'setuptools==75.8.0' 'wheel==0.45.1' 'cython==0.29.37'
CFLAGS="${CFLAGS:-} -Wno-int-conversion" \
    uv pip install --python "$venv_dir/bin/python" --no-build-isolation \
    'python-javabridge==4.0.3'
uv pip install --python "$venv_dir/bin/python" --no-deps \
    -r "$repo_root/reference/requirements.lock"

echo "Reference Python: $venv_dir/bin/python"
echo "Reference source: $source_dir"
