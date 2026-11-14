#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="${CP_ANYWHERE_CP_MEASURE_SOURCE:?Set CP_ANYWHERE_CP_MEASURE_SOURCE to the pinned source checkout}"
venv="${1:-${TMPDIR:-/tmp}/cp-anywhere-cp-measure-baseline/venv}"
commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$commit" != "cdf693187e3dcaf3d067da7b58bdd4ab8896f91e" ]]; then
  echo "cp_measure source commit mismatch: $commit" >&2
  exit 1
fi
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.11 "$venv"
fi
uv pip sync "$root/benchmarks/cp_measure-requirements.lock" --python "$venv/bin/python"
uv pip install --no-deps --python "$venv/bin/python" "$source_dir"
"$venv/bin/python" -c 'import cp_measure; print(cp_measure.__file__)'
