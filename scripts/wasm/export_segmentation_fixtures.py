"""Export the saved segmentation golden fixtures for the browser demo.

Mirrors `scripts/wasm/export_fixtures.py`'s pattern (typed/raw binary for
bulk numeric data, small JSON for metadata) but for
`fixtures/segmentation/` -- the 100 cases independently verified against
the real CellProfiler 4.2.8 oracle by
`tests/segmentation/test_reference_alignment.py`. The browser demo's
in-worker `identifyPrimaryObjects` call is checked against the exact
same saved, checksummed label arrays, proving the WASM build matches the
oracle, not just the native/Python build.

Pixel values are exported *already rescaled* (the same rescale
`cp_anywhere.identify_primary_objects(..., rescale=case["rescale"])`
would apply before calling the Rust core): the Rust/WASM core itself is
unit-agnostic and has no rescale step of its own (see
`crates/cp_anywhere_segmentation`'s module docs), so the browser-side
call must receive pixels already in the units the oracle's own
`threshold`/`threshold_value` was computed against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "fixtures/segmentation"

# Settings fields `ConfigParams` (crates/cp_anywhere_segmentation) expects;
# anything a fixture's raw `settings` dict omits is left out of the
# exported config JSON too, so `build_config`'s own defaults apply --
# matching the native/Python test harness's behavior exactly, not
# re-deriving a second set of defaults here.
_PASSTHROUGH_SETTINGS_FIELDS = (
    "threshold_method",
    "threshold_value",
    "threshold_correction_factor",
    "threshold_range",
    "exclude_size",
    "min_diameter",
    "max_diameter",
    "exclude_border_objects",
    "maximum_object_count",
    "fill_holes",
    "threshold_smoothing_scale",
    "assign_middle_to_foreground",
    "robust_background_lower_outlier_fraction",
    "robust_background_upper_outlier_fraction",
    "robust_background_averaging_method",
    "robust_background_variance_method",
    "robust_background_number_of_deviations",
    "log_transform",
    "threshold_scope",
    "adaptive_window_size",
    "unclump_method",
    "watershed_method",
    "smoothing_filter_size",
    "maxima_suppression_size",
    "low_resolution_maxima",
)


def _rescale(pixels: np.ndarray, rescale: bool) -> np.ndarray:
    if rescale and np.issubdtype(pixels.dtype, np.integer):
        max_value = np.iinfo(pixels.dtype).max
        return pixels.astype(np.float64) / float(max_value)
    return pixels.astype(np.float64)


def write_fixture(case: dict, out_dir: Path) -> None:
    name = case["name"]
    case_json = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    pixels = np.load(FIXTURE_DIR / name / "pixels.npy")
    pixels = _rescale(pixels, case_json["rescale"])
    mask_path = FIXTURE_DIR / name / "mask.npy"
    has_mask = mask_path.exists()
    settings = {
        key: value
        for key, value in case_json["settings"].items()
        if key in _PASSTHROUGH_SETTINGS_FIELDS
    }
    metadata = {
        "shape": list(pixels.shape),
        "settings": settings,
        "has_mask": has_mask,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(metadata))
    (out_dir / f"{name}.pixels.f64").write_bytes(pixels.ravel().tobytes())
    if has_mask:
        mask = np.load(mask_path).astype(np.uint8)
        (out_dir / f"{name}.mask.u8").write_bytes(mask.ravel().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "examples/browser-vite/public/segmentation-fixtures",
    )
    args = parser.parse_args()
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for case in manifest["cases"]:
        write_fixture(case, args.out_dir)
        names.append(case["name"])
    (args.out_dir / "index.json").write_text(json.dumps({"fixtures": names}))
    print(json.dumps({"written": len(names), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
