"""Compare the real in-browser WASM segmentation result with the saved
golden segmentation fixtures.

Reads the JSON written by `examples/browser-vite/run-browser-check.mjs`
(itself produced by a real Playwright Chromium run of the real Vite
production build, using a web worker's `identifyPrimaryObjects` call --
see `scripts/wasm/export_segmentation_fixtures.py`) and checks every
fixture's labels/object_count/threshold EXACTLY against
`fixtures/segmentation/<name>/labels.npy` and its sidecar JSON, the same
oracle `tests/segmentation/test_reference_alignment.py` checks the
native/Python build against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "fixtures/segmentation"
# Threshold is float32 inside real CellProfiler but float64 here (see
# tests/segmentation/test_reference_alignment.py's THRESHOLD_TOLERANCE);
# labels must still match exactly, with no tolerance.
THRESHOLD_TOLERANCE = 1e-5
# Same known, narrow, label-safe gap documented in
# tests/segmentation/test_reference_alignment.py's
# _KNOWN_THRESHOLD_MAP_TOLERANCE_GAP (adaptive, 3-class Otsu,
# log-transformed, a deliberately tiny 4px window -- float32-vs-float64
# histogram-bin sensitivity, not a WASM-specific bug: the same gap shows
# up here because it's the identical Rust core, just a different FFI).
_KNOWN_THRESHOLD_TOLERANCE_GAP = {
    "synthetic_adaptive_otsu_three_class_log_True_range_0.2": 0.02,
}


def _read_labels_npy(path: Path) -> list[int]:
    """Read a saved uint32 .npy array's flat values."""
    return np.load(path).astype("uint32").ravel().tolist()


def compare_fixture(name: str, actual: dict) -> dict:
    case = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    expected_labels = _read_labels_npy(FIXTURE_DIR / name / "labels.npy")
    actual_labels = actual["labels"]
    labels_match = actual_labels == expected_labels
    object_count_match = actual["object_count"] == case["object_count"]
    tolerance = _KNOWN_THRESHOLD_TOLERANCE_GAP.get(name, THRESHOLD_TOLERANCE)
    threshold_match = abs(actual["threshold"] - case["threshold"]) <= tolerance
    status = (
        "pass" if (labels_match and object_count_match and threshold_match) else "fail"
    )
    row = {
        "fixture": name,
        "status": status,
        "labels_match": labels_match,
        "object_count_match": object_count_match,
        "threshold_match": threshold_match,
        "expected_object_count": case["object_count"],
        "actual_object_count": actual["object_count"],
        "expected_threshold": case["threshold"],
        "actual_threshold": actual["threshold"],
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "examples/browser-vite/browser-results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/reports/wasm-segmentation-parity.json",
    )
    args = parser.parse_args()
    browser = json.loads(args.results.read_text())
    if browser["status"] != "done" or browser["errors"]:
        print(
            json.dumps(
                {
                    "error": "browser run did not complete cleanly",
                    "status": browser["status"],
                    "errors": browser["errors"],
                }
            )
        )
        raise SystemExit(1)
    segmentation = browser["results"]["segmentation"]
    rows = [compare_fixture(name, actual) for name, actual in segmentation.items()]
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    all_names = {case["name"] for case in manifest["cases"]}
    missing = sorted(all_names - set(segmentation))
    passed = sum(1 for row in rows if row["status"] == "pass")
    payload = {
        "reference": "CellProfiler 4.2.8",
        "engine": "cp-anywhere WASM identifyPrimaryObjects "
        "(Playwright Chromium, real worker)",
        "fixtures_checked": len(rows),
        "fixtures_declared": len(all_names),
        "missing_fixtures": missing,
        "passed": passed,
        "failed": len(rows) - passed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fixtures_checked": len(rows),
                "passed": passed,
                "failed": len(rows) - passed,
                "missing": len(missing),
                "output": str(args.output),
            }
        )
    )
    if missing or any(row["status"] != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
