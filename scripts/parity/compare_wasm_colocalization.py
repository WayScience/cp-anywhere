"""Compare real Chromium WASM colocalization output with the saved CP
oracle.

First build the WASM package with wasm-pack. This script needs the
browser example's existing Node dependencies and Playwright Chromium
installation.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_IMAGES_OR_OBJECTS = {
    "Across entire image": (True, False),
    "Within objects": (False, True),
    "Both": (True, True),
}
_FAST_COSTES = {"Faster": "faster", "Fast": "fast", "Accurate": "accurate"}


def _matches(got: float, want: float) -> bool:
    if isinstance(got, dict):
        got = {"NaN": math.nan, "+Infinity": math.inf, "-Infinity": -math.inf}[
            got["nonfinite"]
        ]
    if math.isnan(want):
        return math.isnan(got)
    return math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


def _scale_max(fixture: dict) -> float:
    """``get_scale``'s resolved per-pair value, matching the Rust core."""
    scales = fixture.get("scales", {})
    return float(max(scales.values())) if scales else 255.0


def _as_list(value: object) -> list:
    """Normalize a scalar or list into a list (Image-scope fixture
    values are single-element lists while the WASM result holds plain
    floats)."""
    return value if isinstance(value, list) else [value]


def _compare_scope(name: str, scope: str, expected: dict, actual_values: dict) -> int:
    """Compare one case's named columns; return its scalar count."""
    if set(actual_values) != set(expected):
        raise AssertionError(f"{name}: selected {scope} column names differ")
    scalar_count = 0
    for column, values in expected.items():
        want_list = _as_list(values)
        got_list = _as_list(actual_values[column])
        if len(got_list) != len(want_list):
            raise AssertionError(f"{name}:{scope}.{column}: length mismatch")
        for got, want in zip(got_list, want_list, strict=True):
            if not _matches(got, want):
                raise AssertionError(f"{name}:{scope}.{column}: mismatch")
            scalar_count += 1
    return scalar_count


def _compare_case(name: str, expected: dict[str, dict], actual: dict[str, dict]) -> int:
    """Compare one case's Image/object columns; return its scalar count."""
    scalar_count = _compare_scope(
        f"{name}:Image", "Image", expected.get("Image", {}), actual["image"]
    )
    object_scopes = [scope for scope in expected if scope != "Image"]
    for scope in object_scopes:
        scalar_count += _compare_scope(name, scope, expected[scope], actual["objects"])
    if not object_scopes and actual["objects"]:
        raise AssertionError(f"{name}: unexpected object columns")
    return scalar_count


def compare() -> dict[str, int | str]:
    """Execute measureColocalizationJson in real Chromium and compare."""
    fixtures = sorted(
        (ROOT / "fixtures/colocalization").glob("*.json"), key=lambda p: p.stem
    )
    fixtures = [p for p in fixtures if p.stem != "manifest"]
    if not fixtures:
        raise ValueError("no real colocalization fixtures found")
    cases = []
    expectations = []
    for path in fixtures:
        fixture = json.loads(path.read_text())
        rows = len(next(iter(fixture["pixels"].values())))
        cols = len(next(iter(fixture["pixels"].values()))[0])
        settings = fixture["settings"]
        wants_image, wants_object = _IMAGES_OR_OBJECTS[settings["images_or_objects"]]
        labels_by_object = fixture["labels"]
        label_cases = list(labels_by_object.items()) or [("", None)]
        for object_name, label_rows in label_cases:
            cases.append(
                {
                    "images": [
                        {
                            "name": name,
                            "pixels": [v for row in rows_2d for v in row],
                        }
                        for name, rows_2d in fixture["pixels"].items()
                    ],
                    "rows": rows,
                    "cols": cols,
                    "masks": {
                        name: [v for row in rows_2d for v in row]
                        for name, rows_2d in (fixture.get("masks") or {}).items()
                    }
                    or None,
                    "labels": (
                        [v for row in label_rows for v in row]
                        if label_rows is not None
                        else None
                    ),
                    "threshold_percent": settings["thr"],
                    "fast_costes": _FAST_COSTES[settings["fast_costes"]],
                    "costes_scale_max": _scale_max(fixture),
                    "do_corr_and_slope": settings["do_corr_and_slope"],
                    "do_manders": settings["do_manders"],
                    "do_rwc": settings["do_rwc"],
                    "do_overlap": settings["do_overlap"],
                    "do_costes": settings["do_costes"],
                    "wants_image_measurements": wants_image,
                    "wants_object_measurements": wants_object
                    and label_rows is not None,
                }
            )
            scope = next(iter(labels_by_object), None)
            expectations.append((path.stem, scope, fixture["expected"]))
    completed = subprocess.run(
        ["node", str(ROOT / "examples/browser-vite/run-colocalization-check.mjs")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    outputs = [json.loads(line) for line in json.loads(completed.stdout)]
    if len(outputs) != len(cases):
        raise AssertionError("browser output case count does not match inputs")
    scalar_count = 0
    for (name, scope, expected), actual in zip(expectations, outputs, strict=True):
        if scope is None:
            # Image-only case: object columns must be absent. Reuse the
            # same comparator with an empty object-scope expectation by
            # dropping any "objects" the call should not have produced.
            expected_only_image = {"Image": expected.get("Image", {})}
            scalar_count += _compare_case(name, expected_only_image, actual)
        else:
            scalar_count += _compare_case(name, expected, actual)
    return {
        "runtime": "real Playwright Chromium, wasm32-unknown-unknown",
        "cases": len(cases),
        "scalar_comparisons": scalar_count,
        "mismatches": 0,
    }


def main() -> None:
    """Run the comparison and optionally save a replay report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare()
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
