"""Compare real Chromium WASM texture output with the saved CP oracle.

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
    "Images": (True, False),
    "Objects": (False, True),
    "Both": (True, True),
}


def _matches(got: float, want: float) -> bool:
    if isinstance(got, dict):
        got = {"NaN": math.nan, "+Infinity": math.inf, "-Infinity": -math.inf}[
            got["nonfinite"]
        ]
    if math.isnan(want):
        return math.isnan(got)
    return math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


def _compare_case(name: str, expected: dict[str, dict], actual: dict[str, dict]) -> int:
    """Compare one case's Image/Cells columns; return its scalar count."""
    if set(actual["image"]) != set(expected["Image"]):
        raise AssertionError(f"{name}: selected Image column names differ")
    scalar_count = 0
    for column, expected_value in expected["Image"].items():
        if not _matches(actual["image"][column], expected_value):
            raise AssertionError(f"{name}:Image.{column}: mismatch")
        scalar_count += 1
    if set(actual["objects"]) != set(expected["Cells"]):
        raise AssertionError(f"{name}: selected Cells column names differ")
    for column, expected_values in expected["Cells"].items():
        actual_values = actual["objects"][column]
        if len(actual_values) != len(expected_values):
            raise AssertionError(f"{name}:Cells.{column}: length mismatch")
        for got, want in zip(actual_values, expected_values, strict=True):
            if not _matches(got, want):
                raise AssertionError(f"{name}:Cells.{column}: mismatch")
            scalar_count += 1
    return scalar_count


def compare() -> dict[str, int | str]:
    """Execute measureTextureJson in real Chromium and compare every case."""
    fixtures = sorted((ROOT / "fixtures/texture").glob("*.json"), key=lambda p: p.stem)
    fixtures = [p for p in fixtures if p.stem != "manifest"]
    if not fixtures:
        raise ValueError("no real texture fixtures found")
    cases = []
    expectations = []
    for path in fixtures:
        fixture = json.loads(path.read_text())
        pixels = [v for row in fixture["pixels"] for v in row]
        labels = [v for row in fixture["labels"] for v in row]
        rows = len(fixture["pixels"])
        cols = len(fixture["pixels"][0])
        settings = fixture["settings"]
        wants_image, wants_object = _IMAGES_OR_OBJECTS[settings["images_or_objects"]]
        cases.append(
            {
                "pixels": pixels,
                "labels": labels,
                "rows": rows,
                "cols": cols,
                "image_name": "Input",
                "gray_levels": settings["gray_levels"],
                "scales": settings["scales"],
                "wants_image_measurements": wants_image,
                "wants_object_measurements": wants_object,
            }
        )
        expectations.append((path.stem, fixture["expected"]))
    completed = subprocess.run(
        ["node", str(ROOT / "examples/browser-vite/run-texture-check.mjs")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    outputs = [json.loads(line) for line in json.loads(completed.stdout)]
    if len(outputs) != len(cases):
        raise AssertionError("browser output case count does not match inputs")
    scalar_count = sum(
        _compare_case(name, expected, actual)
        for (name, expected), actual in zip(expectations, outputs, strict=True)
    )
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
