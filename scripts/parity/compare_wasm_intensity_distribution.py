"""Compare real Chromium WASM intensity distribution output with the
saved CP oracle.

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
_ZERNIKES = {
    "None": "none",
    "Magnitudes only": "magnitudes",
    "Magnitudes and phase": "magnitudes_and_phase",
}


def _matches(actual: list, expected: list) -> bool:
    if len(actual) != len(expected):
        return False
    markers = {"NaN": math.nan, "+Infinity": math.inf, "-Infinity": -math.inf}
    for actual_value, want in zip(actual, expected, strict=True):
        got = (
            markers[actual_value["nonfinite"]]
            if isinstance(actual_value, dict)
            else actual_value
        )
        if math.isnan(want):
            if not math.isnan(got):
                return False
        elif not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9):
            return False
    return True


def compare() -> dict[str, int | str]:
    """Execute measureIntensityDistributionJson in real Chromium and
    compare every case."""
    fixtures = sorted(
        (ROOT / "fixtures/intensity_distribution").glob("*.json"),
        key=lambda p: p.stem,
    )
    fixtures = [p for p in fixtures if p.stem != "manifest"]
    if not fixtures:
        raise ValueError("no real intensity distribution fixtures found")
    cases = []
    expectations = []
    for path in fixtures:
        fixture = json.loads(path.read_text())
        pixels = [v for row in fixture["pixels"] for v in row]
        labels = [v for row in fixture["labels"] for v in row]
        rows = len(fixture["pixels"])
        cols = len(fixture["pixels"][0])
        settings = fixture["settings"]
        cases.append(
            {
                "pixels": pixels,
                "labels": labels,
                "rows": rows,
                "cols": cols,
                "image_name": "Input",
                "bin_count": settings["bin_count"],
                "wants_scaled": settings["wants_scaled"],
                "maximum_radius": settings["maximum_radius"],
                "zernikes": _ZERNIKES[settings["wants_zernikes"]],
                "zernike_degree": settings["zernike_degree"],
            }
        )
        expectations.append((path.stem, fixture["expected"]))
    completed = subprocess.run(
        [
            "node",
            str(ROOT / "examples/browser-vite/run-intensity-distribution-check.mjs"),
        ],
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
    for (name, expected), actual in zip(expectations, outputs, strict=True):
        if set(actual) != set(expected):
            raise AssertionError(f"{name}: selected column names differ")
        for column, expected_values in expected.items():
            if not _matches(actual[column], expected_values):
                raise AssertionError(f"{name}:{column}: mismatch")
            scalar_count += len(expected_values)
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
