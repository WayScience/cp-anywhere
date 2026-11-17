"""Verify MeasureObjectNeighbors "Expand until adjacent" through the
real WASM boundary (real Playwright Chromium), against the same real
CellProfiler 4.2.8 oracle fixtures used by the Rust core and Python
tests (fixtures/neighbors_expand/module_*.json).

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


def _matches(got: float, want: float) -> bool:
    if isinstance(got, dict):
        got = {"NaN": math.nan, "+Infinity": math.inf, "-Infinity": -math.inf}[
            got["nonfinite"]
        ]
    if math.isnan(want):
        return math.isnan(got)
    return math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


def _compare_case(name: str, expected: dict, actual: dict) -> int:
    """Compare one case's Cells feature columns; return its scalar count."""
    features = actual["features"]["Cells"]
    scalar_count = 0
    for column, values in expected["features"].items():
        got = features[column]
        if len(got) != len(values):
            raise AssertionError(f"{name}:{column}: length mismatch")
        for got_value, want in zip(got, values, strict=True):
            if not _matches(got_value, want):
                raise AssertionError(f"{name}:{column}: mismatch")
            scalar_count += 1
    got_edges = [
        (edge["first_object_id"], edge["second_object_id"])
        for edge in actual["relationships"]
    ]
    want_edges = [tuple(edge) for edge in expected["relationships"]]
    if got_edges != want_edges:
        raise AssertionError(f"{name}: relationship edges differ: {got_edges}")
    return scalar_count


def compare() -> dict[str, int | str]:
    """Execute measureJson in real Chromium and compare."""
    fixtures = sorted((ROOT / "fixtures/neighbors_expand").glob("module_*.json"))
    if not fixtures:
        raise ValueError("no real neighbors_expand module fixtures found")
    cases = []
    expectations = []
    for path in fixtures:
        fixture = json.loads(path.read_text())
        labels = fixture["labels"]
        rows, cols = len(labels), len(labels[0])
        flat_labels = [v for row in labels for v in row]
        cases.append(
            {
                "shape": [rows, cols],
                "images": {"DNA": [0.0] * (rows * cols)},
                "objects": {"Cells": flat_labels},
                "image_id": path.stem,
                "families": ["neighbors"],
                "neighbors_scopes": ["Cells"],
                "neighbors_expand": ["Cells"],
            }
        )
        expectations.append((path.stem, fixture))
    completed = subprocess.run(
        ["node", str(ROOT / "examples/browser-vite/run-neighbors-expand-check.mjs")],
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
    for (name, fixture), actual in zip(expectations, outputs, strict=True):
        scalar_count += _compare_case(name, fixture, actual)
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
