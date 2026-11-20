"""Compare real Chromium WASM size/shape output with the saved CP oracle.

First build the WASM package with wasm-pack. This script needs the browser
example's existing Node dependencies and Playwright Chromium installation.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def family(name: str) -> str:
    """Distinguish the real module's optional column groups."""
    if name.startswith("AreaShape_Zernike_"):
        return "zernike"
    if "Moment" in name or "InertiaTensor" in name:
        return "advanced"
    return "standard"


def fixture_inputs() -> tuple[list[dict], list[dict]]:
    """Prepare strict JSON inputs and matching reference columns."""
    inputs = []
    expected = []
    for path in sorted((ROOT / "fixtures/size_shape").glob("*.json")):
        fixture = json.loads(path.read_text())
        labels = np.asarray(fixture["labels"], dtype=np.uint32)
        for zernikes, advanced in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            enabled = {"standard"}
            if zernikes:
                enabled.add("zernike")
            if advanced:
                enabled.add("advanced")
            inputs.append(
                {
                    "shape": list(labels.shape),
                    "images": {"DNA": [0.0] * labels.size},
                    "objects": {"Cells": labels.ravel().tolist()},
                    "image_id": path.stem,
                    "families": ["object_size_shape"],
                    "size_shape_calculate_zernikes": zernikes,
                    "size_shape_calculate_advanced": advanced,
                }
            )
            expected.append(
                {
                    name: values
                    for name, values in fixture["features"].items()
                    if family(name) in enabled
                }
            )
    if not inputs:
        raise ValueError("no real size/shape fixtures found")
    return inputs, expected


def context_inputs() -> tuple[list[dict], dict]:
    """Load both unchanged full-label planes and their CellProfiler scores."""
    context = ROOT / "fixtures/size_shape/zernike_context"
    scopes = json.loads((context / "oracle.json").read_text())["scopes"]
    inputs = []
    with np.load(context / "labels.npz", allow_pickle=False) as archive:
        for scope, reference in scopes.items():
            labels = archive[f"{scope}_labels"].astype("<u4", copy=False)
            if (
                list(labels.shape) != reference["shape"]
                or hashlib.sha256(labels.tobytes()).hexdigest()
                != reference["labels_sha256"]
            ):
                raise AssertionError(f"{scope}: full-label fixture changed")
            inputs.append(
                {
                    "shape": list(labels.shape),
                    "images": {"DNA": [0.0] * labels.size},
                    "objects": {scope: labels.ravel().tolist()},
                    "image_id": scope,
                    "families": ["object_size_shape"],
                    "size_shape_calculate_zernikes": True,
                    "size_shape_calculate_advanced": False,
                }
            )
    return inputs, scopes


def compare_context(outputs: list[dict], scopes: dict) -> int:
    """Check exact IDs and Zernike values from both browser entry points."""
    checked = 0
    for output, (scope, reference) in zip(outputs, scopes.items(), strict=True):
        ids = list(range(1, reference["label_count"] + 1))
        for entrypoint in ("json", "typed"):
            result = output[entrypoint]
            if result["object_ids"] != {scope: ids}:
                raise AssertionError(f"{entrypoint}.{scope}: object IDs differ")
            features = result["features"][scope]
            names = {name for name in features if family(name) == "zernike"}
            for oid, obj in reference["objects"].items():
                expected = obj["full_field"]
                if names != set(expected):
                    raise AssertionError(
                        f"{entrypoint}.{scope}: Zernike columns differ"
                    )
                for name, value in expected.items():
                    if len(features[name]) != len(ids):
                        raise AssertionError(
                            f"{entrypoint}.{scope}.{name}: length differs"
                        )
                    np.testing.assert_allclose(
                        features[name][int(oid) - 1],
                        value,
                        rtol=1e-10,
                        atol=1e-10,
                        err_msg=f"{entrypoint}.{scope}[{oid}].{name}",
                    )
                    checked += 1
    return checked


def compare() -> dict[str, int | str]:
    """Execute both WASM entry points and compare every selected column."""
    inputs, expected = fixture_inputs()
    full_fields, scopes = context_inputs()
    completed = subprocess.run(
        ["node", str(ROOT / "examples/browser-vite/run-size-shape-check.mjs")],
        input=json.dumps(inputs + full_fields, allow_nan=False),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    outputs = json.loads(completed.stdout)
    if len(outputs) != len(inputs) + len(full_fields):
        raise AssertionError("browser output case count does not match inputs")
    column_count = 0
    scalar_count = 0
    markers = {"NaN": np.nan, "+Infinity": np.inf, "-Infinity": -np.inf}
    for output, columns in zip(outputs[: len(inputs)], expected, strict=True):
        for entrypoint in ("json", "typed"):
            actual = output[entrypoint]["features"]["Cells"]
            if set(actual) != set(columns):
                raise AssertionError(f"{entrypoint}: selected column names differ")
            for name, values in columns.items():
                decoded = [
                    markers[v["nonfinite"]] if isinstance(v, dict) else v
                    for v in actual[name]
                ]
                np.testing.assert_allclose(
                    decoded,
                    values,
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                    err_msg=f"{entrypoint}.{name}",
                )
                column_count += 1
                scalar_count += len(values)
    return {
        "runtime": "real Playwright Chromium, wasm32-unknown-unknown",
        "cases": len(inputs),
        "entrypoints": 2,
        "column_comparisons": column_count,
        "scalar_comparisons": scalar_count,
        "full_label_cases": len(full_fields),
        "full_label_zernike_comparisons": compare_context(
            outputs[len(inputs) :], scopes
        ),
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
