"""Compare the same primary-object batch in native Rust, Python, and Chromium."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

import cp_anywhere as cp

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "fixtures/cppipe/declump_shape_propagate_manual_smoothing.cppipe"
FIELDS = ("real_cell_cc0", "real_bbbc007", "empty_field")


def _manifest(directory: Path, pipeline: cp.Pipeline) -> dict[str, dict[str, Path]]:
    (step,) = pipeline.segmentation_steps
    settings = dict(step.config)
    settings["threshold_method"] = settings.pop("threshold")
    fields = []
    sources = {}
    for name in FIELDS:
        source = (
            ROOT / "fixtures/real" / ("real_bbbc007" if name == "empty_field" else name)
        )
        pixels = cp.load_array(source / "DNA.npy").astype(np.float32) / 255
        if name == "empty_field":
            pixels = np.zeros_like(pixels)
        # Use exactly the same prepared pixels for segmentation and measurement.
        pixels.astype("<f8").tofile(directory / f"{name}.f64")
        np.save(directory / f"{name}.npy", pixels, allow_pickle=False)
        fields.append(
            {"image_id": name, "shape": pixels.shape, "pixels": f"{name}.f64"}
        )
        sources[name] = {step.input_image: directory / f"{name}.npy"}
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "fields": fields,
                "settings": settings,
                "image_name": step.input_image,
                "object_name": step.output_object,
                "families": [module.family for module in pipeline.measurement_steps],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return sources


def _python_results(
    pipeline: cp.Pipeline,
    sources: dict[str, dict[str, Path]],
    directory: Path,
    workers: int,
    backend: Literal["thread", "process"],
) -> list[dict]:
    results = []
    for field in cp.run_batch(pipeline, sources, workers=workers, backend=backend):
        (seg,) = field.primary.values()
        if field.measurements is None:
            raise ValueError(f"Missing measurements for {field.image_id}")
        output = directory / "python-measurements.json"
        field.measurements.to_json(output)
        results.append(
            {
                "imageId": field.image_id,
                "labels": seg.labels.ravel().tolist(),
                "smallRemovedLabels": seg.small_removed_labels.ravel().tolist(),
                "uneditedLabels": seg.unedited_labels.ravel().tolist(),
                "objectCount": seg.object_count,
                "threshold": seg.threshold,
                "measurements": json.loads(output.read_text(encoding="utf-8")),
            }
        )
    return results


def _compare(expected: object, actual: object, path: str, *, approximate: bool) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(
                f"{path}: missing {expected.keys() - actual.keys()}, "
                f"extra {actual.keys() - expected.keys()}"
            )
        for key in expected:
            _compare(
                expected[key], actual[key], f"{path}.{key}", approximate=approximate
            )
        return
    if (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}[{index}]", approximate=approximate)
        return
    # Only quantitative floating results may differ by round-off; no tolerance
    # for integer outputs, keys, IDs, labels, counts or nonfinite markers.
    if (
        approximate
        and isinstance(expected, float)
        and isinstance(actual, (float, int))
        and not isinstance(actual, bool)
        and (".features." in path or path.endswith(".threshold"))
        and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
    ):
        return
    if expected != actual or isinstance(expected, bool) != isinstance(actual, bool):
        raise ValueError(f"{path}: expected {expected!r}, got {actual!r}")


def main() -> None:
    """Run real batch paths and reject any unaccounted difference."""
    pipeline = cp.load_pipeline(PIPELINE)
    # Constant zero with global Otsu is all foreground at threshold zero.
    # Use size filtering so the field-sized object is excluded, as in the
    # existing pinned CellProfiler empty-field regression.
    (primary,) = pipeline.segmentation_steps
    primary.config.update(exclude_size=True, max_diameter=40.0)
    with tempfile.TemporaryDirectory(prefix="cp-anywhere-batch-parity-") as temp:
        directory = Path(temp)
        sources = _manifest(directory, pipeline)
        expected = _python_results(pipeline, sources, directory, 1, "thread")
        if not all(field["objectCount"] > 0 for field in expected[:-1]):
            raise ValueError("Expected foreground objects in both microscopy fixtures")
        if expected[-1]["objectCount"] != 0:
            raise ValueError("Empty field unexpectedly has objects")
        if not any(field["measurements"]["relationships"] for field in expected):
            raise ValueError("Batch must exercise real neighbor relationships")
        for backend in ("thread", "process"):
            actual = _python_results(pipeline, sources, directory, 2, backend)
            _compare(expected, actual, f"Python {backend}", approximate=False)
        rust = subprocess.check_output(
            [
                "cargo",
                "run",
                "--locked",
                "--quiet",
                "-p",
                "cp-anywhere-wasm",
                "--example",
                "batch_parity",
                "--",
                str(directory / "manifest.json"),
            ],
            cwd=ROOT,
            timeout=180,
        )
        _compare(expected, json.loads(rust), "Rust", approximate=True)
        browser = directory / "browser.json"
        subprocess.run(
            [
                "node",
                "examples/browser-vite/run-batch-check.mjs",
                str(directory),
                str(browser),
            ],
            cwd=ROOT,
            check=True,
            timeout=90,
        )
        _compare(
            expected,
            json.loads(browser.read_text(encoding="utf-8")),
            "Chromium WASM",
            approximate=True,
        )
    print(
        f"Batch parity passed: {len(expected)} fields; "
        "native Rust, Python, and Chromium WASM"
    )


if __name__ == "__main__":
    main()
