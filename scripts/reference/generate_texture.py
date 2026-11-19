# ty: ignore[import-not-found]
"""Run real CellProfiler MeasureTexture; capture 2D Haralick-texture fixtures.

Scope: 2D only (this port does not support 3D volumetric correspondence
vectors); real Module.run, real mahotas.features.haralick (the module's
actual numeric core -- not reimplemented or approximated), real
Measurements and pipeline dumps.

Run with the reference environment's Python and CellProfiler on PYTHONPATH.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import cellprofiler_core.preferences
import mahotas
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules import measuretexture as mt
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/texture"
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"


class _Settings(TypedDict):
    """A `MeasureTexture` configuration, matching
    `cp_anywhere::texture::Config`."""

    gray_levels: int
    scales: list[int]
    images_or_objects: str


def _pixels(shape: tuple[int, int], seed: int) -> np.ndarray:
    """A deterministic, non-trivial float64 image in [0, 1]."""
    rng = np.random.default_rng(seed)
    r, c = np.indices(shape)
    base = (np.sin(r / 3.0) + np.cos(c / 4.0) + 2.0) / 4.0
    return np.clip(base + rng.normal(0, 0.05, shape), 0.0, 1.0)


def _disk_labels(
    shape: tuple[int, int], specs: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """``specs`` entries are ``(label, row, col, radius)``."""
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    for label, row, col, radius in specs:
        labels[(r - row) ** 2 + (c - col) ** 2 <= radius**2] = label
    return labels


_Case = tuple[str, np.ndarray, np.ndarray, _Settings]


def cases() -> Iterator[_Case]:
    """Yield (name, pixels, labels, settings) fixture cases for the live oracle."""
    # The scale multiplication reaches half-integers in float32 but not f64.
    ties = np.array(
        [[6.5, 17.5, 28.5], [39.5, 50.5, 61.5], [72.5, 83.5, 94.5]], np.float32
    ) / np.float32(255)
    yield (
        "float32_quantization_ties",
        ties,
        np.ones(ties.shape, dtype=np.uint32),
        {"gray_levels": 256, "scales": [1], "images_or_objects": mt.IO_BOTH},
    )
    shape = (40, 40)
    pixels = _pixels(shape, seed=1)
    labels = _disk_labels(shape, [(1, 10, 10, 8), (2, 28, 28, 6)])
    yield (
        "defaults",
        pixels,
        labels,
        {"gray_levels": 256, "scales": [3], "images_or_objects": mt.IO_BOTH},
    )

    yield (
        "low_gray_levels",
        pixels,
        labels,
        {"gray_levels": 8, "scales": [3], "images_or_objects": mt.IO_BOTH},
    )

    yield (
        "multi_scale",
        pixels,
        labels,
        {"gray_levels": 256, "scales": [2, 7], "images_or_objects": mt.IO_BOTH},
    )

    yield (
        "image_only",
        pixels,
        labels,
        {"gray_levels": 256, "scales": [4], "images_or_objects": mt.IO_IMAGES},
    )

    yield (
        "objects_only",
        pixels,
        labels,
        {"gray_levels": 256, "scales": [4], "images_or_objects": mt.IO_OBJECTS},
    )

    # A tiny 2x2-pixel object: too small for distance=5, exercises mahotas's
    # own ValueError path (real CP converts that to a stored 0.0, not NaN).
    tiny_shape = (20, 20)
    tiny_pixels = _pixels(tiny_shape, seed=2)
    tiny_labels = np.zeros(tiny_shape, np.uint32)
    tiny_labels[4:6, 4:6] = 1
    tiny_r, tiny_c = np.indices(tiny_shape)
    tiny_labels[(tiny_r - 14) ** 2 + (tiny_c - 14) ** 2 <= 6**2] = 2
    yield (
        "tiny_object_value_error",
        tiny_pixels,
        tiny_labels,
        {"gray_levels": 256, "scales": [5], "images_or_objects": mt.IO_OBJECTS},
    )

    # No objects at all -- real CP's own early-return-all-zero path.
    empty_shape = (15, 15)
    yield (
        "no_objects",
        _pixels(empty_shape, seed=3),
        np.zeros(empty_shape, np.uint32),
        {"gray_levels": 256, "scales": [3], "images_or_objects": mt.IO_OBJECTS},
    )

    # A uniform (zero-variance) image -- exercises Correlation's own
    # zero-denominator path live, rather than assuming its behavior.
    uniform_shape = (20, 20)
    uniform_pixels = np.full(uniform_shape, 0.5)
    uniform_labels = _disk_labels(uniform_shape, [(1, 10, 10, 7)])
    yield (
        "uniform_image",
        uniform_pixels,
        uniform_labels,
        {"gray_levels": 16, "scales": [3], "images_or_objects": mt.IO_BOTH},
    )


def _run_case(
    name: str,
    pixels: np.ndarray,
    labels: np.ndarray,
    settings: _Settings,
    provenance: dict[str, str],
) -> tuple[dict[str, object], int]:
    """Run one real MeasureTexture case; dump its ``.cppipe`` and return its
    fixture ``dict`` (not yet written to disk) plus its scalar count."""
    module = mt.MeasureTexture()
    module.images_list.value = "Input"
    module.objects_list.value = "Cells"
    module.gray_levels.value = settings["gray_levels"]
    module.images_or_objects.value = settings["images_or_objects"]
    module.scale_groups.clear()
    for index, scale in enumerate(settings["scales"]):
        module.add_scale(removable=index > 0)
        module.scale_groups[index].scale.value = scale
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)

    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    image_set.add("Input", Image(pixels, convert=False))
    object_set = ObjectSet()
    objects = Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "Cells")
    measurements = Measurements()
    module.run(
        Workspace(pipeline, module, image_set, object_set, measurements, image_sets)
    )

    with (OUT / f"{name}.cppipe").open("w") as stream:
        cppipe = Pipeline()
        cppipe.add_module(module)
        cppipe.dump(stream, save_image_plane_details=False)

    image_columns: dict[str, float] = {}
    if module.wants_image_measurements():
        for feature_name in measurements.get_feature_names("Image"):
            if feature_name.startswith(f"{mt.TEXTURE}_"):
                value = measurements.get_current_measurement("Image", feature_name)
                image_columns[feature_name] = float(value)
    object_columns: dict[str, list[float]] = {}
    if module.wants_object_measurements():
        for feature_name in measurements.get_feature_names("Cells"):
            if feature_name.startswith(f"{mt.TEXTURE}_"):
                raw_values = measurements.get_current_measurement("Cells", feature_name)
                object_columns[feature_name] = [
                    float(v) for v in np.asarray(raw_values, dtype=float)
                ]
    measurements.close()

    scalar_count = len(image_columns) + sum(len(v) for v in object_columns.values())
    print(name, scalar_count)
    return {
        **provenance,
        **({"pixel_dtype": "float32"} if pixels.dtype == np.float32 else {}),
        "settings": settings,
        "pixels": pixels.tolist(),
        "labels": labels.tolist(),
        "expected": {"Image": image_columns, "Cells": object_columns},
    }, scalar_count


def install() -> None:
    """Build a real pipeline+module, run it, and dump the fixture."""
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(mt.__file__)
    provenance = {
        "reference": f"CellProfiler 4.2.8 @ {SOURCE_COMMIT}",
        "mahotas": mahotas.__version__,
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "2D only; real mahotas.features.haralick, real Module.run",
    }
    manifest = []
    total_scalars = 0
    for name, pixels, labels, settings in cases():
        fixture, scalar_count = _run_case(name, pixels, labels, settings, provenance)
        total_scalars += scalar_count
        manifest.append({"name": name, "scalars": scalar_count})
        (OUT / f"{name}.json").write_text(
            json.dumps(fixture, indent=2, allow_nan=True) + "\n"
        )

    manifest_payload = {
        **provenance,
        "cases": manifest,
        "total_scalars": total_scalars,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n")
    print("TOTAL scalar comparisons:", total_scalars)


if __name__ == "__main__":
    install()
