# ty: ignore[import-not-found]
"""Capture OverlayOutlines pixels with CellProfiler 4.2.8 and scikit-image 0.21.0."""

# ruff: noqa: E402
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.overlayoutlines import OverlayOutlines
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/cppipe/overlay_outlines"


def _overlay(
    source: str | None,
    target: str,
    mode: str,
    line: str,
    groups: tuple[tuple[str, str], ...],
) -> OverlayOutlines:
    module = OverlayOutlines()
    module.blank_image.value = source is None
    module.image_name.value = source or "None"
    module.output_image_name.value = target
    module.wants_color.value = mode
    module.line_mode.value = line
    module.max_type.value = "Max of image"
    for index, (name, color) in enumerate(groups):
        if index:
            module.add_outline()
        module.outlines[index].objects_name.value = name
        module.outlines[index].color.value = color
    return module


def _modules() -> tuple[OverlayOutlines, ...]:
    primary = (("Cells", "#008000"),)
    other = (("Edges", "Blue"),)
    white = _overlay("DNA", "GrayWhite", "Grayscale", "Inner", primary)
    white.max_type.value = "Max possible"
    return (
        _overlay(None, "SurveyColor", "Color", "Inner", primary),
        _overlay(None, "JumpGray", "Grayscale", "Inner", (("Cells", "Red"),)),
        _overlay("DNA", "Rgb", "Color", "Thick", primary + other),
        _overlay("DNA", "Reversed", "Color", "Thick", other + primary),
        _overlay("Rgb", "Gray", "Grayscale", "Outer", other),
        _overlay("Gray", "GrayToRgb", "Color", "Inner", (("Edges", "Green"),)),
        _overlay("DNA", "GrayMax", "Grayscale", "Inner", primary),
        white,
        _overlay("RgbInput", "Outer", "Color", "Outer", primary),
        _overlay("RgbInput", "RgbMax", "Grayscale", "Inner", primary),
        _overlay("Binary", "BinaryGray", "Grayscale", "Thick", (("Full", "Red"),)),
        _overlay(None, "FullBackground", "Color", "Inner", (("Full", "White"),)),
        _overlay(None, "EmptyBackground", "Color", "Outer", (("Empty", "White"),)),
    )


def _images() -> dict[str, np.ndarray]:
    dna = (np.arange(35, dtype=np.float32).reshape(5, 7) - 3) / 47
    rgb = np.stack([dna, dna[::-1] / 2, dna[:, ::-1] / 3], axis=-1)
    return {
        "DNA": dna,
        "RgbInput": rgb,
        "Binary": np.array([[[1, 0, 1], [1, 1, 0]]], dtype=bool),
    }


def _labels() -> dict[str, np.ndarray]:
    cells = np.array(
        [
            [2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 30, 30, 30, 0],
            [0, 0, 0, 30, 30, 30, 0],
            [0, 0, 0, 30, 30, 30, 0],
        ],
        dtype=np.int32,
    )
    # Crop the sixth column onward and pad below the second row.
    edges = np.zeros((2, 9), dtype=np.int32)
    edges[:, 2:6] = 9
    edges[:, 7:] = 15
    return {
        "Cells": cells,
        "Edges": edges,
        "Full": np.full((1, 2), 7, dtype=np.int32),
        "Empty": np.zeros((3, 2), dtype=np.int32),
    }


def _plane(pixels: np.ndarray) -> dict:
    return {
        "shape": pixels.shape,
        "dtype": pixels.dtype.name,
        "values": pixels.ravel().tolist(),
    }


def _run(pipeline: Pipeline) -> dict:
    image_sets = ImageSetList()
    images = image_sets.get_image_set(0)
    for name, pixels in _images().items():
        images.add(name, Image(pixels, dimensions=2))
    objects = ObjectSet()
    for name, labels in _labels().items():
        instance = Objects()
        instance.segmented = labels
        objects.add_objects(instance, name)
    measurements = Measurements()
    result = {}
    try:
        for module in pipeline.modules():
            module.run(
                Workspace(pipeline, module, images, objects, measurements, image_sets)
            )
            name = module.output_image_name.value
            result[name] = _plane(images.get_image(name).pixel_data.copy())
    finally:
        measurements.close()
    return result


def generate() -> None:
    """Save original pipeline settings and reference input and output arrays."""
    reference = {
        package: version(package)
        for package in ("CellProfiler-core", "scikit-image", "numpy")
    }
    assert reference == {
        "CellProfiler-core": "4.2.8",
        "scikit-image": "0.21.0",
        "numpy": "1.23.5",
    }
    pipeline = Pipeline()
    for index, module in enumerate(_modules(), 1):
        module.set_module_num(index)
        pipeline.add_module(module)
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = OUT / "overlay.cppipe"
    with pipe.open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)
    reloaded = Pipeline()
    reloaded.load(str(pipe))
    assert len(reloaded.modules()) == len(pipeline.modules())
    fixture = {
        "reference": reference,
        "images": {name: _plane(pixels) for name, pixels in _images().items()},
        "labels": {name: _plane(labels) for name, labels in _labels().items()},
        "outputs": _run(reloaded),
    }
    (OUT / "overlay.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    generate()
