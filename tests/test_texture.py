"""Oracle tests for :func:`cp_anywhere.measure_texture` against real
CellProfiler 4.2.8 fixtures (see ``fixtures/texture/`` and
``scripts/reference/generate_texture.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import measure_texture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/texture"
_CASES = sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "manifest")
_IMAGES_OR_OBJECTS = {
    "Images": (True, False),
    "Objects": (False, True),
    "Both": (True, True),
}


def _assert_matches(got: float, want: float) -> None:
    if math.isnan(want):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_texture_matches_a_fresh_cellprofiler_oracle_run(name: str) -> None:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    pixels = np.array(fixture["pixels"], dtype=fixture.get("pixel_dtype", "float64"))
    labels = np.array(fixture["labels"], dtype=np.uint32)
    settings = fixture["settings"]
    wants_image, wants_object = _IMAGES_OR_OBJECTS[settings["images_or_objects"]]
    result = measure_texture(
        pixels,
        labels,
        image_name="Input",
        gray_levels=settings["gray_levels"],
        scales=settings["scales"],
        wants_image_measurements=wants_image,
        wants_object_measurements=wants_object,
    )
    expected = fixture["expected"]
    assert set(result.image) == set(expected["Image"])
    for column, expected_value in expected["Image"].items():
        _assert_matches(result.image[column], expected_value)
    assert set(result.objects) == set(expected["Cells"])
    for column, expected_values in expected["Cells"].items():
        actual_values = result.objects[column]
        assert len(actual_values) == len(expected_values)
        for got, want in zip(actual_values, expected_values, strict=True):
            _assert_matches(got, want)


def test_texture_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        measure_texture(
            np.zeros((5, 5)), np.zeros((5, 6), dtype=np.uint32), image_name="Input"
        )


def test_texture_rejects_empty_scales() -> None:
    with pytest.raises(ValueError, match="scales"):
        measure_texture(np.zeros((5, 5)), np.zeros((5, 5), dtype=np.uint32), scales=[])


def test_texture_default_settings_match_cellprofiler_defaults() -> None:
    rng = np.random.RandomState(4)
    image = rng.uniform(size=(20, 20))
    r, c = np.indices(image.shape)
    labels = ((r - 10) ** 2 + (c - 10) ** 2 <= 7**2).astype(np.uint32)
    result = measure_texture(image, labels, image_name="DNA")
    expected_feature_count = 13 * 4  # 13 Haralick features x 4 directions
    assert len(result.image) == expected_feature_count
    assert len(result.objects) == expected_feature_count
    for values in result.objects.values():
        assert len(values) == 1
