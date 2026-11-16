"""Oracle tests for :func:`cp_anywhere.measure_granularity` against real
CellProfiler 4.2.8 fixtures (see ``fixtures/granularity/`` and
``scripts/reference/generate_granularity.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import measure_granularity

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/granularity"
_CASES = sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "manifest")


def _assert_matches(actual: np.ndarray, expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_granularity_matches_a_fresh_cellprofiler_oracle_run(name: str) -> None:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    pixels = np.array(fixture["pixels"], dtype=np.float64)
    mask = np.array(fixture["mask"], dtype=bool)
    objects = {
        key: np.array(value, dtype=np.uint32)
        for key, value in fixture["objects"].items()
    }
    settings = fixture["settings"]
    result = measure_granularity(
        pixels,
        objects=objects,
        mask=mask,
        subsample_size=settings["subsample_size"],
        image_sample_size=settings["image_sample_size"],
        element_size=settings["element_size"],
        granular_spectrum_length=settings["granular_spectrum_length"],
    )
    _assert_matches(result.image, fixture["expected"]["image"])
    for key, scales in fixture["expected"]["objects"].items():
        for scale_index, expected_scale in enumerate(scales):
            _assert_matches(result.objects[key][scale_index], expected_scale)


def test_granularity_rejects_a_one_row_image() -> None:
    with pytest.raises(ValueError, match="1-row or 1-column"):
        measure_granularity(np.zeros((1, 9)))


def test_granularity_default_settings_match_cellprofiler_defaults() -> None:
    # CellProfiler's own MeasureGranularity module defaults.
    rng = np.random.RandomState(7)
    image = rng.uniform(size=(40, 40))
    result = measure_granularity(image)
    default_granular_spectrum_length = 16
    assert len(result.image) == default_granular_spectrum_length
    assert result.objects == {}
