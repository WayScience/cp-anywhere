"""Oracle tests for :func:`cp_anywhere.measure_intensity_distribution`
against real CellProfiler 4.2.8 fixtures (see
``fixtures/intensity_distribution/`` and
``scripts/reference/generate_intensity_distribution.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import measure_intensity_distribution
from cp_anywhere.intensity_distribution import _ZernikeOption

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/intensity_distribution"
_CASES = sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "manifest")
_ZERNIKES: dict[str, _ZernikeOption] = {
    "None": "none",
    "Magnitudes only": "magnitudes",
    "Magnitudes and phase": "magnitudes_and_phase",
}


def _assert_matches(actual: np.ndarray, expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_intensity_distribution_matches_a_fresh_cellprofiler_oracle_run(
    name: str,
) -> None:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    pixels = np.array(fixture["pixels"], dtype=np.float64)
    labels = np.array(fixture["labels"], dtype=np.uint32)
    settings = fixture["settings"]
    result = measure_intensity_distribution(
        pixels,
        labels,
        image_name="Input",
        bin_count=settings["bin_count"],
        wants_scaled=settings["wants_scaled"],
        maximum_radius=settings["maximum_radius"],
        zernikes=_ZERNIKES[settings["wants_zernikes"]],
        zernike_degree=settings["zernike_degree"],
    )
    assert set(result.columns) == set(fixture["expected"])
    for column, expected_values in fixture["expected"].items():
        _assert_matches(result.columns[column], expected_values)


def test_intensity_distribution_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        measure_intensity_distribution(
            np.zeros((5, 5)), np.zeros((5, 6), dtype=np.uint32), image_name="DNA"
        )


def test_float32_radial_matches_pinned_oracle() -> None:
    r, c = np.indices((11, 11))
    radius_squared = 16
    labels = (((r - 5) ** 2 + (c - 5) ** 2) <= radius_squared).astype(np.uint32)
    image = (((17 * r + 11 * c + 3 * r * c) % 37 + 1) / 41).astype(np.float32)
    result = measure_intensity_distribution(image, labels, image_name="Input")
    assert result.columns["RadialDistribution_FracAtD_Input_1of4"][0] == pytest.approx(
        0.13833528757095337, rel=1e-9, abs=1e-10
    )
    assert result.columns["RadialDistribution_MeanFrac_Input_4of4"][0] == pytest.approx(
        0.9305979639291758, rel=1e-9, abs=1e-10
    )


def test_intensity_distribution_default_settings_match_cellprofiler_defaults() -> None:
    rng = np.random.RandomState(3)
    image = rng.uniform(size=(21, 23))
    r, c = np.indices(image.shape)
    labels = ((r - 10) ** 2 + (c - 11) ** 2 <= 8**2).astype(np.uint32)
    result = measure_intensity_distribution(image, labels, image_name="DNA")
    expected_columns = 3 * 4  # FracAtD/MeanFrac/RadialCV x 4 default bins
    assert len(result.columns) == expected_columns
    for values in result.columns.values():
        assert len(values) == 1
