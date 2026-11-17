"""Oracle tests for :func:`cp_anywhere.calculate_illumination` against
real CellProfiler 4.2.8 fixtures (see
``fixtures/illumination_calculate/`` and
``scripts/reference/generate_illumination_calculate.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import (
    apply_illumination,
    calculate_illumination,
    calculate_illumination_across,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/illumination_calculate"
_CASES = sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "manifest")


@pytest.mark.parametrize("name", _CASES)
def test_calculate_illumination_matches_a_fresh_cellprofiler_oracle_run(
    name: str,
) -> None:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    pixels = np.array(fixture["pixels"], dtype=np.float64)
    mask = None if fixture["mask"] is None else np.array(fixture["mask"], dtype=bool)
    settings = fixture["settings"]
    actual = calculate_illumination(
        pixels,
        mask=mask,
        intensity_choice=settings["intensity_choice"],
        block_size=settings["block_size"],
        dilate_objects=settings["dilate_objects"],
        object_dilation_radius=settings["object_dilation_radius"],
        smoothing_method=settings["smoothing_method"],
        automatic_object_width=settings["automatic_object_width"],
        object_width=float(settings["object_width"]),
        size_of_smoothing_filter=float(settings["size_of_smoothing_filter"]),
        rescale_option=settings["rescale_option"],
    )
    expected = np.array(fixture["expected"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_calculate_illumination_rejects_a_non_2d_image() -> None:
    with pytest.raises(ValueError, match="2D"):
        calculate_illumination(np.zeros((3, 3, 3)))


def test_calculate_illumination_rejects_a_mismatched_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask"):
        calculate_illumination(np.zeros((5, 5)), mask=np.zeros((3, 3), dtype=bool))


@pytest.mark.parametrize("method", ["Convex Hull", "Splines"])
def test_calculate_illumination_rejects_the_not_yet_implemented_methods(
    method: str,
) -> None:
    with pytest.raises(ValueError, match="not yet implemented"):
        calculate_illumination(np.ones((9, 9)) * 0.5, smoothing_method=method)


def test_calculate_illumination_default_settings_run_without_error() -> None:
    # CellProfiler's own CorrectIlluminationCalculate module defaults
    # (Regular, Gaussian Filter, Automatic, rescale Yes).
    rng = np.random.RandomState(11)
    image = rng.uniform(size=(30, 40))
    result = calculate_illumination(image)
    assert result.shape == image.shape


APPLY_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/illumination_apply"
_APPLY_CASES = sorted(
    p.stem for p in APPLY_FIXTURES.glob("*.json") if p.stem != "manifest"
)


@pytest.mark.parametrize("name", _APPLY_CASES)
def test_apply_illumination_matches_a_fresh_cellprofiler_oracle_run(name: str) -> None:
    fixture = json.loads((APPLY_FIXTURES / f"{name}.json").read_text())
    settings = fixture["settings"]
    actual = apply_illumination(
        np.array(fixture["image"], dtype=np.float64),
        np.array(fixture["illum"], dtype=np.float64),
        divide_or_subtract=settings["divide_or_subtract"],
        truncate_low=settings["truncate_low"],
        truncate_high=settings["truncate_high"],
    )
    expected = np.array(fixture["expected"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_apply_illumination_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        apply_illumination(np.zeros((5, 5)), np.zeros((3, 3)))


def test_apply_illumination_rejects_an_unknown_divide_or_subtract() -> None:
    with pytest.raises(ValueError, match="divide_or_subtract"):
        apply_illumination(np.ones((3, 3)), np.ones((3, 3)), divide_or_subtract="Nope")


ACROSS_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures/illumination_calculate_across"
)
_ACROSS_CASES = sorted(
    p.stem for p in ACROSS_FIXTURES.glob("*.json") if p.stem != "manifest"
)


@pytest.mark.parametrize("name", _ACROSS_CASES)
def test_calculate_illumination_across_matches_a_fresh_cellprofiler_oracle_run(
    name: str,
) -> None:
    fixture = json.loads((ACROSS_FIXTURES / f"{name}.json").read_text())
    settings = fixture["settings"]
    images = [np.array(img, dtype=np.float64) for img in fixture["images"]]
    masks = [None if m is None else np.array(m, dtype=bool) for m in fixture["masks"]]
    actual = calculate_illumination_across(
        images,
        masks=masks,
        intensity_choice=settings["intensity_choice"],
        block_size=settings["block_size"],
        dilate_objects=settings["dilate_objects"],
        object_dilation_radius=settings["object_dilation_radius"],
        smoothing_method=settings["smoothing_method"],
        automatic_object_width=settings["automatic_object_width"],
        object_width=float(settings["object_width"]),
        size_of_smoothing_filter=float(settings["size_of_smoothing_filter"]),
        rescale_option=settings["rescale_option"],
    )
    expected = np.array(fixture["expected"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_calculate_illumination_across_rejects_an_empty_image_list() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        calculate_illumination_across([])


def test_calculate_illumination_across_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_illumination_across([np.zeros((5, 5)), np.zeros((3, 3))])


def test_calculate_illumination_across_rejects_a_mask_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per image"):
        calculate_illumination_across(
            [np.zeros((5, 5)), np.zeros((5, 5))], masks=[None]
        )


def test_calculate_illumination_across_single_image_is_close_to_each_mode() -> None:
    # With exactly one unmasked image, "All" mode's accumulation is
    # numerically close to "Each" mode's own single-image pipeline,
    # but not bit-identical: real CellProfiler's own
    # `calculate_image()` always constructs a new `Image` (an extra
    # float32 round-trip) even for a single accumulated image, while
    # "Each" mode's `Regular`+no-mask branch returns the original
    # image unchanged (no extra round-trip) -- both exactly match
    # their own real-CellProfiler code path (see illumination.rs).
    rng = np.random.RandomState(5)
    image = rng.uniform(size=(20, 24))
    each = calculate_illumination(image, smoothing_method="Fit Polynomial")
    across = calculate_illumination_across([image], smoothing_method="Fit Polynomial")
    np.testing.assert_allclose(each, across, rtol=1e-6, atol=1e-6)
