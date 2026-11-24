"""Tests for `identify_secondary_objects` (IdentifySecondaryObjects):
Propagation (the module's own default), Distance-N, and Watershed -
Gradient. All verified against real, freshly-run CellProfiler 4.2.8 --
not guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.segment import identify_secondary_objects

# Real CellProfiler 4.2.8 IdentifySecondaryObjects, Propagation method,
# regularization_factor=0.05, global Otsu threshold, fill_holes=True,
# on a 10x12 radial-falloff image with two primary seeds at (5,3) and
# (5,8) -- see the Rust segmentation module's secondary.rs
# identical Rust-level test for the full oracle-generation context.
_ROWS, _COLS = 10, 12
_YY, _XX = np.mgrid[0:_ROWS, 0:_COLS]
_D1 = np.sqrt((_YY - 5) ** 2 + (_XX - 3) ** 2)
_D2 = np.sqrt((_YY - 5) ** 2 + (_XX - 8) ** 2)
_IMAGE = np.clip(1 - np.minimum(_D1, _D2) / 6, 0, 1)

_PRIMARY = np.zeros((_ROWS, _COLS), dtype=np.uint32)
_PRIMARY[5, 3] = 1
_PRIMARY[5, 8] = 2

_EXPECTED_PROPAGATION = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint32,
)


def test_propagation_matches_the_real_oracle_exactly() -> None:
    result = identify_secondary_objects(
        _IMAGE,
        _PRIMARY,
        method="propagation",
        threshold="otsu",
        rescale=False,
        regularization_factor=0.05,
        fill_holes=True,
    )
    np.testing.assert_array_equal(result.labels, _EXPECTED_PROPAGATION)
    assert result.threshold is not None


def test_propagation_defaults_labels_in_to_primary_labels() -> None:
    with_default = identify_secondary_objects(
        _IMAGE, _PRIMARY, threshold="otsu", rescale=False
    )
    with_explicit = identify_secondary_objects(
        _IMAGE,
        _PRIMARY,
        primary_labels_in=_PRIMARY,
        threshold="otsu",
        rescale=False,
    )
    np.testing.assert_array_equal(with_default.labels, with_explicit.labels)


def test_distance_n_expands_by_a_fixed_euclidean_radius() -> None:
    # Real scipy.ndimage.distance_transform_edt ground truth (same
    # scenario as the Rust-level nearest_labeled_source test): radius
    # 1.5 keeps only pixels within that distance of a seed.
    primary = np.zeros((1, 10), dtype=np.uint32)
    primary[0, 2] = 1
    primary[0, 7] = 2
    image = np.zeros((1, 10))
    result = identify_secondary_objects(
        image, primary, method="distance_n", distance_to_dilate=1.5
    )
    expected = np.array([[0, 1, 1, 1, 0, 0, 2, 2, 2, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result.labels, expected)
    assert result.threshold is None


def test_distance_n_ignores_threshold_settings_entirely() -> None:
    primary = np.zeros((1, 10), dtype=np.uint32)
    primary[0, 2] = 1
    primary[0, 7] = 2
    image = np.zeros((1, 10))
    a = identify_secondary_objects(
        image, primary, method="distance_n", distance_to_dilate=1.5, threshold="otsu"
    )
    b = identify_secondary_objects(
        image, primary, method="distance_n", distance_to_dilate=1.5, threshold=0.9
    )
    np.testing.assert_array_equal(a.labels, b.labels)


def test_rejects_an_unsupported_method() -> None:
    with pytest.raises(ValueError, match="distance_b"):
        identify_secondary_objects(_IMAGE, _PRIMARY, method="distance_b")


def test_watershed_gradient_matches_the_real_oracle_exactly() -> None:
    # Same inputs and CellProfiler-written expectations as the real
    # IdentifyPrimaryObjects -> IdentifySecondaryObjects fixture replayed
    # through load_pipeline; this exercises the Python wrapper's own
    # argument mapping (threshold="li", smoothing scale 0.0) on the way in.
    directory = Path(__file__).resolve().parents[1] / "fixtures/cppipe/secondary"
    stain = np.load(directory / "identify_secondary_objects_propagation.stain.npy")
    nuclei = np.load(
        directory / "identify_secondary_objects_propagation.nuclei_labels.npy"
    )
    expected = json.loads(
        (
            directory / "identify_secondary_objects_watershed_gradient.expected.json"
        ).read_text()
    )
    result = identify_secondary_objects(
        stain,
        nuclei,
        method="watershed_gradient",
        threshold="li",
        rescale=False,
        threshold_smoothing_scale=0.0,
        fill_holes=True,
    )
    np.testing.assert_array_equal(
        result.labels, np.asarray(expected["cells_labels"], dtype=result.labels.dtype)
    )
    assert result.threshold == pytest.approx(0.10862766577894337, rel=1e-12)


def test_filters_out_a_secondary_object_whose_primary_was_not_kept() -> None:
    # primary_labels (the "kept" set) omits id 2 entirely, even though
    # primary_labels_in (the seed array) still has it -- the grown
    # secondary object for id 2 must be discarded, matching real
    # CellProfiler's own filter_labels step.
    kept = _PRIMARY.copy()
    discarded_id = 2
    kept[kept == discarded_id] = 0
    result = identify_secondary_objects(
        _IMAGE,
        kept,
        primary_labels_in=_PRIMARY,
        method="propagation",
        threshold="otsu",
        rescale=False,
        regularization_factor=0.05,
    )
    assert discarded_id not in result.labels
    assert 1 in result.labels
