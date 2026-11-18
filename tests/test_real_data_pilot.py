"""Checks for the real-data pilot's independent splits and evaluation gates."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import cp_anywhere as cp
from scripts.validation.pilot_checks import (
    compare_features,
    leave_one_out,
    manual_labels,
    normalized_grayscale,
    within_two_pixels,
)
from scripts.validation.real_data_pilot import verify_exports


def test_each_held_out_field_is_absent_from_its_calibration_set() -> None:
    fields = [{"id": "first"}, {"id": "second"}, {"id": "third"}]
    folds = leave_one_out("cohort", fields)
    assert len(folds) == len(fields)
    assert {fold["validation"][0]["id"] for fold in folds} == {
        field["id"] for field in fields
    }
    for fold in folds:
        calibration = {field["id"] for field in fold["calibration"]}
        validation = {field["id"] for field in fold["validation"]}
        assert calibration.isdisjoint(validation)
        assert calibration | validation == {field["id"] for field in fields}


def test_a_single_field_cannot_calibrate_itself() -> None:
    with pytest.raises(ValueError, match="at least two"):
        leave_one_out("cohort", [{"id": "only"}])


def test_duplicate_field_ids_cannot_hide_calibration_leakage() -> None:
    with pytest.raises(ValueError, match="unique"):
        leave_one_out("cohort", [{"id": "same"}, {"id": "same"}])


def test_rgb_files_are_not_silently_converted_to_fluorescence() -> None:
    with pytest.raises(ValueError, match="2D grayscale"):
        normalized_grayscale(np.zeros((8, 8, 3), dtype=np.uint8))
    raw = np.array([[0, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(
        normalized_grayscale(raw), (raw.astype(np.float64) / 255).astype(np.float32)
    )


def test_outline_labels_exclude_the_border_connected_background() -> None:
    outline = np.zeros((9, 9), dtype=bool)
    outline[2:7, 2] = outline[2:7, 6] = True
    outline[2, 2:7] = outline[6, 2:7] = True
    expected = np.zeros_like(outline, dtype=np.int32)
    expected[3:6, 3:6] = 1
    np.testing.assert_array_equal(manual_labels(outline), expected)
    np.testing.assert_array_equal(manual_labels(np.zeros_like(outline)), 0)


def test_two_pixel_boundary_metric_uses_euclidean_distance() -> None:
    outline = np.zeros((9, 9), dtype=bool)
    outline[4, 4] = True
    nearby = within_two_pixels(outline)
    assert nearby[4, 6]
    assert nearby[5, 5]
    assert not nearby[6, 6]
    assert not nearby[4, 7]


def test_feature_comparison_keeps_nonfinite_kinds_and_feature_names() -> None:
    expected = {"Cells": {"Intensity_MeanIntensity_DNA": [0.2, np.nan, np.inf]}}
    result = compare_features(expected, expected)
    assert result["mismatches"] == []
    changed = {"Cells": {"Intensity_MeanIntensity_DNA": [0.2, 0.0, np.inf]}}
    assert compare_features(changed, expected)["mismatches"]
    assert compare_features({"Cells": {}}, expected)["mismatches"]


def test_feature_comparison_does_not_hide_a_numeric_discrepancy() -> None:
    expected = {"Cells": {"AreaShape_Area": [123]}}
    actual = {"Cells": {"AreaShape_Area": [124]}}
    assert compare_features(actual, expected)["mismatches"]


def test_upstream_rounding_does_not_relax_measurement_checks() -> None:
    expected = {"Cells": {"Intensity_MeanIntensity_DNA": [0.4]}}
    # An illumination-sized difference still fails strict measurement parity.
    within_contract = {"Cells": {"Intensity_MeanIntensity_DNA": [0.4 + 4e-7]}}
    assert compare_features(within_contract, expected)["mismatches"]

    # Beyond the illumination bound: still a mismatch, corrected or not.
    beyond = {"Cells": {"Intensity_MeanIntensity_DNA": [0.4 + 4e-6]}}
    assert compare_features(beyond, expected)["mismatches"]

    # Label-derived features are never widened by the illumination tier.
    shape_within = {"Cells": {"AreaShape_Eccentricity": [0.5 + 4e-7]}}
    shape_expected = {"Cells": {"AreaShape_Eccentricity": [0.5]}}
    assert compare_features(shape_within, shape_expected)["mismatches"]


def test_sqlite_export_checks_values_not_only_row_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((3, 3), 0.5)
    labels = np.ones((3, 3), dtype=np.uint32)
    result = cp.measure({"DNA": image}, {"Cells": labels})
    original = cp.MeasurementResult.to_sqlite

    def corrupt_value(self: cp.MeasurementResult, path: Path, **kwargs: str) -> None:
        original(self, path, **kwargs)
        with sqlite3.connect(path) as connection:
            connection.execute(
                'UPDATE Per_Cells SET "Intensity_MeanIntensity_DNA" = 42'
            )

    monkeypatch.setattr(cp.MeasurementResult, "to_sqlite", corrupt_value)
    with pytest.raises(AssertionError):
        verify_exports(result, tmp_path)
