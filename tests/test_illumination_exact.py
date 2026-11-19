"""Exact stage parity for the original BBBC007 illumination calibration."""

import json
from pathlib import Path

import numpy as np

from cp_anywhere import calculate_illumination_across

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/illumination_calculate_across/float32/bbbc007_gaussian_float32.json"
)


def test_bbbc007_gaussian_stages_match_pinned_cellprofiler_exactly() -> None:
    fixture = json.loads(FIXTURE.read_text())
    images = [np.asarray(x, dtype=np.float32) for x in fixture["images"]]
    settings = fixture["settings"]
    for option, key in (("No", "smooth"), ("Median", "final")):
        actual = calculate_illumination_across(
            images,
            intensity_choice=str(settings["intensity_choice"]),
            smoothing_method=str(settings["smoothing_method"]),
            automatic_object_width=str(settings["automatic_object_width"]),
            size_of_smoothing_filter=float(settings["size_of_smoothing_filter"]),
            rescale_option=option,
        ).astype(np.float32)
        expected = np.asarray(fixture[key], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected, err_msg=f"{key} stage")
