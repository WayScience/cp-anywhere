"""Original BBBC007 float32 corrected images retain CP reduction precision."""

import json
from pathlib import Path

import numpy as np
import pytest

import cp_anywhere as cp

PILOT = Path.home() / ".hermes/cache/scratch/cp-anywhere-production-pilot"
MANIFEST = PILOT / "run-after-illum-20261119/inputs/manifest.json"
ORACLE = PILOT / "run-strict/oracle"
RTOL = 1e-9
ATOL = 1e-10


def test_float32_reductions_match_pinned_cp_module() -> None:
    """Values from CP 4.2.8 MeasureImageIntensity with NumPy 1.23.5."""
    pixels = ((np.arange(17 * 23, dtype=np.int32) * 29) % 251).astype(
        np.float32
    ).reshape(17, 23) / np.float32(255)
    labels = {"Cells": np.zeros(pixels.shape, dtype=np.uint32)}
    oracle = {
        "Intensity_LowerQuartileIntensity_DNA": 0.239215686917305,
        "Intensity_MADIntensity_DNA": 0.24705882370471954,
        "Intensity_MaxIntensity_DNA": 0.9803921580314636,
        "Intensity_MeanIntensity_DNA": 0.48632465177179907,
        "Intensity_MedianIntensity_DNA": 0.48627451062202454,
        "Intensity_MinIntensity_DNA": 0.0,
        "Intensity_PercentMaximal_DNA": 0.2557544757033248,
        "Intensity_StdIntensity_DNA": 0.2849389612674713,
        "Intensity_TotalArea_DNA": 391.0,
        "Intensity_TotalIntensity_DNA": 190.15293884277344,
        "Intensity_UpperQuartileIntensity_DNA": 0.7313725650310516,
    }
    actual = cp.measure({"DNA": pixels}, labels, families=["image_intensity"]).features[
        "Image"
    ]
    assert oracle.keys() <= actual.keys()
    for feature, expected in oracle.items():
        assert np.isclose(actual[feature], expected, atol=ATOL, rtol=RTOL), feature

    # Promotion is explicit: a genuine float64 input keeps the old reductions.
    promoted = cp.measure(
        {"DNA": pixels.astype(np.float64)}, labels, families=["image_intensity"]
    ).features["Image"]
    assert np.isclose(
        promoted["Intensity_TotalIntensity_DNA"],
        190.15294509474188,
        atol=ATOL,
        rtol=RTOL,
    )


def test_original_bbbc007_image_intensity_matches_pinned_cp() -> None:
    if not MANIFEST.exists() or not ORACLE.exists():
        pytest.skip("local pinned BBBC007 oracle not installed")
    manifest = json.loads(MANIFEST.read_text())
    for group in manifest["groups"]:
        for field in group["validation"]:
            stem = ORACLE / group["id"] / field["id"]
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
                images = {
                    name: arrays[f"{name}_corrected"] for name in ("DNA", "Actin")
                }
            expected = json.loads(stem.with_suffix(".json").read_text())["features"][
                "Image"
            ]
            actual = cp.measure(
                images,
                {"Cells": np.zeros_like(images["DNA"], dtype=np.uint32)},
                families=["image_intensity"],
            ).features["Image"]
            for feature, value in expected.items():
                if feature.startswith("Intensity_"):
                    assert np.isclose(actual[feature], value, atol=ATOL, rtol=RTOL), (
                        stem,
                        feature,
                        actual[feature],
                        value,
                    )
