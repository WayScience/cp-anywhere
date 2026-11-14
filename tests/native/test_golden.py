"""The native engine must match the saved original CellProfiler result."""

import json
from pathlib import Path

import numpy as np

import cp_anywhere

ROOT = Path(__file__).resolve().parents[2]
TOTAL_FEATURE_COLUMNS = 119  # spec/compatibility.md's declared scope


def test_touching_gradient_native_scope() -> None:
    folder = ROOT / "fixtures/synthetic/touching_gradient"
    expected = json.loads(
        (ROOT / "fixtures/reference/touching_gradient.json").read_text()
    )
    result = cp_anywhere.measure(
        {name: np.load(folder / f"{name}.npy") for name in ("DNA", "Actin")},
        {
            "Cells": np.load(folder / "labels.npy"),
            "Nuclei": np.load(folder / "nuclei_labels.npy"),
        },
        image_id="touching_gradient",
    )
    assert sum(map(len, result.features.values())) == TOTAL_FEATURE_COLUMNS
    assert result.object_ids["Cells"].tolist() == expected["object_ids"]["Cells"]
    assert np.asarray(
        result.features["Cells"]["Neighbors_NumberOfNeighbors_Adjacent"]
    ).tolist() == [1, 2, 1]
    assert (
        result.features["Image"]["Intensity_TotalIntensity_Actin"]
        == expected["features"]["Image"]["Intensity_TotalIntensity_Actin"]
    )
    assert len(result.relationships) == len(expected["relationships"])
