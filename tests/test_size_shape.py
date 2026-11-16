"""Exercise integrated size/shape columns against real CellProfiler fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import load_pipeline, measure

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/size_shape"
CASES = sorted(FIXTURES.glob("*.json"))


def _optional(name: str) -> str:
    if name.startswith("AreaShape_Zernike_"):
        return "zernike"
    if any(part in name for part in ("Moment", "InertiaTensor")):
        return "advanced"
    return "standard"


@pytest.mark.parametrize("path", CASES, ids=lambda path: path.stem)
@pytest.mark.parametrize(
    "zernikes,advanced", [(False, False), (True, False), (False, True), (True, True)]
)
def test_every_selected_column_matches_real_reference(
    path: Path, zernikes: bool, advanced: bool, tmp_path: Path
) -> None:
    fixture = json.loads(path.read_text())
    labels = np.asarray(fixture["labels"], dtype=np.uint32)
    result = measure(
        {"DNA": np.zeros(labels.shape)},
        {"Cells": labels},
        families=["object_size_shape"],
        size_shape_calculate_zernikes=zernikes,
        size_shape_calculate_advanced=advanced,
    )
    enabled = {"standard"}
    if zernikes:
        enabled.add("zernike")
    if advanced:
        enabled.add("advanced")
    expected = {
        name: values
        for name, values in fixture["features"].items()
        if _optional(name) in enabled
    }
    assert set(result.features["Cells"]) == set(expected)
    for name, values in expected.items():
        np.testing.assert_allclose(
            result.features["Cells"][name],
            values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=name,
        )
    # Every enabled column remains safe to serialize/export, including empty sets.
    result.rows()
    result.to_json(tmp_path / "measurements.json")
    assert (
        "Cells" in json.loads((tmp_path / "measurements.json").read_text())["features"]
    )


@pytest.mark.parametrize("path", CASES, ids=lambda path: path.stem)
def test_real_size_shape_pipeline_settings_are_runnable(path: Path) -> None:
    fixture = json.loads(path.read_text())
    labels = np.asarray(fixture["labels"], dtype=np.uint32)
    pipeline = load_pipeline(path.with_suffix(".cppipe"))
    assert "MeasureObjectSizeShape" not in pipeline.skipped_modules
    _, _, result = pipeline.run({"DNA": np.zeros(labels.shape), "Cells": labels})
    assert result is not None
    assert set(result.features["Cells"]) == set(fixture["features"])
    for name, values in fixture["features"].items():
        np.testing.assert_allclose(
            result.features["Cells"][name],
            values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=name,
        )


def test_sparse_shape_labels_raise_instead_of_misaligning_export_rows() -> None:
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[1:3, 1:3] = 2
    with pytest.raises(ValueError, match="consecutive"):
        measure(
            {"DNA": np.zeros(labels.shape)},
            {"Cells": labels},
            families=["object_size_shape"],
        )


def test_3d_shape_request_reports_the_explicit_gap() -> None:
    labels = np.ones((3, 4, 5), dtype=np.uint32)
    with pytest.raises(ValueError, match="2D"):
        measure(
            {"DNA": np.zeros(labels.shape)},
            {"Cells": labels},
            families=["object_size_shape"],
        )


def test_legacy_default_families_do_not_gain_shape_columns() -> None:
    labels = np.ones((4, 4), dtype=np.uint32)
    result = measure({"DNA": np.ones(labels.shape)}, {"Cells": labels})
    assert not any(name.startswith("AreaShape_") for name in result.features["Cells"])
