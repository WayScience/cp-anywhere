"""Regression coverage from real CellProfiler secondary-object runs."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import (
    SecondaryObjectsResult,
    identify_secondary_objects,
    load_pipeline,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe/secondary_regressions"
DIRECT = sorted(FIXTURES.glob("direct_*.json"))
PIPELINES = sorted(FIXTURES.glob("pipeline_*.cppipe"))


@pytest.mark.parametrize("path", DIRECT, ids=lambda path: path.stem)
def test_secondary_labels_and_layers_match_reference(path: Path) -> None:
    case = json.loads(path.read_text())
    primary = np.asarray(case["primary"], dtype=np.uint32)
    result = identify_secondary_objects(
        np.ones(primary.shape),
        primary,
        primary_labels_in=np.asarray(case["unedited"], dtype=np.uint32),
        method=case["method"],
        distance_to_dilate=case["radius"],
        fill_holes=case["fill_holes"],
        threshold=0.5,
        rescale=False,
    )
    np.testing.assert_array_equal(result.labels, case["expected"]["labels"])
    np.testing.assert_array_equal(
        result.unedited_labels, case["expected"]["unedited_labels"]
    )
    np.testing.assert_array_equal(
        result.small_removed_labels, case["expected"]["small_removed_labels"]
    )


@pytest.mark.parametrize("path", PIPELINES, ids=lambda path: path.stem)
def test_pipeline_secondary_and_neighbors_match_reference(path: Path) -> None:
    expected = json.loads(path.with_suffix(".json").read_text())
    primary, derived, measured = load_pipeline(path).run(
        {name: np.asarray(values) for name, values in expected["images"].items()}
    )
    nuclei = primary["Nuclei"]
    for attr in ("labels", "unedited_labels", "small_removed_labels"):
        np.testing.assert_array_equal(getattr(nuclei, attr), expected["Nuclei"][attr])
    cells = derived["Cells"]
    assert isinstance(cells, SecondaryObjectsResult)
    np.testing.assert_array_equal(cells.labels, expected["Cells"]["labels"])
    np.testing.assert_array_equal(
        cells.small_removed_labels, expected["Cells"]["small_removed_labels"]
    )
    assert measured is not None
    for feature, values in expected["neighbors"].items():
        np.testing.assert_allclose(
            measured.features["Cells"][feature],
            values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=feature,
        )
