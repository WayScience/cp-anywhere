"""Real CellProfiler 4.2.8 fixed-distance neighbor regressions."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.api import MeasurementResult, measure
from cp_anywhere.cppipe import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/neighbors_distance"
CASES = json.loads((FIXTURES / "cases.json").read_text())


def _compare(result: MeasurementResult, expected: dict) -> None:
    assert {k: v.tolist() for k, v in result.object_ids.items()} == expected[
        "object_ids"
    ]
    assert set(result.features) == set(expected["features"])
    for scope, columns in expected["features"].items():
        assert set(result.features[scope]) == set(columns)
        for name, values in columns.items():
            actual = np.asarray(result.features[scope][name])
            if "Number" in name:
                assert actual.dtype.kind == "i"
                np.testing.assert_array_equal(actual, values, err_msg=name)
            else:
                decoded = [
                    float(v["nonfinite"]) if isinstance(v, dict) else v for v in values
                ]
                np.testing.assert_allclose(
                    actual, decoded, rtol=1e-12, atol=1e-12, err_msg=name
                )
    assert [
        [row["object_name"], row["first_object_id"], row["second_object_id"]]
        for row in result.relationships
    ] == expected["relationships"]
    for row in result.relationships:
        assert row["first_image_id"] == row["second_image_id"] == result.image_id


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_fixed_distance_matches_cellprofiler(case: dict) -> None:
    labels = np.asarray(case["labels"], dtype=np.uint32)
    diagnostics = case["full_labels"]
    full_objects = None
    if diagnostics is not None:
        full = np.asarray(diagnostics, dtype=np.uint32)
        # Direct measure() takes the chosen diagnostic geometry. With excluded
        # objects disabled, CP masks that layer without changing its IDs.
        full_objects = {
            "Cells": full if case["excluded"] else np.where(labels, full, 0)
        }
    actual = measure(
        {"DNA": np.zeros(labels.shape)},
        {"Cells": labels},
        families=["neighbors"],
        neighbors_distances={"Cells": case["distance"]},
        neighbors_full_objects=full_objects,
    )
    _compare(actual, case)


def test_fixed_distance_cppipe_measures_each_module_in_order() -> None:
    expected = json.loads((FIXTURES / "primary_mixed.json").read_text())
    pipeline = load_pipeline(FIXTURES / "primary_mixed.cppipe")
    assert [step.neighbors_distance for step in pipeline.measurement_steps] == [
        5,
        1,
        1,
        None,
        6,
    ]
    primary, _, result = pipeline.run(
        {"DNA": np.asarray(expected["pixels"], dtype=np.uint16)}
    )
    for name, layers in expected["outputs"].items():
        np.testing.assert_array_equal(primary[name].labels, layers["labels"])
        np.testing.assert_array_equal(
            primary[name].small_removed_labels, layers["full_labels"]
        )
    assert result is not None
    _compare(result, expected)


@pytest.mark.parametrize("distance", ["0", "-1", "1.5", "4294967296"])
def test_fixed_distance_cppipe_rejects_invalid_distance(
    tmp_path: Path, distance: str
) -> None:
    text = (FIXTURES / "primary_mixed.cppipe").read_text()
    path = tmp_path / "invalid.cppipe"
    path.write_text(
        text.replace("Neighbor distance:5", f"Neighbor distance:{distance}", 1)
    )
    with pytest.raises(CppipeError, match="Neighbor distance"):
        load_pipeline(path)


def test_fixed_distance_api_rejects_conflicting_scope_modes() -> None:
    labels = np.asarray([[1, 0, 2]])
    with pytest.raises(ValueError, match="conflicting neighbor distance methods"):
        measure(
            {"DNA": labels},
            {"Cells": labels},
            neighbors_expand=["Cells"],
            neighbors_distances={"Cells": 1},
        )


def test_fixed_distance_rejects_3d() -> None:
    labels = np.asarray([[[1, 0, 2]]])
    with pytest.raises(ValueError, match="supports 2D"):
        measure({"DNA": labels}, {"Cells": labels}, neighbors_distances={"Cells": 1})
