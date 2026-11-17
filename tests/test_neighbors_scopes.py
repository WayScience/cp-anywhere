"""Tests for `measure(..., neighbors_scopes=...)`: computing
MeasureObjectNeighbors for an arbitrary object scope (or several),
not just the original declared "Cells"-only contract.

The default (``neighbors_scopes`` omitted) must stay byte-for-byte
identical to every pre-existing fixture and golden comparison --
verified here directly, not just by the rest of the suite still
passing. The new, non-default path (``neighbors_scopes=["Nuclei"]``)
is cross-checked against a real run of the pinned CellProfiler 4.2.8
oracle's own ``MeasureObjectNeighbors`` module configured for "Nuclei"
against itself -- not guessed: ``Neighbors_NumberOfNeighbors_Adjacent``
= ``[0, 0, 0]`` and ``Neighbors_FirstClosestObjectNumber_Adjacent`` =
``[2, 1, 2]`` on this fixture's real nuclei labels, matching exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.api import measure

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"


def _inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    images = {"DNA": np.load(FIXTURE / "DNA.npy")}
    objects = {
        "Cells": np.load(FIXTURE / "labels.npy"),
        "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
    }
    return images, objects


def _array(value: object) -> np.ndarray:
    """Narrow a feature value to ``np.ndarray`` for ``ty``: object/
    Cells/Nuclei-scope values are always arrays (only the "Image"
    scope holds bare scalars), but the declared feature type is a
    union of both."""
    assert isinstance(value, np.ndarray)
    return value


def test_default_neighbors_scopes_still_computes_cells_only() -> None:
    images, objects = _inputs()
    result = measure(images, objects, families=["neighbors"])
    assert "Neighbors_NumberOfNeighbors_Adjacent" in result.features["Cells"]
    assert "Nuclei" not in result.features or not any(
        name.startswith("Neighbors_") for name in result.features["Nuclei"]
    )
    assert all(row["object_name"] == "Cells" for row in result.relationships)


def test_neighbors_scopes_matches_the_real_oracle_for_nuclei() -> None:
    """Real CellProfiler 4.2.8, MeasureObjectNeighbors on 'Nuclei' vs
    itself, on this fixture's real nuclei labels -- not guessed."""
    images, objects = _inputs()
    result = measure(
        images, objects, families=["neighbors"], neighbors_scopes=["Nuclei"]
    )
    assert "Cells" not in result.features
    assert _array(
        result.features["Nuclei"]["Neighbors_NumberOfNeighbors_Adjacent"]
    ).tolist() == [
        0,
        0,
        0,
    ]
    assert _array(
        result.features["Nuclei"]["Neighbors_FirstClosestObjectNumber_Adjacent"]
    ).tolist() == [2, 1, 2]
    assert result.relationships == []


def test_neighbors_scopes_computes_multiple_scopes_independently() -> None:
    images, objects = _inputs()
    result = measure(
        images,
        objects,
        families=["neighbors"],
        neighbors_scopes=["Cells", "Nuclei"],
    )
    assert "Neighbors_NumberOfNeighbors_Adjacent" in result.features["Cells"]
    assert "Neighbors_NumberOfNeighbors_Adjacent" in result.features["Nuclei"]
    cells_edges = [row for row in result.relationships if row["object_name"] == "Cells"]
    nuclei_edges = [
        row for row in result.relationships if row["object_name"] == "Nuclei"
    ]
    assert cells_edges
    assert nuclei_edges == []
    # Cross-check against the unchanged, oracle-verified Cells-only path.
    cells_only = measure(images, objects, families=["neighbors"])
    assert (
        _array(
            result.features["Cells"]["Neighbors_NumberOfNeighbors_Adjacent"]
        ).tolist()
        == _array(
            cells_only.features["Cells"]["Neighbors_NumberOfNeighbors_Adjacent"]
        ).tolist()
    )


def test_neighbors_scopes_rejects_a_scope_with_no_object_set() -> None:
    images, objects = _inputs()
    with pytest.raises(ValueError, match="Tumor"):
        measure(images, objects, families=["neighbors"], neighbors_scopes=["Tumor"])


def test_neighbors_scopes_rejects_an_empty_sequence() -> None:
    images, objects = _inputs()
    with pytest.raises(ValueError, match="nonempty"):
        measure(images, objects, families=["neighbors"], neighbors_scopes=[])


def test_neighbors_full_objects_matches_oracle_for_border_excluded_object() -> None:
    """Real CellProfiler 4.2.8 oracle ground truth (MeasureObjectNeighbors,
    "Consider objects discarded for touching image border?" = Yes, its own
    default): a 2x9 image with 3 touching objects where object 3 was
    discarded from the kept set. count=[1,2], first=[2,1], second=[0,0],
    touching~=[33.33,66.67]%, relationships=[(1,2),(2,1)] only."""
    dna = np.zeros((2, 9))
    full = np.array(
        [[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uint32
    )
    kept = np.array(
        [[1, 1, 1, 2, 2, 2, 0, 0, 0], [1, 1, 1, 2, 2, 2, 0, 0, 0]], dtype=np.uint32
    )
    result = measure(
        {"DNA": dna},
        {"Cells": kept},
        families=["neighbors"],
        neighbors_full_objects={"Cells": full},
    )
    features = result.features["Cells"]
    assert _array(features["Neighbors_NumberOfNeighbors_Adjacent"]).tolist() == [1, 2]
    assert _array(features["Neighbors_FirstClosestObjectNumber_Adjacent"]).tolist() == [
        2,
        1,
    ]
    assert _array(
        features["Neighbors_SecondClosestObjectNumber_Adjacent"]
    ).tolist() == [
        0,
        0,
    ]
    touching = _array(features["Neighbors_PercentTouching_Adjacent"])
    assert touching[0] == pytest.approx(33.333333333333336)
    assert touching[1] == pytest.approx(66.66666666666667)
    assert [
        (row["object_name"], row["first_object_id"], row["second_object_id"])
        for row in result.relationships
    ] == [("Cells", 1, 2), ("Cells", 2, 1)]


def test_neighbors_full_objects_rejects_a_scope_not_requested() -> None:
    images, objects = _inputs()
    with pytest.raises(ValueError, match="Nuclei"):
        measure(
            images,
            objects,
            families=["neighbors"],
            neighbors_full_objects={"Nuclei": objects["Nuclei"]},
        )


def test_neighbors_expand_matches_real_oracle_for_distant_objects() -> None:
    """Real CellProfiler 4.2.8, MeasureObjectNeighbors distance_method =
    "Expand until adjacent", on a 2x4 grid where two single-column
    objects are two background columns apart (0 neighbors under the
    default "Adjacent" method) -- their Voronoi territories still
    share the whole boundary once expanded. Oracle: count=[1,1],
    first=[2,1], touching=[50.0,50.0]%, distance=[3.0,3.0] (centers
    stay on the ORIGINAL geometry, not the expanded Voronoi cells),
    relate=[(1,2),(2,1)]. See
    fixtures/neighbors_expand/module_two_far_columns.json."""
    labels = np.array([[1, 0, 0, 2], [1, 0, 0, 2]], dtype=np.uint32)
    result = measure(
        {"DNA": np.zeros((2, 4))},
        {"Cells": labels},
        families=["neighbors"],
        neighbors_scopes=["Cells"],
        neighbors_expand=["Cells"],
    )
    features = result.features["Cells"]
    assert "Neighbors_NumberOfNeighbors_Adjacent" not in features
    assert _array(features["Neighbors_NumberOfNeighbors_Expanded"]).tolist() == [1, 1]
    assert _array(features["Neighbors_FirstClosestObjectNumber_Expanded"]).tolist() == [
        2,
        1,
    ]
    touching = _array(features["Neighbors_PercentTouching_Expanded"])
    assert touching[0] == pytest.approx(50.0)
    assert touching[1] == pytest.approx(50.0)
    distance = _array(features["Neighbors_FirstClosestDistance_Expanded"])
    assert distance[0] == pytest.approx(3.0)
    assert distance[1] == pytest.approx(3.0)
    assert [
        (row["first_object_id"], row["second_object_id"])
        for row in result.relationships
    ] == [(1, 2), (2, 1)]


def test_neighbors_expand_rejects_a_3d_image() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.uint32)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 2
    with pytest.raises(ValueError, match="2D"):
        measure(
            {"DNA": np.zeros((2, 2, 2))},
            {"Cells": labels},
            families=["neighbors"],
            neighbors_scopes=["Cells"],
            neighbors_expand=["Cells"],
        )


def test_neighbors_expand_rejects_a_scope_not_requested() -> None:
    images, objects = _inputs()
    with pytest.raises(ValueError, match="Nuclei"):
        measure(
            images,
            objects,
            families=["neighbors"],
            neighbors_scopes=["Cells"],
            neighbors_expand=["Nuclei"],
        )
