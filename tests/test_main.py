"""Tests for the public measurement API against saved golden fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather

from cp_anywhere import measure

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"
GOLDEN = json.loads((ROOT / "fixtures/reference/touching_gradient.json").read_text())
TOTAL_FEATURE_COLUMNS = 119  # spec/compatibility.md's declared scope


def _load_fixture() -> tuple[dict, dict]:
    images = {name: np.load(FIXTURE / f"{name}.npy") for name in ("DNA", "Actin")}
    objects = {
        "Cells": np.load(FIXTURE / "labels.npy"),
        "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
    }
    return images, objects


def test_measure_matches_golden_feature_count() -> None:
    """The full declared scope emits exactly 119 columns."""
    images, objects = _load_fixture()
    result = measure(images, objects, image_id="touching_gradient")
    assert (
        sum(len(columns) for columns in result.features.values())
        == TOTAL_FEATURE_COLUMNS
    )
    assert result.object_ids["Cells"].tolist() == GOLDEN["object_ids"]["Cells"]
    assert len(result.relationships) == len(GOLDEN["relationships"])


def test_measure_rejects_mismatched_shapes() -> None:
    """Images and objects must share one consistent shape."""
    images, objects = _load_fixture()
    objects["Cells"] = objects["Cells"][:-1, :]
    with pytest.raises(ValueError, match="same shape"):
        measure(images, objects, image_id="touching_gradient")


def test_measure_selects_families() -> None:
    """Restricting families limits the emitted feature groups."""
    images, objects = _load_fixture()
    result = measure(
        images, objects, image_id="touching_gradient", families=["image_intensity"]
    )
    assert set(result.features) == {"Image"}


def test_measure_requires_cells_for_neighbors() -> None:
    """Neighbors require a Cells object set by name."""
    images, objects = _load_fixture()
    with pytest.raises(ValueError, match="images and objects must be nonempty"):
        measure({}, {}, image_id="touching_gradient")
    with pytest.raises(ValueError, match="neighbors"):
        measure(images, {"Other": objects["Cells"]}, image_id="touching_gradient")


def test_measurement_result_round_trips_through_csv_and_json(tmp_path: Path) -> None:
    """Exported long-form rows and the JSON structure stay decodable."""
    images, objects = _load_fixture()
    result = measure(images, objects, image_id="touching_gradient")
    csv_path = tmp_path / "out.csv"
    json_path = tmp_path / "out.json"
    arrow_path = tmp_path / "out.arrow"
    result.to_csv(csv_path)
    result.to_json(json_path)
    result.to_arrow(arrow_path)
    assert (
        csv_path.read_text().splitlines()[0] == "image_id,scope,object_id,feature,value"
    )
    decoded = json.loads(json_path.read_text())
    assert decoded["image_id"] == "touching_gradient"
    assert decoded["features"]["Cells"]

    table = feather.read_table(arrow_path)
    assert table.column_names == ["image_id", "scope", "object_id", "feature", "value"]
    assert table.schema.field("object_id").type == pa.int64()
    assert table.schema.field("value").type == pa.float64()
    assert table.num_rows == len(result.rows())


def test_arrow_export_preserves_nonfinite_values_as_real_floats(
    tmp_path: Path,
) -> None:
    """Unlike to_json's string markers, Arrow keeps NaN/Infinity as floats."""
    folder = ROOT / "fixtures/synthetic/nonfinite"
    images = {name: np.load(folder / f"{name}.npy") for name in ("DNA", "Actin")}
    objects = {
        "Cells": np.load(folder / "labels.npy"),
        "Nuclei": np.load(folder / "nuclei_labels.npy"),
    }
    result = measure(images, objects, image_id="nonfinite")
    arrow_path = tmp_path / "nonfinite.arrow"
    result.to_arrow(arrow_path)
    table = feather.read_table(arrow_path)
    values = table.column("value").to_pylist()
    assert any(math.isnan(v) for v in values)
    assert any(math.isinf(v) for v in values)
