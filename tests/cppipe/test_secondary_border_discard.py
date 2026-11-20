"""Secondary border discard against CellProfiler 4.2.8."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, identify_secondary_objects, load_pipeline

FIXTURES = (
    Path(__file__).resolve().parents[2] / "fixtures/cppipe/secondary_border_discard"
)
METHODS = ("propagation", "watershed_image")
PRIMARY_DISCARD = "Discard the associated primary objects?"


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_discards_edge_cells_and_keeps_primary_ids(method: str) -> None:
    expected = json.loads((FIXTURES / f"{method}.json").read_text())
    images = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in expected["images"].items()
    }
    primary, secondary, _ = load_pipeline(FIXTURES / f"{method}.cppipe").run(images)
    for name, result in (("Nuclei", primary["Nuclei"]), ("Cells", secondary["Cells"])):
        for layer, labels in expected[name].items():
            np.testing.assert_array_equal(getattr(result, layer), labels)


@pytest.mark.parametrize("method", METHODS)
def test_mask_border_discards_only_cardinal_contacts(method: str) -> None:
    case = json.loads((FIXTURES / f"masked_{method}.json").read_text())
    primary = np.asarray(case["primary"], dtype=np.uint32)
    result = identify_secondary_objects(
        (primary > 0).astype(np.float32),
        primary,
        mask=np.asarray(case["mask"], dtype=bool),
        method=method,
        threshold=0.5,
        discard_edge=True,
    )
    for layer, labels in case["expected"].items():
        np.testing.assert_array_equal(getattr(result, layer), labels)
    np.testing.assert_array_equal(primary, case["primary"])


def test_pipeline_rejects_discarding_associated_primaries(tmp_path: Path) -> None:
    source = (FIXTURES / "propagation.cppipe").read_text()
    path = tmp_path / "discard_associated.cppipe"
    path.write_text(source.replace(f"{PRIMARY_DISCARD}:No", f"{PRIMARY_DISCARD}:Yes"))
    with pytest.raises(CppipeError, match="Discard the associated primary objects"):
        load_pipeline(path)
