"""Verify three-compartment size/shape execution and per-module flags."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_shape_settings_do_not_leak_between_compartments(case: str) -> None:
    path = FIXTURES / "size_shape" / f"{case}.cppipe"
    expected = json.loads(path.with_suffix(".json").read_text())
    pipeline = load_pipeline(path)
    shape_steps = [
        s for s in pipeline.measurement_steps if s.family == "object_size_shape"
    ]
    assert [
        (s.objects, s.calculate_zernikes, s.calculate_advanced) for s in shape_steps
    ] == [
        (("Nuclei",), True, False),
        (("Cells",), False, False),
        (("Cytoplasm",), True, True),
    ]
    images = {
        name: np.load(
            FIXTURES
            / "secondary"
            / f"identify_secondary_objects_propagation.{suffix}.npy"
        )
        for name, suffix in (("DNA", "dna"), ("Stain", "stain"))
    }
    _, _, result = pipeline.run(images)
    assert result is not None
    for scope, features in expected["features"].items():
        expected_shape = {name for name in features if name.startswith("AreaShape_")}
        actual_shape = {
            name for name in result.features[scope] if name.startswith("AreaShape_")
        }
        assert actual_shape == expected_shape
        for name, values in features.items():
            np.testing.assert_allclose(
                result.features[scope][name],
                values,
                rtol=1e-7,
                atol=1e-10,
                equal_nan=True,
                err_msg=f"{scope}.{name}",
            )
    result.rows()


def test_size_shape_revision_is_checked(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[2] / "fixtures/size_shape/rectangle.cppipe"
    )
    text = source.read_text().replace(
        "variable_revision_number:3", "variable_revision_number:999"
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"MeasureObjectSizeShape.*999"):
        load_pipeline(path)
