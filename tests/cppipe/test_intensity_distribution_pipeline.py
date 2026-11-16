"""Verify MeasureObjectIntensityDistribution cppipe translation."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
_BIN_COUNT = 3
_ZERNIKE_DEGREE = 4


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_intensity_distribution_settings_and_execution(case: str) -> None:
    path = FIXTURES / "intensity_distribution" / f"{case}.cppipe"
    expected = json.loads(path.with_suffix(".json").read_text())
    pipeline = load_pipeline(path)
    assert len(pipeline.intensity_distribution_steps) == 1
    step = pipeline.intensity_distribution_steps[0]
    assert step.images == ("Stain",)
    assert step.objects == ("Cells",)
    assert step.bin_count == _BIN_COUNT
    assert step.wants_scaled is True
    assert step.maximum_radius == pytest.approx(100.0)
    assert step.zernikes == "magnitudes"
    assert step.zernike_degree == _ZERNIKE_DEGREE
    # See tests/cppipe/test_granularity_pipeline.py's own note: DNA (uint16
    # here) must be passed RAW so identify_primary_objects applies real
    # CellProfiler's own integer rescale; Stain is already float64 in
    # [0, 1], so only its real float32 downcast precision loss matters.
    images = {
        "DNA": np.load(
            FIXTURES / "secondary/identify_secondary_objects_propagation.dna.npy"
        ),
        "Stain": np.load(
            FIXTURES / "secondary/identify_secondary_objects_propagation.stain.npy"
        )
        .astype(np.float32)
        .astype(np.float64),
    }
    _, _, result = pipeline.run(images)
    assert result is not None
    for scope, features in expected["features"].items():
        for name, values in features.items():
            np.testing.assert_allclose(
                result.features[scope][name],
                values,
                rtol=1e-5,
                atol=1e-7,
                equal_nan=True,
                err_msg=f"{scope}.{name}",
            )
    result.rows()


def test_intensity_distribution_revision_is_checked(tmp_path: Path) -> None:
    source = FIXTURES / "intensity_distribution" / "shrink.cppipe"
    text = source.read_text().replace(
        "MeasureObjectIntensityDistribution:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:6|",
        "MeasureObjectIntensityDistribution:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:999|",
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"MeasureObjectIntensityDistribution.*999"):
        load_pipeline(path)


def test_intensity_distribution_rejects_a_non_default_center_choice(
    tmp_path: Path,
) -> None:
    source = FIXTURES / "intensity_distribution" / "shrink.cppipe"
    text = source.read_text().replace(
        "Object to use as center?:These objects",
        "Object to use as center?:Centers of other objects",
    )
    path = tmp_path / "other_center.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="Centers of other objects"):
        load_pipeline(path)
