"""Verify MeasureObjectIntensityDistribution cppipe translation."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
_BIN_COUNT = 3
_ZERNIKE_DEGREE = 4
RADIAL = FIXTURES / "intensity_distribution"


@pytest.mark.parametrize(
    "case", ["jump_groups", "scaled_then_unscaled", "unscaled_then_scaled"]
)
def test_radial_groups_match_cellprofiler(case: str) -> None:
    fixture = json.loads((RADIAL / f"{case}.oracle.json").read_text())
    pipeline = load_pipeline(RADIAL / f"{case}.cppipe")
    assert len(pipeline.intensity_distribution_steps) == 1
    assert pipeline.intensity_distribution_steps[0].objects == (
        "Nuclei",
        "Cytoplasm",
        "Cells",
    )
    images = {
        name: np.asarray(plane["values"], dtype=plane["dtype"])
        for name, plane in fixture["inputs"].items()
    }
    _, objects, result = pipeline.run(images)
    assert result is not None
    np.testing.assert_array_equal(objects["Cytoplasm"].labels, fixture["cytoplasm"])
    assert result.features.keys() == fixture["features"].keys()
    for scope, features in fixture["features"].items():
        np.testing.assert_array_equal(
            result.object_ids[scope], fixture["object_ids"][scope]
        )
        assert result.features[scope].keys() == features.keys()
        for name, values in features.items():
            expected = [np.nan if v == {"nonfinite": "NaN"} else v for v in values]
            np.testing.assert_allclose(
                result.features[scope][name],
                expected,
                rtol=1e-5,
                atol=1e-7,
                equal_nan=True,
                err_msg=f"{scope}.{name}",
            )
    result.rows()


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("Hidden:3\n    Hidden:3", "Hidden:2\n    Hidden:3", "settings"),
        ("Hidden:3\n    Hidden:3", "Hidden:3\n    Hidden:-1", "bin group count"),
        ("Hidden:0", "Hidden:1", "heatmap"),
        ("Number of bins:3", "Number of bins:2", "duplicate bin count"),
        ("Scale the bins?:No", "Scale the bins?:Maybe", "Yes/No"),
        (
            "Select objects to measure:Cells\n"
            "    Object to use as center?:These objects",
            "Select objects to measure:Cells\n"
            "    Object to use as center?:Edges of other objects",
            "Edges of other objects",
        ),
    ],
)
def test_radial_groups_validate_settings(
    tmp_path: Path, old: str, new: str, error: str
) -> None:
    original = (RADIAL / "scaled_then_unscaled.cppipe").read_text()
    assert old in original
    path = tmp_path / "invalid.cppipe"
    path.write_text(original.replace(old, new, 1))
    with pytest.raises(CppipeError, match=error):
        load_pipeline(path)


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
