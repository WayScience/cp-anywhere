"""Verify MeasureColocalization cppipe translation."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
_THRESHOLD_PERCENT = 15.0


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_colocalization_settings_and_execution(case: str) -> None:
    path = FIXTURES / "colocalization" / f"{case}.cppipe"
    expected = json.loads(path.with_suffix(".json").read_text())
    pipeline = load_pipeline(path)
    assert len(pipeline.colocalization_steps) == 1
    step = pipeline.colocalization_steps[0]
    assert step.images == ("DNA", "Stain")
    assert step.objects == ("Cells",)
    assert step.threshold_percent == pytest.approx(_THRESHOLD_PERCENT)
    assert step.fast_costes == "faster"
    assert step.do_corr_and_slope is True
    assert step.do_manders is True
    assert step.do_rwc is True
    assert step.do_overlap is True
    assert step.do_costes is True
    assert step.wants_image_measurements is True
    assert step.wants_object_measurements is True
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


def test_colocalization_revision_is_checked(tmp_path: Path) -> None:
    source = FIXTURES / "colocalization" / "shrink.cppipe"
    text = source.read_text().replace(
        "MeasureColocalization:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:5|",
        "MeasureColocalization:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:999|",
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"MeasureColocalization.*999"):
        load_pipeline(path)


def test_colocalization_rejects_a_single_selected_image(tmp_path: Path) -> None:
    source = FIXTURES / "colocalization" / "shrink.cppipe"
    text = source.read_text().replace(
        "Select images to measure:DNA, Stain",
        "Select images to measure:DNA",
    )
    path = tmp_path / "one_image.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="at least 2 selected images"):
        load_pipeline(path)
