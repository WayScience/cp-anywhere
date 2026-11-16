"""Verify MeasureTexture cppipe translation."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
_GRAY_LEVELS = 16
_SCALE = 3


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_texture_settings_and_execution(case: str) -> None:
    path = FIXTURES / "texture" / f"{case}.cppipe"
    expected = json.loads(path.with_suffix(".json").read_text())
    pipeline = load_pipeline(path)
    assert len(pipeline.texture_steps) == 1
    step = pipeline.texture_steps[0]
    assert step.images == ("Stain",)
    assert step.objects == ("Cells",)
    assert step.gray_levels == _GRAY_LEVELS
    assert step.scales == (_SCALE,)
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


def test_texture_revision_is_checked(tmp_path: Path) -> None:
    source = FIXTURES / "texture" / "shrink.cppipe"
    text = source.read_text().replace(
        "MeasureTexture:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:7|",
        "MeasureTexture:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:999|",
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"MeasureTexture.*999"):
        load_pipeline(path)


def test_texture_rejects_a_mismatched_scale_count(tmp_path: Path) -> None:
    source = FIXTURES / "texture" / "shrink.cppipe"
    text = source.read_text().replace("Hidden:1", "Hidden:2")
    path = tmp_path / "bad_scale_count.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="MeasureTexture"):
        load_pipeline(path)
