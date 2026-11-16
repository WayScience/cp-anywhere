"""Verify MeasureGranularity cppipe translation and multi-object execution."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_granularity_settings_and_multi_object_execution(case: str) -> None:
    path = FIXTURES / "granularity" / f"{case}.cppipe"
    expected = json.loads(path.with_suffix(".json").read_text())
    pipeline = load_pipeline(path)
    assert len(pipeline.granularity_steps) == 1
    step = pipeline.granularity_steps[0]
    assert step.images == ("Stain",)
    assert step.objects == ("Nuclei", "Cells")
    assert step.subsample_size == pytest.approx(0.5)
    assert step.image_sample_size == pytest.approx(0.5)
    expected_element_size = 4
    expected_granular_spectrum_length = 6
    assert step.element_size == expected_element_size
    assert step.granular_spectrum_length == expected_granular_spectrum_length
    # Real CellProfiler's Image(..., convert=True) auto-rescales integer
    # input by dtype max and always downcasts through float32 (confirmed in
    # cellprofiler_core/image/_image.py's pixel_data setter). cp_anywhere's
    # own identify_primary_objects already reproduces that integer rescale
    # internally (see segment.py's rescale=True default), so DNA (uint16
    # here) must be passed RAW, not pre-rescaled -- pre-converting it to
    # float64 without dividing by 65535 would silently corrupt the
    # threshold entirely. Stain is already float64 in [0, 1] (no rescale
    # needed), so only its float32 downcast precision loss is replicated
    # here, matching what real MeasureGranularity actually measured.
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


def test_granularity_revision_is_checked(tmp_path: Path) -> None:
    source = FIXTURES / "granularity" / "shrink.cppipe"
    text = source.read_text().replace(
        "MeasureGranularity:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:4|",
        "MeasureGranularity:[module_num:5|svn_version:'Unknown'|"
        "variable_revision_number:999|",
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"MeasureGranularity.*999"):
        load_pipeline(path)


def test_granularity_requires_an_untranslated_object_to_raise(tmp_path: Path) -> None:
    source = FIXTURES / "granularity" / "shrink.cppipe"
    text = source.read_text().replace("Nuclei, Cells", "Nuclei, Cells, Mystery")
    path = tmp_path / "missing_object.cppipe"
    path.write_text(text)
    pipeline = load_pipeline(path)
    images = {
        name: np.load(
            FIXTURES
            / "secondary"
            / f"identify_secondary_objects_propagation.{suffix}.npy"
        )
        for name, suffix in (("DNA", "dna"), ("Stain", "stain"))
    }
    with pytest.raises(CppipeError, match="Mystery"):
        pipeline.run(images)
