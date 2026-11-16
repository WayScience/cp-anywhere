"""Compare imported tertiary pipelines with freshly generated CP fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, identify_tertiary_objects, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
TERTIARY = FIXTURES / "tertiary"
SECONDARY = FIXTURES / "secondary"
PREFIX = "identify_secondary_objects_propagation"


def _images() -> dict[str, np.ndarray]:
    return {
        name: np.load(SECONDARY / f"{PREFIX}.{suffix}.npy")
        for name, suffix in (("DNA", "dna"), ("Stain", "stain"))
    }


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_tertiary_pipeline_matches_real_labels_and_intensity(case: str) -> None:
    expected = json.loads((TERTIARY / f"{case}.expected.json").read_text())
    pipeline = load_pipeline(TERTIARY / f"{case}.cppipe")
    assert "IdentifyTertiaryObjects" not in pipeline.skipped_modules
    assert len(pipeline.tertiary_steps) == 1
    assert pipeline.tertiary_steps[0].shrink_primary == expected["shrink_primary"]
    primary, derived, measured = pipeline.run(_images())
    np.testing.assert_array_equal(
        primary["Nuclei"].labels, expected["labels"]["Nuclei"]
    )
    for name in ("Cells", "Cytoplasm"):
        np.testing.assert_array_equal(derived[name].labels, expected["labels"][name])
    assert measured is not None
    values = measured.features["Cytoplasm"]["Intensity_MeanIntensity_Stain"]
    np.testing.assert_allclose(values, expected["mean_intensity"], rtol=1e-6)


@pytest.mark.parametrize("case", ["shrink", "no_shrink"])
def test_tertiary_array_api_matches_real_oracle(case: str) -> None:
    expected = json.loads((TERTIARY / f"{case}.expected.json").read_text())
    labels = expected["labels"]
    result = identify_tertiary_objects(
        np.asarray(labels["Nuclei"], dtype=np.uint32),
        np.asarray(labels["Cells"], dtype=np.uint32),
        shrink_primary=expected["shrink_primary"],
    )
    np.testing.assert_array_equal(result.labels, labels["Cytoplasm"])


def test_missing_tertiary_input_is_reported(tmp_path: Path) -> None:
    text = (
        (TERTIARY / "shrink.cppipe")
        .read_text()
        .replace(
            "Select the smaller identified objects:Nuclei",
            "Select the smaller identified objects:Missing",
        )
    )
    path = tmp_path / "missing.cppipe"
    path.write_text(text)
    pipeline = load_pipeline(path)
    with pytest.raises(CppipeError, match="Missing"):
        pipeline.run(_images())


def test_tertiary_revision_is_checked(tmp_path: Path) -> None:
    text = (
        (TERTIARY / "shrink.cppipe")
        .read_text()
        .replace(
            "IdentifyTertiaryObjects:[module_num:3|svn_version:'Unknown'|variable_revision_number:3",
            "IdentifyTertiaryObjects:[module_num:3|svn_version:'Unknown'|variable_revision_number:999",
        )
    )
    path = tmp_path / "revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"IdentifyTertiaryObjects.*999"):
        load_pipeline(path)
