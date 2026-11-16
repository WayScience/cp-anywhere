"""Verify ``cp_anywhere.cppipe`` against real, oracle-generated
``.cppipe`` files (``scripts/cppipe/generate_fixtures.py``), not
hand-crafted text: parsing must recover the exact settings used to
build each file, and running the translated pipeline on the same
fixed test image must reproduce the real CellProfiler oracle's labels
exactly (not approximately) -- the scalar threshold alone gets a
narrow float32(CellProfiler)-vs-float64(this project) tolerance,
matching the same pattern established in
tests/segmentation/test_reference_alignment.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.cppipe import CppipeError, load_pipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "fixtures/cppipe"
THRESHOLD_TOLERANCE = 1e-3
MINIMUM_CASE_COUNT = 10

CASE_NAMES = sorted(p.stem for p in FIXTURE_DIR.glob("*.cppipe"))


def _load_case(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.expected.json").read_text())


def _test_image(name: str | None = None) -> np.ndarray:
    if name is not None:
        per_case = FIXTURE_DIR / f"{name}.test_image.npy"
        if per_case.exists():
            return np.load(per_case)
    return np.load(FIXTURE_DIR / "test_image.npy")


def test_fixture_directory_is_not_empty() -> None:
    assert len(CASE_NAMES) >= MINIMUM_CASE_COUNT


@pytest.mark.parametrize("name", CASE_NAMES)
def test_every_real_cppipe_fixture_segments_exactly_like_the_oracle(
    name: str,
) -> None:
    expected = _load_case(name)
    pipeline = load_pipeline(FIXTURE_DIR / f"{name}.cppipe")

    assert len(pipeline.segmentation_steps) == 1
    step = pipeline.segmentation_steps[0]
    assert step.input_image == "DNA"
    assert step.output_object == "Cells"

    segmentations, _secondary, _measurement = pipeline.run({"DNA": _test_image(name)})
    result = segmentations["Cells"]

    expected_labels = np.load(FIXTURE_DIR / f"{name}.labels.npy")
    np.testing.assert_array_equal(result.labels, expected_labels)
    assert result.object_count == expected["object_count"]
    assert abs(result.threshold - expected["threshold"]) <= THRESHOLD_TOLERANCE


@pytest.mark.parametrize(
    "name", [n for n in CASE_NAMES if n != "segmentation_only_no_measure"]
)
def test_measure_modules_are_translated_and_runnable(name: str) -> None:
    pipeline = load_pipeline(FIXTURE_DIR / f"{name}.cppipe")
    families = {step.family for step in pipeline.measurement_steps}
    assert families == {"image_intensity", "object_intensity", "neighbors"}

    _segmentations, _secondary, result = pipeline.run({"DNA": _test_image(name)})
    assert result is not None
    assert "Image" in result.features
    assert "Cells" in result.features
    assert "Cells" in result.object_ids


def test_segmentation_only_pipeline_has_no_measurement_steps() -> None:
    pipeline = load_pipeline(FIXTURE_DIR / "segmentation_only_no_measure.cppipe")
    assert pipeline.measurement_steps == []
    segmentations, _secondary, measurement = pipeline.run({"DNA": _test_image()})
    assert measurement is None
    assert "Cells" in segmentations


def test_neighbors_excluded_border_object_matches_a_fresh_oracle_run() -> None:
    """``wants_excluded_objects=True`` (CellProfiler's own default) end
    to end through the full cppipe import + run() path: the object
    discarded for touching the image border must be routed via
    ``SegmentationResult.small_removed_labels`` into
    ``neighbors_full_objects``, reproducing a real, freshly-run
    CellProfiler oracle's own numbers exactly for this fixture's real
    kept/small_removed label pair (see
    scripts/cppipe/generate_fixtures.py's `_real_neighbors_excluded`)."""
    name = "neighbors_excluded_border_object"
    expected = _load_case(name)
    image = _test_image(name)
    pipeline = load_pipeline(FIXTURE_DIR / f"{name}.cppipe")

    neighbors_steps = [s for s in pipeline.measurement_steps if s.family == "neighbors"]
    assert len(neighbors_steps) == 1
    assert neighbors_steps[0].wants_excluded_objects is True

    segmentations, _secondary, result = pipeline.run({"DNA": image})
    assert result is not None
    assert segmentations["Cells"].object_count == expected["object_count"]
    features = result.features["Cells"]
    ground_truth = expected["neighbors_excluded"]
    count = features["Neighbors_NumberOfNeighbors_Adjacent"]
    assert isinstance(count, np.ndarray)
    assert count.tolist() == [int(v) for v in ground_truth["count"]]
    first = features["Neighbors_FirstClosestObjectNumber_Adjacent"]
    assert isinstance(first, np.ndarray)
    assert first.tolist() == ground_truth["first"]
    touching = features["Neighbors_PercentTouching_Adjacent"]
    assert isinstance(touching, np.ndarray)
    for actual, want in zip(touching, ground_truth["touching"], strict=True):
        assert actual == pytest.approx(want)


def test_unknown_leading_pipeline_modules_are_skipped_not_dropped_silently(
    tmp_path: Path,
) -> None:
    text = (FIXTURE_DIR / "otsu_simple.cppipe").read_text()
    # Splice in a module this importer does not translate, matching a
    # real pipeline's typical shape (Images always precedes everything).
    header, _, rest = text.partition("\n\n")
    injected = (
        f"{header}\n\n"
        "Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2"
        "|show_window:False|notes:[]|batch_state:array([], dtype=uint8)"
        "|enabled:True|wants_pause:False]\n"
        "    :\n"
        "    Filter images?:No filtering\n\n"
        f"{rest}"
    )
    injected = injected.replace("ModuleCount:4", "ModuleCount:5")
    path = tmp_path / "with_images_module.cppipe"
    path.write_text(injected)

    pipeline = load_pipeline(path)
    assert "Images" in pipeline.skipped_modules
    assert len(pipeline.segmentation_steps) == 1


def test_not_a_cppipe_file_raises_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "not_a_pipeline.cppipe"
    path.write_text("just some random text\n")
    with pytest.raises(CppipeError, match="not a CellProfiler"):
        load_pipeline(path)


def test_wrong_variable_revision_number_raises_a_clear_error(tmp_path: Path) -> None:
    text = (FIXTURE_DIR / "otsu_simple.cppipe").read_text()
    text = text.replace(
        "IdentifyPrimaryObjects:[module_num:1|svn_version:'Unknown'|variable_revision_number:15",
        "IdentifyPrimaryObjects:[module_num:1|svn_version:'Unknown'|variable_revision_number:11",
    )
    path = tmp_path / "old_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="variable_revision_number=11"):
        load_pipeline(path)


def test_use_advanced_no_raises_a_clear_error(tmp_path: Path) -> None:
    text = (FIXTURE_DIR / "otsu_simple.cppipe").read_text()
    text = text.replace("Use advanced settings?:Yes", "Use advanced settings?:No")
    path = tmp_path / "basic_mode.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="advanced settings"):
        load_pipeline(path)


def test_non_adjacent_neighbors_distance_method_raises_a_clear_error(
    tmp_path: Path,
) -> None:
    text = (FIXTURE_DIR / "otsu_simple.cppipe").read_text()
    text = text.replace(
        "Method to determine neighbors:Adjacent",
        "Method to determine neighbors:Within",
    )
    path = tmp_path / "expanded_neighbors.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="distance method"):
        load_pipeline(path)


def test_identify_secondary_objects_matches_a_fresh_oracle_run() -> None:
    secondary_dir = FIXTURE_DIR / "secondary"
    pipeline = load_pipeline(
        secondary_dir / "identify_secondary_objects_propagation.cppipe"
    )
    assert pipeline.segmentation_steps == [
        pipeline.segmentation_steps[0]
    ]  # exactly one IdentifyPrimaryObjects step
    assert len(pipeline.secondary_steps) == 1
    step = pipeline.secondary_steps[0]
    assert step.input_image == "Stain"
    assert step.primary_object == "Nuclei"
    assert step.output_object == "Cells"
    assert step.config["method"] == "propagation"
    assert step.config["regularization_factor"] == pytest.approx(0.05)
    assert step.config["fill_holes"] is True

    dna = np.load(secondary_dir / "identify_secondary_objects_propagation.dna.npy")
    stain = np.load(secondary_dir / "identify_secondary_objects_propagation.stain.npy")
    expected = json.loads(
        (
            secondary_dir / "identify_secondary_objects_propagation.expected.json"
        ).read_text()
    )

    segmentations, secondary_results, _measurement = pipeline.run(
        {"DNA": dna, "Stain": stain}
    )
    assert segmentations["Nuclei"].object_count == expected["nuclei_object_count"]
    cells = secondary_results["Cells"]
    np.testing.assert_array_equal(
        cells.labels, np.array(expected["cells_labels"], dtype=np.uint32)
    )


def test_identify_secondary_objects_rejects_an_unknown_primary_object(
    tmp_path: Path,
) -> None:
    secondary_dir = FIXTURE_DIR / "secondary"
    text = (secondary_dir / "identify_secondary_objects_propagation.cppipe").read_text()
    text = text.replace(
        "Select the input objects:Nuclei", "Select the input objects:OtherNuclei"
    )
    path = tmp_path / "unknown_primary.cppipe"
    path.write_text(text)
    pipeline = load_pipeline(path)
    dna = np.load(secondary_dir / "identify_secondary_objects_propagation.dna.npy")
    stain = np.load(secondary_dir / "identify_secondary_objects_propagation.stain.npy")
    with pytest.raises(CppipeError, match="OtherNuclei"):
        pipeline.run({"DNA": dna, "Stain": stain})
