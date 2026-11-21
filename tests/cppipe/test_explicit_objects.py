"""Preflight and run pipelines that consume labels supplied by the caller."""

from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, inspect_pipeline, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"


def _without_primary(tmp_path: Path, *, plugin: bool = True) -> Path:
    header, primary, *steps = (
        (FIXTURES / "manual.cppipe").read_text().strip().split("\n\n")
    )
    assert primary.startswith("IdentifyPrimaryObjects:")
    if plugin:
        steps.insert(
            0, "RunCellpose:[module_num:1|variable_revision_number:3|enabled:True]"
        )
    else:
        header = header.replace("ModuleCount:4", "ModuleCount:3")
    path = tmp_path / "caller_objects.cppipe"
    path.write_text("\n\n".join([header, *steps]) + "\n")
    return path


def test_preflight_reports_missing_required_labels(tmp_path: Path) -> None:
    path = _without_primary(tmp_path)
    entries = inspect_pipeline(path)
    assert entries[0].status == "unsupported_module"
    assert entries[1].status == "missing_dependency"
    assert "Cells" in entries[1].detail
    assert inspect_pipeline(path, available_objects=("Cells",))[1].status == "supported"


def test_missing_labels_stop_before_segmentation(tmp_path: Path) -> None:
    path = tmp_path / "missing_labels.cppipe"
    path.write_text(
        (FIXTURES / "manual.cppipe")
        .read_text()
        .replace(
            "Select objects to measure:Cells", "Select objects to measure:Missing", 1
        )
    )
    pipeline = load_pipeline(path)
    with pytest.raises(
        CppipeError, match=r"MeasureObjectIntensity needs object set 'Missing'"
    ):
        pipeline.run({"DNA": np.zeros((1, 5, 6), dtype=np.float32)})


def test_later_producer_cannot_satisfy_an_earlier_consumer(tmp_path: Path) -> None:
    header, primary, intensity, *other = (
        (FIXTURES / "manual.cppipe").read_text().strip().split("\n\n")
    )
    path = tmp_path / "forward_reference.cppipe"
    path.write_text("\n\n".join((header, intensity, primary, *other)) + "\n")
    assert inspect_pipeline(path)[0].status == "missing_dependency"
    with pytest.raises(CppipeError, match=r"MeasureObjectIntensity.*Cells.*later"):
        load_pipeline(path).run({"DNA": np.zeros((8, 8), dtype=np.float32)})


def test_inspection_rejects_a_secondary_output_as_seeds(tmp_path: Path) -> None:
    _, _, step, _ = (
        (FIXTURES / "secondary_regressions/pipeline_border_distance_n.cppipe")
        .read_text()
        .strip()
        .split("\n\n")
    )
    chained = step.replace(":Cells\n", ":MoreCells\n").replace(":Nuclei\n", ":Cells\n")
    path = tmp_path / "secondary_chain.cppipe"
    path.write_text(f"CellProfiler Pipeline:\nModuleCount:2\n\n{step}\n\n{chained}\n")
    assert [x.status for x in inspect_pipeline(path, available_objects=["Nuclei"])] == [
        "supported",
        "missing_dependency",
    ]


def test_run_accepts_explicit_labels_without_modifying_them(tmp_path: Path) -> None:
    pipeline = load_pipeline(_without_primary(tmp_path))
    images = {"DNA": np.ones((5, 6), dtype=np.float32)}
    with pytest.raises(
        CppipeError, match=r"MeasureObjectIntensity needs object set 'Cells'"
    ):
        pipeline.run_detailed(images)
    labels = np.zeros((5, 6), dtype=np.uint32)
    labels[1:3, 1:3] = 1
    labels[2:4, 3:5] = 2
    labels.setflags(write=False)
    result = pipeline.run_detailed(images, objects={"Cells": labels})
    assert result.measurements is not None
    assert result.measurements.object_ids["Cells"].tolist() == [1, 2]
    assert not result.primary and not result.secondary
    assert not result.derived_images


def test_excluded_neighbors_require_an_explicit_diagnostic_plane(
    tmp_path: Path,
) -> None:
    path = _without_primary(tmp_path, plugin=False)
    text = path.read_text().replace(
        "Consider objects discarded for touching image border?:No",
        "Consider objects discarded for touching image border?:Yes",
    )
    assert text != path.read_text()
    path.write_text(text)
    pipeline = load_pipeline(path)
    images = {"DNA": np.ones((5, 6), dtype=np.float32)}
    labels = np.zeros((5, 6), dtype=np.uint32)
    labels[1:3, 1:3] = 1
    diagnostic = labels.copy()
    diagnostic[1:3, 3:5] = 2
    assert (
        inspect_pipeline(path, available_objects=("Cells",))[-1].status
        == "missing_dependency"
    )
    assert (
        inspect_pipeline(
            path, available_objects=("Cells",), available_object_diagnostics=("Cells",)
        )[-1].status
        == "supported"
    )
    with pytest.raises(
        CppipeError, match=r"MeasureObjectNeighbors.*Cells.*object_diagnostics"
    ):
        pipeline.run_detailed(images, objects={"Cells": labels})
    result = pipeline.run_detailed(
        images, objects={"Cells": labels}, object_diagnostics={"Cells": diagnostic}
    )
    assert result.measurements is not None
    assert result.measurements.object_ids["Cells"].tolist() == [1]
    np.testing.assert_array_equal(
        result.measurements.features["Cells"]["Neighbors_NumberOfNeighbors_Adjacent"],
        [1],
    )


def test_neighbors_can_run_on_labels_without_an_intensity_image(tmp_path: Path) -> None:
    source = (FIXTURES / "manual.cppipe").read_text()
    header, *steps = source.strip().split("\n\n")
    path = tmp_path / "neighbors.cppipe"
    path.write_text(
        header.replace("ModuleCount:4", "ModuleCount:1") + "\n\n" + steps[-1] + "\n"
    )
    labels = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint32)
    _, _, result = load_pipeline(path).run({}, objects={"Cells": labels})
    assert result is not None
    assert result.object_ids["Cells"].tolist() == [1]


@pytest.mark.parametrize(
    ("name", "labels", "match"),
    [
        ("", np.ones((2, 2), dtype=np.uint32), "object names must not be empty"),
        ("Cells", np.ones((2, 2), dtype=np.float32), "integer dtype"),
        ("Cells", np.full((2, 2), -1, dtype=np.int32), "between zero"),
        ("Cells", np.zeros((0, 2), dtype=np.uint32), "nonempty 2D or 3D"),
    ],
)
def test_bad_explicit_labels_are_rejected(
    tmp_path: Path, name: str, labels: np.ndarray, match: str
) -> None:
    pipeline = load_pipeline(_without_primary(tmp_path))
    with pytest.raises(CppipeError, match=match):
        pipeline.run_detailed({}, objects={name: labels})


def test_diagnostic_shape_must_match_kept_labels(tmp_path: Path) -> None:
    pipeline = load_pipeline(_without_primary(tmp_path))
    with pytest.raises(CppipeError, match=r"Cells.*same shape"):
        pipeline.run_detailed(
            {},
            objects={"Cells": np.zeros((2, 3), dtype=np.uint32)},
            object_diagnostics={"Cells": np.zeros((3, 2), dtype=np.uint32)},
        )


def test_explicit_objects_cannot_override_pipeline_segmentation() -> None:
    pipeline = load_pipeline(FIXTURES / "manual.cppipe")
    labels = np.zeros((5, 6), dtype=np.uint32)
    with pytest.raises(CppipeError, match=r"Cells.*produced by"):
        pipeline.run_detailed(
            {"DNA": labels.astype(np.float32)}, objects={"Cells": labels}
        )
    assert (
        inspect_pipeline(FIXTURES / "manual.cppipe", available_objects=("Cells",))[
            0
        ].status
        == "missing_dependency"
    )
