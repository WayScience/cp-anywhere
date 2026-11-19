"""Run real translated CellProfiler pipelines on independent image sets."""

from multiprocessing import active_children
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import pytest

import cp_anywhere as cp

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PIPELINE = FIXTURES / "cppipe/tertiary/no_shrink.cppipe"


def test_batch_neighbor_relationships_use_the_field_id() -> None:
    path = FIXTURES / "cppipe/declump_shape_propagate_manual_smoothing.cppipe"
    dna = cp.load_array(_image_sets()["real_bbbc007"]["DNA"]).astype(np.float32) / 255
    (run,) = cp.run_batch(path, {"site-2": {"DNA": dna}})
    assert run.measurements is not None
    assert run.measurements.relationships
    for edge in run.measurements.relationships:
        assert edge["first_image_id"] == edge["second_image_id"] == "site-2"


def _image_sets() -> dict[str, dict[str, Path]]:
    return {
        field: {"DNA": path / "DNA.npy", "Stain": path / "Actin.npy"}
        for field in ("real_cell_cc0", "real_bbbc007")
        for path in (FIXTURES / "real" / field,)
    }


def test_batch_loads_a_pipeline_path_and_accepts_arrays() -> None:
    inputs = {"field": {"DNA": cp.load_array(_image_sets()["real_bbbc007"]["DNA"])}}
    (run,) = cp.run_batch(
        FIXTURES / "cppipe/segmentation_only_no_measure.cppipe", inputs
    )
    assert run.image_id == "field"
    assert run.primary
    assert run.measurements is None


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_batch_error_identifies_the_image_set(
    tmp_path: Path, backend: Literal["thread", "process"]
) -> None:
    image_sets = _image_sets()
    image_sets["missing-field"] = {"DNA": tmp_path / "missing.npy"}
    children = set(active_children())
    with pytest.raises(FileNotFoundError, match=r"missing\.npy") as error:
        list(cp.run_batch(PIPELINE, image_sets, workers=2, backend=backend))
    assert error.value.__notes__ == ["cp-anywhere image set: 'missing-field'"]
    assert set(active_children()) <= children


@pytest.mark.parametrize("workers", [0, -1, True])
def test_batch_requires_positive_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list(cp.run_batch(PIPELINE, _image_sets(), workers=workers))


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_batch_image_ids_round_trip_through_parquet(
    tmp_path: Path, backend: Literal["thread", "process"]
) -> None:
    fields = _image_sets()
    for index, run in enumerate(
        cp.run_batch(PIPELINE, fields, workers=2, backend=backend)
    ):
        assert run.measurements is not None
        run.measurements.to_parquet(tmp_path / f"{index}.parquet")
    data = pq.read_table(tmp_path, columns=["image_id", "scope", "object_id"])
    assert data.num_rows > 0
    assert set(data["image_id"].to_pylist()) == set(fields)


@pytest.mark.parametrize(
    ("workers", "backend"), [(1, "thread"), (2, "thread"), (2, "process")]
)
def test_batch_matches_atomic_pipeline_on_real_images(
    workers: int, backend: Literal["thread", "process"]
) -> None:
    pipeline = cp.load_pipeline(PIPELINE)
    image_sets = _image_sets()
    runs = list(cp.run_batch(pipeline, image_sets, workers=workers, backend=backend))
    assert [run.image_id for run in runs] == list(image_sets)
    for run in runs:
        arrays = {
            name: cp.load_array(path) for name, path in image_sets[run.image_id].items()
        }
        primary, derived, measurements = pipeline.run(arrays)
        assert list(run.primary) == list(primary)
        assert list(run.derived) == list(derived)
        for name, result in (primary | derived).items():
            actual = (run.primary | run.derived)[name]
            np.testing.assert_array_equal(actual.labels, result.labels, strict=True)
        assert measurements is not None and run.measurements is not None
        assert run.measurements.image_id == run.image_id
        assert measurements.image_id == "Image"
        assert run.measurements.relationships == measurements.relationships
        assert list(run.measurements.object_ids) == list(measurements.object_ids)
        for scope, ids in measurements.object_ids.items():
            np.testing.assert_array_equal(
                run.measurements.object_ids[scope], ids, strict=True
            )
        assert list(run.measurements.features) == list(measurements.features)
        for scope, features in measurements.features.items():
            assert list(run.measurements.features[scope]) == list(features)
            for name, value in features.items():
                np.testing.assert_array_equal(
                    run.measurements.features[scope][name], value, strict=True
                )


def test_process_batch_accepts_array_mappings_and_joins_workers_on_close() -> None:
    images = {
        image_id: MappingProxyType(
            {name: cp.load_array(path) for name, path in sources.items()}
        )
        for image_id, sources in _image_sets().items()
    }
    before = set(active_children())
    runs = cp.run_batch(PIPELINE, images, workers=2, backend="process")
    try:
        result = next(runs)
        assert result.image_id == next(iter(images))
        assert result.primary
        pool = set(active_children()) - before
        assert pool
    finally:
        runs.close()
    assert not any(child.is_alive() for child in pool)


def test_process_backend_with_one_worker_runs_in_caller() -> None:
    # A local subclass cannot be transported to a spawned process.
    class LocalPipeline(cp.Pipeline):
        pass

    fields = _image_sets()
    runs = cp.run_batch(LocalPipeline(), fields, workers=1, backend="process")
    assert [run.image_id for run in runs] == list(fields)


def test_batch_rejects_unknown_backend_even_in_serial_mode() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        list(cp.run_batch(PIPELINE, {}, backend="unknown"))  # ty: ignore[invalid-argument-type]
