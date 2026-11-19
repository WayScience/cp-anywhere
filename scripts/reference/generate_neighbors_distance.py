# ty: ignore[import-not-found]
"""Capture fixed-distance neighbors from the pinned CellProfiler 4.2.8 source."""

# ruff: noqa: E402
from __future__ import annotations

import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Union

import cellprofiler
import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.measureobjectneighbors import (
    D_ADJACENT,
    D_WITHIN,
    MeasureObjectNeighbors,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/neighbors_distance"
Labels = Union[list[list[int]], np.ndarray]


def _neighbors(
    scope: str, distance: int | None, excluded: bool
) -> MeasureObjectNeighbors:
    module = MeasureObjectNeighbors()
    module.object_name.value = scope
    module.neighbors_name.value = scope
    module.distance_method.value = D_ADJACENT if distance is None else D_WITHIN
    module.distance.value = 5 if distance is None else distance
    module.wants_excluded_objects.value = excluded
    assert module.distance.value == (5 if distance is None else distance)
    return module


def _single(
    name: str,
    labels: Labels,
    distance: int,
    full: Labels | None = None,
    *,
    excluded: bool = False,
) -> dict:
    labels = np.array(labels, dtype=np.int32)
    if full is not None:
        full = np.array(full, dtype=np.int32)
    objects = Objects()
    objects.segmented = labels.copy()
    if full is not None:
        objects.small_removed_segmented = full.copy()
    return {
        "name": name,
        "distance": distance,
        "labels": labels.tolist(),
        "full_labels": None if full is None else full.tolist(),
        "excluded": excluded,
        **_run(
            [_neighbors("Cells", distance, excluded)],
            np.zeros(labels.shape),
            {"Cells": objects},
        ),
    }


def _run(
    modules: list[MeasureObjectNeighbors | IdentifyPrimaryObjects],
    pixels: np.ndarray,
    inputs: dict[str, Objects],
    pipeline_file: Path | None = None,
) -> dict:
    pipeline = Pipeline()
    for num, module in enumerate(modules, 1):
        module.set_module_num(num)
        pipeline.add_module(module)
    if pipeline_file is not None:
        with pipeline_file.open("w") as fp:
            pipeline.dump(fp, save_image_plane_details=False)
    images = ImageSetList()
    image_set = images.get_image_set(0)
    image_set.add("DNA", Image(pixels, dimensions=2))
    objects = ObjectSet()
    for name, value in inputs.items():
        objects.add_objects(value, name)
    measurements = Measurements()
    measurements.group_number = 1
    measurements.group_index = 1
    try:
        outputs = {}
        features = {}
        object_ids = {}
        for module in modules:
            with np.errstate(invalid="ignore", divide="ignore"):
                module.run(
                    Workspace(
                        pipeline, module, image_set, objects, measurements, images
                    )
                )
            if isinstance(module, IdentifyPrimaryObjects):
                output = objects.get_objects(module.y_name.value)
                outputs[module.y_name.value] = {
                    "labels": output.segmented.copy().tolist(),
                    "full_labels": output.small_removed_segmented.copy().tolist(),
                }
            else:
                scope = module.object_name.value
                object_ids[scope] = objects.get_objects(scope).indices.tolist()
                columns = features.setdefault(scope, {})
                for feature in module.all_features:
                    key = module.get_measurement_name(feature)
                    values = measurements.get_current_measurement(scope, key)
                    columns[key] = [
                        int(v)
                        if feature
                        in (
                            "NumberOfNeighbors",
                            "FirstClosestObjectNumber",
                            "SecondClosestObjectNumber",
                        )
                        else _number(v)
                        for v in np.asarray(values)
                    ]
        edges = []
        for group in sorted(
            measurements.get_relationship_groups(), key=lambda g: g.module_number
        ):
            rows = measurements.get_relationships(
                group.module_number,
                group.relationship,
                group.object_name1,
                group.object_name2,
            )
            for row in rows:
                edges.append(
                    [
                        group.object_name1,
                        int(row["ObjectNumber_First"]),
                        int(row["ObjectNumber_Second"]),
                    ]
                )
                assert row["ImageNumber_First"] == row["ImageNumber_Second"] == 1
        return {
            "features": features,
            "object_ids": object_ids,
            "relationships": edges,
            "outputs": outputs,
        }
    finally:
        measurements.close()


def _number(value: np.floating) -> float | dict[str, str]:
    if np.isfinite(value):
        return float(value)
    return {
        "nonfinite": "NaN"
        if np.isnan(value)
        else "+Infinity"
        if value > 0
        else "-Infinity"
    }


def _primary(scope: str) -> IdentifyPrimaryObjects:
    primary = IdentifyPrimaryObjects()
    primary.x_name.value = "DNA"
    primary.y_name.value = scope
    primary.use_advanced.value = True
    primary.size_range.value = (2, 7)
    primary.exclude_size.value = True
    primary.exclude_border_objects.value = False
    primary.unclump_method.value = "None"
    primary.watershed_method.value = "None"
    primary.fill_holes.value = "Never"
    primary.threshold.threshold_scope.value = "Global"
    primary.threshold.global_operation.value = "Manual"
    primary.threshold.manual_threshold.value = 0.5
    primary.threshold.threshold_smoothing_scale.value = 0
    return primary


def _chain() -> dict:
    pixels = np.zeros((15, 23), dtype=np.uint16)
    pixels[2:12, 2:10] = 50000  # Removed by the primary size filter.
    pixels[4:6, 11:13] = pixels[4:6, 17:19] = pixels[9:11, 14:16] = 55000
    modules = [
        _primary("Cells"),
        _primary("Nuclei"),
        _neighbors("Cells", 5, True),
        _neighbors("Nuclei", 1, True),
        _neighbors("Cells", 1, False),
        _neighbors("Cells", None, True),
        _neighbors("Cells", 6, True),
    ]
    return {
        "pixels": pixels.tolist(),
        **_run(modules, pixels, {}, OUT / "primary_mixed.cppipe"),
    }


def generate() -> None:
    """Run the pinned source checkout and write the oracle fixtures."""
    lock = json.loads((ROOT / "reference/source-lock.json").read_text())
    assert cellprofiler.__file__ is not None
    source_dir = Path(cellprofiler.__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == lock["cellprofiler"]["commit"]
    assert cellprofiler.__version__ == version("CellProfiler-core") == "4.2.8"
    assert version("centrosome") == "1.2.3"
    assert np.__version__ == "1.23.5"
    assert version("scipy") == "1.9.3"
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        _single("diagonal_contact", [[1, 0], [0, 2]], 1),
        _single(
            "disk_and_touching_radii",
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0],
            ],
            4,
        ),
        _single(
            "fft_branch",
            [
                [1, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 3, 0],
            ],
            6,
        ),
        _single("empty", [[0, 0], [0, 0]], 5),
        _single(
            "only_object_excluded", [[0, 0], [0, 0]], 5, [[1, 0], [0, 0]], excluded=True
        ),
        _single("one_object", [[0, 1], [0, 0]], 5),
        _single("sparse", [[0, 1, 0, 0, 0], [0, 0, 0, 3, 0]], 5),
        _single("sparse_one_object", [[0, 0, 3], [0, 0, 0]], 5),
        _single("equal_centers", [[1, 1, 1], [1, 2, 1], [1, 1, 1], [3, 0, 0]], 1),
        _single("partition_tie", [[0, 1, 0], [2, 3, 4], [0, 5, 0]], 1),
    ]
    full = [[5, 0, 1, 0, 0, 2, 0, 3], [5, 0, 1, 0, 0, 2, 0, 3]]
    kept = [[0, 0, 3, 0, 0, 1, 0, 2], [0, 0, 3, 0, 0, 1, 0, 2]]
    cases += [
        _single("renumbered_excluded", kept, 5, full, excluded=True),
        _single("renumbered_not_excluded", kept, 5, full),
    ]
    for name, payload in (("cases", cases), ("primary_mixed", _chain())):
        (OUT / f"{name}.json").write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    generate()
