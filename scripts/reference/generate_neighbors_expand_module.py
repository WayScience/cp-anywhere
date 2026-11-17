# ty: ignore[import-not-found]
"""Oracle: real CellProfiler 4.2.8 MeasureObjectNeighbors, distance
method "Expand until adjacent" (D_EXPAND), self-neighbors only
(matching this project's existing wants_excluded_objects-only scope;
cross-object-set neighbors remain an explicit, separate gap).
"""

# ruff: noqa: E402

import json
from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.measureobjectneighbors import (
    D_EXPAND,
    MeasureObjectNeighbors,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/neighbors_expand"

_CASES = {
    # Columns 0-1 belong to object 1, columns 2-3 to object 2: not
    # Adjacent (two columns apart), but their Voronoi territories
    # share the whole vertical boundary once expanded.
    "two_far_columns": np.array(
        [
            [1, 0, 0, 2],
            [1, 0, 0, 2],
        ],
        dtype=np.uint32,
    ),
    "three_objects_uneven": np.array(
        [
            [1, 1, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "single_object": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "already_touching": np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint32,
    ),
}


def _run(labels: np.ndarray) -> dict:
    pipeline = Pipeline()
    module = MeasureObjectNeighbors()
    module.set_module_num(1)
    module.object_name.value = "Cells"
    module.neighbors_name.value = "Cells"
    module.distance_method.value = D_EXPAND
    module.wants_excluded_objects.value = False
    pipeline.add_module(module)

    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("DNA", Image(np.zeros(labels.shape), convert=False, dimensions=2))
    object_set = ObjectSet()
    objects = Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "Cells")
    measurements = Measurements()
    measurements.group_number = 1
    measurements.group_index = 1
    module.run(
        Workspace(pipeline, module, image_set, object_set, measurements, image_set_list)
    )
    features = {}
    for feature in module.all_features:
        name = module.get_measurement_name(feature)
        value = measurements.get_current_measurement("Cells", name)
        features[name] = np.asarray(value).tolist()
    relationships = measurements.get_relationship_groups()
    edges = []
    for group in relationships:
        rel = measurements.get_relationships(
            group.module_number,
            group.relationship,
            group.object_name1,
            group.object_name2,
        )
        edges = sorted(
            zip(
                rel["ObjectNumber_First"].tolist(),
                rel["ObjectNumber_Second"].tolist(),
            )
        )
    return {"features": features, "relationships": edges}


def generate() -> None:
    """Write each case's real MeasureObjectNeighbors D_EXPAND output."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, labels in _CASES.items():
        result = _run(labels)
        payload = {
            "reference": "CellProfiler 4.2.8 MeasureObjectNeighbors D_EXPAND",
            "labels": labels.tolist(),
            **result,
        }
        (OUT / f"module_{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(name, result["features"].keys())


if __name__ == "__main__":
    generate()
