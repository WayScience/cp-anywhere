# ty: ignore[import-not-found]
"""Capture secondary-object regression cases with the real reference modules.

Run in the isolated CellProfiler 4.2.8 environment, not the project venv.
Both final labels and intermediate layers are reference outputs.
"""

# ruff: noqa: ANN001, ANN201, E402

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.identifysecondaryobjects import IdentifySecondaryObjects
from cellprofiler.modules.measureobjectneighbors import MeasureObjectNeighbors
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/cppipe/secondary_regressions"
METHODS = {"distance_n": "Distance - N", "propagation": "Propagation"}


def secondary(method, radius=30, holes=False):
    module = IdentifySecondaryObjects()
    module.x_name.value = "Nuclei"
    module.y_name.value = "Cells"
    module.image_name.value = "Stain"
    module.method.value = METHODS[method]
    module.distance_to_dilate.value = radius
    module.fill_holes.value = holes
    module.threshold.global_operation.value = "Manual"
    module.threshold.manual_threshold.value = 0.5
    module.threshold.threshold_smoothing_scale.value = 0
    return module


def layers(objects):
    return {
        name: getattr(objects, source).astype(int).tolist()
        for name, source in (
            ("labels", "segmented"),
            ("unedited_labels", "unedited_segmented"),
            ("small_removed_labels", "small_removed_segmented"),
        )
    }


def execute(pipeline, images, objects):
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    for name, array in images.items():
        image_set.add(name, Image(array, convert=False, dimensions=2))
    measurements = Measurements()
    for module in pipeline.modules():
        module.run(
            Workspace(pipeline, module, image_set, objects, measurements, image_sets)
        )
    return measurements


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(name)


def direct_cases():
    kept = np.zeros((9, 9), dtype=np.uint32)
    kept[4, 1] = 1
    interior = kept.copy()
    interior[4, 5] = 2
    renumbered = np.zeros_like(kept)
    renumbered[4, 5] = 1
    old_ids = np.zeros_like(kept)
    old_ids[4, 5] = 2
    ring = np.zeros_like(kept)
    ring[2:7, 2] = ring[2:7, 6] = 1
    ring[2, 2:7] = ring[6, 2:7] = 1
    ring[4, 4] = 2
    for method in METHODS:
        for name, primary, unedited, radius, holes in (
            ("discarded_interior", kept, interior, 30, False),
            ("renumbered", renumbered, old_ids, 30, False),
            ("nested", ring, ring, 1, True),
        ):
            module = secondary(method, radius, holes)
            module.set_module_num(1)
            pipeline = Pipeline()
            pipeline.add_module(module)
            objects = ObjectSet()
            nuclei = Objects()
            nuclei.segmented = primary
            nuclei.unedited_segmented = unedited
            objects.add_objects(nuclei, "Nuclei")
            image = np.ones(primary.shape)
            measurements = execute(pipeline, {"Stain": image}, objects)
            save(
                f"direct_{name}_{method}",
                {
                    "method": method,
                    "radius": radius,
                    "fill_holes": holes,
                    "primary": primary.tolist(),
                    "unedited": unedited.tolist(),
                    "expected": layers(objects.get_objects("Cells")),
                },
            )
            measurements.close()


def pipeline_cases():
    for method in METHODS:
        for name in ("border", "bottom_border", "oversized", "renumbered_neighbors"):
            shape = (9, 15) if name == "border" else (13, 21)
            dna = np.zeros(shape)
            if name == "border":
                dna[:3, 1:4] = dna[4:7, 10:13] = 1
            elif name == "bottom_border":
                dna[2:5, 2:5] = dna[10:, 15:18] = 1
            elif name == "oversized":
                dna[2:5, 2:5] = dna[6:11, 12:17] = 1
            else:
                dna[:3, 1:4] = dna[4:7, 7:10] = dna[8:11, 15:18] = 1
            primary = IdentifyPrimaryObjects()
            primary.x_name.value = "DNA"
            primary.y_name.value = "Nuclei"
            primary.use_advanced.value = True
            primary.exclude_size.value = name == "oversized"
            primary.size_range.value = (0, 4)
            primary.exclude_border_objects.value = name != "oversized"
            primary.unclump_method.value = "None"
            primary.watershed_method.value = "None"
            primary.fill_holes.value = "Never"
            primary.threshold.global_operation.value = "Manual"
            primary.threshold.manual_threshold.value = 0.5
            primary.threshold.threshold_smoothing_scale.value = 0
            pipeline = Pipeline()
            primary.set_module_num(1)
            pipeline.add_module(primary)
            module = secondary(method)
            module.set_module_num(2)
            pipeline.add_module(module)
            neighbors = MeasureObjectNeighbors()
            neighbors.object_name.value = neighbors.neighbors_name.value = "Cells"
            neighbors.distance_method.value = "Adjacent"
            neighbors.set_module_num(3)
            pipeline.add_module(neighbors)
            objects = ObjectSet()
            measurements = execute(
                pipeline, {"DNA": dna, "Stain": np.ones(shape)}, objects
            )
            filename = f"pipeline_{name}_{method}"
            save(
                filename,
                {
                    "images": {"DNA": dna.tolist(), "Stain": np.ones(shape).tolist()},
                    "Nuclei": layers(objects.get_objects("Nuclei")),
                    "Cells": layers(objects.get_objects("Cells")),
                    "neighbors": {
                        feature: measurements.get_current_measurement(
                            "Cells", feature
                        ).tolist()
                        for feature in measurements.get_feature_names("Cells")
                        if feature.startswith("Neighbors_")
                    },
                },
            )
            with (OUT / f"{filename}.cppipe").open("w") as stream:
                pipeline.dump(stream, save_image_plane_details=False)
            measurements.close()


if __name__ == "__main__":
    direct_cases()
    pipeline_cases()
