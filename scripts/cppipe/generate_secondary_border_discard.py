# ty: ignore[import-not-found]
"""Record secondary border-discard fixtures in the CellProfiler 4.2.8 environment."""

# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np
from cellprofiler import __version__

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.identifysecondaryobjects import IdentifySecondaryObjects
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from generate_secondary_regressions import layers

OUT = Path(__file__).resolve().parents[2] / "fixtures/cppipe/secondary_border_discard"
METHODS = {"propagation": "Propagation", "watershed_image": "Watershed - Image"}


def _manual_threshold(
    module: IdentifyPrimaryObjects | IdentifySecondaryObjects,
) -> None:
    module.threshold.global_operation.value = "Manual"
    module.threshold.manual_threshold.value = 0.5
    module.threshold.threshold_smoothing_scale.value = 0


def _secondary(method: str, *, fill_holes: bool = False) -> IdentifySecondaryObjects:
    module = IdentifySecondaryObjects()
    module.x_name.value = "Nuclei"
    module.y_name.value = "Cells"
    module.image_name.value = "Stain"
    module.method.value = METHODS[method]
    module.wants_discard_edge.value = True
    module.wants_discard_primary.value = False
    module.fill_holes.value = fill_holes
    _manual_threshold(module)
    assert module.wants_discard_edge and not module.wants_discard_primary
    return module


def _run(
    modules: list[IdentifyPrimaryObjects | IdentifySecondaryObjects],
    images: dict[str, Image],
    objects: ObjectSet,
) -> Pipeline:
    pipeline = Pipeline()
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    for name, image in images.items():
        image_set.add(name, image)
    measurements = Measurements()
    try:
        for index, module in enumerate(modules, 1):
            module.set_module_num(index)
            pipeline.add_module(module)
            module.run(
                Workspace(
                    pipeline, module, image_set, objects, measurements, image_sets
                )
            )
    finally:
        measurements.close()
    return pipeline


def _save(name: str, data: dict[str, object]) -> None:
    # Keep the small input/output planes visible side by side on separate lines.
    rows = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in data.items()]
    (OUT / f"{name}.json").write_text("{\n" + ",\n".join(rows) + "\n}\n")


def _pipeline_case(method: str) -> None:
    dna = np.zeros((7, 12), dtype=np.float32)
    dna[1, 2] = dna[3, 8] = dna[5, 5] = 1
    stain = dna.copy()
    stain[:3, 1:4] = stain[2:5, 8:11] = stain[4:6, 4:7] = 1
    stain[3, 9] = 0  # Hole in the first surviving secondary object.
    primary = IdentifyPrimaryObjects()
    primary.x_name.value = "DNA"
    primary.y_name.value = "Nuclei"
    primary.use_advanced.value = True
    primary.exclude_size.value = primary.exclude_border_objects.value = False
    primary.unclump_method.value = primary.watershed_method.value = "None"
    primary.fill_holes.value = "Never"
    _manual_threshold(primary)
    objects = ObjectSet()
    images = {"DNA": dna, "Stain": stain}
    pipeline = _run(
        [primary, _secondary(method, fill_holes=True)],
        {name: Image(pixels) for name, pixels in images.items()},
        objects,
    )
    nuclei = layers(objects.get_objects("Nuclei"))
    cells = layers(objects.get_objects("Cells"))
    assert np.unique(nuclei["labels"]).tolist() == [0, 1, 2, 3]
    assert np.unique(cells["small_removed_labels"]).tolist() == [0, 1, 2, 3]
    assert np.unique(cells["labels"]).tolist() == [0, 1, 2]
    final = np.asarray(cells["labels"])
    np.testing.assert_array_equal(final[[1, 3, 3, 5], [2, 8, 9, 5]], [0, 1, 1, 2])
    _save(
        method,
        {
            "images": {key: value.tolist() for key, value in images.items()},
            "Nuclei": nuclei,
            "Cells": cells,
        },
    )
    with (OUT / f"{method}.cppipe").open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)


def _masked_case(method: str) -> None:
    mask = np.ones((7, 10), dtype=bool)
    mask[2, 2] = False
    primary = np.zeros(mask.shape, dtype=np.int32)
    primary[2, 3] = 9  # Cardinal contact with an interior mask hole.
    primary[3, 1] = 7  # Diagonal-only contact with that hole.
    primary[5, 7] = 3
    primary[6, 8] = 5  # Image edge is also a masked-image border.
    stain = (primary > 0).astype(np.float32)
    nuclei = Objects()
    nuclei.segmented = primary
    objects = ObjectSet()
    objects.add_objects(nuclei, "Nuclei")
    _run([_secondary(method)], {"Stain": Image(stain, mask=mask)}, objects)
    cells = layers(objects.get_objects("Cells"))
    final = np.asarray(cells["labels"])
    np.testing.assert_array_equal(final[[2, 6, 3, 5], [3, 8, 1, 7]], [0, 0, 2, 1])
    _save(
        f"masked_{method}",
        {"primary": primary.tolist(), "mask": mask.tolist(), "expected": cells},
    )


if __name__ == "__main__":
    assert __version__ == "4.2.8"
    OUT.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        _pipeline_case(method)
        _masked_case(method)
