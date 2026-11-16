# ty: ignore[import-not-found]
"""Capture size/shape measurements with real CellProfiler, including settings."""

# ruff: noqa: E402

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/size_shape"


def cases() -> dict[str, np.ndarray]:
    rectangle = np.zeros((12, 14), dtype=np.uint32)
    rectangle[2:8, 3:11] = 1
    ring = rectangle.copy()
    ring[4:6, 5:9] = 0
    diagonal = np.zeros_like(rectangle)
    for i in range(2, 9):
        diagonal[i, i : i + 2] = 1
    nested = ring.copy()
    nested[4:6, 6:8] = 2
    multiple = np.zeros_like(rectangle)
    multiple[1:4, 1:5] = 1
    multiple[7:11, 8:13] = 2
    border = np.zeros_like(rectangle)
    border[:5, :4] = 1
    point = np.zeros_like(rectangle)
    point[5, 5] = 1
    return {
        "rectangle": rectangle,
        "ring": ring,
        "diagonal": diagonal,
        "nested": nested,
        "multiple": multiple,
        "border": border,
        "point": point,
        "empty": np.zeros_like(rectangle),
    }


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, labels in cases().items():
        module = MeasureObjectSizeShape()
        module.objects_list.value = "Cells"
        module.calculate_advanced.value = True
        module.set_module_num(1)
        pipeline = Pipeline()
        pipeline.add_module(module)
        with (OUT / f"{name}.cppipe").open("w") as stream:
            pipeline.dump(stream, save_image_plane_details=False)
        objects = Objects()
        objects.segmented = labels
        object_set = ObjectSet()
        object_set.add_objects(objects, "Cells")
        image_sets = ImageSetList()
        measurements = Measurements()
        module.run(
            Workspace(
                pipeline,
                module,
                image_sets.get_image_set(0),
                object_set,
                measurements,
                image_sets,
            )
        )
        features = {
            feature: measurements.get_current_measurement("Cells", feature).tolist()
            for feature in measurements.get_feature_names("Cells")
        }
        result = {
            "reference": "CellProfiler 4.2.8",
            "settings": {"calculate_advanced": True, "calculate_zernikes": True},
            "labels": labels.tolist(),
            "features": features,
        }
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {len(features)} features")
        measurements.close()


if __name__ == "__main__":
    generate()
