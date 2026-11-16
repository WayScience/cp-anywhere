# ty: ignore[import-not-found]
"""Exercise size/shape settings in a real three-compartment pipeline."""

# ruff: noqa: E402

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/size_shape"
INPUTS = ROOT / "fixtures/cppipe/secondary"
PREFIX = "identify_secondary_objects_propagation"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for case in ("shrink", "no_shrink"):
        pipeline = Pipeline()
        pipeline.load(str(ROOT / f"fixtures/cppipe/tertiary/{case}.cppipe"))
        for object_name, zernikes, advanced in (
            ("Nuclei", True, False),
            ("Cells", False, False),
            ("Cytoplasm", True, True),
        ):
            module = MeasureObjectSizeShape()
            module.set_module_num(len(pipeline.modules()) + 1)
            module.objects_list.value = object_name
            module.calculate_zernikes.value = zernikes
            module.calculate_advanced.value = advanced
            pipeline.add_module(module)
        with (OUT / f"{case}.cppipe").open("w") as stream:
            pipeline.dump(stream, save_image_plane_details=False)
        image_sets = ImageSetList()
        image_set = image_sets.get_image_set(0)
        for name, suffix in (("DNA", "dna"), ("Stain", "stain")):
            pixels = np.load(INPUTS / f"{PREFIX}.{suffix}.npy")
            image_set.add(name, Image(pixels, convert=True, dimensions=2))
        objects = ObjectSet()
        measurements = Measurements()
        for module in pipeline.modules():
            module.run(
                Workspace(
                    pipeline,
                    module,
                    image_set,
                    objects,
                    measurements,
                    image_sets,
                )
            )
        expected = {"reference": "CellProfiler 4.2.8", "features": {}}
        for name in ("Nuclei", "Cells", "Cytoplasm"):
            expected["features"][name] = {
                feature: measurements.get_current_measurement(name, feature).tolist()
                for feature in measurements.get_feature_names(name)
                if feature.startswith(("AreaShape_", "Intensity_"))
            }
        (OUT / f"{case}.json").write_text(json.dumps(expected, indent=2) + "\n")
        print(
            case,
            {name: len(features) for name, features in expected["features"].items()},
        )
        measurements.close()


if __name__ == "__main__":
    generate()
