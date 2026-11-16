# ty: ignore[import-not-found]
"""Exercise MeasureObjectIntensityDistribution settings in a real
multi-module pipeline."""

# ruff: noqa: E402

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measureobjectintensitydistribution import (
    MeasureObjectIntensityDistribution,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/intensity_distribution"
INPUTS = ROOT / "fixtures/cppipe/secondary"
PREFIX = "identify_secondary_objects_propagation"


def generate() -> None:
    """Chain a real segmentation pipeline with MeasureObjectIntensityDistribution."""
    OUT.mkdir(parents=True, exist_ok=True)
    for case in ("shrink", "no_shrink"):
        pipeline = Pipeline()
        pipeline.load(str(ROOT / f"fixtures/cppipe/tertiary/{case}.cppipe"))
        module = MeasureObjectIntensityDistribution()
        module.set_module_num(len(pipeline.modules()) + 1)
        module.images_list.value = "Stain"
        # This port only supports one object group and one bin-count group
        # (the default center choice, "These objects"); real CellProfiler's
        # settings add more groups via add_object()/add_bin_count(), which
        # is an explicit, documented translation gap -- see
        # _translate_measure_object_intensity_distribution's field-count
        # check in src/cp_anywhere/cppipe.py.
        module.objects[0].object_name.value = "Cells"
        module.bin_counts[0].wants_scaled.value = True
        module.bin_counts[0].bin_count.value = 3
        module.bin_counts[0].maximum_radius.value = 100
        module.wants_zernikes.value = "Magnitudes only"
        module.zernike_degree.value = 4
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
        for pipeline_module in pipeline.modules():
            pipeline_module.run(
                Workspace(
                    pipeline,
                    pipeline_module,
                    image_set,
                    objects,
                    measurements,
                    image_sets,
                )
            )
        expected = {"reference": "CellProfiler 4.2.8", "features": {}}
        for name in ("Cells",):
            expected["features"][name] = {
                feature: measurements.get_current_measurement(name, feature).tolist()
                for feature in measurements.get_feature_names(name)
                if feature.startswith("RadialDistribution_")
            }
        (OUT / f"{case}.json").write_text(json.dumps(expected, indent=2) + "\n")
        print(
            case,
            {name: len(features) for name, features in expected["features"].items()},
        )
        measurements.close()


if __name__ == "__main__":
    generate()
