# ty: ignore[import-not-found]
"""Generate tertiary labels and intensity values with real CellProfiler 4.2.8.

Run this script in the isolated reference environment. The secondary
fixture supplies a real saved pipeline and its input images.
"""

# ruff: noqa: ANN001, ANN201, E402

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifytertiaryobjects import IdentifyTertiaryObjects
from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity
from cellprofiler_core.image import Image, ImageSetList, VanillaImage
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fixtures/cppipe/secondary"
OUT = ROOT / "fixtures/cppipe/tertiary"
PREFIX = "identify_secondary_objects_propagation"


def generate(shrink):
    pipeline = Pipeline()
    pipeline.load(str(SOURCE / f"{PREFIX}.cppipe"))
    assert [m.module_name for m in pipeline.modules()] == [
        "IdentifyPrimaryObjects",
        "IdentifySecondaryObjects",
    ]
    tertiary = IdentifyTertiaryObjects()
    tertiary.set_module_num(3)
    tertiary.secondary_objects_name.value = "Cells"
    tertiary.primary_objects_name.value = "Nuclei"
    tertiary.subregion_objects_name.value = "Cytoplasm"
    tertiary.shrink_primary.value = shrink
    pipeline.add_module(tertiary)
    intensity = MeasureObjectIntensity()
    intensity.set_module_num(4)
    intensity.images_list.value = "Stain"
    intensity.objects_list.value = "Cytoplasm"
    pipeline.add_module(intensity)
    name = "shrink" if shrink else "no_shrink"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.cppipe").open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)

    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    for image_name, suffix in (("DNA", "dna"), ("Stain", "stain")):
        pixels = np.load(SOURCE / f"{PREFIX}.{suffix}.npy")
        image_set.providers.append(
            VanillaImage(image_name, Image(pixels, convert=True, dimensions=2))
        )
    objects = ObjectSet()
    measurements = Measurements()
    for module in pipeline.modules():
        module.run(
            Workspace(pipeline, module, image_set, objects, measurements, image_sets)
        )
    expected = {
        "reference": "CellProfiler 4.2.8",
        "shrink_primary": shrink,
        "labels": {
            key: objects.get_objects(key).segmented.astype(int).tolist()
            for key in ("Nuclei", "Cells", "Cytoplasm")
        },
        "mean_intensity": measurements.get_current_measurement(
            "Cytoplasm", "Intensity_MeanIntensity_Stain"
        ).tolist(),
    }
    assert np.any(objects.get_objects("Cytoplasm").segmented)
    (OUT / f"{name}.expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(json.dumps({"case": name, "mean_intensity": expected["mean_intensity"]}))


if __name__ == "__main__":
    for shrink_primary in (True, False):
        generate(shrink_primary)
