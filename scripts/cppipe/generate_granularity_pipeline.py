# ty: ignore[import-not-found]
"""Exercise MeasureGranularity settings in a real multi-module pipeline."""

# ruff: noqa: E402

import json
from pathlib import Path
from typing import Any

import cellprofiler_core.preferences
import numpy as np
import skimage.morphology

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measuregranularity import MeasureGranularity
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/granularity"
INPUTS = ROOT / "fixtures/cppipe/secondary"
PREFIX = "identify_secondary_objects_propagation"


def _install_selem_compatibility_adapter() -> None:
    """Rename only selem->footprint; forward to the installed skimage ops.

    Real CellProfiler 4.2.8's MeasureGranularity source still passes the
    removed ``selem`` keyword to ``skimage.morphology.erosion``/
    ``dilation``; the pinned reference env's scikit-image 0.21.0 only
    accepts ``footprint``. See docs/reference.md's dependency-conflict
    note -- this is a real, confirmed incompatibility, not guessed.
    """
    import functools

    for name in ("erosion", "dilation", "reconstruction"):
        original: Any = getattr(skimage.morphology, name)

        @functools.wraps(original)
        def compatible(
            *args: object, _original: Any = original, **kwargs: object
        ) -> object:
            if "selem" in kwargs:
                kwargs["footprint"] = kwargs.pop("selem")
            return _original(*args, **kwargs)

        setattr(skimage.morphology, name, compatible)


def generate() -> None:
    """Chain a real segmentation pipeline with MeasureGranularity and dump it."""
    _install_selem_compatibility_adapter()
    OUT.mkdir(parents=True, exist_ok=True)
    for case in ("shrink", "no_shrink"):
        pipeline = Pipeline()
        pipeline.load(str(ROOT / f"fixtures/cppipe/tertiary/{case}.cppipe"))
        module = MeasureGranularity()
        module.set_module_num(len(pipeline.modules()) + 1)
        module.images_list.value = "Stain"
        module.wants_objects.value = True
        module.objects_list.value = "Nuclei, Cells"
        module.subsample_size.value = 0.5
        module.image_sample_size.value = 0.5
        module.element_size.value = 4
        module.granular_spectrum_length.value = 6
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
        for name in ("Image", "Nuclei", "Cells"):
            expected["features"][name] = {
                feature: (
                    float(measurements.get_current_image_measurement(feature))
                    if name == "Image"
                    else measurements.get_current_measurement(name, feature).tolist()
                )
                for feature in measurements.get_feature_names(name)
                if feature.startswith("Granularity_")
            }
        (OUT / f"{case}.json").write_text(json.dumps(expected, indent=2) + "\n")
        print(
            case,
            {name: len(features) for name, features in expected["features"].items()},
        )
        measurements.close()


if __name__ == "__main__":
    generate()
