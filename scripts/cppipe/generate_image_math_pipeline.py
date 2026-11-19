# ty: ignore[import-not-found]
"""Generate an ordered ImageMath fixture with CellProfiler 4.2.8."""

# ruff: noqa: E402
import json
from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.correctilluminationapply import CorrectIlluminationApply
from cellprofiler.modules.correctilluminationcalculate import (
    CorrectIlluminationCalculate,
)
from cellprofiler.modules.imagemath import ImageMath
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/cppipe/illumination"


def _apply(source: str, flat: str, target: str) -> CorrectIlluminationApply:
    module = CorrectIlluminationApply()
    module.images[0].image_name.value = source
    module.images[0].illum_correct_function_image_name.value = flat
    module.images[0].corrected_image_name.value = target
    return module


def generate() -> None:
    """Replay apply → math → calculate → apply in the original order."""
    math = ImageMath()
    math.operation.value = "None"
    math.images[0].image_name.value = "PreDNA"
    math.images[0].factor.value = 1.5
    math.output_image_name.value = "DNA"
    math.after_factor.value = 0.5
    math.addend.value = -0.125
    math.truncate_low.value = True
    math.truncate_high.value = True
    math.replace_nan.value = True
    calc = CorrectIlluminationCalculate()
    calc.image_name.value = "DNA"
    calc.illumination_image_name.value = "Flat2"
    calc.each_or_all.value = "Each"
    calc.smoothing_method.value = "No smoothing"
    calc.rescale_option.value = "No"
    pipeline = Pipeline()
    modules = (
        _apply("Raw", "Flat", "PreDNA"),
        math,
        calc,
        _apply("DNA", "Flat2", "Final"),
    )
    for i, module in enumerate(modules, 1):
        module.set_module_num(i)
        pipeline.add_module(module)
    inputs = {
        "Raw": np.array([[0.0, 0.25, 0.5], [0.75, 1.0, np.nan]], dtype=np.float32),
        "Flat": np.full((2, 3), 0.5, dtype=np.float32),
    }
    lists = ImageSetList()
    images = lists.get_image_set(0)
    for name, pixels in inputs.items():
        images.add(name, Image(pixels, convert=True))
    objects = ObjectSet()
    measurements = Measurements()
    try:
        for module in modules:
            with np.errstate(invalid="ignore", divide="ignore"):
                module.run(
                    Workspace(pipeline, module, images, objects, measurements, lists)
                )
        result = {
            name: images.get_image(name).pixel_data.tolist()
            for name in ("DNA", "Flat2")
        }
    finally:
        measurements.close()
    with (OUT / "image_math_order.cppipe").open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)
    (OUT / "image_math_order.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    generate()
