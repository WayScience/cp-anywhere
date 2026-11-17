# ty: ignore[import-not-found]
"""Generate a real .cppipe pipeline chaining CorrectIlluminationCalculate
and CorrectIlluminationApply, plus a fresh oracle run of that exact
combined pipeline on a real test image -- not a hand-derived result."""

# ruff: noqa: E402

from pathlib import Path

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import json

import numpy as np
from cellprofiler.modules.correctilluminationapply import CorrectIlluminationApply
from cellprofiler.modules.correctilluminationcalculate import (
    CorrectIlluminationCalculate,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/illumination"


def _test_image(shape: tuple[int, int] = (18, 22)) -> np.ndarray:
    rng = np.random.default_rng(42)
    r, c = np.indices(shape)
    base = ((r * 11 + c * 7) % 29) / 32.0
    return np.clip(base + rng.normal(0, 0.03, shape), 0.0, 1.0)


def _build_module() -> tuple:
    calculate = CorrectIlluminationCalculate()
    calculate.image_name.value = "DNA"
    calculate.illumination_image_name.value = "Illum"
    calculate.intensity_choice.value = "Regular"
    calculate.each_or_all.value = "Each"
    calculate.smoothing_method.value = "Gaussian Filter"
    calculate.automatic_object_width.value = "Automatic"
    calculate.rescale_option.value = "Yes"
    calculate.set_module_num(1)

    apply_module = CorrectIlluminationApply()
    del apply_module.images[:]
    apply_module.add_image(can_delete=False)
    image = apply_module.images[0]
    image.image_name.value = "DNA"
    image.corrected_image_name.value = "DNACorrected"
    image.illum_correct_function_image_name.value = "Illum"
    image.divide_or_subtract.value = "Divide"
    apply_module.truncate_low.value = True
    apply_module.truncate_high.value = True
    apply_module.set_module_num(2)
    return calculate, apply_module


def _run_oracle(pixels: np.ndarray) -> np.ndarray:
    calculate, apply_module = _build_module()
    pipeline = Pipeline()
    pipeline.add_module(calculate)
    pipeline.add_module(apply_module)
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    image_set.add("DNA", Image(pixels.copy(), convert=False))
    object_set = ObjectSet()
    measurements = Measurements()
    workspace = Workspace(
        pipeline, calculate, image_set, object_set, measurements, image_sets
    )
    calculate.run(workspace)
    workspace = Workspace(
        pipeline, apply_module, image_set, object_set, measurements, image_sets
    )
    apply_module.run(workspace)
    return image_set.get_image("DNACorrected").pixel_data


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pixels = _test_image()
    calculate, apply_module = _build_module()
    pipeline = Pipeline()
    pipeline.add_module(calculate)
    pipeline.add_module(apply_module)
    with (OUT / "illumination_calculate_apply.cppipe").open("w") as stream:
        pipeline.dump(stream, save_image_plane_details=False)
    expected = _run_oracle(pixels)
    np.save(OUT / "illumination_calculate_apply.test_image.npy", pixels)
    (OUT / "illumination_calculate_apply.expected.json").write_text(
        json.dumps({"corrected": expected.tolist()}, indent=2) + "\n"
    )
    print("illumination_calculate_apply written")


if __name__ == "__main__":
    generate()
