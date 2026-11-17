# ty: ignore[import-not-found]
"""Oracle: real CellProfiler 4.2.8 CorrectIlluminationApply. Runs the
real module's own run_image() against a real Workspace/Image."""

# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.correctilluminationapply import CorrectIlluminationApply
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/illumination_apply"


def _image(shape: tuple[int, int], *, seed: int, low: float, high: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, shape)


def cases() -> list[dict[str, Any]]:
    result = []
    shape = (12, 15)
    for divide_or_subtract in ("Divide", "Subtract"):
        for truncate_low in (True, False):
            for truncate_high in (True, False):
                name = (
                    f"{divide_or_subtract.lower()}_"
                    f"tl{int(truncate_low)}_th{int(truncate_high)}"
                )
                result.append(
                    {
                        "name": name,
                        "image": _image(shape, seed=1, low=-0.3, high=1.4),
                        "illum": _image(shape, seed=2, low=0.2, high=2.0),
                        "divide_or_subtract": divide_or_subtract,
                        "truncate_low": truncate_low,
                        "truncate_high": truncate_high,
                    }
                )
    return result


def _run_case(case: dict[str, Any]) -> np.ndarray:
    module = CorrectIlluminationApply()
    del module.images[:]
    module.add_image(can_delete=False)
    image = module.images[0]
    image.image_name.value = "Input"
    image.corrected_image_name.value = "Corrected"
    image.illum_correct_function_image_name.value = "Illum"
    image.divide_or_subtract.value = case["divide_or_subtract"]
    module.truncate_low.value = case["truncate_low"]
    module.truncate_high.value = case["truncate_high"]
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    image_set.add("Input", Image(case["image"].copy(), convert=False))
    image_set.add("Illum", Image(case["illum"].copy(), convert=False))
    object_set = ObjectSet()
    measurements = Measurements()
    workspace = Workspace(
        pipeline, module, image_set, object_set, measurements, image_sets
    )
    module.run(workspace)
    return image_set.get_image("Corrected").pixel_data


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases():
        name = case["name"]
        output = _run_case(case)
        result = {
            "settings": {
                k: v for k, v in case.items() if k not in ("name", "image", "illum")
            },
            "image": case["image"].tolist(),
            "illum": case["illum"].tolist(),
            "expected": output.tolist(),
        }
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest.append(name)
        print(name)
    (OUT / "manifest.json").write_text(json.dumps({"cases": manifest}, indent=2) + "\n")
    print(f"wrote {len(manifest)} cases")


if __name__ == "__main__":
    generate()
