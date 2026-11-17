# ty: ignore[import-not-found]
"""Oracle: real CellProfiler 4.2.8 CorrectIlluminationCalculate, "Each"
mode, the five non-exotic smoothing methods (everything except Convex
Hull and Splines, which are separate, much harder follow-on primitives
-- see spec/illumination.md). Runs the real module's own run() against
a real Workspace/Image, not a hand-derived formula.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import cellprofiler.modules.correctilluminationcalculate as _correctillumination_module
import numpy as np
from cellprofiler.modules.correctilluminationcalculate import (
    CorrectIlluminationCalculate,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/illumination_calculate"
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"


def _pixels(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r, c = np.indices(shape)
    base = ((r * 17 + c * 13 + r * c * 7) % 41) / 45.0
    blob = (
        (r - shape[0] / 2) ** 2 + (c - shape[1] / 2) ** 2 < (min(shape) / 3) ** 2
    ) * 0.4
    noise = rng.normal(0, 0.02, shape)
    return np.clip(base + blob + noise, 0.0, 1.0)


def _mask(shape: tuple[int, int], *, kind: str) -> np.ndarray | None:
    if kind == "none":
        return None
    r, c = np.indices(shape)
    if kind == "partial":
        return (r + c * 2) % 5 != 0
    raise ValueError(kind)


def cases() -> list[dict[str, Any]]:
    """Every case: pixels/mask + every CorrectIlluminationCalculate
    setting this generator exercises, as explicit named values."""
    result = []
    shapes = {"small": (24, 28), "tall": (31, 19)}
    for shape_name, shape in shapes.items():
        for mask_kind in ("none", "partial"):
            pixels = _pixels(shape, seed=hash((shape_name, mask_kind)) % (2**31))
            mask = _mask(shape, kind=mask_kind)
            for intensity_choice in ("Regular", "Background"):
                for smoothing_method, width_choice, width_value in (
                    ("No smoothing", "Automatic", 0),
                    ("Median Filter", "Automatic", 0),
                    ("Median Filter", "Manually", 4),
                    ("Gaussian Filter", "Automatic", 0),
                    ("Gaussian Filter", "Object size", 12),
                    ("Fit Polynomial", "Automatic", 0),
                    ("Smooth to Average", "Automatic", 0),
                ):
                    for rescale_option in ("Yes", "No", "Median"):
                        for dilate in (False, True):
                            if dilate and intensity_choice != "Regular":
                                continue
                            name = (
                                f"{shape_name}_{mask_kind}_{intensity_choice.lower()}_"
                                f"{smoothing_method.lower().replace(' ', '_')}_"
                                f"{width_choice.lower().replace(' ', '_')}_"
                                f"{rescale_option.lower()}"
                                + ("_dilated" if dilate else "")
                            )
                            result.append(
                                {
                                    "name": name,
                                    "pixels": pixels,
                                    "mask": mask,
                                    "intensity_choice": intensity_choice,
                                    "block_size": 6,
                                    "rescale_option": rescale_option,
                                    "smoothing_method": smoothing_method,
                                    "automatic_object_width": width_choice,
                                    "object_width": width_value or 10,
                                    "size_of_smoothing_filter": width_value or 10,
                                    "dilate_objects": dilate,
                                    "object_dilation_radius": 2,
                                }
                            )
    return result


def _run_case(case: dict[str, Any]) -> np.ndarray:
    module = CorrectIlluminationCalculate()
    module.image_name.value = "Input"
    module.illumination_image_name.value = "Illum"
    module.intensity_choice.value = case["intensity_choice"]
    module.dilate_objects.value = case["dilate_objects"]
    module.object_dilation_radius.value = case["object_dilation_radius"]
    module.block_size.value = case["block_size"]
    module.rescale_option.value = case["rescale_option"]
    module.each_or_all.value = "Each"
    module.smoothing_method.value = case["smoothing_method"]
    module.automatic_object_width.value = case["automatic_object_width"]
    module.object_width.value = case["object_width"]
    module.size_of_smoothing_filter.value = case["size_of_smoothing_filter"]
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    mask = case["mask"]
    if mask is None:
        image_set.add("Input", Image(case["pixels"].copy(), convert=False))
    else:
        image_set.add(
            "Input", Image(case["pixels"].copy(), mask=mask.copy(), convert=False)
        )
    object_set = ObjectSet()
    measurements = Measurements()
    workspace = Workspace(
        pipeline, module, image_set, object_set, measurements, image_sets
    )
    module.run(workspace)
    return image_set.get_image("Illum").pixel_data


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(_correctillumination_module.__file__)
    provenance = {
        "reference": f"CellProfiler 4.2.8 @ {SOURCE_COMMIT}",
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": (
            "2D, 'Each' mode only; 5 of 7 smoothing methods "
            "(everything except Convex Hull and Splines)"
        ),
    }
    manifest = []
    for case in cases():
        name = case["name"]
        output = _run_case(case)
        result = {
            **provenance,
            "settings": {
                k: v for k, v in case.items() if k not in ("name", "pixels", "mask")
            },
            "pixels": case["pixels"].tolist(),
            "mask": None if case["mask"] is None else case["mask"].tolist(),
            "expected": output.tolist(),
        }
        (OUT / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=True) + "\n"
        )
        manifest.append({"name": name, "shape": list(case["pixels"].shape)})
        print(name)
    (OUT / "manifest.json").write_text(
        json.dumps({**provenance, "cases": manifest}, indent=2) + "\n"
    )
    print(f"wrote {len(manifest)} cases")


if __name__ == "__main__":
    generate()
