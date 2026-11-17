# ty: ignore[import-not-found]
"""Oracle: real CellProfiler 4.2.8 CorrectIlluminationCalculate's "All"
mode accumulation (`CorrectIlluminationImageProvider`), run directly
against several real `Image` objects -- not a hand-derived average."""

# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
from cellprofiler.modules.correctilluminationcalculate import (
    CorrectIlluminationCalculate,
    CorrectIlluminationImageProvider,
)
from cellprofiler_core.image import Image

OUT = Path(__file__).resolve().parents[2] / "fixtures/illumination_calculate_across"


def _pixels(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, shape)


def _mask(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masked_fraction = 0.2
    return rng.uniform(size=shape) > masked_fraction


def cases() -> list[dict[str, Any]]:
    shape = (10, 12)
    result = []
    for intensity_choice in ("Regular", "Background"):
        for mask_kind in ("none", "mixed"):
            images = [_pixels(shape, seed=100 + i) for i in range(4)]
            if mask_kind == "none":
                masks: list[np.ndarray | None] = [None, None, None, None]
            else:
                # A mix of masked and unmasked images in the same group.
                masks = [
                    None,
                    _mask(shape, seed=200),
                    _mask(shape, seed=201),
                    None,
                ]
            name = f"{intensity_choice.lower()}_{mask_kind}"
            result.append(
                {
                    "name": name,
                    "images": images,
                    "masks": masks,
                    "intensity_choice": intensity_choice,
                    "block_size": 4,
                    "rescale_option": "Yes",
                    "smoothing_method": "Gaussian Filter",
                    "automatic_object_width": "Automatic",
                    "object_width": 10,
                    "size_of_smoothing_filter": 10,
                    "dilate_objects": False,
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
    module.each_or_all.value = "All: Across cycles"
    module.smoothing_method.value = case["smoothing_method"]
    module.automatic_object_width.value = case["automatic_object_width"]
    module.object_width.value = case["object_width"]
    module.size_of_smoothing_filter.value = case["size_of_smoothing_filter"]

    provider = CorrectIlluminationImageProvider("Illum", module)
    for pixels, mask in zip(case["images"], case["masks"]):
        if mask is None:
            image = Image(pixels.copy(), convert=False)
        else:
            image = Image(pixels.copy(), mask=mask.copy(), convert=False)
        provider.add_image(image)
    return provider.provide_image(None).pixel_data


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases():
        name = case["name"]
        output = _run_case(case)
        result = {
            "settings": {
                k: v for k, v in case.items() if k not in ("name", "images", "masks")
            },
            "images": [img.tolist() for img in case["images"]],
            "masks": [None if m is None else m.tolist() for m in case["masks"]],
            "expected": output.tolist(),
        }
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest.append(name)
        print(name)
    (OUT / "manifest.json").write_text(json.dumps({"cases": manifest}, indent=2) + "\n")
    print(f"wrote {len(manifest)} cases")


if __name__ == "__main__":
    generate()
