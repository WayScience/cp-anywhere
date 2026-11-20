# ty: ignore[import-not-found]
"""Capture an image-quality, flag, and image-preparation branch from CP 4.2.8."""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.flagimage import FlagImage
from cellprofiler.modules.imagemath import ImageMath
from cellprofiler.modules.measureimagequality import O_SELECT, MeasureImageQuality
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/cppipe/image_quality"


def pipeline_modules() -> tuple[MeasureImageQuality, FlagImage, ImageMath]:
    quality = MeasureImageQuality()
    quality.images_choice.value = O_SELECT
    group = quality.image_groups[0]
    group.image_names.value = "DNA"
    group.include_image_scalings.value = True
    group.check_blur.value = True
    group.scale_groups[0].scale.value = 20
    group.check_saturation.value = True
    group.check_intensity.value = False
    group.calculate_threshold.value = False

    flag = FlagImage()
    entry = flag.flags[0]
    entry.category.value = "QC"
    entry.feature_name.value = "Rejected"
    entry.wants_skip.value = True
    criteria = entry.measurement_settings[0]
    criteria.measurement.value = "ImageQuality_PercentMaximal_DNA"
    criteria.wants_minimum.value = False
    criteria.wants_maximum.value = True
    criteria.maximum_value.value = 10

    math = ImageMath()
    math.operation.value = "None"
    math.images[0].image_name.value = "DNA"
    math.output_image_name.value = "AfterQC"
    return quality, flag, math


def main() -> None:
    pipeline = Pipeline()
    modules = pipeline_modules()
    for i, module in enumerate(modules, 1):
        module.set_module_num(i)
        pipeline.add_module(module)
    with (OUT / "quality_flag.cppipe").open("w", encoding="utf-8") as handle:
        pipeline.dump(handle)
    rows, cols = np.indices((64, 80))
    pattern = ((39 * rows + 11 * cols + 7 * rows * cols) % 53 + (rows + cols) % 5) / 256
    record = {}
    for name, source in (("kept", pattern), ("skipped", np.full_like(pattern, 0.25))):
        pixels = source.astype(np.float32)
        image_sets = ImageSetList()
        image_set = image_sets.get_image_set(0)
        image_set.add("DNA", Image(pixels, scale=4095))
        measurements = Measurements()
        try:
            skipped = False
            for module in modules:
                workspace = Workspace(
                    pipeline, module, image_set, ObjectSet(), measurements, image_sets
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    module.run(workspace)
                if workspace.disposition == "Skip":
                    skipped = True
                    break
            record[name] = {
                "skipped": skipped,
                "columns": {
                    column: np.asarray(
                        measurements.get_current_image_measurement(column)
                    ).item()
                    for column in sorted(measurements.get_feature_names("Image"))
                },
            }
            if not skipped:
                np.testing.assert_array_equal(
                    image_set.get_image("AfterQC").pixel_data, pixels
                )
            np.save(OUT / f"quality_flag.{name}.npy", pixels)
        finally:
            measurements.close()
    (OUT / "quality_flag.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
