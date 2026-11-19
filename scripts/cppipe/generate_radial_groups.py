# ty: ignore[import-not-found]
"""Generate radial group fixtures with the pinned CellProfiler 4.2.8 oracle.

Pass the original CPJUMP1 analysis .cppipe from JUMP commit
6ba3fcd1495d9e844e4607373a568641981ffcd8 as the sole argument.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import sys
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from pathlib import Path

import cellprofiler
import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures/cppipe/intensity_distribution"
NAMES = ("Nuclei", "Cytoplasm", "Cells")
SOURCE_COMMIT = "6ba3fcd1495d9e844e4607373a568641981ffcd8"
SOURCE_SHA256 = "b2661d63bffc6ea09cc35fdd4a800491e38b9c3a1c5b96d0d477a42e31dc4bae"


def _block(text: str, name: str) -> str:
    return next(block for block in text.split("\n\n") if block.startswith(name + ":"))


def _pipeline(jump: Path) -> Pipeline:
    radial = _block(jump.read_text(), "MeasureObjectIntensityDistribution")
    source = (ROOT / "fixtures/cppipe/tertiary/no_shrink.cppipe").read_text()
    tertiary = _block(source, "IdentifyTertiaryObjects")
    header = source.split("\n\n", 1)[0].replace("ModuleCount:4", "ModuleCount:2")
    pipeline = Pipeline()
    pipeline.load(
        StringIO(
            f"{header}\n\n{tertiary.replace('module_num:3', 'module_num:1', 1)}"
            f"\n\n{radial.replace('module_num:25', 'module_num:2', 1)}\n"
        )
    )
    assert [module.module_name for module in pipeline.modules()] == [
        "IdentifyTertiaryObjects",
        "MeasureObjectIntensityDistribution",
    ]
    return pipeline


def _inputs(images: list[str], *, empty: bool) -> dict[str, np.ndarray]:
    rows, cols = np.indices((10, 14))
    nuclei = np.zeros(rows.shape, np.int32)
    cells = np.zeros(rows.shape, np.int32)
    cells[1:8, 1:6] = 1
    cells[2:9, 7:13] = 3
    if not empty:
        nuclei[3:6, 2:5] = 1
        nuclei[4:7, 8:11] = 3
    # Binary fractions keep float32 histogram sums exact for these shapes.
    return {
        **{
            name: (
                ((3 * rows + 5 * cols + (idx + 1) * rows * cols) % 63 + 1) / 64
            ).astype(np.float32)
            for idx, name in enumerate(images)
        },
        "Nuclei": nuclei,
        "Cells": cells,
    }


def _number(value: float) -> float | dict[str, str]:
    assert math.isfinite(value) or math.isnan(value)
    return {"nonfinite": "NaN"} if math.isnan(value) else float(value)


def _measure(pipeline: Pipeline, inputs: dict[str, np.ndarray]) -> dict:
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    objects = ObjectSet()
    for name, data in inputs.items():
        if name in {"Nuclei", "Cells"}:
            labels = Objects()
            labels.segmented = data
            objects.add_objects(labels, name)
        else:
            image_set.add(name, Image(data, convert=True, dimensions=2))

    measurements = Measurements()
    radial = pipeline.modules()[-1]
    try:
        for module in pipeline.modules():
            workspace = Workspace(
                pipeline, module, image_set, objects, measurements, image_sets
            )
            module.run(workspace)
        declared = {
            (scope, name) for scope, name, _ in radial.get_measurement_columns(pipeline)
        }
        actual = {
            (scope, name)
            for scope in NAMES
            for name in measurements.get_feature_names(scope)
            if name.startswith("RadialDistribution_")
        }
        assert declared == actual
        return {
            "cytoplasm": objects.get_objects("Cytoplasm").segmented.tolist(),
            "object_ids": {
                scope: measurements.get_current_measurement(
                    scope, "ObjectNumber"
                ).tolist()
                for scope in NAMES
            },
            "features": {
                scope: {
                    name: [
                        _number(value)
                        for value in measurements.get_current_measurement(scope, name)
                    ]
                    for obj, name in sorted(actual)
                    if obj == scope
                }
                for scope in NAMES
            },
        }
    finally:
        measurements.close()


def generate(jump: Path) -> None:
    """Replay JUMP's settings and both orders of mixed bin groups."""
    if sha256(jump.read_bytes()).hexdigest() != SOURCE_SHA256:
        msg = f"Expected the original CPJUMP1 pipeline at commit {SOURCE_COMMIT}"
        raise ValueError(msg)
    cases = {
        "jump_groups": (),
        "scaled_then_unscaled": ((True, 4, 100), (False, 2, 1), (False, 3, 2)),
        "unscaled_then_scaled": ((False, 2, 1), (True, 4, 100), (False, 3, 2)),
    }
    for case, bins in cases.items():
        pipeline = _pipeline(jump)
        radial = pipeline.modules()[-1]
        if bins:
            radial.images_list.value = "DNA, RNA"
            for _ in bins[1:]:
                radial.add_bin_count()
            for group, (scaled, count, radius) in zip(radial.bin_counts, bins):
                group.wants_scaled.value = scaled
                group.bin_count.value = count
                group.maximum_radius.value = radius
        inputs = _inputs(
            list(radial.images_list.value), empty=case == "scaled_then_unscaled"
        )
        # JUMP's block has no intensity Zernikes. Keep that setting in every case.
        assert radial.wants_zernikes.value == "None"
        with (OUT / f"{case}.cppipe").open("w") as stream:
            pipeline.dump(stream, save_image_plane_details=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = _measure(pipeline, inputs)
        expected.update(
            reference={
                "cellprofiler": cellprofiler.__version__,
                **{
                    package: version(package)
                    for package in ("CellProfiler-core", "centrosome", "numpy", "scipy")
                },
                "jump_commit": SOURCE_COMMIT,
                "jump_pipeline_sha256": SOURCE_SHA256,
            },
            inputs={
                name: {"dtype": array.dtype.name, "values": array.tolist()}
                for name, array in inputs.items()
            },
        )
        (OUT / f"{case}.oracle.json").write_text(
            json.dumps(expected, sort_keys=True) + "\n"
        )
        print(case, {name: len(cols) for name, cols in expected["features"].items()})


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
