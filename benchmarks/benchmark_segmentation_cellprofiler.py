#  ty: ignore[import-not-found]
"""Time original CellProfiler 4.2.8's IdentifyPrimaryObjects.run() on
real microscopy fixtures -- same methodology as
benchmark_segmentation_native.py, for direct comparison.

Runs in the separate pinned-oracle environment, same as
scripts/segmentation/generate_reference.py.
"""

# ruff: noqa: ANN001, ANN201, E402

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler_core.image import Image, ImageSetList, VanillaImage
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ["real_cell_cc0_dna_otsu", "real_bbbc007_dna_otsu"]
WARMUPS = 3
RUNS = 20


def run_once(pixels) -> None:
    module = IdentifyPrimaryObjects()
    module.x_name.value = "image"
    module.y_name.value = "objects"
    module.use_advanced.value = True
    module.threshold.threshold_scope.value = "Global"
    module.threshold.global_operation.value = "Otsu"
    module.threshold.two_class_otsu.value = "Two classes"
    module.threshold.threshold_smoothing_scale.value = 0
    module.unclump_method.value = "None"
    module.watershed_method.value = "None"
    module.fill_holes.value = "Never"
    module.exclude_border_objects.value = False
    module.exclude_size.value = False

    image = Image(pixels, convert=True, dimensions=2)
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.providers.append(VanillaImage("image", image))
    object_set = ObjectSet()
    measurements = Measurements()
    pipeline = Pipeline()
    module.run(Workspace(pipeline, module, image_set, object_set, measurements, None))


def samples(function, warmups, runs):
    for _ in range(warmups):
        function()
    out = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        out.append((time.perf_counter_ns() - start) / 1e6)
    return out


def summarize(values):
    return {
        "raw_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    report = {
        "baseline": "CellProfiler 4.2.8 (original, headless)",
        "warmups": WARMUPS,
        "timed_runs": RUNS,
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "workloads": [],
    }
    for name in WORKLOADS:
        pixels = np.load(ROOT / f"fixtures/segmentation/{name}/pixels.npy")
        case = json.loads((ROOT / f"fixtures/segmentation/{name}.json").read_text())

        times = samples(lambda pixels=pixels: run_once(pixels), WARMUPS, RUNS)
        report["workloads"].append(
            {
                "fixture": name,
                "shape": list(pixels.shape),
                "object_count": case["object_count"],
                "identify_primary_objects": summarize(times),
            }
        )
        print(f"{name}: {statistics.median(times):.4f} ms", flush=True)

    path = ROOT / "benchmarks/reports/segmentation-cellprofiler-baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
