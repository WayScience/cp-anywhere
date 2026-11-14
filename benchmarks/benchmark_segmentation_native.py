"""Time cp_anywhere.identify_primary_objects (native Rust) on real
microscopy fixtures -- same methodology as benchmark_cp_measure.py, for
direct comparison with benchmark_segmentation_cellprofiler.py's real
CellProfiler ``IdentifyPrimaryObjects.run()`` timing on the same images.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np

import cp_anywhere

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ["real_cell_cc0_dna_otsu", "real_bbbc007_dna_otsu"]
WARMUPS = 3
RUNS = 20


def samples(function: Callable[[], object], warmups: int, runs: int) -> list[float]:
    for _ in range(warmups):
        function()
    out = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        out.append((time.perf_counter_ns() - start) / 1e6)
    return out


def summarize(values: list[float]) -> dict:
    return {
        "raw_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    report = {
        "baseline": "cp_anywhere (native Rust)",
        "warmups": WARMUPS,
        "timed_runs": RUNS,
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "machine": platform.machine(),
            "packages": {"cp_anywhere": metadata.version("cp-anywhere")},
        },
        "workloads": [],
    }
    for name in WORKLOADS:
        folder = ROOT / "fixtures/segmentation" / name
        pixels = np.load(folder / "pixels.npy")
        case = json.loads((ROOT / f"fixtures/segmentation/{name}.json").read_text())

        times = samples(
            lambda pixels=pixels: cp_anywhere.identify_primary_objects(
                pixels, threshold="otsu"
            ),
            WARMUPS,
            RUNS,
        )
        report["workloads"].append(
            {
                "fixture": name,
                "shape": list(pixels.shape),
                "object_count": case["object_count"],
                "identify_primary_objects": summarize(times),
            }
        )
        print(f"{name}: {statistics.median(times):.4f} ms", flush=True)

    path = ROOT / "benchmarks/reports/segmentation-native-baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
