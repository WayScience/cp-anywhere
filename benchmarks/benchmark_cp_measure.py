"""Measure actual pinned cp_measure kernels on the representative fixtures."""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from cp_measure.core.measureobjectintensity import (  # ty: ignore[import-not-found]
    get_intensity,
)
from cp_measure.multimask.measureobjectneighbors import (  # ty: ignore
    measureobjectneighbors,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parity.compare_cp_measure import load_arrays, load_catalog  # noqa: E402


def prepare(arrays: dict) -> dict:
    """Make the same masked labels and preserved pixel values as parity."""
    mask = arrays.get("mask")
    result = {
        channel: np.ascontiguousarray(
            arrays[channel] if mask is None else np.where(mask, arrays[channel], np.nan)
        )
        for channel in ("DNA", "Actin")
    }
    for scope, name in (("Cells", "labels"), ("Nuclei", "nuclei_labels")):
        labels = arrays[name]
        result[scope] = np.ascontiguousarray(labels)
    result["neighbors_labels"] = np.ascontiguousarray(arrays["labels"])
    return result


def intensity(prepared: dict) -> dict:
    result = {}
    for scope in ("Cells", "Nuclei"):
        for channel in ("DNA", "Actin"):
            features = get_intensity(
                prepared[scope], prepared[channel], edge_measurements=True, legacy=True
            )
            result.update(
                {
                    f"{scope}/{name}_{channel}": values.tolist()
                    for name, values in features.items()
                }
            )
    return result


def neighbors(prepared: dict) -> dict:
    labels = prepared["neighbors_labels"]
    return {
        name: np.asarray(values).tolist()
        for name, values in measureobjectneighbors(
            labels, labels, distance_method="Adjacent"
        ).items()
    }


def samples(function: Callable, warmups: int, runs: int) -> tuple[list[float], Any]:
    for _ in range(warmups):
        function()
    timings = []
    output = None
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = function()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return timings, output


def summarize(values: list[float]) -> dict:
    return {
        "raw_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    contract = yaml.safe_load((ROOT / "spec/benchmark-contract.yaml").read_text())
    manifest = json.loads((ROOT / "fixtures/reference/manifest.json").read_text())
    entries = {entry["name"]: entry for entry in manifest["fixtures"]}
    warmups, runs = (
        contract["measurement"]["warmups"],
        contract["measurement"]["timed_runs"],
    )
    startup_code = (
        "import numpy, scipy, skimage, cp_measure; "
        "from cp_measure.core.measureobjectintensity import get_intensity; "
        "from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors"
    )
    startup, _ = samples(
        lambda: subprocess.run(
            [sys.executable, "-c", startup_code],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        1,
        3,
    )
    report = {
        "contract_version": contract["contract_version"],
        "baseline": "cp_measure",
        "source_commit": contract["native_baseline"]["commit"],
        "warmups": warmups,
        "timed_runs": runs,
        "startup_import_ms": summarize(startup),
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names
            else None,
            "packages": {
                name: metadata.version(name)
                for name in (
                    "cp_measure",
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "centrosome",
                    "mahotas",
                    "PyYAML",
                )
            },
            "threads": {
                key: os.environ.get(key, "unset")
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "workloads": [],
    }
    catalog = load_catalog()
    mapped_count = sum(row["status"] == "mapped" for row in catalog.values())
    for workload in contract["representative_workloads"]:
        entry = entries[workload["fixture"]]
        arrays = load_arrays(entry)
        conversion, prepared = samples(lambda: prepare(arrays), warmups, runs)
        intensity_times, intensity_output = samples(
            lambda: intensity(prepared), warmups, runs
        )
        neighbor_times, neighbor_output = samples(
            lambda: neighbors(prepared), warmups, runs
        )

        def end_to_end() -> bytes:
            current = prepare(load_arrays(entry))
            output = {**intensity(current), **neighbors(current)}
            return json.dumps(output, allow_nan=True).encode()

        e2e_times, output_bytes = samples(end_to_end, warmups, runs)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["workloads"].append(
            {
                "fixture": entry["name"],
                "shape": entry["shape"],
                "pixels": int(np.prod(entry["shape"])),
                "object_count": int(np.count_nonzero(np.unique(arrays["labels"]))),
                "equivalent_feature_count": mapped_count,
                "actual_output_feature_count": len(intensity_output)
                + len(neighbor_output),
                "output_bytes_json": len(output_bytes),
                "peak_rss_bytes_process": peak
                if platform.system() == "Darwin"
                else peak * 1024,
                "conversion": summarize(conversion),
                "object_intensity": summarize(intensity_times),
                "neighbors": summarize(neighbor_times),
                "end_to_end": summarize(e2e_times),
            }
        )
        print(
            f"{entry['name']}: intensity {statistics.median(intensity_times):.3f} ms, "
            f"neighbors {statistics.median(neighbor_times):.3f} ms, "
            f"e2e {statistics.median(e2e_times):.3f} ms",
            flush=True,
        )
    report["limitations"] = [
        "Peak RSS is a process high-water mark, not a per-kernel allocation.",
        "Startup is a separate fresh-process import timer; kernel samples "
        "exclude imports.",
        "End to end includes saved NPY loading, conversion, kernels, and "
        "JSON output; it excludes CellProfiler and Parquet export.",
        "Only 91 mapped object columns are timed; image intensity and "
        "directed relationship rows have no cp_measure implementation.",
        "Timing is a comparison baseline, not evidence of CellProfiler parity.",
    ]
    path = ROOT / "benchmarks/reports/cp_measure-baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
