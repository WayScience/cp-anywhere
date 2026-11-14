"""Benchmark the installed cp-anywhere Python binding (quantifies the
PyO3/NumPy FFI overhead over the bare Rust core; plan 5.2 "Python binding
overhead: Quantified and bounded").

Same methodology, same four representative workloads, and the same
`object_intensity` + `neighbors` family scope as
`benchmarks/benchmark_cp_measure.py` and
`scripts/benchmarks/report_native_rust.py`, so all three reports are
directly comparable.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np

import cp_anywhere

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ["touching_gradient", "real_cell_cc0", "grid_medium", "grid_large"]
WARMUPS = 3
RUNS = 10


def fixture_dir(name: str) -> Path:
    family = "real" if name == "real_cell_cc0" else "synthetic"
    return ROOT / f"fixtures/{family}/{name}"


def load(name: str) -> tuple[dict, dict]:
    folder = fixture_dir(name)
    images = {
        channel: np.load(folder / f"{channel}.npy") for channel in ("DNA", "Actin")
    }
    objects = {
        "Cells": np.load(folder / "labels.npy"),
        "Nuclei": np.load(folder / "nuclei_labels.npy"),
    }
    return images, objects


def samples(function: Callable[[], object], warmups: int, runs: int) -> list[float]:
    for _ in range(warmups):
        function()
    out = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        out.append((time.perf_counter_ns() - start) / 1e6)
    return out


def main() -> None:
    workloads = []
    for name in WORKLOADS:
        images, objects = load(name)
        times = samples(
            lambda: cp_anywhere.measure(
                images,
                objects,
                image_id=name,
                families=["object_intensity", "neighbors"],
            ),
            WARMUPS,
            RUNS,
        )
        workloads.append(
            {
                "fixture": name,
                "raw_ms": times,
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
        )
        print(f"{name}: median {statistics.median(times):.4f} ms", flush=True)
    engine = "cp-anywhere Python binding (cp_anywhere.measure, installed release wheel)"
    report = {
        "contract_version": 1,
        "engine": engine,
        "families_timed": ["object_intensity", "neighbors"],
        "warmups": WARMUPS,
        "timed_runs": RUNS,
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                "cp-anywhere": metadata.version("cp-anywhere"),
                "numpy": metadata.version("numpy"),
            },
        },
        "peak_rss_bytes_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin"
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "workloads": workloads,
        "limitations": [
            "This times Python call overhead plus the Rust kernel plus NumPy "
            "array marshaling; it is not an isolated kernel time (see "
            "scripts/benchmarks/report_native_rust.py for that).",
        ],
    }
    out = ROOT / "benchmarks/reports/python-binding-baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    main()
