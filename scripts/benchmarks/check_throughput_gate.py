"""Compute the native_acceptance throughput gate from spec/benchmark-contract.yaml.

Compares the pinned cp_measure baseline
(benchmarks/baseline/initial-cp-measure-timing.json) against cp-anywhere's
native Rust kernel (benchmarks/reports/native-rust-baseline.json) on the
identical `object_intensity` + `neighbors` scope, per workload and as
an equal-weight arithmetic mean -- and reports every per-workload ratio so an
aggregate pass cannot hide an individual regression (plan 5.2: "report
per-feature regressions rather than allowing aggregate speed to hide a
major slowdown").
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    contract = yaml.safe_load((ROOT / "spec/benchmark-contract.yaml").read_text())
    cp_measure = json.loads(
        (ROOT / "benchmarks/baseline/initial-cp-measure-timing.json").read_text()
    )["timing_report"]
    rust = json.loads(
        (ROOT / "benchmarks/reports/native-rust-baseline.json").read_text()
    )

    cp_measure_sum = {}
    for workload in cp_measure["workloads"]:
        cp_measure_sum[workload["fixture"]] = (
            workload["object_intensity"]["median_ms"]
            + workload["neighbors"]["median_ms"]
        )
    rust_ms = {w["fixture"]: w["median_ms"] for w in rust["workloads"]}

    ratios = {}
    for workload in contract["representative_workloads"]:
        name = workload["fixture"]
        ratios[name] = cp_measure_sum[name] / rust_ms[name]

    mean_ratio = statistics.mean(ratios.values())
    minimum = contract["native_acceptance"]["minimum"]
    per_workload_pass = {name: ratio >= minimum for name, ratio in ratios.items()}
    report = {
        "metric": contract["native_acceptance"]["metric"],
        "minimum": minimum,
        "cp_measure_object_intensity_plus_neighbors_ms": cp_measure_sum,
        "cp_anywhere_rust_ms": rust_ms,
        "per_workload_throughput_ratio": ratios,
        "per_workload_pass": per_workload_pass,
        "equal_weight_mean_ratio": mean_ratio,
        "aggregate_pass": mean_ratio >= minimum,
        "all_workloads_pass_individually": all(per_workload_pass.values()),
        "note": (
            "A workload ratio below 1.0 means cp-anywhere Rust is slower than "
            "cp_measure for that workload; the aggregate mean must not be "
            "read as a pass if any per-workload ratio is below the minimum."
        ),
    }
    out = ROOT / "benchmarks/reports/native-throughput-gate.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "per_workload_throughput_ratio",
                    "equal_weight_mean_ratio",
                    "all_workloads_pass_individually",
                )
            }
        )
    )
    if not report["all_workloads_pass_individually"]:
        print(
            "FAIL: at least one representative workload is slower "
            "than the cp_measure baseline",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
