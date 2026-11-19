"""Parse Criterion's raw sample data into the project's benchmark-report shape.

Reads `target/criterion/object_intensity_and_neighbors/<fixture>/new/sample.json`
(written by `cargo bench --manifest-path crates/cp_anywhere/Cargo.toml`)
for the four representative workloads, families `object_intensity` +
`neighbors` only (matching exactly what the pinned `cp_measure` baseline in
`benchmarks/baseline/initial-cp-measure-timing.json` computed), and writes
`benchmarks/reports/native-rust-baseline.json` in the same shape as the
other baseline reports.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ["touching_gradient", "real_cell_cc0", "grid_medium", "grid_large"]


def load_samples(fixture: str) -> list[float]:
    path = (
        ROOT
        / f"target/criterion/object_intensity_and_neighbors/{fixture}/new/sample.json"
    )
    if not path.exists():
        msg = (
            "Run `cargo bench --manifest-path "
            f"crates/cp_anywhere/Cargo.toml` first: {path}"
        )
        raise FileNotFoundError(msg)
    data = json.loads(path.read_text())
    return [t / i / 1e6 for t, i in zip(data["times"], data["iters"], strict=True)]


def main() -> None:
    workloads = []
    for fixture in WORKLOADS:
        samples = load_samples(fixture)
        workloads.append(
            {
                "fixture": fixture,
                "raw_ms": samples,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
        )
    report = {
        "contract_version": 1,
        "engine": (
            "cp-anywhere native Rust (cp_anywhere::measure, no Python/FFI boundary)"
        ),
        "families_timed": ["object_intensity", "neighbors"],
        "comparison_note": (
            "families_timed matches exactly what cp_measure 0.3.0 implements "
            "(no image_intensity, no relationship rows); see "
            "benchmarks/baseline/initial-cp-measure-timing.json for the "
            "baseline computed on the same scope."
        ),
        "harness": (
            "Criterion.rs, cargo bench --manifest-path crates/cp_anywhere/Cargo.toml"
        ),
        "workloads": workloads,
    }
    out = ROOT / "benchmarks/reports/native-rust-baseline.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "written": str(out),
                "medians_ms": {w["fixture"]: w["median_ms"] for w in workloads},
            }
        )
    )


if __name__ == "__main__":
    main()
