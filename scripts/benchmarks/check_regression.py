"""CI regression check: compare a freshly measured native-rust-baseline.json
against the committed baseline with a tolerance margin.

Per plan 5.2 ("add CI regression thresholds and trend reports; avoid
hard-failing on noisy single-run results"), this only fails when a workload
is slower than the committed baseline by more than `TOLERANCE`, never on an
improvement, and never on noise within the margin. It reports every
workload's delta so a regression in one workload cannot hide behind others
improving.

Usage:
    cargo bench --manifest-path crates/cp_anywhere/Cargo.toml
    uv run --frozen python scripts/benchmarks/report_native_rust.py
    uv run --frozen python scripts/benchmarks/check_regression.py

To intentionally accept a new baseline after a verified, deliberate change
(e.g. a real optimization or a documented tradeoff), copy the fresh
benchmarks/reports/native-rust-baseline.json over
benchmarks/baseline/native-rust-committed.json and commit both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 0.20  # 20% slower than committed is a regression; faster never fails


def main() -> None:
    committed_path = ROOT / "benchmarks/baseline/native-rust-committed.json"
    fresh_path = ROOT / "benchmarks/reports/native-rust-baseline.json"
    if not committed_path.exists():
        print(f"No committed baseline at {committed_path} yet; nothing to compare.")
        return
    committed = {
        w["fixture"]: w["median_ms"]
        for w in json.loads(committed_path.read_text())["workloads"]
    }
    fresh = {
        w["fixture"]: w["median_ms"]
        for w in json.loads(fresh_path.read_text())["workloads"]
    }

    rows = []
    regressed = []
    for fixture, committed_ms in committed.items():
        fresh_ms = fresh.get(fixture)
        if fresh_ms is None:
            continue
        delta = (fresh_ms - committed_ms) / committed_ms
        rows.append(
            {
                "fixture": fixture,
                "committed_ms": committed_ms,
                "fresh_ms": fresh_ms,
                "delta": delta,
            }
        )
        if delta > TOLERANCE:
            regressed.append(fixture)

    print(
        json.dumps(
            {"tolerance": TOLERANCE, "rows": rows, "regressed": regressed}, indent=2
        )
    )
    if regressed:
        print(
            f"REGRESSION: {regressed} slower than committed baseline "
            f"by more than {TOLERANCE:.0%}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("No workload regressed beyond tolerance.")


if __name__ == "__main__":
    main()
