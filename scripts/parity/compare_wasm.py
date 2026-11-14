"""Compare the real in-browser WASM result with the saved golden fixtures.

Reads the JSON written by `examples/browser-vite/run-browser-check.mjs`
(itself produced by a real Playwright Chromium run of the real Vite
production build, using a web worker) and runs the same per-feature
parity rules as `scripts/parity/compare_native.py`, over the 2D fixture
subset the browser demo exports (see `scripts/wasm/export_fixtures.py`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.parity.compare_cp_measure import (  # noqa: E402
    compare_feature,
    compare_relationships,
)


def compare_fixture(name: str, output_json: str, features: list[dict]) -> list[dict]:
    """Compare one browser-produced fixture result with its golden output."""
    actual = json.loads(output_json)
    reference = json.loads((ROOT / f"fixtures/reference/{name}.json").read_text())
    rows = []
    for feature in features:
        scope, feature_name = feature["object_scope"], feature["name"]
        expected = reference["features"][scope][feature_name]
        got = actual["features"].get(scope, {}).get(feature_name)
        ref_ids = [1] if scope == "Image" else reference["object_ids"][scope]
        got_ids = [1] if scope == "Image" else actual["object_ids"].get(scope)
        outcome = compare_feature(
            [expected] if scope == "Image" else expected,
            [got] if scope == "Image" and got is not None else got,
            ref_ids,
            got_ids,
            feature["comparison"],
            actual_dtype=None,
            expected_dtype=feature["dtype"],
        )
        rows.append(
            {
                "fixture": name,
                "object_scope": scope,
                "feature": feature_name,
                "module": feature["module"],
                **outcome,
            }
        )
    rows.append(
        {
            "fixture": name,
            "object_scope": "relationship",
            "feature": "Neighbors/Cells/Cells/directed",
            "module": "MeasureObjectNeighbors",
            **compare_relationships(
                reference["relationships"], actual["relationships"]
            ),
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "examples/browser-vite/browser-results.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/reports/wasm-parity.json"
    )
    args = parser.parse_args()
    browser = json.loads(args.results.read_text())
    if browser["status"] != "done" or browser["errors"]:
        print(
            json.dumps(
                {
                    "error": "browser run did not complete cleanly",
                    "status": browser["status"],
                    "errors": browser["errors"],
                }
            )
        )
        raise SystemExit(1)
    features = yaml.safe_load((ROOT / "spec/features.yaml").read_text())["features"]
    parity = browser["results"]["parity"]
    rows = [
        row
        for name, output in parity.items()
        for row in compare_fixture(name, output, features)
    ]
    summary = dict(Counter(row["status"] for row in rows))
    payload = {
        "reference": "CellProfiler 4.2.8",
        "engine": "cp-anywhere WASM (Playwright Chromium, real worker)",
        "feature_count": len(features),
        "fixtures": len(parity),
        "fixture_names": sorted(parity),
        "memory_bytes": browser["results"]["memoryBytes"],
        "wasm_transfer_bytes": browser["results"]["wasmTransferBytes"],
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "fixture",
                "object_scope",
                "feature",
                "module",
                "status",
                "max_abs_error",
                "max_rel_error",
                "mismatches",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**row, "mismatches": json.dumps(row.get("mismatches", []))}
            )
    print(
        json.dumps({"rows": len(rows), "summary": summary, "output": str(args.output)})
    )
    if any(not status.startswith("pass") for status in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
