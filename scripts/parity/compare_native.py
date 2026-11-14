"""Compare the native Rust result with all saved CellProfiler 4.2.8 fixtures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import cp_anywhere

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.parity.compare_cp_measure import (  # noqa: E402
    compare_feature,
    compare_relationships,
    load_arrays,
)


def compare_fixture(entry: dict, features: list[dict]) -> list[dict]:
    """Compare one saved input with its original CellProfiler output."""
    arrays = load_arrays(entry)
    result = cp_anywhere.measure(
        {name: arrays[name] for name in entry["channel_names"]},
        {"Cells": arrays["labels"], "Nuclei": arrays["nuclei_labels"]},
        mask=arrays.get("mask"),
        image_id=entry["name"],
    )
    reference = json.loads((ROOT / entry["result"]).read_text())
    rows = []
    for feature in features:
        scope, name = feature["object_scope"], feature["name"]
        expected = reference["features"][scope][name]
        actual = result.features.get(scope, {}).get(name)
        ref_ids = [1] if scope == "Image" else reference["object_ids"][scope]
        got_ids = [1] if scope == "Image" else result.object_ids[scope].tolist()
        outcome = compare_feature(
            [expected] if scope == "Image" else expected,
            None
            if actual is None
            else [actual]
            if scope == "Image"
            else np.asarray(actual).tolist(),
            ref_ids,
            None if actual is None else got_ids,
            feature["comparison"],
            actual_dtype=None if actual is None else str(np.asarray(actual).dtype),
            expected_dtype=feature["dtype"],
        )
        rows.append(
            {
                "fixture": entry["name"],
                "object_scope": scope,
                "feature": name,
                "module": feature["module"],
                **outcome,
            }
        )
    rows.append(
        {
            "fixture": entry["name"],
            "object_scope": "relationship",
            "feature": "Neighbors/Cells/Cells/directed",
            "module": "MeasureObjectNeighbors",
            **compare_relationships(reference["relationships"], result.relationships),
        }
    )
    return rows


def main() -> None:
    """Write full JSON and CSV reports and fail on any difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/reports/native-parity.json"
    )
    args = parser.parse_args()
    features = yaml.safe_load((ROOT / "spec/features.yaml").read_text())["features"]
    entries = json.loads((ROOT / "fixtures/reference/manifest.json").read_text())[
        "fixtures"
    ]
    rows = [row for entry in entries for row in compare_fixture(entry, features)]
    summary = dict(Counter(row["status"] for row in rows))
    payload = {
        "reference": "CellProfiler 4.2.8",
        "engine": "cp-anywhere Rust",
        "feature_count": len(features),
        "fixtures": len(entries),
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
