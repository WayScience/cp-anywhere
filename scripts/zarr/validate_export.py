"""Native validation for the Zarr/ONNX/DuckDB browser integration demo.

Reads the real Parquet file the browser's DuckDB-WASM instance exported
(examples/zarr-onnx-duckdb/cp_anywhere_features.parquet, produced by
`npm run browser-check`), validates it with *native* DuckDB (not the WASM
build under test), and cross-checks every value against a fresh, from-source
native Python `cp_anywhere.measure()` call fed the exact same Zarr bytes and
a fresh native `cp_anywhere.identify_primary_objects()` segmentation (the
same settings `main.ts` passes to `identifyPrimaryObjects`) -- proving the
browser pipeline (both its own segmentation and its measurement) is
numerically correct, not just that it ran without crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

import cp_anywhere

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/zarr-onnx-duckdb"
TOTAL_ROWS = 21  # 1 object x the 21 object_intensity columns for a single DNA channel


def main() -> None:
    parquet_path = EXAMPLE / "cp_anywhere_features.parquet"
    if not parquet_path.exists():
        msg = (
            f"Run `npm run browser-check` in {EXAMPLE} first: {parquet_path} not found"
        )
        raise FileNotFoundError(msg)

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT * FROM read_parquet('{parquet_path}') ORDER BY feature"
    ).fetchall()
    columns = [d[0] for d in con.description]
    browser_values = {
        row[columns.index("feature")]: row[columns.index("value")] for row in rows
    }

    if len(rows) != TOTAL_ROWS:
        msg = f"expected {TOTAL_ROWS} rows in the exported Parquet, got {len(rows)}"
        raise AssertionError(msg)

    # Re-derive the identical inputs from source: the exact Zarr chunk bytes
    # (what zarrita.js actually read), fresh-segmented natively with the
    # same settings main.ts passes to cp-anywhere-wasm's
    # `identifyPrimaryObjects` (not the ONNX demo mask -- that model is
    # still run in the browser for comparison/logging only, see
    # docs/browser-integration.md).
    raw = (EXAMPLE / "public/dna.zarr/0.0").read_bytes()
    image = np.frombuffer(raw, dtype="<f8").reshape(12, 12).copy()
    seg = cp_anywhere.identify_primary_objects(
        image,
        threshold="otsu",
        unclump_method="intensity",
        watershed_method="intensity",
        min_diameter=4.0,
    )

    native = cp_anywhere.measure(
        {"DNA": image},
        {"Cells": seg.labels},
        image_id="zarr-onnx-demo",
        families=["object_intensity"],
    )
    native_values = {
        name: float(np.asarray(values)[0])
        for name, values in native.features["Cells"].items()
    }

    mismatches = []
    for name, native_value in native_values.items():
        browser_value = browser_values.get(name)
        if browser_value is None:
            mismatches.append({"feature": name, "issue": "missing from browser export"})
            continue
        if not np.isclose(browser_value, native_value, rtol=1e-6, atol=1e-9):
            mismatches.append(
                {"feature": name, "native": native_value, "browser": browser_value}
            )

    report = {
        "rows_in_parquet": len(rows),
        "features_compared": len(native_values),
        "mismatches": mismatches,
        "cp_anywhere_object_count": int(seg.object_count),
        "cp_anywhere_threshold": float(seg.threshold),
    }
    print(json.dumps(report, indent=2))
    if mismatches:
        msg = (
            f"{len(mismatches)} feature(s) mismatched between "
            "browser export and native measure()"
        )
        raise AssertionError(msg)
    print("OK: browser-exported Parquet matches native cp_anywhere.measure() exactly.")


if __name__ == "__main__":
    main()
