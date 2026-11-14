"""Compare pinned cp_measure outputs with saved CellProfiler fixtures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_catalog(root: Path = ROOT) -> dict[tuple[str, str], dict]:
    """Validate the independent cp_measure map against the oracle inventory."""
    inventory = yaml.safe_load((root / "spec/features.yaml").read_text())["features"]
    mapping = yaml.safe_load((root / "spec/cp_measure-map.yaml").read_text())
    expected = {(f["object_scope"], f["name"]) for f in inventory}
    rows = {(f["object_scope"], f["name"]): f for f in mapping["features"]}
    if len(rows) != len(mapping["features"]) or set(rows) != expected:
        raise ValueError("cp_measure map must cover each original feature once")
    for f in inventory:
        key = (f["object_scope"], f["name"])
        rows[key] = {
            **rows[key],
            "rule": f["comparison"],
            "dtype": f["dtype"],
            "module": f["module"],
        }
        if rows[key]["status"] == "mapped" and not rows[key]["cp_measure_feature"]:
            raise ValueError(f"Mapped feature lacks cp_measure name: {key}")
    return rows


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict) and set(value) == {"nonfinite"}:
        return str(value["nonfinite"])
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        return "+Infinity" if value == "Infinity" else value
    if isinstance(value, (float, int)) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
        )
    return "finite"


def _json_value(value: object) -> object:
    kind = _kind(value)
    return {"nonfinite": kind} if kind not in {"finite", "null"} else value


def compare_feature(  # noqa: C901, PLR0912, PLR0913
    reference: list | None,
    actual: list | None,
    reference_ids: list[int],
    actual_ids: list[int] | None,
    rule: dict,
    *,
    actual_dtype: str | None = None,
    expected_dtype: str = "float64",
) -> dict:
    """Compare presence, IDs, shape, dtype, null/nonfinite, then values."""
    result = {
        "status": "pass_exact",
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "mismatches": [],
    }
    if actual is None:
        return {
            **result,
            "status": "missing",
            "mismatches": [
                {"kind": "presence", "expected": "present", "actual": "missing"}
            ],
        }
    if reference is None:
        raise ValueError("Reference feature missing")
    if actual_ids is None:
        actual_ids = []
    if len(reference) != len(reference_ids) or len(actual) != len(actual_ids):
        result["mismatches"].append(
            {"kind": "shape", "expected": len(reference), "actual": len(actual)}
        )
    if actual_dtype is not None and actual_dtype != expected_dtype:
        result["mismatches"].append(
            {"kind": "dtype", "expected": expected_dtype, "actual": actual_dtype}
        )
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(reference_ids):
        result["mismatches"].append(
            {"kind": "ids", "expected": reference_ids, "actual": actual_ids}
        )
        for object_id in sorted(set(reference_ids) - set(actual_ids)):
            result["mismatches"].append(
                {
                    "kind": "missing_id",
                    "object_id": object_id,
                    "expected": reference[reference_ids.index(object_id)],
                    "actual": None,
                }
            )
    values = dict(zip(actual_ids, actual, strict=False))
    for object_id, expected in zip(reference_ids, reference, strict=False):
        if object_id not in values:
            continue
        got = values[object_id]
        ek, gk = _kind(expected), _kind(got)
        if ek != gk:
            result["mismatches"].append(
                {
                    "kind": "null" if "null" in (ek, gk) else "nonfinite",
                    "object_id": object_id,
                    "expected": _json_value(expected),
                    "actual": _json_value(got),
                }
            )
            continue
        if ek != "finite":
            continue
        error = abs(float(expected) - float(got))
        relative = (
            error / abs(float(expected))
            if expected != 0
            else (0.0 if error == 0 else "Infinity")
        )
        result["max_abs_error"] = max(result["max_abs_error"], error)
        if relative == "Infinity":
            result["max_rel_error"] = "Infinity"
        elif result["max_rel_error"] != "Infinity":
            result["max_rel_error"] = max(result["max_rel_error"], relative)
        tolerance = float(rule.get("absolute_tolerance") or 0) + float(
            rule.get("relative_tolerance") or 0
        ) * abs(float(expected))
        if error > tolerance:
            result["mismatches"].append(
                {
                    "kind": "numeric",
                    "object_id": object_id,
                    "expected": expected,
                    "actual": got,
                    "abs_error": error,
                    "rel_error": relative,
                }
            )
        elif error:
            result["status"] = "pass_tolerance"
    kinds = {item["kind"] for item in result["mismatches"]}
    for kind, status in (
        ("presence", "missing"),
        ("ids", "fail_ids"),
        ("shape", "fail_shape"),
        ("dtype", "fail_dtype"),
        ("null", "fail_null"),
        ("nonfinite", "fail_nonfinite"),
        ("numeric", "fail_numeric"),
    ):
        if kind in kinds:
            result["status"] = status
            break
    return result


def compare_relationships(reference: list[dict], actual: list[dict] | None) -> dict:
    """Compare directed image/object edges without normalizing direction."""
    if actual is None:
        return {
            "status": "unsupported",
            "expected_topology": reference,
            "actual_topology": None,
        }
    keys = ("first_image_id", "first_object_id", "second_image_id", "second_object_id")

    def edge(row: dict) -> tuple:
        return tuple(row.get(key) for key in keys)

    expected, got = Counter(map(edge, reference)), Counter(map(edge, actual))
    return {
        "status": "pass_exact" if expected == got else "fail_topology",
        "expected_topology": reference,
        "actual_topology": actual,
    }


def load_arrays(entry: dict, root: Path = ROOT) -> dict:
    """Load the saved input bytes without regenerating segmentation."""
    import numpy as np

    return {
        Path(path).stem: np.load(root / path, allow_pickle=False)
        for path in entry["sha256"]
    }


def measure_fixture(arrays: dict) -> dict:
    """Run only the two families implemented by cp_measure 0.3.0."""
    import numpy as np
    from cp_measure.core.measureobjectintensity import (  # ty: ignore[import-not-found]
        get_intensity,
    )
    from cp_measure.multimask.measureobjectneighbors import (  # ty: ignore
        measureobjectneighbors,
    )

    out: dict[str, Any] = {
        "Image": {},
        "Cells": {},
        "Nuclei": {},
        "object_ids": {},
        "relationships": None,
        "errors": {},
    }
    mask = arrays.get("mask")
    for scope, label_name in (("Cells", "labels"), ("Nuclei", "nuclei_labels")):
        labels = arrays[label_name]
        ids = [int(x) for x in np.unique(labels) if x > 0]
        out["object_ids"][scope] = ids
        for channel in ("DNA", "Actin"):
            try:
                pixels = (
                    arrays[channel]
                    if mask is None
                    else np.where(mask, arrays[channel], np.nan)
                )
                measured = get_intensity(
                    labels, pixels, edge_measurements=True, legacy=True
                )
                for name, data in measured.items():
                    out[scope][f"{name}_{channel}"] = {
                        "ids": ids,
                        "values": data.tolist(),
                        "dtype": str(data.dtype),
                    }
            except Exception as exc:  # Capture a baseline failure per family.
                out["errors"][f"{scope}/{channel}"] = f"{type(exc).__name__}: {exc}"
    try:
        measured = measureobjectneighbors(
            arrays["labels"], arrays["labels"], distance_method="Adjacent"
        )
        ids = [int(x) for x in np.unique(arrays["labels"]) if x > 0]
        for name, data in measured.items():
            values = np.asarray(data)
            out["Cells"][name] = {
                "ids": ids,
                "values": values.tolist(),
                "dtype": str(values.dtype),
            }
    except Exception as exc:
        out["errors"]["Neighbors"] = f"{type(exc).__name__}: {exc}"
    return out


def compare_fixture(entry: dict, catalog: dict, root: Path = ROOT) -> list[dict]:
    reference = json.loads((root / entry["result"]).read_text())
    measured = measure_fixture(load_arrays(entry, root))
    rows = []
    for (scope, name), mapping in catalog.items():
        base = {
            "fixture": entry["name"],
            "object_scope": scope,
            "feature": name,
            "module": mapping["module"],
            "cp_measure_feature": mapping["cp_measure_feature"],
        }
        if mapping["status"] == "unsupported":
            rows.append(
                {
                    **base,
                    "status": "unsupported",
                    "reason": mapping["reason"],
                    "mismatches": [],
                    "max_abs_error": None,
                    "max_rel_error": None,
                }
            )
            continue
        candidate = measured[scope].get(mapping["cp_measure_feature"])
        if candidate is None and measured["errors"]:
            family = (
                "Neighbors"
                if mapping["module"] == "MeasureObjectNeighbors"
                else f"{scope}/{name.rsplit('_', 1)[-1]}"
            )
            if family in measured["errors"]:
                rows.append(
                    {
                        **base,
                        "status": "blocked",
                        "reason": measured["errors"][family],
                        "mismatches": [],
                        "max_abs_error": None,
                        "max_rel_error": None,
                    }
                )
                continue
        ref_values = reference["features"][scope][name]
        ref_ids = [1] if scope == "Image" else reference["object_ids"][scope]
        result = compare_feature(
            [ref_values] if scope == "Image" else ref_values,
            None if candidate is None else candidate["values"],
            ref_ids,
            None if candidate is None else candidate["ids"],
            mapping["rule"],
            actual_dtype=None if candidate is None else candidate["dtype"],
            expected_dtype=mapping["dtype"],
        )
        reason = None
        if (
            result["status"] == "fail_dtype"
            and name == "Neighbors_NumberOfNeighbors_Adjacent"
        ):
            reason = (
                "cp_measure allocates neighbor_count as float64; "
                "original declares int64"
            )
        elif result["status"] == "fail_ids" and entry["name"] == "sparse_ids":
            reason = (
                "cp_measure emits present IDs only; original retains "
                "positions 1..max(label)"
            )
        elif result["status"] == "fail_nonfinite" and entry["name"] == "nonfinite":
            reason = (
                "cp_measure get_intensity filters nonfinite pixels; "
                "original preserves them in several reductions"
            )
        elif (
            result["status"] == "fail_numeric"
            and name.startswith("Location_MaxIntensity_")
            and entry["name"] == "one_constant"
        ):
            reason = (
                "Tied maximum position differs; cp_measure source selects "
                "first C-order maximum"
            )
        elif result["status"].startswith("fail"):
            reason = "unknown"
        rows.append({**base, **result, "reason": reason})
    # Relationships are a separate, required part of the declared scope.
    rows.append(
        {
            "fixture": entry["name"],
            "object_scope": "relationship",
            "feature": "Neighbors/Cells/Cells/directed",
            "module": "MeasureObjectNeighbors",
            "cp_measure_feature": None,
            **compare_relationships(reference["relationships"], None),
            "reason": "cp_measure does not emit relationship rows",
            "mismatches": [],
            "max_abs_error": None,
            "max_rel_error": None,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/baseline/cp_measure-parity.json",
    )
    args = parser.parse_args()
    source = os.environ.get("CP_ANYWHERE_CP_MEASURE_SOURCE")
    if source:
        commit = subprocess.check_output(
            ["git", "-C", source, "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != "cdf693187e3dcaf3d067da7b58bdd4ab8896f91e":
            raise ValueError(f"Unexpected cp_measure source commit: {commit}")
    catalog = load_catalog()
    manifest = json.loads((ROOT / "fixtures/reference/manifest.json").read_text())
    rows = [
        row for entry in manifest["fixtures"] for row in compare_fixture(entry, catalog)
    ]
    summary = dict(Counter(row["status"] for row in rows))
    payload = {
        "reference": "CellProfiler 4.2.8",
        "baseline": "cp_measure 0.3.0",
        "feature_count": len(catalog),
        "fixtures": len(manifest["fixtures"]),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "fixture",
                "object_scope",
                "feature",
                "module",
                "cp_measure_feature",
                "status",
                "reason",
                "max_abs_error",
                "max_rel_error",
                "mismatches_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get("mismatches", []))
                    if key == "mismatches_json"
                    else row.get(key)
                    for key in writer.fieldnames
                }
            )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
