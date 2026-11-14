"""Comparator contracts independent of the optional baseline environment."""

# ruff: noqa: PLR2004

import json
from pathlib import Path

from scripts.parity.compare_cp_measure import (
    compare_feature,
    compare_relationships,
    load_catalog,
)


def test_finite_and_nonfinite_are_separate() -> None:
    rule = {"mode": "numeric", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-9}
    result = compare_feature(
        [1.0, "NaN", "Infinity", None],
        [1.0 + 1e-11, "NaN", "-Infinity", None],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        rule,
    )
    assert result["status"] == "fail_nonfinite"
    assert result["mismatches"][0]["object_id"] == 3
    assert result["mismatches"][0]["kind"] == "nonfinite"


def test_ids_and_presence_are_checked_before_values() -> None:
    rule = {"mode": "exact", "absolute_tolerance": 0, "relative_tolerance": 0}
    result = compare_feature([2, 3], [2], [1, 4], [1], rule)
    assert result["status"] == "fail_ids"
    assert any(item.get("object_id") == 4 for item in result["mismatches"])
    assert compare_feature([1], None, [1], None, rule)["status"] == "missing"


def test_catalog_covers_exact_original_inventory() -> None:
    catalog = load_catalog()
    assert len(catalog) == 119
    assert len(set(catalog)) == 119
    assert sum(row["status"] == "unsupported" for row in catalog.values()) >= 28


def test_known_unsupported_image_feature() -> None:
    catalog = load_catalog()
    assert catalog[("Image", "Intensity_TotalIntensity_DNA")]["status"] == "unsupported"


def test_reordered_ids_pass_but_missing_id_fails() -> None:
    rule = {"mode": "exact", "absolute_tolerance": 0, "relative_tolerance": 0}
    assert (
        compare_feature([5, 8], [8, 5], [1, 2], [2, 1], rule)["status"] == "pass_exact"
    )


def test_directed_relationship_topology() -> None:
    expected = [{"first_object_id": 1, "second_object_id": 2}]
    reversed_edge = [{"first_object_id": 2, "second_object_id": 1}]
    assert compare_relationships(expected, reversed_edge)["status"] == "fail_topology"
    assert compare_relationships(expected, None)["status"] == "unsupported"


def test_observed_cp_measure_mismatches_are_cataloged() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/baseline/cp_measure-parity.json"
    )
    report = json.loads(path.read_text())
    rows = report["rows"]
    assert len(rows) == 12 * 120
    assert any(
        r["fixture"] == "one_constant"
        and r["feature"] == "Location_MaxIntensity_X_DNA"
        and r["status"] == "fail_numeric"
        for r in rows
    )
    assert any(
        r["fixture"] == "nonfinite"
        and r["feature"] == "Intensity_IntegratedIntensity_DNA"
        and r["status"] == "fail_nonfinite"
        for r in rows
    )
