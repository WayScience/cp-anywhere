"""Compare pinned-oracle replays without treating platform rounding as new science.

Inputs, label layers, provenance, IDs, counts, relationships and settings stay
exact. Measurement outputs use the unchanged per-feature rules in features.yaml.
Segmentation's float32 log/Li calculations and float64 spline interpolation vary
across macOS arm64 and Linux x86_64 even with identical pinned dependencies:
observed maximum map drift is 6.94e-8, with every label byte unchanged. The replay
limit below is deliberately tighter than the native-port parity limit (1e-5).
Golden bytes and their integrity checks are never rewritten or rounded here.

This module must also run under the oracle's Python 3.9, not only project Python.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

SEGMENTATION_REPLAY_ATOL = 1e-7


def _assert_value(
    expected: object,
    actual: object,
    path: str,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Keep structure/types/nonfinite markers exact; tolerate finite floats only."""
    assert type(expected) is type(actual), (
        f"{path}: type {type(expected)} != {type(actual)}"
    )
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys(), f"{path}: keys differ"
        for key, value in expected.items():
            _assert_value(value, actual[key], f"{path}/{key}", atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual), f"{path}: lengths differ"
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_value(left, right, f"{path}/{index}", atol=atol, rtol=rtol)
    elif isinstance(expected, float):
        assert isinstance(actual, float)
        assert math.isfinite(expected) and math.isfinite(actual), (
            f"{path}: nonfinite float"
        )
        error = abs(expected - actual)
        limit = atol + rtol * abs(expected)
        assert error <= limit, (
            f"{path}: expected {expected!r}, actual {actual!r}, "
            f"abs_error={error!r}, limit={limit!r}"
        )
    else:
        assert expected == actual, f"{path}: expected {expected!r}, actual {actual!r}"


def _feature_value(expected: object, actual: object, path: str, rule: dict) -> None:
    numeric = rule["mode"] == "numeric"
    _assert_value(
        expected,
        actual,
        path,
        atol=rule["absolute_tolerance"] if numeric else 0.0,
        rtol=rule["relative_tolerance"] if numeric else 0.0,
    )


def _measurement_json(expected: dict, actual: dict, path: str, rules: dict) -> None:
    left, right = expected.pop("features"), actual.pop("features")
    _assert_value(expected, actual, path)
    assert left.keys() == right.keys(), f"{path}/features: scopes differ"
    for scope, columns in left.items():
        assert columns.keys() == right[scope].keys(), (
            f"{path}/features/{scope}: keys differ"
        )
        for name, value in columns.items():
            location = f"{path}/features/{scope}/{name}"
            assert (scope, name) in rules, f"{location}: unknown feature"
            _feature_value(value, right[scope][name], location, rules[scope, name])


def _measurement_csv(expected: bytes, actual: bytes, path: str, rules: dict) -> None:
    left = csv.DictReader(io.StringIO(expected.decode()))
    right = csv.DictReader(io.StringIO(actual.decode()))
    _assert_value(left.fieldnames, right.fieldnames, path + "/header")
    left_rows, right_rows = list(left), list(right)
    assert len(left_rows) == len(right_rows), f"{path}: row counts differ"
    for index, (want, got) in enumerate(zip(left_rows, right_rows)):
        location = f"{path}/row/{index}"
        left_value, right_value = want.pop("value_json"), got.pop("value_json")
        _assert_value(want, got, location)
        scope, feature = want.get("object_scope", "Image"), want["feature"]
        assert (scope, feature) in rules, f"{location}: unknown feature"
        _feature_value(
            json.loads(left_value),
            json.loads(right_value),
            location + "/value_json",
            rules[scope, feature],
        )


def verify_measurement_replay(root: Path, files: dict[str, bytes]) -> list[str]:
    """Verify every generated file; return paths whose bytes differ numerically."""
    inventory = yaml.safe_load((root / "spec/features.yaml").read_text())
    rules = {
        (feature["object_scope"], feature["name"]): feature["comparison"]
        for feature in inventory["features"]
    }
    differences = []
    for name, payload in files.items():
        path = root / name
        assert path.exists(), f"{name}: missing reference replay file"
        expected = path.read_bytes()
        if expected == payload:
            continue
        if name.startswith("fixtures/reference/") and name.endswith(".json"):
            if name.endswith("/manifest.json"):
                assert expected == payload, f"{name}: byte mismatch"
            _measurement_json(json.loads(expected), json.loads(payload), name, rules)
        elif name in {"fixtures/reference/image.csv", "fixtures/reference/objects.csv"}:
            _measurement_csv(expected, payload, name, rules)
        else:
            assert expected == payload, f"{name}: byte mismatch"
        differences.append(name)
    return differences


def _threshold_fields(expected: dict, actual: dict, path: str) -> None:
    assert expected.keys() == actual.keys(), f"{path}: keys differ"
    for field in ("threshold", "guide_threshold"):
        if field in expected:
            _assert_value(
                expected.pop(field),
                actual.pop(field),
                f"{path}/{field}",
                atol=SEGMENTATION_REPLAY_ATOL,
            )
    _assert_value(expected, actual, path)


def _segmentation_json(
    root: Path, files: dict[str, bytes], expected: dict, actual: dict, path: str
) -> None:
    if path.endswith("/manifest.json"):
        left, right = expected.pop("cases"), actual.pop("cases")
        _assert_value(expected, actual, path)
        assert len(left) == len(right), f"{path}/cases: lengths differ"
        for index, (want, got) in enumerate(zip(left, right)):
            _threshold_fields(want, got, f"{path}/cases/{index}")
        return
    left, right = expected.pop("sha256"), actual.pop("sha256")
    assert left.keys() == right.keys(), f"{path}/sha256: keys differ"
    for name, digest in left.items():
        location = f"{path}/sha256/{name}"
        assert name in files, f"{location}: missing generated payload"
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, (
            f"{location}: committed checksum mismatch"
        )
        assert hashlib.sha256(files[name]).hexdigest() == right[name], (
            f"{location}: replay checksum mismatch"
        )
        # The map's numeric values, shape and dtype are checked separately below.
        # Hash strings are NOT numerically compared or silently discarded.
        if not name.endswith("/threshold_map.npy"):
            _assert_value(digest, right[name], location)
    _threshold_fields(expected, actual, path)


def _threshold_map(expected: bytes, actual: bytes, path: str) -> None:
    left = np.load(io.BytesIO(expected), allow_pickle=False)
    right = np.load(io.BytesIO(actual), allow_pickle=False)
    assert left.dtype == right.dtype and left.dtype.kind == "f", (
        f"{path}: dtype differs"
    )
    assert left.shape == right.shape, f"{path}: shape differs"
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right)), (
        f"{path}: nonfinite map value"
    )
    np.testing.assert_allclose(
        right,
        left,
        rtol=0,
        atol=SEGMENTATION_REPLAY_ATOL,
        equal_nan=False,
        err_msg=path,
    )


def verify_segmentation_replay(root: Path, files: dict[str, bytes]) -> list[str]:
    """Verify exact arrays/metadata and tightly bounded numeric threshold outputs."""
    differences = []
    for name, payload in files.items():
        path = root / name
        assert path.exists(), f"{name}: missing reference replay file"
        expected = path.read_bytes()
        if name.startswith("fixtures/segmentation/") and name.endswith(".json"):
            _segmentation_json(
                root, files, json.loads(expected), json.loads(payload), name
            )
        elif name.startswith("fixtures/segmentation/") and name.endswith(
            "/threshold_map.npy"
        ):
            _threshold_map(expected, payload, name)
        else:
            assert expected == payload, f"{name}: byte mismatch"
        if expected != payload:
            differences.append(name)
    return differences
