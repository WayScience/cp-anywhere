"""Run a bounded BBBC007 pilot against the pinned CellProfiler oracle.

This is validation tooling, not a replacement for experiment-specific QC.
Downloaded source data and full arrays stay in the caller's work directory.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def normalized_grayscale(image: np.ndarray) -> np.ndarray:
    """Match float32 image storage without guessing how to strip RGB overlays."""
    expected_ndim = 2
    if image.ndim != expected_ndim or 0 in image.shape:
        raise ValueError("real pilot inputs must be nonempty 2D grayscale arrays")
    return (image.astype(np.float64) / np.iinfo(image.dtype).max).astype(np.float32)


def leave_one_out(cohort: str, fields: Sequence[dict]) -> list[dict]:
    """Use other fields from the same acquisition as calibration, never self."""
    minimum_fields = 2
    if len(fields) < minimum_fields:
        raise ValueError("independent calibration needs at least two fields")
    if len({field["id"] for field in fields}) != len(fields):
        raise ValueError("field IDs must be unique within a cohort")
    ordered = sorted(fields, key=lambda field: field["id"])
    return [
        {
            "id": f"{cohort}--{field['id']}",
            "kind": "real_leave_one_out",
            "calibration": [other for other in ordered if other["id"] != field["id"]],
            "validation": [field],
        }
        for field in ordered
    ]


def manual_labels(outline: np.ndarray) -> np.ndarray:
    """Label enclosed manual-outline interiors using four-neighbor connectivity.

    Border-connected components and the outline pixels themselves are excluded,
    matching the documented conversion used for the vendored BBBC007 crop.
    """
    outline = np.asarray(outline, dtype=bool)
    rows, cols = outline.shape
    free = (~outline).ravel()
    visited = np.zeros(free.shape, dtype=bool)
    labels = np.zeros(free.shape, dtype=np.int32)
    next_label = 0
    for start in np.flatnonzero(free):
        if visited[start]:
            continue
        queue = deque([int(start)])
        visited[start] = True
        component = []
        touches_border = False
        while queue:
            index = queue.popleft()
            component.append(index)
            row, col = divmod(index, cols)
            if row in (0, rows - 1) or col in (0, cols - 1):
                touches_border = True
            for y, x in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if 0 <= y < rows and 0 <= x < cols:
                    neighbor = y * cols + x
                    if free[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        if not touches_border:
            next_label += 1
            labels[component] = next_label
    return labels.reshape(outline.shape)


def within_two_pixels(outline: np.ndarray) -> np.ndarray:
    """Pixels at Euclidean distance at most two from a manual boundary."""
    radius = 2
    rows, cols = outline.shape
    padded = np.pad(np.asarray(outline, dtype=bool), radius)
    nearby = np.zeros((rows, cols), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                nearby |= padded[
                    radius + dy : radius + dy + rows, radius + dx : radius + dx + cols
                ]
    return nearby


def compare_features(actual: Mapping, expected: Mapping) -> dict:
    """Compare schema, shapes, nonfinite kinds, and published numeric tolerances.

    Core measurements use rtol=1e-9/atol=1e-10. Shape measurements use the
    existing shape module's rtol=1e-7/atol=1e-10. Discrete measurements
    remain exact. Upstream illumination differences do not relax these
    checks; compare measurements on identical inputs to isolate stages.
    """
    mismatches = []
    checked = 0
    for scope in sorted(set(actual) | set(expected)):
        actual_columns = actual.get(scope, {})
        expected_columns = expected.get(scope, {})
        for feature in sorted(set(actual_columns) | set(expected_columns)):
            name = f"{scope}.{feature}"
            if feature not in actual_columns or feature not in expected_columns:
                mismatches.append({"feature": name, "kind": "feature_presence"})
                continue
            left = np.asarray(actual_columns[feature], dtype=np.float64)
            right = np.asarray(expected_columns[feature], dtype=np.float64)
            checked += right.size
            if left.shape != right.shape:
                mismatches.append({"feature": name, "kind": "shape"})
                continue
            if not (
                np.array_equal(np.isnan(left), np.isnan(right))
                and np.array_equal(np.isposinf(left), np.isposinf(right))
                and np.array_equal(np.isneginf(left), np.isneginf(right))
            ):
                mismatches.append({"feature": name, "kind": "nonfinite"})
                continue
            discrete = feature == "AreaShape_Area" or feature.startswith(
                "Neighbors_Number"
            )
            if discrete:
                rtol, atol = 0.0, 0.0
            elif feature.startswith("AreaShape_"):
                rtol, atol = 1e-7, 1e-10

            else:
                rtol, atol = 1e-9, 1e-10
            if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
                finite = np.isfinite(left) & np.isfinite(right)
                max_error = (
                    float(np.max(np.abs(left[finite] - right[finite])))
                    if finite.any()
                    else None
                )
                mismatches.append(
                    {"feature": name, "kind": "numeric", "max_abs_error": max_error}
                )
    return {"scalars_checked": checked, "mismatches": mismatches}
