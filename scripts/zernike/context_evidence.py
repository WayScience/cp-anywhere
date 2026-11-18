# ty: ignore[import-not-found]
# The CP-only and native-only modes run in separate Python environments.
# ruff: noqa: PLC0415
"""Capture real CP context-sensitive Zernikes, or check strict native parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/size_shape/zernike_context"
TARGETS = {"Nuclei": (6, 34), "Cells": (6,)}
RTOL = 1e-10
ATOL = 1e-10


def label_digest(labels: np.ndarray) -> str:
    """Hash exact row-major labels with a platform-independent integer dtype."""
    return hashlib.sha256(labels.astype("<u4").tobytes()).hexdigest()


def run_cp(labels: np.ndarray) -> dict[str, list[float]]:
    """Run the actual module, not a native-derived reference implementation."""
    # Reference-only dependencies are absent from the normal project environment.
    import cellprofiler_core.preferences
    from cellprofiler.modules.measureobjectsizeshape import (
        MeasureObjectSizeShape,
    )
    from cellprofiler_core.image import ImageSetList
    from cellprofiler_core.measurement import Measurements
    from cellprofiler_core.object import Objects, ObjectSet
    from cellprofiler_core.pipeline import Pipeline
    from cellprofiler_core.workspace import Workspace

    cellprofiler_core.preferences.set_headless()
    module = MeasureObjectSizeShape()
    module.objects_list.value = "Objects"
    module.calculate_zernikes.value = True
    module.calculate_advanced.value = False
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    objects = Objects()
    objects.segmented = labels
    object_set = ObjectSet()
    object_set.add_objects(objects, "Objects")
    image_sets = ImageSetList()
    measurements = Measurements()
    try:
        module.run(
            Workspace(
                pipeline,
                module,
                image_sets.get_image_set(0),
                object_set,
                measurements,
                image_sets,
            )
        )
        return {
            name: measurements.get_current_measurement("Objects", name).tolist()
            for name in sorted(measurements.get_feature_names("Objects"))
            if name.startswith("AreaShape_Zernike_")
        }
    finally:
        measurements.close()


def circle_evidence(labels: np.ndarray, targets: tuple[int, ...]) -> dict:
    """Observe the real SciPy tie decisions without replacing their behavior."""
    import scipy.ndimage
    from centrosome.cpmorphology import minimum_enclosing_circle

    decisions = {str(oid): [] for oid in targets}
    original = scipy.ndimage.minimum_position

    def observe(values: np.ndarray, groups: np.ndarray, indexes: np.ndarray) -> list:
        result = original(values, groups, indexes)
        for oid in targets:
            positions = np.flatnonzero(groups == oid)
            if positions.size:
                tied = positions[values[positions] == values[positions].min()]
                if tied.size > 1:
                    chosen = int(result[list(indexes).index(oid)][0])
                    decisions[str(oid)].append(
                        {
                            "candidate_count": int(values.size),
                            "tied_positions": tied.tolist(),
                            "selected_position": chosen,
                            "selected_tie_rank": int(np.flatnonzero(tied == chosen)[0]),
                        }
                    )
        return result

    indexes = np.arange(1, int(labels.max()) + 1, dtype=np.int32)
    with patch("scipy.ndimage.minimum_position", side_effect=observe):
        centers, radii = minimum_enclosing_circle(labels, indexes)
    evidence = {}
    for oid in targets:
        center, radius = centers[oid - 1], radii[oid - 1]
        rows, cols = np.where(labels == oid)
        x = (cols - center[1]) / radius
        y = (rows - center[0]) / radius
        inside = x * x + y * y <= 1
        evidence[str(oid)] = {
            "center": center.tolist(),
            "radius": float(radius),
            "radius_hex": float(radius).hex(),
            "area": int(rows.size),
            "included_pixels": int(inside.sum()),
            "excluded_pixels": np.column_stack((rows[~inside], cols[~inside])).tolist(),
            "minimum_position_ties": decisions[str(oid)],
        }
    return evidence


def capture(arrays: dict[str, np.ndarray]) -> dict:
    """Capture both full-field and same-frame isolated CP results."""
    import cellprofiler

    result = {
        "versions": {
            "CellProfiler": cellprofiler.__version__,
            **{
                name: importlib.metadata.version(name)
                for name in ("centrosome", "numpy", "scipy", "scikit-image")
            },
        },
        "scopes": {},
    }
    for scope, targets in TARGETS.items():
        labels = arrays[f"{scope}_labels"].astype(np.uint32)
        full = run_cp(labels)
        circles = circle_evidence(labels, targets)
        objects = {}
        for oid in targets:
            isolated = (labels == oid).astype(np.uint32)
            objects[str(oid)] = {
                "full_field": {name: values[oid - 1] for name, values in full.items()},
                "isolated": {
                    name: values[0] for name, values in run_cp(isolated).items()
                },
                "full_circle": circles[str(oid)],
                "isolated_circle": circle_evidence(isolated, (1,))["1"],
            }
        result["scopes"][scope] = {
            "shape": list(labels.shape),
            "label_count": int(labels.max()),
            "labels_sha256": label_digest(labels),
            "objects": objects,
        }
    return result


def check_native(arrays: dict[str, np.ndarray], expected: dict) -> int:
    """Return nonzero for strict full-field parity failures, never waive them."""
    from cp_anywhere import measure

    labels = {scope: arrays[f"{scope}_labels"] for scope in TARGETS}
    actual = measure(
        {"DNA": np.zeros(next(iter(labels.values())).shape)},
        labels,
        families=["object_size_shape"],
        size_shape_calculate_zernikes=True,
        size_shape_calculate_advanced=False,
    ).features
    failures = 0
    for scope, targets in TARGETS.items():
        for oid in targets:
            oracle = expected["scopes"][scope]["objects"][str(oid)]["full_field"]
            errors = {
                name: abs(np.asarray(actual[scope][name])[oid - 1] - value)
                for name, value in oracle.items()
                if not np.isclose(
                    np.asarray(actual[scope][name])[oid - 1],
                    value,
                    rtol=RTOL,
                    atol=ATOL,
                )
            }
            failures += len(errors)
            print(
                f"{scope} object {oid}: {len(errors)} mismatches; "
                f"max absolute error {max(errors.values(), default=0):.17g}"
            )
    print(f"Strict full-field parity: {failures} failures (rtol={RTOL}, atol={ATOL})")
    return int(failures != 0)


def main() -> int:
    """Capture, reproduce, or strictly compare the checked-in evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path)
    parser.add_argument("--source-oracle", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--native-check", action="store_true")
    args = parser.parse_args()
    source = args.source_npz or FIXTURE / "labels.npz"
    with np.load(source, allow_pickle=False) as archive:
        arrays = {f"{scope}_labels": archive[f"{scope}_labels"] for scope in TARGETS}
    if args.native_check:
        return check_native(arrays, json.loads((FIXTURE / "oracle.json").read_text()))
    observed = capture(arrays)
    if args.source_oracle:
        pilot = json.loads(args.source_oracle.read_text())["features"]
        for scope, targets in TARGETS.items():
            for oid in targets:
                for name, value in observed["scopes"][scope]["objects"][str(oid)][
                    "full_field"
                ].items():
                    np.testing.assert_allclose(
                        value, pilot[scope][name][oid - 1], rtol=RTOL, atol=ATOL
                    )
        observed["source"] = {
            "field": args.source_npz.stem,
            "pilot_oracle_verified": True,
        }
    if args.write:
        if args.source_npz is None or args.source_oracle is None:
            parser.error("--write requires --source-npz and --source-oracle")
        FIXTURE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FIXTURE / "labels.npz", **arrays)
        (FIXTURE / "oracle.json").write_text(json.dumps(observed, indent=2) + "\n")
        print("Wrote full, unchanged labels and independently measured CP evidence")
    else:
        expected = json.loads((FIXTURE / "oracle.json").read_text())
        assert observed["versions"] == expected["versions"], (
            "Reference environment changed"
        )
        assert observed["scopes"] == expected["scopes"], "CP evidence changed"
        print(
            "Real CP exactly reproduces all recorded scores, circles, and tie decisions"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
