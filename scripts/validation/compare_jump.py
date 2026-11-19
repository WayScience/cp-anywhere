"""Compare a bounded cp-anywhere run with saved CPJUMP1 reference fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from .pilot_checks import compare_features

if TYPE_CHECKING:
    from cp_anywhere import MeasurementResult

LABELS = {
    "Nuclei": {
        "segmented": "labels",
        "unedited_segmented": "unedited_labels",
        "small_removed_segmented": "small_removed_labels",
    },
    "Cells": {"segmented": "labels", "small_removed_segmented": "small_removed_labels"},
    "Cytoplasm": {"segmented": "labels"},
}
MEASUREMENT_BLOCKS = tuple(str(i) for i in range(19, 28))
FOCUSED_MEASUREMENTS = {
    "correlation": ("19", "colocalization_steps"),
    "granularity": ("20", "granularity_steps"),
    "radial": ("25", "intensity_distribution_steps"),
    "texture": ("27", "texture_steps"),
}
FIELD_NAMES = ("BR00118045_A01_1", "BR00118045_A01_2")


def _selected_fields(site: str | None) -> tuple[str, ...]:
    return FIELD_NAMES if site is None else (FIELD_NAMES[int(site) - 1],)


def _measurement_scope(focus: str | None) -> tuple[tuple[str, ...], dict]:
    if focus is None:
        return MEASUREMENT_BLOCKS, {}
    block, keep = FOCUSED_MEASUREMENTS[focus]
    steps = {steps for _, steps in FOCUSED_MEASUREMENTS.values()}
    return (block,), {key: [] for key in (steps | {"measurement_steps"}) - {keep}}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _archived_measurements(directory: Path, target: Path) -> None:
    # Only the independent reference environment needs h5py.
    import h5py  # ty: ignore[import-not-found]

    schema = _json(directory / "schema.json")
    requested = {
        (scope, name)
        for block in MEASUREMENT_BLOCKS
        for scope, name, *_ in schema["declared_by_module"][block]
    }
    requested.update((scope, "Number_Object_Number") for scope in LABELS)
    with h5py.File(directory / "measurements.h5", "r") as source:
        (run,) = source["Measurements"].values()
        arrays = {}
        for scope, name in sorted(requested):
            observed = schema["observed"].get(scope, {}).get(name)
            if observed is None:
                continue
            column = run[scope][name]
            (row,) = column["index"][:]
            image, first, last = row
            if image != 1:
                raise ValueError(f"Expected image number 1, got {image}")
            data = np.asarray(column["data"][first:last]).reshape(observed["shape"])
            digest = hashlib.sha256(data.tobytes(order="C")).hexdigest()
            if (
                digest != observed["sha256_c_order"]
                or data.dtype.str != observed["dtype"]
            ):
                raise ValueError(f"Reference data changed: {scope}.{name}")
            arrays[f"{scope}/{name}"] = data
    # Keys always contain a scope separator, so none can bind allow_pickle.
    np.savez(target, **arrays)  # ty: ignore[invalid-argument-type]


def _inputs(directory: Path) -> dict[str, np.ndarray]:
    from PIL import Image

    arrays = {}
    for name, info in _json(directory / "inputs.json")["assets"].items():
        path = Path(info["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Input bytes changed: {path}")
        if name.startswith("Illum"):
            arrays[name] = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                arrays[name] = np.array(image)
        raw = arrays[name]
        if (
            list(raw.shape) != info["raw"]["shape"]
            or raw.dtype.str != info["raw"]["dtype"]
            or hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            != info["raw"]["sha256_c_order"]
        ):
            raise ValueError(f"Decoded raw pixels changed: {path}")
    return arrays


def _compare_labels(directory: Path, results: dict) -> dict:
    comparison = {}
    manifest = _json(directory / "result.json")["labels"]
    for scope, layers in LABELS.items():
        for layer, record in manifest[scope].items():
            if layer not in layers:
                comparison[f"{scope}/{layer}"] = "not exposed by cp-anywhere"
                continue
            expected = np.load(directory / record["file"], allow_pickle=False)
            if (
                hashlib.sha256(expected.tobytes(order="C")).hexdigest()
                != record["sha256_c_order"]
            ):
                raise ValueError(f"Reference label bytes changed: {scope}.{layer}")
            actual = getattr(results[scope], layers[layer])
            comparison[f"{scope}/{layer}"] = {
                "different_pixels": int(np.count_nonzero(expected != actual))
                if actual.shape == expected.shape
                else None,
                "same_ids": bool(
                    np.array_equal(np.unique(expected), np.unique(actual))
                ),
            }
    return comparison


def compare(
    directory: Path,
    features: Path,
    results: dict,
    measurement: MeasurementResult,
    *,
    blocks: tuple[str, ...] = MEASUREMENT_BLOCKS,
) -> dict:
    """Compare only columns requested by the analysis measurement blocks."""
    actual = measurement.features
    ids = measurement.object_ids
    schema = _json(directory / "schema.json")
    requested = {
        (scope, name)
        for block in blocks
        for scope, name, *_ in schema["declared_by_module"][block]
    }
    wanted: dict[str, dict] = {}
    received: dict[str, dict] = {}
    with np.load(features, allow_pickle=False) as archive:
        unobserved = []
        for scope, name in sorted(requested):
            key = f"{scope}/{name}"
            if key not in archive:
                unobserved.append(key)
                continue
            wanted.setdefault(scope, {})[name] = archive[key]
            if name in actual.get(scope, {}):
                received.setdefault(scope, {})[name] = actual[scope][name]
        object_ids = {
            scope: bool(
                np.array_equal(ids[scope], archive[f"{scope}/Number_Object_Number"])
            )
            for scope in LABELS
        }
    report = compare_features(received, wanted)
    extra = {
        scope: sorted(set(columns) - set(wanted.get(scope, {})))
        for scope, columns in actual.items()
    }
    return {
        "labels": _compare_labels(directory, results),
        "object_ids_match": object_ids,
        "measurement_blocks": list(blocks),
        "columns_requested": len(requested),
        "reference_columns_not_realized": unobserved,
        "extra_columns": extra,
        "measurement_values_compared": report["scalars_checked"],
        "discrepancies_by_family_and_kind": dict(
            sorted(
                Counter(
                    f"{entry['feature'].partition('.')[2].partition('_')[0]}/"
                    f"{entry['kind']}"
                    for entry in report["mismatches"]
                ).items()
            )
        ),
        "mismatches": report["mismatches"],
    }


def _failed(item: dict) -> bool:
    return bool(
        item["mismatches"]
        or item["reference_columns_not_realized"]
        or any(item["extra_columns"].values())
        or not all(item["object_ids_match"].values())
        or any(
            not record["same_ids"] or record["different_pixels"] != 0
            for record in item["labels"].values()
            if isinstance(record, dict)
        )
    )


def _run(args: argparse.Namespace) -> int:
    from cp_anywhere import inspect_pipeline, load_pipeline

    path = args.root / "normalized.cppipe"
    pipeline = load_pipeline(path)
    unsupported = [
        m.name for m in inspect_pipeline(path) if m.status.startswith("unsupported")
    ]
    if unsupported != ["OverlayOutlines", "OverlayOutlines"]:
        raise ValueError(f"Unexpected unsupported modules: {unsupported}")
    blocks, overrides = _measurement_scope(args.focus)
    pipeline = replace(pipeline, **overrides)
    results = {}
    with TemporaryDirectory(prefix="compare-jump-") as scratch:
        for field_name in _selected_fields(args.site):
            directory = args.root / "output" / field_name
            reference = Path(scratch) / f"{directory.name}.npz"
            subprocess.run(
                [
                    str(args.reference_python),
                    "-m",
                    "scripts.validation.compare_jump",
                    "reference",
                    str(directory),
                    str(reference),
                ],
                check=True,
                timeout=120,
            )
            primary, derived, measurement = pipeline.run(_inputs(directory))
            if measurement is None:
                raise ValueError("Expected measurements")
            item = compare(
                directory,
                reference,
                primary | derived,
                measurement,
                blocks=blocks,
            )
            results[directory.name] = item
            print(directory.name, item["discrepancies_by_family_and_kind"], flush=True)
            args.report.write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    return int(any(_failed(item) for item in results.values()))


def main() -> int:
    """Return nonzero on discrepancies, while retaining the observed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    local = modes.add_parser("compare")
    for flag in ("root", "reference-python", "report"):
        local.add_argument(f"--{flag}", required=True, type=Path)
    local.add_argument("--focus", choices=FOCUSED_MEASUREMENTS)
    local.add_argument("--site", choices=("1", "2"))
    archive = modes.add_parser("reference")
    archive.add_argument("directory", type=Path)
    archive.add_argument("target", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        return _run(args)
    _archived_measurements(args.directory, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
