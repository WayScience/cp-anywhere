"""Run an independent, bounded BBBC007 pilot against real CellProfiler.

Run as a module from the checkout root. Full source arrays, labels, and
exports go to --work-dir; --report receives the small reviewable summary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import cp_anywhere as cp
from scripts.validation.bbbc007_data import prepare_inputs, sha256
from scripts.validation.pilot_checks import compare_features, within_two_pixels

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
CHANNELS = ("DNA", "Actin")


def read_images(field: dict, directory: Path) -> dict[str, np.ndarray]:
    """Load the exact normalized inputs shared with the reference process."""
    return {
        name: np.load(directory / field[name], allow_pickle=False) for name in CHANNELS
    }


def finite_ratio(numerator: float, denominator: float) -> float | None:
    """Represent undefined QC explicitly rather than treating it as perfect."""
    return float(numerator / denominator) if denominator > 0 else None


def foreground_score(labels: np.ndarray, manual: np.ndarray) -> dict:
    """Measure foreground overlap; this is not an instance-matching score."""
    predicted = labels > 0
    truth = manual > 0
    overlap = int(np.count_nonzero(predicted & truth))
    total = int(np.count_nonzero(predicted) + np.count_nonzero(truth))
    return {
        "foreground_dice": finite_ratio(2 * overlap, total),
        "manual_enclosed_objects": int(manual.max()),
        "predicted_objects": int(np.count_nonzero(np.unique(labels))),
    }


def boundary_score(labels: np.ndarray, outline: np.ndarray) -> dict:
    """Score internal cell boundaries within two pixels of the manual outline.

    A relevant pixel differs from a neighboring foreground label and has
    no background pixel in its eight-neighborhood. The two-pixel disk is
    Euclidean. Empty relevant sets are reported as undefined, not 100%.
    """
    rows, cols = labels.shape
    padded = np.pad(labels, 1)
    near_background = np.zeros(labels.shape, dtype=bool)
    boundary = np.zeros(labels.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            adjacent = padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
            near_background |= adjacent == 0
            boundary |= (adjacent != labels) & (adjacent > 0)
    relevant = (labels > 0) & boundary & ~near_background
    hits = int(np.count_nonzero(relevant & within_two_pixels(outline)))
    count = int(np.count_nonzero(relevant))
    return {
        "relevant_pixels": int(count),
        "within_two_pixels": int(hits),
        "fraction": finite_ratio(hits, count),
    }


def background_variation(image: np.ndarray, manual: np.ndarray) -> float | None:
    """Spatial CV of tile means over manual-background pixels; diagnostic only."""
    tile_means = []
    for rows in np.array_split(np.arange(image.shape[0]), 4):
        for cols in np.array_split(np.arange(image.shape[1]), 4):
            pixels = image[np.ix_(rows, cols)]
            background = manual[np.ix_(rows, cols)] == 0
            if background.any():
                tile_means.append(float(pixels[background].mean()))
    values = np.asarray(tile_means)
    return (
        finite_ratio(float(values.std()), float(values.mean())) if values.size else None
    )


def quality_metrics(
    field: dict, inputs: Path, raw: dict, corrected: dict, labels: dict
) -> dict:
    """Report annotation agreement and correction effects without assuming benefit."""
    metrics = {"channels": {}}
    for name in CHANNELS:
        before = raw[name]
        after = corrected[name]
        record: dict = {
            "input_saturated_fraction": float(np.mean(before >= 1)),
            "output_saturated_fraction": float(np.mean(after >= 1)),
            "mean_before": float(before.mean()),
            "mean_after": float(after.mean()),
        }
        if not field.get("skip_manual_evaluation"):
            manual = np.load(
                inputs / field[f"{name}_manual_labels"], allow_pickle=False
            )
            scope = "Nuclei" if name == "DNA" else "Cells"
            record["segmentation"] = foreground_score(labels[scope], manual)
            record["background_spatial_cv_before"] = background_variation(
                before, manual
            )
            record["background_spatial_cv_after"] = background_variation(after, manual)
            if name == "Actin":
                outline = np.load(inputs / field["Actin_outline"], allow_pickle=False)
                record["internal_boundary_agreement"] = boundary_score(
                    labels[scope], outline
                )
        if "reference_fields" in field:
            reference = np.load(
                inputs / field["reference_fields"][name], allow_pickle=False
            )
            record["reference_mae_before"] = float(np.abs(before - reference).mean())
            record["reference_mae_after"] = float(np.abs(after - reference).mean())
        metrics["channels"][name] = record
    return metrics


def verify_exports(result: cp.MeasurementResult, destination: Path) -> dict:
    """Actually write and reload table formats on every held-out field."""
    result.to_csv(destination / "features.csv")
    result.to_parquet(destination / "features.parquet")
    result.to_sqlite(destination / "features.sqlite", if_exists="replace")
    expected = pd.DataFrame(result.rows())
    parquet = pd.read_parquet(destination / "features.parquet")
    csv = pd.read_csv(destination / "features.csv")
    pd.testing.assert_frame_equal(parquet, expected, check_dtype=False)
    pd.testing.assert_frame_equal(
        csv, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
    )
    with sqlite3.connect(destination / "features.sqlite") as connection:
        for scope, columns in result.features.items():
            ids = result.object_ids.get(scope, [])
            data: dict[str, list] = {
                "image_id": [result.image_id] * (1 if scope == "Image" else len(ids))
            }
            if scope != "Image":
                data["object_id"] = list(ids)
            data.update(
                {
                    name: [values] if scope == "Image" else np.asarray(values).tolist()
                    for name, values in columns.items()
                }
            )
            table = f"Per_{scope}".replace('"', '""')
            stored = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
            pd.testing.assert_frame_equal(
                stored, pd.DataFrame(data), check_dtype=False, check_exact=True
            )
    return {"CSV": "passed", "Parquet": "passed", "SQLite": "passed"}


def measure_case(
    images: dict, labels: dict, image_id: str, settings: dict
) -> cp.MeasurementResult:
    """Measure supported scopes without changing sparse cytoplasm IDs."""
    intensity = cp.measure(
        images,
        labels,
        image_id=image_id,
        families=["image_intensity", "object_intensity", "neighbors"],
        neighbors_scopes=["Cells"],
    )
    shape = cp.measure(
        images,
        {name: labels[name] for name in settings["shape_scopes"]},
        image_id=image_id,
        families=["object_size_shape"],
    )
    return cp.MeasurementResult(
        image_id=image_id,
        features={
            scope: {**columns, **shape.features.get(scope, {})}
            for scope, columns in intensity.features.items()
        },
        object_ids=intensity.object_ids,
        relationships=intensity.relationships,
    )


def native_case(
    field: dict, inputs: Path, destination: Path, functions: dict, settings: dict
) -> dict:
    """Exercise correction → primary → secondary → tertiary → features."""
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw = read_images(field, inputs)
    corrected = {
        name: cp.apply_illumination(raw[name], functions[name]) for name in CHANNELS
    }
    nuclei = cp.identify_primary_objects(corrected["DNA"], **settings["primary"])
    cells = cp.identify_secondary_objects(
        corrected["Actin"],
        nuclei.labels,
        primary_labels_in=nuclei.unedited_labels,
        **settings["secondary"],
    )
    cytoplasm = cp.identify_tertiary_objects(
        nuclei.labels, cells.labels, **settings["tertiary"]
    )
    labels = {
        "Nuclei": nuclei.labels,
        "Cells": cells.labels,
        "Cytoplasm": cytoplasm.labels,
    }
    image_id = f"{destination.parent.name}/{field['id']}"
    # Empty cytoplasm regions leave sparse IDs. Keep those IDs and measure
    # their intensities, but do not request unsupported sparse shape columns.
    # CP keeps these corrected images as float32. Our illumination core emits
    # float64 values that are *exactly* the same pixels, but a float64 image
    # uses different NumPy reduction precision in MeasureImageIntensity.
    # Preserve corrected arrays/segmentation as-is; restore CP's dtype only
    # at the measurement boundary (checked for lossless conversion).
    measurement_images = {name: corrected[name].astype(np.float32) for name in CHANNELS}
    if any(
        not np.array_equal(measurement_images[name].astype(np.float64), corrected[name])
        for name in CHANNELS
    ):
        raise ValueError("corrected pixels cannot be represented exactly as float32")
    result = measure_case(measurement_images, labels, image_id, settings)
    np.savez(
        destination.with_suffix(".npz"),
        DNA_corrected=corrected["DNA"],
        Actin_corrected=corrected["Actin"],
        Nuclei_labels=nuclei.labels,
        Cells_labels=cells.labels,
        Cytoplasm_labels=cytoplasm.labels,
        Nuclei_unedited=nuclei.unedited_labels,
        Nuclei_small_removed=nuclei.small_removed_labels,
        Cells_small_removed=cells.small_removed_labels,
    )
    features = {
        scope: {name: np.asarray(value).tolist() for name, value in columns.items()}
        for scope, columns in result.features.items()
    }
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "features": features,
                "object_ids": {
                    name: ids.tolist() for name, ids in result.object_ids.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "field": field["id"],
        "shape": list(raw["DNA"].shape),
        "objects": {
            name: int(np.count_nonzero(np.unique(value)))
            for name, value in labels.items()
        },
        "features_per_scope": {
            name: len(columns) for name, columns in result.features.items()
        },
        "quality": quality_metrics(field, inputs, raw, corrected, labels),
        "exports": verify_exports(result, destination),
        "elapsed_seconds": time.perf_counter() - started,
    }


def native_group(
    group: dict, inputs: Path, destination: Path, settings: dict
) -> list[dict]:
    """Calibrate only on the declared independent fields of one cohort."""
    destination.mkdir(parents=True, exist_ok=True)
    calibration = {field["id"] for field in group["calibration"]}
    held_out = {field["id"] for field in group["validation"]}
    if calibration & held_out:
        raise ValueError(f"calibration leakage: {group['id']}")
    config = {**settings["illumination"], **group.get("illumination", {})}
    functions = {
        name: cp.calculate_illumination_across(
            [
                np.load(inputs / field[name], allow_pickle=False)
                for field in group["calibration"]
            ],
            **config,
        )
        for name in CHANNELS
    }
    if any(
        not np.isfinite(array).all() or np.any(array <= 0)
        for array in functions.values()
    ):
        raise ValueError(f"nonpositive/nonfinite calibration: {group['id']}")
    np.savez(
        destination / "illumination.npz", DNA=functions["DNA"], Actin=functions["Actin"]
    )
    records = []
    for field in group["validation"]:
        print(f"native {group['id']}/{field['id']}", flush=True)
        record = native_case(
            field, inputs, destination / field["id"], functions, settings
        )
        records.append({"group": group["id"], "kind": group["kind"], **record})
    return records


def compare_arrays(actual_path: Path, expected_path: Path) -> dict:
    """Labels are exact; float images use the existing illumination tolerance."""
    mismatches = []
    checked = 0
    with (
        np.load(actual_path, allow_pickle=False) as actual,
        np.load(expected_path, allow_pickle=False) as expected,
    ):
        for name in sorted(set(actual.files) | set(expected.files)):
            if name not in actual.files or name not in expected.files:
                mismatches.append({"array": name, "kind": "presence"})
                continue
            left, right = actual[name], expected[name]
            checked += right.size
            if left.shape != right.shape:
                mismatches.append({"array": name, "kind": "shape"})
                continue
            discrete = np.issubdtype(left.dtype, np.integer)
            matches = (
                np.array_equal(left, right)
                if discrete
                else np.allclose(left, right, rtol=1e-6, atol=1e-6, equal_nan=True)
            )
            if not matches:
                count = int(np.count_nonzero(left != right))
                mismatches.append(
                    {
                        "array": name,
                        "kind": "exact_labels" if discrete else "numeric",
                        "differing_pixels": count,
                    }
                )
    return {"pixels_checked": checked, "mismatches": mismatches}


def compare_pilot(manifest: dict, native: Path, oracle: Path) -> list[dict]:
    """Compare every named case, feature, and diagnostic label layer."""
    records = []
    for group in manifest["groups"]:
        group_id = group["id"]
        functions = compare_arrays(
            native / group_id / "illumination.npz",
            oracle / group_id / "illumination.npz",
        )
        for field in group["validation"]:
            name = field["id"]
            native_stem, oracle_stem = (
                native / group_id / name,
                oracle / group_id / name,
            )
            actual = json.loads(native_stem.with_suffix(".json").read_text())
            expected = json.loads(oracle_stem.with_suffix(".json").read_text())
            features = compare_features(actual["features"], expected["features"])
            # Keep end-to-end discrepancies above. This additional, isolated
            # comparison uses CP's actual saved pixels and labels, so upstream
            # illumination rounding cannot disguise a measurement mismatch.
            with np.load(oracle_stem.with_suffix(".npz"), allow_pickle=False) as data:
                shared = measure_case(
                    # The separate CP shared-input pass explicitly promotes
                    # these same pixels to float64 before measuring.
                    {
                        channel: data[f"{channel}_corrected"].astype(np.float64)
                        for channel in CHANNELS
                    },
                    {
                        scope: data[f"{scope}_labels"]
                        for scope in ("Nuclei", "Cells", "Cytoplasm")
                    },
                    f"{group_id}/{name}",
                    manifest["settings"],
                )
            isolated = compare_features(
                shared.features, expected["features_on_shared_inputs"]
            )
            if actual["object_ids"] != expected["object_ids"]:
                features["mismatches"].append({"kind": "object_ids"})
            arrays = compare_arrays(
                native_stem.with_suffix(".npz"), oracle_stem.with_suffix(".npz")
            )
            records.append(
                {
                    "group": group_id,
                    "field": name,
                    "illumination": functions,
                    "arrays": arrays,
                    "features": features,
                    "measurements_on_reference_inputs": isolated,
                    "identification_layer_features": {
                        "status": "not_compared",
                        "features": {
                            scope: sorted(columns)
                            for scope, columns in expected[
                                "identification_layer_features"
                            ]["features"].items()
                        },
                        "values_recorded_in": str(oracle_stem.with_suffix(".json")),
                    },
                }
            )
    return records


def run_reference(args: argparse.Namespace, manifest: Path) -> None:
    """Invoke the separate pinned process; never import a substitute oracle."""
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"wrong reference source commit: {revision}")
    environment = dict(os.environ, PYTHONPATH=str(args.reference_source))
    command = [
        str(args.reference_python),
        str(ROOT / "scripts/reference/real_data_pilot.py"),
        "--input",
        str(manifest),
        "--output",
        str(args.work_dir / "oracle"),
    ]
    subprocess.run(command, env=environment, check=True, timeout=600)


def main() -> None:
    """Prepare, execute, compare, and write a bounded validation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    manifest_path = prepare_inputs(args.data_dir, args.work_dir / "inputs")
    manifest = json.loads(manifest_path.read_text())
    if args.prepare_only:
        print(manifest_path)
        return
    if not args.native_only and (
        args.reference_python is None or args.reference_source is None
    ):
        parser.error(
            "full validation requires --reference-python and --reference-source"
        )
    for relative, expected in manifest["array_sha256"].items():
        if sha256(manifest_path.parent / relative) != expected:
            raise ValueError(f"prepared input changed: {relative}")
    quality = [
        record
        for group in manifest["groups"]
        for record in native_group(
            group,
            manifest_path.parent,
            args.work_dir / "native" / group["id"],
            manifest["settings"],
        )
    ]
    comparisons = []
    if not args.native_only:
        run_reference(args, manifest_path)
        comparisons = compare_pilot(
            manifest, args.work_dir / "native", args.work_dir / "oracle"
        )
    mismatches = sum(
        len(record[key]["mismatches"])
        for record in comparisons
        for key in (
            "illumination",
            "arrays",
            "features",
            "measurements_on_reference_inputs",
        )
    )
    report = {
        "status": "native_only"
        if args.native_only
        else ("pass" if not mismatches else "fail"),
        "scope": (
            "Public multi-field benchmark pilot; not a full plate or "
            "validated instrument calibration."
        ),
        "dataset": {
            key: manifest[key]
            for key in (
                "dataset",
                "source_url",
                "license",
                "archives",
                "normalization",
                "cohorts",
                "excluded",
            )
        },
        "settings": manifest["settings"],
        "groups": [
            {
                "id": group["id"],
                "kind": group["kind"],
                "calibration": [field["id"] for field in group["calibration"]],
                "validation": [field["id"] for field in group["validation"]],
                "illumination": {
                    **manifest["settings"]["illumination"],
                    **group.get("illumination", {}),
                },
            }
            for group in manifest["groups"]
        ],
        "reference": {"source_commit": SOURCE_COMMIT, "executed": not args.native_only},
        "production_readiness": "not_established",
        "input_manifest_sha256": sha256(manifest_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cp_anywhere": version("cp-anywhere"),
            "numpy": np.__version__,
        },
        "held_out_cases": len(quality),
        "mismatches": mismatches,
        "feature_scalars_checked": sum(
            record["features"]["scalars_checked"] for record in comparisons
        ),
        "quality": quality,
        "comparisons": comparisons,
        "limitations": [
            (
                "No treatment/control metadata or instrument flat-field truth: "
                "real-data QC is diagnostic, not evidence that correction "
                "preserves biological differences."
            ),
            "RGB sources are excluded explicitly rather than silently converted.",
            (
                "Boundary scoring uses eight-neighbor relevant-pixel exclusion "
                "and Euclidean two-pixel matching; do not equate results to the "
                "published benchmark without matching all conventions."
            ),
            (
                "Real-field calibration folds are leave-one-out within the same "
                "source cohort; controlled noise and saturation are labeled "
                "synthetic stress inputs."
            ),
            (
                "Shape measurements exclude Cytoplasm because empty regions "
                "leave sparse IDs, an explicitly unsupported shape setting. "
                "Cytoplasm intensity measurements retain the original IDs."
            ),
            (
                "Location_Center_X/Y/Z from CP identification modules and CP "
                "relationship rows are outside this measurement comparison; "
                "they are reported explicitly rather than counted as passing."
            ),
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "held_out_cases": len(quality),
                "mismatches": mismatches,
                "report": str(args.report),
            }
        )
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
