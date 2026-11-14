# ty: ignore[import-not-found]
"""Run original CellProfiler 4.2.8 modules against fixed array fixtures.

This script runs with the separate environment installed by reference/install.sh.
It never imports cp_anywhere or substitutes another measurement engine.
"""

# The reference API is dynamically typed, and fixture literals define the test cases.
# Set headless mode before importing CellProfiler measurement modules.
# ruff: noqa: ANN001, ANN201, PLR0912, PLR0913, PLR0915, PLR2004, C901, E402
# The project venv intentionally omits packages in the separate reference venv.

import argparse
import csv
import hashlib
import io
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import cellprofiler
import cellprofiler_core.preferences
import numpy as np
import yaml

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measureimageintensity import MeasureImageIntensity
from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity
from cellprofiler.modules.measureobjectneighbors import (
    D_ADJACENT,
    MeasureObjectNeighbors,
)
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
PIPELINE_PATH = "reference/pipelines/intensity_neighbors.cppipe"
CHANNELS = ("DNA", "Actin")
OBJECT_SET = "Cells"
OBJECT_SETS = ("Cells", "Nuclei")
REAL_FIXTURE_NAMES = ("real_cell_cc0", "real_bbbc007")
# Populated by fixture_arrays() for fixtures with real, independently
# segmented nuclei (not the generic single-pixel-per-object placeholder
# every other fixture uses); see produce()'s measure() call.
EXPLICIT_NUCLEI = {}


class IndentedDumper(yaml.SafeDumper):
    """Use list indentation accepted by the repository YAML linter."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, indentless=False)


def encoded(value):
    """Preserve JSON-distinct null, NaN, and signed infinity values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return {"nonfinite": "NaN"}
        if np.isposinf(value):
            return {"nonfinite": "+Infinity"}
        if np.isneginf(value):
            return {"nonfinite": "-Infinity"}
    return value


def canonical(data):
    """Serialize results without timestamps or platform-specific float text."""
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def fixture_arrays():
    """Yield arrays and the fixed segmentation method for each workload."""
    z = np.zeros((8, 8), dtype=np.float64)
    yield (
        "empty_image",
        {"DNA": z, "Actin": z.copy()},
        np.zeros_like(z, dtype=np.int32),
        None,
        (1.0, 1.0),
        "supplied",
    )

    one = np.zeros((8, 8), dtype=np.int32)
    one[2:6, 2:6] = 1
    yield (
        "empty_mask",
        {"DNA": np.ones((8, 8)), "Actin": np.full((8, 8), 2.0)},
        one,
        np.zeros((8, 8), dtype=bool),
        (1.0, 1.0),
        "supplied",
    )
    yield (
        "one_constant",
        {"DNA": np.ones((8, 8)), "Actin": np.zeros((8, 8))},
        one,
        None,
        (1.0, 1.0),
        "supplied",
    )

    y, x = np.indices((12, 12))
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:5, 1:5] = 1
    labels[5:9, 1:5] = 2
    labels[5:9, 5:9] = 3
    yield (
        "touching_gradient",
        {"DNA": (x + 2 * y) / 33.0, "Actin": (x * y).astype(np.uint16)},
        labels,
        None,
        (0.25, 0.50),
        "supplied",
    )
    partial_mask = (x < 7) | ((y >= 5) & (x < 10))
    yield (
        "partial_mask",
        {"DNA": (x + 2 * y) / 33.0, "Actin": (x * y).astype(np.uint16)},
        labels,
        partial_mask,
        (0.25, 0.50),
        "supplied",
    )

    labels = np.zeros((13, 13), dtype=np.int32)
    labels[:4, :4] = 1
    labels[1:3, 1:3] = 0  # A hole in the border object.
    labels[6:8, 5:7] = 2
    labels[10:12, 10:12] = 2  # One ID in disconnected regions.
    labels[7:11, 7:10] = 3
    sparse = np.zeros((13, 13), dtype=np.float64)
    sparse[0, 0] = 1.0
    sparse[7, 5] = 0.5
    yield (
        "holes_disconnected_border",
        {"DNA": sparse, "Actin": np.full((13, 13), 255, dtype=np.uint8)},
        labels,
        None,
        (1.0, 1.0),
        "supplied",
    )

    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[4:6, 4:6] = 5
    labels[7:9, 7:9] = 16
    yield (
        "sparse_ids",
        {
            "DNA": np.arange(100, dtype=np.float64).reshape(10, 10) / 99,
            "Actin": np.ones((10, 10)),
        },
        labels,
        None,
        (1.0, 1.0),
        "supplied",
    )

    labels = np.zeros((7, 8, 8), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 1
    labels[4:6, 1:4, 1:4] = 2
    labels[2:5, 5:7, 5:7] = 3
    zz, yy, xx = np.indices(labels.shape)
    yield (
        "volume_3d",
        {"DNA": (zz + yy + xx) / 20.0, "Actin": (zz * yy + xx).astype(np.uint16)},
        labels,
        None,
        (2.0, 0.5, 0.5),
        "supplied",
    )

    labels = np.zeros((9, 9), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[5:8, 5:8] = 2
    dna = np.arange(81, dtype=np.float64).reshape(9, 9) / 80
    dna[2, 2] = np.nan
    dna[6, 6] = np.inf
    yield (
        "nonfinite",
        {"DNA": dna, "Actin": np.ones((9, 9))},
        labels,
        None,
        (1.0, 1.0),
        "supplied",
    )

    # scikit-image 0.21.0 bundles this CC0 quantitative phase microscopy image.
    from skimage.data import cell
    from skimage.measure import label

    source = cell()
    crop = source[260:356, 210:306].copy()
    candidates = label(crop > np.percentile(crop, 85), connectivity=1)
    labels = np.zeros(crop.shape, dtype=np.int32)
    next_id = 1
    for old_id in range(1, int(candidates.max()) + 1):
        region = candidates == old_id
        if 4 <= int(region.sum()) <= 2000 and next_id <= 20:
            labels[region] = next_id
            next_id += 1
    assert next_id > 2, "The real crop must contain multiple threshold objects"
    yield (
        "real_cell_cc0",
        {"DNA": crop, "Actin": 255 - crop},
        labels,
        None,
        (0.107, 0.107),
        "threshold at 85th percentile; connected labels; 4..2000 pixels; first 20",
    )

    # BBBC007v1 (Jones et al. 2005; Broad Bioimage Benchmark Collection),
    # CC0. Real DNA + Actin channels, real independently-segmented cell
    # and nuclei labels (not the generic single-pixel placeholder every
    # other fixture uses) -- see reference/vendor/bbbc007/README.md for
    # full provenance and the derivation from the dataset's own manual
    # outline ground truth.
    vendor = ROOT / "reference/vendor/bbbc007"
    bbbc_dna = np.load(vendor / "dna.npy")
    bbbc_actin = np.load(vendor / "actin.npy")
    bbbc_cells = np.load(vendor / "cells.npy")
    bbbc_nuclei = np.load(vendor / "nuclei.npy")
    EXPLICIT_NUCLEI["real_bbbc007"] = bbbc_nuclei
    yield (
        "real_bbbc007",
        {"DNA": bbbc_dna, "Actin": bbbc_actin},
        bbbc_cells,
        None,
        (1.0, 1.0),
        "BBBC007v1 manual outline ground truth; connected labels; "
        "see reference/vendor/bbbc007/README.md",
    )

    for side, grid, name in ((96, 8, "grid_medium"), (192, 16, "grid_large")):
        yy, xx = np.indices((side, side))
        labels = (yy // 12 * grid + xx // 12 + 1).astype(np.int32)
        labels[(yy % 12 >= 8) | (xx % 12 >= 8)] = 0
        dna = ((xx + yy) % 37).astype(np.float64) / 36
        actin = ((xx * 3 + yy * 7) % 256).astype(np.uint8)
        yield (
            name,
            {"DNA": dna, "Actin": actin},
            labels,
            None,
            (1.0, 1.0),
            "supplied grid: 8x8-pixel objects in 12x12-pixel cells",
        )


def build_pipeline():
    pipeline = Pipeline()
    image = MeasureImageIntensity()
    image.set_module_num(1)
    image.images_list.value = ", ".join(CHANNELS)
    image.wants_objects.value = False
    image.wants_percentiles.value = True
    image.percentiles.value = "10, 50, 90"
    pipeline.add_module(image)
    intensity = MeasureObjectIntensity()
    intensity.set_module_num(2)
    intensity.images_list.value = ", ".join(CHANNELS)
    intensity.objects_list.value = ", ".join(OBJECT_SETS)
    pipeline.add_module(intensity)
    neighbors = MeasureObjectNeighbors()
    neighbors.set_module_num(3)
    neighbors.object_name.value = OBJECT_SET
    neighbors.neighbors_name.value = OBJECT_SET
    neighbors.distance_method.value = D_ADJACENT
    neighbors.wants_excluded_objects.value = False
    pipeline.add_module(neighbors)
    stream = io.StringIO()
    pipeline.dump(stream)
    return stream.getvalue().encode()


def load_pipeline(payload):
    pipeline = Pipeline()
    pipeline.load(io.StringIO(payload.decode()))
    assert [module.module_name for module in pipeline.modules()] == [
        "MeasureImageIntensity",
        "MeasureObjectIntensity",
        "MeasureObjectNeighbors",
    ]
    return pipeline


def measure(name, channels, labels, mask, spacing, pipeline, nuclei=None):
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    for channel in CHANNELS:
        image_set.add(
            channel,
            Image(
                channels[channel],
                mask=mask,
                convert=False,
                dimensions=labels.ndim,
                spacing=spacing,
            ),
        )
    object_set = ObjectSet()
    if nuclei is None:
        nuclei = np.zeros_like(labels)
        for object_id in np.unique(labels):
            if object_id == 0:
                continue
            coords = np.argwhere(labels == object_id)
            nuclei[tuple(coords[len(coords) // 2])] = object_id
    for set_name, set_labels in (("Cells", labels), ("Nuclei", nuclei)):
        objects = Objects()
        objects.segmented = set_labels
        objects.parent_image = image_set.get_image(CHANNELS[0])
        object_set.add_objects(objects, set_name)
    measurements = Measurements()
    measurements.group_number = 1
    measurements.group_index = 1
    declared = {}
    for module in pipeline.modules():
        workspace = Workspace(
            pipeline, module, image_set, object_set, measurements, image_set_list
        )
        module.run(workspace)
        for scope, feature, dtype in module.get_measurement_columns(pipeline):
            key = (scope, feature)
            assert key not in declared, key
            declared[key] = (module.module_name, dtype)
    features = {"Image": {}, **{set_name: {} for set_name in OBJECT_SETS}}
    for (scope, feature), _ in sorted(declared.items()):
        assert feature in measurements.get_feature_names(scope), (name, scope, feature)
        data = measurements.get_current_measurement(scope, feature)
        if scope == "Image":
            features[scope][feature] = encoded(data)
        else:
            values = np.asarray(data).tolist()
            features[scope][feature] = [encoded(value) for value in values]
    for scope in features:
        actual = set(measurements.get_feature_names(scope))
        expected = {
            feature for declared_scope, feature in declared if declared_scope == scope
        }
        allowed_metadata = (
            {"Group_Number", "Group_Index"} if scope == "Image" else set()
        )
        assert actual - expected == allowed_metadata, (name, scope, actual - expected)
    groups = measurements.get_relationship_groups()
    assert len(groups) <= 1, (name, groups)
    if groups:
        group = groups[0]
        assert (
            group.module_number,
            group.relationship,
            group.object_name1,
            group.object_name2,
        ) == (3, "Neighbors", OBJECT_SET, OBJECT_SET)
    rows = measurements.get_relationships(3, "Neighbors", OBJECT_SET, OBJECT_SET)
    relationships = [
        {
            "first_image_id": name,
            "first_object_id": int(row[1]),
            "second_image_id": name,
            "second_object_id": int(row[3]),
        }
        for row in rows
    ]
    relationships.sort(
        key=lambda row: (row["first_object_id"], row["second_object_id"])
    )
    return (
        {
            "image_id": name,
            "image_number": 1,
            "object_ids": {
                "Cells": list(range(1, int(labels.max()) + 1)),
                "Nuclei": list(range(1, int(nuclei.max()) + 1)),
            },
            "features": features,
            "relationships": relationships,
        },
        declared,
        nuclei,
    )


def produce():
    files = {}
    pipeline_bytes = build_pipeline()
    files[PIPELINE_PATH] = pipeline_bytes
    pipeline = load_pipeline(pipeline_bytes)
    fixtures = []
    observed = {}
    csv_rows = {"image": [], "objects": [], "relationships": []}
    for name, channels, labels, mask, spacing, segmentation in fixture_arrays():
        assert set(channels) == set(CHANNELS)
        assert all(channel.shape == labels.shape for channel in channels.values())
        result, declared, nuclei = measure(
            name,
            channels,
            labels,
            mask,
            spacing,
            pipeline,
            nuclei=EXPLICIT_NUCLEI.get(name),
        )
        sha = {}
        arrays = {**channels, "labels": labels, "nuclei_labels": nuclei}
        if mask is not None:
            arrays["mask"] = mask
        for array_name, array in arrays.items():
            family = "real" if name in REAL_FIXTURE_NAMES else "synthetic"
            path = "fixtures/{}/{}/{}.npy".format(family, name, array_name)
            payload = npy_bytes(array)
            files[path] = payload
            sha[path] = hashlib.sha256(payload).hexdigest()
        result_path = "fixtures/reference/{}.json".format(name)
        files[result_path] = canonical(result)
        fixtures.append(
            {
                "name": name,
                "dimensions": labels.ndim,
                "shape": list(labels.shape),
                "channel_names": list(CHANNELS),
                "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                "mask_present": mask is not None,
                "spacing": list(spacing),
                "segmentation": segmentation,
                "pipeline": PIPELINE_PATH,
                "environment_lock": "reference/source-lock.json",
                "result": result_path,
                "sha256": sha,
            }
        )
        for key, (module, dtype) in declared.items():
            record = observed.setdefault(
                key,
                {
                    "module": module,
                    "dtype": dtype,
                    "dimensions": set(),
                    "nonfinite": set(),
                    "null": False,
                },
            )
            assert (record["module"], record["dtype"]) == (module, dtype)
            record["dimensions"].add(labels.ndim)
            values = result["features"][key[0]][key[1]]
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, dict) and "nonfinite" in value:
                    record["nonfinite"].add(value["nonfinite"])
                if value is None:
                    record["null"] = True
        for feature, value in result["features"]["Image"].items():
            csv_rows["image"].append(
                [name, 1, feature, json.dumps(value, allow_nan=False)]
            )
        for set_name in OBJECT_SETS:
            for feature, values in result["features"][set_name].items():
                for object_id, value in enumerate(values, 1):
                    csv_rows["objects"].append(
                        [
                            name,
                            1,
                            set_name,
                            object_id,
                            feature,
                            json.dumps(value, allow_nan=False),
                        ]
                    )
        for row in result["relationships"]:
            csv_rows["relationships"].append(
                [
                    name,
                    3,
                    "Neighbors",
                    OBJECT_SET,
                    row["first_object_id"],
                    OBJECT_SET,
                    row["second_object_id"],
                ]
            )
    features = []
    for (scope, feature), record in sorted(observed.items()):
        discrete = record["dtype"] == "integer"
        module_settings = {
            "MeasureImageIntensity": {
                "images": list(CHANNELS),
                "wants_objects": False,
                "wants_percentiles": True,
                "percentiles": [10, 50, 90],
            },
            "MeasureObjectIntensity": {
                "images": list(CHANNELS),
                "objects": list(OBJECT_SETS),
            },
            "MeasureObjectNeighbors": {
                "objects": OBJECT_SET,
                "neighbors": OBJECT_SET,
                "distance_method": D_ADJACENT,
                "wants_excluded_objects": False,
            },
        }[record["module"]]
        features.append(
            {
                "name": feature,
                "module": record["module"],
                "object_scope": scope,
                "dimensionality": sorted(record["dimensions"]),
                "inputs": ["image", "mask"]
                if scope == "Image"
                else (
                    ["labels"]
                    if record["module"] == "MeasureObjectNeighbors"
                    else ["image", "labels", "mask"]
                ),
                "settings": {"pipeline": PIPELINE_PATH, **module_settings},
                "dtype": "int64" if discrete else "float64",
                "missing_value_semantics": (
                    "Nonfinite values use explicit JSON markers. "
                    "Sparse object IDs retain array positions."
                ),
                "observed_nonfinite": sorted(record["nonfinite"]),
                "observed_null": record["null"],
                "cp_measure_status": "unknown_m2",
                "rust_status": "not_started",
                "wasm_status": "not_started",
                "comparison": {
                    "mode": "exact" if discrete else "numeric",
                    "absolute_tolerance": 0 if discrete else 1e-10,
                    "relative_tolerance": 0 if discrete else 1e-9,
                    "rationale": "Discrete ID/count"
                    if discrete
                    else (
                        "Float64 reduction order; tighten per feature "
                        "during parity review"
                    ),
                },
            }
        )
    inventory = {
        "cellprofiler_version": "4.2.8",
        "source_commit": SOURCE_COMMIT,
        "features": features,
        "relationships": [
            {
                "module": "MeasureObjectNeighbors",
                "name": "Neighbors",
                "first_scope": OBJECT_SET,
                "second_scope": OBJECT_SET,
                "direction": "both directed rows",
                "topology": "same object set, Adjacent mode",
                "fields": [
                    "first_image_id",
                    "first_object_id",
                    "second_image_id",
                    "second_object_id",
                ],
                "comparison": "exact rows after key sort",
            }
        ],
    }
    files["spec/features.yaml"] = yaml.dump(
        inventory,
        Dumper=IndentedDumper,
        sort_keys=True,
        width=70,
        explicit_start=True,
    ).encode()
    headers = {
        "image": ["image_id", "image_number", "feature", "value_json"],
        "objects": [
            "image_id",
            "image_number",
            "object_scope",
            "object_id",
            "feature",
            "value_json",
        ],
        "relationships": [
            "image_id",
            "module_number",
            "relationship",
            "first_scope",
            "first_object_id",
            "second_scope",
            "second_object_id",
        ],
    }
    for table, rows in csv_rows.items():
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers[table])
        writer.writerows(rows)
        files["fixtures/reference/{}.csv".format(table)] = stream.getvalue().encode()
    manifest = {
        "cellprofiler_version": "4.2.8",
        "source_commit": SOURCE_COMMIT,
        "cp_measure_commit": "cdf693187e3dcaf3d067da7b58bdd4ab8896f91e",
        "environment_lock": "reference/source-lock.json",
        "requirements_sha256": hashlib.sha256(
            (ROOT / "reference/requirements.lock").read_bytes()
        ).hexdigest(),
        "pipeline": PIPELINE_PATH,
        "pipeline_sha256": hashlib.sha256(pipeline_bytes).hexdigest(),
        "fixtures": fixtures,
        "tables": {
            table: "fixtures/reference/{}.csv".format(table) for table in csv_rows
        },
        "feature_count": len(features),
        "relationship_type_count": 1,
        "real_sources": [
            {
                "name": "skimage.data.cell",
                "package": "scikit-image 0.21.0",
                "license": "CC0",
                "source_file_sha256": hashlib.sha256(
                    Path(__import__("skimage.data").data.__file__)
                    .parent.joinpath("cell.png")
                    .read_bytes()
                ).hexdigest(),
                "crop": "rows 260:356, columns 210:306",
                "fixture": "real_cell_cc0",
            },
            {
                "name": "BBBC007v1 (Drosophila Kc167 cells, A9 p10)",
                "package": "Broad Bioimage Benchmark Collection",
                "license": "CC0",
                "crop": "rows 256:384, columns 192:320 of a 450x450 field",
                "fixture": "real_bbbc007",
                "provenance": "reference/vendor/bbbc007/README.md",
            },
        ],
    }
    files["fixtures/reference/manifest.json"] = canonical(manifest)
    return files, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify", action="store_true", help="Replay and compare all committed bytes"
    )
    args = parser.parse_args()
    source_root = Path(cellprofiler.__file__).resolve().parent.parent
    assert (
        subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        == SOURCE_COMMIT
    )
    assert cellprofiler.__version__ == "4.2.8"
    assert version("cellprofiler-core") == "4.2.8"
    files, manifest = produce()
    for name, payload in files.items():
        path = ROOT / name
        if args.verify:
            assert path.read_bytes() == payload, "Reference replay mismatch: " + name
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    digest = hashlib.sha256(b"".join(files[name] for name in sorted(files))).hexdigest()
    print(
        json.dumps(
            {
                "mode": "verify" if args.verify else "write",
                "files": len(files),
                "fixtures": len(manifest["fixtures"]),
                "features": manifest["feature_count"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
