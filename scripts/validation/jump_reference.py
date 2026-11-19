# ty: ignore[import-not-found]
"""Run the pinned CPJUMP1 analysis on two public fields with real CellProfiler."""

# The reference API is intentionally absent from the project environment.
# ruff: noqa: ANN401, E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cellprofiler
import numpy as np
from cellprofiler_core.preferences import set_headless
from PIL import Image as PillowImage

set_headless()

from cellprofiler_core.image import FileImage, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.utilities.java import start_java, stop_java
from cellprofiler_core.workspace import Workspace

from scripts.reference.generate_granularity import (
    ADAPTER,
    install_compatibility_adapter,
)

JUMP_REPO = "jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1"
JUMP_SHA = "6ba3fcd1495d9e844e4607373a568641981ffcd8"
BATCH = "2020_11_04_CPJUMP1"
PLATE = "BR00118045"
WELL = "A01"
SITES = ("1", "2")
PIPE_URL = (
    f"https://raw.githubusercontent.com/{JUMP_REPO}/{JUMP_SHA}/pipelines/{BATCH}/"
    "CPJUMP1_analysis_without_batchfile_406.cppipe"
)
PIPE_SHA256 = "b2661d63bffc6ea09cc35fdd4a800491e38b9c3a1c5b96d0d477a42e31dc4bae"
GALLERY = "https://cellpainting-gallery.s3.amazonaws.com"
CSV_URL = (
    f"{GALLERY}/cpg0000-jump-pilot/source_4/workspace/load_data_csv/"
    f"{BATCH}/{PLATE}/load_data_with_illum.csv"
)
CSV_SHA256 = "a990e61092efed397d6761c70935d2ad4831ecb5f37f5070edbc51708e0b64e9"
CP_SHA = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
CP_VERSION = "4.2.8"
CHANNELS = ("DNA", "ER", "AGP", "Mito", "RNA", "Brightfield", "HighZBF", "LowZBF")
LABEL_PRODUCERS = {16: "Nuclei", 17: "Cells", 18: "Cytoplasm"}
LAYERS = ("segmented", "unedited_segmented", "small_removed_segmented")
# LoadData/SaveImages/ExportToSpreadsheet are caller I/O; OverlayOutlines is UI.
CALLER = {1, 28, 29, 30, 31, 32}
DISABLED = {11, 12, 13, 33}


def digest(payload: bytes) -> str:
    """Hash unmodified artifact bytes or C-order array bytes."""
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write a strict JSON evidence record."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def download(url: str, path: Path, expected: str | None = None) -> str:
    """Stage public bytes in scratch and return their SHA-256."""
    if not path.exists():
        pending = path.with_suffix(path.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "2",
                "--max-time",
                "90",
                url,
                "-o",
                str(pending),
            ],
            check=True,
        )
        pending.replace(path)
    sha = digest(path.read_bytes())
    if expected is not None and sha != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: {sha}")
    return sha


def versions() -> dict:
    """Verify the pinned reference and record the versions used by this run."""
    if cellprofiler.__file__ is None:
        raise RuntimeError("CellProfiler source checkout not found")
    source = Path(cellprofiler.__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "cellprofiler"],
        text=True,
    )
    dependencies = {
        pkg: version(pkg)
        for pkg in (
            "cellprofiler-core",
            "centrosome",
            "numpy",
            "scipy",
            "scikit-image",
            "h5py",
            "mahotas",
            "python-bioformats",
            "python-javabridge",
            "Pillow",
        )
    }
    if commit != CP_SHA or dirty or dependencies["cellprofiler-core"] != CP_VERSION:
        raise RuntimeError(
            "Use the clean, pinned CellProfiler 4.2.8 reference environment"
        )
    return {
        "cellprofiler_source": str(source),
        "cellprofiler_commit": commit,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def module_state(pipeline: Pipeline) -> list[tuple]:
    """Record CP's loaded values rather than interpreting display strings."""
    return [
        (
            m.module_num,
            m.module_name,
            m.variable_revision_number,
            m.enabled,
            tuple(s.value_text for s in m.settings()),
        )
        for m in pipeline.modules(exclude_disabled=False)
    ]


def resave(root: Path) -> Pipeline:
    """Perform a real CP resave and verify the loaded state round-trips."""
    original = root / "original.cppipe"
    normalized = root / "normalized.cppipe"
    download(PIPE_URL, original, PIPE_SHA256)
    pipeline = Pipeline()
    pipeline.loadtxt(str(original), raise_on_error=True)
    before = module_state(pipeline)
    if [s[0] for s in before] != list(range(1, 34)) or {
        s[0] for s in before if not s[3]
    } != DISABLED:
        raise ValueError("Unexpected module order or disabled blocks")
    with normalized.open("w", encoding="utf-8") as stream:
        pipeline.dump(stream, save_image_plane_details=False)
    reloaded = Pipeline()
    reloaded.loadtxt(str(normalized), raise_on_error=True)
    if before != module_state(reloaded):
        raise ValueError("CP resave changed loaded module state")
    return reloaded


def select_fields(input_dir: Path) -> tuple[int, dict[str, dict[str, str]]]:
    """Select the real published rows by plate, well, and site."""
    source = input_dir / "load_data_with_illum.csv"
    download(CSV_URL, source, CSV_SHA256)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        site: [
            r
            for r in rows
            if (r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
            == (PLATE, WELL, site)
        ]
        for site in SITES
    }
    if any(len(matches) != 1 for matches in selected.values()):
        raise ValueError("Each requested field must match exactly one published row")
    return len(rows), {site: matches[0] for site, matches in selected.items()}


def array_record(array: np.ndarray) -> dict:
    """Identify exact array bytes without changing dtypes or NaNs."""
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256_c_order": digest(array.tobytes(order="C")),
    }


def stage_images(
    row: dict[str, str], input_dir: Path, image_set: Any, *, rescale: bool
) -> dict:
    """Use the same CP FileImage decoder as LoadData, without invoking LoadData."""
    names = {prefix + ch for prefix in ("Orig", "Illum") for ch in CHANNELS}
    if {key[4:] for key in row if key.startswith("URL_")} != names:
        raise ValueError("Published field does not match all sixteen expected inputs")
    assets = {}
    for name in sorted(names):
        remote = urlsplit(row[f"URL_{name}"])
        if (remote.scheme, remote.netloc) != ("s3", "cellpainting-gallery"):
            raise ValueError(f"Unexpected gallery URL: {remote.geturl()}")
        path = input_dir / Path(remote.path).name
        sha = download(GALLERY + remote.path, path)
        if name.startswith("Illum"):
            raw = np.load(path, allow_pickle=False)
        else:
            with PillowImage.open(path) as tiff:
                raw = np.array(tiff)
        image = FileImage(
            name, str(input_dir), path.name, rescale=rescale
        ).provide_image(None)
        expected_shape = (
            int(row["Metadata_ImageSizeY"]),
            int(row["Metadata_ImageSizeX"]),
        )
        if raw.shape != expected_shape or image.pixel_data.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {name}")
        image_set.add(name, image)
        assets[name] = {
            "source": remote.geturl(),
            "path": str(path),
            "sha256": sha,
            "raw": array_record(raw),
            "decoded": array_record(image.pixel_data),
            "cp_scale": float(image.scale),
        }
    return assets


def snapshot_labels(objects: ObjectSet, output: Path, scope: str) -> dict:
    """Save exact label IDs at their producer, before measurement modules."""
    obj = objects.get_objects(scope)
    layers = {}
    for layer in LAYERS:
        array = getattr(obj, layer).copy()
        path = output / f"{scope}.{layer}.npy"
        np.save(path, array, allow_pickle=False)
        layers[layer] = {
            **array_record(array),
            "file": path.name,
            "ids": np.unique(array[array > 0]).tolist(),
        }
    return layers


def schema(measurements: Measurements) -> dict:
    """Record raw realized column names, shapes and dtypes as CP exposes them."""
    return {
        scope: {
            key: array_record(
                np.asarray(measurements.get_current_measurement(scope, key))
            )
            for key in sorted(measurements.get_feature_names(scope))
        }
        for scope in sorted(measurements.get_object_names())
    }


def save_relationships(measurements: Measurements, output: Path) -> list[dict]:
    """Save CP's relationship records with unmodified image and object IDs."""
    records = []
    for group in measurements.get_relationship_groups():
        key = (
            group.module_number,
            group.relationship,
            group.object_name1,
            group.object_name2,
        )
        edges = measurements.get_relationships(*key)
        path = output / f"relationships.{'.'.join(map(str, key))}.npy"
        np.save(path, edges, allow_pickle=False)
        records.append(
            {
                "key": key,
                "file": path.name,
                "columns": edges.dtype.descr,
                **array_record(edges),
            }
        )
    return records


def run_field(
    pipeline: Pipeline, row: dict[str, str], input_dir: Path, output: Path
) -> None:
    """Run the enabled scientific blocks in file order on a shared workspace."""
    output.mkdir()
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    objects = ObjectSet()
    # Use the original LoadData rescale setting but provide the field ourselves.
    rescale = pipeline.modules()[0].rescale.value
    illumination = pipeline.module(14)
    if illumination.each_or_all.value != "Each":
        raise ValueError("Bounded replay requires the published per-image illumination")
    assets = stage_images(row, input_dir, image_set, rescale=rescale)
    write_json(output / "inputs.json", {"row": row, "assets": assets})
    measurements = Measurements(filename=str(output / "measurements.h5"))
    labels, declared, executed = {}, {}, []
    try:
        for module in pipeline.modules():
            num = module.module_num
            if num in CALLER:
                continue
            print(
                f"site {row['Metadata_Site']} module {num}: {module.module_name}",
                flush=True,
            )
            workspace = Workspace(
                pipeline, module, image_set, objects, measurements, image_set_list
            )
            module.run(workspace)
            executed.append(num)
            if num in LABEL_PRODUCERS:
                scope = LABEL_PRODUCERS[num]
                labels[scope] = snapshot_labels(objects, output, scope)
            declared[num] = module.get_measurement_columns(pipeline)
        write_json(
            output / "schema.json",
            {"declared_by_module": declared, "observed": schema(measurements)},
        )
        write_json(
            output / "result.json",
            {
                "executed": executed,
                "labels": labels,
                "relationships": save_relationships(measurements, output),
            },
        )
    finally:
        measurements.close()


def main() -> None:
    """Expose the bounded reference pilot as a standalone command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "work_dir", type=Path, help="scratch root for all inputs and outputs"
    )
    args = parser.parse_args()
    root = args.work_dir.resolve()
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    environment = versions()
    pipeline = resave(root)
    csv_rows, fields = select_fields(input_dir)
    write_json(
        output_dir / "provenance.json",
        {
            "environment": environment,
            "pipeline_url": PIPE_URL,
            "original_sha256": PIPE_SHA256,
            "normalized_sha256": digest((root / "normalized.cppipe").read_bytes()),
            "csv_url": CSV_URL,
            "csv_sha256": CSV_SHA256,
            "csv_rows": csv_rows,
            "plate": PLATE,
            "well": WELL,
            "sites": SITES,
            "caller_managed": sorted(CALLER),
            "disabled": sorted(DISABLED),
            "granularity_compatibility": ADAPTER,
        },
    )
    install_compatibility_adapter()
    start_java()
    try:
        for site, row in fields.items():
            run_field(pipeline, row, input_dir, output_dir / f"{PLATE}_{WELL}_{site}")
    finally:
        stop_java()


if __name__ == "__main__":
    main()
