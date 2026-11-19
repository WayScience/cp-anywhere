"""Replay a bounded CPJUMP1 group against the pinned CellProfiler oracle.

Supply the unmodified illum_without_batchfile.cppipe and load_data.csv from
CPJUMP1 plate BR00117010. Keep the source TIFFs for A01 sites 1 and 2 and A02
site 1 in --images, using their original basenames. The caller supplies raw
arrays in place of LoadData. CellProfiler writes the reference NPY files.

The CSV is under cpg0000-jump-pilot/source_4/workspace/load_data_csv/
2020_11_04_CPJUMP1/BR00117010/ in the cellpainting-gallery S3 bucket. Use
the URL_* columns for the exact TIFFs. Run the compare subcommand with
the pinned CellProfiler source and isolated reference interpreter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as TiffImage

if TYPE_CHECKING:
    from collections.abc import Sequence

PIPELINE_SHA256 = "d54afdfdae18d8107412673ac2e21f76c86da63dbf84c76b81bcbbc475081cf0"
REFERENCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
JUMP_COMMIT = "6ba3fcd1495d9e844e4607373a568641981ffcd8"
PLATE = "BR00117010"
FIELDS = (("A01", "1"), ("A01", "2"), ("A02", "1"))
CHANNELS = (
    "OrigDNA",
    "OrigER",
    "OrigRNA",
    "OrigAGP",
    "OrigMito",
    "OrigBrightfield",
    "OrigLowZBF",
    "OrigHighZBF",
)
ATOL = 5e-7


def _cycles(table: Path, images: Path) -> list[dict[str, Path]]:
    with table.open(newline="", encoding="utf-8") as stream:
        rows = {
            (row["Metadata_Well"], row["Metadata_Site"]): row
            for row in csv.DictReader(stream)
            if row["Metadata_Plate"] == PLATE
        }
    return [
        {key: images / rows[field][f"URL_{key}"].rsplit("/", 1)[1] for key in CHANNELS}
        for field in FIELDS
    ]


def _read_tiff(path: str | Path) -> np.ndarray:
    with TiffImage.open(path) as image:
        return np.array(image)


def _reference(pipeline_path: Path, manifest: Path, target: Path) -> None:
    # Keep CellProfiler and its dependencies in the isolated oracle process.
    from cellprofiler_core import (  # ty: ignore[import-not-found]
        preferences,
    )
    from cellprofiler_core.image import (  # ty: ignore[import-not-found]
        Image,
        ImageSetList,
    )
    from cellprofiler_core.measurement import (  # ty: ignore[import-not-found]
        Measurements,
    )
    from cellprofiler_core.object import (  # ty: ignore[import-not-found]
        ObjectSet,
    )
    from cellprofiler_core.pipeline import (  # ty: ignore[import-not-found]
        Pipeline,
    )
    from cellprofiler_core.workspace import (  # ty: ignore[import-not-found]
        Workspace,
    )

    preferences.set_headless()
    preferences.set_default_output_directory(str(target))
    pipeline = Pipeline()
    pipeline.load(str(pipeline_path))
    modules = [
        module for module in pipeline.modules() if module.module_name != "LoadData"
    ]
    cycles = json.loads(manifest.read_text(encoding="utf-8"))
    grouping = {"Plate": PLATE}
    image_numbers = list(range(1, len(cycles) + 1))
    image_sets = ImageSetList()
    outputs: dict[str, str] = {}
    with closing(Measurements()) as measurements:
        first = Workspace(pipeline, None, None, None, measurements, image_sets)
        for module in modules:
            module.prepare_group(first, grouping, image_numbers)
        workspaces = []
        for index, file_map in enumerate(cycles):
            measurements.next_image_set(index + 1)
            measurements.add_image_measurement("Metadata_Plate", PLATE)
            image_set = image_sets.get_image_set(index)
            for name, path in file_map.items():
                image_set.add(name, Image(_read_tiff(path), dimensions=2))
            objects = ObjectSet()
            workspaces = [
                Workspace(
                    pipeline, module, image_set, objects, measurements, image_sets
                )
                for module in modules
            ]
            for module, workspace in zip(modules, workspaces):
                module.run(workspace)
        for module, workspace in zip(modules, workspaces):
            module.post_group(workspace, grouping)
            if module.module_name == "SaveImages":
                outputs[module.image_name.value] = Path(
                    module.get_filename(workspace, check_overwrite=False)
                ).name
    packages = ("cellprofiler-core", "centrosome", "scikit-image", "scipy", "numpy")
    (target / "oracle.json").write_text(
        json.dumps(
            {"outputs": outputs, "packages": {name: version(name) for name in packages}}
        ),
        encoding="utf-8",
    )


def _differences(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"Expected {expected.shape}/{expected.dtype}, "
            f"got {actual.shape}/{actual.dtype}"
        )
    deviation = np.abs(actual.astype(np.float64) - expected)
    if not np.isfinite(deviation).all():
        raise AssertionError("Expected finite images")
    return {
        "shape": actual.shape,
        "dtype": actual.dtype.name,
        "max_abs_error": float(deviation.max()),
        "different_pixels": int(np.count_nonzero(deviation)),
        "pixels_outside_tolerance": int(np.count_nonzero(deviation > ATOL)),
    }


def _compare(args: argparse.Namespace) -> None:
    from cp_anywhere.plate_illumination import (
        load_plate_illumination_pipeline,
    )

    if hashlib.sha256(args.pipeline.read_bytes()).hexdigest() != PIPELINE_SHA256:
        raise ValueError("Pipeline does not match the pinned CPJUMP1 file")
    git = ("git", "-C", str(args.reference_source))
    if (
        subprocess.check_output((*git, "rev-parse", "HEAD"), text=True).strip()
        != REFERENCE_COMMIT
    ):
        raise ValueError("Wrong reference source commit")
    if subprocess.check_output((*git, "status", "--porcelain"), text=True).strip():
        raise ValueError("Reference source checkout is not clean")
    files = _cycles(args.load_data, args.images.resolve())
    plan = load_plate_illumination_pipeline(args.pipeline)
    arrays = [
        {name: _read_tiff(path) for name, path in cycle.items()} for cycle in files
    ]
    with TemporaryDirectory(prefix="plate-illumination-") as directory:
        root = Path(directory)
        manifest = root / "cycles.json"
        manifest.write_text(json.dumps(files, default=str), encoding="utf-8")
        environment = dict(os.environ, PYTHONPATH=str(args.reference_source.resolve()))
        subprocess.run(
            [
                str(args.reference_python),
                __file__,
                "oracle",
                str(args.pipeline.resolve()),
                str(manifest),
                directory,
            ],
            env=environment,
            check=True,
            timeout=180,
        )
        oracle = json.loads((root / "oracle.json").read_text(encoding="utf-8"))
        actual = plan.run(arrays)
        if tuple(actual) != tuple(oracle["outputs"]):
            raise AssertionError("Different named outputs or output order")
        outputs = {
            name: _differences(actual[name], np.load(root / file, allow_pickle=False))
            for name, file in oracle["outputs"].items()
        }
    report = {
        "jump_commit": JUMP_COMMIT,
        "pipeline_sha256": PIPELINE_SHA256,
        "cellprofiler_commit": REFERENCE_COMMIT,
        "reference_packages": oracle["packages"],
        "load_data_sha256": hashlib.sha256(args.load_data.read_bytes()).hexdigest(),
        "plate": PLATE,
        "well_site": FIELDS,
        "input_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for cycle in files
            for path in cycle.values()
        },
        "absolute_tolerance": ATOL,
        "outputs": outputs,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(result["pixels_outside_tolerance"] for result in outputs.values()):
        raise AssertionError(f"Output exceeds tolerance; see {args.report}")
    print(f"Compared {len(outputs)} outputs within atol={ATOL:g}; see {args.report}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    compare = commands.add_parser("compare")
    for name in (
        "pipeline",
        "load-data",
        "images",
        "reference-python",
        "reference-source",
        "report",
    ):
        compare.add_argument(f"--{name}", type=Path, required=True)
    oracle = commands.add_parser("oracle")
    for name in ("pipeline", "manifest", "target"):
        oracle.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "oracle":
        _reference(args.pipeline, args.manifest, args.target)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
