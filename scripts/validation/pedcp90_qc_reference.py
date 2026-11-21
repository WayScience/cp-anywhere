# ty: ignore[import-not-found]
"""Execute the staged QC prefix in the pinned CellProfiler environment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from cellprofiler_core.preferences import set_headless

set_headless()

from cellprofiler_core.constants.workspace import DISPOSITION_SKIP
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.utilities.java import start_java, stop_java
from cellprofiler_core.workspace import Workspace

from .jump_reference import versions
from .pedcp90_qc import (
    FIELD_COLUMNS,
    FLAG,
    Decision,
    Observation,
    metric,
    read_inputs,
    sha256,
    write_json,
)

IMAGE_NDIM = 2


def load_pipeline(run: Path) -> Pipeline:
    """Load the literal prefix and change only the LoadData CSV location."""
    pipeline = Pipeline()
    pipeline.loadtxt(str(run / "qc.cppipe"), raise_on_error=True)
    names = tuple(module.module_name for module in pipeline.modules())
    if names not in (
        ("LoadData", "MeasureImageQuality", "FlagImage", "CorrectIlluminationApply"),
        ("LoadData", "MeasureImageQuality", "FlagImage", "ImageMath"),
    ):
        raise ValueError(f"Unexpected enabled QC prefix: {names}")
    load = pipeline.modules()[0]
    load.csv_directory.value = f"Elsewhere...|{run}"
    load.csv_file_name.value = "selected.csv"
    return pipeline


def archive_decoded(workspace: Workspace, run: Path, field: dict) -> dict:
    """Archive the exact arrays and scales supplied by original CP LoadData."""
    decoded = {}
    for name, asset in field["assets"].items():
        if sha256(Path(asset["path"]).read_bytes()) != asset["file_sha256"]:
            raise ValueError(f"Original input changed: {asset['path']}")
        img = workspace.image_set.get_image(name, must_be_grayscale=True)
        pixels = img.pixel_data
        if pixels.ndim != IMAGE_NDIM or img.has_mask or not np.isfinite(pixels).all():
            raise ValueError(f"Expected a finite, unmasked 2D input: {name}")
        rel = f"decoded/{workspace.measurements.image_set_number:02d}-{name}.npy"
        with (run / rel).open("xb") as stream:
            np.save(stream, pixels, allow_pickle=False)
        decoded[name] = {
            "file": rel,
            "dtype": pixels.dtype.str,
            "shape": list(pixels.shape),
            "scale": metric(float(img.scale)) if img.scale is not None else None,
            "sha256": sha256(pixels.tobytes()),
        }
    return decoded


def observe(workspace: Workspace, outputs: list[str]) -> Observation:
    """Run CP's measurement and flag modules and obey its skip disposition."""
    _, quality, flag, downstream = workspace.pipeline.modules()
    m = workspace.measurements
    quality.run(workspace)
    flag.run(workspace)
    flag_value = m.get_current_image_measurement(FLAG)
    if flag_value not in (0, 1):
        raise ValueError(f"Invalid CellProfiler flag: {flag_value}")
    skipped = workspace.disposition == DISPOSITION_SKIP
    failed = {
        criterion.measurement.value: not bool(
            flag.eval_measurement(workspace, criterion)[0]
        )
        for group in flag.flags
        for criterion in group.measurement_settings
    }
    if not skipped:
        downstream.run(workspace)
        for name in outputs:
            workspace.image_set.get_image(name)
    return {
        "decision": Decision(int(flag_value), skipped, not skipped).record(),
        "criteria_failed": failed,
        "metrics": {
            key: metric(float(m.get_current_image_measurement(key)))
            for key in sorted(m.get_feature_names("Image"))
            if key.startswith("ImageQuality_")
        },
    }


def run_reference(run: Path) -> None:
    """Write oracle observations, using distinct image sets and real CP LoadData."""
    inputs = read_inputs(run)
    env = versions()
    pipeline = load_pipeline(run)
    load = pipeline.modules()[0]
    (run / "decoded").mkdir()
    observations = []
    start_java()
    try:
        m = Measurements()
        try:
            sets = ImageSetList()
            for number, field in enumerate(inputs["fields"], 1):
                m.next_image_set(number)
                workspace = Workspace(pipeline, load, m, ObjectSet(), m, sets)
                if number == 1 and not load.prepare_run(workspace):
                    raise ValueError(
                        "CellProfiler LoadData did not prepare the bounded CSV"
                    )
                load.run(workspace)
                if [
                    str(m.get_current_image_measurement(key)) for key in FIELD_COLUMNS
                ] != field["field"]:
                    raise ValueError(
                        f"CellProfiler loaded the wrong field: {field['field']}"
                    )
                decoded = archive_decoded(workspace, run, field)
                observations.append(
                    {
                        "field": field["field"],
                        "decoded": decoded,
                        "observation": observe(workspace, inputs["after_qc_outputs"]),
                    }
                )
        finally:
            m.close()
    finally:
        stop_java()
    write_json(
        run / "cellprofiler.json",
        {
            "environment": env,
            "inputs_sha256": sha256((run / "inputs.json").read_bytes()),
            "fields": observations,
        },
    )


def main() -> None:
    """Run the CellProfiler half of the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_reference(args.run.resolve())


if __name__ == "__main__":
    main()
