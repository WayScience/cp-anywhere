"""Tests for the cp-anywhere command-line interface."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.feather
import pyarrow.parquet
import tifffile

from cp_anywhere.api import read_parquet_feature_names

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"
TOTAL_FEATURE_COLUMNS = 119  # spec/compatibility.md's declared scope


def test_cli_help() -> None:
    """The CLI exposes usage without requiring input arrays."""
    output = subprocess.run(
        ["uv", "run", "cp-anywhere", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    assert "--image" in output.stdout
    assert "--objects" in output.stdout


def test_cli_measures_saved_arrays_to_json(tmp_path: Path) -> None:
    """The CLI loads NPY arrays and writes a decodable JSON result."""
    output_path = tmp_path / "result.json"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    decoded = json.loads(output_path.read_text())
    assert decoded["image_id"] == "touching_gradient"
    assert (
        sum(len(columns) for columns in decoded["features"].values())
        == TOTAL_FEATURE_COLUMNS
    )


def test_cli_measures_tiff_images_to_json(tmp_path: Path) -> None:
    """The CLI also accepts ordinary TIFF image files, not just NPY."""
    dna_path = tmp_path / "dna.tif"
    actin_path = tmp_path / "actin.tif"
    tifffile.imwrite(dna_path, np.load(FIXTURE / "DNA.npy"))
    tifffile.imwrite(actin_path, np.load(FIXTURE / "Actin.npy"))
    output_path = tmp_path / "result.json"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={dna_path}",
            "--image",
            f"Actin={actin_path}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    decoded = json.loads(output_path.read_text())
    assert decoded["image_id"] == "touching_gradient"
    assert (
        sum(len(columns) for columns in decoded["features"].values())
        == TOTAL_FEATURE_COLUMNS
    )


def test_cli_measures_to_arrow(tmp_path: Path) -> None:
    """A .arrow --output extension writes all feature data as Arrow IPC."""
    output_path = tmp_path / "result.arrow"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    table = pyarrow.feather.read_table(output_path)
    assert table.column_names == ["image_id", "scope", "object_id", "feature", "value"]
    pairs = set(
        zip(table.column("scope").to_pylist(), table.column("feature").to_pylist())
    )
    assert len(pairs) == TOTAL_FEATURE_COLUMNS


def test_cli_measures_to_parquet(tmp_path: Path) -> None:
    """A .parquet --output extension writes a real Parquet file."""
    output_path = tmp_path / "result.parquet"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    table = pyarrow.parquet.read_table(output_path)
    assert table.column_names == ["image_id", "scope", "object_id", "feature", "value"]


def test_cli_measures_to_sqlite_strict_tables(tmp_path: Path) -> None:
    """A .sqlite --output extension writes STRICT Per_Image/Per_<Object> tables."""
    output_path = tmp_path / "result.sqlite"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    connection = sqlite3.connect(str(output_path))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert tables == {"Per_Image", "Per_Cells", "Per_Nuclei"}
    finally:
        connection.close()


def test_cli_measures_to_parquet_joined_mode(tmp_path: Path) -> None:
    """--parquet-mode joined writes a single combined CytoTable-style
    wide table to a --output .parquet file."""
    output_path = tmp_path / "joined.parquet"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_path),
            "--parquet-mode",
            "joined",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    table = pyarrow.parquet.read_table(output_path)
    assert "Cells_Intensity_MeanIntensity_DNA" in table.column_names
    assert "Nuclei_Intensity_MeanIntensity_DNA" in table.column_names
    assert "Image_Intensity_TotalIntensity_DNA" in table.column_names


def test_cli_measures_to_parquet_wide_mode(tmp_path: Path) -> None:
    """--parquet-mode wide treats --output as a directory of per-scope
    files, each with one real named column per feature."""
    output_dir = tmp_path / "wide"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_dir),
            "--parquet-mode",
            "wide",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    written = {path.name for path in output_dir.glob("*.parquet")}
    assert written == {"Per_Image.parquet", "Per_Cells.parquet", "Per_Nuclei.parquet"}
    table = pyarrow.parquet.read_table(output_dir / "Per_Cells.parquet")
    assert "features" not in table.column_names
    assert "Intensity_MeanIntensity_DNA" in table.column_names


def test_cli_measures_to_parquet_array_mode(tmp_path: Path) -> None:
    """--parquet-mode array treats --output as a directory of per-scope
    files, each with features in one fixed-size array column."""
    output_dir = tmp_path / "arrays"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--image",
            f"DNA={FIXTURE / 'DNA.npy'}",
            "--image",
            f"Actin={FIXTURE / 'Actin.npy'}",
            "--objects",
            f"Cells={FIXTURE / 'labels.npy'}",
            "--objects",
            f"Nuclei={FIXTURE / 'nuclei_labels.npy'}",
            "--image-id",
            "touching_gradient",
            "--output",
            str(output_dir),
            "--parquet-mode",
            "array",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    written = {path.name for path in output_dir.glob("*.parquet")}
    assert written == {"Per_Image.parquet", "Per_Cells.parquet", "Per_Nuclei.parquet"}
    feature_names = read_parquet_feature_names(output_dir / "Per_Cells.parquet")
    table = pyarrow.parquet.read_table(output_dir / "Per_Cells.parquet")
    assert table.schema.field("features").type.list_size == len(feature_names)


def test_cli_runs_a_real_cppipe_pipeline(tmp_path: Path) -> None:
    """``--pipeline`` runs a real .cppipe's translated segmentation and
    measurement steps, writing the measurement result like any other run."""
    cppipe_fixture = ROOT / "fixtures/cppipe/otsu_simple.cppipe"
    image_path = ROOT / "fixtures/cppipe/test_image.npy"
    output_path = tmp_path / "result.json"
    subprocess.run(
        [
            "uv",
            "run",
            "cp-anywhere",
            "--pipeline",
            str(cppipe_fixture),
            "--image",
            f"DNA={image_path}",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    decoded = json.loads(output_path.read_text())
    assert decoded["image_id"] == "Image"
    assert "Cells" in decoded["object_ids"]
