"""Tests for the cp-anywhere command-line interface."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    import numpy as np
    import tifffile

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
