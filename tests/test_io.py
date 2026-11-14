"""Tests for the file-format-aware simple calling convention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

import cp_anywhere
from cp_anywhere.io import load_array, measure_files

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"
TOTAL_FEATURE_COLUMNS = 119  # spec/compatibility.md's declared scope


def test_load_array_reads_npy() -> None:
    array = load_array(FIXTURE / "DNA.npy")
    assert np.array_equal(array, np.load(FIXTURE / "DNA.npy"))


def test_load_array_reads_tiff(tmp_path: Path) -> None:
    source = np.load(FIXTURE / "DNA.npy")
    path = tmp_path / "dna.tif"
    tifffile.imwrite(path, source)
    assert np.array_equal(load_array(path), source)


def test_load_array_reads_png(tmp_path: Path) -> None:
    source = np.load(FIXTURE / "labels.npy").astype(np.uint8)
    path = tmp_path / "labels.png"
    Image.fromarray(source).save(path)
    assert np.array_equal(load_array(path), source)


def test_load_array_passes_through_arrays() -> None:
    array = np.load(FIXTURE / "DNA.npy")
    assert load_array(array) is array


def test_load_array_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported file extension"):
        load_array(path)


def test_measure_files_matches_in_memory_result() -> None:
    expected = cp_anywhere.measure(
        {name: np.load(FIXTURE / f"{name}.npy") for name in ("DNA", "Actin")},
        {
            "Cells": np.load(FIXTURE / "labels.npy"),
            "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
        },
        image_id="touching_gradient",
    )
    frame = measure_files(
        {"DNA": FIXTURE / "DNA.npy", "Actin": FIXTURE / "Actin.npy"},
        {"Cells": FIXTURE / "labels.npy", "Nuclei": FIXTURE / "nuclei_labels.npy"},
        image_id="touching_gradient",
    )
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(expected.rows())
    assert len(frame[["scope", "feature"]].drop_duplicates()) == TOTAL_FEATURE_COLUMNS
    assert frame.to_dict("records") == expected.rows()


def test_measure_files_matches_oracle_on_real_bbbc007() -> None:
    folder = ROOT / "fixtures/real/real_bbbc007"
    expected = json.loads((ROOT / "fixtures/reference/real_bbbc007.json").read_text())
    frame = measure_files(
        {"DNA": folder / "DNA.npy", "Actin": folder / "Actin.npy"},
        {"Cells": folder / "labels.npy", "Nuclei": folder / "nuclei_labels.npy"},
        image_id="real_bbbc007",
    )
    cells_edge = (
        frame[
            (frame["scope"] == "Cells")
            & (frame["feature"] == "Intensity_IntegratedIntensityEdge_Actin")
        ]
        .sort_values("object_id")["value"]
        .tolist()
    )
    assert (
        cells_edge
        == expected["features"]["Cells"]["Intensity_IntegratedIntensityEdge_Actin"]
    )


def test_measure_files_accepts_bare_paths_without_mapping() -> None:
    frame = measure_files(FIXTURE / "DNA.npy", FIXTURE / "labels.npy")
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["scope"]) <= {"Image", "Cells"}
