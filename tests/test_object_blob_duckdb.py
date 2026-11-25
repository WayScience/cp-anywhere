"""Tests for writing the object-blob and whole-frame-label exports
directly into DuckDB database files (same row content and decode
guarantees as the Parquet writers, ingested with the same
create/append/replace semantics as MeasurementResult.to_duckdb).
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from cp_anywhere.api import MeasurementResult, measure
from cp_anywhere.object_export import (
    decode_labels_rle,
    decode_mask_rle,
    read_objects_parquet,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"


@pytest.fixture
def result() -> MeasurementResult:
    return measure(
        {
            "DNA": np.load(FIXTURE / "DNA.npy"),
            "Actin": np.load(FIXTURE / "Actin.npy"),
        },
        {
            "Cells": np.load(FIXTURE / "labels.npy"),
            "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
        },
        image_id="touching_gradient",
    )


@pytest.fixture
def labels_arrays() -> dict[str, np.ndarray]:
    return {
        "Cells": np.load(FIXTURE / "labels.npy"),
        "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
    }


def test_to_objects_duckdb_stores_decodable_blobs(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.duckdb"
    images = {
        "DNA": np.load(FIXTURE / "DNA.npy"),
        "Actin": np.load(FIXTURE / "Actin.npy"),
    }
    dna = np.asarray(images["DNA"])
    result.to_objects_duckdb(
        destination,
        images=images,
        objects=labels_arrays,
        channels=["DNA"],
        image_format="tiff",
    )
    connection = duckdb.connect(str(destination))
    try:
        table_info = connection.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'SegmentationObjects' ORDER BY ordinal_position"
        ).fetchall()
        names = {name for name, _ in table_info}
        types = dict(table_info)
        assert {
            "image_id",
            "scope",
            "object_id",
            "row_min",
            "row_max",
            "col_min",
            "col_max",
            "area_pixels",
            "channels",
            "image_tiff",
            "mask_rle",
        } <= names
        assert types["image_tiff"] == "BLOB"
        assert types["mask_rle"] == "BLOB"
        rows = connection.execute(
            "SELECT image_id, scope, object_id, row_min, row_max, col_min, "
            "col_max, area_pixels, image_tiff, mask_rle "
            "FROM SegmentationObjects ORDER BY scope, object_id"
        ).fetchall()
    finally:
        connection.close()
    expected_rows = sum(len(result.object_ids[s]) for s in result.object_ids)
    assert len(rows) == expected_rows
    for image_id, scope, object_id, r0, r1, c0, c1, area, tiff, mask in rows:
        assert image_id == "touching_gradient"
        owned = labels_arrays[scope][r0 : r1 + 1, c0 : c1 + 1] == object_id
        # blob decode guarantees: exact mask and exact TIFF crop
        assert np.array_equal(decode_mask_rle(mask, owned.shape), owned)
        assert np.array_equal(
            tifffile.imread(io.BytesIO(tiff)), dna[r0 : r1 + 1, c0 : c1 + 1]
        )
        assert int(np.count_nonzero(owned)) == area


def test_to_objects_duckdb_accumulates_across_images(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.duckdb"
    images = {"DNA": np.load(FIXTURE / "DNA.npy")}
    result.select_scopes(["Nuclei"]).to_objects_duckdb(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    second = measure(
        images,
        {"Nuclei": labels_arrays["Nuclei"]},
        image_id="touching_gradient_2",
        families=["image_intensity", "object_intensity"],
    )
    second.to_objects_duckdb(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    connection = duckdb.connect(str(destination))
    try:
        ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT image_id FROM SegmentationObjects"
            ).fetchall()
        }
    finally:
        connection.close()
    assert ids == {"touching_gradient", "touching_gradient_2"}
    result.select_scopes(["Nuclei"]).to_objects_duckdb(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
        if_exists="replace",
    )
    connection = duckdb.connect(str(destination))
    try:
        ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT image_id FROM SegmentationObjects"
            ).fetchall()
        }
        fetched = connection.execute(
            "SELECT count(*) FROM SegmentationObjects"
        ).fetchone()
        assert fetched is not None
        count = fetched[0]
    finally:
        connection.close()
    assert ids == {"touching_gradient"}
    assert count > 0


def test_to_labels_duckdb_stores_the_whole_frame(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "labels.duckdb"
    result.to_labels_duckdb(destination, objects=labels_arrays)
    connection = duckdb.connect(str(destination))
    try:
        rows = connection.execute(
            "SELECT image_id, scope, rows, cols, object_count, labels_rle "
            "FROM SegmentationLabels ORDER BY scope"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == len(result.object_ids)
    for image_id, scope, n_rows, n_cols, object_count, blob in rows:
        assert image_id == "touching_gradient"
        decoded = decode_labels_rle(blob, (n_rows, n_cols))
        assert np.array_equal(decoded, labels_arrays[scope]), scope
        assert object_count == len(result.object_ids[scope])


def test_to_objects_duckdb_matches_the_parquet_contents(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    images = {"DNA": np.load(FIXTURE / "DNA.npy")}
    parquet = tmp_path / "objects.parquet"
    duckdb_file = tmp_path / "objects.duckdb"
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        parquet,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    result.select_scopes(["Nuclei"]).to_objects_duckdb(
        duckdb_file,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    parquet_rows = read_objects_parquet(parquet).to_pylist()
    connection = duckdb.connect(str(duckdb_file))
    try:
        duckdb_rows = connection.execute(
            "SELECT image_id, scope, object_id, mask_rle FROM "
            "SegmentationObjects ORDER BY object_id"
        ).fetchall()
    finally:
        connection.close()
    parquet_by_id = {row["object_id"]: row for row in parquet_rows}
    for image_id, scope, object_id, mask in duckdb_rows:
        assert (image_id, scope) == (parquet_by_id[object_id]["image_id"], scope)
        assert mask == parquet_by_id[object_id]["mask_rle"]


def test_to_objects_duckdb_rejects_unknown_format(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="image_format"):
        result.select_scopes(["Nuclei"]).to_objects_duckdb(
            tmp_path / "x.duckdb",
            images={"DNA": np.load(FIXTURE / "DNA.npy")},
            objects={"Nuclei": labels_arrays["Nuclei"]},
            channels=["DNA"],
            image_format="webp",
        )


def test_to_objects_duckdb_and_labels_live_in_one_database(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "segmentation.duckdb"
    images = {"DNA": np.load(FIXTURE / "DNA.npy")}
    result.select_scopes(["Nuclei"]).to_objects_duckdb(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    result.select_scopes(["Nuclei"]).to_labels_duckdb(
        destination, objects={"Nuclei": labels_arrays["Nuclei"]}
    )
    connection = duckdb.connect(str(destination))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        connection.close()
    assert {"SegmentationObjects", "SegmentationLabels"} <= tables


def test_to_objects_duckdb_jpeg_blobs_stay_queryable_and_decodable(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.duckdb"
    result.select_scopes(["Nuclei"]).to_objects_duckdb(
        destination,
        images={"DNA": np.load(FIXTURE / "DNA.npy")},
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    connection = duckdb.connect(str(destination))
    try:
        fetched = connection.execute(
            "SELECT image_jpeg FROM SegmentationObjects LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert fetched is not None
    crop = PILImage.open(io.BytesIO(fetched[0]))
    assert crop.format == "JPEG"
