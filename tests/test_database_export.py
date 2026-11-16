"""Tests for MeasurementResult's database-export capabilities: Parquet,
DuckDB, and SQLite (STRICT tables). See spec/compatibility.md for the
declared feature scope these exports carry; this file only tests the
export mechanics themselves (schema, typing, accumulation), reusing the
cppipe-coverage test image/fixtures' Cells/Nuclei object sets for real,
non-trivial measurement data (not synthetic single-pixel toys).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.dataset
import pyarrow.parquet
import pytest

from cp_anywhere.api import MeasurementResult, measure, read_parquet_feature_names

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
def second_result() -> MeasurementResult:
    """A second, differently-named image's measurement, for testing
    multi-image accumulation into one database file."""
    return measure(
        {
            "DNA": np.load(FIXTURE / "DNA.npy"),
            "Actin": np.load(FIXTURE / "Actin.npy"),
        },
        {
            "Cells": np.load(FIXTURE / "labels.npy"),
            "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
        },
        image_id="touching_gradient_2",
    )


# --------------------------------------------------------------------------
# Parquet: same long-form rows as to_csv/to_arrow, as a real Parquet file.
# --------------------------------------------------------------------------


def test_to_parquet_writes_the_same_long_form_rows_as_to_arrow(
    result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.parquet"
    result.to_parquet(path)
    table = pyarrow.parquet.read_table(path)
    assert table.column_names == ["image_id", "scope", "object_id", "feature", "value"]
    assert table.num_rows == len(result.rows())
    assert set(table.column("image_id").to_pylist()) == {"touching_gradient"}


def test_to_parquet_preserves_nonfinite_values_natively(
    result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.parquet"
    result.to_parquet(path)
    table = pyarrow.parquet.read_table(path)
    values = table.column("value").to_pylist()
    # Real feature data; just confirm every value round-trips as a float
    # (no string "NaN"/"Infinity" markers, unlike to_json's encoding).
    assert all(isinstance(value, float) for value in values)


# --------------------------------------------------------------------------
# Parquet, mode="array": one file per scope, features packed into a
# single fixed-size-list column, with the ordered feature names
# embedded in the file's own schema metadata (self-describing, no
# sidecar file to go out of sync).
# --------------------------------------------------------------------------


def test_to_parquet_array_mode_writes_one_file_per_scope(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "arrays"
    result.to_parquet(directory, mode="array")
    written = {path.name for path in directory.glob("*.parquet")}
    assert written == {"Per_Image.parquet", "Per_Cells.parquet", "Per_Nuclei.parquet"}


def test_to_parquet_array_mode_features_are_a_fixed_size_list_column(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "arrays"
    result.to_parquet(directory, mode="array")
    table = pyarrow.parquet.read_table(directory / "Per_Cells.parquet")

    assert table.column_names == ["image_id", "object_id", "features"]
    assert table.num_rows == len(result.object_ids["Cells"])

    feature_names = list(result.features["Cells"].keys())
    features_type = table.schema.field("features").type
    assert isinstance(features_type, pyarrow.FixedSizeListType)
    assert features_type.list_size == len(feature_names)

    first_row = table.column("features")[0].as_py()
    assert len(first_row) == len(feature_names)
    assert all(isinstance(value, float) for value in first_row)


def test_to_parquet_array_mode_embeds_feature_names_in_schema_metadata(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "arrays"
    result.to_parquet(directory, mode="array")

    feature_names = read_parquet_feature_names(directory / "Per_Cells.parquet")
    assert feature_names == list(result.features["Cells"].keys())

    image_feature_names = read_parquet_feature_names(directory / "Per_Image.parquet")
    assert image_feature_names == list(result.features["Image"].keys())


def test_to_parquet_array_mode_matches_values_from_long_form(
    result: MeasurementResult, tmp_path: Path
) -> None:
    """The array-mode column and the long-form rows describe the same
    measurements -- prove it by cross-checking one object's values."""
    long_path = tmp_path / "result.parquet"
    result.to_parquet(long_path)
    long_table = pyarrow.parquet.read_table(long_path).to_pylist()

    directory = tmp_path / "arrays"
    result.to_parquet(directory, mode="array")
    cells_table = pyarrow.parquet.read_table(directory / "Per_Cells.parquet")
    feature_names = read_parquet_feature_names(directory / "Per_Cells.parquet")

    object_id = cells_table.column("object_id")[0].as_py()
    array_row = cells_table.column("features")[0].as_py()
    for name, array_value in zip(feature_names, array_row, strict=True):
        (long_row,) = [
            row
            for row in long_table
            if row["scope"] == "Cells"
            and row["object_id"] == object_id
            and row["feature"] == name
        ]
        assert array_value == pytest.approx(long_row["value"], nan_ok=True)


def test_to_parquet_rejects_an_unknown_mode(
    result: MeasurementResult, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="mode"):
        result.to_parquet(tmp_path / "result.parquet", mode="bogus")


def test_read_parquet_feature_names_rejects_a_plain_long_form_file(
    result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.parquet"
    result.to_parquet(path)
    with pytest.raises(ValueError, match="feature_names"):
        read_parquet_feature_names(path)


# --------------------------------------------------------------------------
# Parquet, mode="wide": one file per scope, one real named column per
# feature -- the traditional wide/tabular shape (same as to_duckdb/
# to_sqlite's own tables), not packed into an array.
# --------------------------------------------------------------------------


def test_to_parquet_wide_mode_writes_one_file_per_scope(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "wide"
    result.to_parquet(directory, mode="wide")
    written = {path.name for path in directory.glob("*.parquet")}
    assert written == {"Per_Image.parquet", "Per_Cells.parquet", "Per_Nuclei.parquet"}


def test_to_parquet_wide_mode_has_one_named_column_per_feature(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "wide"
    result.to_parquet(directory, mode="wide")
    table = pyarrow.parquet.read_table(directory / "Per_Cells.parquet")

    feature_names = list(result.features["Cells"].keys())
    assert table.column_names == ["image_id", "object_id", *feature_names]
    assert table.num_rows == len(result.object_ids["Cells"])
    assert table.schema.field(feature_names[0]).type == pyarrow.float64()


def test_to_parquet_wide_mode_image_table_has_no_object_id(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "wide"
    result.to_parquet(directory, mode="wide")
    table = pyarrow.parquet.read_table(directory / "Per_Image.parquet")
    assert "object_id" not in table.column_names
    assert table.num_rows == 1


def test_to_parquet_wide_mode_matches_values_from_long_form(
    result: MeasurementResult, tmp_path: Path
) -> None:
    long_path = tmp_path / "result.parquet"
    result.to_parquet(long_path)
    long_table = pyarrow.parquet.read_table(long_path).to_pylist()

    directory = tmp_path / "wide"
    result.to_parquet(directory, mode="wide")
    cells_table = pyarrow.parquet.read_table(directory / "Per_Cells.parquet")

    object_id = cells_table.column("object_id")[0].as_py()
    first_row = {
        name: cells_table.column(name)[0].as_py() for name in cells_table.column_names
    }
    for name, value in first_row.items():
        if name in ("image_id", "object_id"):
            continue
        (long_row,) = [
            row
            for row in long_table
            if row["scope"] == "Cells"
            and row["object_id"] == object_id
            and row["feature"] == name
        ]
        assert value == pytest.approx(long_row["value"], nan_ok=True)


# --------------------------------------------------------------------------
# Parquet, mode="joined": one combined, cytotable-style wide table --
# one row per object, joined across every object scope on object_id,
# columns prefixed by scope name, Image-scope features broadcast onto
# every row. Written either as a single .parquet file or, for a
# directory path, as a paginated Parquet dataset (one or more
# row-group-bounded fragment files) that can accumulate across many
# images.
# --------------------------------------------------------------------------


def test_to_parquet_joined_mode_single_file(
    result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "joined.parquet"
    result.to_parquet(path, mode="joined")
    table = pyarrow.parquet.read_table(path)

    assert "image_id" in table.column_names
    assert "object_id" in table.column_names
    assert "Image_Intensity_TotalIntensity_DNA" in table.column_names
    assert "Cells_Intensity_MeanIntensity_DNA" in table.column_names
    assert "Nuclei_Intensity_MeanIntensity_DNA" in table.column_names
    assert table.num_rows == len(result.object_ids["Cells"])


def test_to_parquet_joined_mode_values_match_long_form(
    result: MeasurementResult, tmp_path: Path
) -> None:
    long_path = tmp_path / "result.parquet"
    result.to_parquet(long_path)
    long_rows = pyarrow.parquet.read_table(long_path).to_pylist()

    joined_path = tmp_path / "joined.parquet"
    result.to_parquet(joined_path, mode="joined")
    joined_rows = pyarrow.parquet.read_table(joined_path).to_pylist()

    for row in joined_rows:
        object_id = row["object_id"]
        for scope in ("Cells", "Nuclei"):
            feature_names = list(result.features[scope].keys())
            name = feature_names[0]
            (long_row,) = [
                long_row
                for long_row in long_rows
                if long_row["scope"] == scope
                and long_row["object_id"] == object_id
                and long_row["feature"] == name
            ]
            assert row[f"{scope}_{name}"] == pytest.approx(
                long_row["value"], nan_ok=True
            )
        image_feature_names = list(result.features["Image"].keys())
        image_name = image_feature_names[0]
        (image_row,) = [
            long_row
            for long_row in long_rows
            if long_row["scope"] == "Image" and long_row["feature"] == image_name
        ]
        assert row[f"Image_{image_name}"] == pytest.approx(
            image_row["value"], nan_ok=True
        )


def test_to_parquet_joined_mode_as_a_dataset_directory(
    result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "joined_dataset"
    result.to_parquet(directory, mode="joined")
    table = pyarrow.dataset.dataset(str(directory), format="parquet").to_table()
    assert table.num_rows == len(result.object_ids["Cells"])
    assert "Cells_Intensity_MeanIntensity_DNA" in table.column_names


def test_to_parquet_joined_mode_dataset_appends_across_multiple_images(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "joined_dataset"
    result.to_parquet(directory, mode="joined")
    second_result.to_parquet(directory, mode="joined", if_exists="append")

    table = pyarrow.dataset.dataset(str(directory), format="parquet").to_table()
    assert table.num_rows == 2 * len(result.object_ids["Cells"])
    assert set(table.column("image_id").to_pylist()) == {
        "touching_gradient",
        "touching_gradient_2",
    }


def test_to_parquet_joined_mode_dataset_replace_overwrites(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    directory = tmp_path / "joined_dataset"
    result.to_parquet(directory, mode="joined")
    second_result.to_parquet(directory, mode="joined", if_exists="replace")

    table = pyarrow.dataset.dataset(str(directory), format="parquet").to_table()
    assert set(table.column("image_id").to_pylist()) == {"touching_gradient_2"}


def test_to_parquet_joined_mode_respects_max_rows_per_file(
    result: MeasurementResult, tmp_path: Path
) -> None:
    """A small max_rows_per_file pages the dataset into multiple
    fragment files rather than one big file -- real Parquet-native
    pagination, not a bespoke chunking scheme."""
    directory = tmp_path / "joined_dataset"
    result.to_parquet(directory, mode="joined", max_rows_per_file=1)
    fragment_files = list(directory.glob("*.parquet"))
    assert len(fragment_files) == len(result.object_ids["Cells"])
    table = pyarrow.dataset.dataset(str(directory), format="parquet").to_table()
    assert table.num_rows == len(result.object_ids["Cells"])


def test_to_parquet_joined_mode_default_join_is_outer() -> None:
    """Object scopes with mismatched object_ids still produce a row for
    every object_id seen in *any* scope by default, with null feature
    values for scopes missing that object."""
    crafted = MeasurementResult(
        image_id="crafted",
        features={
            "Cells": {"Feature_A": np.array([1.0, 2.0, 3.0])},
            "Nuclei": {"Feature_B": np.array([10.0, 20.0])},
        },
        object_ids={
            "Cells": np.array([1, 2, 3]),
            "Nuclei": np.array([1, 2]),
        },
        relationships=[],
    )
    table = crafted._joined_table(join="outer")
    rows = {row["object_id"]: row for row in table.to_pylist()}
    assert set(rows) == {1, 2, 3}
    assert rows[3]["Nuclei_Feature_B"] is None
    assert rows[1]["Nuclei_Feature_B"] == pytest.approx(10.0)


def test_to_parquet_joined_mode_inner_join_drops_unmatched_objects() -> None:
    crafted = MeasurementResult(
        image_id="crafted",
        features={
            "Cells": {"Feature_A": np.array([1.0, 2.0, 3.0])},
            "Nuclei": {"Feature_B": np.array([10.0, 20.0])},
        },
        object_ids={
            "Cells": np.array([1, 2, 3]),
            "Nuclei": np.array([1, 2]),
        },
        relationships=[],
    )
    table = crafted._joined_table(join="inner")
    assert set(table.column("object_id").to_pylist()) == {1, 2}


def test_to_parquet_joined_mode_rejects_an_unknown_join(
    result: MeasurementResult, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="join"):
        result.to_parquet(tmp_path / "joined.parquet", mode="joined", join="bogus")


# --------------------------------------------------------------------------
# DuckDB: wide Per_Image / Per_<ObjectName> tables, CellProfiler's own
# ExportToDatabase naming convention.
# --------------------------------------------------------------------------


def test_to_duckdb_creates_per_image_and_per_object_tables(
    result: MeasurementResult, tmp_path: Path
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "result.duckdb"
    result.to_duckdb(path)

    connection = duckdb.connect(str(path))
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert tables == {"Per_Image", "Per_Cells", "Per_Nuclei"}

        image_rows = connection.execute("SELECT * FROM Per_Image").fetchdf()
        assert len(image_rows) == 1
        assert image_rows["image_id"].iloc[0] == "touching_gradient"

        cells_rows = connection.execute("SELECT * FROM Per_Cells").fetchdf()
        assert len(cells_rows) == len(result.object_ids["Cells"])
        assert set(cells_rows["object_id"]) == {
            int(value) for value in result.object_ids["Cells"]
        }
    finally:
        connection.close()


def test_to_duckdb_append_accumulates_rows_across_multiple_images(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "result.duckdb"
    result.to_duckdb(path)
    second_result.to_duckdb(path, if_exists="append")

    connection = duckdb.connect(str(path))
    try:
        image_rows = connection.execute(
            "SELECT image_id FROM Per_Image ORDER BY image_id"
        ).fetchall()
        assert [row[0] for row in image_rows] == [
            "touching_gradient",
            "touching_gradient_2",
        ]
    finally:
        connection.close()


def test_to_duckdb_replace_overwrites_rather_than_accumulates(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "result.duckdb"
    result.to_duckdb(path)
    second_result.to_duckdb(path, if_exists="replace")

    connection = duckdb.connect(str(path))
    try:
        image_rows = connection.execute("SELECT image_id FROM Per_Image").fetchall()
        assert [row[0] for row in image_rows] == ["touching_gradient_2"]
    finally:
        connection.close()


# --------------------------------------------------------------------------
# SQLite: the same wide tables, but as STRICT tables.
# --------------------------------------------------------------------------


def test_to_sqlite_creates_per_image_and_per_object_strict_tables(
    result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.sqlite"
    result.to_sqlite(path)

    connection = sqlite3.connect(str(path))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert tables == {"Per_Image", "Per_Cells", "Per_Nuclei"}

        for table in tables:
            (schema_sql,) = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (table,)
            ).fetchone()
            assert "STRICT" in schema_sql

        image_rows = connection.execute("SELECT * FROM Per_Image").fetchall()
        assert len(image_rows) == 1

        cells_rows = connection.execute("SELECT * FROM Per_Cells").fetchall()
        assert len(cells_rows) == len(result.object_ids["Cells"])
    finally:
        connection.close()


def test_to_sqlite_strict_tables_reject_a_type_violation(
    result: MeasurementResult, tmp_path: Path
) -> None:
    """A STRICT table genuinely enforces column types -- prove it by
    attempting an insert SQLite's legacy (non-STRICT) type affinity
    would have silently accepted."""
    path = tmp_path / "result.sqlite"
    result.to_sqlite(path)

    connection = sqlite3.connect(str(path))
    try:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(Per_Image)").fetchall()
        ]
        placeholders = ", ".join("?" for _ in columns)
        values = ["not-a-number" for _ in columns]
        values[columns.index("image_id")] = "bad-row"
        # STRICT tables raise here; a legacy (non-STRICT) table's type
        # affinity would have silently stored the string instead.
        with pytest.raises(sqlite3.IntegrityError, match="cannot store TEXT"):
            connection.execute(
                f"INSERT INTO Per_Image ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
    finally:
        connection.close()


def test_to_sqlite_append_accumulates_rows_across_multiple_images(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.sqlite"
    result.to_sqlite(path)
    second_result.to_sqlite(path, if_exists="append")

    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute(
            "SELECT image_id FROM Per_Image ORDER BY image_id"
        ).fetchall()
        assert [row[0] for row in rows] == [
            "touching_gradient",
            "touching_gradient_2",
        ]
    finally:
        connection.close()


def test_to_sqlite_replace_overwrites_rather_than_accumulates(
    result: MeasurementResult, second_result: MeasurementResult, tmp_path: Path
) -> None:
    path = tmp_path / "result.sqlite"
    result.to_sqlite(path)
    second_result.to_sqlite(path, if_exists="replace")

    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute("SELECT image_id FROM Per_Image").fetchall()
        assert [row[0] for row in rows] == ["touching_gradient_2"]
    finally:
        connection.close()


def test_to_sqlite_rejects_too_old_sqlite_for_strict_tables(
    result: MeasurementResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 36, 0))
    with pytest.raises(RuntimeError, match=r"3\.37"):
        result.to_sqlite(tmp_path / "result.sqlite")
