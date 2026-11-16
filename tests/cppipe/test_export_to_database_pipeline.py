"""Verify ExportToDatabase cppipe translation and write()."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import CppipeError, load_pipeline
from cp_anywhere.api import MeasurementResult, measure

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe"
ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = ROOT / "fixtures/synthetic/touching_gradient"


@pytest.fixture
def result() -> MeasurementResult:
    return measure(
        {
            "DNA": np.load(SYNTHETIC / "DNA.npy"),
            "Actin": np.load(SYNTHETIC / "Actin.npy"),
        },
        {
            "Cells": np.load(SYNTHETIC / "labels.npy"),
            "Nuclei": np.load(SYNTHETIC / "nuclei_labels.npy"),
        },
        image_id="touching_gradient",
    )


def test_export_to_database_all_objects_overwrite_schema() -> None:
    pipeline = load_pipeline(
        FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    )
    assert len(pipeline.export_steps) == 1
    step = pipeline.export_steps[0]
    assert step.sqlite_file == "MyExpt.db"
    assert step.objects_choice == "all"
    assert step.objects == ()
    assert step.if_exists == "replace"


def test_export_to_database_select_objects_never_overwrite() -> None:
    pipeline = load_pipeline(
        FIXTURES / "export_to_database/select_objects_never_overwrite.cppipe"
    )
    assert len(pipeline.export_steps) == 1
    step = pipeline.export_steps[0]
    assert step.sqlite_file == "MyExpt.db"
    assert step.objects_choice == "select"
    assert step.objects == ("Cells",)
    assert step.if_exists == "append"


def _table_names(db_path: Path) -> set[str]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def test_export_to_database_write_all_objects(
    tmp_path: Path, result: MeasurementResult
) -> None:
    pipeline = load_pipeline(
        FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    )
    step = pipeline.export_steps[0]
    db_path = tmp_path / step.sqlite_file
    step.write(result, db_path)
    assert _table_names(db_path) == {"Per_Image", "Per_Cells", "Per_Nuclei"}


def test_export_to_database_write_selected_objects_only(
    tmp_path: Path, result: MeasurementResult
) -> None:
    pipeline = load_pipeline(
        FIXTURES / "export_to_database/select_objects_never_overwrite.cppipe"
    )
    step = pipeline.export_steps[0]
    db_path = tmp_path / step.sqlite_file
    step.write(result, db_path)
    # objects_choice == "select" with objects == ("Cells",): Per_Nuclei
    # must NOT appear, matching real CellProfiler's own object-choice
    # filtering (Per_Image always gets written regardless).
    assert _table_names(db_path) == {"Per_Image", "Per_Cells"}


def test_export_to_database_selected_scopes_none(result: MeasurementResult) -> None:
    pipeline = load_pipeline(
        FIXTURES / "export_to_database/select_objects_never_overwrite.cppipe"
    )
    step = pipeline.export_steps[0]
    none_step = replace(step, objects_choice="none", objects=())
    assert none_step.selected_scopes(result) == ("Image",)


def test_export_to_database_revision_is_checked(tmp_path: Path) -> None:
    source = FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    text = source.read_text().replace(
        "variable_revision_number:28", "variable_revision_number:999"
    )
    path = tmp_path / "wrong_revision.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match=r"ExportToDatabase.*999"):
        load_pipeline(path)


def test_export_to_database_rejects_mysql(tmp_path: Path) -> None:
    source = FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    text = source.read_text().replace("Database type:SQLite", "Database type:MySQL")
    path = tmp_path / "mysql.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="MySQL"):
        load_pipeline(path)


def test_export_to_database_rejects_combined_object_table(tmp_path: Path) -> None:
    source = FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    text = source.read_text().replace(
        "a single object view?:One table per object type",
        "a single object view?:Single object table",
    )
    path = tmp_path / "combined.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="single object table or a single"):
        load_pipeline(path)


def test_export_to_database_rejects_table_prefix(tmp_path: Path) -> None:
    source = FIXTURES / "export_to_database/all_objects_overwrite_schema.cppipe"
    text = source.read_text().replace(
        "Add a prefix to table names?:No", "Add a prefix to table names?:Yes"
    )
    path = tmp_path / "prefixed.cppipe"
    path.write_text(text)
    with pytest.raises(CppipeError, match="prefix"):
        load_pipeline(path)
