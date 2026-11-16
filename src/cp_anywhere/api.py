"""NumPy input validation and the public CP-Anywhere measurement result."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import pyarrow as pa

DEFAULT_FAMILIES = ("image_intensity", "object_intensity", "neighbors")
FAMILIES = (*DEFAULT_FAMILIES, "object_size_shape")
MAX_LABEL = 1_000_000
MAX_IMAGE_DTYPE_ITEMSIZE = 8  # bytes; i.e. up to a 64-bit int/uint/float

# Parquet schema-metadata key under which to_parquet(mode="array") embeds
# a scope's ordered feature name list, so a fixed-size `features` array
# column stays self-describing without a separate sidecar file.
_FEATURE_NAMES_METADATA_KEY = b"cp_anywhere.feature_names"


@dataclass(frozen=True)
class MeasurementResult:
    """Typed feature columns and directed neighbor edges for one image."""

    image_id: str
    features: dict[str, dict[str, float | int | np.ndarray]]
    object_ids: dict[str, np.ndarray]
    relationships: list[dict[str, str | int]]

    def rows(self) -> list[dict[str, str | int | float]]:
        """Return a long table with one row per feature and object ID."""
        rows: list[dict[str, str | int | float]] = []
        for scope, columns in self.features.items():
            for name, values in columns.items():
                if scope == "Image":
                    rows.append(
                        {
                            "image_id": self.image_id,
                            "scope": scope,
                            "object_id": 0,
                            "feature": name,
                            "value": float(values),
                        }
                    )
                else:
                    for object_id, value in zip(
                        self.object_ids[scope], np.asarray(values), strict=True
                    ):
                        rows.append(
                            {
                                "image_id": self.image_id,
                                "scope": scope,
                                "object_id": int(object_id),
                                "feature": name,
                                "value": value.item(),
                            }
                        )
        return rows

    def select_scopes(self, scopes: Iterable[str]) -> MeasurementResult:
        """Return a copy keeping only the named feature scopes (plus
        their ``object_ids``) -- e.g. to restrict :meth:`to_sqlite`/
        :meth:`to_duckdb`/:meth:`to_parquet` to a chosen subset of this
        result's object tables (mirrors real CellProfiler's own
        ``ExportToDatabase`` "export measurements for all/no/selected
        objects" setting; see ``spec/database-export.md``).
        ``relationships`` passes through unchanged -- it carries no
        scope of its own, and none of the wide-table export methods
        read it."""
        selected = set(scopes)
        return MeasurementResult(
            image_id=self.image_id,
            features={k: v for k, v in self.features.items() if k in selected},
            object_ids={k: v for k, v in self.object_ids.items() if k in selected},
            relationships=self.relationships,
        )

    def to_csv(self, path: str | Path) -> None:
        """Write long-form feature rows with nonfinite values kept as strings."""
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("image_id", "scope", "object_id", "feature", "value"),
            )
            writer.writeheader()
            writer.writerows(self.rows())

    def to_arrow(self, path: str | Path) -> None:
        """Write every feature row as an Arrow IPC (Feather v2) file.

        One row per object per feature column, the same long-form shape as
        :meth:`rows`/:meth:`to_csv`, covering all feature data (every
        scope, object, and column) in a single typed Arrow table --
        ``image_id``/``scope``/``feature`` as strings, ``object_id`` as
        int64, ``value`` as float64 (NaN/Infinity preserved natively,
        unlike :meth:`to_json`'s string markers).
        """
        import pyarrow as pa
        from pyarrow import feather

        rows = self.rows()
        table = pa.table(
            {
                "image_id": pa.array(
                    [row["image_id"] for row in rows], type=pa.string()
                ),
                "scope": pa.array([row["scope"] for row in rows], type=pa.string()),
                "object_id": pa.array(
                    [row["object_id"] for row in rows], type=pa.int64()
                ),
                "feature": pa.array([row["feature"] for row in rows], type=pa.string()),
                "value": pa.array([row["value"] for row in rows], type=pa.float64()),
            }
        )
        feather.write_feather(table, str(path))

    def to_parquet(
        self,
        path: str | Path,
        *,
        mode: str = "long",
        join: str = "outer",
        if_exists: str = "append",
        max_rows_per_file: int = 100_000,
    ) -> None:
        """Write this result's feature data as real Apache Parquet file(s).

        ``mode="long"`` (the default): ``path`` is a single file, written
        in the same long-form shape as :meth:`to_arrow` (one row per
        object per feature column).

        ``mode="wide"``: the traditional, tabular shape -- ``path`` is a
        *directory* (created if missing); one file per feature scope is
        written into it, ``Per_Image.parquet`` plus one
        ``Per_<ObjectName>.parquet`` per object set (CellProfiler's own
        ``ExportToDatabase`` table-naming convention, matching
        :meth:`to_duckdb`/:meth:`to_sqlite`'s own tables exactly). Each
        file has ``image_id`` (and, for object scopes, ``object_id``)
        columns plus one real, named column per feature -- feature
        names are the column names, no extra lookup needed.

        ``mode="array"``: same per-scope files and id columns as
        ``"wide"``, but every feature is packed into a single
        ``features`` column instead of one column each, as one
        fixed-size list per row (Arrow ``FixedSizeListArray`` -- the
        most direct way Parquet/Arrow represents a per-row numeric
        vector of known length; it round-trips straight into a dense
        ``(n_rows, n_features)`` numpy array with no separate decoding
        step). Reading a bare feature array back tells you nothing
        about *which* feature each position is; the file's own Parquet
        schema metadata carries the exact ordered feature name list for
        that column, read back with :func:`read_parquet_feature_names`
        -- no sidecar file to lose or fall out of sync.

        ``mode="joined"``: a single combined, `CytoTable
        <https://github.com/cytomining/CytoTable>`_-style wide table --
        one row per object, every object scope joined together on
        ``object_id`` (``join="outer"``, the default: every object_id
        seen in *any* scope, with nulls where a scope has no matching
        object; ``join="inner"``: only object_ids present in *every*
        scope), each scope's feature columns prefixed with that scope's
        name (``Cells_Intensity_MeanIntensity_DNA``,
        ``Nuclei_AreaShape_Area``, ...) to avoid name collisions between
        scopes, with every ``Image``-scope feature broadcast onto every
        row with an ``Image_`` prefix. This assumes matching
        ``object_id`` values across scopes identify the same physical
        object -- true whenever one object set was derived from another
        (as most of this project's own fixtures are; see
        ``spec/compatibility.md``), *not* guaranteed for independently
        segmented object sets with no declared parent-child
        relationship (`cp_anywhere` has no ``RelateObjects``
        equivalent). ``Neighbors`` relationship rows have no natural
        per-object column and are not included.

        ``path`` may be a single ``.parquet`` file (one write, no
        pagination beyond Parquet's own row groups) or a directory, in
        which case this writes a real Parquet *dataset* via
        ``pyarrow.dataset.write_dataset`` -- ``max_rows_per_file``
        bounds each fragment file's row count (Parquet's own built-in
        pagination mechanism, not a bespoke chunking scheme; the
        100,000-row default is a common balance between per-file
        overhead and query/memory efficiency). Calling this repeatedly
        with a directory ``path`` -- the usual way to build one combined
        multi-image dataset, since :func:`measure` always processes one
        image at a time -- defaults to ``if_exists="append"``, adding a
        new, uniquely-named fragment file each call without touching
        existing ones; ``if_exists="replace"`` deletes the directory
        first.
        """
        import pyarrow as pa
        import pyarrow.parquet

        if mode not in ("long", "wide", "array", "joined"):
            msg = f"mode must be 'long', 'wide', 'array', or 'joined', not {mode!r}"
            raise ValueError(msg)

        if mode == "long":
            rows = self.rows()
            table = pa.table(
                {
                    "image_id": pa.array(
                        [row["image_id"] for row in rows], type=pa.string()
                    ),
                    "scope": pa.array([row["scope"] for row in rows], type=pa.string()),
                    "object_id": pa.array(
                        [row["object_id"] for row in rows], type=pa.int64()
                    ),
                    "feature": pa.array(
                        [row["feature"] for row in rows], type=pa.string()
                    ),
                    "value": pa.array(
                        [row["value"] for row in rows], type=pa.float64()
                    ),
                }
            )
            pyarrow.parquet.write_table(table, str(path))
            return

        if mode == "joined":
            if if_exists not in ("append", "replace"):
                raise ValueError("if_exists must be 'append' or 'replace'")
            table = self._joined_table(join=join)
            target = Path(path)
            if target.suffix == ".parquet":
                pyarrow.parquet.write_table(
                    table, str(target), row_group_size=max_rows_per_file
                )
                return
            import shutil
            import uuid

            import pyarrow.dataset

            if if_exists == "replace" and target.exists():
                shutil.rmtree(target)
            pyarrow.dataset.write_dataset(
                table,
                str(target),
                format="parquet",
                basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
                max_rows_per_file=max_rows_per_file,
                max_rows_per_group=max_rows_per_file,
                existing_data_behavior="overwrite_or_ignore",
            )
            return

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        id_column_names = ("image_id", "object_id")
        pyarrow_type = {
            "TEXT": pa.string(),
            "INTEGER": pa.int64(),
            "REAL": pa.float64(),
        }

        if mode == "wide":
            for table_name, columns in self._wide_tables():
                arrow_table = pa.table(
                    {
                        name: pa.array(values, type=pyarrow_type[sql_type])
                        for name, (sql_type, values) in columns.items()
                    }
                )
                pyarrow.parquet.write_table(
                    arrow_table, str(directory / f"{table_name}.parquet")
                )
            return

        for table_name, columns in self._wide_tables():
            feature_names = [name for name in columns if name not in id_column_names]
            row_count = len(next(iter(columns.values()))[1])
            feature_rows = [
                [float(columns[name][1][row]) for name in feature_names]
                for row in range(row_count)
            ]
            arrays = {
                name: pa.array(
                    columns[name][1],
                    type=pa.string() if name == "image_id" else pa.int64(),
                )
                for name in id_column_names
                if name in columns
            }
            arrays["features"] = pa.array(
                feature_rows, type=pa.list_(pa.float64(), len(feature_names))
            )
            arrow_table = pa.table(arrays)
            arrow_table = arrow_table.replace_schema_metadata(
                {
                    **(arrow_table.schema.metadata or {}),
                    _FEATURE_NAMES_METADATA_KEY: json.dumps(feature_names).encode(),
                }
            )
            pyarrow.parquet.write_table(
                arrow_table, str(directory / f"{table_name}.parquet")
            )

    def _joined_table(self, *, join: str = "outer") -> pa.Table:
        """Build the combined, cytotable-style wide table :meth:`to_parquet`
        (``mode="joined"``) writes out -- one row per object_id, joined
        across every object scope, columns prefixed by scope name, with
        ``Image``-scope features broadcast onto every row. Returns a
        ``pyarrow.Table``.
        """
        import pyarrow as pa

        if join not in ("outer", "inner"):
            msg = f"join must be 'outer' or 'inner', not {join!r}"
            raise ValueError(msg)

        object_scopes = [scope for scope in self.features if scope != "Image"]
        if not object_scopes:
            msg = "joined mode needs at least one object scope with measurements"
            raise ValueError(msg)

        id_sets = [
            {int(value) for value in self.object_ids[scope]} for scope in object_scopes
        ]
        if join == "outer":
            all_ids = sorted(set().union(*id_sets))
        else:
            all_ids = sorted(set.intersection(*id_sets))

        columns: dict[str, list] = {
            "image_id": [self.image_id] * len(all_ids),
            "object_id": list(all_ids),
        }
        if "Image" in self.features:
            for name, value in self.features["Image"].items():
                columns[f"Image_{name}"] = [value] * len(all_ids)
        for scope in object_scopes:
            scope_ids = [int(value) for value in self.object_ids[scope]]
            index_by_id = {
                object_id: index for index, object_id in enumerate(scope_ids)
            }
            for name, values in self.features[scope].items():
                array = np.asarray(values)
                columns[f"{scope}_{name}"] = [
                    float(array[index_by_id[object_id]])
                    if object_id in index_by_id
                    else None
                    for object_id in all_ids
                ]
        return pa.table(columns)

    def _scope_columns(self, scope: str) -> dict[str, tuple[str, list]]:
        """``{feature_name: (sql_type, column_values)}`` for one feature
        scope, typed from each column's actual numpy/Python dtype
        (``"INTEGER"`` for integer-typed features such as neighbor
        counts, ``"REAL"`` for everything else)."""
        columns: dict[str, tuple[str, list]] = {}
        for name, value in self.features[scope].items():
            if scope == "Image":
                is_integer = isinstance(value, (int, np.integer)) and not isinstance(
                    value, bool
                )
                columns[name] = ("INTEGER" if is_integer else "REAL", [value])
            else:
                array = np.asarray(value)
                is_integer = array.dtype.kind in "iu"
                columns[name] = (
                    "INTEGER" if is_integer else "REAL",
                    [int(v) if is_integer else float(v) for v in array],
                )
        return columns

    def _wide_tables(self) -> list[tuple[str, dict[str, tuple[str, list]]]]:
        """Return ``(table_name, {column: (sql_type, values)})`` for
        ``Per_Image`` plus one ``Per_<ObjectName>`` table per object set
        with measurements -- CellProfiler's own ``ExportToDatabase``
        table-naming convention, used by :meth:`to_duckdb`/:meth:`to_sqlite`.
        """
        tables: list[tuple[str, dict[str, tuple[str, list]]]] = []
        if "Image" in self.features:
            columns: dict[str, tuple[str, list]] = {
                "image_id": ("TEXT", [self.image_id])
            }
            columns.update(self._scope_columns("Image"))
            tables.append(("Per_Image", columns))
        for scope in self.features:
            if scope == "Image":
                continue
            object_ids = self.object_ids[scope]
            row_count = len(object_ids)
            columns = {
                "image_id": ("TEXT", [self.image_id] * row_count),
                "object_id": ("INTEGER", [int(value) for value in object_ids]),
            }
            columns.update(self._scope_columns(scope))
            tables.append((f"Per_{scope}", columns))
        return tables

    def to_duckdb(self, path: str | Path, *, if_exists: str = "append") -> None:
        """Write/append wide ``Per_Image``/``Per_<ObjectName>`` tables
        (CellProfiler's own ``ExportToDatabase`` naming convention, one
        row per image or object, one column per feature) into a DuckDB
        database file.

        ``if_exists="append"`` (the default) adds this result's rows to
        any existing tables at ``path`` -- the usual way to accumulate
        one combined dataset across many images, since :func:`measure`
        itself always processes one image at a time. ``"replace"``
        drops and recreates every table first.
        """
        import duckdb
        import pyarrow as pa

        if if_exists not in ("append", "replace"):
            raise ValueError("if_exists must be 'append' or 'replace'")
        pyarrow_type = {
            "TEXT": pa.string(),
            "INTEGER": pa.int64(),
            "REAL": pa.float64(),
        }

        connection = duckdb.connect(str(path))
        try:
            for table_name, columns in self._wide_tables():
                arrow_table = pa.table(
                    {
                        name: pa.array(values, type=pyarrow_type[sql_type])
                        for name, (sql_type, values) in columns.items()
                    }
                )
                cursor = connection.cursor()
                if if_exists == "replace":
                    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"').fetchall()
                exists = (
                    len(
                        cursor.execute(
                            "SELECT 1 FROM information_schema.tables "
                            "WHERE table_name = ?",
                            [table_name],
                        ).fetchall()
                    )
                    > 0
                )
                cursor.register("_cp_anywhere_rows", arrow_table)
                if exists:
                    cursor.execute(
                        f'INSERT INTO "{table_name}" SELECT * FROM _cp_anywhere_rows'
                    ).fetchall()
                else:
                    cursor.execute(
                        f'CREATE TABLE "{table_name}" AS '
                        "SELECT * FROM _cp_anywhere_rows"
                    ).fetchall()
                cursor.unregister("_cp_anywhere_rows")
                cursor.close()
        finally:
            connection.close()

    def to_sqlite(self, path: str | Path, *, if_exists: str = "append") -> None:
        """Write/append the same wide ``Per_Image``/``Per_<ObjectName>``
        tables as :meth:`to_duckdb`, but as SQLite `STRICT tables
        <https://www.sqlite.org/stricttables.html>`_ -- every column has
        a declared, enforced type (``TEXT``/``INTEGER``/``REAL``) instead
        of SQLite's legacy, permissive type affinity. Requires SQLite
        3.37+ (``STRICT`` syntax support); raises :class:`RuntimeError`
        naming the running SQLite's version otherwise, rather than
        failing on the first ``CREATE TABLE`` with a confusing syntax
        error.

        ``if_exists`` has the same "append" (default, accumulate across
        images) / "replace" (drop and recreate) meaning as
        :meth:`to_duckdb`.
        """
        import sqlite3

        if sqlite3.sqlite_version_info < (3, 37, 0):
            msg = (
                "SQLite STRICT tables require SQLite >= 3.37; this "
                f"Python's sqlite3 module reports {sqlite3.sqlite_version}"
            )
            raise RuntimeError(msg)
        if if_exists not in ("append", "replace"):
            raise ValueError("if_exists must be 'append' or 'replace'")

        connection = sqlite3.connect(str(path))
        try:
            for table_name, columns in self._wide_tables():
                if if_exists == "replace":
                    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                column_defs = ", ".join(
                    f'"{name}" {sql_type}' for name, (sql_type, _) in columns.items()
                )
                connection.execute(
                    f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_defs}) STRICT'
                )
                names = list(columns.keys())
                placeholders = ", ".join("?" for _ in names)
                rows_to_insert = list(
                    zip(*(values for _, values in columns.values()), strict=True)
                )
                connection.executemany(
                    f'INSERT INTO "{table_name}" ({", ".join(names)}) '
                    f"VALUES ({placeholders})",
                    rows_to_insert,
                )
            connection.commit()
        finally:
            connection.close()

    def to_json(self, path: str | Path) -> None:
        """Write the reference-compatible feature and relationship structure."""

        def encode(value: object) -> object:
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return [encode(item) for item in value]
            if isinstance(value, np.generic):
                return encode(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return {
                    "nonfinite": "NaN"
                    if np.isnan(value)
                    else "+Infinity"
                    if value > 0
                    else "-Infinity"
                }
            return value

        payload = {
            "image_id": self.image_id,
            "features": encode(self.features),
            "object_ids": encode(self.object_ids),
            "relationships": self.relationships,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _image_array(value: object, shape: tuple[int, ...] | None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3) or 0 in array.shape:
        raise ValueError("images must have two or three nonzero dimensions")
    if shape is not None and array.shape != shape:
        raise ValueError("all inputs must have the same shape")
    if array.dtype.kind not in "iuf" or array.dtype.itemsize > MAX_IMAGE_DTYPE_ITEMSIZE:
        raise TypeError("images must have a supported real numeric dtype")
    return np.ascontiguousarray(array, dtype=np.float64)


def _label_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("all inputs must have the same shape")
    if array.dtype.kind not in "iu":
        raise TypeError("labels must have an integer dtype")
    if array.size and (np.min(array) < 0 or np.max(array) > MAX_LABEL):
        raise ValueError(f"labels must be between zero and {MAX_LABEL}")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _validate_families(families: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(DEFAULT_FAMILIES if families is None else families)
    if len(selected) != len(set(selected)) or any(
        name not in FAMILIES for name in selected
    ):
        raise ValueError(f"families must be drawn from {FAMILIES}")
    return selected


def _convert_images(
    images: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    converted: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name, value in images.items():
        if not isinstance(name, str) or not name:
            raise ValueError("image names must be nonempty strings")
        array = _image_array(value, shape)
        shape = array.shape
        converted[name] = array
    if shape is None:
        raise ValueError("images and objects must be nonempty")
    return converted, shape


def _convert_objects(
    objects: Mapping[str, object], shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    if any(not isinstance(name, str) or not name for name in objects):
        raise ValueError("object names must be nonempty strings")
    return {name: _label_array(value, shape) for name, value in objects.items()}


def _convert_mask(mask: object | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    converted = np.asarray(mask)
    if converted.shape != shape or converted.dtype.kind != "b":
        raise ValueError("mask must be a boolean array with the image shape")
    return np.ascontiguousarray(converted)


def measure(
    images: Mapping[str, object],
    objects: Mapping[str, object],
    *,
    mask: object | None = None,
    image_id: str = "Image",
    families: Sequence[str] | None = None,
    neighbors_scopes: Sequence[str] | None = None,
    neighbors_full_objects: Mapping[str, object] | None = None,
    size_shape_calculate_zernikes: bool = True,
    size_shape_calculate_advanced: bool = False,
) -> MeasurementResult:
    """Measure 2D or 3D arrays with the selected CellProfiler 4.2.8 settings.

    Float64 C-contiguous images are borrowed by Rust. Other numeric images and
    strided arrays are converted once. Do not mutate inputs during this call.

    The original three default families are unchanged. Select
    ``families=["object_size_shape"]`` explicitly for 2D shape measurements.
    ``size_shape_calculate_zernikes=True`` adds 30 Zernike columns to the
    25 standard columns; ``size_shape_calculate_advanced=True`` adds 53
    moment/inertia columns. Shape measurements ignore the image mask and
    require consecutive object IDs starting at 1. Sparse shape labels and
    3D shape requests raise errors, not misaligned or substitute columns.

    ``neighbors_scopes`` selects which object set(s) ``MeasureObjectNeighbors``
    runs against itself (CellProfiler's own ``Adjacent`` distance method;
    cross-object-set and non-"Adjacent" neighbors remain out of scope -- see
    ``spec/compatibility.md``). Defaults to ``("Cells",)`` when the
    ``"neighbors"`` family is selected and ``neighbors_scopes`` is omitted,
    matching this project's original declared scope exactly, byte-for-byte
    (every pre-existing fixture, golden comparison, and caller is
    unaffected). Pass e.g. ``neighbors_scopes=("Nuclei", "Cells")`` to
    compute it for multiple object sets in one call instead -- each one
    independently, with its own feature columns and relationship rows
    tagged by an ``"object_name"`` key, matching how real CellProfiler
    pipelines invoke ``MeasureObjectNeighbors`` once per object set.

    ``neighbors_full_objects`` supplies each scope's diagnostic geometry
    (CellProfiler's ``objects.small_removed_segmented``). Both primary and
    secondary segmentation results expose it as ``small_removed_labels``.
    The core matches diagnostic IDs to retained IDs by pixel overlap.
    Diagnostic objects omitted from the kept set can affect counts,
    touching percentages, distances, and angles, but do not get output
    rows or relationship edges. Closest-object numbers are recomputed
    among retained objects, matching CellProfiler. Which excluded objects
    appear in the diagnostic layer depends on the segmentation module;
    primary segmentation's layer already excludes border objects.
    A scope with no entry here (the default) uses its kept labels.
    """
    from cp_anywhere import _native

    if not images or not objects:
        raise ValueError("images and objects must be nonempty")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    selected = _validate_families(families)
    converted_images, shape = _convert_images(images)
    converted_objects = _convert_objects(objects, shape)
    resolved_neighbors_scopes: list[str] = []
    if neighbors_scopes is not None:
        resolved_neighbors_scopes = list(neighbors_scopes)
        if not resolved_neighbors_scopes:
            raise ValueError("neighbors_scopes must be nonempty if given")
        missing = [name for name in resolved_neighbors_scopes if name not in objects]
        if missing:
            raise ValueError(f"neighbors_scopes {missing} are not provided object sets")
    elif "neighbors" in selected and "Cells" not in objects:
        raise ValueError("neighbors require a Cells object set")
    converted_full_objects: dict[str, np.ndarray] = {}
    if neighbors_full_objects is not None:
        converted_full_objects = _convert_objects(neighbors_full_objects, shape)
        active_scopes = resolved_neighbors_scopes or (
            ["Cells"] if "neighbors" in selected else []
        )
        extra = [name for name in converted_full_objects if name not in active_scopes]
        if extra:
            raise ValueError(
                f"neighbors_full_objects {extra} are not requested neighbors scopes"
            )
    converted_mask = _convert_mask(mask, shape)
    raw = _native.measure(
        shape,
        converted_images,
        converted_objects,
        converted_mask,
        image_id,
        selected,
        resolved_neighbors_scopes or None,
        converted_full_objects or None,
        size_shape_calculate_zernikes,
        size_shape_calculate_advanced,
    )
    features = raw["features"]
    if "Image" in features:
        features["Image"] = {
            name: value[0].item() for name, value in features["Image"].items()
        }
    return MeasurementResult(
        image_id, features, raw["object_ids"], raw["relationships"]
    )


def read_parquet_feature_names(path: str | Path) -> list[str]:
    """Read back the ordered feature name list embedded in a
    ``MeasurementResult.to_parquet(..., mode="array")`` file's Parquet
    schema metadata -- the position of each name in the returned list
    matches the position of its value in that file's ``features``
    fixed-size-list column. Raises :class:`ValueError` naming the path
    if it has no such metadata (e.g. a plain ``mode="long"`` file).
    """
    import pyarrow.parquet

    schema = pyarrow.parquet.read_schema(str(path))
    raw = (schema.metadata or {}).get(_FEATURE_NAMES_METADATA_KEY)
    if raw is None:
        msg = (
            f"{path} has no embedded feature_names metadata -- was it "
            "written by MeasurementResult.to_parquet(mode='array')?"
        )
        raise ValueError(msg)
    return json.loads(raw)
