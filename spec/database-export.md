# Database export

`MeasurementResult` gained three new export methods, alongside the
existing `to_csv`/`to_arrow`/`to_json`:

| Method       | Format                                | Extra dependency         |
| ------------ | ------------------------------------- | ------------------------ |
| `to_parquet` | Apache Parquet                        | none (`pyarrow` is core) |
| `to_duckdb`  | DuckDB database file                  | `cp-anywhere[duckdb]`    |
| `to_sqlite`  | SQLite database file, `STRICT` tables | none (stdlib `sqlite3`)  |

The CLI dispatches on `--output`'s extension, same as the existing
formats: `.parquet`, `.duckdb`, `.sqlite`/`.sqlite3`/`.db`.

## Parquet: a long-form file, same shape as `to_arrow`

`to_parquet` writes the exact same long-form rows as `to_csv`/
`to_arrow` (`image_id`, `scope`, `object_id`, `feature`, `value`), as a
real Parquet file via `pyarrow.parquet`. It is a drop-in alternative
file format, not a different schema.

## DuckDB and SQLite: wide `Per_Image`/`Per_<ObjectName>` tables

`to_duckdb` and `to_sqlite` instead write **wide** tables -- one row
per image or per object, one column per feature -- matching real
CellProfiler's own `ExportToDatabase` module's table-naming convention
(`Per_Image`, `Per_Cells`, `Per_Nuclei`, ...) so the exported database
looks like what a CellProfiler user already expects from a SQL export,
not a project-specific shape.

Each table's id columns are `image_id` (the string passed to
`measure(..., image_id=...)`, not a CellProfiler-style sequential
`ImageNumber` -- `cp_anywhere.measure()` has no concept of a pipeline
run's image ordering) and, for object tables, `object_id` (the same
1-based object numbering `MeasurementResult.object_ids` already uses).
Every feature column's SQL type (`REAL` or `INTEGER`) is taken from
that feature's actual numpy/Python dtype, not assumed to always be
`REAL`.

### Accumulating across images

`cp_anywhere.measure()` always processes one image at a time; real
multi-image datasets are built by calling `to_duckdb`/`to_sqlite`
**repeatedly**, once per image, into the same database file. Both
methods default to `if_exists="append"`: each call adds its rows to
whatever tables already exist (creating them on the first call),
rather than requiring the caller to pre-create a schema. Pass
`if_exists="replace"` to drop and recreate every table fresh instead.

## Why SQLite must use `STRICT` tables

SQLite's classic tables use *type affinity*: a column declared `REAL`
will silently store a `TEXT` value if you insert one. `STRICT` tables
(SQLite 3.37+) remove that behavior -- every column has a declared,
enforced type (`TEXT`/`INTEGER`/`REAL` here), and an insert that
doesn't match raises `sqlite3.IntegrityError` naming the exact column
and the offending type, instead of silently corrupting a feature
column. Every table `to_sqlite` creates uses `CREATE TABLE ... STRICT`.
`to_sqlite` checks `sqlite3.sqlite_version_info` up front and raises a
clear `RuntimeError` naming the running SQLite version if it predates
3.37, rather than failing confusingly on the first `CREATE TABLE`.

DuckDB has no non-strict mode to opt out of in the first place --
every DuckDB column is always strongly typed -- so `to_duckdb` needs
no equivalent keyword.

## Parquet, `mode="array"`: fixed-size feature-array storage

`to_parquet` also has a second mode, for workflows that want each
object's feature vector as a single dense array column (e.g. to load
straight into a `(n_rows, n_features)` numpy array) instead of
long-form rows:

```python
result.to_parquet("output_dir", mode="array")
```

Parquet files hold one table each, and the scopes genuinely have
different numbers of features, so `path` is a **directory** in this
mode: one file per scope is written into it, using the same
`Per_Image`/`Per_<ObjectName>` naming as `to_duckdb`/`to_sqlite`. Each
file has `image_id` (and, for object scopes, `object_id`) columns plus
a single `features` column -- an Arrow `FixedSizeListArray` (the
native Arrow/Parquet representation of a per-row numeric vector of a
known, fixed length; it round-trips directly into a dense numpy array
with `np.array(table.column("features").to_pylist())`, no padding or
ragged-list handling needed).

A bare array of numbers says nothing about which position is which
feature. Rather than a sidecar file that can silently drift out of
sync, the exact ordered feature name list is embedded directly in
that file's own Parquet schema metadata (key `cp_anywhere.feature_names`,
JSON-encoded) -- read it back with:

```python
from cp_anywhere import read_parquet_feature_names

feature_names = read_parquet_feature_names("output_dir/Per_Cells.parquet")
# feature_names[i] names column i of every row's `features` array
```

The CLI exposes this as `--output <directory> --parquet-mode array`.

## Parquet, `mode="joined"`: one combined CytoTable-style wide table

A fourth mode, for workflows that want one combined table across every
object scope -- matching the convention
[CytoTable](https://github.com/cytomining/CytoTable) uses to turn
CellProfiler's separate per-compartment tables into a single
analysis-ready table:

```python
result.to_parquet("joined.parquet", mode="joined")  # single file
result.to_parquet("joined_dataset", mode="joined")  # dataset directory
```

One row per object, every object scope joined together on `object_id`
(`join="outer"`, the default: every `object_id` seen in *any* scope,
with nulls where a scope has no matching object; `join="inner"`: only
`object_id`s present in *every* scope). Each scope's feature columns
are prefixed with that scope's name (`Cells_Intensity_MeanIntensity_DNA`,
`Nuclei_AreaShape_Area`, ...) to avoid collisions between scopes that
share feature names, and every `Image`-scope feature is broadcast onto
every row with an `Image_` prefix. `Neighbors` relationship rows have
no natural per-object column and are not included.

This assumes matching `object_id` values across scopes identify the
same physical object -- true whenever one object set was derived from
another, which is how most of this project's own fixtures are built
(see `spec/compatibility.md`), but **not** guaranteed for independently
segmented object sets with no declared parent-child relationship
(`cp_anywhere` has no `RelateObjects` equivalent). This is a named,
honest limitation, not a hidden one.

### File or dataset, with real Parquet-native pagination

`path` may be:

- a single `.parquet` file -- one write, `max_rows_per_file` becomes
  that file's internal Parquet row-group size (Parquet's own built-in
  pagination, not a bespoke chunking scheme);
- a directory -- written as a real Parquet *dataset* via
  `pyarrow.dataset.write_dataset`, one or more row-group-bounded
  fragment files (`max_rows_per_file` caps each fragment's row count;
  100,000 by default, a common balance of per-file overhead against
  query/memory efficiency). Reading it back uses
  `pyarrow.dataset.dataset(path).to_table()`, which transparently
  unions every fragment file.

Calling this repeatedly with a directory `path` -- the usual way to
build one combined multi-image dataset, since `measure()` always
processes one image at a time -- defaults to `if_exists="append"`,
adding a new, uniquely-named fragment file each call without touching
existing ones; `if_exists="replace"` deletes the directory first.

The CLI exposes this as `--parquet-mode joined --join outer|inner --max-rows-per-file N`, with `--output` either a `.parquet` file or a
directory.

## Reproduce

```sh
uv run --frozen pytest tests/test_database_export.py
```

## Driving these from a real `.cppipe` pipeline's `ExportToDatabase` module

`cp_anywhere.load_pipeline(...).export_steps` translates a real
`ExportToDatabase` module's settings into an `ExportToDatabaseStep`,
which has its own `write(result, path)` calling `to_sqlite` with that
step's `objects_choice`/`objects` object-table selection and
`if_exists` overwrite policy already applied:

```python
pipeline = cp_anywhere.load_pipeline("my_pipeline.cppipe")
for step in pipeline.export_steps:
    step.write(result, step.sqlite_file)
```

Only SQLite output and "one table per object type" are supported --
real CellProfiler's MySQL connection, its default "Single object
table"/"Single object view" combined-table modes (use
`to_parquet(mode="joined")` for that shape instead), its table
prefixes, its CellProfiler Analyst properties file, its
object-relationship table, and its per-image aggregate
mean/median/std-dev columns are all explicit, documented gaps --
`load_pipeline` raises a clear `CppipeError` naming the unsupported
setting rather than silently ignoring it. See
`ExportToDatabaseStep`'s own docstring and
`spec/cppipe-compatibility.md` for the full settings-mapping table.
