# Database export

`MeasurementResult` can be written directly to a real Parquet file, a
DuckDB database, or a SQLite database (using `STRICT` tables), on top
of the existing `to_csv`/`to_arrow`/`to_json`:

```python
result.to_parquet("result.parquet")  # same long-form rows as to_arrow
result.to_duckdb("results.duckdb")  # wide Per_Image / Per_Cells / ...
result.to_sqlite("results.sqlite")  # same, as SQLite STRICT tables
```

`to_duckdb`/`to_sqlite` write **wide** tables named `Per_Image`/
`Per_<ObjectName>` -- matching real CellProfiler's own
`ExportToDatabase` naming convention -- and default to
`if_exists="append"`, so processing many images in a loop and calling
one of these once per image builds up a single combined dataset, the
way CellProfiler's own batch runs do. Pass `if_exists="replace"` to
start fresh instead.

`to_duckdb` needs the optional `duckdb` dependency
(`pip install cp-anywhere[duckdb]`); `to_parquet` and `to_sqlite` need
nothing beyond this project's own core dependencies.

For a dense, fixed-size feature array per object instead of long-form
rows (e.g. to load straight into a `(n_rows, n_features)` numpy
array), use `to_parquet(..., mode="array")`: `path` becomes a
directory, one `Per_Image.parquet`/`Per_<ObjectName>.parquet` file is
written per scope, and each file's `features` column is a single
fixed-size array per row. The exact ordered feature names for that
array are embedded in the file's own Parquet schema metadata (no
sidecar file to go out of sync) -- read them back with
`cp_anywhere.read_parquet_feature_names(path)`. The CLI exposes this
as `--output <directory> --parquet-mode array`.

`to_parquet(..., mode="wide")` writes the same per-scope files as
`mode="array"`, but as the traditional tabular shape instead -- one
real, named column per feature, no array/metadata lookup needed
(`--output <directory> --parquet-mode wide`).

For one combined, [CytoTable](https://github.com/cytomining/CytoTable)
-style table across every object scope -- one row per object, scopes
joined on `object_id`, feature columns prefixed by scope name -- use
`to_parquet(..., mode="joined")`. `path` is either a single `.parquet`
file or a directory, in which case it's written as a real,
`max_rows_per_file`-paginated Parquet dataset via
`pyarrow.dataset.write_dataset` that can accumulate across many images
with `if_exists="append"` (the default): `--output <file-or-directory> --parquet-mode joined --join outer|inner --max-rows-per-file N`.

The CLI dispatches on `--output`'s file extension the same way:

```sh
cp-anywhere --image DNA=dna.tif --objects Cells=labels.npy \
  --output results.sqlite --if-exists append
```

See `spec/database-export.md` for the exact schema (column names and
types), why SQLite specifically needs `STRICT` tables, and how it's
verified (`tests/test_database_export.py`).
