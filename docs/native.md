# M3 native Rust core

## Result

The Rust core (`crates/cp_anywhere_core`) implements the complete declared
scope from `spec/compatibility.md`: `MeasureImageIntensity`,
`MeasureObjectIntensity` over `Cells`/`Nuclei` × `DNA`/`Actin`, and
`MeasureObjectNeighbors` same-set `Adjacent` with directed relationship
rows. `crates/cp_anywhere_python` is a thin PyO3/NumPy binding with no
scientific logic of its own; `src/cp_anywhere/api.py` validates inputs and
exposes `cp_anywhere.measure(...)`.

`scripts/parity/compare_native.py` compares the Rust result against all 13
saved golden fixtures and all 119 declared columns plus the directed
relationship table: **1,560 of 1,560 rows pass** (1,535 exact, 25 within
the `spec/features.yaml` float64 tolerance). The 25 tolerance rows are all
`MeasureImageIntensity` scalar reductions with errors between 1e-17 and
1e-14 — float64 summation-order variance, not a scope or semantics bug.
Full counts and reasons are recorded in `spec/native-coverage.yaml` and
`benchmarks/reports/native-parity.json`.

```sh
cargo test --manifest-path crates/cp_anywhere_core/Cargo.toml
env -u CONDA_PREFIX uv run maturin develop --release --uv
uv run --frozen python scripts/parity/compare_native.py
CP_ANYWHERE_REFERENCE_PYTHON=/path/to/reference/venv/bin/python \
  PYTHONPATH=/path/to/pinned/CellProfiler \
  uv run --frozen pytest
```

`env -u CONDA_PREFIX` is required on hosts where both `CONDA_PREFIX` and
`VIRTUAL_ENV` are set; maturin refuses to build with both present.

## Hard-case source decisions

These follow `spec/reference-semantics.md`, which records the exact
original behavior read from the pinned CellProfiler 4.2.8 and NumPy 1.23.5
sources, not a cleaned-up reimplementation:

- Integer images are promoted to `f64` without dtype-maximum rescaling,
  matching the reference runner's `Image(..., convert=False)`.
- Object-quantile and MAD formulas use `n * fraction` (not `(n-1) * fraction`) and `n / dimensions`, matching the original module exactly.
- Maximum-intensity coordinate ties are resolved by a literal Rust port of
  NumPy 1.23.5's introsort (`numpy_argsort`/`argsort_range`/
  `numpy_heapsort` in `lib.rs`), derived from
  `numpy/core/src/npysort/quicksort.cpp` and `npysort_heapsort.h` at
  `de82cd9468704a033702974010ee7e7efc85b393` (BSD-3-Clause; see
  `licenses/numpy.txt` and `THIRD_PARTY_NOTICES.md`). A stable per-object
  sort does not reproduce the golden tie-break positions; this port does.
- Same-set adjacent neighbors use only the first two center axes even in
  3D (z/y, omitting x) for nearest-distance and angle, a separate ordering
  for distance/angle selection versus the ID ordering, and the original
  `centrosome.outline.outline`-equivalent first-two-axes diagonal outline
  (also applied unchanged in 3D). These are preserved reference quirks,
  not defects introduced by this port.
- Sparse object IDs keep their array position through the maximum label;
  missing slots are not compacted.

## Resource and memory contract

- Inputs: 2D or 3D only, nonzero shape, checked shape-product
  multiplication (`checked_mul`), matching channel/object/mask lengths,
  labels must be unsigned and `<= MAX_LABEL` (1,000,000; a documented
  limit, not an attempt to allocate `u32::MAX` rows).
- `crates/cp_anywhere_python/src/lib.rs` borrows C-contiguous `f64` image
  arrays and `u32` label arrays directly as slices — no copy for that
  common case. `src/cp_anywhere/api.py` performs exactly one conversion
  via `np.ascontiguousarray` for other numeric dtypes or non-contiguous
  input; it does not claim zero-copy for those cases.
- The kernel runs with `py.detach(...)` (PyO3's GIL release), so Rust work
  does not hold the GIL. Callers must not mutate input arrays while
  `measure()` is running, since the Rust side holds borrowed slices for
  the call's duration.
- Rejected inputs: 1D or higher-than-3D shapes, boolean/float label
  arrays, non-real-numeric image dtypes, mismatched shapes, labels above
  `MAX_LABEL`, and (when neighbors are selected) no `Cells` object set.
- The core (`cp_anywhere_core`) has no Python, filesystem, or platform
  dependency: `Input`/`ResultTable` are plain Rust types over borrowed
  slices, which keeps a future WASM target possible without a rewrite.

## API

`cp_anywhere.measure(images, objects, *, mask=None, image_id="Image", families=None)` returns a frozen `MeasurementResult` with `.features`
(mirroring the golden JSON structure: `Image` scope holds scalars, object
scopes hold NumPy arrays aligned to `.object_ids`), `.object_ids`, and
`.relationships` (directed edges with both image/object IDs). `.rows()`
gives a long table; `.to_csv()`/`.to_json()` write it and the
reference-shaped structure respectively, with nonfinite floats encoded
explicitly (`{"nonfinite": "NaN"}`, `"+Infinity"`, `"-Infinity"`) rather
than silently dropped or turned into `null`. `families` restricts
computation to a subset of `("image_intensity", "object_intensity", "neighbors")` so later benchmarking can time an equivalent subset without
computing unselected families.

The CLI (`cp-anywhere --image NAME=PATH --objects NAME=PATH [--mask PATH] --output PATH`) loads saved NPY arrays and writes CSV or JSON depending on
the output suffix.

## Wheel build and installed-wheel smoke test

`maturin build --release` produced
`cp_anywhere-0.1.0-cp311-abi3-macosx_11_0_arm64.whl`. It was installed into
a **fresh, isolated `uv venv`** with no project checkout on `sys.path`. The
imported module resolved to that venv's `site-packages` (not the source
tree), and `cp_anywhere.measure(...)` on the `touching_gradient` fixture
returned all 119 declared columns. This proves the installed wheel, not
just the editable development install, produces the full declared scope.

`cargo fmt --check` and `cargo clippy --all-targets` are clean for both
crates. `cargo test` passes the one Rust unit test (invalid shape and
over-limit label rejection). The full project `pytest` suite (22 tests,
including the real CellProfiler 4.2.8 replay) passes.

## Known limitations

- Only the one `rejects_bad_shape_and_large_id` Rust unit test exists; most
  correctness assurance currently comes from the Python-side golden parity
  script, not from Rust-level property tests (e.g. proptest) on the
  sorting or neighbor kernels.
- Timing against `cp_measure` and against a WASM build is explicitly out
  of scope for M3; see `docs/decisions/0002-rust-core.md` and M4/M6.
