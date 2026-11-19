# M4 WASM feasibility and browser integration spike

## Result

The Rust core compiles to `wasm32-unknown-unknown` with its default
features, including the `.cppipe` executor. The core does not depend on
Python or browser bindings.
`crates/cp_anywhere_wasm` is a thin `wasm-bindgen` binding (JSON in/out for
small metadata, typed arrays for the numeric payload). The build script
`scripts/wasm/build_package.py` uses wasm-pack and names the npm package
`cp-anywhere`. The Vite example
(`examples/browser-vite`) that runs the module inside a dedicated web
worker.

**Both measurement and segmentation run entirely client-side with
verified parity against the real CellProfiler 4.2.8 oracle.** A real,
reproducible headless-Chromium run (Playwright, not a mock DOM;
`examples/browser-vite/run-browser-check.mjs`) executes the worker
against:

- All **13** declared measurement golden fixtures (2D and 3D) via
  `measure()`, compared by `scripts/parity/compare_wasm.py` --
  **1,560 of 1,560 rows pass** (1,533 exact, 27 within the declared
  float64 tolerance; zero failures). 13 fixtures × 120 rows (119
  features + 1 relationship table) -- the identical row count the
  native build passes (`docs/native.md`), closing M5's "100% ... in
  native Rust, Python, and WASM" gate for the current declared
  manifest.
- All **100** segmentation golden fixtures (`fixtures/segmentation/`)
  via `identifyPrimaryObjects()`, compared by
  `scripts/parity/compare_wasm_segmentation.py` -- **100 of 100 pass**
  exactly on labels and object count (one fixture's scalar threshold
  uses the same documented, narrow, label-safe float32/float64
  tolerance exception as the native/Python build; see
  `tests/segmentation/test_reference_alignment.py`'s
  `_KNOWN_THRESHOLD_MAP_TOLERANCE_GAP`) -- see
  `docs/segmentation-coverage.md`.

The 27 measurement tolerance rows are the same `MeasureImageIntensity`
reduction-order families already seen in the native build
(`docs/native.md`), plus two `Neighbors_AngleBetweenNeighbors_Adjacent`
rows at 1e-14-scale absolute / 1e-16-scale relative error --
consistent with a transcendental-function (`acos`/`atan2`) last-bit
difference between wasm32's and native arm64's libm, not a computation
bug.

```sh
cargo fmt --manifest-path crates/cp_anywhere_wasm/Cargo.toml --check
cargo clippy --manifest-path crates/cp_anywhere_wasm/Cargo.toml --target wasm32-unknown-unknown --all-targets
python scripts/wasm/build_package.py

cd examples/browser-vite
npm install
uv run --frozen python ../../scripts/wasm/export_fixtures.py               # measurement fixtures (gitignored)
uv run --frozen python ../../scripts/wasm/export_segmentation_fixtures.py  # segmentation fixtures (gitignored)
npm run build
npx playwright install chromium   # once
npm run browser-check             # writes browser-results.json
cd ../..
uv run --frozen python scripts/parity/compare_wasm.py
uv run --frozen python scripts/parity/compare_wasm_segmentation.py
```

## Browser demo and npm package

`examples/browser-vite` is a real Vite project (not a static mockup): its
`src/worker.ts` imports the generated `cp-anywhere` package, fetches
each fixture's raw binary arrays (see below), calls `measure()` and
`identifyPrimaryObjects()`, and posts the results back; `src/main.ts`
drives the worker from the page and renders status in `#status`.
`npm run build` produces a genuine production bundle:

```text
dist/index.html                             0.46 kB
dist/assets/worker-*.js                     ~9.0 kB
dist/assets/cp_anywhere_wasm_bg-*.wasm    377.62 kB  (146.80 kB gzip-equivalent)
dist/assets/index-*.js                      ~2.5 kB
```

`npm run preview` (or `run-browser-check.mjs`'s own tiny static server)
serves this real build; nothing is dev-server-only or dynamically
recompiled per request.

## Segmentation: `identifyPrimaryObjects`

`identifyPrimaryObjects(pixels, rows, cols, mask, config_json)` shares
its exact settings surface with the Python binding
(`cp_anywhere::segmentation::build_config`/`ConfigParams` -- see
`spec/segmentation-modules.md`), so the same `config_json` field names
(`threshold_method`, `unclump_method`, `watershed_method`, ...) work
identically in both bindings, eliminating a second, driftable mapping.
Its result (`SegmentationResult`) is exposed as typed-array/scalar
getters (`labels`, `smallRemovedLabels`, `objectCount`, `threshold`,
`thresholdMap`, `guideThreshold`), not a JSON string: the label arrays
are the same size as the input image, unlike `measure()`'s much smaller
per-object feature output, so JSON-encoding them would be wasteful.
`identifyPrimaryObjectsJson(input)` is a JSON-string overload (pixels as
a plain array) kept for parity with `measureJson`, for debugging/small
inputs without constructing a typed array by hand.

`examples/zarr-onnx-duckdb` demonstrates this chained into a full
Zarr -> segmentation -> measurement -> Parquet pipeline in a real
browser (see `docs/browser-integration.md`) -- this is the shape a
browser-based consumer needing **both** segmentation and feature
extraction from cp-anywhere would use, not a theoretical API surface.

## Typed-array inputs, output conversion, memory

- **Input:** `measure(shape, image_names, images, object_names, objects, mask, image_id, families)` takes real typed arrays
  (`Uint32Array`/`Float64Array`/`Uint8Array`) across the wasm-bindgen
  boundary — the plan's "typed-array inputs" requirement. `wasm-bindgen`
  copies `&[T]` parameters into wasm linear memory once per call; this is
  the standard, documented cost of that FFI pattern (see
  `crates/cp_anywhere_wasm/src/lib.rs`), not something hidden.
- **Numeric fixture data is fetched as raw binary, not JSON.** The first
  export attempt used plain JSON and failed immediately: `nonfinite`'s
  pixel array contains literal NaN/Infinity, which are not valid JSON
  numbers (`JSON.parse` correctly rejects them). `scripts/wasm/export_fixtures.py`
  now writes small JSON metadata (shape, names, `image_id`) plus
  `.images.f64` / `.objects.u32` / `.mask.u8` raw binary blobs; the worker
  fetches each as an `ArrayBuffer` and constructs a typed-array view
  directly over the bytes, which preserves the exact IEEE754 bit pattern
  and is also the more realistic typed-array transfer path for a browser
  demo.
- **Output conversion:** `measure()` returns one JSON string matching
  `MeasurementResult.to_json()`'s shape exactly (same `{"nonfinite": "NaN"}`-style markers), so the same Python comparator
  (`scripts/parity/compare_cp_measure.py`'s `compare_feature`/
  `compare_relationships`) verifies both the native and the WASM result
  without a second, independently-trusted comparison implementation.
- **Memory:** `WebAssembly.Memory.buffer.byteLength` was 3,538,944 bytes
  (~3.46 MiB) after running all 13 fixtures sequentially in one worker
  instance — a post-hoc read, not a tracked peak (see Known limitations).

## Worker execution and browser compatibility

The module runs inside a dedicated `Worker({ type: "module" })`, loaded via
Vite's `new URL(...)` + `new Worker(...)` pattern, which Vite bundles
correctly (confirmed by the production build above). Testing used real
headless Chromium (Playwright, Chrome for Testing 159.0.8154.6); Firefox
and Safari were not tested in this spike (see Known limitations).
`crossOriginIsolated` / COOP-COEP were not required for this
single-threaded `wasm-bindgen` build.

## Native vs WASM comparison

Both the native build (`docs/native.md`) and this WASM build pass the same
golden fixtures exactly (modulo the same class of float64
reduction/transcendental-order tolerance). Measured per-call timing
(`benchmarks/reports/wasm-baseline.json`, 3 warmups / 10 runs, same four
representative workloads from `spec/benchmark-contract.yaml`):

| Fixture           | WASM median ms (worker round trip) |
| ----------------- | ---------------------------------: |
| touching_gradient |                               0.15 |
| real_cell_cc0     |                               2.85 |
| grid_medium       |                              11.80 |
| grid_large        |                             174.00 |

Each sample times one `measure()` call, which includes the in-WASM
`serde_json::to_string()` encoding of the full 119-column result — a
worker-round-trip cost, not an isolated kernel time. It is **not yet**
comparable to the native/`cp_measure` Python baseline
(`benchmarks/baseline/initial-cp-measure-timing.json`) on an
apples-to-apples kernel-only basis; that isolation, plus a real
cross-target regression gate, is M6's job. `spec/benchmark-contract.yaml`'s
`browser_worker_budget`, `browser_memory_budget`, and
`browser_package_size_budget` are updated from "provisional until M4" to
real first-measured numbers with a documented margin.

## Known limitations

- Only Chromium was tested (via Playwright). Firefox and Safari are
  untested; no WASM feature used here (no SIMD, no threads, no
  SharedArrayBuffer) is Chromium-specific, so cross-browser support is
  expected but not verified in this spike.
- Per-call timing includes JSON-encoding every result; kernel-only timing
  (e.g., via a numeric-only output mode) is left to M6.
- `wasm_linear_memory_bytes_after_run` is a single post-run read, not a
  tracked peak across the run.
- This spike did not test a `crossOriginIsolated`/threaded build; the
  current build is intentionally single-threaded, matching the plan's
  "test... browser compatibility" bar without requiring COOP/COEP
  deployment headers.
