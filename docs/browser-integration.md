# Browser integration: Zarr, ONNX Runtime Web, DuckDB-WASM

## What this is

`examples/zarr-onnx-duckdb/` is a real, end-to-end browser integration
proving the pieces named in the plan work together with cp-anywhere's own
WASM build, with **no mocked stage**:

1. **zarrita.js** reads a real OME-Zarr v2 array (`public/dna.zarr`,
   written by `scripts/zarr/export_demo_fixture.py`) over HTTP.
1. **onnxruntime-web** runs a real, checked ONNX graph
   (`public/threshold.onnx`) against that array, producing a reference
   threshold mask (logged for comparison; not used downstream -- see
   "Why the mask is only a reference" below).
1. **cp-anywhere**'s own `identifyPrimaryObjects` (the same
   CellProfiler-compatible, oracle-verified segmentation the Python
   binding exposes -- Otsu threshold + Intensity declumping/watershed in
   this demo) segments the same Zarr pixels into real object labels.
1. **cp-anywhere**'s `measure()` (this project's own Rust/WASM
   kernel, the same build verified in `docs/wasm.md`) measures the Zarr
   image using cp-anywhere's own segmentation output as the `Cells`
   object labels -- both segmentation and feature extraction now come
   from cp-anywhere, not just the latter.
1. **DuckDB-WASM** loads cp-anywhere's own measurement output into a SQL
   table and exports it as a real Parquet file (`COPY ... FORMAT PARQUET`).

`scripts/zarr/validate_export.py` then validates that Parquet file with
**native** DuckDB (not the WASM build under test) and cross-checks every
value -- and the segmentation's own `object_count`/`threshold` -- against
a fresh, from-source native Python `cp_anywhere.identify_primary_objects()`

- `cp_anywhere.measure()` call on the identical Zarr bytes -- proving the
  browser pipeline (both its segmentation and its measurement) produced
  numerically correct output, not just that it ran without an error.

## Why the mask is only a reference

The ONNX threshold model (`public/threshold.onnx`, a checked
`Greater(image, 0.5) -> Cast(..., INT32)` graph) still runs in this demo
to exercise the real Zarr -> ONNX -> cp-anywhere -> DuckDB pipeline named
in the plan, and its foreground pixel count is logged for comparison.
Its output is **not** fed into measurement: `main.ts` segments with
cp-anywhere's own `identifyPrimaryObjects` instead (step 3 above), so the
object labels measurement actually uses come from real CellProfiler-
compatible segmentation (with declumping), not a disconnected-foreground
threshold mask that would merge every touching object into a single
label. A future revision could use the ONNX path for a learned model
(e.g. StarDist) instead of the fixed-threshold demo graph; that remains
independent of cp-anywhere's own segmentation path either way.

## Run it

```sh
uv sync --group demo-assets
uv run --frozen python scripts/zarr/export_demo_fixture.py

cd examples/zarr-onnx-duckdb
npm install
npm run build
npm run browser-check   # real Playwright/Chromium run; writes
                         # cp_anywhere_features.parquet and browser-results.json
cd ../..

uv run --group demo-assets python scripts/zarr/validate_export.py
```

## Verified results

A real run (Playwright 1.65.1, Chrome for Testing, `npm run browser-check`)
on the `touching_gradient` fixture's DNA channel (12x12, normalized to
`[0, 1]`):

- Zarr read: shape `[12, 12]`, 144 pixels.
- ONNX inference (reference only): 72 foreground pixels (`mask > 0.5`).
- cp-anywhere `identifyPrimaryObjects()` (Otsu + Intensity
  declump/watershed, `min_diameter=4.0`): 1 object, threshold
  `0.486328125`.
- cp-anywhere `measure()`: 21 `object_intensity` columns for that
  object.
- DuckDB-WASM Parquet export: real `PAR1` magic bytes at both ends of the
  file, independently readable by native DuckDB.
- `scripts/zarr/validate_export.py`: all 21 feature values, plus the
  segmentation's own threshold, match a from-source native Python
  `cp_anywhere.identify_primary_objects()` + `cp_anywhere.measure()` run
  exactly (`rtol=1e-6, atol=1e-9`) -- zero mismatches.

## Known limitations

- **Network required at run time.** DuckDB-WASM fetches its `parquet`
  extension from `extensions.duckdb.org` on the first `LOAD parquet`; this
  demo does not vendor that extension. For a fully offline/CSP-restricted
  deployment, fetch and serve
  `https://extensions.duckdb.org/v1.5.4/wasm_mvp/parquet.duckdb_extension.wasm`
  locally and set `custom_extension_repository` before `LOAD parquet`
  (verify the embedded engine version first -- it does not always match
  the npm package version; see the exact version-pinning caveats this was
  derived from in
  `~/.hermes/cache/scratch/cp-anywhere-browser-integration-research.md`
  if still present, or re-derive from DuckDB-WASM's own docs).
- **Single browser tested.** Only Chromium (via Playwright) was exercised.
  No SharedArrayBuffer/COOP/COEP cross-origin isolation was used or is
  required for this single-threaded WASM MVP path.
- **2D only.** The Zarr stage of this demo uses one 2D channel; this
  does not affect cp-anywhere's own 2D/3D coverage (`docs/native.md`,
  `docs/wasm.md`), which is unrelated and already complete.
- **Not an OME-Zarr multiscale/metadata demo.** The written Zarr array
  has an empty `.zattrs` (no OME-NGFF multiscale metadata, no
  multichannel axes); it is a plain single-chunk array read for its pixel
  data only.
- **Segmentation scope is this demo's own fixed choice, not every
  setting.** `main.ts` hard-codes `threshold_method="otsu"`,
  `unclump_method="intensity"`, `watershed_method="intensity"`,
  `min_diameter=4.0`; `identifyPrimaryObjects`'s full settings surface
  (every threshold method, adaptive scope, Shape/Propagate declump, hole
  filling, border/size filtering) is documented and independently
  oracle-verified in `docs/segmentation-coverage.md` and
  `spec/segmentation-modules.md`, not re-demonstrated here.
