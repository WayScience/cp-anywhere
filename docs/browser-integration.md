# Browser integration: Zarr, ONNX Runtime Web, DuckDB-WASM

## What this is

`examples/zarr-onnx-duckdb/` is a real, end-to-end browser integration
proving the pieces named in the plan work together with cp-anywhere's own
WASM build, with **no mocked stage**:

1. **zarrita.js** reads a real OME-Zarr v2 array (`public/dna.zarr`,
   written by `scripts/zarr/export_demo_fixture.py`) over HTTP.
1. **onnxruntime-web** runs a real, checked ONNX graph
   (`public/threshold.onnx`) against that array, producing an integer
   label mask.
1. **cp-anywhere-wasm** (this project's own Rust/WASM kernel, the same
   build verified in `docs/wasm.md`) measures the Zarr image using that
   ONNX-produced mask as the `Cells` object labels.
1. **DuckDB-WASM** loads cp-anywhere's own measurement output into a SQL
   table and exports it as a real Parquet file (`COPY ... FORMAT PARQUET`).

`scripts/zarr/validate_export.py` then validates that Parquet file with
**native** DuckDB (not the WASM build under test) and cross-checks every
value against a fresh, from-source native Python `cp_anywhere.measure()`
call on the identical Zarr bytes and mask -- proving the browser pipeline
produced numerically correct output, not just that it ran without an
error.

## Why the mask, not a trained segmentation model

The plan names "ONNX Runtime Web label masks" as an integration target, not
a trained segmentation model (cp-anywhere does not ship or depend on one).
`scripts/zarr/export_demo_fixture.py` builds a tiny, real ONNX graph --
`Greater(image, 0.5) -> Cast(..., INT32)` -- checked with
`onnx.checker.check_model` and independently evaluated with
`onnx.reference.ReferenceEvaluator` before the browser ever sees it. This
is a deterministic threshold, not instance segmentation: disconnected
foreground pixels all share label `1`. It exists to exercise the real
Zarr -> ONNX -> cp-anywhere -> DuckDB pipeline end to end, not to claim
CellProfiler-parity object identification from a thresholding operator (see
"Known limitations" below).

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
- ONNX inference: 72 foreground pixels (`mask > 0.5`).
- cp-anywhere-wasm `measure()`: 21 `object_intensity` columns for the
  single foreground object.
- DuckDB-WASM Parquet export: 1,924 bytes, real `PAR1` magic bytes at both
  ends of the file, independently readable by native DuckDB.
- `scripts/zarr/validate_export.py`: all 21 feature values match a
  from-source native Python `cp_anywhere.measure()` run exactly (`rtol=1e-6, atol=1e-9`) -- zero mismatches.

## Known limitations

- **Not scientific segmentation.** The ONNX model is a fixed-threshold
  demo graph (see above), not a learned model or connected-component
  instance segmentation. Scientific parity for this demo's mask is not
  claimed or meaningful; cp-anywhere's own measurement parity is
  established separately and exhaustively in `docs/native.md` and
  `docs/wasm.md` against real CellProfiler 4.2.8 fixtures.
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
- **2D only.** The Zarr/ONNX stage of this demo uses one 2D channel; this
  does not affect cp-anywhere's own 2D/3D coverage (`docs/native.md`,
  `docs/wasm.md`), which is unrelated and already complete.
- **Not an OME-Zarr multiscale/metadata demo.** The written Zarr array
  has an empty `.zattrs` (no OME-NGFF multiscale metadata, no
  multichannel axes); it is a plain single-chunk array read for its pixel
  data only.
