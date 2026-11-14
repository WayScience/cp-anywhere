# CI/CD

## On every pull request (`.github/workflows/run-tests.yml`)

- **`pre_commit_checks`**: all pre-commit hooks (ruff, cargo fmt/clippy via
  the Rust hooks below, yamllint, codespell, mdformat, zizmor, vulture,
  `ty`). Syncs every `uv` dependency group (`--all-groups`) so `ty` can
  resolve the demo-only `onnx`/`duckdb` imports in `scripts/zarr/*.py`.
- **`run_tests`** (matrix: Python 3.11-3.15 x Ubuntu/macOS): `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, `maturin develop` +
  `pytest`, and `scripts/parity/compare_native.py` (golden-fixture parity;
  does not need the live CellProfiler oracle, so it runs on every PR
  across the full OS/Python matrix).
- **`wasm_build_and_browser_check`**: builds `crates/cp_anywhere_wasm`
  with `wasm-pack`, builds the `examples/browser-vite` Vite demo, runs it
  in a real headless Chromium via Playwright
  (`npm run browser-check`), and checks the captured output with
  `scripts/parity/compare_wasm.py` -- a real WASM build and browser smoke
  test on every PR, not a native-only check.

## Nightly (`.github/workflows/nightly-full-parity.yml`)

Installs the pinned CellProfiler 4.2.8 oracle (`reference/install.sh`,
which needs a JDK for `python-javabridge` -- too heavy to run on every
PR) and re-runs the full suite against it: `pytest` with the oracle
configured, `compare_native.py`, and a full WASM build + browser check +
`compare_wasm.py`. This is the plan's "full feature inventory parity
across native/Python/WASM" nightly check. Also runnable on demand
(`workflow_dispatch`).

## Weekly (`.github/workflows/benchmark.yml`)

Native Rust throughput benchmark (Criterion) against the committed
baseline, with a tolerance-based regression gate
(`scripts/benchmarks/check_regression.py`) -- see `docs/performance.md`
for why this runs on a schedule rather than every PR (CI runner timing
noise).

## On release (`.github/workflows/publish-*.yml`)

See `docs/release.md` for the full publish pipeline (PyPI, crates.io,
npm, GitHub Release assets).

## Known gaps

- Property-based tests for invariants such as label permutation and
  object ordering (plan section 8) are not implemented. The current test
  suite checks golden-fixture correctness, not permutation-invariance
  properties; this would be a reasonable follow-up using `proptest`
  (Rust) or `hypothesis` (Python).
- None of the new CI workflows added in M7 (`wasm_build_and_browser_check`,
  `nightly-full-parity.yml`) have been exercised by a real GitHub Actions
  run in this session -- only validated locally (the exact same commands
  each step runs were run directly in this session and passed) and
  statically (`actionlint`/`zizmor` via pre-commit).
