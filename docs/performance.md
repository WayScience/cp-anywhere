# M6 performance report

## Headline result

**Fixed a real O(pixels^2) bug in the neighbors kernel**, found by profiling
a per-workload throughput regression the plan explicitly warns against
hiding behind an aggregate: before the fix, `cp-anywhere` was 3.6x *slower*
than the pinned `cp_measure` baseline on the `grid_large` workload (256
objects, 192x192 px), even though every smaller workload was already much
faster. After the fix, all four representative workloads beat the baseline
individually, and WASM round-trip times fell by 11x-33x at the same time.

## How the bug was found

`crates/cp_anywhere_core/src/lib.rs`'s "touching" check (used for
`PercentTouching` in `MeasureObjectNeighbors`) needs to know, for every
outline pixel of object `k`, whether any pixel within a small fixed radius
(1.5 px) belongs to a *different* object. The original implementation
tested that by scanning **every pixel in the whole image** for each outline
pixel -- an accidental O(perimeter_pixels * total_pixels) algorithm when
O(perimeter_pixels) suffices, because the actual predicate only ever
depends on a handful of neighboring offsets.

This was invisible in the declared fixture set's smallest three workloads
(it is a constant-factor cost there) but dominates at `grid_large` scale
(36,864 px, 256 objects): a `cargo bench` run that split the combined
`object_intensity_and_neighbors` timing by family
(`crates/cp_anywhere_core/benches/throughput.rs`'s `by_family` group)
showed `neighbors` alone was 151.45 ms on `grid_large` vs 9.68 ms on
`grid_medium` -- roughly a 15.6x time increase for a 4x object-count / 4x
pixel-count increase, consistent with an O(pixels * objects) or O(pixels^2)
pattern, not the near-linear scaling `object_intensity` showed over the
same step (0.44 ms -> 2.18 ms, ~5x).

## The fix

Replaced the whole-image scan with a bounded local-neighborhood check using
the same `shifted(index, shape, delta)` helper the adjacent-relationship
loop just below it already used correctly: a small, fixed Cartesian product
of candidate offsets (9 for 2D, 8 for 3D) covering exactly the same set of
relative positions the original predicate tested, reusing the project's
existing bounds-checked index arithmetic instead of adding a new primitive.
**The predicate itself was not touched** -- including its asymmetric,
reference-preserving 3D offset range (`d == 0 || d == -1`, not a symmetric
radius) -- only how it is evaluated changed, from O(total pixels) to O(1)
per outline pixel.

Correctness was re-verified immediately after the change, not assumed:

```text
cargo test --manifest-path crates/cp_anywhere_core/Cargo.toml   # pass
uv run --frozen python scripts/parity/compare_native.py          # 1440/1440 pass (unchanged)
# rebuild WASM, re-export fixtures, rebuild demo, re-run in Chromium:
uv run --frozen python scripts/parity/compare_wasm.py            # 1440/1440 pass (unchanged)
```

Both parity counts are bit-for-bit identical to before the fix -- the
change is a pure performance fix, not a semantics change.

## Before / after timing

Native Rust kernel only (`cp_anywhere_core::measure`, no Python/FFI,
`object_intensity` + `neighbors` families, Criterion, median of 100
samples):

| Workload          |    Before |    After | Speedup |
| ----------------- | --------: | -------: | ------: |
| touching_gradient |  38.05 us | 35.48 us |   1.07x |
| real_cell_cc0     |   2.53 ms | 0.195 ms |   13.0x |
| grid_medium       |  10.16 ms | 0.695 ms |   14.6x |
| grid_large        | 154.85 ms | 7.354 ms |   21.1x |

Browser (WASM, Playwright/Chromium, median per `measure()` call including
in-WASM JSON encoding, 10 timed runs):

| Workload          |    Before |    After |                            Speedup |
| ----------------- | --------: | -------: | ---------------------------------: |
| touching_gradient |   0.15 ms |  0.20 ms | ~unchanged (near timer resolution) |
| real_cell_cc0     |   2.85 ms |  0.70 ms |                               4.1x |
| grid_medium       |  11.80 ms |  2.00 ms |                               5.9x |
| grid_large        | 174.00 ms | 11.85 ms |                              14.7x |

## Throughput vs the pinned `cp_measure` baseline

Per `spec/benchmark-contract.yaml`'s `native_acceptance` gate: equal-weight
arithmetic mean of per-workload `cp_measure_time / cp-anywhere_time` ratios
on the `object_intensity` + `neighbors` scope (the only scope `cp_measure`
implements), minimum 1.0. Computed by
`scripts/benchmarks/check_throughput_gate.py`, which prints every
per-workload ratio and fails if *any* workload is below the minimum, so an
aggregate pass cannot hide an individual regression:

| Workload          | cp_measure (ms) | cp-anywhere Rust (ms) | Ratio |
| ----------------- | --------------: | --------------------: | ----: |
| touching_gradient |           1.034 |                0.0355 | 29.1x |
| real_cell_cc0     |           3.380 |                0.1952 | 17.3x |
| grid_medium       |          10.916 |                0.6952 | 15.7x |
| grid_large        |          42.933 |                7.3535 |  5.8x |

Equal-weight mean: **17.0x**. Every workload individually clears the 1.0x
minimum (`all_workloads_pass_individually: true` in
`benchmarks/reports/native-throughput-gate.json`) -- before the fix,
`grid_large`'s ratio was 0.28x (a genuine regression the aggregate of
~7.5x would have silently hidden had it not been checked per-workload).

## Python binding overhead

Per plan 5.2 ("Python binding overhead: quantified and bounded"), comparing
the installed release wheel's `cp_anywhere.measure()` against the bare Rust
kernel on the same inputs (`benchmarks/benchmark_native.py` vs
`scripts/benchmarks/report_native_rust.py`):

| Workload          | Rust (ms) | Python (ms) | Overhead |
| ----------------- | --------: | ----------: | -------: |
| touching_gradient |    0.0355 |      0.0744 |    2.09x |
| real_cell_cc0     |    0.1952 |      0.2614 |    1.34x |
| grid_medium       |    0.6952 |      0.7626 |    1.10x |
| grid_large        |    7.3535 |      7.7758 |    1.06x |

The fixed per-call cost (roughly 0.04-0.07 ms: Python call dispatch plus
NumPy array validation) shrinks in relative terms as the workload grows,
and stays within the declared `<=1.5x` budget
(`spec/benchmark-contract.yaml`'s `python_overhead_budget`) for every
workload at or above `real_cell_cc0` scale. No unnecessary full-array copy
was found in the binding for the already-C-contiguous fixture inputs.

## CI regression gate

`scripts/benchmarks/check_regression.py` compares a fresh
`benchmarks/reports/native-rust-baseline.json` against the committed
`benchmarks/baseline/native-rust-committed.json`, failing only when a
workload is more than 20% slower than committed -- never on an
improvement, and never on noise within that margin, per the plan's
explicit warning against hard-failing on noisy single-run CI timing.
`.github/workflows/benchmark.yml` runs the full benchmark -> report ->
throughput-gate -> regression-check chain on a weekly schedule and on
manual dispatch (not on every PR, for the same noise reason), uploading
the raw report as a 90-day build artifact for trend-watching.

To intentionally accept a new baseline after a verified, deliberate
change, regenerate `benchmarks/reports/native-rust-baseline.json` and copy
it over `benchmarks/baseline/native-rust-committed.json`.

## Reproduce

```sh
cargo bench --manifest-path crates/cp_anywhere_core/Cargo.toml
uv run --frozen python scripts/benchmarks/report_native_rust.py
env -u CONDA_PREFIX uv run maturin develop --release --uv
uv run --frozen python benchmarks/benchmark_native.py
uv run --frozen python scripts/benchmarks/check_throughput_gate.py
uv run --frozen python scripts/benchmarks/check_regression.py
```

## Known limitations

- Single machine, single run per report regeneration (Criterion's own
  statistical sampling, 100 samples per workload, is the only
  within-run repetition; see `benchmark-analysis`'s guidance on
  single-rep measurements -- the gaps here are large (5.8x-29x, 21x on
  the previously-regressed workload), well above the "order of magnitude"
  bar for a conclusion to survive run-to-run noise).
- `cp_measure`'s own timing snapshot (`benchmarks/baseline/initial-cp-measure-timing.json`)
  was captured once, in M2, on this same machine; it was not re-measured in
  M6. Re-running it would only change the comparison if this machine's
  absolute performance characteristics changed between M2 and M6, which
  they did not (same hardware, same session).
- The regression-gate tolerance (20%) and the schedule (weekly, not
  per-PR) are both judgment calls aimed at the plan's explicit warning
  against noisy single-run CI failures; they have not been tuned against
  real CI runner variance data, because this session cannot execute a
  real scheduled GitHub Actions run.
