# M2 cp_measure comparison and baseline

## Scope and provenance

The original CellProfiler 4.2.8 fixture outputs are the scientific reference. The comparison uses `cp_measure` 0.3.0 at commit `cdf693187e3dcaf3d067da7b58bdd4ab8896f91e`. The [feature map](../spec/cp_measure-map.yaml) covers all 119 original columns. It maps 84 object intensity columns and seven neighbor columns. It marks all 28 image intensity columns unsupported. The pinned cp_measure source has no `MeasureImageIntensity` module. Its neighbor function contains a TODO for related measurements and emits no directed relationship rows. The 12 relationship tables are therefore unsupported too.

The original runner constructs each `Image` with `convert=False` in `scripts/reference/generate.py`. In the original `cellprofiler_core.image._image.Image.set_image`, that option retains the input array. The conversion path would scale unsigned integer inputs, but the saved fixtures do **not** use that path. This comparison keeps uint8 and uint16 pixel values unchanged. Floats also stay unchanged. For image masks, it keeps labels intact and replaces excluded pixels with NaN for cp_measure's finite-pixel filter. The original neighbor module ignores the image mask. The comparison uses `legacy=True` for cp_measure object intensity quantiles and `Adjacent` for neighbors.

## Observed parity

The [JSON catalog](../benchmarks/baseline/cp_measure-parity.json) and [CSV catalog](../benchmarks/baseline/cp_measure-parity.csv) record one row for every fixture and original column, plus one directed relationship row per fixture. Each row includes fixture, scope, feature, status, source feature name, error maxima, and exact mismatching values. The JSON also records expected relationship edges. The harness checks presence, row IDs, shape, dtype, null, nonfinite, and numeric values separately. It applies each rule from `spec/features.yaml`. It aligns object rows by ID.

| Status             | Fixture entries |
| ------------------ | --------------: |
| Exact pass         |             857 |
| Tolerance pass     |              46 |
| Numeric mismatch   |              43 |
| Nonfinite mismatch |              26 |
| ID mismatch        |              91 |
| Dtype mismatch     |               8 |
| Blocked call       |              21 |
| Unsupported        |             348 |

There are 1,440 entries: 12 fixtures × (119 columns + one relationship table). The 91 ID mismatches occur in `sparse_ids`: the original keeps positions from 1 through the largest label, while cp_measure emits present labels. The eight dtype mismatches are `Neighbors_NumberOfNeighbors_Adjacent`. The original declares `int64`; cp_measure allocates its count array as `float64`. In `nonfinite`, cp_measure's object intensity function filters nonfinite pixels, while the original propagates some of them. The constant-image maximum-position mismatch follows cp_measure's first C-order maximum selection. The neighbor call raises `UnboundLocalError` on three small fixtures, producing 21 blocked column entries. Other numeric differences retain `reason: unknown` until source analysis supports a cause. A cp_measure mismatch does not weaken the future Rust parity gate.

## Measured Python baseline

The [benchmark report](../benchmarks/reports/cp_measure-baseline.json) contains all raw samples, medians, ranges, output byte counts, feature counts, environment pins, and process peak RSS. Each listed kernel and end-to-end path had three warmups and ten timed runs. The separate fresh-process startup timer had one warmup and three samples. The measured machine was macOS 26.7.2 arm64, with 12 logical CPUs and 48 GiB RAM. Python was 3.11.13; NumPy 2.4.6, SciPy 1.17.1, scikit-image 0.26.0, centrosome 1.3.4, and mahotas 1.4.19. No thread environment variable was set.

| Fixture           | Conversion median ms | Object intensity median ms | Neighbors median ms | End-to-end median ms | JSON bytes |
| ----------------- | -------------------: | -------------------------: | ------------------: | -------------------: | ---------: |
| touching_gradient |             0.000354 |                      0.627 |               0.407 |                1.281 |      6,999 |
| real_cell_cc0     |             0.000292 |                      2.066 |               1.314 |                3.689 |     10,951 |
| grid_medium       |             0.000334 |                      8.145 |               2.771 |               12.086 |     64,451 |
| grid_large        |             0.000333 |                     31.750 |              11.183 |               46.911 |    247,221 |

The startup import median was 695.794 ms. Conversion is almost free on these contiguous unmasked arrays because it returns views. End-to-end time includes NPY loading, conversion, the two cp_measure families, and JSON output. Kernel time excludes imports and file loading. The process peak RSS high-water mark ranged from about 115 to 122 MB; it is not a per-kernel allocation. The baseline times only the 91 mapped columns. It does not time missing image intensity or relationship outputs. M6 can compare Rust speed on the same equivalent subset after Rust passes the original CellProfiler reference.

An immutable pre-Rust timing snapshot, source version, scientific-fixture digest, and timestamp are retained in `benchmarks/baseline/initial-cp-measure-timing.json`. Re-running the benchmark updates the current report, not that snapshot.

## Reproduce

Use Python 3.11 and the pinned source checkout. The installation script checks the source commit and installs the locked support packages into an isolated environment.

```sh
export CP_ANYWHERE_CP_MEASURE_SOURCE=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cp-measure
bash benchmarks/install_cp_measure.sh /Users/buntend/.hermes/cache/scratch/cp-anywhere-cp-measure-baseline/venv
/Users/buntend/.hermes/cache/scratch/cp-anywhere-cp-measure-baseline/venv/bin/python scripts/parity/compare_cp_measure.py
/Users/buntend/.hermes/cache/scratch/cp-anywhere-cp-measure-baseline/venv/bin/python benchmarks/benchmark_cp_measure.py
uv run --frozen pytest
```

The support package pins are in `benchmarks/cp_measure-requirements.lock`. The project runtime has no cp_measure or SciPy dependency. These commands write only the M2 report files. They do not rewrite the golden fixtures.
