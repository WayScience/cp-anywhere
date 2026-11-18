# WASM illumination bindings

These exports use typed arrays for image data and a JSON object for settings.
They leave all input arrays unchanged and return a new row-major `Float64Array`.

```typescript
calculateIllumination(
  pixels: Float64Array,
  rows: number,
  cols: number,
  mask: Uint8Array | undefined,
  config_json: string,
): Float64Array;

applyIllumination(
  pixels: Float64Array,
  _function: Float64Array,
  rows: number,
  cols: number,
  config_json: string,
): Float64Array;
```

The generated wasm-bindgen declarations also accept plain numeric arrays and
`null` for the optional mask. Use typed arrays and `undefined` in application
code. Both dimensions must be positive integers, and each supplied buffer must
contain exactly `rows * cols` entries. Mask bytes are false when zero and true
otherwise. An absent mask is distinct from a supplied all-true mask: the Rust
core preserves CellProfiler's mask-dependent float32 conversion stages.

## Calculation settings

`calculateIllumination` wraps the existing Rust `illumination::calculate_each`
implementation. Settings use the same names, defaults, and case-sensitive
CellProfiler choices as Python's `calculate_illumination`:

| Key                        | Default             | Choices or type                                                                                     |
| -------------------------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| `intensity_choice`         | `"Regular"`         | `"Regular"`, `"Background"`                                                                         |
| `block_size`               | `60`                | Unsigned integer                                                                                    |
| `dilate_objects`           | `false`             | Boolean                                                                                             |
| `object_dilation_radius`   | `1`                 | Unsigned integer                                                                                    |
| `smoothing_method`         | `"Gaussian Filter"` | `"No smoothing"`, `"Median Filter"`, `"Gaussian Filter"`, `"Fit Polynomial"`, `"Smooth to Average"` |
| `automatic_object_width`   | `"Automatic"`       | `"Automatic"`, `"Object size"`, `"Manually"`                                                        |
| `object_width`             | `10.0`              | Number                                                                                              |
| `size_of_smoothing_filter` | `10.0`              | Number                                                                                              |
| `rescale_option`           | `"Yes"`             | `"Yes"`, `"No"`, `"Median"`                                                                         |

Only single-image, 2D, `Each` calculation is exposed. `Convex Hull` and `Splines`
raise explicit unsupported-method errors. There is no across-group API here.

## Application settings

| Key                  | Default    | Choices or type          |
| -------------------- | ---------- | ------------------------ |
| `divide_or_subtract` | `"Divide"` | `"Divide"`, `"Subtract"` |
| `truncate_low`       | `true`     | Boolean                  |
| `truncate_high`      | `true`     | Boolean                  |

Application computes division or subtraction in float64, performs clipping,
then casts the result to float32 and back to float64. Subtraction always clips
negative results to zero, even when `truncate_low` is false. Division does not
substitute an epsilon. NaN, infinity, and signed zero follow the Python/real-CP
contract; comparisons do not replace NaN with clipping bounds. Clipping occurs
before the final cast, so a large finite unclipped result can become infinity.
The binding does not enforce a positive finite calibration map; that policy
belongs to the calling application.

For both functions, pass `"{}"` for defaults. Malformed JSON, non-object JSON,
unknown keys, wrong setting types, unknown enum choices, and mismatched shapes
throw JavaScript string errors. Existing WASM exports are unchanged.

## Build and verify

From the repository root:

```bash
env -u CONDA_PREFIX wasm-pack build crates/cp_anywhere_wasm \
  --target web --release --out-dir pkg --out-name cp_anywhere
node scripts/wasm/check_illumination.mjs
env -u CONDA_PREFIX cargo fmt -p cp-anywhere-wasm -- --check
env -u CONDA_PREFIX cargo clippy -p cp-anywhere-wasm \
  --target wasm32-unknown-unknown -- -D warnings
```

The Node check initializes the generated web-target module with explicit WASM
bytes; it runs actual WebAssembly, not a JavaScript implementation. It checks
all 252 existing real-CP calculation cases (including 126 with masks), all eight
existing application cases, defaults, input immutability, invalid inputs, and
retention of existing exports. Calculation and finite application comparisons
use the existing native test tolerances (`rtol=1e-6`, `atol=1e-6`).

Eight additional binding-only edge cases cover both methods and all clipping
combinations with zero, signed zero, nonfinite values, tiny divisors, and
float32 overflow. Their 576 output values are compared exactly, including NaN
and signed zero. They were generated by executing real
`CorrectIlluminationApply.run`, not by computing expectations in cp-anywhere.
Their source module hash is recorded in the fixture.

To regenerate only the binding edge fixture, supply a real CellProfiler
reference environment. On the current verification machine:

```bash
env -u CONDA_PREFIX \
  PYTHONPATH=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cellprofiler \
  /Users/buntend/.hermes/cache/scratch/cp-anywhere-reference-clean/venv/bin/python \
  scripts/wasm/generate_illumination_edges.py
```

This writes only
`crates/cp_anywhere_wasm/tests/fixtures/illumination_apply_edges.json`.
The main check needs only Node and the generated package, not Python or
CellProfiler. Browser/app integration remains a separate check.
