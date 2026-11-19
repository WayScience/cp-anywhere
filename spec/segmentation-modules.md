# Segmentation module coverage

This declares the exact scope of `cp_anywhere::segmentation` /
`cp_anywhere.identify_primary_objects` (native Python) /
`identifyPrimaryObjects` (WASM, `crates/cp_anywhere_wasm`, same settings
surface via the shared `build_config`/`ConfigParams`), against
CellProfiler 4.2.8's `IdentifyPrimaryObjects` module (and the shared
`Threshold` module it uses) as the compatibility oracle. Verified by
`tests/segmentation/test_reference_alignment.py` (native/Python) and
`scripts/parity/compare_wasm_segmentation.py` (WASM, via a real
Playwright/Chromium run) against
`scripts/segmentation/generate_reference.py`'s real-CellProfiler output,
on both synthetic and real microscopy fixtures (`fixtures/segmentation/`,
100 cases as of this update).

## Covered settings

| CellProfiler setting                                                                                                            | Values in this release                                                                          | Status  |
| ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------- |
| `use_advanced`                                                                                                                  | `True`                                                                                          | Fixed   |
| `threshold.threshold_scope`                                                                                                     | `"Global"` or `"Adaptive"`                                                                      | Covered |
| `threshold.global_operation`                                                                                                    | `"Otsu"` (two- or three-class), `"Manual"`, `"Minimum Cross-Entropy"`, `"Robust Background"`    | Covered |
| `threshold.local_operation`                                                                                                     | same as `global_operation`, plus `"Sauvola"` (adaptive-only)                                    | Covered |
| `threshold.adaptive_window_size`                                                                                                | configurable                                                                                    | Covered |
| `threshold.two_class_otsu`                                                                                                      | `"Two classes"` or `"Three classes"`                                                            | Covered |
| `threshold.assign_middle_to_foreground`                                                                                         | configurable (three-class Otsu only)                                                            | Covered |
| `threshold.threshold_smoothing_scale`                                                                                           | configurable (Gaussian pre-blur of the comparison image, mask-bleed-corrected)                  | Covered |
| `threshold.threshold_correction_factor`                                                                                         | configurable                                                                                    | Covered |
| `threshold.threshold_range`                                                                                                     | configurable (defaults to `(0, 1)`, matching CellProfiler's own Setting bounds)                 | Covered |
| `threshold.manual_threshold`                                                                                                    | configurable, `[0, 1]` units                                                                    | Covered |
| `threshold.log_transform`                                                                                                       | configurable (Otsu two/three-class and Li only, matching CellProfiler's own Setting visibility) | Covered |
| `threshold.lower_outlier_fraction` / `upper_outlier_fraction` / `averaging_method` / `variance_method` / `number_of_deviations` | configurable (Robust Background only)                                                           | Covered |
| `unclump_method`                                                                                                                | `"None"`, `"Intensity"`, `"Shape"`                                                              | Covered |
| `watershed_method`                                                                                                              | `"None"`, `"Intensity"`, `"Shape"`, `"Propagate"`                                               | Covered |
| `automatic_smoothing` / `smoothing_filter_size`                                                                                 | configurable (automatic formula or manual override)                                             | Covered |
| `automatic_suppression` / `maxima_suppression_size`                                                                             | configurable (automatic formula or manual override)                                             | Covered |
| `fill_holes`                                                                                                                    | `"Never"`, `"After declumping only"`, `"After both thresholding and declumping"`                | Covered |
| `exclude_border_objects`                                                                                                        | configurable                                                                                    | Covered |
| `exclude_size` / `size_range`                                                                                                   | configurable                                                                                    | Covered |
| `limit_choice` / `maximum_object_count`                                                                                         | configurable (`"Continue"` or `"Erase"`)                                                        | Covered |

"Covered" means every value listed is implemented and independently
verified against the real CellProfiler 4.2.8 oracle (not assumed
correct by inspection, and not a silent approximation) — see
`docs/segmentation-coverage.md` for the verification method and current
fixture count.

## Not yet implemented (explicit gaps, not silent fallbacks)

- `unclump_method = "Laplacian of Gaussian"` (`UN_LOG`). CellProfiler's
  own three-way `unclump_method` Choice includes this value; this
  release's `UnclumpMethod` enum only has `None`/`Intensity`/`Shape`,
  so passing an equivalent string raises rather than silently
  approximating it with `"Intensity"`.
- `low_res_maxima = True` combined with `min_diameter > 10.0`
  (CellProfiler's own gate for downsampled maxima detection, for
  speed on large objects). `declump()` detects this combination and
  returns an explicit error rather than running the un-downsampled
  path silently, or guessing at the downsampling math.
- `threshold.global_operation = "Measurement"` (a per-image threshold
  read from an upstream measurement column, not computed from pixel
  statistics at all).
- 3D/volumetric segmentation (`IdentifyPrimaryObjects` itself only
  supports 2D; see CellProfiler's own **Watershed** module for 3D).
- Secondary and tertiary segmentation in WASM. Native Python supports
  propagation, fixed-radius secondary expansion, and tertiary subtraction.
  See `spec/cppipe-compatibility.md` for their settings and limits.

A future release extending this scope needs its own fixtures and a
scope-decision update here, matching this project's existing
`spec/compatibility.md` convention for the measurement side.

## Pixel units: the "rescale" convention

CellProfiler's `Threshold.manual_threshold`/`threshold_range` Settings
are hard-bounded to `[0, 1]` (`FloatRange((0, 1), minval=0, maxval=1)`
in `cellprofiler/modules/threshold.py`) -- CellProfiler's Global
threshold machinery assumes images have been rescaled into `[0, 1]`.
Its default image loading (`Image(..., convert=True)`) does this
automatically: integer-dtype pixels are divided by their dtype's max
representable value (confirmed directly against the pinned
CellProfiler source: `uint8` divides by 255, `uint16` by 65535);
float-dtype pixels are never rescaled, regardless of their value range.

`cp_anywhere.identify_primary_objects` reproduces this exactly via
`rescale=True` (the default), so a raw `uint8`/`uint16`/`uint32` image
works like a default CellProfiler pipeline with no manual conversion.
`rescale=False` operates on raw pixel values directly instead (matching
this project's **measurement** API's own `convert=False` convention,
which is a deliberate choice for *that* API, not CellProfiler's own
default).

### Two Otsu algorithms, because CellProfiler's own Otsu has two paths

`skimage.filters.threshold_otsu` (which CellProfiler's `Threshold`
module calls directly) dispatches on the pixel array's dtype *at the
time it runs* -- not on whether the values happen to be whole numbers:

- `threshold="otsu"` (default): 256-bin uniform histogram over
  `[min, max]` of the (possibly rescaled) pixel values, matching
  `threshold_otsu`'s float-dtype path -- this is what real CellProfiler
  uses after its default rescale, since rescaled pixels are always
  float32/float64.
- `threshold="otsu_integer"`: exact one-bin-per-value histogram,
  matching `threshold_otsu`'s integer-dtype path -- only correct if
  CellProfiler would also have seen a genuinely integer-dtype,
  *unrescaled* array (i.e. paired with `rescale=False` on an integer
  image).

Verified bit-close (float32-vs-float64 precision; CellProfiler computes
in float32 after its rescale, this implementation in float64) against
the real oracle on two independent real microscopy datasets:
`real_cell_cc0` (threshold 0.25099573 oracle vs 0.2509957... here) and
`real_bbbc007` (0.14417893 oracle vs 0.14417892... here) --
see `tests/segmentation/test_reference_alignment.py`.

### Global threshold operations

Besides the two Otsu paths above, four other `global_operation`/
`local_operation` values are implemented, each verified directly
against the real `centrosome.threshold`/`skimage.filters` function it
corresponds to, not re-derived from a paper:

- **Minimum Cross-Entropy (Li)**: `skimage.filters.threshold_li`, plus
  CellProfiler's own tolerance formula
  (`max(min(diff(unique(values)))/2, 0.5/65536)`).
- **Three-class Otsu**: `skimage.filters.threshold_multiotsu` with
  128 bins; `assign_middle_to_foreground` controls whether the middle
  class joins the foreground or background mask.
- **Robust Background**: the full trimmed-mean/median/mode,
  standard-deviation/median-absolute-deviation formula from
  `centrosome.threshold.get_threshold_robust_background`, including
  its `low_chop == 0` quirk (if the lower trim rounds to zero pixels,
  the upper trim is *also* skipped entirely) and `centrosome`'s
  `binned_mode` tie-breaking (first-index-wins on ties, matching
  `numpy.argmax`, not Rust's default last-index-wins `max_by_key`).
- **Sauvola** (adaptive scope only; it is not even one of
  CellProfiler's own `global_operation` Choice values): sliding-window
  local mean/std via `skimage.filters.threshold_sauvola`, including
  `numpy.pad(mode="reflect")` boundary semantics.

### Adaptive (local) thresholding

`threshold_scope="adaptive"` computes a per-pixel threshold map by
blending block-grid threshold values with a from-scratch tensor-product
B-spline interpolator (not-a-knot knots, FindSpan basis evaluation,
Gaussian-elimination solve), verified to reproduce real
`scipy.interpolate.RectBivariateSpline(s=0)` to within float epsilon on
fixed test grids. The final map is clamped to
`[0.7, 1.5] * global_guide_threshold`, matching
`Threshold._correct_local_threshold` exactly. `log_transform` interacts
correctly with the block/spline path for Otsu/Li; Sauvola bypasses the
block/spline machinery entirely (a separate, direct sliding-window
computation, matching CellProfiler's own code structure) and zero-fills
masked-out pixels, unlike every other method.

One narrow, label-safe tolerance exception is documented directly in
`tests/segmentation/test_reference_alignment.py`
(`_KNOWN_THRESHOLD_MAP_TOLERANCE_GAP`): a single adversarial fixture
(adaptive, 3-class Otsu, log-transformed, a deliberately tiny 4-pixel
window) can land threshold_map values up to ~0.012 off the oracle due
to a `float32` (real CellProfiler's internal pixel dtype) vs `float64`
(this implementation's) precision difference in log-transform's noise
floor. Labels are unaffected in every case; a full `float32` rewrite of
the native core was tried and reverted after it fixed nothing here and
introduced a new, unrelated label-level regression elsewhere (see
`tests/segmentation/test_reference_alignment.py`'s comment for detail).

### Declumping and watershed

`unclump_method`/`watershed_method` separate touching objects right
after initial 8-connected labeling and before border/size filtering,
matching `separate_neighboring_objects`'s exact position in
CellProfiler's pipeline (`IdentifyPrimaryObjects.run()`). Every
primitive was verified independently against its real source before
being assembled:

- **Maxima detection**: `centrosome.cpmorphology.is_local_maximum` +
  `binary_shrink` (the latter's four 512-entry directional-thinning
  lookup tables extracted bit-for-bit from the running oracle, not
  re-derived), plus `strel_disk`.
- **Smoothing**: `IdentifyPrimaryObjects.smooth_image`'s own Gaussian
  (limited to +/-2SD, mask-bleed-corrected) -- a *different* Gaussian
  from `threshold_smoothing_scale`'s.
- **Shape unclump**: an exact Euclidean distance transform
  (Felzenszwalt & Huttenlocher's lower-envelope-of-parabolas
  algorithm, the same one `scipy.ndimage.distance_transform_edt` uses
  internally) plus a from-scratch MT19937 generator matching
  `numpy.random.seed(0); numpy.random.uniform(0, 0.001, shape)` bit
  for bit (CellProfiler's own fixed-seed jitter to break exact
  distance ties).
- **Watershed** (Intensity/Shape): a negative-marker FIFO watershed
  matching `skimage.segmentation.watershed`, including its degenerate
  symmetric-tie-break behavior.
- **Propagate**: `centrosome.propagate.propagate`'s distinct
  (non-watershed) algorithm.
- **Shape + Shape** (unclump and watershed both `"Shape"`) correctly
  *reuses* the same already-jittered distance transform computed
  during unclump, matching CellProfiler's own
  `if distance_transformed_image is None` check -- it does not
  recompute (and re-jitter) a second one.

### Hole filling, including the nested-object-absorption case

`fill_holes` is implemented via a full foreground/background label
graph (`crates/cp_anywhere/src/segmentation/holes.rs`, adapted from
`centrosome.cpmorphology.fill_labeled_holes`/
`_cpmorphology2.fill_labeled_holes_loop`), not a simplified
single-owner heuristic. This correctly handles:

- A background region enclosed by exactly one label: filled.
- A background region touching two *different* labels: left
  unfilled (not absorbed into either).
- **A small object fully nested inside a larger one: absorbed into
  the enclosing label.** This was a real, previously-known gap in an
  earlier simplified implementation; fixed and verified against the
  real oracle on single-level and multi-level ("bullseye") nesting.
- `max_area`-gated filling, matching CellProfiler's own
  `size_fn(area, is_foreground)` predicate semantics exactly.

`fill_holes="After both thresholding and declumping"` runs a
pre-labeling fill (size-limited to `max_diameter^2`, on the raw
thresholded mask before 8-connected labeling even happens) in addition
to the post-filter fill every non-`"Never"` value runs.

## Chaining into measurement

`identify_primary_objects(...).labels` is a plain label array, directly
usable as a `cp_anywhere.measure()` object-set input:

```python
seg = cp_anywhere.identify_primary_objects(dna_image, threshold="otsu")
result = cp_anywhere.measure({"DNA": dna_image}, {"Cells": seg.labels})
```

## Benchmark

See `benchmarks/reports/segmentation-native-vs-cellprofiler.json` and
`docs/segmentation-performance.md` for Rust-vs-real-CellProfiler timing
on the real microscopy fixtures.
