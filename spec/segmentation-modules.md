# Segmentation module coverage (v0)

This declares the exact scope of `crates/cp_anywhere_segmentation` /
`cp_anywhere.identify_primary_objects`, against CellProfiler 4.2.8's
`IdentifyPrimaryObjects` module (and the shared `Threshold` module it
uses) as the compatibility oracle. Verified by
`tests/segmentation/test_reference_alignment.py` against
`scripts/segmentation/generate_reference.py`'s real-CellProfiler output,
on both synthetic and real microscopy fixtures (`fixtures/segmentation/`).

## Covered settings

| CellProfiler setting                    | Value in this release                                                           | Status  |
| --------------------------------------- | ------------------------------------------------------------------------------- | ------- |
| `use_advanced`                          | `True`                                                                          | Fixed   |
| `threshold.threshold_scope`             | `"Global"`                                                                      | Fixed   |
| `threshold.global_operation`            | `"Otsu"` (two-class) or `"Manual"`                                              | Covered |
| `threshold.two_class_otsu`              | `"Two classes"`                                                                 | Fixed   |
| `threshold.threshold_smoothing_scale`   | `0` (no Gaussian pre-blur)                                                      | Fixed   |
| `threshold.threshold_correction_factor` | configurable                                                                    | Covered |
| `threshold.threshold_range`             | configurable (defaults to `(0, 1)`, matching CellProfiler's own Setting bounds) | Covered |
| `threshold.manual_threshold`            | configurable, `[0, 1]` units                                                    | Covered |
| `unclump_method`                        | `"None"`                                                                        | Fixed   |
| `watershed_method`                      | `"None"`                                                                        | Fixed   |
| `fill_holes`                            | `"Never"`                                                                       | Fixed   |
| `exclude_border_objects`                | `False`                                                                         | Fixed   |
| `exclude_size` / `size_range`           | configurable                                                                    | Covered |

"Fixed" means this release only implements that one value (not a
silent default -- CellProfiler's own `separate_neighboring_objects` is
an exact no-op for `unclump_method="None"`/`watershed_method="None"`,
and both of `run()`'s hole-fill calls are skipped exactly for
`fill_holes="Never"`, so this is not an approximation of those settings,
it is the exact algorithm CellProfiler itself runs for this
configuration).

## Not yet implemented (explicit gaps, not silent fallbacks)

- `unclump_method`/`watershed_method` other than `"None"` (Intensity/
  Shape declumping, distance-transform or intensity watershed).
- `fill_holes` other than `"Never"`.
- `exclude_border_objects = True`.
- `threshold.threshold_scope = "Adaptive"` (local/windowed thresholding).
- `threshold.global_operation` values other than `"Otsu"`/`"Manual"`
  (Minimum Cross-Entropy, Robust Background, Measurement).
- `threshold.two_class_otsu = "Three classes"`.
- `threshold.log_transform`.
- 3D/volumetric segmentation (`IdentifyPrimaryObjects` itself only
  supports 2D; see CellProfiler's own **Watershed** module for 3D).
- `IdentifySecondaryObjects`/`IdentifyTertiaryObjects` (this release is
  primary-object segmentation only).

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
