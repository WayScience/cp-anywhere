# Reference semantics to preserve during implementation

These notes come from the pinned original sources and installed reference environment, not from assumed textbook definitions. The golden data remains the authority.

## Input units

The real `Image` objects in `scripts/reference/generate.py` explicitly use `convert=False`. Integer pixel values are therefore retained in their supplied units, not divided by the dtype maximum. A port must promote integer arithmetic to avoid overflow without applying hidden intensity normalization. This differs from a file-loader pipeline that rescales pixels before measurement.

## Object intensity

- Original `measureobjectintensity.py` line 349 writes `masked_labels > 0 & numpy.isfinite(img)`. Python precedence means this does not filter NaN/Inf as its comment suggests. Do not silently fix that behavior in a compatible port.
- Object percentiles use `area * fraction`, not NumPy's usual `(area - 1) * fraction`, and clamp to the final sample.
- The MAD sample position uses `area / image.dimensions`: one-half in 2D, one-third in 3D.
- The original calls `find_boundaries(..., mode="inner")` after reshaping a 2D array to depth one. Edges use original labels before the image mask; masking does not introduce new boundaries.
- Maximum coordinates come from `scipy.ndimage.maximum_position` on the flattened selected foreground, using its labeled/index-array path. SciPy 1.9.3 globally argsorts intensity with NumPy's default unstable quicksort and overwrites each label's position as it scans that order. Equal maxima do not generally mean first or last row-major pixel. A per-object stable sort changes emitted coordinates.
- Missing sparse-label slots and all-masked objects need separate treatment. Read original `Objects.get_labels` and actual fixture values rather than assuming every absent row is NaN.

## Same-set adjacent object neighbors

- Nearest-neighbor IDs are recomputed with a stable lexicographic distance order, but distances and angle vectors originate from a separate `argpartition` call. Ties must preserve both algorithms, not one generic nearest-neighbor policy.
- Original centers have one column per dimension, yet nearest distances and angle vectors use only columns zero and one. Thus the 3D mode uses z/y and omits x for these features. Preserve and disclose this reference behavior.
- Adjacency dilation uses radius one. The percent-touching dilation uses radius 1.5, and the denominator is the original `centrosome.outline.outline` result.
- `centrosome.outline.outline` compares the first two axes and their diagonals, marking the first/last planes of those axes. Its indexing also runs on the 3D array; it is not a conventional 26-connected volumetric outline.
- Image masks do not affect neighbors.
- Directed relationship rows include both endpoints. Missing labels are not real objects, even when feature arrays retain their row slots.

## Available sorting reference

NumPy v1.23.5 source has been cloned to `/Users/buntend/.hermes/cache/scratch/cp-anywhere-numpy-reference` at `de82cd9468704a033702974010ee7e7efc85b393`. Relevant files are `numpy/core/src/npysort/quicksort.cpp` and `selection.cpp`. Its BSD-3-Clause license was read before using any source; if code is translated into Rust, preserve the NumPy license notice and record the derivation. The scientific environment's NumPy version is pinned to 1.23.5, so substituting a later sorting implementation may change equal-maximum coordinates.
