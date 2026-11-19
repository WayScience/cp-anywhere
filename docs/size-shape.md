# Size and shape measurements

`MeasureObjectSizeShape` is available for 2D label arrays in native Python
and WASM. It uses the pinned CellProfiler 4.2.8 reference, with
scikit-image 0.21.0 and centrosome 1.2.3.

```python
from cp_anywhere import measure

result = measure(
    {"DNA": dna_image},
    {"Cells": cell_labels},
    families=["object_size_shape"],
    size_shape_calculate_zernikes=True,
    size_shape_calculate_advanced=False,
)
```

The family measures each supplied object set, without using image intensity
or the image mask. Images are still required by the shared `measure` API.
The original three default families are unchanged: request
`object_size_shape` explicitly to add shape columns.

## Columns and settings

All columns have the `AreaShape_` prefix. The module supplies:

- 25 standard columns: area, perimeter, axes, eccentricity, orientation,
  center coordinates, bounding boxes, form factor, extent, solidity,
  compactness, Euler number, radii, convex area, Feret diameters, and
  equivalent diameter.
- 30 Zernike columns, orders 0 through 9. Enabled by default, matching
  CellProfiler's `calculate_zernikes` setting.
- 53 advanced columns: spatial, central and normalized moments, Hu moments,
  the inertia tensor and its eigenvalues. Disabled by default, matching
  CellProfiler's `calculate_advanced` setting.

The two flags select 25, 55, 78, or 108 total columns. Undefined values
remain NaN or infinity, rather than becoming zero. The normal result
serialization and export methods retain their existing nonfinite-value
rules. An empty object set still has the selected schema with empty arrays.

The `.cppipe` importer accepts revision 3. Its three settings are the
object list, Zernike flag, and advanced flag. Each imported module keeps
its own settings and object selection, even when a pipeline measures
Nuclei, Cells, and Cytoplasm with different flags.

## Limits

- 3D size/shape measurements are not implemented. A 3D request raises a
  clear error rather than returning 2D substitutes.
- The public measurement API requires consecutive object IDs starting at
  1. The real reference produces inconsistently packed columns for sparse
     labels. The low-level Rust routines preserve that behavior for numerical
     comparison, but the integrated API rejects it to prevent corrupt row
     alignment in exports. Other measurement families keep their existing
     sparse-label support.
- Numerical checks target the pinned reference environment. Advanced
  moment reductions reproduce its NumPy/OpenBLAS ARMv8 order; this is not
  a claim of bitwise agreement with every BLAS implementation.

## Verification

Real reference fixtures cover rectangles, rings, diagonal shapes, nested
objects, multiple objects, border objects, single pixels, and empty images.
Tests compare every selected column under all four flag combinations.
Separate real pipelines run primary, secondary, and tertiary segmentation
before size/shape measurement, with different flags for each compartment.

The browser check runs both `measureJson` and `measureWithOptions` in real
Playwright Chromium, not a simulated WASM runtime. The original
eight-argument `measure` export remains available for existing callers;
`measureWithOptions` adds diagnostic neighbor layers and shape flags.

To regenerate and check the reference data:

```sh
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/python \
  scripts/reference/generate_size_shape.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/python \
  scripts/cppipe/generate_size_shape_pipeline.py
uv run --frozen pytest tests/test_size_shape.py tests/cppipe/test_size_shape_pipeline.py
python scripts/wasm/build_package.py
uv run --frozen python scripts/parity/compare_wasm_size_shape.py
```

The browser check requires the browser example's existing Node dependencies
and Playwright Chromium installation. Building a package does not publish it.
