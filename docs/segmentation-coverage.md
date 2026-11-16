# Segmentation coverage dashboard

## Declared scope

`cp_anywhere.identify_primary_objects` targets CellProfiler 4.2.8's
`IdentifyPrimaryObjects` module (and the `Threshold` module it shares
with every other segmentation module) as its compatibility oracle.
Full declared scope, including every explicit gap, is in
`spec/segmentation-modules.md`; this page is the distilled status
dashboard.

## Cross-target verification status

Both native Rust/Python and the WASM (browser) build pass all 100
fixtures exactly (see `docs/wasm.md` for the WASM-specific run and its
one documented tolerance exception, identical to the one below).

| Tier                                                                                            | Status                         |
| ----------------------------------------------------------------------------------------------- | ------------------------------ |
| Global threshold: Manual, Otsu (two- and three-class), Li, Robust Background                    | Covered                        |
| Adaptive (local) threshold: block-grid + B-spline interpolation, Sauvola                        | Covered                        |
| `threshold_smoothing_scale`, `threshold_correction_factor`, `threshold_range`                   | Covered                        |
| `log_transform`                                                                                 | Covered                        |
| Border exclusion, size/diameter filtering, object-count limiting                                | Covered                        |
| Hole filling (`Never`/`After declumping only`/`After both`), including nested-object absorption | Covered                        |
| Declumping (`unclump_method`: None/Intensity/Shape)                                             | Covered                        |
| Watershed separation (`watershed_method`: None/Intensity/Shape/Propagate)                       | Covered                        |
| Manual smoothing-filter-size / maxima-suppression-size overrides                                | Covered                        |
| `unclump_method = "Laplacian of Gaussian"`                                                      | Not implemented (explicit gap) |
| `low_res_maxima` downsampled maxima detection                                                   | Not implemented (explicit gap) |
| `threshold.global_operation = "Measurement"`                                                    | Not implemented (explicit gap) |
| 3D/volumetric segmentation, `IdentifySecondaryObjects`/`IdentifyTertiaryObjects`                | Out of scope for this release  |

**Every "Covered" row above is independently verified against the real,
pinned CellProfiler 4.2.8 oracle** — not assumed correct by code
inspection. `tests/segmentation/test_reference_alignment.py` replays
`scripts/segmentation/generate_reference.py`'s real-CellProfiler output
against `cp_anywhere.identify_primary_objects` on 100 fixture cases
(`fixtures/segmentation/`), spanning synthetic edge cases and two
independent real microscopy datasets (BBBC007, real_cell_cc0). No
fixture has a silent fallback or relaxed tolerance beyond the one
documented, narrow, label-safe exception below.

One known tolerance exception: a single adversarial fixture (adaptive
scope, three-class Otsu, log-transformed, a deliberately tiny 4-pixel
window) can land `threshold_map` values up to ~0.012 off the oracle,
from a `float32`-vs-`float64` pixel-precision difference that a full
pipeline rewrite was tried and reverted for (see
`spec/segmentation-modules.md`'s "Adaptive (local) thresholding"
section). Object labels are unaffected in every case.

## Chaining into measurement

```python
seg = cp_anywhere.identify_primary_objects(dna_image, threshold="otsu")
result = cp_anywhere.measure({"DNA": dna_image}, {"Cells": seg.labels})
```

## Reproduce

```sh
env -u CONDA_PREFIX uv run maturin develop --release --uv
CP_ANYWHERE_REFERENCE_PYTHON=/path/to/reference/venv/bin/python \
  PYTHONPATH=/path/to/pinned/CellProfiler \
  uv run --frozen pytest tests/segmentation

# regenerate fixtures from the real oracle (after a reference-semantics change)
CP_ANYWHERE_REFERENCE_PYTHON=/path/to/reference/venv/bin/python \
  PYTHONPATH=/path/to/pinned/CellProfiler \
  /path/to/reference/venv/bin/python scripts/segmentation/generate_reference.py
```
