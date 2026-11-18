# cp-anywhere

<img src="https://raw.githubusercontent.com/d33bs/cp-anywhere/main/docs/src/_static/cp-anywhere.png" alt="cp-anywhere logo" width="300">

CellProfiler-compatible cell profiling in Python and the browser, powered by Rust.
Correct illumination, segment images, measure cell features, and export analysis-ready tables.

cp-anywhere turns microscopy images into per-cell feature tables without requiring
the CellProfiler desktop application. Use image files or NumPy arrays in Python,
work with pandas tables, or run the shared measurement engine in a browser through
WebAssembly. See the [compatibility guide](https://github.com/d33bs/cp-anywhere/blob/main/spec/compatibility.md) for supported modules and settings.

## Usage

To run from this repository, install Rust and uv, then run `uv sync --frozen`.
Run the Python examples with `uv run python` or in the repository's virtual environment.

For a hands-on tour, read
[Cell Profiling Anywhere](https://d33bs.github.io/cp-anywhere/cell-profiling-anywhere.html)
or open the [source notebook](https://github.com/d33bs/cp-anywhere/blob/main/docs/src/cell-profiling-anywhere.ipynb).
The notebook follows a complete Python workflow: illumination correction,
segmentation, measurements, plots, and image/table exports.
Run `uv sync --frozen --group docs` first, then select the repository's `.venv`
Python interpreter in your notebook editor.

The simplest way to call cp-anywhere is `measure_files`, which accepts
TIFF/OME-TIFF, PNG, or saved NumPy array file paths directly (or
already-loaded arrays) and returns a tidy `pandas.DataFrame`:

```python
import cp_anywhere

frame = cp_anywhere.measure_files(
    images={"DNA": "dna.tif", "Actin": "actin.tif"},
    objects={"Cells": "cells.tif", "Nuclei": "nuclei.tif"},
)
```

A single image/object-set pair can skip the dict form entirely:

```python
frame = cp_anywhere.measure_files("dna.tif", "cells.tif")
```

The same file formats work from the command line:

```sh
uv run cp-anywhere \
  --image DNA=dna.tif --image Actin=actin.tif \
  --objects Cells=cells.tif --objects Nuclei=nuclei.tif \
  --output result.json
```

`--output` also accepts `.csv` or `.arrow` (Arrow IPC/Feather, covering
every feature value as one long-form table, with NaN/Infinity kept as
real floats instead of JSON's string markers). `measure_files`'s
DataFrame can be written the same way with `frame.to_feather("result.arrow")`.

For arrays already in memory, `cp_anywhere.measure(...)` is the lower-level
entry point that returns a `MeasurementResult` instead of a `DataFrame`;
it has matching `to_json`/`to_csv`/`to_arrow` methods
(see the [Python API reference](https://d33bs.github.io/cp-anywhere/python-api.html)).

### Segmentation

`cp_anywhere.identify_primary_objects(...)` segments a 2D grayscale image
into labeled objects. You can pass those labels directly to `measure`.
See [segmentation coverage](https://github.com/d33bs/cp-anywhere/blob/main/spec/segmentation-modules.md) for supported settings.

```python
seg = cp_anywhere.identify_primary_objects(dna_image, threshold="otsu")
result = cp_anywhere.measure({"DNA": dna_image}, {"Nuclei": seg.labels})
```

The [segmentation tests](https://github.com/d33bs/cp-anywhere/tree/main/tests/segmentation) compare labels with CellProfiler 4.2.8 on synthetic and real microscopy fixtures.

#### Preview segmented objects

Use `preview_segmentation` to draw object outlines on the input image or show
the objects as filled colors. The preview does not change the image or labels.

```python
overlay = cp_anywhere.preview_segmentation(seg.labels, dna_image)
colored = cp_anywhere.preview_segmentation(seg.labels, mode="labels")

cp_anywhere.save_array(overlay, "segmentation-preview.png", bit_depth="No conversion")
```

To inspect the overlay in a notebook, use `plt.imshow(overlay)` with Matplotlib.

## Project links

- [Documentation](https://d33bs.github.io/cp-anywhere/)
- [Browser and WebAssembly integration](https://github.com/d33bs/cp-anywhere/blob/main/docs/wasm.md)
- [Development and contributing](https://github.com/d33bs/cp-anywhere/blob/main/CONTRIBUTING.md)
- [Report a security vulnerability](https://github.com/d33bs/cp-anywhere/security/advisories/new)
- [Citation](https://github.com/d33bs/cp-anywhere/blob/main/CITATION.cff)
- [BSD 3-Clause license](https://github.com/d33bs/cp-anywhere/blob/main/LICENSE)
