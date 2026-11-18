# cp-anywhere

<img src="https://raw.githubusercontent.com/d33bs/cp-anywhere/main/docs/src/_static/cp-anywhere.png" alt="cp-anywhere logo" width="300">

CellProfiler-compatible cell profiling in Python and the browser, powered by Rust.
Correct illumination, segment images, measure cell features, and export analysis-ready tables.

cp-anywhere turns microscopy images into per-cell feature tables without requiring
the CellProfiler desktop application. Use image files or NumPy arrays in Python,
work with pandas tables, or run the shared measurement engine in a browser through
WebAssembly. Supported modules and settings are listed in `spec/compatibility.md`.

## Usage

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
cp-anywhere \
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
(see `docs/src/python-api.md`).

### Segmentation

`cp_anywhere.identify_primary_objects(...)` segments a 2D grayscale image
into labeled objects (CellProfiler's `IdentifyPrimaryObjects`, Global Otsu
or a manual threshold -- see `spec/segmentation-modules.md` for the exact
scope), and its output chains directly into `measure`:

```python
seg = cp_anywhere.identify_primary_objects(dna_image, threshold="otsu")
result = cp_anywhere.measure({"DNA": dna_image}, {"Cells": seg.labels})
```

Verified against real CellProfiler 4.2.8, exact label-for-label, on both
synthetic and real microscopy fixtures: `tests/segmentation/`,
`fixtures/segmentation/`. Performance: `docs/segmentation-performance.md`.

## High-level overview

This template gives you a ready-to-run Python research software project with:

- `uv`-managed environments and dependencies
- Testing and coverage defaults via `pytest` + `coverage.py`
- Pre-commit automation for formatting, linting, and type checks
- GitHub Actions workflows for linting, tests, docs, and release-related automation
- Starter package + CLI scaffold under `src/`
- Optional high-performance Rust bindings (PyO3 + maturin) with a Cargo workspace under `crates/`
- Documentation scaffold under `docs/`
- Poe task entrypoints for common local workflows (including a full local pipeline task)

## Included agent skills

This project includes agent guidance in `.agents/skills/` with common skills for:

- Test-driven development
- Incremental implementation
- Code review and quality checks
- CI/CD and automation workflow alignment
- Targeted correctness properties and honest verification reports (`formal-verification`)
- Debugging and error recovery
- Optional learning exercises (`learning-opportunities`)
- Simplified Technical English (ASD-STE100) writing checks (`simple-english`)

If you do not want to use local agent guidance in your project, remove `AGENTS.md` and the `.agents/` directory.

## Working with the Rust bindings

This project includes a native extension written in Rust. The performance-critical work lives in a Cargo workspace under `crates/`:

- `crates/cp_anywhere_core` — a plain Rust library with the core routines (unit-tested with `cargo test`)
- `crates/cp_anywhere_python` — a thin [PyO3](https://pyo3.rs/) layer that exposes them to Python as the native module `cp_anywhere._native`

The Python package under `src/cp_anywhere/` calls into the compiled extension, so build it into your environment before running the Python code:

```sh
# build the native extension into the active uv environment
uv run maturin develop

# or use the provided poe task
uv run poe develop
```

`uv run poe test` rebuilds the extension and runs the Python tests; `uv run poe pipeline` runs the Rust tests, rebuilds, lints, and runs the Python tests. See the [versioning notes](https://github.com/d33bs/cp-anywhere/blob/main/CONTRIBUTING.md#versioning) for how release versions flow from git tags into the built wheel.

## Post template copy instructions

While we provide some customizations to the files in this template based on your specification there's likely a chance some things aren't perfect.
We recommend taking a look at each file used within this template to ensure it meets your expectations for the project you're working on.
In addition, consider the following steps to help ensure the project is in good shape before proceeding too far.

- [ ] Remove files you plan on not using (e.g. `src/notebooks`, `.github/workflows/publish-pypi.yml`, etc.).
- [ ] Update the `LICENSE` file based on the project.
- [ ] Update the `CITATION.cff` file based on the project.
- [ ] Update `.github/CODEOWNERS` with the right GitHub handle(s) for review ownership, and expand it to additional owners/teams and path-specific ownership rules as the project grows.
- [ ] Update the project dependencies using `uv remove` or `uv add`.
- [ ] Update the `pyproject.toml` file based on the project.
- [ ] Enable `pre-commit-lite` to help automate corrections to code during pull request updates. Otherwise, consider removing the step: labeled with: `pre-commit-ci/lite-action` within `.github/workflows/run-tests.yml`.
- [ ] Enable private security vulnerability issue reporting within the repo settings (e.g. https://github.com/repo_org/repo_name/settings/security_analysis)
- [ ] Enable [branch protection rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule) to require one pull request review approval per pull request to help with maintainer expectations.
- [ ] Add collaborators to access the repository.
- [ ] Create a `pages` branch and enable GitHub Pages on the repository (for documentation).
- [ ] Update the GitHub repository description.
- [ ] Remove these instructions!
