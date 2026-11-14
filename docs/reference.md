# CellProfiler reference baseline (M0/M1)

## Result

The generator runs the original CellProfiler 4.2.8 modules on real `cellprofiler-core` `Workspace` objects. It saves and reloads one `.cppipe` before measurement. The modules read the saved fixture arrays as CellProfiler images and objects. The run emits 119 exact feature columns: 28 image intensity, 84 object intensity, and seven object neighbor columns. It also emits the directed `Neighbors` relationship table. Twelve synthetic fixtures plus a real microscopy crop (`real_cell_cc0`) and a real two-channel microscopy crop with independent ground-truth labels (`real_bbbc007`) — thirteen fixtures total — cover 2D, 3D, two channels, two object sets, masks, sparse IDs, and degenerate data. This baseline makes no Rust or `cp_measure` compatibility claim.

The source versions, commit IDs, platform, and package versions are in `reference/source-lock.json` and `reference/requirements.lock`. CellProfiler source stays unmodified. The project BSD-3-Clause license and Dave Bunten attribution stay in place.

## Reproduce

Install the separate reference environment. These commands use a cached source checkout to avoid a second clone. Omit `CP_ANYWHERE_REFERENCE_SOURCE` to clone the pinned source from GitHub.

```bash
export CP_ANYWHERE_REFERENCE_ROOT="$HOME/.cache/cp-anywhere/reference"
export CP_ANYWHERE_REFERENCE_SOURCE="/path/to/CellProfiler-at-2df090ff"
reference/install.sh
reference/run.sh
reference/run.sh --verify
```

`reference/install.sh` installs Python 3.9.25 in the reference root. It installs the 51 pinned packages in `reference/requirements.lock`. It builds `python-javabridge==4.0.3` with `-Wno-int-conversion` on macOS arm64. This compiler flag addresses a generated C warning. It does not change the CellProfiler source or algorithms. `reference/run.sh` checks the source commit before executing the generator.

Run the repository checks. Set both environment variables so the replay test uses the original source and reference Python.

```bash
export CP_ANYWHERE_REFERENCE_PYTHON="$CP_ANYWHERE_REFERENCE_ROOT/venv/bin/python"
export PYTHONPATH="$CP_ANYWHERE_REFERENCE_SOURCE"
uv run --frozen pytest tests/reference -q
uv run --frozen pytest
```

Without `CP_ANYWHERE_REFERENCE_PYTHON`, the replay test reports an explicit skip. The other tests still check recorded checksums and the feature inventory. A skip does not count as reference execution.

## Data layout and interpretation

`reference/pipelines/intensity_neighbors.cppipe` stores the three module settings. `fixtures/synthetic/<name>/*.npy` holds synthetic arrays. `fixtures/real/<name>/*.npy` holds a real crop and its labels for `real_cell_cc0` and `real_bbbc007`. The `Nuclei` label array selects one pixel per nonzero `Cells` ID for every fixture except `real_bbbc007`, which uses genuinely independent, multi-pixel nuclei segmented from the dataset's own ground truth (see below). `fixtures/reference/<name>.json` holds all feature values by image and object scope. `fixtures/reference/image.csv`, `objects.csv`, and `relationships.csv` provide long-form tables with stable IDs. `fixtures/reference/manifest.json` links these files to SHA-256 checksums, spacing, source, segmentation method, and the environment lock. `spec/features.yaml` contains the exact CellProfiler columns from `get_measurement_columns`, with their module, scope, input, settings, dtype, and comparison rule.

The JSON marker `{"nonfinite":"NaN"}` means NaN. The markers `+Infinity` and `-Infinity` preserve signed infinity. JSON `null` remains distinct. Missing sparse object IDs keep their array positions. The CSV `value_json` field contains the same encoded value. Image IDs are fixture names. Object IDs are CellProfiler's one-based label numbers. Relationship rows retain both endpoints and direction.

The fixtures are `empty_image`, `empty_mask`, `one_constant`, `touching_gradient`, `partial_mask`, `holes_disconnected_border`, `sparse_ids`, `volume_3d`, `nonfinite`, `real_cell_cc0`, `real_bbbc007`, `grid_medium`, and `grid_large`. The grids have 64 and 256 objects. The intensity arrays include zeros, constants, gradients, sparse pixels, uint8 saturation, uint16 values, float64 values, NaN, and infinity. The 2D and 3D spacing entries include anisotropy. CellProfiler uses pixel units for the selected features. The runner records spacing without rescaling results.

The real crop comes from `skimage.data.cell()` in scikit-image 0.21.0. Its bundled `cell.png` has SHA-256 `8d23a7fb81f7cc877cd09f330357fc7f595651306e84e17252f6e0a1b3f61515`. The crop is rows 260:356 and columns 210:306. [The dataset documentation states that the image is CC0](https://scikit-image.org/docs/0.21.x/api/skimage.data.html#skimage.data.cell). The `DNA` array is the original crop. `Actin` is a synthetic inversion of it. The fixture segmentation is a deterministic threshold and connected-component label operation, as recorded in the manifest. All engines must receive these saved arrays, not recompute segmentation.

`real_bbbc007` is a second, independent real source: a 128x128 crop of a real two-channel (DNA + Actin) fluorescence image of Drosophila Kc167 cells from BBBC007v1 (Broad Bioimage Benchmark Collection), also CC0. Unlike every other fixture, both its `Cells` and `Nuclei` label arrays are genuine, independently segmented, multi-pixel regions — derived from the dataset's own manually-outlined ground truth, not a synthetic single-pixel placeholder. See `reference/vendor/bbbc007/README.md` for exact provenance, source checksums, and the derivation method.

## Environment limits and observations

The original 4.2.8 app lists older dependencies that lack practical macOS arm64 wheels for this setup. The measurement-only environment retains original CellProfiler 4.2.8 source and core. It uses `h5py==3.11.0` instead of core `~=3.6.0`, `pyzmq==25.1.2` instead of `~=22.3`, and `scikit-image==0.21.0` instead of `0.18.3`. `uv pip check` reports these three conflicts. Compared with the original app metadata, it also uses `scipy==1.9.3` instead of `1.9.0` and `tifffile==2024.8.30` instead of a version before `2022.4.22`. The full installed version list is pinned, so the golden values belong to this exact environment. A stock CellProfiler 4.2.8 installation with its original dependency pins has not been shown to give identical values. The original GUI, MySQL exporter, and file-reader pipeline are not installed. App dependencies for those unused paths, such as `wxPython` and `mysqlclient`, are absent. Pipeline loading logs a missing `MySQLdb` warning while scanning unused modules. Degenerate and nonfinite fixtures produce NumPy runtime warnings. The generator retains their output values and nonfinite markers.

The `sparse_ids` fixture uses maximum ID 16 with gaps. The original modules index arrays by maximum label. A very large sparse ID can require excessive memory, so it is a documented CellProfiler limit and has no golden value. The neighbor mode ignores image masks by design in the original module. The object intensity module can produce nonfinite results for zero intensity or nonfinite input. The image intensity module filters NaN and infinity before its statistics. These rules are captured in fixture values. Parent-child relationships and other neighbor modes have no reference data in this scope.

## Validation record

The clean reference installation completed using `reference/install.sh` in a fresh scratch environment. Two consecutive `reference/run.sh --verify` executions returned exit code 0 and identical results:

```json
{"features":119,"files":68,"fixtures":12,"mode":"verify","sha256":"5df1a4dbc886e952f228d60863b704ab09e7702d13f77a46b25a42fe5c0633b5"}
```

The verified shell environment used:

```bash
export CP_ANYWHERE_REFERENCE_ROOT=/Users/buntend/.hermes/cache/scratch/cp-anywhere-reference-clean
export CP_ANYWHERE_REFERENCE_SOURCE=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cellprofiler
reference/install.sh
reference/run.sh --verify
reference/run.sh --verify
```

The project suite also passed with the original-reference replay enabled: nine tests passed, none skipped.

```bash
CP_ANYWHERE_REFERENCE_PYTHON=/Users/buntend/.hermes/cache/scratch/cp-anywhere-reference-py39/bin/python \
  PYTHONPATH=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cellprofiler \
  uv run --frozen pytest
```

These absolute paths record the verification environment, not installation requirements. The portable commands above use a configurable reference root.
