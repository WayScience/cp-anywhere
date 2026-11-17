# CellProfiler `.cppipe` pipeline import

## What this is

`cp_anywhere.load_pipeline(path)` parses a real CellProfiler `.cppipe`
text file and translates the settings of every module this project
implements (`IdentifyPrimaryObjects`, `IdentifySecondaryObjects`,
`IdentifyTertiaryObjects`,
`MeasureImageIntensity`, `MeasureObjectIntensity`, `MeasureObjectSizeShape`,
`MeasureGranularity`, `MeasureObjectIntensityDistribution`,
`MeasureTexture`, `MeasureColocalization`,
`MeasureObjectNeighbors`, `ExportToDatabase`,
`CorrectIlluminationCalculate`, `CorrectIlluminationApply`, and `SaveImages`)
into this project's own configuration.
Supported configuration can be reused without manual re-entry.
The limits below apply to each module.
Import success does not imply full CellProfiler execution or automatic output.

**This is a settings importer, not a full pipeline execution engine.**
File-matching/organization modules (`Images`, `Metadata`,
`NamesAndTypes`, `Groups`, and anything else not explicitly listed
above) are recognized and skipped -- reported in
`Pipeline.skipped_modules`, never silently dropped -- not executed.
The caller supplies already-loaded image arrays by the same names the
pipeline's own modules reference (`cp_anywhere.io`/`cp_anywhere.cli`
already handle loading files by name; replicating CellProfiler's own
file-matching/grouping rules is out of scope).

## The `.cppipe` text format (verified, not guessed)

The classic `.cppipe` text format -- `CellProfiler Pipeline: ...`
header, then one block per module (`ModuleName:[attr:'val'|...]`
followed by indented, strictly *positional* `Label:Value` setting
lines) -- was confirmed by generating real pipeline files with the
pinned CellProfiler 4.2.8 oracle
(`cellprofiler_core.pipeline.io._v5.dump`, the real serializer) and
reading their literal text (`scripts/cppipe/generate_fixtures.py`),
not reverse-engineered from reading the loader's source alone. Every
setting's label is decorative and discarded -- only its position in
the module's declared `settings()` order matters, exactly matching
real CellProfiler's own loader.

`.json`-extension pipeline files (a separate, newer serialization in
`cellprofiler_core.pipeline.io._v6`) are **not** supported; `.cppipe`
is the format CellProfiler's own GUI and essentially every shared
pipeline in the wild uses.

## Why no CellProfiler dependency, and why only the current revision

This importer has **no CellProfiler dependency** at all (it ships as
part of `cp_anywhere` itself). Replicating arbitrary older pipeline
files would require re-implementing every supported module's own
`upgrade_settings` version-migration chain -- years of legacy-format
branches per module -- which this project deliberately does not do.

Instead, `load_pipeline` accepts only the **current**
`variable_revision_number` for each supported module (the exact value
the pinned CellProfiler 4.2.8 oracle itself writes: `15` for
`IdentifyPrimaryObjects`, plus its embedded `Threshold` sub-module's
own `threshold_setting_version` of `12`; `10` for
`IdentifySecondaryObjects` (sharing the same `threshold_setting_version`
tail); `4` for `MeasureObjectIntensity` and `MeasureImageIntensity`;
`3` for `MeasureObjectNeighbors`) and raises `cp_anywhere.CppipeError`,
naming the unsupported revision, for anything older -- never a silent,
partial, or guessed translation. Re-opening and re-saving the pipeline
in a current CellProfiler upgrades it to the supported revision.

## `IdentifyPrimaryObjects`: settings mapping

Every setting `IdentifyPrimaryObjects.settings()` declares (34 total
at revision 15: `ImageSegmentation`'s `x_name`/`y_name`, 14 of the
module's own settings, `threshold_setting_version`, then
`Threshold.settings()[2:]`'s 17 settings) maps directly to
`cp_anywhere.identify_primary_objects`'s own parameter, using the same
display-string vocabulary CellProfiler's own module constants declare
(`UN_*`/`WA_*`/`FH_*`/`LIMIT_*` in `identifyprimaryobjects.py`;
`TS_*`/`TM_*`/`O_*`/`RB_*` in `threshold.py`) -- see
`spec/segmentation-modules.md` for `cp_anywhere`'s own side of this
mapping. `threshold_scope` selects whether `global_operation` or
`local_operation` is the active threshold method (matching
CellProfiler's own `Threshold.threshold_operation` property exactly);
`limit_choice="Erase"` maps to `maximum_object_count=<value>`,
`"Continue"` to `maximum_object_count=None`.

### `IdentifyPrimaryObjects` explicit gaps

- **`Use advanced settings? = No`** ("basic mode"). This is not a
  cosmetic GUI toggle: real CellProfiler's own `run()` branches on
  `self.basic` (`not use_advanced`) to *override* several saved
  settings at run time regardless of their literal saved values
  (always auto-threshold-correct, always fill holes at both stages,
  always declump with Intensity/Intensity, never erase on an object
  count limit, and more -- confirmed directly in
  `cellprofiler/modules/identifyprimaryobjects.py`'s `run()`/
  `separate_neighboring_objects()`). Replicating this would mean a
  second, parallel execution path, not a settings translation; an
  explicit error is raised instead. Re-save with "Use advanced
  settings?" set to Yes.
- **`unclump_method = "Laplacian of Gaussian"`.** `cp_anywhere`'s
  `unclump_method` has no `"log"` value at all (see
  `spec/segmentation-modules.md`'s own declared gap for this).
- **`threshold operation = "Measurement"`** (a per-image threshold
  read from an upstream measurement column). Not implemented by
  `cp_anywhere`'s segmentation core at all.

## `IdentifySecondaryObjects`: settings mapping and explicit gaps

Every setting `IdentifySecondaryObjects.settings()` declares (28 total
at revision 10: `ImageSegmentation`'s `x_name`/`y_name`, 8 of the
module's own settings, `threshold_setting_version`, then the *same*
17-setting `Threshold.settings()[2:]` tail `IdentifyPrimaryObjects`
also carries) maps to `cp_anywhere.identify_secondary_objects`'s own
parameters (`spec/segmentation-modules.md` has `cp_anywhere`'s own
side). `x_name` becomes `primary_object` (the object set this step
grows around, resolved against a translated `IdentifyPrimaryObjects`
step's output in the same pipeline -- `CppipeError` naming the object
set if it isn't one); `y_name` becomes `output_object`; `image_name`
becomes `input_image`.

Only **`Propagation`** (the module's own real default) and
**`Distance - N`** are implemented, each with their own disjoint
settings: `Propagation` uses `regularization_factor` plus the full
threshold tail (identical mapping to `IdentifyPrimaryObjects`'s own,
reusing the same display-string dictionaries); `Distance - N` uses
only `distance_to_dilate` and ignores every threshold setting entirely
(CellProfiler's own `Distance - N` never thresholds the secondary
image). `fill_holes` (a plain Yes/No here, unlike
`IdentifyPrimaryObjects`'s three-way setting) applies to both methods.

Secondary seeds come from the primary result's `unedited_labels`, not
`small_removed_labels`. Edge-touching seeds remain even if primary
filtering excluded them; other seed pixels survive only where kept
primary labels are nonzero. Grown labels are mapped to final primary
IDs by overlap, so filtering and renumbering do not erase valid cells.
Hole filling changes background only and preserves nested foreground
objects. Secondary results retain `small_removed_labels` before this
mapping for downstream neighbor geometry.

`fixtures/cppipe/secondary_regressions/` covers interior exclusions,
border competition, oversized primaries, renumbering, nested objects,
and all neighbor measurement columns against real reference runs.

### `IdentifySecondaryObjects` explicit gaps

- **`method = "Watershed - Gradient"` / `"Watershed - Image"` /
  `"Distance - B"`.** Not implemented by `cp_anywhere`'s segmentation
  core at all; `CppipeError` names the unsupported method.
- **`wants_discard_edge = Yes`** ("Discard secondary objects touching
  the border of the image?"). Not implemented; `CppipeError` raised.
- **`wants_discard_primary`** (discarding/renaming the primary object
  set that fed this step) is read but has no effect -- `cp_anywhere`
  never mutates the primary object set as a side effect of running
  `IdentifySecondaryObjects`.

## `IdentifyTertiaryObjects`: settings mapping and limits

Revision 3 stores four settings: larger objects, smaller objects,
output name, and `shrink_primary`. The importer maps these to
`TertiaryObjectsStep` without changing their order or values.
`Pipeline.run()` resolves the object names, subtracts the smaller
objects, and makes the output available to measurement modules.

Both `shrink_primary = Yes` and `No` are supported. `Yes` retains the
smaller objects' one-pixel, eight-connected outline. `No` removes every
smaller-object pixel. Output labels preserve the larger objects' IDs,
even when subtraction leaves gaps in the numbering.

The second item returned by `Pipeline.run()` contains secondary and
tertiary results. No fourth return item is added. Inputs must already
share a nonempty 2D grid. CellProfiler crop/resize metadata and automatic
parent, child-count, and location measurements are not implemented here.

The two real CellProfiler fixtures in `fixtures/cppipe/tertiary/` cover
both shrink settings. Tests compare every label and downstream mean
intensity with the reference output.

## `MeasureObjectSizeShape`: settings mapping and limits

Revision 3 has three settings: the object list, `calculate_zernikes`,
and `calculate_advanced`. All 25 standard 2D columns, all 30 Zernike
columns, and all 53 advanced columns are implemented in the shared
Rust core, native Python, and WASM. Each imported module's flags apply
only to its own requested object sets; columns are merged with other
measurements without sharing flags across compartments.

3D shape measurement is an explicit gap. The integrated API also
rejects sparse label IDs because the reference packs different shape
columns inconsistently. It never silently renumbers input objects or
returns misaligned export rows. See `docs/size-shape.md` for the API,
nonfinite-value rules, and native/browser reproduction commands.

## `MeasureGranularity`: settings mapping and limits

Revision 4 has seven settings: the image list, "measure within
objects?", the object list, both subsampling factors, the structuring
element radius, and the spectrum length. 2D granularity spectra
(image-level and per-object) are implemented in the shared Rust core,
native Python, and WASM as a standalone `measure_granularity` call --
not a `measure()` family, since its per-scale result shape and
per-image settings do not fit that unified contract. Every listed
image is measured independently; results merge into the same
`MeasurementResult` under real CellProfiler's own
`Granularity_<scale>_<image name>` feature names.

A 1-row or 1-column image is an explicit gap: real CellProfiler's own
module divides by `(dimension - 1)` while rescaling its working image
and raises `ZeroDivisionError` there, confirmed live against the
pinned reference -- there is no oracle value to match. See
`docs/granularity.md` for the API, the resampling/erosion/
reconstruction algorithm, the real-CellProfiler `Image(..., convert=True)` precision contract intensity measurements must match,
and native/browser reproduction commands.

## `Measure*` modules: settings mapping and explicit gaps

Per `spec/compatibility.md`'s own declared scope:

- **`MeasureObjectIntensity`**: `images_list`/`objects_list` (CellProfiler's
  `", "`-joined `ListSubscriber` serialization, confirmed against a real
  saved pipeline) map directly; no settings outside this scope exist.
- **`MeasureImageIntensity`**: `images_list` maps directly.
  `wants_objects = Yes` ("Measure the intensity only from areas
  enclosed by objects?") is an explicit gap (object-restricted
  `MeasureImageIntensity` is out of `cp_anywhere`'s declared scope);
  `wants_percentiles`/`percentiles` are accepted but have no effect --
  `cp_anywhere.measure`'s `image_intensity` family always computes its
  fixed, declared percentile columns (`spec/features.yaml`) regardless
  of this module's own percentile setting.
- **`MeasureObjectNeighbors`**: any self-paired object set supports
  `distance_method = "Adjacent"`. With `wants_excluded_objects = Yes`,
  translated primary and secondary results supply `small_removed_labels`;
  tertiary results use their final labels, as CellProfiler does. These
  diagnostic layers are module-specific: the primary layer already had
  border filtering, while the secondary layer retains pre-filter grown
  labels. Diagnostic IDs are matched to final IDs by pixel overlap.
  Counts, touching percentages, distances, and angles use diagnostic
  geometry. Closest-object numbers and relationship rows use retained
  IDs. An object set supplied only as a raw array has no diagnostic layer;
  requesting excluded-object geometry for it raises `CppipeError`.
  Cross-object-set neighbors and `"Within X pixels"` remain explicit gaps.
  The 2D `"Expand until adjacent"` method is implemented as described below.

### Real-world coverage note

A November 2026 survey of 10 real CellProfiler pipelines from the
WayScience GitHub org found `MeasureObjectNeighbors`'s old `"Cells"`
-only, `wants_excluded_objects = No`-only restriction blocked **all
10/10** of them from loading at all (every one uses
`wants_excluded_objects = Yes`, CellProfiler's own default, and most
also run it once per compartment -- e.g. once for `Nuclei`, once for
`Cells`). Both restrictions were lifted the same session; 9/10 now
load end to end (the remaining one uses `"Expand until adjacent"`, a
distinct, still-open gap). `IdentifySecondaryObjects` support (this
section) was added next, covering its real default (`Propagation`)
and `Distance - N`. Tertiary-object subtraction is also implemented.
The measurement modules each used by 8-10 of the 10 surveyed pipelines
(`MeasureGranularity`, `MeasureObjectIntensityDistribution`,
`MeasureTexture`, and `MeasureColocalization`) are all implemented
now, with real-reference fixtures for each module's own settings
surface. `MeasureObjectNeighbors`'s `"Expand until adjacent"` distance
method -- the one remaining blocker from the survey -- has since been
implemented too (see this file's own section below); every surveyed
pipeline now loads and runs end to end.

## `MeasureObjectNeighbors`: "Expand until adjacent"

`distance_method = "Expand until adjacent"` (CellProfiler's own
`D_EXPAND`) is translated into `MeasurementStep.wants_expand = True`
and threaded through `measure(..., neighbors_expand=[...])`. Real
CellProfiler's own algorithm: every background pixel of the object's
geometry (`neighbors_full_objects` when supplied, else its own kept
labels) is first assigned the label of its nearest foreground pixel
(a Euclidean nearest-label fill -- real CellProfiler's own
`scipy.ndimage.distance_transform_edt(labels == 0, return_indices=True)` then `labels[i, j]`, oracle-verified against
real scipy; see `scripts/reference/generate_neighbors_expand.py` and
`fixtures/neighbors_expand/*.json`), and then the exact same
touching/relationship-edge detection used for `"Adjacent"` runs on
that fully covered label image. Object centers and the closest-
object/distance/angle columns are unaffected -- they always use the
original, unexpanded geometry, matching real CellProfiler's own
`run()` (it only reassigns a local variable, never
`objects.small_removed_segmented` itself). Feature columns get an
`_Expanded` suffix instead of `_Adjacent`
(`Neighbors_PercentTouching_Expanded`, etc. -- CellProfiler's own
`S_EXPANDED` scale string). 2D only: a 2D+3D mixed pipeline's 3D scope
would need volumetric nearest-label fill, an explicit, out-of-scope
gap (raises a clear error naming the scope). `"Within X pixels"`
(`D_WITHIN`) remains a separate, out-of-scope gap.

## `MeasureObjectIntensityDistribution`, `MeasureTexture`, `MeasureColocalization`: settings mapping and limits

All three are 2D-only ports of real CellProfiler 4.2.8 modules (their
3D volumetric modes are explicit gaps, consistent with every other
module in this project), each implemented as a standalone
`cp_anywhere.measure_*` call -- not a `measure()` family -- with its
own real-reference fixtures and its settings translated by
`load_pipeline`:

- **`MeasureObjectIntensityDistribution`** (revision 6): the default
  center choice `"These objects"` only. `"Centers of other objects"`
  and `"Edges of other objects"` (which need a second, separately
  selected object set) are explicit gaps, rejected at translation
  time. Real CellProfiler adds extra object/bin-count groups via
  `add_object()`/`add_bin_count()`; only the single-group case
  (one object set, one bin-count group) is translated, and the
  field-count check rejects more with a clear error.
- **`MeasureTexture`** (revision 7): the "Hidden" scale count
  validates the number of trailing "Texture scale to measure"
  settings; any number of scales is supported. Image-wide texture
  genuinely never applies the image mask in real CellProfiler
  (`run_image` has no mask parameter) -- this port replicates that
  quirk exactly, documented rather than "fixed".
- **`MeasureColocalization`** (revision 5): the full `do_all`/five
  individual `do_*` flag surface, all three `fast_costes` modes
  (`Faster` bisection, `Fast`, and `Accurate` linear search), and the
  real `Correlation_{Feature}_{img1}_{img2}` column names including
  the asymmetric Correlation/Slope pair orientation from
  `get_image_pairs()`. The port shares one `costes_scale_max` across
  every pair in a call (real CP resolves per-pair via `get_scale`);
  pass the largest bit-depth scale when your images differ. 3D is an
  explicit gap.

One shared behavior all measurement modules depend on: `Pipeline.run()`
hands every measurement the same pixel data real CellProfiler's
`ImageSet` holds -- `Image(..., convert=True)` semantics (integer
pixels rescaled to `[0, 1]` float32, floats downcast to float32),
replicated by `_cp_converted_pixels`. Segmentation keeps the raw
arrays because `identify_primary_objects` applies that same integer
rescale itself while thresholding (its own oracle-verified numerics).

## `ExportToDatabase`: settings mapping and limits

`ExportToDatabase` (revision 28) is translated into an
`ExportToDatabaseStep`, not merged into `measure()`'s own combined
call -- it is a terminal write action, not a measurement, and real
CellProfiler itself only runs it once at the end of a pipeline (often
after every image in an experiment), not per image the way
`Pipeline.run()` operates. Call its own `step.write(result, path)`
once you have accumulated whatever `MeasurementResult`(s) you want
exported. Only a deliberately narrow slice of this very large, mostly
MySQL/CellProfiler-Analyst-oriented module is supported -- see
`spec/database-export.md`'s own section and `ExportToDatabaseStep`'s
docstring for the full list of explicit, rejected-at-translation-time
gaps (MySQL, the default combined-object-table modes, table prefixes,
the CPA properties file, the relationship table, per-image
aggregates). `load_pipeline` validates the real module's four
"Hidden:" group counts (image/group-field/filter-field/
workspace-measurement groups) to confirm the settings list's total
length, but never reads their content -- none of that is implemented.

## Illumination correction: configuration and limits

`CorrectIlluminationCalculate` revision 2 maps its 23 positional fields to
`IlluminationCalculateStep`.
The importer supports `Each` mode and five smoothing methods:
`No smoothing`, `Median Filter`, `Gaussian Filter`, `Fit Polynomial`, and
`Smooth to Average`.
It rejects Convex Hull, Splines, and both `All` modes.
The standalone `calculate_illumination_across` API supplies shared-function
calculation without CellProfiler's cycle scheduling.
Retained averaged/dilated image configuration has no effect.

`CorrectIlluminationApply` revision 5 maps repeated four-field image groups
and two trailing clipping flags to `IlluminationApplyStep`.
Each group identifies the source, output, correction image, and divide/subtract method.

`Pipeline.run()` calculates correction images before application, segmentation,
and measurements.
It groups calculation and application modules rather than preserving arbitrary
interleaving from the source pipeline.
A later calculation that depends on an earlier application is therefore not supported.
The runner also does not return intermediate correction images.
`spec/illumination.md` gives the standalone API and numerical evidence.

## `SaveImages`: explicit writes, not automatic pipeline output

Revision 16 has 18 positional fields and maps to `SaveImagesStep`.
The importer accepts `Image`, `Single name`, and TIFF/PNG/JPEG/NPY output.
It rejects mask/cropping output, movie stacks, HDF5, and other naming methods.

`step.write(image, path)` writes a caller-supplied array to an explicit destination.
Its overwrite check respects the imported `overwrite` flag.
`effective_bit_depth` honors the imported depth for TIFF and forces 8-bit
conversion for PNG/JPEG.
NPY bypasses conversion.
The direct `save_array` API instead uses the caller's explicit bit depth.

`Pipeline.run()` does not execute save steps or expose corrected arrays for them.
Callers use the standalone correction API for explicit image output.
Directory resolution, metadata substitution, cycle timing, filename measurements,
subfolders, and TIFF compression configuration are not implemented.
The importer currently consumes some unsupported configuration without rejecting it.
These are compatibility gaps, not guarantees from a successful import.
`spec/illumination.md` records the full current scope and reproduction commands.

## A combined-measurement superset, not a subset

`cp_anywhere.measure()` has no per-module feature-subset switch: one
call computes every declared feature for every named image against
every named object. `Pipeline.run()` unions every translated
intensity and neighbor module's requested image/object names into a single
`measure()` call. Size/shape modules run separately with their own flags
and object selection, then merge their named columns into that result. If a real pipeline deliberately measures only some
image/object combinations (e.g. `MeasureObjectIntensity` with just
`DNA` against `Cells`, deliberately *not* `Actin` against `Cells`),
`cp_anywhere`'s output is a **superset**: every value the source
pipeline explicitly requested is present and correct, but extra
columns for combinations it didn't request may also appear. This is
not claimed as an exact column-set match, only a correctness guarantee
for everything that *was* requested.

## Reproduce

```sh
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/cppipe/generate_fixtures.py   # writes fixtures/cppipe/
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/cppipe/generate_secondary_fixture.py   # writes fixtures/cppipe/secondary/

PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/cppipe/generate_tertiary_fixtures.py

uv run --frozen pytest tests/cppipe
```
