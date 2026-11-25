# AutoSegment: image quality, pipeline selection, and reporting

AutoSegment picks and runs a segmentation pipeline with minimal manual
tuning, after checking that each image meets a basic technical quality
bar. It follows a five-stage design: Stage 0 image
quality, Stage 1 pipeline selection, Stage 2 parameter search, Stage 3
apply, Stage 4 report and optional human review.

**Scope in this release:** Stage 0 (`cp_anywhere.quality`) is complete.
Stage 1-4 (`cp_anywhere.autosegment`) wrap this project's own existing,
oracle-verified classical segmentation (`identify_primary_objects` +
`identify_secondary_objects` + `identify_tertiary_objects`) into one
named pipeline and run real selection, search, apply, and reporting
logic over it. A second, learned nuclei candidate -- `"stardist"` --
is implemented against the real pretrained
StarDist `2D_versatile_fluo` model, but is **off by default**: it
needs the native extension rebuilt with cp-anywhere's `stardist-model`
Cargo feature (`maturin develop --features stardist-model`). Without
that build, requesting `"stardist"` raises `RuntimeError` naming the
rebuild command rather than silently falling back to the classical
threshold. External scientific validation against expert-reviewed masks
is out of scope for this module; it needs real annotated data this
release does not ship.

## Stage 0: the image-quality gate

```python
from cp_anywhere import quality

profile = quality.QcProfile.from_cppipe("published_qc_pipeline.cppipe")
result = quality.assess(images, profile)  # images: id -> {channel: array}
result.accepted  # image IDs eligible for AutoSegment's pipeline search
```

A `QcProfile` is calibration evidence, not an invented universal
standard. Build one from a real, pinned pipeline's own
`MeasureImageQuality`/`FlagImage` pair with `QcProfile.from_cppipe`, or
from explicit, caller-calibrated bounds with `QcProfile.from_criteria`.
This module ships no built-in numeric thresholds and no preset named
profile (e.g. no built-in `"cell-painting"` profile) -- supply your own.

Metrics reuse this project's own `MeasureImageQuality` kernel
(`ImageQuality_*`, already verified against CellProfiler 4.2.8) plus a
few new, non-CellProfiler `Intensity_*` robust statistics (median, MAD,
extreme quantiles, a simple background estimate, near-zero fraction).
Every image gets `pass`, `review`, or `fail`:

- **fail**: a structural problem (missing/non-2D/wrong-dtype/non-finite
  channel), or a configured bound was outside range.
- **review**: the profile needed a channel or measurement the caller
  did not supply -- ambiguous, not confidently bad.
- **pass**: every configured criterion was satisfied.

`review` images need a recorded override (`QualityGateResult.accept_overrides`)
to enter AutoSegment's selection; `fail` images never enter automatic
selection.

## Stage 1-4: `cp_anywhere.autosegment.auto`

```python
from cp_anywhere import autosegment

result = autosegment.auto(
    images,  # id -> {channel: array}, same shape as quality.assess
    channels={"nuclei": "DNA", "cells": "AGP"},
    quality_profile=profile,
    pipelines=["classical"],  # or ["stardist"] (see the M2 note below)
    trials=16,
    seed=42,
    review_mode="optional",  # default: never blocks Stage 3
)
result.save("segmentation-config.json")
masks = result.segment(images)  # Stage 3, works regardless of review_mode
result.report("report.html")  # Stage 4, self-contained and offline
```

`result.status` is `"selected"`, `"awaiting_approval"` (selected, but
`review_mode="required"` withholds final approval), or
`"review_required"` (too few accepted images, no candidate pipeline
passed its hard constraints, or the tuned configuration did not hold up
on the held-out check). Stage 2's random search is seeded and
reproducible: the same `seed` and inputs always pick the same
parameters. Hard constraints reject a trial with no nuclei, an empty or
near-total whole-cell mask, or a nucleus with no matching cell.

The accepted images are split into a tuning set and a held-out set.
Search and screening run only on the tuning images; after tuning, the
winning configuration is re-run on the held-out images, which it never
saw during tuning. Pass `acquisition_groups` (a map from image ID to a
group name, such as its plate) and images from the same acquisition
always land on the same side of the split, so held-out evidence comes
from acquisitions the search never touched. Without groups the split
only alternates the sorted image IDs. Every screening, tuning, and
held-out run records one evaluation row (image, phase, pipeline,
runtime seconds, failures) on `result.evaluations`, with the two sides
listed on `result.tuning_ids` and `result.validation_ids`. Pass
`max_field_seconds` to bound how long one image may take in a
candidate run; images that run longer fail that run with a "runtime
limit" failure and leave the search, which ends the run as
`"review_required"` if no candidate evidence remains.

`result.review` records accept/override/reject decisions
(`result.review.accept(reviewer=...)`, `.override(selected=..., ...)`,
`.reject(...)`) without altering the original automatic recommendation;
`result.save_review("review.json")` persists that audit trail.

## The StarDist (M2) nuclei candidate

StarDist uses the pretrained `2D_versatile_fluo` model on one 2D
fluorescent nuclei channel. It scales the image by its 1st and 99.8th
percentiles, then runs ONNX inference in Rust on the CPU. Existing
propagation and subtraction kernels produce the cell and cytoplasm
masks. Both AutoSegment candidates exclude all nucleus pixels from
cytoplasm. This does not change `IdentifyTertiaryObjects` defaults.

From a source checkout, enable the native feature:

```sh
uv run --frozen maturin develop --features stardist-model
```

Use `pipelines=["stardist"]` to evaluate only StarDist, or pass
`pipelines=["classical", "stardist"]` to compare both. If the feature is
absent, requesting StarDist raises an error; it does not fall back.
The model is included in the Python package. Inference needs no network
access, TensorFlow, or separate Python StarDist installation.

The initial probability and suppression thresholds (about 0.479 and
0.3) come from the pretrained model. The saved configuration includes
the model name and SHA-256. See `src/cp_anywhere/data/stardist/NOTICE.md`
for the model source, license, and conversion details.

The Rust decoder uses rasterized overlap for non-maximum suppression.
It can differ from upstream StarDist at boundaries or near suppression
thresholds. Tests exercise the real model on synthetic fluorescence
fields. They do not establish accuracy on a new assay. This integration
supports native CPU builds only; the browser API does not expose StarDist.

Run the opt-in checks with `uv run --frozen poe test-stardist`.

## The InstanSeg nuclei candidate

InstanSeg uses the pretrained `single_channel_nuclei` model (upstream
instanseg/instanseg, `instanseg_models_v0.1.2` release pin, model id
26423248\) on one 2D nuclei channel. It scales the image by its 0.1st and
99.9th percentiles (the model's own bioimage-io `scale_range` pin, with
eps 1e-6), then runs the ONNX fcn network in Rust on the CPU and applies
a native port of upstream's deterministic postprocessing: peak
local-maxima seeding on the seed map, per-centroid feature windows with
the pixel classifier, fragment cleanup, and an
intersection-over-minimum-area overlap merge. Cells and cytoplasm grow
classically, exactly like the other pipelines.

From a source checkout, enable the native feature:

```sh
uv run --frozen maturin develop --features instanseg-model
```

Use `pipelines=["instanseg"]` to evaluate only InstanSeg, or pass
`["classical", "instanseg"]` (or all three) to compare. If the feature is
absent, requesting InstanSeg raises `RuntimeError` naming the rebuild
command; it does not fall back. The models are included in the Python
package. Inference needs no network access, no torch, and no separate
Python InstanSeg installation.

InstanSeg's parameter search space is intentionally empty: the released
model's own graph freezes all of its postprocessing parameters
(`min_size`, `mask_threshold`, `seed_threshold`, `peak_distance`,
`overlap_threshold`, `fg_threshold`, `window_size`, `cleanup_fragments`)
to the model's saved v0.1.2 values, and the port reproduces upstream's
own published test output label-for-label. The saved configuration
records the model name, the shipped ONNX SHA-256, the upstream
TorchScript SHA-256, and the release URL. See
`src/cp_anywhere/data/instanseg/NOTICE.md` for the full provenance,
licensing (Apache-2.0 code and weights; CC0/CC-BY training datasets),
and the released-graph divergences from the v0.1.2 wheel sources the
port documents. Declared gap: upstream's cells-and-nuclei model variant
is not implemented; InstanSeg here always produces nuclei only.

Run the opt-in checks with `uv run --frozen poe test-instanseg`.

## The SplineDist nuclei candidate

SplineDist ships the pretrained `bbbc038_10` fluorescence-nuclei model
(upstream uhlmanngroup/splinedist, trained on DSB2018/BBBC038; weights
pinned to Zenodo record 7193306) on one 2D nuclei channel. It scales
the image by its 1st and 99.8th percentiles (upstream's csbdeep
normalizer default that the training run used, with csbdeep's eps
1e-20), reflect-pads to the network's 16-voxel divisibility exactly
like `SplineDistPadAndCropResizer`, then runs the ONNX graph in Rust on
the CPU (grid-(2,2)-compressed probability and 20-channel spline
control-point heads) and applies a native port of upstream's decode:
control points through the `phi` B3 basis into closed polygons,
bbox-windowed intersection-over-min-area non-maximum suppression
against the model's own thresholds, and probability-ordered
rasterization where higher-probability objects overwrite. Cells and
cytoplasm grow classically, exactly like the other pipelines.

From a source checkout, enable the native feature:

```sh
uv run --frozen maturin develop --features splinedist-model
```

Use `pipelines=["splinedist"]` to evaluate only SplineDist, or include
it alongside the other candidates. If the feature is absent, requesting
SplineDist raises `RuntimeError` naming the rebuild command; it does
not fall back. The model and its `phi` basis are included in the Python
package. Inference needs no network access, no TensorFlow, and no
Python splinedist installation.

SplineDist's parameter search space is intentionally empty: the shipped
model's own `thresholds.json` freezes the decode (`prob` 0.4775366187095642,
`nms` 0.3), and the native port reproduces upstream's real
`predict_instances` output label-for-label with those exact thresholds
(123-instance oracle in `crates/cp_anywhere/tests/splinedist_fixtures/`).
The saved configuration records the model name, the shipped ONNX
SHA-256, the upstream weights SHA-256, and the Zenodo source URL. See
`src/cp_anywhere/data/splinedist/NOTICE.md` for the full provenance
and licensing (BSD-3-Clause code; CC-BY-4.0 weights; StarDist/BSD-3
built). Declared gap: only the DSB2018 fluorescence-nuclei model is
shipped; upstream's H&E `conic` model variants exist but are not
implemented. SplineDist here always produces nuclei only.

Run the opt-in checks with `uv run --frozen poe test-splinedist`.

## What this module intentionally does not do yet

- No acquisition-group (plate/well/batch) calibration inside Stage 0 --
  callers calibrate their own profile.
- The Stage 4 report has the plan's section structure (image-quality
  gate, decision summary, parameter search, human review,
  reproducibility) but not yet the full literature technique catalog,
  response-surface plots, or a failure-crop gallery.
- No external validation against expert-reviewed ground-truth masks.

See the Stage 1-2 notes above for what is and is not implemented.
