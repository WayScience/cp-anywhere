# AutoSegment: image quality, pipeline selection, and reporting

AutoSegment picks and runs a segmentation pipeline with minimal manual
tuning, after checking that each image meets a basic technical quality
bar. It follows the five-stage design in the project's own
`cp-anywhere-autosegment-plan.md` (repository root): Stage 0 image
quality, Stage 1 pipeline selection, Stage 2 parameter search, Stage 3
apply, Stage 4 report and optional human review.

**Scope in this release:** Stage 0 (`cp_anywhere.quality`) is complete.
Stage 1-4 (`cp_anywhere.autosegment`) wrap this project's own existing,
oracle-verified classical segmentation (`identify_primary_objects` +
`identify_secondary_objects` + `identify_tertiary_objects`) into one
named pipeline and run real selection, search, apply, and reporting
logic over it. A second, learned nuclei candidate -- `"stardist"`
(plan milestone M2) -- is implemented against the real pretrained
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
`"review_required"` (too few accepted images, or no candidate pipeline
passed its hard constraints). Stage 2's random search is seeded and
reproducible: the same `seed` and inputs always pick the same
parameters. Hard constraints reject a trial with no nuclei, an empty or
near-total whole-cell mask, or a nucleus with no matching cell.

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

## What this module intentionally does not do yet

- No acquisition-group (plate/well/batch) calibration inside Stage 0 --
  callers calibrate their own profile.
- The Stage 4 report has the plan's section structure (image-quality
  gate, decision summary, parameter search, human review,
  reproducibility) but not yet the full literature technique catalog,
  response-surface plots, or a failure-crop gallery.
- No external validation against expert-reviewed ground-truth masks.

See `cp-anywhere-autosegment-plan.md` for the full design and the
remaining milestone checklist.
