# Production readiness: bounded validation

This pilot does not establish production readiness.
It separates agreement with CellProfiler from useful correction on real images.
The machine-readable evidence is in `spec/real-data-pilot-summary.json`.

## Reference replay and fixture reproducibility

Reference replay distinguishes floating-point differences from changed scientific outputs.
Inputs, labels, object IDs, counts, configuration, and nonfinite kinds remain strict.
Each array must match its own stored checksum before a numeric comparison can pass.
Measurement comparisons retain the existing feature tolerances.
Threshold replay permits absolute drift up to `1e-7`, with zero relative tolerance.
Native scientific tolerances did not change.

The illumination generator previously used Python's process-dependent `hash()` to seed input images.
It now uses explicit seeds and records the real reference module's hash.
Two fresh reference processes, with `PYTHONHASHSEED=1` and `987654321`, produced 253 identical files for 252 cases.
Native calculation matched all 252 cases at the existing illumination tolerance.
The one-time regeneration changed inputs and corresponding CellProfiler outputs, not masks or configuration.
No native output replaced a reference output.

The canonical segmentation generator also includes the empty-field regression.
A fresh replay matched all 499 files across 101 cases.

## Real-data pilot

The source is BBBC007v1, with DNA and Actin channels and manual outlines.
Archive hashes, excluded RGB images, configuration, cohort membership, and reference versions appear in the evidence file.
The reference is real CellProfiler 4.2.8 at commit `2df090ff281cff80943fe679a1c3192fbda9cd6d`.

Ten complete grayscale fields form leave-one-out calibration groups within their source cohorts.
No held-out field contributes to its own illumination function.
Four additional cases cover two known-shading simulations with noisy independent calibration, an empty field, and a saturated field.
These cases are simulations, not instrument calibration measurements.

The run includes illumination calculation, correction, primary/secondary/tertiary segmentation, measurements, and exports.

| Comparison                                          | Result                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| Illumination functions and corrected images         | All 14 cases pass the existing `1e-6` image tolerance                   |
| Final and diagnostic labels                         | All 14 cases match exactly                                              |
| CSV and Parquet exports                             | All 14 cases retain measured values and keys                            |
| SQLite exports                                      | All 14 cases retain every table's values and keys, not only row counts  |
| End-to-end feature comparison                       | Fails: 1,430 feature-vector checks across 13 cases                      |
| Measurements on identical float64 pixels and labels | 13 of 14 cases pass; 60 Zernike feature-vector checks fail in one field |

Both feature comparisons use the original tolerances: core `rtol=1e-9, atol=1e-10`, shape `rtol=1e-7, atol=1e-10`, and exact discrete values.
Schema, shape, and nonfinite-kind comparisons remain strict.
The 1,490 failed checks in the report include both comparisons, not 1,490 distinct faulty features.

### Fixed defects

An all-foreground distance transform used a constant native sentinel instead of SciPy's actual no-background behavior.
Native declumping then produced 1,555 false nuclei from the empty input.
The corrected implementation matches SciPy's distance transform and produces zero final objects with the pilot configuration.
Real CellProfiler generated the regression fixture.

The reference runner also saved a diagnostic array after a later module changed its underlying object state.
It now copies the secondary array immediately after identification.
This fixes the reference capture rather than changing correct native labels.

### Remaining blockers

End-to-end intensity differences include float32 arithmetic in CellProfiler and float64 arithmetic in the native measurement API.
The report retains these disagreements.
A separate real-CellProfiler measurement pass uses identical float64 inputs to isolate measurement behavior.
This pass does not replace the original pipeline results.

Zernike measurements remain a release blocker for strict full-field parity.
The failing field is `f9620-20P1-POS0002`.
The common-input comparison isolates the disagreement from illumination and segmentation.
No wider tolerance or expected-failure marker hides this result.

The investigation confirmed context-dependent tie selection in CellProfiler's minimum-enclosing-circle calculation.
Equal-angle vertices can be selected differently when unrelated objects share the label image.
A tiny radius change then includes or excludes different boundary pixels.
The 60 failing feature-vector checks contain 90 disagreeing values across three objects.
Native values match isolated CellProfiler measurements, but not the original full-field measurements.

Full labels, real reference measurements, and reproduction commands are preserved in `fixtures/size_shape/zernike_context/README.md`.
The real-reference reproduction passes exactly; the strict native comparison correctly exits with status 1.
Production Zernike code remains unchanged rather than introducing a field-specific tie rule or boundary epsilon.

Cytoplasm can contain missing object IDs after subtraction.
The pilot retains those IDs and measures intensity, but explicitly excludes unsupported cytoplasm shape measurements.
Identification-only `Location_Center_X/Y/Z` values and relationship rows are outside this bounded measurement comparison.
The reference JSON preserves identification-only values separately.

## Scientific quality

Agreement with CellProfiler does not establish that this correction helps an experiment.
Background spatial variation increased in 9 of 10 DNA fields and 8 of 10 Actin fields with the tested configuration.
These results do not support recommending this configuration for those acquisitions.

Foreground Dice overlap with manual regions ranges from approximately 0.687 to 0.835 for nuclei and 0.617 to 0.779 for cells.
These diagnostic scores do not claim to reproduce the published benchmark's scoring conventions.
No raw-versus-corrected segmentation comparison or treatment/control analysis establishes biological preservation.

Known-shading simulations show recovery against their unmodified references, but do not validate real instrument correction.
Saturated inputs remain lossy.
Acquisition-specific calibration, acceptance criteria, and independent biological controls remain necessary.

## Reproduction

The reference interpreter must use the pinned CellProfiler source and dependencies.
`REFERENCE_PYTHON` and `REFERENCE_SOURCE` below identify that environment.
`PILOT_DATA` holds the source archives, and `PILOT_RUN` must be a fresh output directory.

```sh
env -u CONDA_PREFIX uv sync --frozen --all-groups
env -u CONDA_PREFIX uv run --no-sync maturin develop --uv
env -u CONDA_PREFIX uv run --no-sync python -m scripts.validation.real_data_pilot \
  --data-dir "$PILOT_DATA" \
  --work-dir "$PILOT_RUN" \
  --report "$PILOT_REPORT" \
  --reference-python "$REFERENCE_PYTHON" \
  --reference-source "$REFERENCE_SOURCE"
```

The pilot saves its report and exits nonzero when strict parity fails.
It refuses to overwrite a nonempty reference output directory.
Heavy arrays stay outside the repository.

## Local quality gates

The reviewed tree passed 633 Python tests with both real-reference replay tests enabled.
All 160 Rust unit tests, Clippy with warnings denied, and the full pre-commit suite passed.
The real-data pilot deliberately reports failure despite these software checks.
A hosted CI run of these uncommitted edits remains unverified.
