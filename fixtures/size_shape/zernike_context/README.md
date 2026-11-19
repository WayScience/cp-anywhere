# Zernike full-field parity evidence

## Current status

Native now coordinates circle iterations across all labels and uses NumPy 1.23.5's global angle sort. The strict check passes all 90 target scores at the original `rtol=1e-10`, `atol=1e-10`. A fresh 14-case BBBC007 run against independently executed CP 4.2.8 passes with zero mismatches, including identical-input measurements. Report: `~/.hermes/cache/scratch/cp-anywhere-production-pilot/zernike-global-fixed-20261119/report.json`. Neither oracle nor tolerances changed.

Browser WASM passes 32 existing size/shape fixture cases but fails an additional full-field check for Nuclei object 6: 30 Zernike values on each of its two entry points. Report: `~/.hermes/cache/scratch/cp-anywhere-production-pilot/zernike-full-field-wasm-20261119.json`. Native parity does not establish WASM full-field parity.

## Historical diagnosis (before the native fix)

The following measurements preserve the former per-object discrepancy.

The source field is `f9620-20P1-POS0002`, group `f9620--f9620-20P1-POS0002`.
Both 400-by-400 arrays contain all 101 original object labels.
`labels.npz` preserves the complete `Nuclei_labels` and `Cells_labels` arrays, including background.
The fixture contains no intensity image or cropped replacement geometry.
`oracle.json` contains hashes, versions, actual CP scores, circle values, excluded pixels, and tie decisions.

## Verified cause

`centrosome.minimum_enclosing_circle` calls `scipy.ndimage.minimum_position` on angles from multiple objects.
SciPy 1.9.3 uses `input.ravel().argsort()` without a stable sort.
Equal angles can select different vertices when unrelated objects change the sort input.
The selected vertex changes the computed circle radius by a few floating-point units.
The circle center remains identical for these objects.

The Zernike calculation normalizes pixel coordinates before applying the exact `x*x + y*y <= 1` mask.
Small radius changes move boundary pixels across that mask.
The resulting score error is about 0.01, not a small floating-point residual.

The isolated comparison keeps the original 400-by-400 frame and pixel coordinates.
It removes other labels and maps the selected object to label 1.
The reference script runs the actual `MeasureObjectSizeShape` module in both contexts.
It also observes the real minimum-position calls without changing their return values.

| Scope / ID  | Full-field radius | Isolated radius    | Included pixels, full / isolated | Maximum native error against full CP |
| ----------- | ----------------- | ------------------ | -------------------------------- | ------------------------------------ |
| Nuclei / 6  | 5.439837932759932 | 5.4398379327599296 | 84 / 83                          | 0.010756678912417182                 |
| Nuclei / 34 | 9.806807298389982 | 9.80680729838999   | 255 / 258                        | 0.0099292420730007491                |
| Cells / 6   | 5.439837932759932 | 5.4398379327599296 | 84 / 83                          | 0.010756678912417182                 |

For these three objects, full-field CP selects the second tied vertex. Isolated CP selects the first.
A fixed tie rule is not a general solution: full-field calls also select other tie ranks.
Across all full-field circle calls, Nuclei selects tie ranks 0, 1, and 2. Cells selects ranks 0 and 1.
Native computes each object's circle independently and matches the isolated reference for these objects.

Matching the full-field oracle requires its context-dependent vertex selection, or an agreed upstream change to the numerical contract.
A fixed radius adjustment, mask epsilon, or tolerance increase cannot reproduce that contract reliably.
This investigation did not port NumPy's unstable sort or add field-specific rules.
The full-field mismatch remains unresolved.

## Full-label sweep and bounded contract

A fresh run of the actual CP module measured all 101 IDs in each unchanged
400-by-400 label image. At `rtol=1e-10`, `atol=1e-10`, native output differed
for 90 of the 6,060 Zernike values. Each of Nuclei 6, Nuclei 34, and Cells 6
had 30 mismatches. The other 199 object records matched. The fresh CP values
for the three recorded objects exactly matched `oracle.json`. Both input array
hashes matched the checked-in hashes. This result applies to this field only.

The global SciPy calls found 8 tied minimum-angle decisions for Nuclei and
3 for Cells. NumPy's default global `argsort` selected the second tied vertex
for each failing object:

- Nuclei 6: position 49 before 44, among 687 candidates.
- Nuclei 34: position 392 before 388, among 1,190 candidates.
- Cells 6: position 46 before 41, among 528 candidates.

Other tied decisions chose ranks 0, 1, and 2 for Nuclei and ranks 0 and 1
for Cells. Thus, a fixed first/last-tie rule cannot reproduce these decisions.

The bounded claim is isolated-object parity for the three recorded objects
in their original image frame. The 199 other object records passed on this
field, but that does not establish parity for other fields. Exact general
full-field parity requires the multi-object iteration and NumPy 1.23.5's
unstable global `minimum_position` sort, including its equal-value ordering.
Native computes circles per object. Changing its local tie rule cannot
reproduce that context-dependent contract. The strict full-field check below
remains a failing diagnostic.

## Original 14-case pilot replay

The current native pipeline freshly reran all 14 original BBBC007 cases in
`zernike-recheck-20261119`. The strict `compare_pilot` comparison used the
pinned original CP outputs (`run-strict/oracle`); the newly prepared input
manifest was byte-identical to the original. This was a new native segmentation
and measurement run, not a new reference run. The native-only runner reports
`native_only` with zero comparisons by design; **only the subsequent strict
comparison** establishes the counts below. All 5,160,000 illumination pixels
and 20,640,000 array pixels passed.
Of 308,591 end-to-end feature scalars, 60 feature columns mismatched;
the same 60 columns mismatched when measuring the pinned CP inputs.
All 60 belong to `f9620-20P1-POS0002`: 30 Nuclei and 30 Cells
`AreaShape_Zernike_*` columns. No non-Zernike columns mismatched.

A separate fresh run of the real CP module on both unchanged full-label
arrays reproduced all 6,060 original pilot Zernike scores (30 columns × 101
IDs × two scopes). Native missed 90 individual scores: all 30 columns for
Nuclei IDs 6 and 34 and Cells ID 6. Thus the pilot's 60 mismatched *columns*
and the full-label sweep's 90 mismatched *scalars* count different things.
Do not report the pilot as passing. No general fix was justified: reproducing
this version-pinned contract needs simultaneous multi-object circle iteration
and NumPy's unstable global argsort ordering, not a local geometry patch.

## Reference environment

- CellProfiler 4.2.8, source checkout `2df090ff281cff80943fe679a1c3192fbda9cd6d`
- centrosome 1.2.3
- NumPy 1.23.5
- SciPy 1.9.3
- scikit-image 0.21.0

Every recorded full-field score was compared with the independently generated pilot oracle.
The comparison used the existing size/shape tolerances: `rtol=1e-10`, `atol=1e-10`.
A second real-CP execution reproduced every recorded score, circle, and tie decision exactly.

## Reproduce

From the repository root, reproduce the checked-in oracle evidence:

```sh
PYTHONPATH=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cellprofiler \
  /Users/buntend/.hermes/cache/scratch/cp-anywhere-reference-clean/venv/bin/python \
  scripts/zernike/context_evidence.py
```

Expected result: exact evidence reproduction, exit status 0.
This command requires the reference environment, not the normal native environment.

Run the strict native comparison:

```sh
uv run --frozen python scripts/zernike/context_evidence.py --native-check
```

Current native result: zero mismatches, exit status **0**.
No result is excluded, and the tolerances remain `1e-10`.

Run the passing integrity and isolated-parity checks:

```sh
uv run --frozen pytest tests/test_zernike_context.py tests/test_size_shape.py -q --no-cov
```

Verified result: 49 passed, including two full-field checks.
No test is skipped or marked as an expected failure.

To regenerate evidence from the original pilot outputs:

```sh
PILOT=/Users/buntend/.hermes/cache/scratch/cp-anywhere-production-pilot/run-full
GROUP=f9620--f9620-20P1-POS0002
FIELD=f9620-20P1-POS0002
PYTHONPATH=/Users/buntend/.hermes/cache/scratch/cp-anywhere-cellprofiler \
  /Users/buntend/.hermes/cache/scratch/cp-anywhere-reference-clean/venv/bin/python \
  scripts/zernike/context_evidence.py \
  --source-npz "$PILOT/native/$GROUP/$FIELD.npz" \
  --source-oracle "$PILOT/oracle/$GROUP/$FIELD.json" --write
```

The NPZ container uses compression. Array hashes, not ZIP metadata, identify the preserved labels.
