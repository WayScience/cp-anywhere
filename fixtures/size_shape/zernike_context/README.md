# Zernike full-field parity limitation

## Status

The native implementation does **not** match real CellProfiler for all Zernike measurements on this field.
No production code, scientific tolerances, or input labels changed during this investigation.
The fixture records the limitation. It does not grant a parity exemption.

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

Current result: 30 mismatches per target object, 90 total, exit status **1**.
No result is excluded from this comparison, and the tolerances remain `1e-10`.
This command must pass before anyone claims full-field parity for these targets.

Run the passing integrity and isolated-parity checks:

```sh
uv run --frozen pytest tests/test_zernike_context.py tests/test_size_shape.py -q --no-cov
```

Verified result: 47 passed. These tests do **not** establish full-field parity.
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
