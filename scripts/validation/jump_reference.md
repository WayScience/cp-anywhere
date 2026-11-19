# CPJUMP1 reference pilot

CellProfiler 4.2.8 ran all 23 enabled scientific blocks on plate `BR00118045`,
well `A01`, sites `1` and `2`, at the original 1080 × 1080 resolution. The two
fields came from the plate's published `load_data_with_illum.csv` (3,456 rows).
Each used eight original TIFFs and the same eight published plate maps. See
`output/provenance.json` for source URLs, pipeline hashes, and package versions.

| Site | Nuclei | Cells | Cytoplasm | First cell `Intensity_MeanIntensity_DNA` |
| ---- | -----: | ----: | --------: | ---------------------------------------: |
| 1    |     96 |    96 |        96 |                      0.01804063868325423 |
| 2    |    132 |   132 |       132 |                     0.014390638865235587 |

## Run

From the repository root, use the existing pinned reference environment. Choose
a fresh scratch path for a rerun; the script does not overwrite field results.

```bash
scratch="$HOME/.hermes/cache/scratch"
JAVA_HOME="$(/usr/libexec/java_home)" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$scratch/cp-anywhere-cellprofiler" \
"$scratch/cp-anywhere-reference-clean/venv/bin/python" \
  -m scripts.validation.jump_reference "$scratch/jump-reference-rerun"
```

The completed run is under `~/.hermes/cache/scratch/jump-reference/`:

- `original.cppipe`, `normalized.cppipe`, `resave.diff`: source, CP 4.2.8
  serialization, and text diff. CP reload preserved all 33 blocks and all
  loaded setting values and enabled states.
- `input/`: original bytes. The pipeline's `LoadData` rescale setting is
  honored through CP's `FileImage` decoder. The TIFFs have `uint16` pixels
  and CP scale 65,535; the maps have `float32` pixels and CP scale 1.
- `output/BR00118045_A01_{1,2}/`: CP's raw `measurements.h5`, nine raw label
  planes, six relationship tables, and JSON manifests. `schema.json` has the
  declared columns by block and the realized names, shapes, dtypes and
  hashes. `result.json` has exact label IDs and hashes at each producer.
- `output/readback.json`: read-back check of all 18 label planes, both HDF5
  files, and all 12 relationship tables. Each field has 7,182 realized
  measurement keys: 1,387 Image, 1,932 Nuclei, 1,939 Cells, and 1,924 Cytoplasm.

## Scope and blockers

CP upgraded `DateRevision` from 406 to 428. It upgraded `IdentifyPrimaryObjects`
from revision 14 to 15 and both `SaveImages` blocks from 15 to 16. In the
`IdentifySecondaryObjects` block, it kept revision 10 and upgraded the embedded
threshold settings from 11 to 12. This repairs the old setting layout but
leaves the segmentation method `Watershed - Image` unchanged. The initial
preflight rejected that method. The current cp-anywhere executor supports it.

The caller supplies input and owns output and UI. The script does not invoke
blocks 1 or 28–32, or the disabled blocks 11–13 and 33. Block 14 stays in its
published `Each` mode, so no plate-wide estimate is made from the two fields.
Each field has its own CP workspace and local image number 1; join by the
published plate, well, and site, not that local image number.

On the installed scikit-image, block 20 fails on the removed `selem` keyword.
The script uses the repository's existing keyword-only compatibility adapter
(`selem` → `footprint`). It forwards the same structuring element to the
installed implementation. It changes no pipeline setting, dependency, or
parity tolerance. Raw nonfinite features remain in the HDF5 output.

## Compare with cp-anywhere

Build the Python extension before you run this comparison. Use the reference
interpreter to read the saved CellProfiler measurements.

```sh
scratch="$HOME/.hermes/cache/scratch"
uv run --no-sync python -m scripts.validation.compare_jump compare \
  --root "$scratch/jump-reference" \
  --reference-python "$scratch/cp-anywhere-reference-clean/venv/bin/python" \
  --report "$scratch/jump-reference/comparison.json"
```

All six exposed label planes matched exactly in each field. The object IDs
matched in all three compartments. The comparison does not count the three
label planes per field that cp-anywhere does not expose. It does not compare
identification-module measurements, saved files, or relationship tables.

The exact names of all 7,154 requested measurement columns matched in both
fields, with no extras. After the float32 precision fixes, the full comparison
passed on both fields. It checked 556,064 values on site 1 and 764,072 values
on site 2. No value was outside the existing tolerances. The comparison
returned exit status 0 and saved `comparison-all-after-precision.json`.

The table shows the previous mismatch counts. Each count is the number of
columns with one or more values outside the same tolerances.

| Site | Values compared | Correlation | Granularity | Radial intensity | Texture |
| ---- | --------------: | ----------: | ----------: | ---------------: | ------: |
| 1    |         556,064 |         248 |         510 |              288 |       0 |
| 2    |         764,072 |         249 |         512 |              288 |       0 |

The corrected calculations keep float32 precision at the affected stages.
This includes texture byte scaling, granularity, radial distribution, and
per-object colocalization products and thresholds. These results cover only
the two tested CPJUMP1 fields and the requested measurement blocks. They do
not establish parity for every `.cppipe` module or a full plate.

Before the texture fix, 1,777 columns at site 1 and 2,493 at site 2 were
outside tolerance. At site 1, all eight prepared float32 planes matched
CellProfiler exactly. Widening them before scaling to bytes changed the
rounding of 12 pixels in the brightfield channels. Keeping the multiplication
in float32 resolved the texture differences on both sites without changing
tolerances. The `float32_quantization_ties` fixture checks the same rounding
boundary against CellProfiler.

The command writes the report before returning a nonzero status for a mismatch.
This is a comparison against a pinned CellProfiler replay, not a comparison
with the published feature exports.

For a focused comparison, add `--focus granularity`, `--focus radial`,
`--focus correlation`, or `--focus texture`. Use a separate report path.
This keeps image preparation, segmentation, label checks, and object ID
checks intact, but runs only the selected measurement block. Add `--site 1`
or `--site 2` to restrict a probe to one original field. Omit both options
for the full measurement comparison.

## Group illumination

`plate_illumination_oracle.py compare` checks the unchanged CPJUMP1 illumination
pipeline. The replay used eight original TIFFs per field in three fields on
plate `BR00117010`. All eight final float32 maps matched CellProfiler exactly
(9,331,200 pixels total). This tests the group-final result, not a full plate
or an intermediate map from an earlier cycle.
