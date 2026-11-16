# CellProfiler pipeline (`.cppipe`) import

`cp_anywhere.load_pipeline(path)` reads a real CellProfiler `.cppipe`
file and translates the settings of every module this project
implements (`IdentifyPrimaryObjects`, `IdentifySecondaryObjects`,
`IdentifyTertiaryObjects`,
`MeasureImageIntensity`, `MeasureObjectIntensity`, `MeasureObjectSizeShape`,
`MeasureGranularity`, `MeasureObjectIntensityDistribution`,
`MeasureTexture`, `MeasureColocalization`,
`MeasureObjectNeighbors`) into `cp_anywhere`'s own configuration -- an
existing pipeline's settings for those modules carry over 1-for-1,
with no manual re-entry.

```python
import cp_anywhere

pipeline = cp_anywhere.load_pipeline("my_pipeline.cppipe")
segmentations, derived_objects, measurement = pipeline.run(
    {"DNA": dna_image, "Actin": actin_image}
)
```

The second result contains both secondary and tertiary object sets.
For example, `derived_objects["Cytoplasm"].labels` contains the tertiary
labels. The return value remains a three-item tuple.

This is a **settings importer, not a full pipeline execution engine**:
file-matching/organization modules (`Images`, `Metadata`,
`NamesAndTypes`, `Groups`, ...) are recognized and skipped --
`pipeline.skipped_modules` lists them -- not executed; you supply
already-loaded image arrays by name, same as `cp_anywhere.measure`.

Every out-of-scope setting (an older pipeline revision, "basic mode",
a non-`"Adjacent"` `MeasureObjectNeighbors` distance method, an
unimplemented `IdentifySecondaryObjects` method, and a handful more)
raises `cp_anywhere.CppipeError` naming the exact setting, rather than
silently mis-translating it -- see `spec/cppipe-compatibility.md` for
the full settings-mapping table, every explicit gap, and how it is
verified (real, oracle-generated `.cppipe` fixtures in
`fixtures/cppipe/`, checked in `tests/cppipe/test_cppipe_import.py`
against the pinned CellProfiler 4.2.8 oracle's own segmentation output
on the same test image). The CLI also accepts a pipeline directly:

```sh
cp-anywhere --pipeline my_pipeline.cppipe --image DNA=dna.tif --output result.json
```
