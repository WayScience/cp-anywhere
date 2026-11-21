# Bounded PedCP90 QC validation

The runner compares CellProfiler 4.2.8 with cp-anywhere on the exact `Blur_and_Saturation_QCFlag` value, all ten criterion decisions, skip disposition, and whether the next stage produces its outputs. It also checks that the QC metric names match. It records numeric values separately and does not require them to be bit-identical.

The runner pins `WayScience/pedcp90@3c185d3b0fa4879bd55413d8682cbea4dfdbdff9`, `2.feature_extraction/analysis.cppipe`, by SHA-256. It preserves the original `LoadData`, `MeasureImageQuality`, and `FlagImage` blocks. The flag fails on any fluorescence channel with `PowerLogLogSlope` outside `[-2.7976730166487322, -1.1262448982947686]` or `PercentMaximal` greater than `4.275261792593619`. The five channels are DNA, AGP, ER, Mito, and RNA.

## Results

| Input                  | Field                    | Exact decision in both runtimes | Next stage                                   |
| ---------------------- | ------------------------ | ------------------------------- | -------------------------------------------- |
| Original PedCP90       | `BR00143976 / C03 / 3`   | `0`, kept                       | Original correction produced all six outputs |
| Public CPJUMP1 control | `BR00118045 / A01 / 1–7` | `0`, kept                       | Probe ran                                    |
| Public CPJUMP1 control | `BR00118045 / A01 / 8`   | `1`, skipped                    | Probe did not run                            |

There were zero criterion, flag, skip, or downstream-execution differences across the nine fields. Both runtimes flagged the Mito blur criterion in the rejected control: CellProfiler measured `-1.071378424391396`, and cp-anywhere measured `-1.0713784070615722`. The control uses an `ImageMath` copy probe after the unchanged flag. It does not run PedCP90 illumination correction.

### Input provenance

- **PedCP90:** `cytomining/CytoDataFrame@e3e39626df5b49081df41ef53b488393611f9005`, `tests/data/cytotable/pediatric_cancer_atlas_profiling/images/orig`. All six public TIFFs match the file MD5 values in the pinned PedCP90 `1.illumination_correction/img_quality_control/whole_img_qc_output/BR00143976/Image.csv`. DNA, AGP, ER, Mito, RNA, and brightfield all came from the original 1080 × 1080 TIFFs. The runner also used the six maps from the pinned project, `1.illumination_correction/illum_directory/Round_1_data/BR00143976/`, and the plate CSV in `2.feature_extraction/loaddata_csvs/Round_1_data/`.
- **CPJUMP1:** Cell Painting Gallery `cpg0000-jump-pilot/source_4`, batch `2020_11_04_CPJUMP1`, plate `BR00118045`. The plate `load_data_with_illum.csv` identifies the original objects. All 40 TIFFs match the file MD5 values in the corresponding public per-site `Image.csv` exports under `workspace/analysis/2020_11_04_CPJUMP1/BR00118045/analysis/`.

**Nominal acceptance remains open.** The one public original only covers a retained PedCP90 field. The rejected CPJUMP1 control does not replace a rejected original PedCP90 field or validate either side of the original thresholds on the nominated sample. The pinned PedCP90 data README points to Alpine/PetaLibrary for the wider image set. No Alpine connection or credentials were used in this validation. For the next bounded run, stage original TIFFs and plate maps for passing and rejected PedCP90 fields near the cutoffs. On `BR00143976`, the published QC table identifies `H15 / 2` and `E17 / 6` on opposite sides of the RNA blur floor, and `H12 / 5` and `E08 / 7` on opposite sides of the Mito saturation ceiling. These are candidate fields, not validated decisions.

## Run from the repository root

Stage the original files by manifest basename in one local directory per plate. Use the pinned CellProfiler environment from `reference/install.sh` and a cp-anywhere environment with the native extension built. Repeat `--field` for up to eight distinct fields. Choose a new run directory for each run.

```sh
# Set RUN, PIPELINE, MANIFEST, IMAGE_ROOT, PUBLISHED_QC, CP_SOURCE, and CP_PYTHON first.
.venv/bin/python -m scripts.validation.pedcp90_qc stage \
  --run "$RUN" --pipeline "$PIPELINE" --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" --published-qc "$PUBLISHED_QC" \
  --field BR00143976 C03 3 --scope pedcp90-original
PYTHONPATH="$CP_SOURCE" "$CP_PYTHON" \
  -m scripts.validation.pedcp90_qc_reference --run "$RUN"
.venv/bin/python -m scripts.validation.pedcp90_qc compare --run "$RUN"
```

For a CPJUMP1 control, use `--scope cpjump1-control` and a CPJUMP1 plate manifest. If the per-site digests are separate, combine their rows in one CSV for `--published-qc`. The digest table is optional; without it, the runner records SHA-256 but does not independently verify original-file identity. Use `--provenance` to record any additional source locations.

`inputs.json` records the pin, file provenance, and hashes. The reference script uses the real CP `LoadData` decoder and archives the rescaled arrays and their scale metadata. The comparison replays those exact arrays in cp-anywhere to isolate QC and skip parity from image decoding. It writes `comparison.json` and exits nonzero on any decision, criterion, or QC metric-name mismatch.

Completed evidence is under `~/.hermes/cache/scratch/pedcp90-qc/`: `final-original/` and `final-controls/`. These are bounded QC runs, not full profiling, browser-parity, or corrected-pixel comparisons.
