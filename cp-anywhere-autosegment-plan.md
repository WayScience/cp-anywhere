# CP-Anywhere AutoSegment: implementation plan

**Status:** Proposed MVP  
**Goal:** Automatically select and configure a reproducible segmentation pipeline for single-cell morphological profiling, with an initial image-quality gate, low-cost execution, and literature-backed visual explanations.  
**Primary scope:** 2D fluorescence microscopy, especially Cell Painting.  
**Principle:** CP-Anywhere (CPA) owns the complete segmentation pipeline, including model inference, mask decoding, postprocessing, compartment relationships, measurements, and reporting.

## 1. Summary

AutoSegment is a small addition to CP-Anywhere with **five ordered stages (0–4)**:

0. **Gate image quality:** Check basic acquisition and image quality before segmentation. Mark images `pass`, `review`, or `fail` using literature-grounded measurements and acquisition-calibrated thresholds. Preserve all images and QC results; only accepted images feed automatic pipeline selection.
1. **Select a pipeline:** Compare a small set of candidate nuclei/whole-cell segmentation pipelines on representative images with reasonable initial settings. Shortlist up to two candidates.
2. **Optimize the pipeline:** Tune a limited number of parameters for each shortlisted candidate, check performance on held-out images, and select the fastest configuration that meets quality requirements.
3. **Apply the selected pipeline:** Generate masks and measurements using the selected, versioned configuration. Preserve all QC outcomes and exclude failed inputs according to the Stage 0 policy.
4. **Generate report; optionally review:** Always generate the literature-backed visual report and complete automated outputs. Human judgment is optional and **non-blocking by default**; only an explicitly enabled approval gate holds downstream release.

Generate a self-contained **HTML report** with image overlays, measured quality indicators, parameter comparisons, a table explaining segmentation techniques, and clickable scientific references. Save machine-readable results so the decision can be reproduced.

**MVP restraint:** No custom-trained segmentation model, no required GPU, no learned selector, no separate orchestration service, and no requirement for Python at runtime. CPU support is required; browser/WASM inference should be tested rather than assumed.

## 2. Architecture and ownership

All stages are initiated and controlled through CPA:

```text
Images + channel mapping + acquisition metadata + optional annotations/replicates
                         |
                  CPA AutoSegment
                         |
       Stage 0: image quality gate
             /           |          \
           pass        review        fail
             |          |             |
             |     override only     log reasons
             |          |             |
             +----------+       exclude from search
             |
          Representative image sampling
                         |
       Stage 1: compare pipeline candidates
                         |
         Shortlist up to two pipelines
                         |
       Stage 2: parameter optimization
                         |
        Validation on held-out images
                         |
       Stage 3: apply selected pipeline
                         |
   Selected pipeline + masks + measurements
                         |
     Stage 4: report + human judgment
         /                        \
   default: continue       approval mode: hold
         |                        |
    final artifacts       accept / revise / reject
         |                        |
         +----------> report + JSON + Parquet + provenance
```

A *pipeline* is a compatible set of operations, not a single interchangeable algorithm:

```text
Preprocess image
  -> identify nuclei (classical or StarDist)
  -> identify whole cells (seeded watershed / propagation)
  -> derive cytoplasm = cell mask minus matched nucleus mask
  -> validate object relationships
  -> extract morphology measurements
```

Implement these interfaces inside CPA's Rust core:

- `SegmentationMethod`: accepts an image and settings; returns a labeled instance mask and run metadata.
- `Pipeline`: combines nuclei, cells, cytoplasm, and preprocessing/postprocessing stages.
- `QualityEvaluator`: measures segmentation plausibility and, when supplied, annotation agreement and profiling usefulness.
- `AutoSelector`: runs the two-stage search with a bounded time/trial budget.
- `ReportWriter`: generates HTML plus structured results.
- `ReviewController`: records optional human decisions and enforces an approval gate **only** when explicitly enabled.

Add an independently usable `ImageQualityEvaluator` that measures image-level acquisition quality **without any cell segmentation**, applies a versioned QC policy, and returns accepted/flagged images plus reasons. AutoSegment always invokes this evaluator before representative-image sampling.

The same normalized `u32` instance-mask representation, spatial coordinates, compartment IDs, and provenance schema should apply to every method. A pretrained model may be packaged as a pinned asset, but **CPA runs it locally and implements its mask decoding**. Do not rely on an external Python segmentation service.

## 3. Small initial algorithm set

**Implement only two nuclei methods and one shared whole-cell method in the MVP.** This yields two full candidate pipelines:

| Candidate | Nuclei stage | Whole-cell stage | Cytoplasm | Execution |
|---|---|---|---|---|
| A: Classical | Threshold + distance-transform / marker watershed | Seeded watershed / CellProfiler-style propagation | Cell minus nucleus | Native Rust, CPU/WASM |
| B: Learned | Pretrained 2D StarDist fluorescence nuclei model | Same seeded watershed / propagation | Cell minus nucleus | CPA-managed model inference + Rust mask decoding |

### Implementation sequence

1. **Classical nuclei:** Otsu thresholding, optional smoothing/illumination handling, size filtering, declumping using distance-transform watershed.
2. **Shared secondary objects:** Nuclear-seed-based whole-cell watershed/propagation constrained by cell-channel foreground; stable nucleus-to-cell assignment.
3. **StarDist nuclei:** Bundle a known, license-compatible pretrained 2D fluorescence model; run inference from CPA, decode polygon outputs, and apply non-maximum suppression in Rust. Validate model export, ONNX operator support, CPU performance, and WASM viability before promising browser support.
4. **Cytoplasm:** Derive non-overlapping cytoplasm masks after validating matched parent nucleus/cell identifiers.

The classical and learned candidates differ only in the nuclei stage at first. This helps isolate which step improves the resulting morphology measurements. If StarDist integration proves unexpectedly difficult, ship the classical pipeline and reporting first, while keeping the second candidate behind a feature flag.

### Literature-aware technique catalog (shown in every report)

Distinguish **implemented**, **evaluated in this run**, and **background-only** methods. The table below communicates published design strengths, not universal rankings.

| Technique | Common strengths / suitable context | Limitations / reasons it may fail | MVP status | References |
|---|---|---|---|---|
| Otsu threshold + watershed | Fast, interpretable; useful for clear fluorescent nuclei and separation by intensity/shape | Uneven staining, overlapping/touching objects, irregular boundaries | Implement and evaluate | [Otsu (1979)](https://doi.org/10.1109/TSMC.1979.4310076); [CellProfiler IdentifyPrimaryObjects](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifyPrimaryObjects.html) |
| Seeded watershed / propagation | Whole-cell boundaries informed by nucleus seeds and a cell-associated channel; natural Cell Painting workflow | Requires reasonably correct nuclei and informative cell boundaries; may merge or truncate cells | Implement as common stage | [CellProfiler IdentifySecondaryObjects](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifySecondaryObjects.html); [Carpenter et al. (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1794559/) |
| StarDist | Separates crowded, approximately star-convex fluorescent nuclei | Shape representation less suitable for strongly branching/irregular objects; model domain shift | Implement and evaluate | [Schmidt et al. (2018)](https://arxiv.org/abs/1806.03535); [StarDist project](https://stardist.net/) |
| InstanSeg | Efficient pretrained segmentation of cells/nuclei in supported microscopy settings | Model domain and deployment constraints need validation | Catalog only; future | [InstanSeg publication/preprint](https://arxiv.org/abs/2408.15954) |
| Cellpose | Generalist learned instance segmentation across diverse cells/images | Heavier inference/runtime; performance depends on model/version and data | Catalog only; future | [Stringer et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/33318659/) |
| Omnipose | Particularly suited to elongated or branched bacterial shapes | Less directly aligned with typical mammalian Cell Painting; additional complexity | Catalog only; future | [Cutler et al. (2022)](https://www.nature.com/articles/s41592-022-01639-4) |

Each catalog entry should include: algorithm family; target compartment; assumptions; typical failure modes; supported channels/modalities; adjustable settings; local implementation support; source DOI or persistent URL; and the version/date reviewed. Literature claims must be clearly separate from locally observed results.

## 4. Stage 0: image quality gate (required)

**Purpose:** Decide whether microscopy images meet a minimum *technical image-quality* standard before comparing segmentation pipelines. **This is not a cell segmentation or biological phenotype classifier.** Run it as a native Rust CPU operation; optionally expose it independently as `cpa.quality.assess(...)`. It is the first step of `cpa.segmentation.auto(...)` by default.

### Initial QC measurements

| Check | Suggested inexpensive measurement (without masks) | Interpretation / failure mode | Scientific basis |
|---|---|---|---|
| Focus / blur | Power-spectrum slope (as in high-content QC), optionally Tenengrad gradient energy | Low-focus or blurred acquisition; requires channel-aware calibration | [Bray et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/) |
| Saturation / clipping | Fraction of pixels at detector maximum or specified high-intensity clipping threshold, by channel | Lost intensity information from overexposure | [Caicedo et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6871000/); [CellProfiler MeasureImageQuality](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html) |
| Signal / intensity | Per-channel intensity quantiles, robust dispersion, background estimate, fraction of near-zero pixels | Empty or unusually dim channels, unexpected background / exposure | [CellProfiler MeasureImageQuality](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html); [Bray et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/) |
| Illumination uniformity (later) | Low-frequency background variation or corner-to-center intensity trends | Spatial shading or nonuniform illumination | [Bray et al., 2016, Cell Painting protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC5223290/) |
| Artifacts (later) | Extremely bright contiguous regions, suspicious local texture / intensity anomalies; optional small supervised classifier | Debris, bubbles, unusual staining or imaging defects | [Bray et al., 2018, machine-learning QC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6112602/) |

**MVP:** Implement only focus, saturation/clipping, and robust per-channel intensity statistics. Include structured input checks: decoding, expected channels, shape/dimensions, bit depth, and finite/valid values. Save the other metrics as future extensions rather than requiring a trained QC classifier.

### Quality policy and threshold calibration

- Use a **versioned, configurable QC profile** (for example, `cell-painting`) with documented metric definitions and channel requirements. The profile provides *candidate checks*, not claimed universal biological standards.
- **Hard fail:** Corrupt/unreadable input, missing required channel, invalid shape, or acquisition data demonstrably unusable under explicit detector settings.
- **Calibrated fail:** A focus, clipping, or other metric crosses an experimentally validated threshold with adequate evidence. Determine thresholds from reviewed examples or trusted control images collected under comparable settings; do not hard-code literature-wide numeric cutoffs.
- **Review:** Outlier or uncertain result, missing calibration, sparse biological signal, or conflicting metrics. Unusual cells or treatment effects must not be automatically rejected solely for looking unlike controls.
- **Pass:** Required checks are valid and calibrated limits pass. If calibration is unavailable, use a conservative policy that flags ambiguous cases rather than manufacturing a confident pass/fail.
- Compare within **acquisition-compatible groups** (channel, objective, exposure, imaging batch/plate where relevant). Avoid learning cutoffs from a plate so severely degraded that its distribution becomes the baseline.
- Preserve `pass`, `review`, and `fail` inputs and metric values in output; retain original image IDs and reasons. Review requires a recorded manual override to enter selection. Failed images never enter automatic selection unless explicitly overridden, with provenance.
- Exclude failed images from the **segmentation-method search**, not silently from raw data or downstream provenance. If too few accepted images remain, return `review_required` and do not run Stage 1.

### Image-quality reporting

The first report section shows image counts and proportions by status, thresholds with the source of calibration, per-channel focus/saturation/intensity distributions, acquisition-group comparisons, and raw thumbnails of **actual flagged images**. Each flag links to the image, relevant measured values, check definition, threshold, and source reference. Include manual review decisions and overrides. Distinguish technical acquisition issues from plausible biological differences.

**Validation:** Compare gate decisions against blinded human review on diverse plates/channels; measure false rejection of biologically interesting images, sensitivity to known blur/clipping, and CPU execution cost. Stress-test low-cell-density wells and treatment-induced morphology differences.

## 5. Stage 1: pipeline selection

**Inputs:** Images marked `pass` by Stage 0 (plus approved `review` images), explicit channel map (e.g., DNA/nuclear and cell-boundary/whole-cell stain), optional plate/well metadata and annotations, CPU and runtime budget.

1. Stratify a *small* sample of **Stage 0 accepted images** across wells, plates, intensity, focus, and cell-density ranges. Start with 10–20 images for engineering tests; calibrate sample size experimentally, and reserve separate images for validation.
2. Run pipeline A and B using documented, reasonable defaults and a small coarse variation of settings so one method is not unfairly rejected due to a poor default.
3. Record all masks, paired source-image crops, image characteristics, QC measurements, runtime, and failures.
4. Reject candidates failing hard constraints, such as invalid labels, broken parent-child relationships, unreasonable empty/near-total coverage, or a user-defined maximum runtime. Treat biological plausibility as a warning or configurable constraint, not universal ground truth.
5. Shortlist up to two plausible pipelines. If none passes, **abstain and request review** rather than declare a winner.

**Important:** An unlabeled image alone does not establish true segmentation accuracy. Reference-free QC is evidence about plausibility and consistency, not proof of correct boundaries.

## 6. Stage 2: parameter optimization

Perform a bounded **random search** (initially, around 10–20 trials per shortlisted candidate, adjustable) using a reproducible seed. Do not add a tuning framework as a core dependency.

| Stage | Initial tunable parameters |
|---|---|
| Classical nuclei | Smoothing radius, threshold correction, minimum object area, minimum peak/seed separation, declumping parameters |
| StarDist nuclei | Detection probability threshold, polygon NMS threshold; optionally normalization settings if supported |
| Shared whole-cell stage | Foreground threshold, propagation regularization, maximum expansion / size constraints |
| Postprocessing | Border-object policy, minimum cell/nucleus area, parent-child consistency rules |

- Use sensible, method-specific ranges, not a large Cartesian grid.
- Cache reusable preprocessing and nucleus masks when tuning only downstream settings.
- Avoid selecting based on one unusually easy image or well. Aggregate across the stratified sample.
- Re-run the best settings for each candidate on **held-out** images, ideally from unseen wells or plates when available.
- Select a pipeline only after the held-out comparison; if quality is statistically or practically indistinguishable, prefer the faster/simpler candidate.
- Return the complete configuration and a clear account of how the selected method compared with alternatives.

For future work, consider sequential optimization or a small tabular predictor trained from benchmark outcomes. Neither belongs in the first release.

## 7. Evaluation and decision policy

Use a small number of **interpretable measurements**. Do not invent fixed literature-backed weights or equate a high proxy score with biological correctness.

| Evaluation level | Metrics / checks | Purpose |
|---|---|---|
| Object plausibility | Counts, foreground fraction, areas, eccentricity, border-touching fraction, unexpectedly small/large objects | Detect gross failures and differences in mask populations |
| Compartment consistency | Fraction of cells with exactly one matched nucleus, nucleus containment, invalid overlaps, cytoplasm non-overlap | Protect biologically meaningful compartment measurements |
| Image agreement | Boundary-to-image-gradient correspondence, foreground intensity, threshold sensitivity | Flag masks poorly supported by available stain channels |
| Stability | Consistency under *modest* image perturbations; variation in derived morphology features | Flag fragile segmentation choices; stability alone does not prove accuracy |
| Efficiency | CPU wall time, peak memory, images or megapixels per second | Support low-resource decisions |
| Optional annotated evaluation | Instance-level precision/recall, matching at specified IoU thresholds, boundary agreement | Check segmentation accuracy when expert labels exist |
| Optional profiling evaluation | Morphology feature reproducibility, percent replicating, percent matching (when suitable experimental replicates and annotations exist) | Test whether segmentation supports useful downstream biology |

**Selection policy for MVP:**

1. Apply explicit validity and minimum-quality constraints.
2. Compare the *individual* QC metrics and held-out performance; use a simple documented ranking/tie-breaking policy rather than an unsupported universal composite score.
3. When labels are available, prioritize annotated object accuracy. When replicate experimental designs are available, present downstream profiling outcomes separately as stronger biological evidence.
4. Choose the fastest candidate that meets required quality criteria. Otherwise return `review_required` with supporting evidence.

Report sampling design and uncertainty. Do not overinterpret replicate-based outcomes when sample sizes are too small or when experimental metadata is unavailable.

## 8. Stage 3: apply the selected pipeline

Once Stage 2 identifies an acceptable configuration, CPA runs the chosen pipeline on the eligible images and produces labeled nuclei, cell, and cytoplasm masks plus morphological measurements. Preserve input IDs and Stage 0 exclusion reasons. Save a reproducible segmentation configuration before execution, and retain the original algorithm and parameter results even if the researcher later overrides the recommendation.

Stage 3 is automated. A human reviewer is **not** required to start or finish segmentation in the default mode.

## 9. Stage 4: visual report and human judgment

Stage 4 **always generates its report**, regardless of whether a reviewer participates. Report generation is automatic; human judgment is a separate, **non-blocking by default**, decision-support action. The automated recommendation and Stage 3 results are complete before optional human review. Users can inspect evidence and accept the recommendation, select another evaluated pipeline or configuration (with an explicit rerun), or reject the result. A human decision is tracked as separate provenance rather than silently rewriting the original automated decision.

**Review behavior:**

- `review_mode="optional"` (**default**): always write the report and finalize results immediately; record review status as `not_reviewed` until a human chooses to act. No waiting, approval prompt, or required manual click.
- `review_mode="required"` (**explicit opt-in**): write the report and provisional masks, but mark the run `awaiting_approval` and withhold **final acceptance/publication/export to downstream production workflows** until a reviewer accepts or overrides it. Never discard or hide the provisional artifacts.
- `review_mode="disabled"`: still generate the same report and provenance, but omit human-decision controls and any expectation of review.
- Approval actions are `accept`, `override` (selected existing candidate/configuration; rerun where necessary), and `reject`. Record reviewer identity if available, timestamp, reason, selected configuration, and links to the previous automated choice. Unsupported choices require a new evaluation, not a false claim they were benchmarked.
- Distinguish **Stage 0 image-quality flags** from **Stage 4 human approval**. Stage 0 still excludes known unusable images from automatic selection; Stage 4 cannot silently alter those exclusions.

### Visual and literature-backed HTML report

Generate a **self-contained, offline-readable HTML file**, plus structured data. The report is part of the output contract, not an optional debugging view.

### Report sections

1. **Image-quality gate:** pass/review/fail totals, per-channel focus, saturation and intensity distributions, cutoff calibration, actual flagged-image gallery, and recorded overrides; link to published QC methods.
2. **Decision summary:** selected pipeline (or review required), settings, CPU cost, sample coverage, and stated decision rules.
3. **Technique comparison table:** the catalog above, literature-based strengths/failure modes, references, support status, and whether a method was actually run.
4. **Same-image overlays:** raw image, nuclei outlines, whole-cell outlines, and cytoplasm on **identical crops** for each candidate, with controls to inspect alternate examples.
5. **Object distributions:** area, shape, counts by well/image, and nucleus-cell assignment statistics.
6. **QC and performance:** per-candidate metrics, failure counts, CPU runtime, and optional memory usage; indicate held-out versus tuning images.
7. **Parameter search:** trials and settings table; compact response plots or heatmaps when enough data exist; clearly mark selected parameters.
8. **Failure gallery:** representative merged, split, missing, border-truncated, or inconsistent objects, linked to the source image.
9. **Evidence and limitations:** method-selection rationale distinguishing literature expectations, locally measured observations, and remaining uncertainty.
10. **Reproducibility:** CPA version, OS/CPU, image/channel metadata, method/model version and checksum, random seed, settings, metric definitions, and source references.
11. **Human review:** optional accept/override/reject controls, review status, decision history, and whether approval is required before downstream release. In default mode, no action is needed.

**Avoid misleading visuals:** Use the same raw pixels and crop coordinates for direct comparison. Do not use stock or separately sourced microscopy pictures in generated decision reports.

### Artifacts

```text
output/
  image-quality.json         # pass/review/fail, thresholds, overrides, metric definitions
  segmentation-config.json   # selected pipeline, settings, provenance
  methods.json               # versioned literature/technique catalog
  image-quality.parquet      # per-image, per-channel presegmentation QC
  evaluation.parquet         # per-image, per-method, per-trial measurements
  report.html                # fully self-contained comparative report + review controls
  review.json                # review mode, status, optional human decisions, audit trail
  masks/                     # native CPA-supported labeled-mask storage
```

For `report.html`, prefer a small, embedded static visualization implementation with no network calls. Preserve exact links to paper DOIs/URLs as plain hyperlinks; the rest of the report should remain readable offline.

## 10. Proposed public API (illustrative)

```python
import cp_anywhere as cpa

qc = cpa.quality.assess(
    images,
    channels={"nuclei": "DNA", "cells": "AGP"},
    profile="cell-painting",
)
qc.report("output/image-quality.html")

# AutoSegment also runs Stage 0 itself when given raw images.
result = cpa.segmentation.auto(
    images,
    channels={"nuclei": "DNA", "cells": "AGP"},
    pipelines=["classical", "stardist"],
    quality_profile="cell-painting",  # mandatory Stage 0
    top_k=2,
    trials=16,
    device="cpu",
    seed=42,
    review_mode="optional",  # default: never blocks automated results
)

result.save("output/segmentation-config.json")
masks = result.segment(images)  # Stage 3: selected pipeline
result.report("output/report.html")  # Stage 4: optional human judgment
# review_mode="required" holds final downstream release until approved.
```

These names describe a proposed API; they are not claims about existing CP-Anywhere functions. Expose equivalent configuration and execution through the Rust library and CLI. Python should be a binding, not a required runtime dependency. For browser use, the CPA runtime should consume the same serialized pipeline configuration.

Example CLI concept:

```bash
cpa quality assess --config experiment.json --out output/
cpa segment auto --config experiment.json --out output/  # runs quality gate first
cpa segment run --pipeline output/segmentation-config.json --out masks/
```

## 11. Implementation milestones

### M0 — Image quality gate (prerequisite)

- [ ] Define `ImageQualityEvaluator`, per-channel metric schema, QC profiles, and pass/review/fail with per-image reasons.
- [ ] Implement focus, detector saturation/clipping, and robust intensity statistics in Rust without segmentation.
- [ ] Add acquisition-group calibration, conservative review behavior, overrides, and valid/missing-channel checks.
- [ ] Emit `image-quality.json` / `image-quality.parquet` and a simple QC report with actual flagged-image thumbnails.
- [ ] Test bad-focus, clipped, empty-channel, sparse-cell, and biologically unusual images against expert review.

**Done when:** CPA can independently assess images and AutoSegment refuses to start method selection without enough accepted images; all decisions and exceptions are traceable.

### M1 — Common execution and mask contracts

- [ ] Define image channels, labeled-instance masks, parent-child IDs, and `Pipeline` interface.
- [ ] Implement classical nuclei thresholding, declumping, and shared whole-cell propagation in Rust.
- [ ] Add reproducibility metadata and unit tests for touching cells, border objects, empty images, and label integrity.

**Done when:** CPA can generate nuclei, cell, and cytoplasm masks and measure basic properties on a small 2D fluorescence test set.

### M2 — CPA-managed learned inference

- [ ] Validate and package one pretrained StarDist 2D fluorescence nuclei model with license and checksum.
- [ ] Implement the inference adapter and Rust postprocessing inside CPA.
- [ ] Test native CPU; verify WASM feasibility separately and document unsupported targets honestly.

**Done when:** Both candidate pipelines produce comparable masks through a common CPA API, without invoking an external Python segmentation process.

### M3 — Two-stage AutoSegment

- [ ] Run Stage 0 automatically and use only accepted images for stratified selection and held-out splits.
- [ ] Implement bounded comparison, shortlist, random search, and runtime accounting.
- [ ] Add validity constraints and a `review_required` outcome.
- [ ] Emit `segmentation-config.json` and `evaluation.parquet`.

**Done when:** Given a fixed seed and dataset, CPA makes a reproducible choice or abstains, and the reason can be inspected.

### M4 — Stage 3 application + Stage 4 report and optional review

- [ ] Apply the chosen versioned pipeline across accepted images and preserve masks, measurements, and excluded-image provenance.
- [ ] Add a versioned methods catalog and paper references.
- [ ] Generate presegmentation image-quality distributions and flagged-image examples, followed by same-image segmentation overlays, QC tables/distributions, failure crops, and parameter summaries.
- [ ] Separate literature statements from dataset-specific findings in the report.
- [ ] Produce a single offline-readable `report.html`.
- [ ] Add `review_mode=optional|required|disabled`; default `optional` must never block automated results.
- [ ] Record human accept/override/reject actions in `review.json`, preserving the original automatic recommendation.
- [ ] In `required` mode, keep artifacts provisional and gate only final downstream acceptance/publication until approval.

**Done when:** A researcher can understand the selected pipeline, competing outcomes, limitations, and literature without reading raw logs; unattended runs complete without review, and explicitly approval-gated runs cannot be finalized without a recorded decision.

### M5 — Scientific validation

- [ ] Compare selection against a small expert-reviewed mask set using object-level metrics.
- [ ] Test across different wells/plates and challenging cell-density or illumination conditions.
- [ ] When metadata allow, compare morphology feature stability and replicate retrieval.
- [ ] Benchmark CPU performance and memory against fixed-pipeline baselines.

**Done when:** The team can show where automated selection agrees or disagrees with independent annotations and whether the added search cost is justified by better downstream results.

## 12. Success criteria and safeguards

- Stage 0 image-quality assessment is mandatory before automatic selection, independently usable, and does not require segmentation.
- Every QC decision includes metric values, calibrated thresholds, reason codes, and override provenance; uncalibrated values do not masquerade as universal image-quality standards.
- CPA owns **all** segmentation execution, including its packaged pretrained model.
- Default installation supports a useful classical CPU workflow with no model download.
- Both pipelines produce consistent masks, compartment relationships, and metadata.
- Automated search has configurable time and trial limits and can run on lightweight CPU hardware.
- Decisions are reproducible given pinned model/runtime versions, image input, settings, and seed.
- Reports include literature comparisons, real same-image overlays, measured QC, performance, and a decision explanation.
- Stage 4 reports are always generated; human judgment is non-blocking by default. Only an explicitly enabled approval mode gates final downstream acceptance.
- Human decisions and overrides are auditable and never erase the original automated selection.
- Unavailable methods are **never** reported as benchmarked; reference-free quality metrics are **never** described as ground-truth accuracy.
- The system may decline to select a pipeline when evidence is insufficient.

## 13. Out of scope for MVP

- Training or fine-tuning new segmentation neural networks.
- Large foundation-model segmentation dependencies or mandatory GPU acceleration.
- Full multi-method catalogs implemented as executable plugins.
- 3D segmentation, organoid-specific pipelines, or time-lapse tracking.
- Automated choice of a different method for every individual cell.
- A learned meta-selector (e.g., Random Forest) before benchmark data and labels justify it.
- Guaranteeing browser model inference on all WebAssembly runtimes without compatibility testing.

## 14. Selected literature and documentation

1. Bray MA et al. (2012). *Workflow and Metrics for Image Quality Control in Large-Scale High-Content Screens*. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/). Basis for image-level QC measurements and threshold assessment.
2. Bray MA et al. (2018). *Quality Control for High-Throughput Imaging Experiments Using Machine Learning in CellProfiler*. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6112602/). Basis for optional supervised artifact detection.
3. Caicedo JC et al. (2017). *Data-analysis strategies for image-based cell profiling*. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6871000/). Connects image QC to profiling workflows.
4. CellProfiler documentation. [MeasureImageQuality](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html). Direct implementation reference.
5. Otsu N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. [DOI](https://doi.org/10.1109/TSMC.1979.4310076).
6. Carpenter AE et al. (2006). *CellProfiler: image analysis software for identifying and quantifying cell phenotypes*. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC1794559/).
7. CellProfiler documentation. [IdentifyPrimaryObjects](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifyPrimaryObjects.html) and [IdentifySecondaryObjects](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/IdentifySecondaryObjects.html).
8. Schmidt U et al. (2018). *Cell Detection with Star-Convex Polygons* (StarDist). [Paper](https://arxiv.org/abs/1806.03535).
9. Stringer C et al. (2021). *Cellpose: a generalist algorithm for cellular segmentation*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/33318659/).
10. Cutler KJ et al. (2022). *Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation*. [Paper](https://www.nature.com/articles/s41592-022-01639-4).
11. *Evaluation of cell segmentation methods without reference segmentations* (2023). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10208095/). Reference-free evaluation ideas; validate their suitability for Cell Painting.
12. *SEG: Segmentation Evaluation in absence of Ground truth labels* (2023). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9980141/). Perturbation-based and multi-method comparisons, with limitations.
13. Cimini BA et al. (2023). *Optimizing the Cell Painting assay for image-based profiling*. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536784/). Percent replicating and percent matching as profiling evaluation tools.
14. *A systematic evaluation of computational methods for cell segmentation* (2024). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11330341/). Variation in performance with modality, channels, training data, and morphology.
15. *Metrics reloaded: recommendations for image analysis validation* (2024). [DOI](https://doi.org/10.1038/s41592-023-02151-z). Metric selection and validation cautions.
16. InstanSeg (2024 preprint). [Preprint](https://arxiv.org/abs/2408.15954). A potential future efficient learned candidate; not an MVP dependency.

---

**Implementation priority:** Ship image-quality gating first, then classical segmentation + common mask contracts, then CPA-controlled StarDist, then two-stage selection, application, and a Stage 4 evidence-based report with non-blocking optional review. Keep the method catalog larger than the executable algorithm set so readers can understand alternatives without unnecessarily enlarging the software.
