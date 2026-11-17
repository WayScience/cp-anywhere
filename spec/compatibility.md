# Compatibility contract (M0/M1)

## Reference and claim

The scientific reference is the unmodified CellProfiler 4.2.8 source at commit `2df090ff281cff80943fe679a1c3192fbda9cd6d`. It runs with `cellprofiler-core==4.2.8` and the exact headless package versions in `reference/requirements.lock`. The installation was validated on macOS arm64 with Python 3.9.25 and Java 25.0.2. Other platforms are unverified. The project package uses Python 3.11 or newer. The separate reference environment does not change that requirement.

`cp_measure==0.3.0` at commit `cdf693187e3dcaf3d067da7b58bdd4ab8896f91e` is the later comparison and performance baseline. Its output is not the scientific reference. The project license remains BSD-3-Clause, Dave Bunten. The original CellProfiler source is BSD-3-Clause. The real image crop is CC0.

The initial declared release scope contains every measurement emitted by these CellProfiler modules with the saved pipeline settings:

| Module                   | Settings in scope                                             | Output                                                          |
| ------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------- |
| `MeasureImageIntensity`  | Whole image, `DNA` and `Actin`, percentiles 10, 50, 90        | All emitted image columns                                       |
| `MeasureObjectIntensity` | `DNA` and `Actin` against both `Cells` and `Nuclei`           | All emitted object columns, including location columns          |
| `MeasureObjectNeighbors` | `Cells` against itself, `Adjacent`, excluded objects disabled | All seven object columns and the `Neighbors` relationship table |

The golden fixtures cover 2D and 3D runs. Both channels are separate grayscale arrays. Both object sets use supplied integer labels. In most fixtures `Nuclei` is derived deterministically from `Cells` by selecting one pixel per object ID; `real_bbbc007` is the one exception, using genuinely independent, multi-pixel `Cells` and `Nuclei` labels segmented from a real dataset's own ground truth. Masks apply to intensity measurements. `MeasureObjectNeighbors` does not respect image masks in the original module. Spacing is recorded as metadata. The measured values use CellProfiler pixel coordinates and counts, with no physical-unit scaling.

For this initial scope, “full feature set” means the complete 119-column inventory in `spec/features.yaml` and the directed relationship rows. A future implementation cannot claim parity by implementing a subset of those columns. More channels, additional object sets, and different names require new manifests and fixtures before a claim. The checked-in fixture names and settings are the present contract.

## Later extensions

The table above remains the initial golden-fixture contract. Later
extensions add secondary and tertiary segmentation, broader neighbor
object selection, and 2D `MeasureObjectSizeShape` with Zernike and
advanced columns. These have separate real-reference fixtures and
limits documented in `spec/cppipe-compatibility.md` and
`docs/size-shape.md`; they do not change the original 119-column inventory.

Illumination correction and array output are later, partial extensions.
Native Python supports five calculation smoothers, shared-function aggregation,
divide/subtract application, and common image formats.
The `.cppipe` importer also reads a restricted subset of their configuration.
`SaveImagesStep.write()` requires a caller-supplied array and destination.
`Pipeline.run()` does not save corrected images automatically.
`spec/illumination.md` records the supported configuration, reference evidence,
and remaining gaps, including Convex Hull, Splines, HDF5, movie stacks, and WASM.

## Explicit exclusions

The following CellProfiler 4.2.8 measurement modules remain outside this release scope: `MeasureImageAreaOccupied`, `MeasureImageOverlap`, `MeasureImageQuality`, `MeasureImageSkeleton`, `MeasureObjectOverlap`, and `MeasureObjectSkeleton`. (`MeasureGranularity`, `MeasureObjectSizeShape`, `MeasureObjectIntensityDistribution`, `MeasureTexture`, and `MeasureColocalization` have since been implemented with their own real-reference fixtures; see `spec/cppipe-compatibility.md`.) Any measurement from other CellProfiler modules is also outside scope.

Other settings of the selected modules are outside scope: object-restricted `MeasureImageIntensity`; other percentile lists or disabled percentiles; different image or object selections for `MeasureObjectIntensity`; and cross-set or distance-based (`"Within X pixels"`) `MeasureObjectNeighbors`. (`MeasureObjectNeighbors`'s `"Expand until adjacent"` distance method has since been implemented -- `measure(..., neighbors_expand=...)` and the `.cppipe` importer's `MeasurementStep.wants_expand`; see `spec/cppipe-compatibility.md`. Excluded-object handling has also since been implemented via `neighbors_full_objects`.) The optional neighbor visualization images are outside scope. Parent-child relationships, upstream segmentation, and calibration-driven value conversion are outside scope. The fixture runner consumes arrays directly in a real CellProfiler `Workspace`; it is not a full file-import pipeline.

Large sparse label IDs are limited by the original modules. They allocate or index arrays proportional to the maximum label ID. The `sparse_ids` fixture reaches ID 16 and preserves gaps. An ID near the 32-bit limit is not run because it would require excessive memory. This is an explicit CellProfiler limitation, not a passing fixture.

## Parity gate

Compare feature presence, object scope, image ID, object ID, relationship topology, shape, dtype, null and nonfinite values, then numeric values. Match discrete counts and IDs exactly. The initial float64 rule is per-feature `atol=1e-10`, `rtol=1e-9`. This is a provisional numerical rule for reduction-order variation. It does not permit a feature to pass with an unexplained mismatch. Tighten it during feature implementation where exact arithmetic is feasible. Never relax a global threshold to hide a failure. Each claimed native, Python, and WASM feature must pass the original reference. A failed feature is excluded from the claim and reported.

No Rust, Python binding, WASM, or `cp_measure` parity is claimed in M0/M1.
