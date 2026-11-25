# SplineDist `bbbc038_10` (ONNX)

## Source and licenses

- **Model:** The bundled `splinedist.onnx` is converted from the
  `bbbc038_10` weights in the uhlmanngroup/splinedist
  [Trained Weights release](https://zenodo.org/records/7193306)
  (Zenodo record 7193306, "Splinedist Trained Weights"; DOI
  10.5281/zenodo.7193306). Weights are CC-BY-4.0 (the record's license);
  `LICENSE` contains the upstream *code* license (BSD-3-Clause,
  uhlmanngroup/splinedist).
- **Code:** SplineDist builds on StarDist (BSD-3-Clause,
  https://github.com/stardist/stardist, see upstream's own
  `LICENSE_stardist.txt`); its decode pipeline (spline control-point
  decode, grid rescaling, IoMA non-maximum suppression through
  StarDist's C kernel, and probability-ordered rasterization) follows
  upstream's own Python/C kernels. The native port in
  `crates/cp_anywhere/src/segmentation/splinedist.rs` mirrors that
  pipeline under the same BSD-3-Clause terms.
- **Training dataset** (per the model's provenance; users remain
  responsible for using the model in accordance with the dataset
  license): DSB2018 / BBBC038 (Broad Bioimage Benchmark Collection
  BBBC038v1, https://bbbc.broadinstitute.org/BBBC038).

| Artifact                                         | SHA-256                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `splinedist_models.zip` (Zenodo 7193306 archive) | `75b56f3d8f1741cb727db5bdd5c851184e3134d0f77501a2b9a48d46ab0081cc` |
| `bbbc038_10/weights_best.h5` (upstream weights)  | `4c1bf5547d91f95ebd9723a03a75ec605c0d117cdec0272c10936125a03b9995` |
| `bbbc038_10/phi_10.npy` (upstream phi basis)     | `fe1dc838c73a5a461b2bbc3432c74ae6059efcf03a676417f81a3c87e97e829c` |
| `splinedist.onnx` (bundled)                      | `c36989c35bdbdcf42f2a750439316a1d109b64fba1cc3f64740089179cbb8953` |
| `phi_basis.npy` (bundled, copy of `phi_10.npy`)  | `fe1dc838c73a5a461b2bbc3432c74ae6059efcf03a676417f81a3c87e97e829c` |

## Conversion and verification

`scripts/splinedist/convert.py` converts the pinned upstream Keras
`weights_best.h5` to ONNX (float32 input `(1, H, W, 1)` with dynamic
H/W, opset 13; two grid-(2,2)-compressed heads: probability
`(1, H/2, W/2, 1)` and 20-channel spline control-point coordinates
`(1, H/2, W/2, 20)`), verifies the archive hash before extracting, and
generated the parity fixtures in
`crates/cp_anywhere/tests/splinedist_fixtures/`.

Parity was verified against upstream in two steps: (1) the bundled ONNX
heads reproduce upstream Keras' exact head tensors (per-element error
below `1e-3` on both the recorded input and a second-shape probe);
(2) the full native path (the bundled ONNX through tract plus the Rust
port of upstream's `dist_to_coord` -> spline sampling -> IoMA NMS ->
`polygons_to_label` decode) reproduces upstream's own
`SplineDist2D.predict_instances` output (123 instances) **exactly
(pixel-for-pixel label equality)** on the reference real image. The
shipped model's decode thresholds are the model's own `thresholds.json`
(`prob` 0.4775366187095642, `nms` 0.3), not invented values; they are
constants in the native decode (see
`crates/cp_anywhere/tests/splinedist_fixtures/details.json`).

Only the fluorescence-nuclei `bbbc038_10` model is bundled; upstream's
H&E `conic` model variants exist but are not shipped here.

No external imagery beyond the model record's own example image is
bundled or used for these checks.
