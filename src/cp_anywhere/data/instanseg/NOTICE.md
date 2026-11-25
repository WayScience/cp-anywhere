# InstanSeg `single_channel_nuclei` (ONNX)

## Source and licenses

- **Model:** The bundled `instanseg_fcn.onnx` and
  `instanseg_pixel_classifier.onnx` are converted from the InstanSeg
  [instanseg_models_v0.1.2 release archive](https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.2/single_channel_nuclei.zip)
  (model id 26423248 on the InstanSeg model zoo). The upstream
  TorchScript archive is `single_channel_nuclei.pt` (inside the ZIP).
  `LICENSE` contains the InstanSeg code/weights license (Apache-2.0,
  upstream instanseg/instanseg).
- **Code:** The Rust postprocessing port mirrors upstream
  `instanseg.utils.loss.instanseg_loss` (`InstanSeg_Torchscript.forward`
  and its helpers at the `instanseg_models_v0.1.2` pin), which is also
  Apache-2.0; the port follows the same upstream repo license.
- **Training datasets** (per the model's own README; users remain
  responsible for using the model in accordance with the dataset
  licenses): `cpdmi_2023` (CC BY 4.0,
  https://www.nature.com/articles/s41597-023-02108-z), `dsb_2018` (CC0,
  https://bbbc.broadinstitute.org/BBBC038), and `bsst265` (CC0,
  https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265).

| Artifact                                                      | SHA-256                                                            |
| ------------------------------------------------------------- | ------------------------------------------------------------------ |
| `single_channel_nuclei.zip` (upstream release archive)        | `6bc2e4cd8acd9bea64a04e926a74a53f922d76fd6c1dc633ca0bca335df4d4d7` |
| `single_channel_nuclei.pt` (TorchScript model inside the ZIP) | `118066231fb7753ffb048bcd2164186397bcc5c8fc4ad7f826efe928b515794c` |
| `instanseg_fcn.onnx` (bundled)                                | `5707bb92e3ddc097f32498c4507a9739280c506ed0a4bf782561a77b9e8aaeb7` |
| `instanseg_pixel_classifier.onnx` (bundled)                   | `a77cde586576ab480d2e2f2dbeae3ba8bcc569494426fa77aa7590435e97c80d` |

## Conversion and verification

`scripts/instanseg/convert.py` splits the upstream TorchScript model into
two ONNX exports (the fcn/trunk network and the small pixel classifier)
and checks the ZIP hash before extracting. The fcn export uses a float32
input of shape `(1, 1, height, width)` with dynamic height and width
(opset 13); the pixel classifier takes a float32 `(batch, features)`
input with 4 embedding features (the released graph pins the trace probe
shape `(3, 4)`, but it is pure Gemm/Relu and runs at any batch size). The
single-channel-nuclei variant emits 5 fcn channels: 2 coordinate offsets,
2 sigmas, and 1 seed map; it segments nuclei only (the cells-and-nuclei
variant of upstream is a declared gap here). See
`scripts/instanseg/README.md` and
`crates/cp_anywhere/tests/instanseg_fixtures/` for the parity fixtures.

Parity was verified against upstream in two steps: (1) the bundled fcn
ONNX reproduces the TorchScript fcn output on the upstream test image
and a separate seeded probe within `1e-3` per element; (2) the full
native path (padding + fcn + the Rust port of upstream postprocessing)
reproduces upstream's own `test-output_instance_segmentation` label map
exactly (25 labels, upstream's test image).

**Released-graph divergences from the installed wheel source.** The
shipped `single_channel_nuclei.pt` graph differs from the
`instanseg_torch==0.1.2` wheel's Python sources in three ways, all
verified against upstream's oracle and ported faithfully in
`crates/cp_anywhere/src/segmentation/instanseg.rs`: the overlap merge
uses `fast_sparse_intersection_over_minimum_area` (IoMA), not `IoU`; the
model's own saved defaults are `seed_threshold = 0.1`,
`overlap_threshold = 0.5`, and `min_size = 20` (the wheel source's
`params.get` fallbacks are 0.7 / 0.3 / 10); and the graph actively
filters objects by `min_size` (raw foreground count strictly greater
than 20) before cleanup, which the wheel sources never do inside
`forward`.

No external imagery beyond the upstream model release's own test
inputs is bundled or used for these checks.
