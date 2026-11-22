# StarDist `2D_versatile_fluo` (ONNX)

## Source and licenses

- **Model:** The bundled `2D_versatile_fluo.onnx` comes from the StarDist
  [v0.1 release archive](https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip).
  `LICENSE-model` contains the release's [BSD 3-Clause license](https://raw.githubusercontent.com/stardist/stardist-models/v0.1/LICENSE.txt)
  (2018–2020, Martin Weigert and Uwe Schmidt).
- **Code:** `LICENSE` contains the separate [StarDist code license](https://raw.githubusercontent.com/stardist/stardist/e80c6de700693bc228ed3c9ba1dc19c3785667ee/LICENSE.txt)
  (BSD 3-Clause, 2018–2025, Uwe Schmidt and Martin Weigert). It covers the
  upstream code adapted for the Rust decoder, not the model release.

| Artifact                                                                                                                                                            | SHA-256                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `python_2D_versatile_fluo.zip` ([upstream pin](https://github.com/stardist/stardist/blob/e80c6de700693bc228ed3c9ba1dc19c3785667ee/stardist/models/__init__.py#L19)) | `4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd` |
| `weights_best.h5` (inside the ZIP)                                                                                                                                  | `42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349` |
| `2D_versatile_fluo.onnx` (bundled)                                                                                                                                  | `4d0c5663417544c88a6be2abaf09c4052f1202a71577425baa408d1a2d73741d` |

The release's `thresholds.json` specifies `prob=0.479071463157368` and
`nms=0.3`. Its `config.json` specifies 32 rays and a `(2, 2)` grid.

## Conversion and verification

`scripts/stardist/convert.py` exports with `tf2onnx` (opset 13) and a float32
input of shape `(1, height, width, 1)` with dynamic height and width. The script
checks the ZIP hash even on cache hits and extracts only the three expected
files. Export requires an explicit `--out` path and cannot overwrite a file.
Use `--verify-only` to check the bundled model without replacing it.

On November 22, 2026, the bundled ONNX file and the original Keras model received
the same seeded synthetic inputs. The check compared every element of both
outputs on macOS arm64 (TensorFlow 2.16.2, Keras 3.12.0, StarDist 0.9.2,
ONNX Runtime 1.23.2 with the CPU provider):

| Input (height × width) | Probability max absolute difference | Distance max absolute difference |
| ---------------------- | ----------------------------------: | -------------------------------: |
| 64 × 64                |                     `2.1457672e-06` |                  `6.1035156e-05` |
| 96 × 48                |                     `2.3841858e-06` |                  `1.0681152e-04` |

These probes check the model outputs, not Rust decoding or segmentation accuracy.
A fresh export also passed both probes, but did not reproduce the bundled file
byte for byte. The bundled file was not replaced.
No external imagery is bundled with the model or used for these probes.
