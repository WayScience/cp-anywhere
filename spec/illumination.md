# Illumination correction and image output

## Reference and current scope

The reference is CellProfiler 4.2.8 at commit
`2df090ff281cff80943fe679a1c3192fbda9cd6d`, with the packages in
`reference/requirements.lock`.
The fixtures come from that reference environment, not `cp_measure`.

This feature is partial, not a claim of full `CorrectIlluminationCalculate`
or `SaveImages` compatibility.

| Operation                              | Available entry point           | Current limits                                                    |
| -------------------------------------- | ------------------------------- | ----------------------------------------------------------------- |
| Calculate one correction function      | `calculate_illumination`        | Nonempty 2D grayscale arrays, five smoothing methods              |
| Calculate a shared correction function | `calculate_illumination_across` | Same-shaped 2D arrays, optional mask per image                    |
| Apply a correction function            | `apply_illumination`            | Equal-shaped arrays, divide or subtract, optional output clipping |
| Save an array                          | `save_array`                    | NPY, TIFF, and Pillow-supported PNG/JPEG/BMP/GIF combinations     |
| Import calculation configuration       | `IlluminationCalculateStep`     | Revision 2, `Each` mode only                                      |
| Import application configuration       | `IlluminationApplyStep`         | Revision 5, repeated image groups                                 |
| Import save configuration              | `SaveImagesStep`                | Revision 16, `Image`, `Single name`, TIFF/PNG/JPEG/NPY            |

Calculation uses the shared Rust core through the native Python binding.
Application uses NumPy.
File output uses NumPy, tifffile, or Pillow.
These operations do not have WASM bindings or browser verification yet.

## Calculate a correction function

`calculate_illumination(image, ...)` implements the `Each` calculation mode.
Its `mask` must share the image shape.
The API converts the input to float64 without normalizing integer intensities.
Callers supply pixels in the intended intensity range.
The imported pipeline instead applies CellProfiler-style input conversion.

The implemented smoothing methods use the exact CellProfiler strings:

- `No smoothing`
- `Median Filter`
- `Gaussian Filter`
- `Fit Polynomial`
- `Smooth to Average`

`Convex Hull` and `Splines` remain unimplemented.
Both raise `ValueError` through the calculation API and `CppipeError` during import.

`intensity_choice="Regular"` uses the source intensity.
`intensity_choice="Background"` uses block minima controlled by `block_size`.
The remaining configuration includes object dilation, dilation radius, smoothing width, and output rescaling.

`automatic_object_width` accepts `Automatic`, `Object size`, or `Manually`.
`object_width` controls `Object size`.
`size_of_smoothing_filter` controls `Manually`.
`rescale_option` accepts `Yes`, `No`, or `Median`.
`Yes` clamps values below the reference's robust minimum before division.
`Median` divides by the median without that clamp.

The implementation preserves float32 rounding at the corresponding reference image-construction stages.
A no-op stage does not introduce an extra conversion.
The returned array uses float64 storage, but intermediate float32 rounding remains visible.

## Calculate across images

`calculate_illumination_across(images, masks=..., ...)` calculates one shared function from all supplied images.
The images must share a nonempty 2D shape.
An optional mask list must contain one entry per image.
`None` marks an unmasked image.

The function averages each image's contribution with a per-pixel mask count.
Then it applies dilation, smoothing, and rescaling once to the average.
This supplies the numerical aggregation used by the reference's `All` modes.
It does not implement CellProfiler's cycle scheduling or image groups.

The `.cppipe` importer rejects `All: First cycle` and `All: Across cycles`.
`Pipeline.run()` accepts one image set, not the full sequence those modes require.
Callers can calculate a shared function separately and supply it as a named image to an imported application module.

## Apply and save

`apply_illumination(image, function, ...)` divides or subtracts arrays with equal shapes.
`divide_or_subtract="Divide"` pairs with `Regular` calculation.
`divide_or_subtract="Subtract"` pairs with `Background` calculation.
Subtraction always replaces negative results with zero, as the reference does.
`truncate_low` and `truncate_high` control the additional output limits at zero and one.
The result includes the reference output's float32 round-trip.

This example creates a 2D image, calculates its correction function, and saves both outputs:

```python
import numpy as np

from cp_anywhere import apply_illumination, calculate_illumination, save_array

image = np.linspace(0.1, 0.9, 480).reshape(20, 24)
function = calculate_illumination(image, smoothing_method="Gaussian Filter")
corrected = apply_illumination(image, function)
save_array(function, "illumination.npy")
save_array(corrected, "corrected.tiff", bit_depth="16-bit integer")
```

`save_array` dispatches on the destination suffix.
NPY preserves the supplied array and bypasses bit-depth conversion.
Other formats accept these conversion names:

- `8-bit integer`
- `16-bit integer`
- `32-bit floating point`
- `No conversion`

The conversion fixtures cover float32 and float64 pixels in `[0, 1]`.
Integer output scales those pixels to 255 or 65535, rounds half to even, and casts to the destination dtype.
This is not a per-image minimum/maximum rescale.
The current helper rejects values outside `[0, 1]` for integer output.
Integer-input conversion and negative floating-point values do not yet reproduce all `skimage.util.img_as_*` behavior.

The destination format must support the converted dtype and channel layout.
For PNG/JPEG, an explicit `bit_depth="8-bit integer"` is the portable choice.
Lossless TIFF/PNG conversion tests do not imply byte-identical container files.
JPEG is lossy.
The caller creates destination directories.
Direct `save_array` calls can overwrite an existing file.
NPY writes use the exact destination, including uppercase or mixed-case extensions.
They do not append a second `.npy` suffix.

## Imported save configuration

`load_pipeline(path).save_images_steps` contains the imported save modules.
Each step exposes `image_name`, `file_name`, `file_format`, `bit_depth`, and `overwrite`.
`step.write(image, destination)` writes a caller-supplied array to an explicit destination.
It refuses an existing destination when `overwrite` is false.

`step.effective_bit_depth` honors the imported depth for TIFF.
It forces 8-bit conversion for PNG/JPEG, matching the reference's `get_bit_depth()`.
NPY still bypasses conversion.

`Pipeline.run()` does not write files automatically.
It returns segmentation and measurement results, not corrected image arrays.
The standalone calculation/application API exposes arrays for explicit saving.
An imported `SaveImages` module alone is not an end-to-end pipeline save action.

The importer rejects `Mask`, `Cropping`, `Movie/Stack`, HDF5, and naming methods other than `Single name`.
Directory resolution, metadata substitution, source-file naming, sequence numbering, and subfolder creation are not implemented.
Cycle timing, filename measurements, and TIFF compression configuration are not implemented either.
Some inactive or unsupported configuration fields are currently consumed without validation or execution.
Retained averaged/dilated images from calculation modules are also unavailable.
These limits remain follow-up work, not completed compatibility.

## Evidence and reproduction

The checked-in fixtures contain:

- 252 `Each` calculation cases
- Four shared-function calculation cases, including mixed masked/unmasked images
- Eight application cases
- Four conversion arrays at three output depths, with 14,619 scalar comparisons
- Real serialized calculation/application and save modules in `fixtures/cppipe/illumination/`

The calculation and application tests use `rtol=1e-6` and `atol=1e-6`.
Conversion tests require exact array equality.
The existing two-module pipeline test exercises `Pipeline.run()` but compares separately calculated arrays with the reference.
It does not inspect the runner's intermediate images or prove automatic saving.

Run the tests:

```sh
env -u CONDA_PREFIX uv run --frozen pytest --no-cov \
  tests/test_illumination.py tests/test_save_images.py \
  tests/cppipe/test_illumination_pipeline.py
```

Regenerate fixtures only in the pinned reference environment:

```sh
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/reference/generate_illumination_calculate.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/reference/generate_illumination_across.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/reference/generate_illumination_apply.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/reference/generate_save_images_bitdepth.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/cppipe/generate_illumination_pipeline.py
PYTHONPATH=/path/to/pinned/CellProfiler /path/to/reference/venv/bin/python \
  scripts/cppipe/generate_save_images_pipeline.py
```

The `Each` generator uses Python's process-randomized `hash()` in its seed.
An unchanged script can therefore produce different input images on regeneration.
The checked-in inputs define the current comparisons.
