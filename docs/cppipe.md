# Run a CellProfiler pipeline on decoded arrays

Use the same `.cppipe` text and the same named image planes in Python, native Rust, and browser WASM. Supply image and label names from the pipeline, not file names. See the [module and settings limits](../spec/cppipe-compatibility.md).

## Python

Use `cp_anywhere.load_pipeline(path).run(images)` for the original three-value result. Use `run_detailed(images)` to also get `derived_images` and the `skipped` flag. Pass `scaling={"DNA": 65535}` only when that is the original acquisition scale. Use [`run_batch`](batch.md) to process multiple fields.

Use `cp_anywhere.inspect_pipeline(path)` before running an unfamiliar file.
It reports each module in file order, including disabled modules, caller-managed
I/O, unsupported modules, rejected settings, and missing object dependencies.
For example, if `RunCellpose` is unsupported and nothing else produces `Nuclei`,
its supported downstream consumers are reported as `missing_dependency`.
The check does not validate image inputs or label pixels.

If you already have labels, pass `objects={"Nuclei": nuclei_labels}` to `run`
or `run_detailed`. These are integer label planes, not intensity images; zero
is background. A supplied plane cannot replace an object set produced by a
supported module. If `MeasureObjectNeighbors` includes excluded objects,
also pass `object_diagnostics={"Nuclei": nuclei_small_removed}` with the
original small-removed label plane, not another copy of the kept labels.
`IdentifySecondaryObjects` can use external kept labels as seeds, but cannot
recreate edge seeds that existed only in the original unedited primary labels.
To check this configuration without running, pass the corresponding names
as `available_objects` and `available_object_diagnostics` to `inspect_pipeline`.
Legacy callers that supply labels through `images` can continue to do so
where that route was already supported, but should declare those object names
for preflight. Pass `available_objects=None` for a settings-only check.

For shared illumination maps, use `cp_anywhere.load_plate_illumination_pipeline(path)`.
Call `plan.run(image_sets)` with one ordered group of named image dictionaries.
This Python-only runner supports the CPJUMP1 bilinear resize and
`All: Across cycles` sequence. It returns named final maps; you supply the
groups and choose where to save the maps. See the [CPJUMP1 replay](../scripts/validation/jump_reference.md).

## Native Rust

Add `cp-anywhere` to your Cargo dependencies. Import `Pipeline`, `Plane`, and `Dtype` from `cp_anywhere::pipeline`, which is enabled by default. Call `Pipeline::parse(&text)` once. Call `pipeline.run(images)` with a vector of `(String, Plane)` pairs for each image set. A `Plane` has `shape`, `values` in row-major order, and an original `Dtype`. Use `pipeline.plan()` to inspect translated settings, skipped modules, and output descriptors.

For caller-supplied labels, call `pipeline.run_with_inputs(inputs)` with
`PipelineInputs`. Put kept labels in `objects` and small-removed labels in
`object_diagnostics`, keyed by the same object name.

## Browser WASM

Build the [web package](wasm.md), then initialize it as usual. Use `compile_cppipe(text)` to inspect the plan or `run_cppipe(text, imagesJson, pixels)` to process one field. `imagesJson` is a JSON array of `{name, shape, dtype}` descriptors. Pass a `Float64Array` with the raw plane values concatenated in descriptor order. For example, a 16-bit image has `dtype: "uint16"` and unscaled numeric pixel values.

To reuse a parsed pipeline, construct `new CppipePipeline(text)` once per worker. Call its `run(imagesJson, pixels)` method for each complete image set. Call `free()` when the worker no longer needs the pipeline.

Use `role: "objects"` for kept labels or `role: "object_diagnostics"` for
small-removed labels. For both descriptors, `name` is the object set name and
`dtype` must be an integer type. Image descriptors can omit `role` (the
default is `"image"`). Only image descriptors accept `scaling`.

The image descriptors also accept optional `scaling` metadata. Parse the returned JSON to read `primary`, `secondary`, `tertiary`, `object_shapes`, `derived_images`, `measurements`, and `skipped`. Arrays are flat and row-major. A skipped field retains its QC measurements but does not run later stages. Numeric nonfinite outputs use the strings `NaN`, `Infinity`, or `-Infinity`. When packing 64-bit integer inputs, check that each value converts to `Float64` without loss; not every 64-bit integer is representable.

## Execution limits

The single-field executor takes decoded arrays. It does not read image files, match metadata, group fields, or execute arbitrary modules. It preserves file order within the image preparation stage, followed by segmentation and measurement. It skips disabled and unsupported modules, but a field that passes QC cannot proceed to segmentation or measurement if a required object set is missing. Shared illumination maps can be prepared separately and supplied by name. `SaveImages` and `ExportToDatabase` translate to settings, but `run` does not write files.

## Verify parity

Build the WASM package and install Playwright Chromium as described in [the browser build instructions](wasm.md). Run from the repository root:

```sh
uv run --frozen poe cppipe-parity
```

This compares translated settings and execution against Python using the original pipeline text. It exercises the actual compiled WASM in Chromium. It checks agreement among cp-anywhere runtimes, not new independent CellProfiler agreement. The original CellProfiler oracle checks remain separate.
