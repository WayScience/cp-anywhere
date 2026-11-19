# Batch processing independent image sets

Use `cp_anywhere.run_batch` to apply the same supported `.cppipe` pipeline to many fields. Pass a mapping from unique image IDs to named image paths or arrays. Use the names in the pipeline, not the file names.

`workers=1` runs in the caller and is the default. With two or more workers, choose `backend="thread"` (the default) or `backend="process"`. A single field always runs in the caller. Both backends bound the number of in-flight fields by the worker count and yield results in input order.

Use threads in notebooks. The process backend starts fresh Python interpreters with the `spawn` start method. For process mode, put the batch call in a Python script under a main guard:

```python
import cp_anywhere as cp

if __name__ == "__main__":
    image_sets = {
        "A01-site1": {"DNA": "a01_s1_dna.tif", "Stain": "a01_s1_actin.tif"},
        "A01-site2": {"DNA": "a01_s2_dna.tif", "Stain": "a01_s2_actin.tif"},
    }
    for field in cp.run_batch(
        "pipeline.cppipe", image_sets, workers=2, backend="process"
    ):
        if field.measurements is not None:
            field.measurements.to_parquet(f"{field.image_id}.parquet")
```

Prefer file paths in process mode. If you pass arrays, Python copies their data to workers. Python also copies results back to the caller. If you use a custom `Pipeline` subclass, define it at module level so workers can import it. Save each result before you read the next to limit memory use. Do not change the pipeline or inputs while the batch runs.

Errors stop iteration and identify the affected image set in the traceback. Outputs you already saved remain in place. If you stop reading early, call `.close()` on the batch iterator to cancel queued work where possible and wait for running work to finish.

Each result contains the same primary and derived objects as `Pipeline.run`. If the pipeline measures objects, the result also contains measurements with the caller's image ID. Use `field.measurements.to_parquet(...)` to write a separate file per field. Use a new output directory for each run or choose your own overwrite policy.

For more control, call `Pipeline.run` for a single field or compose the individual functions. This helper does not match files, infer plate layout, schedule cluster jobs, retry failures, or execute pipeline export steps. Keep shared illumination estimation outside the batch: estimate each map once with `calculate_illumination_across`, then supply that saved map under the name expected by the pipeline's correction step. See the [pipeline import limits](../spec/cppipe-compatibility.md).

## Batch parity across runtimes

For full supported pipeline execution on a single field, see the [cross-runtime `.cppipe` guide](cppipe.md). The batch check below tests scheduling separately on a smaller operation graph.

- Python: `cp_anywhere.run_batch` runs a supported `.cppipe` with serial, thread, or process execution and yields results in input order.
- Native Rust: `cp_anywhere::batch::run_batch` runs one caller-supplied operation per input, on the caller's thread or a bounded set of threads. It returns results in input order. It does not parse `.cppipe` or load images. Process scheduling remains external in Rust.
- Browser WASM: the [browser example](../examples/browser-vite/src/batch.ts) runs primary segmentation and measurement in one or more Web Workers. Each worker loads one WASM instance. The example takes prepared grayscale pixels and segmentation settings, not `.cppipe` files. It does not require shared WASM memory or cross-origin isolation. The worker pool is example code, not an npm package export.

The CI comparison runs two real microscopy crops and a synthetic empty field through all three paths. It uses the same prepared pixels and verifies input order, image IDs, all three label layers, counts, object IDs, neighbor edges, and intensity and neighbor features. Discrete outputs and nonfinite kinds match exactly. Floating features and thresholds use relative and absolute tolerances of `1e-10`. Serial and parallel execution must agree exactly within each runtime.

This check uses `fixtures/cppipe/declump_shape_propagate_manual_smoothing.cppipe` with size filtering enabled and a maximum diameter of 40 pixels. It does not establish batch parity for secondary or tertiary segmentation, illumination, other measurement modules, or input and output formats. The separate CellProfiler oracle checks still cover the supported individual operations.

After [building the WASM package and browser example](wasm.md) and installing Chromium through Playwright, run from the repository root:

```sh
uv run --frozen python -m scripts.parity.compare_batch
```

## Python experiment: 10 full microscopy fields

The [benchmark script](../benchmarks/benchmark_batch.py) processes the grayscale DNA and actin pairs in [BBBC007 v1](https://bbbc.broadinstitute.org/BBBC007). It uses an existing translated CellProfiler fixture: primary nuclei, secondary cells, tertiary cytoplasm, and cytoplasm intensity. Six fields stored as non-grayscale images were excluded rather than converted. No new image data is vendored.

On an Apple M4 Pro with Python 3.11.13 and a release build of the Rust extension:

| Execution           | Median time for 10 fields | Speedup vs. serial |
| ------------------- | ------------------------: | -----------------: |
| Serial              |                   0.606 s |                  — |
| 2 threads           |                   0.318 s |              1.91× |
| 4 threads           |                   0.181 s |              3.34× |
| 2 spawned processes |                   0.426 s |              1.42× |
| 4 spawned processes |                   0.303 s |              2.00× |

Each mode uses the public `run_batch` API and had one warmup and three timed runs in rotated order. Timing includes loading prepared NumPy arrays, creating workers, processing all fields, and writing three label arrays and one Parquet file per field. It excludes downloading and preparing the images, parsing the pipeline, and hashing the outputs. All 40 files matched the serial run byte for byte in every trial. This checks batch equivalence, not agreement with manual outlines or a new CellProfiler run.

For this workload, threads were faster than creating a fresh process pool. This does not establish the fastest worker count for other workloads or for a long-lived process pool. The benchmark does not measure peak memory or cold reads from remote storage. Correctness tests use bundled images and have no timing threshold.

Run from the repository root. The benchmark checks the pinned archive hashes and downloads the archives only if they are missing from the cache directory.

```sh
uv run --no-sync maturin develop --release --uv
uv run --no-sync python -m benchmarks.benchmark_batch --data-dir /path/to/bbbc007-cache
```
