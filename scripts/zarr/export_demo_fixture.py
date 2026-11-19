"""Export one golden fixture image as a real OME-Zarr v2 array, and build a
tiny, real ONNX threshold model that produces a binary foreground mask from
that image (plan M7: "Integrate with zarrita.js/OME-Zarr, ONNX Runtime Web
label masks").

This is a demo-scale integration test fixture, not a scientific
segmentation model: it is a deterministic `Greater(image, threshold)` graph,
not instance segmentation or a learned model (see docs/browser-integration.md
"Known limitations"). It exists to prove the real browser pipeline --
Zarr read -> ONNX Runtime Web inference -> cp-anywhere measure() ->
DuckDB-WASM Parquet export -- runs end to end with no mocked stage, per the
plan's explicit requirement.

Writes a hand-built Zarr v2 array directory (no `zarr` package dependency
at runtime; the format is simple enough, and this keeps the demo-generation
script dependency-free beyond numpy/onnx, which are build-time tools only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"
OUT = ROOT / "examples/zarr-onnx-duckdb/public"
THRESHOLD = 0.5


def write_zarr_v2_array(out_dir: Path, array: np.ndarray) -> None:
    """Write a single-chunk Zarr v2 array: .zarray, .zattrs, chunk 0.0 / 0.0.0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype_code = {"float64": "<f8", "float32": "<f4", "uint32": "<u4"}[str(array.dtype)]
    zarray = {
        "zarr_format": 2,
        "shape": list(array.shape),
        "chunks": list(array.shape),
        "dtype": dtype_code,
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    (out_dir / ".zarray").write_text(json.dumps(zarray))
    (out_dir / ".zattrs").write_text(json.dumps({}))
    chunk_name = ".".join(["0"] * array.ndim)
    (out_dir / chunk_name).write_bytes(np.ascontiguousarray(array).tobytes())


def build_threshold_model(shape: tuple[int, ...]) -> onnx.ModelProto:
    """Greater(image, threshold) -> Cast(..., INT32): a real, checked ONNX graph."""
    threshold_initializer = helper.make_tensor(
        "threshold", TensorProto.FLOAT, [], [THRESHOLD]
    )
    image_input = helper.make_tensor_value_info("image", TensorProto.FLOAT, list(shape))
    mask_output = helper.make_tensor_value_info("mask", TensorProto.INT32, list(shape))
    greater = helper.make_node("Greater", ["image", "threshold"], ["is_foreground"])
    cast = helper.make_node("Cast", ["is_foreground"], ["mask"], to=TensorProto.INT32)
    graph = helper.make_graph(
        [greater, cast],
        "cp_anywhere_threshold_mask",
        [image_input],
        [mask_output],
        initializer=[threshold_initializer],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    checker.check_model(model)
    return model


def main() -> None:
    dna = np.load(FIXTURE / "DNA.npy").astype(np.float32)
    normalized = dna / float(dna.max()) if dna.max() > 0 else dna
    write_zarr_v2_array(OUT / "dna.zarr", normalized.astype(np.float64))

    model = build_threshold_model(normalized.shape)
    model_path = OUT / "threshold.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, model_path)

    evaluator = ReferenceEvaluator(model)
    (raw_mask,) = evaluator.run(None, {"image": normalized})
    expected_mask = np.asarray(raw_mask)
    expected_path = ROOT / "examples/zarr-onnx-duckdb/expected-mask.json"
    expected_path.write_text(
        json.dumps(
            {
                "shape": list(normalized.shape),
                "mask": expected_mask.astype(int).tolist(),
                "foreground_pixels": int(expected_mask.sum()),
                "model_sha256": __import__("hashlib")
                .sha256(model_path.read_bytes())
                .hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "zarr_dir": str(OUT / "dna.zarr"),
                "onnx_model": str(model_path),
                "expected_mask": str(expected_path),
                "foreground_pixels": int(expected_mask.sum()),
                "shape": list(normalized.shape),
            }
        )
    )


if __name__ == "__main__":
    main()
