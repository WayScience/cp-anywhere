#!/usr/bin/env python3
"""Export the StarDist 2D_versatile_fluo model to ONNX or check an existing export.

Use a disposable Python 3.11 environment. None of the conversion dependencies
are required by cp-anywhere at runtime:

    python3.11 -m venv /tmp/stardist-convert-venv
    /tmp/stardist-convert-venv/bin/pip install 'tensorflow==2.16.2' \
        'keras==3.12.0' 'stardist==0.9.2' 'tf2onnx==1.17.0' \
        'onnx==1.17.0' 'onnxruntime==1.23.2'
    /tmp/stardist-convert-venv/bin/python scripts/stardist/convert.py \
        --workdir /tmp/stardist-convert-work --out /tmp/2D_versatile_fluo.onnx

Use --verify-only FILE in place of --out to compare an existing ONNX file with
Keras without writing a new export. The converter never overwrites a file.
See src/cp_anywhere/data/stardist/NOTICE.md for provenance and licenses.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO

MODEL_URL = "https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip"
MODEL_NAME = "python_2D_versatile_fluo"
ARCHIVE_SHA256 = "4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd"
_ARCHIVE_SIZE = 5_320_433
_MODEL_MEMBERS = ("config.json", "thresholds.json", "weights_best.h5")
# Absolute error limits for the two float32 outputs.
_MAX_PROB_DIFF = 1e-3
_MAX_DIST_DIFF = 1e-2
_PROBE_SHAPES = ((64, 64), (96, 48))
_OUTPUT_COUNT = 2


def _read_verified_archive(source: BinaryIO, label: str) -> bytes:
    data = source.read(_ARCHIVE_SIZE + 1)
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"{label}: SHA-256 mismatch (expected {ARCHIVE_SHA256})")
    return data


def _write_new_file(path: Path, data: bytes) -> None:
    """Publish a complete file without replacing an existing destination."""
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        staged = Path(staging) / path.name
        staged.write_bytes(data)
        path.hardlink_to(staged)


def download_and_extract(workdir: Path, run_dir: Path) -> Path:
    archive_path = workdir / "model.zip"
    try:
        cached = archive_path.open("rb")
    except FileNotFoundError:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            data = _read_verified_archive(response, MODEL_URL)
        _write_new_file(archive_path, data)
    else:
        with cached:
            data = _read_verified_archive(cached, str(archive_path))

    # The checksum authenticates all three members. Never use loose cached
    # config, thresholds or weights from a previous run.
    model_dir = run_dir / MODEL_NAME
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sorted(archive.namelist()) != list(_MODEL_MEMBERS):
            raise SystemExit("Unexpected members in the model archive")
        model_dir.mkdir()
        for name in _MODEL_MEMBERS:
            with archive.open(name) as source, (model_dir / name).open("xb") as dest:
                shutil.copyfileobj(source, dest)
    return model_dir


def convert(model_dir: Path, run_dir: Path) -> Path:
    import tensorflow as tf  # noqa: PLC0415  # ty: ignore[import-not-found]
    import tf2onnx  # noqa: PLC0415  # ty: ignore[import-not-found]
    from stardist.models import (  # noqa: PLC0415  # ty: ignore[import-not-found]
        StarDist2D,
    )

    model = StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
    spec = (tf.TensorSpec((1, None, None, 1), tf.float32, name="input"),)
    onnx_path = run_dir / "2D_versatile_fluo.onnx"
    tf2onnx.convert.from_keras(
        model.keras_model, input_signature=spec, opset=13, output_path=str(onnx_path)
    )
    return onnx_path


def verify(model_dir: Path, onnx_path: Path) -> None:
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415  # ty: ignore[import-not-found]
    from stardist.models import (  # noqa: PLC0415  # ty: ignore[import-not-found]
        StarDist2D,
    )

    model = StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for height, width in _PROBE_SHAPES:
        rng = np.random.default_rng(0)
        image = np.clip(rng.normal(0.4, 0.1, size=(1, height, width, 1)), 0, 1).astype(
            np.float32
        )
        reference = model.keras_model.predict(image, verbose=0)
        outputs = session.run(None, {"input": image})
        if len(outputs) != _OUTPUT_COUNT or len(reference) != _OUTPUT_COUNT:
            raise SystemExit("Expected two model outputs: probability and distance")
        outputs.sort(key=lambda output: output.shape[-1])
        for name, expected, output, limit in zip(
            ("prob", "dist"),
            reference,
            outputs,
            (_MAX_PROB_DIFF, _MAX_DIST_DIFF),
            strict=True,
        ):
            if output.shape != expected.shape or output.dtype != expected.dtype:
                raise SystemExit(
                    f"{height}x{width} {name}: output shape or dtype mismatch"
                )
            diff = float(np.abs(output - expected).max())
            print(f"{height}x{width} {name}: max abs diff vs Keras = {diff:.8g}")
            if not diff <= limit:
                raise SystemExit(f"{height}x{width} {name}: difference exceeds {limit}")
    print("ONNX passed both Keras probe comparisons")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, required=True, help="archive cache folder"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="write a new, verified ONNX file")
    action.add_argument("--verify-only", type=Path, help="check an existing ONNX file")
    args = parser.parse_args()
    out: Path | None = args.out
    if out is not None and (out.exists() or out.is_symlink()):
        parser.error(f"will not replace existing path: {out}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.workdir) as staging:
        run_dir = Path(staging)
        model_dir = download_and_extract(args.workdir, run_dir)
        if out is None:
            verify(model_dir, args.verify_only)
            return
        onnx_path = convert(model_dir, run_dir)
        verify(model_dir, onnx_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_new_file(out, onnx_path.read_bytes())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
