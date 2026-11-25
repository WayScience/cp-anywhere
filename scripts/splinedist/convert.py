# ty: ignore[import-not-found]
"""Export the SplineDist bbbc038_10 model to ONNX and write oracle fixtures.

SplineDist (uhlmanngroup/splinedist, BSD-3-Clause) is a StarDist-architecture
CNN whose network predicts spline control points as (rho, phi) pairs per
pixel instead of star-convex ray distances, enabling non-star-convex shapes.
The pretrained weights ship on Zenodo (DOI 10.5281/zenodo.7193306,
CC-BY-4.0). The `bbbc038_10` model (DSB2018 fluorescence nuclei, the same
data family as StarDist's `2D_versatile_fluo`) uses n_params=20 control
points, grid (2,2), contoursize_max=399, and ships its own thresholds.json,
phi basis, grid_10.npy, and a bundled example image.

This mirrors the StarDist arrangement (scripts/stardist/convert.py): keras
-> tf2onnx ONNX export verified against the real model, then the spline
decode is ported natively in Rust and verified against real
`predict_instances` labels. The oracle fixtures embed upstream's own NMS
kernel via the installed stardist C library, so the labels are genuine
SplineDist outputs -- never re-derived.

Use a disposable Python 3.10 environment. None of the conversion
dependencies are required by cp-anywhere at runtime:

    python3.10 -m venv /tmp/splinedist-convert-venv
    /tmp/splinedist-convert-venv/bin/pip install 'numpy==1.23.5' \
        'tensorflow-macos==2.10.0'  # or 'tensorflow==2.10.1' on x86-64
    /tmp/splinedist-convert-venv/bin/pip install csbdeep scikit-image \
        opencv-python-headless scipy 'stardist==0.8.3' 'tf2onnx==1.16.1' \
        'onnx==1.17.0' onnxruntime 'setuptools<81'
    /tmp/splinedist-convert-venv/bin/pip install \
        'splinedist @ git+https://github.com/uhlmanngroup/splinedist'
    # macOS/arm64 only: stardist's compiled NMS kernel links LLVM OpenMP
    # without declaring it, so preload it:
    export DYLD_INSERT_LIBRARIES=/opt/homebrew/opt/libomp/lib/libomp.dylib
    /tmp/splinedist-convert-venv/bin/python scripts/splinedist/convert.py \
        --workdir /tmp/splinedist-convert-work \
        --out-dir /tmp/splinedist-export

``--out-dir`` writes `splinedist.onnx` and the fixture files (used by
``crates/cp_anywhere/tests/splinedist_fixtures/``); the converter never
overwrites existing destinations. See
src/cp_anywhere/data/splinedist/NOTICE.md for provenance and licenses.
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

# Zenodo record 7193306 "Splinedist Trained Weights", CC-BY-4.0.
MODEL_URL = "https://zenodo.org/api/records/7193306/files/splinedist_models.zip/content"
ZIP_SIZE = 306_585_771
ZIP_MD5 = "c121caf72e2c89acb98f7a60d6411f32"
ZIP_SHA256 = "75b56f3d8f1741cb727db5bdd5c851184e3134d0f77501a2b9a48d46ab0081cc"
MODEL_NAME = "bbbc038_10"
# keras->onnx float32 tolerance on the two network heads.
_MAX_OUTPUT_DIFF = 5e-3
_PROBE_H, _PROBE_W = 128, 128


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_and_extract(workdir: Path) -> Path:
    archive_path = workdir / "splinedist_models.zip"
    workdir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists() or _sha256_file(archive_path) != ZIP_SHA256:
        print(f"downloading {MODEL_URL} ... (~306 MB; resumable across attempts)")
        for _ in range(12):
            size = archive_path.stat().st_size if archive_path.exists() else 0
            if size >= ZIP_SIZE and _sha256_file(archive_path) == ZIP_SHA256:
                break
            subprocess.run(
                [
                    "curl", "-sL", "-C", "-", "--retry", "5", "--retry-delay", "5",
                    "-o", str(archive_path), MODEL_URL,
                ],
                check=False,
            )
    if _sha256_file(archive_path) != ZIP_SHA256 or hashlib.md5(
        archive_path.read_bytes()
    ).hexdigest() != ZIP_MD5:
        raise SystemExit(
            f"splinedist_models.zip checksums do not match "
            f"(md5 {ZIP_MD5}, sha256 {ZIP_SHA256})"
        )
    model_root = workdir / "splinedist_models"
    model_dir = model_root / MODEL_NAME
    if not (model_dir / "weights_best.h5").exists():
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(workdir)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the SplineDist bbbc038_10 CNN to ONNX")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import tf2onnx
    import tensorflow as tf
    from PIL import Image
    from csbdeep.utils import normalize
    from splinedist.models import SplineDist2D

    model_dir = download_and_extract(args.workdir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "splinedist.onnx"
    if onnx_path.exists():
        raise SystemExit(f"{onnx_path} already exists; the converter never overwrites")

    thresholds = json.loads((model_dir / "thresholds.json").read_text())
    print("thresholds:", thresholds)
    config = json.loads((model_dir / "config.json").read_text())
    n_params = int(config["n_params"])  # network channel count: 2 * control points
    grid = tuple(int(g) for g in config["grid"])
    assert grid == (2, 2), str(config)

    # The model zip bundles one real example image; normalize it exactly the
    # way upstream's prediction.ipynb does (csbdeep 1-99.8 percentile).
    raw_img = np.asarray(Image.open(model_dir / "bbbc038.png").convert("L"))
    img_normalized = normalize(raw_img.astype(np.float32), 1, 99.8, axis=(0, 1)).astype(
        np.float32
    )
    print("example image:", raw_img.shape, raw_img.dtype)

    model = SplineDist2D(None, name=MODEL_NAME, basedir=str(model_dir.parent))

    # Record the exact network input/output tensors that flow into
    # _instances_from_selection so the native port replays real tensors.
    recorded: dict = {}

    keras_predict = model.keras_model.predict

    def recording_predict(inputs, **kwargs):
        outputs = keras_predict(inputs, **kwargs)
        recorded["input"] = np.asarray(inputs).copy()
        if isinstance(outputs, list):
            recorded["outputs"] = [np.asarray(o).copy() for o in outputs]
        else:
            recorded["outputs"] = [np.asarray(outputs).copy()]
        return outputs

    model.keras_model.predict = recording_predict
    labels, details = model.predict_instances(
        img_normalized, prob_thresh=thresholds["prob"], nms_thresh=thresholds["nms"]
    )
    model.keras_model.predict = keras_predict
    np.save(out_dir / "labels_oracle.npy", labels.astype(np.int32))
    print(
        "oracle labels:", int(labels.max()), "instances over shape", labels.shape
    )

    # The spline basis ships with the model; its second axis gives the real
    # control-point count (config n_params counts the 2*points channels).
    phi_files = sorted(model_dir.glob("phi_*.npy"))
    if len(phi_files) != 1:
        raise SystemExit(f"expected exactly one phi basis file, got {phi_files}")
    phi_path = phi_files[0]
    phi_basis = np.load(phi_path)
    control_points = int(phi_basis.shape[1])
    contoursize = int(phi_basis.shape[0])
    print(f"phi basis {phi_path.name}: {phi_basis.shape} -> {control_points} control points, {contoursize} contour samples")
    assert control_points * 2 == n_params, (control_points, n_params)
    np.save(out_dir / "phi_basis.npy", phi_basis)

    net_input = recorded["input"]
    net_outputs = recorded["outputs"]
    print("recorded network input:", net_input.shape)
    print("recorded network outputs:", [o.shape for o in net_outputs])
    np.save(out_dir / "network_input.npy", net_input)
    for i, out in enumerate(net_outputs):
        np.save(out_dir / f"network_output_keras_{i}.npy", out)

    # tf2onnx export with symbolic H/W (the network is fully convolutional,
    # and real fields arrive at arbitrary grid-divisible padded sizes).
    spec = (tf.TensorSpec([1, None, None, 1], tf.float32, name="image"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model.keras_model,
        input_signature=spec,
        opset=13,
        output_path=str(onnx_path),
        large_model=False,
    )
    onnx.checker.check_model(onnx.load_model_from_string(onnx_path.read_bytes()))
    print("onnx check ok")

    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    diffs = []
    for i, keras_out in enumerate(net_outputs):
        onnx_out = session.run(None, {"image": net_input})[i]
        diff = float(
            np.abs(onnx_out.astype("float64") - keras_out.astype("float64")).max()
        )
        diffs.append(diff)
    print("per-head max|keras-onnx| on the recorded input:", diffs)
    if max(diffs) > _MAX_OUTPUT_DIFF:
        raise SystemExit(f"keras/ONNX parity beyond the {_MAX_OUTPUT_DIFF} limit: {diffs}")

    # Second probe at a different size (divisible by the unet's axes_div_by),
    # recorded straight from real keras predictions.
    div_by = model._axes_div_by("YX")
    probe_h = _PROBE_H - (_PROBE_H % div_by[0])
    probe_w = _PROBE_W - (_PROBE_W % div_by[1])
    probe_input = np.zeros((1, probe_h, probe_w, 1), dtype=np.float32)
    probe_input[0, :, :, 0] = np.sin(
        np.linspace(0.0, 12.0, probe_h)[:, None]
    ) + np.cos(np.linspace(0.0, 9.0, probe_w))[None, :]
    probe_outputs = model.keras_model.predict(probe_input, verbose=0)
    probe_outputs = probe_outputs if isinstance(probe_outputs, list) else [probe_outputs]
    recorded["probe_input"] = probe_input.copy()
    recorded["probe_outputs"] = [np.asarray(o).copy() for o in probe_outputs]
    for i, out in enumerate(recorded["probe_outputs"]):
        np.save(out_dir / f"probe_output_keras_{i}.npy", out)
    np.save(out_dir / "probe_input.npy", probe_input)
    onnx_probe = session.run(None, {"image": probe_input})
    probe_diffs = [
        float(
            np.abs(
                np.asarray(onnx_probe[i]).astype("float64")
                - np.asarray(p).astype("float64")
            ).max()
        )
        for i, p in enumerate(recorded["probe_outputs"])
    ]
    print("per-head max|keras-onnx| on the probe:", probe_diffs)
    if max(probe_diffs) > _MAX_OUTPUT_DIFF:
        raise SystemExit(f"keras/ONNX probe parity beyond the {_MAX_OUTPUT_DIFF} limit: {probe_diffs}")
    print("keras/ONNX parity within limits")

    (out_dir / "details.json").write_text(
        json.dumps(
            {
                "model_url": MODEL_URL,
                "zip_md5": ZIP_MD5,
                "zip_sha256": ZIP_SHA256,
                "model": MODEL_NAME,
                "n_params": n_params,
                "grid": list(grid),
                "contoursize_max": contoursize,
                "thresholds": thresholds,
                "weights_sha256": _sha256_file(model_dir / "weights_best.h5"),
                "phi_sha256": _sha256_file(phi_path),
                "control_points": control_points,
                "contoursize_max": contoursize,
                "example_image": "bbbc038.png",
                "normalizer": "csbdeep normalize percentile 1-99.8 axes (0,1)",
                "onnx_sha256": _sha256_file(onnx_path),
                "opset": 13,
                "axes_div_by": list(model._axes_div_by("YX")),
                "network_input_shape": list(net_input.shape),
                "note": (
                    "network_input/network_output_keras_* are the exact tensors that "
                    "upstream _instances_from_prediction consumed to produce "
                    "labels_oracle; the native port must turn network_input through "
                    "the ONNX heads plus the ported spline decode into exactly "
                    "labels_oracle."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("done")


if __name__ == "__main__":
    main()
