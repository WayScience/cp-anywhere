# ty: ignore[import-not-found]
"""Export the InstanSeg single_channel_nuclei model's CNN and pixel classifier
to ONNX, or check an existing export.

InstanSeg's released ``instanseg.pt`` is a TorchScript module embedding the
full postprocessing pipeline. PyTorch's ONNX exporters reject that scripted
graph (an internal ``input_values.size() == param_count_list.size()`` assert
in ``graph_utils.cpp``, reproduced on torch 2.8.0 and 2.15.0 with both the
legacy and ``torch.export`` exporters), and the postprocessing itself is
data-dependent (peak-local-maxima seed counts change per image), so a traced
whole-module export would bake one image's structure into the graph.

Instead this converter exports the two control-flow-free submodules -- the
traced CNN (``fcn``) and the small linear pixel classifier -- and leaves the
deterministic postprocessing to cp-anywhere's native implementation,
verified against the oracle pair upstream ships inside the same zip
(``test-input.npy`` / ``test-output_instance_segmentation.npy``). This
mirrors the StarDist arrangement (ONNX network + native decode), not an
invention.

Use a disposable Python 3.11 environment. None of the conversion
dependencies are required by cp-anywhere at runtime:

    python3.11 -m venv /tmp/instanseg-convert-venv
    /tmp/instanseg-convert-venv/bin/pip install --index-url \
        https://download.pytorch.org/whl/cpu 'torch==2.8.0'
    /tmp/instanseg-convert-venv/bin/pip install numpy onnx==1.17.0 onnxruntime \
        'instanseg_torch @ https://github.com/instanseg/instanseg/releases/download/v0.1.2/instanseg_torch-0.1.2-py3-none-any.whl' \
        pandas tifffile scipy seaborn matplotlib
    /tmp/instanseg-convert-venv/bin/python scripts/instanseg/convert.py \
        --workdir /tmp/instanseg-convert-work \
        --out-dir /tmp/instanseg-export

``--out-dir`` writes ``instanseg_fcn.onnx`` and
``instanseg_pixel_classifier.onnx`` plus fixture files under ``--fixtures``
(used by ``crates/cp_anywhere/tests/instanseg_fixtures/``);
``--verify-only DIR`` compares existing exports against torch without
writing anything; the converter never overwrites existing destinations.
Run ``--record-pins`` once to print the archive and TorchScript digests this
script pins. See src/cp_anywhere/data/instanseg/NOTICE.md for provenance
and licenses.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

# instanseg/instanseg release instanseg_models_v0.1.2, asset
# single_channel_nuclei.zip (model id 26423248 per its rdf.yaml;
# Apache-2.0 code and weights, trained on CC0/CC-BY datasets).
MODEL_URL = (
    "https://github.com/instanseg/instanseg/releases/download/"
    "instanseg_models_v0.1.2/single_channel_nuclei.zip"
)
MODEL_NAME = "single_channel_nuclei"
ZIP_SHA256 = "6bc2e4cd8acd9bea64a04e926a74a53f922d76fd6c1dc633ca0bca335df4d4d7"
TORCHSCRIPT_SHA256 = "118066231fb7753ffb048bcd2164186397bcc5c8fc4ad7f826efe928b515794c"
_OPSET = 13
# Per the bioimage.io rdf.yaml: float32, per-plane scale_range over the
# 0.1-99.9 percentiles, eps 1e-06. InstanSeg's own percentile_normalize
# defaults to eps 1e-3; both agree whenever the intensity range is not
# degenerate, and the rdf is the bioimage.io contract QuPath follows.
_PREPROCESS_PERCENTILE = 0.1
_PREPROCESS_EPS = 1e-6
# Largest absolute ONNX-vs-torch difference tolerated: float32 conv noise
# only (measured 7.4e-05 on the fixture).
_MAX_FCN_DIFF = 1e-3
_MAX_CLASSIFIER_DIFF = 1e-5


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def download_and_extract(workdir: Path) -> Path:
    archive_path = workdir / "single_channel_nuclei.zip"
    workdir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"downloading {MODEL_URL} ...")
        with urllib.request.urlopen(MODEL_URL) as response:
            archive_path.write_bytes(response.read())
    if _sha256_file(archive_path) != ZIP_SHA256:
        raise SystemExit(
            f"single_channel_nuclei.zip SHA-256 mismatch (expected {ZIP_SHA256})"
        )
    model_dir = workdir / MODEL_NAME
    model_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(model_dir)
    return model_dir


def load_torchscript(model_dir: Path):
    import torch

    script_path = model_dir / "instanseg.pt"
    if _sha256_file(script_path) != TORCHSCRIPT_SHA256:
        raise SystemExit(
            f"instanseg.pt SHA-256 mismatch (expected {TORCHSCRIPT_SHA256})"
        )
    return torch.jit.load(str(script_path), map_location="cpu").eval()


def preprocess(raw):
    """rdf.yaml preprocessing: float32, per-plane scale_range 0.1/99.9."""
    import numpy as np
    import torch

    x = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))
    flat = x.reshape(-1)
    pmin = torch.quantile(flat, _PREPROCESS_PERCENTILE / 100.0)
    pmax = torch.quantile(flat, 1.0 - _PREPROCESS_PERCENTILE / 100.0)
    return (x - pmin) / max(_PREPROCESS_EPS, float(pmax - pmin))


def pad_like_forward(normalized):
    """Replicate InstanSeg_Torchscript.forward's padding before ``fcn``."""
    import torch
    from instanseg.utils.loss import instanseg_loss

    with torch.no_grad():
        padded, pad = instanseg_loss._instanseg_padding(
            normalized, extra_pad=0, min_dim=16, ensure_square=False
        )
    return padded, pad.numpy().astype("int64")


def record_pins(workdir: Path) -> None:
    archive_path = workdir / "single_channel_nuclei.zip"
    workdir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        with urllib.request.urlopen(MODEL_URL) as response:
            archive_path.write_bytes(response.read())
    print("zip sha256:", _sha256_file(archive_path))
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if name.endswith(".pt"):
                print(f"{name} sha256:", _sha256_bytes(zf.read(name)))


def export_artifacts(model, model_dir: Path, out_dir: Path) -> dict:
    """Export fcn + pixel_classifier ONNX into ``out_dir`` (never overwrites)."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    probe = None
    probe_ref = None
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fcn": out_dir / "instanseg_fcn.onnx",
        "pixel_classifier": out_dir / "instanseg_pixel_classifier.onnx",
    }
    for path in paths.values():
        if path.exists():
            raise SystemExit(f"{path} already exists; the converter never overwrites")

    raw = np.load(model_dir / "test-input.npy")
    padded, pad = pad_like_forward(preprocess(raw))
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    print(
        f"fixture input {raw.shape} -> padded {tuple(padded.shape)}, pad {pad.tolist()}"
    )

    with torch.no_grad():
        torch_out = model.fcn(torch.from_numpy(padded)).numpy()
    torch.onnx.export(
        model.fcn,
        (torch.from_numpy(padded),),
        str(paths["fcn"]),
        input_names=["image"],
        output_names=["channels"],
        opset_version=_OPSET,
        # The CNN is fully convolutional: real fields arrive at arbitrary
        # (min_dim-16-padded) sizes, so the graph must stay dynamic.
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "channels": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load_model_from_string(paths["fcn"].read_bytes()))
    session = ort.InferenceSession(
        paths["fcn"].read_bytes(), providers=["CPUExecutionProvider"]
    )
    fcn_diff = float(
        np.abs(
            session.run(None, {"image": padded})[0].astype("float64")
            - torch_out.astype("float64")
        ).max()
    )
    if fcn_diff > _MAX_FCN_DIFF:
        raise SystemExit(
            f"fcn ONNX/torch parity beyond the {_MAX_FCN_DIFF} limit: {fcn_diff}"
        )
    print(f"fcn fixture parity ok (max|onnx - torch| = {fcn_diff:.3e})")

    # Second shape (not the fixture's): a different padded size must hold too.
    probe_shape = (1, 1, 128, 208)
    probe = torch.from_numpy(
        np.ascontiguousarray(_probe_input(rng_seed=7), dtype=np.float32)
    )
    with torch.no_grad():
        probe_ref = model.fcn(probe).numpy()
    probe_diff = float(
        np.abs(
            session.run(None, {"image": probe.numpy()})[0].astype("float64") - probe_ref
        ).max()
    )
    if probe_diff > _MAX_FCN_DIFF:
        raise SystemExit(
            f"fcn probe parity beyond the {_MAX_FCN_DIFF} limit: {probe_diff}"
        )
    print(f"fcn 128x208 probe parity ok (max|onnx - torch| = {probe_diff:.3e})")

    classifier = model.pixel_classifier
    width = int(classifier.fc1.weight.shape[1])
    probe = torch.zeros(3, width)
    with torch.no_grad():
        reference = classifier(probe).numpy()
    torch.onnx.export(
        classifier,
        (probe,),
        str(paths["pixel_classifier"]),
        input_names=["features"],
        output_names=["logits"],
        opset_version=_OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(
        onnx.load_model_from_string(paths["pixel_classifier"].read_bytes())
    )
    pc_session = ort.InferenceSession(
        paths["pixel_classifier"].read_bytes(), providers=["CPUExecutionProvider"]
    )
    pc_diff = float(
        np.abs(pc_session.run(None, {"features": probe.numpy()})[0] - reference).max()
    )
    if pc_diff > _MAX_CLASSIFIER_DIFF:
        raise SystemExit(
            f"pixel classifier parity beyond the {_MAX_CLASSIFIER_DIFF} limit: {pc_diff}"
        )
    print(
        f"pixel classifier parity ok (features width {width}, max diff {pc_diff:.3e})"
    )
    return {
        "paths": paths,
        "pad": [int(pad[0]), int(pad[1])],
        "torch_out": torch_out,
        "probe": probe.numpy(),
        "reference": reference,
        "probe_ref": probe_ref,
        "pixel_classifier_width": width,
    }


def write_fixtures(fixtures: Path, model_dir: Path, artifacts: dict) -> None:
    import numpy as np

    fixtures.mkdir(parents=True, exist_ok=True)
    np.save(fixtures / "fcn_input.npy", _fixture_input(model_dir))
    np.save(fixtures / "fcn_output_torch.npy", artifacts["torch_out"])
    np.save(fixtures / "fcn_probe_input.npy", _probe_input(rng_seed=7))
    np.save(fixtures / "fcn_probe_output_torch.npy", artifacts["probe_ref"])
    np.save(
        fixtures / "labels_oracle.npy",
        np.load(model_dir / "test-output_instance_segmentation.npy"),
    )
    np.save(fixtures / "classifier_probe_input.npy", artifacts["probe"])
    np.save(fixtures / "classifier_probe_torch.npy", artifacts["reference"])
    (fixtures / "details.json").write_text(
        json.dumps(_details(artifacts), indent=2) + "\n"
    )


def _fixture_input(model_dir: Path):
    import numpy as np

    raw = np.load(model_dir / "test-input.npy")
    padded, _ = pad_like_forward(preprocess(raw))
    return np.ascontiguousarray(padded, dtype=np.float32)


def _probe_input(rng_seed: int):
    import torch

    # Deterministic probe at a different (multiple-of-16) size; the Rust
    # fixture suite replays exactly this input through tract.
    generator = torch.Generator().manual_seed(rng_seed)
    return torch.rand(1, 1, 128, 208, generator=generator).numpy()


def _details(artifacts: dict) -> dict:
    return {
        "model_url": MODEL_URL,
        "zip_sha256": ZIP_SHA256,
        "torchscript_sha256": TORCHSCRIPT_SHA256,
        "torchscript_model": f"{MODEL_NAME}.pt",
        "opset": _OPSET,
        "pad_x": artifacts["pad"][0],
        "pad_y": artifacts["pad"][1],
        "probe_shape": [1, 1, 128, 208],
        "pixel_classifier_width": artifacts["pixel_classifier_width"],
        "preprocess_percentile": _PREPROCESS_PERCENTILE,
        "preprocess_eps": _PREPROCESS_EPS,
        "note": (
            "fcn_input is the rdf scale_range-preprocessed (0.1/99.9 percentiles, "
            "eps 1e-06) test-input, padded exactly like InstanSeg_Torchscript's own "
            "forward; labels_oracle is upstream's test-output_instance_segmentation. "
            "The native postprocessing port must reproduce labels_oracle from fcn_input."
        ),
        "onnx_sha256_fcn": _sha256_file(artifacts["paths"]["fcn"]),
        "onnx_sha256_pixel_classifier": _sha256_file(
            artifacts["paths"]["pixel_classifier"]
        ),
    }


CONVERTER_TITLE = "Export the InstanSeg single_channel_nuclei CNN to ONNX"


def main() -> None:
    parser = argparse.ArgumentParser(description=CONVERTER_TITLE)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--record-pins", action="store_true")
    args = parser.parse_args()
    if args.record_pins:
        record_pins(args.workdir)
        return
    if bool(args.out_dir) == bool(args.verify_only):
        raise SystemExit("pass exactly one of --out-dir / --verify-only")

    model_dir = download_and_extract(args.workdir)
    model = load_torchscript(model_dir)
    if args.verify_only:
        verify_only(model, model_dir, args.verify_only)
        print("verify done")
        return
    artifacts = export_artifacts(model, model_dir, args.out_dir)
    if args.fixtures:
        write_fixtures(args.fixtures, model_dir, artifacts)
    print("done")


def run_parity(onnx_bytes: bytes, torch_out, padded) -> float:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    run = session.run(None, {"image": padded})[0]
    diff = np.abs(run.astype("float64") - np.asarray(torch_out).astype("float64"))
    return float(diff.max())


def verify_only(model, model_dir: Path, existing_dir: Path) -> None:
    """Re-run both parity checks against existing ONNX files without writing."""
    import numpy as np
    import torch

    fcn_path = existing_dir / "instanseg_fcn.onnx"
    classifier_path = existing_dir / "instanseg_pixel_classifier.onnx"
    for path in (fcn_path, classifier_path):
        if not path.exists():
            raise SystemExit(f"verify-only: {path} does not exist")

    raw = np.load(model_dir / "test-input.npy")
    padded, _pad = pad_like_forward(preprocess(raw))
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    with torch.no_grad():
        torch_out = model.fcn(torch.from_numpy(padded)).numpy()
    fcn_diff = run_parity(fcn_path.read_bytes(), torch_out, padded)
    print(f"fixture parity: max|onnx - torch| = {fcn_diff:.3e}")
    if fcn_diff > _MAX_FCN_DIFF:
        raise SystemExit(
            f"fcn ONNX/torch parity beyond the {_MAX_FCN_DIFF} limit: {fcn_diff}"
        )

    classifier = model.pixel_classifier
    width = int(classifier.fc1.weight.shape[1])
    probe = torch.zeros(3, width)
    with torch.no_grad():
        reference = classifier(probe).numpy()
    import onnxruntime as ort

    session = ort.InferenceSession(
        classifier_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    pc_diff = float(
        np.abs(session.run(None, {"features": probe.numpy()})[0] - reference).max()
    )
    print(f"pixel classifier parity ok (max diff {pc_diff:.3e})")
    if pc_diff > _MAX_CLASSIFIER_DIFF:
        raise SystemExit(
            f"pixel classifier parity beyond the {_MAX_CLASSIFIER_DIFF} limit: {pc_diff}"
        )


if __name__ == "__main__":
    main()
