"""Export the saved golden fixture input arrays for the browser demo.

This is a browser-demo convenience: real deployments decode pixel data with
zarrita.js and labels with ONNX Runtime Web (see docs/decisions/0002-rust-core.md
and plan section 9). This script stands in for that decode step so the
Vite example and the in-browser parity check can run against the exact same
saved, checksummed arrays used by ``scripts/parity/compare_native.py``, for
all declared fixtures (2D and 3D).

Numeric payloads are written as raw native-endian binary (float64 images,
uint32 labels, uint8 mask), not JSON: a few fixtures contain NaN/Infinity
pixel values, which are not valid in strict JSON numbers. Typed-array views
over raw bytes preserve those IEEE754 bit patterns exactly, and exercising
real `ArrayBuffer`/typed-array transfer is also the point of a WASM demo.
Only small, always-finite metadata (shape, names, image_id) is JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_fixture(entry: dict, out_dir: Path) -> None:
    """Write one fixture's metadata JSON plus its raw binary arrays."""
    folder = ROOT / Path(next(iter(entry["sha256"]))).parent
    image_names = entry["channel_names"]
    images = {
        name: np.load(folder / f"{name}.npy").astype(np.float64) for name in image_names
    }
    objects = {
        "Cells": np.load(folder / "labels.npy").astype(np.uint32),
        "Nuclei": np.load(folder / "nuclei_labels.npy").astype(np.uint32),
    }
    mask_path = folder / "mask.npy"
    has_mask = mask_path.exists()
    name = entry["name"]
    metadata = {
        "shape": list(entry["shape"]),
        "image_names": image_names,
        "object_names": ["Cells", "Nuclei"],
        "has_mask": has_mask,
        "image_id": name,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(metadata))
    images_blob = np.concatenate([images[n].ravel() for n in image_names])
    (out_dir / f"{name}.images.f64").write_bytes(images_blob.tobytes())
    objects_blob = np.concatenate([objects[n].ravel() for n in ("Cells", "Nuclei")])
    (out_dir / f"{name}.objects.u32").write_bytes(objects_blob.tobytes())
    if has_mask:
        mask = np.load(mask_path).astype(np.uint8)
        (out_dir / f"{name}.mask.u8").write_bytes(mask.ravel().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "examples/browser-vite/public/fixtures"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "fixtures/reference/manifest.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for entry in manifest["fixtures"]:
        write_fixture(entry, args.out_dir)
        names.append(entry["name"])
    (args.out_dir / "index.json").write_text(json.dumps({"fixtures": names}))
    print(json.dumps({"written": len(names), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
