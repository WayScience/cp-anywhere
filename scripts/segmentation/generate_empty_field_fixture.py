"""Oracle fixture for the all-zero ("empty field") stress input.

Adds the ``synthetic_declump_shape_propagate_empty_field`` segmentation
fixture: the production pilot's stress-empty-field validation input
(``A9-A9-p10-empty``, the all-zero 450x450 float DNA channel -- a
constant image whose global two-class Otsu threshold is 0.0, making
every pixel foreground) run with the pilot's primary settings
(Shape unclump + Propagate watershed, min/max diameter 10/40,
exclude_size, fill Never).

Real CellProfiler 4.2.8 declumps the single all-foreground blob into
ONE object -- ``scipy.ndimage.distance_transform_edt`` on an image
with no background pixel returns a strictly monotone bowl
(``sqrt((row + 1)^2 + col^2)``, an artifact of scipy 1.9.3's
zero-initialized feature transform), whose seed-0-jittered maxima
detection finds exactly one maximum at pixel (449, 449) -- and the
max-size filter then erases that one 202500-px object (0 final
objects; ``small_removed`` keeps it). See
``crates/cp_anywhere_segmentation/src/declump.rs``'s
``distance_transform_edt`` for the native-side counterpart.

This script runs ONLY under the pinned reference environment (same
interpreter as ``scripts/segmentation/generate_reference.py`` -- real
CellProfiler 4.2.8, never cp_anywhere). It reuses that script's
``run_segmentation``/``configure_module`` machinery verbatim so the
oracle configuration is not re-derived, writes the fixture arrays and
case JSON in the same on-disk format, and appends the manifest entry
idempotently (an existing entry with the same name is replaced).

Run:
    PYTHONPATH=<pinned cellprofiler source> <reference venv python> \\
        scripts/segmentation/generate_empty_field_fixture.py
"""

# The reference API is dynamically typed.
# ruff: noqa: E402

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.segmentation.generate_reference import run_segmentation

CASE_NAME = "synthetic_declump_shape_propagate_empty_field"
OUT_DIR = ROOT / "fixtures/segmentation"
# The production pilot's stress-empty-field shape (450x450) and dtype.
PIXELS = np.zeros((450, 450), dtype=np.float32)
# The pilot's primary settings (run-full/inputs/manifest.json "settings").
KWARGS = {
    "threshold_method": "otsu",
    "exclude_size": True,
    "min_diameter": 10.0,
    "max_diameter": 40.0,
    "unclump_method": "shape",
    "watershed_method": "propagate",
    "fill_holes": "never",
    "threshold_smoothing_scale": 0.0,
    "threshold_correction_factor": 1.0,
    "threshold_range": (0.0, 1.0),
    "log_transform": False,
    "threshold_scope": "global",
    "exclude_border_objects": False,
    "low_resolution_maxima": False,
}


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def main() -> None:
    labels, threshold, small_removed, _, _ = run_segmentation(PIXELS, **KWARGS)
    folder = f"fixtures/segmentation/{CASE_NAME}"
    case_files = {
        f"{folder}/pixels.npy": npy_bytes(PIXELS),
        f"{folder}/labels.npy": npy_bytes(labels),
        f"{folder}/small_removed_labels.npy": npy_bytes(small_removed),
    }
    for relative, payload in case_files.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    result = {
        "image_id": CASE_NAME,
        "settings": {**KWARGS, "threshold_range": [0.0, 1.0]},
        "rescale": True,
        "threshold": threshold,
        "object_count": int(labels.max()),
        "shape": list(PIXELS.shape),
        "sha256": {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in case_files.items()
        },
    }
    (OUT_DIR / f"{CASE_NAME}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = {
        "name": CASE_NAME,
        "result": f"{folder}.json",
        "threshold": threshold,
        "object_count": result["object_count"],
    }
    manifest["cases"] = [
        case for case in manifest["cases"] if case["name"] != CASE_NAME
    ] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "case": CASE_NAME,
                "threshold": threshold,
                "object_count": result["object_count"],
                "small_removed_object_count": int(small_removed.max()),
                "labels_all_zero": bool(np.all(labels == 0)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
