# ty: ignore[import-not-found]
"""Oracle for the 2D nearest-nonzero-label fill ("Voronoi expand")
MeasureObjectNeighbors' "Expand until adjacent" distance method needs:
every background pixel gets the label of its nearest foreground pixel,
via real scipy's own `distance_transform_edt(labels == 0,
return_indices=True)` -- the exact same primitive real CellProfiler's
`measureobjectneighbors.py` itself calls for D_EXPAND.
"""

import json
from pathlib import Path

import numpy as np
import scipy.ndimage

OUT = Path(__file__).resolve().parents[2] / "fixtures/neighbors_expand"

_CASES = {
    "two_distant_objects": np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "three_objects_uneven": np.array(
        [
            [1, 1, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "single_object": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "sparse_ids_with_gap": np.array(
        [
            [5, 0, 0, 0, 9],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    ),
    "already_touching": np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint32,
    ),
    "all_foreground_no_background": np.array(
        [
            [1, 2],
            [1, 2],
        ],
        dtype=np.uint32,
    ),
}


def generate() -> None:
    """Write each case's real scipy nearest-label fill as JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, labels in _CASES.items():
        i, j = scipy.ndimage.distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        expanded = labels[i, j]
        payload = {
            "reference": "scipy distance_transform_edt(..., return_indices=True)",
            "labels": labels.tolist(),
            "expanded": expanded.tolist(),
        }
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(name, labels.shape, "ok")


if __name__ == "__main__":
    generate()
