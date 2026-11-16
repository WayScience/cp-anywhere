#  ty: ignore[import-not-found]
"""Generate real .cppipe pipeline files from the pinned CellProfiler
4.2.8 oracle, for round-trip-verifying ``cp_anywhere.cppipe``'s
settings importer against genuine CellProfiler-written text (never
hand-crafted) -- see ``spec/cppipe-compatibility.md``.

Each case reuses ``scripts/segmentation/generate_reference.py``'s own
``configure_module`` so the *exact same* settings already
oracle-verified for segmentation (``fixtures/segmentation/``) are what
gets saved into the pipeline file, and saves alongside each ``.cppipe``
file the Python kwargs used to build it (the ground truth the parsed
result must match).

Run with the separate reference environment (same pinned CellProfiler
source as ``scripts/segmentation/generate_reference.py``), never with
this project's own ``cp_anywhere``.
"""

from __future__ import annotations

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.measureimageintensity import MeasureImageIntensity
from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity
from cellprofiler.modules.measureobjectneighbors import MeasureObjectNeighbors
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "fixtures/cppipe"

sys.path.insert(0, str(ROOT))
from scripts.segmentation.generate_reference import (
    configure_module,
    run_segmentation,
)

# A fixed twin-blob test image, shared by every case, already
# oracle-verified many times over this project's segmentation work
# (crates/cp_anywhere_segmentation's declump end-to-end test,
# fixtures/segmentation/synthetic_declump_*). uint16 so the real
# default `Image(..., convert=True)` rescale this generator's own
# `run_segmentation` applies is exercised, matching
# cp_anywhere.identify_primary_objects's own default `rescale=True`.
_ROWS, _COLS = 20, 24
_YY, _XX = np.mgrid[0:_ROWS, 0:_COLS]
_D1 = np.sqrt((_YY - 9) ** 2 + (_XX - 8) ** 2)
_D2 = np.sqrt((_YY - 9) ** 2 + (_XX - 15) ** 2)
TEST_IMAGE = (
    np.maximum(np.clip(1 - _D1 / 7, 0, 1), np.clip(1 - _D2 / 7, 0, 1)) * 65535
).astype(np.uint16)

# A dedicated 3-blob image for the "wants_excluded_objects=True" case:
# three overlapping circular blobs (same radial-falloff pattern as
# TEST_IMAGE's own twin blobs), split into 3 separate objects by
# shape-based declumping/watershed at each blob's own local maximum,
# with the rightmost blob reaching the image's right edge exactly so a
# real `exclude_border_objects=True` run genuinely discards it.
_EX_ROWS, _EX_COLS = 20, 30
_EYY, _EXX = np.mgrid[0:_EX_ROWS, 0:_EX_COLS]
_ED1 = np.sqrt((_EYY - 9) ** 2 + (_EXX - 5) ** 2)
_ED2 = np.sqrt((_EYY - 9) ** 2 + (_EXX - 13) ** 2)
_ED3 = np.sqrt((_EYY - 9) ** 2 + (_EXX - 28) ** 2)
_NEIGHBORS_EXCLUDED_IMAGE = (
    np.maximum(
        np.maximum(np.clip(1 - _ED1 / 6, 0, 1), np.clip(1 - _ED2 / 6, 0, 1)),
        np.clip(1 - _ED3 / 6, 0, 1),
    )
    * 65535
).astype(np.uint16)


def build_pipeline(
    name: str,
    ipo_kwargs: dict,
    *,
    measure: bool = True,
    wants_excluded: bool = False,
    test_image: np.ndarray | None = None,
) -> None:
    image = TEST_IMAGE if test_image is None else test_image
    pipeline = Pipeline()

    m1 = IdentifyPrimaryObjects()
    m1.set_module_num(1)
    m1.x_name.value = "DNA"
    m1.y_name.value = "Cells"
    configure_module(m1, **ipo_kwargs)
    pipeline.add_module(m1)

    if measure:
        m2 = MeasureObjectIntensity()
        m2.set_module_num(2)
        m2.images_list.value = "DNA"
        m2.objects_list.value = "Cells"
        pipeline.add_module(m2)

        m3 = MeasureImageIntensity()
        m3.set_module_num(3)
        m3.images_list.value = "DNA"
        m3.wants_objects.value = False
        pipeline.add_module(m3)

        m4 = MeasureObjectNeighbors()
        m4.set_module_num(4)
        m4.object_name.value = "Cells"
        m4.neighbors_name.value = "Cells"
        m4.distance_method.value = "Adjacent"
        m4.wants_excluded_objects.value = wants_excluded
        pipeline.add_module(m4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{name}.cppipe").open("w") as fp:
        pipeline.dump(fp, save_image_plane_details=False)
    if test_image is not None:
        np.save(OUT_DIR / f"{name}.test_image.npy", test_image)

    labels, threshold, small_removed, _threshold_map, _guide = run_segmentation(
        image, **ipo_kwargs
    )
    object_count = int(labels.max())
    np.save(OUT_DIR / f"{name}.labels.npy", labels.astype(np.uint32))
    expected: dict = {
        "ipo_kwargs": ipo_kwargs,
        "has_measure": measure,
        "object_count": object_count,
        "threshold": threshold,
    }
    if measure and wants_excluded:
        # Real oracle ground truth for MeasureObjectNeighbors with
        # "Consider objects discarded for touching image border?" =
        # Yes, run directly (not hand-derived) against this case's own
        # real kept/small_removed label pair.
        expected["neighbors_excluded"] = _real_neighbors_excluded(labels, small_removed)
    (OUT_DIR / f"{name}.expected.json").write_text(
        json.dumps(expected, indent=2) + "\n"
    )


def _real_neighbors_excluded(labels: np.ndarray, small_removed: np.ndarray) -> dict:
    pipeline = Pipeline()
    neighbors = MeasureObjectNeighbors()
    neighbors.set_module_num(1)
    neighbors.object_name.value = "Cells"
    neighbors.neighbors_name.value = "Cells"
    neighbors.distance_method.value = "Adjacent"
    neighbors.wants_excluded_objects.value = True
    pipeline.add_module(neighbors)
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    object_set = ObjectSet()
    objects = Objects()
    objects.segmented = labels
    objects.small_removed_segmented = small_removed
    object_set.add_objects(objects, "Cells")
    measurements = Measurements()
    workspace = Workspace(
        pipeline, neighbors, image_set, object_set, measurements, image_set_list
    )
    neighbors.run(workspace)
    return {
        "count": list(
            measurements.get_current_measurement(
                "Cells", "Neighbors_NumberOfNeighbors_Adjacent"
            )
        ),
        "first": [
            int(v)
            for v in measurements.get_current_measurement(
                "Cells", "Neighbors_FirstClosestObjectNumber_Adjacent"
            )
        ],
        "touching": list(
            measurements.get_current_measurement(
                "Cells", "Neighbors_PercentTouching_Adjacent"
            )
        ),
    }


CASES = {
    "otsu_simple": {"threshold_method": "otsu"},
    "manual": {"threshold_method": "manual", "threshold_value": 0.3},
    "otsu_three_class_background": {
        "threshold_method": "otsu_three_class",
        "assign_middle_to_foreground": False,
    },
    "li": {"threshold_method": "li"},
    "robust_background": {
        "threshold_method": "robust_background",
        "robust_background_lower_outlier_fraction": 0.1,
        "robust_background_upper_outlier_fraction": 0.02,
        "robust_background_averaging_method": "median",
        "robust_background_variance_method": "mad",
        "robust_background_number_of_deviations": 3.0,
    },
    "adaptive_otsu": {
        "threshold_method": "otsu",
        "threshold_scope": "adaptive",
        "adaptive_window_size": 8,
    },
    "adaptive_sauvola": {
        "threshold_method": "sauvola",
        "threshold_scope": "adaptive",
        "adaptive_window_size": 8,
    },
    "declump_intensity_intensity": {
        "threshold_method": "otsu",
        "min_diameter": 10.0,
        "unclump_method": "intensity",
        "watershed_method": "intensity",
    },
    "declump_shape_propagate_manual_smoothing": {
        "threshold_method": "otsu",
        "min_diameter": 10.0,
        "unclump_method": "shape",
        "watershed_method": "propagate",
        "smoothing_filter_size": 3.0,
        "maxima_suppression_size": 4.0,
    },
    "fill_holes_border_limit": {
        "threshold_method": "otsu",
        "fill_holes": "after_thresholding_and_declump",
        "exclude_border_objects": True,
        "exclude_size": True,
        "min_diameter": 5.0,
        "max_diameter": 50.0,
        "maximum_object_count": 200,
    },
    "threshold_smoothing_log_transform": {
        "threshold_method": "otsu",
        "threshold_smoothing_scale": 1.3488,
        "log_transform": True,
        "threshold_correction_factor": 0.9,
        "threshold_range": (0.1, 0.9),
    },
    "segmentation_only_no_measure": {"threshold_method": "otsu"},
}

_NEIGHBORS_EXCLUDED_CASE = (
    "neighbors_excluded_border_object",
    {
        "threshold_method": "otsu",
        "min_diameter": 8.0,
        "unclump_method": "shape",
        "watershed_method": "shape",
        "exclude_border_objects": True,
    },
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "test_image.npy", TEST_IMAGE)
    for name, kwargs in CASES.items():
        build_pipeline(name, kwargs, measure=name != "segmentation_only_no_measure")
    name, kwargs = _NEIGHBORS_EXCLUDED_CASE
    build_pipeline(
        name,
        kwargs,
        measure=True,
        wants_excluded=True,
        test_image=_NEIGHBORS_EXCLUDED_IMAGE,
    )
    print(json.dumps({"written": len(CASES) + 1, "out_dir": str(OUT_DIR)}))


if __name__ == "__main__":
    main()
