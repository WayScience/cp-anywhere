"""Pairwise image/object colocalization and correlation, implemented
against CellProfiler 4.2.8's ``MeasureColocalization`` module as the
compatibility oracle.

Declared scope: 2D images only (real CellProfiler's module also
supports 3D, with 13 image-pair correspondence vectors instead of the
2D case's implicit single comparison -- that volumetric case is an
explicit, documented gap, matching every other module in this project).
This port also shares a single ``costes_scale_max`` across every image
pair in one call, rather than real CellProfiler's own per-pair
``get_scale(first_image.scale, second_image.scale)`` resolution -- pass
the largest bit-depth scale across your images if they differ::

    result = cp_anywhere.measure_colocalization(
        {"DNA": dna_image, "GFP": gfp_image}, objects={"Cells": cell_labels},
    )
    result.image["Correlation_Correlation_DNA_GFP"]       # a float
    result.objects["Correlation_Correlation_DNA_GFP"]      # one value per object

All seven feature families (Correlation+Slope, Overlap+K1/K2, Manders,
RWC, and Costes-thresholded Manders) are independently gated by the
``do_*`` flags, matching real CellProfiler's own "Run all metrics?"/
individual-flag UI. A pair whose combined mask has no unmasked pixels
reports ``NaN`` for every feature of that pair, matching real
CellProfiler's own empty-mask branch exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_EXPECTED_NDIM = 2
_MIN_IMAGES = 2

FastCostesMode = Literal["faster", "fast", "accurate"]


@dataclass(frozen=True)
class ColocalizationResult:
    """Whole-image and per-object pairwise colocalization measurements,
    keyed by the real ``Correlation_{Feature}_{img1}_{img2}`` column
    name CellProfiler itself uses."""

    image: dict[str, float]
    """Present only when ``wants_image_measurements`` is set."""

    objects: dict[str, np.ndarray]
    """``objects[column][i]`` is one value per object (ascending label
    ID); present only when ``wants_object_measurements`` is set."""


def measure_colocalization(
    images: Mapping[str, object],
    *,
    objects: object | None = None,
    masks: Mapping[str, object] | None = None,
    threshold_percent: float = 15.0,
    fast_costes: FastCostesMode = "faster",
    costes_scale_max: float = 255.0,
    do_corr_and_slope: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    wants_image_measurements: bool = True,
    wants_object_measurements: bool = False,
) -> ColocalizationResult:
    """Run CellProfiler's pairwise colocalization/correlation on two or
    more 2D images.

    ``images`` maps image name to a 2D array; every pair (in the
    canonical ``i < j`` order of ``images``'s own iteration order,
    matching real CellProfiler's ``get_image_pairs()``) is measured.
    ``objects``, if given, is a single 2D label array (one object set
    per call, like every other measurement in this package);
    ``wants_object_measurements`` requires it. ``threshold_percent`` is
    the real "Set threshold as percentage of maximum intensity" setting
    (``0..=99``); ``fast_costes`` selects which Costes' automated
    threshold search real CellProfiler's "Method for Costes
    thresholding" setting would pick.
    """
    from cp_anywhere import _native

    image_names = list(images)
    if len(image_names) < _MIN_IMAGES:
        raise ValueError("at least two images are required")
    arrays, (rows, cols) = _validated_images(images)

    mask_arrays: dict[str, np.ndarray] = {}
    for name, mask in (masks or {}).items():
        if name not in arrays:
            raise ValueError(f"mask given for unknown image {name!r}")
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=np.bool_)
        if mask_array.shape != (rows, cols):
            raise ValueError(f"mask for {name!r} must have the image shape")
        mask_arrays[name] = mask_array

    labels_array = None
    if objects is not None:
        labels_array = np.ascontiguousarray(np.asarray(objects), dtype=np.uint32)
        if labels_array.shape != (rows, cols):
            raise ValueError("objects must have the image shape")
    if wants_object_measurements and labels_array is None:
        raise ValueError("wants_object_measurements requires objects")

    image_dict, object_dict = _native.measure_colocalization(
        arrays,
        rows,
        cols,
        masks=mask_arrays or None,
        labels=labels_array,
        threshold_percent=threshold_percent,
        fast_costes=fast_costes,
        costes_scale_max=costes_scale_max,
        do_corr_and_slope=do_corr_and_slope,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        wants_image_measurements=wants_image_measurements,
        wants_object_measurements=wants_object_measurements,
    )
    return ColocalizationResult(
        image=dict(image_dict),
        objects={name: np.asarray(values) for name, values in object_dict.items()},
    )


def _validated_images(
    images: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Validate every image as a nonempty 2D array of one shared shape."""
    arrays: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in images:
        array = np.ascontiguousarray(np.asarray(images[name]), dtype=np.float64)
        if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
            raise ValueError(f"image {name!r} must be a nonempty 2D array")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("all images must have the same shape")
        arrays[name] = array
    assert shape is not None
    return arrays, (shape[0], shape[1])
