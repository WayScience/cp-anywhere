"""Image/object granularity, implemented against CellProfiler 4.2.8's
``MeasureGranularity`` module as the compatibility oracle.

Declared scope: 2D images only. Real CellProfiler's own module divides by
``(dimension - 1)`` while rescaling its working image back to full
resolution, and raises ``ZeroDivisionError`` for any image with a 1-row or
1-column dimension; :func:`measure_granularity` raises a clear
:class:`ValueError` for that case instead of silently producing 0/NaN --
see ``fixtures/granularity/`` and ``crates/cp_anywhere/src/granularity.rs``
for the oracle-verified numerics (bilinear and cubic-spline resampling,
grayscale erosion/dilation, and morphological reconstruction all
independently matched against the pinned reference)::

    result = cp_anywhere.measure_granularity(
        dna_image, objects={"Cells": cell_labels},
    )
    result.image[0]           # GS1, the image-level granularity spectrum
    result.objects["Cells"][0]  # GS1, one value per Cells object

An object with zero pixels under the mask reports ``NaN`` at every scale,
matching real CellProfiler (its own per-object baseline mean is already
``NaN``, and ``NaN`` never recovers through the per-scale subtraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_EXPECTED_NDIM = 2


@dataclass(frozen=True)
class GranularityResult:
    """Image-level and per-object-set granularity spectra."""

    image: np.ndarray
    """``image[i]`` is the image-level ``GSi+1`` value."""

    objects: dict[str, np.ndarray]
    """``objects[name][i]`` is one value per object (ascending 1-based ID)
    at scale ``i + 1``; an empty array for an object set with no objects."""


def measure_granularity(
    image: object,
    *,
    objects: Mapping[str, object] | None = None,
    mask: object | None = None,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,
    _float32: bool | None = None,
) -> GranularityResult:
    """Run CellProfiler's 2D granularity spectrum on one image.

    ``subsample_size``/``image_sample_size`` are CellProfiler's own
    "Subsampling factor for granularity/background reduction" settings
    (``(0, 1]``; ``1.0`` disables that resampling step).
    ``element_size`` is the structuring element radius in pixels of the
    (possibly subsampled) working image. ``granular_spectrum_length`` is
    the number of erosion/reconstruction scales to report.
    """
    from cp_anywhere import _native

    source = np.asarray(image)
    float32 = source.dtype == np.float32 if _float32 is None else _float32
    array = np.ascontiguousarray(source, dtype=np.float64)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    rows, cols = array.shape
    if rows == 1 or cols == 1:
        raise ValueError(
            "MeasureGranularity on a 1-row or 1-column image divides by "
            "(dimension - 1) in real CellProfiler and raises "
            "ZeroDivisionError there; this is a real CellProfiler "
            "limitation, not something cp-anywhere papers over"
        )
    mask_array = None
    if mask is not None:
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=np.bool_)
        if mask_array.shape != array.shape:
            raise ValueError("mask must have the image shape")
    object_arrays = {
        name: _label_array(value, array.shape)
        for name, value in (objects or {}).items()
    }
    raw = _native.measure_granularity(
        array,
        rows,
        cols,
        object_arrays,
        mask=mask_array,
        subsample_size=subsample_size,
        image_sample_size=image_sample_size,
        element_size=element_size,
        granular_spectrum_length=granular_spectrum_length,
        float32=float32,
    )
    return GranularityResult(
        image=np.asarray(raw["image"]),
        objects={name: np.asarray(scales) for name, scales in raw["objects"].items()},
    )


def _label_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.uint32)
    if array.shape != shape:
        raise ValueError("object label arrays must have the image shape")
    return array
