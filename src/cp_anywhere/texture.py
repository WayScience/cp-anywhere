"""Image/object Haralick texture features, implemented against
CellProfiler 4.2.8's ``MeasureTexture`` module as the compatibility
oracle.

Declared scope: 2D images only. Real CellProfiler's module also supports
3D volumetric correspondence vectors (13 directions instead of 4); that
case is an explicit, documented gap -- see
``crates/cp_anywhere_core/src/texture.rs`` for the oracle-verified 2D
numerics (``skimage.util.img_as_ubyte``'s exact rounding,
``skimage.exposure.rescale_intensity``'s division-before-multiply
order, and ``mahotas.features.haralick``'s own co-occurrence-matrix and
13-feature formulas, all independently matched against the pinned
reference)::

    result = cp_anywhere.measure_texture(
        image, cell_labels, image_name="DNA", scales=[3],
    )
    result.image["Texture_Contrast_DNA_3_00_256"]       # a float
    result.objects["Texture_Contrast_DNA_3_00_256"]      # one value per object

A real CellProfiler quirk this port replicates exactly, not "fixes": a
per-object co-occurrence matrix that comes out empty (e.g. an object
smaller than the requested ``scale``) reports ``0.0`` for every feature
in that direction, not ``NaN`` -- matching real CellProfiler's own
``result[~numpy.isfinite(result)] = 0`` step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_EXPECTED_NDIM = 2
_MIN_GRAY_LEVELS = 2
_MAX_GRAY_LEVELS = 256


@dataclass(frozen=True)
class TextureResult:
    """Image-level and per-object Haralick texture measurements, keyed by
    the real ``Texture_{Feature}_{image_name}_{scale}_{direction:02}_
    {gray_levels}`` column name CellProfiler itself uses."""

    image: dict[str, float]
    """Present only when ``wants_image_measurements`` is set."""

    objects: dict[str, np.ndarray]
    """``objects[column][i]`` is one value per object (ascending label
    ID); present only when ``wants_object_measurements`` is set."""


def measure_texture(
    image: object,
    labels: object,
    *,
    image_name: str = "Input",
    gray_levels: int = 256,
    scales: Sequence[int] = (3,),
    wants_image_measurements: bool = True,
    wants_object_measurements: bool = True,
) -> TextureResult:
    """Run CellProfiler's 2D Haralick texture measurements on one image.

    ``gray_levels`` is CellProfiler's own "Enter how many gray levels to
    measure the texture at" setting (``2..=256``). ``scales`` are
    "Texture scale to measure" settings, one measurement set per scale.
    """
    from cp_anywhere import _native

    array = np.ascontiguousarray(np.asarray(image), dtype=np.float64)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    labels_array = np.ascontiguousarray(np.asarray(labels), dtype=np.uint32)
    if labels_array.shape != array.shape:
        raise ValueError("labels must have the image shape")
    if not image_name:
        raise ValueError("image_name must not be empty")
    if not _MIN_GRAY_LEVELS <= gray_levels <= _MAX_GRAY_LEVELS:
        raise ValueError("gray_levels must be between 2 and 256")
    scales_list = [int(scale) for scale in scales]
    if not scales_list:
        raise ValueError("scales must not be empty")
    rows, cols = array.shape
    image_dict, object_dict = _native.measure_texture(
        array,
        labels_array,
        rows,
        cols,
        image_name,
        gray_levels,
        scales_list,
        wants_image_measurements,
        wants_object_measurements,
    )
    return TextureResult(
        image=dict(image_dict),
        objects={name: np.asarray(values) for name, values in object_dict.items()},
    )
