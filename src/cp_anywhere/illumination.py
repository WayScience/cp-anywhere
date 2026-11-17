"""Illumination correction, implemented against CellProfiler 4.2.8's
``CorrectIlluminationCalculate``/``CorrectIlluminationApply`` modules as
the compatibility oracle.

Declared scope so far: 2D images only. ``CorrectIlluminationCalculate``
covers both ``"Each"`` mode (one image at a time,
:func:`calculate_illumination`) and ``"All"`` mode (accumulated across
many image sets, :func:`calculate_illumination_across`), and 5 of its 7
smoothing methods (``No smoothing``, ``Median Filter``,
``Gaussian Filter``, ``Fit Polynomial``, ``Smooth to Average``).
``Convex Hull`` and ``Splines`` (a multi-pass geometric transform and
an iterative 2D spline background fit, respectively) raise a clear
:class:`ValueError` naming them as not yet implemented.
``CorrectIlluminationApply`` (:func:`apply_illumination`) supports
same-shaped arrays with divide/subtract and output clipping.
See ``spec/illumination.md`` and
``crates/cp_anywhere_core/src/illumination.rs`` for the oracle-verified
numerics (confirmed against every real-fixture case in
``fixtures/illumination_calculate/``, ``fixtures/illumination_calculate_
across/``, and ``fixtures/illumination_apply/``, including the
``float32`` round-trip precision contract real CellProfiler's own
``Image`` construction introduces at each pipeline stage)::

    function = cp_anywhere.calculate_illumination(
        dna_image, smoothing_method="Gaussian Filter",
    )
    corrected = cp_anywhere.apply_illumination(dna_image, function)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_EXPECTED_NDIM = 2


def calculate_illumination(
    image: object,
    *,
    mask: object | None = None,
    intensity_choice: str = "Regular",
    block_size: int = 60,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    smoothing_method: str = "Gaussian Filter",
    automatic_object_width: str = "Automatic",
    object_width: float = 10.0,
    size_of_smoothing_filter: float = 10.0,
    rescale_option: str = "Yes",
) -> np.ndarray:
    """Calculate a ``CorrectIlluminationCalculate`` illumination function
    for one image ("Each" mode).
    Use :func:`calculate_illumination_across` for a shared function
    calculated from a sequence of images.

    ``intensity_choice`` is ``"Regular"`` or ``"Background"``.
    ``smoothing_method`` is one of ``"No smoothing"``,
    ``"Median Filter"``, ``"Gaussian Filter"``, ``"Fit Polynomial"``,
    ``"Smooth to Average"`` (``"Convex Hull"`` and ``"Splines"`` raise
    a clear :class:`ValueError` naming them as not yet implemented).
    ``automatic_object_width`` is ``"Automatic"``, ``"Object size"``,
    or ``"Manually"`` (CellProfiler's own ``FI_*`` constants).
    ``rescale_option`` is ``"Yes"``, ``"No"``, or ``"Median"``.
    """
    from cp_anywhere import _native

    array = np.ascontiguousarray(np.asarray(image), dtype=np.float64)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    rows, cols = array.shape
    mask_array = None
    if mask is not None:
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=np.bool_)
        if mask_array.shape != array.shape:
            raise ValueError("mask must have the image shape")
    result = _native.calculate_illumination_each(
        array.ravel(),
        rows,
        cols,
        mask=None if mask_array is None else mask_array.ravel(),
        intensity_choice=intensity_choice,
        block_size=block_size,
        dilate_objects=dilate_objects,
        object_dilation_radius=object_dilation_radius,
        smoothing_method=smoothing_method,
        automatic_object_width=automatic_object_width,
        object_width=object_width,
        size_of_smoothing_filter=size_of_smoothing_filter,
        rescale_option=rescale_option,
    )
    return np.asarray(result).reshape(rows, cols)


def calculate_illumination_across(
    images: Sequence[object],
    *,
    masks: Sequence[object | None] | None = None,
    intensity_choice: str = "Regular",
    block_size: int = 60,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    smoothing_method: str = "Gaussian Filter",
    automatic_object_width: str = "Automatic",
    object_width: float = 10.0,
    size_of_smoothing_filter: float = 10.0,
    rescale_option: str = "Yes",
) -> np.ndarray:
    """Calculate a ``CorrectIlluminationCalculate`` illumination
    function accumulated across many image sets ("All: First cycle" /
    "All: Across cycles" mode): every image's own contribution
    (zeroed-mask ``"Regular"`` pixels, or a per-block minimum for
    ``"Background"``) is averaged pixel-by-pixel (weighted by how many
    images had an unmasked pixel there), then the same dilation,
    smoothing, and rescaling as :func:`calculate_illumination` runs
    once on that average.

    ``images`` must all share one shape; ``masks[i]`` (when given)
    applies to ``images[i]`` and may be ``None`` for an unmasked image
    in an otherwise-masked group, matching real CellProfiler pipelines
    that mix masked and unmasked image sets. See
    :func:`calculate_illumination` for every other parameter.
    """
    from cp_anywhere import _native

    if not images:
        raise ValueError("calculate_illumination_across needs at least one image")
    arrays = [np.ascontiguousarray(np.asarray(img), dtype=np.float64) for img in images]
    shape = arrays[0].shape
    if len(shape) != _EXPECTED_NDIM or 0 in shape:
        raise ValueError("every image must be a nonempty 2D array")
    if any(a.shape != shape for a in arrays):
        raise ValueError("every image must have the same shape")
    resolved_masks: Sequence[object | None] = (
        masks if masks is not None else [None] * len(arrays)
    )
    if len(resolved_masks) != len(arrays):
        raise ValueError("masks must have one entry per image (use None for no mask)")
    mask_arrays: list[np.ndarray | None] = []
    for mask in resolved_masks:
        if mask is None:
            mask_arrays.append(None)
            continue
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=np.bool_)
        if mask_array.shape != shape:
            raise ValueError("every mask must have the image shape")
        mask_arrays.append(mask_array)
    rows, cols = shape
    pairs = [
        (array.ravel(), None if mask is None else mask.ravel())
        for array, mask in zip(arrays, mask_arrays, strict=True)
    ]
    result = _native.calculate_illumination_across(
        pairs,
        rows,
        cols,
        intensity_choice=intensity_choice,
        block_size=block_size,
        dilate_objects=dilate_objects,
        object_dilation_radius=object_dilation_radius,
        smoothing_method=smoothing_method,
        automatic_object_width=automatic_object_width,
        object_width=object_width,
        size_of_smoothing_filter=size_of_smoothing_filter,
        rescale_option=rescale_option,
    )
    return np.asarray(result).reshape(rows, cols)


def apply_illumination(
    image: object,
    illumination_function: object,
    *,
    divide_or_subtract: str = "Divide",
    truncate_low: bool = True,
    truncate_high: bool = True,
) -> np.ndarray:
    """Apply a ``CorrectIlluminationCalculate`` illumination function to
    one image (``CorrectIlluminationApply``).

    ``divide_or_subtract`` is ``"Divide"`` (illumination calculated
    with ``"Regular"``) or ``"Subtract"`` (illumination calculated
    with ``"Background"``); ``"Subtract"`` always clips negative
    results to ``0`` regardless of ``truncate_low``, matching real
    CellProfiler's own unconditional clip inside that branch.
    ``truncate_low``/``truncate_high`` are CellProfiler's own
    "Set output image values less/greater than 0/1 equal to 0/1?"
    settings, applied after the divide/subtract.
    """
    array = np.ascontiguousarray(np.asarray(image), dtype=np.float64)
    illum = np.ascontiguousarray(np.asarray(illumination_function), dtype=np.float64)
    if array.shape != illum.shape:
        raise ValueError(
            "image and illumination_function must have equal dimensions "
            f"({array.shape} vs {illum.shape})"
        )
    if divide_or_subtract == "Divide":
        output = array / illum
    elif divide_or_subtract == "Subtract":
        output = array - illum
        output = np.where(output < 0, 0, output)
    else:
        raise ValueError(
            "divide_or_subtract must be 'Divide' or 'Subtract', "
            f"got {divide_or_subtract!r}"
        )
    if truncate_low:
        output = np.where(output < 0, 0, output)
    if truncate_high:
        output = np.where(output > 1, 1, output)
    # Matches real CellProfiler's own `Image(output_pixels,
    # parent_image=orig_image)` construction, which always downcasts
    # to float32 (see illumination.rs's module docs for the same
    # contract in CorrectIlluminationCalculate).
    return output.astype(np.float32).astype(np.float64)
