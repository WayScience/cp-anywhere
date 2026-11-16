"""Object segmentation, implemented against CellProfiler 4.2.8's
``IdentifyPrimaryObjects`` module (and the shared ``Threshold`` module it
uses) as the compatibility oracle.

Declared scope: every global and adaptive threshold operation (Manual,
Otsu two/three-class, Li, Robust Background, Sauvola), log transform,
threshold smoothing, declumping/watershed (Intensity/Shape/Propagate),
hole filling (including nested-object absorption), border exclusion,
and size/count filtering -- see ``spec/segmentation-modules.md`` for
the exact settings mapping and the explicit remaining gaps (not silent
approximations).

CellProfiler's own ``Threshold`` module hard-bounds ``manual_threshold``
and ``threshold_range`` to ``[0, 1]`` and assumes images have been
rescaled to that range (its default ``Image(..., convert=True)`` divides
integer-dtype pixels by their dtype's max value; float-dtype pixels are
never rescaled). ``identify_primary_objects`` reproduces that default
behavior automatically (``rescale=True``) so passing a raw ``uint8``/
``uint16`` image works exactly like a default CellProfiler pipeline,
with no manual conversion needed.

The result chains directly into :func:`cp_anywhere.measure`::

    seg = cp_anywhere.identify_primary_objects(dna_image, threshold="otsu")
    result = cp_anywhere.measure({"DNA": dna_image}, {"Cells": seg.labels})
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_DIAMETER_DEFAULT = 0.0
_MAX_DIAMETER_DEFAULT = float("inf")
_EXPECTED_NDIM = 2
_RESCALE_DIVISORS = {
    np.dtype("uint8"): 255.0,
    np.dtype("uint16"): 65535.0,
    np.dtype("uint32"): 4294967295.0,
}


@dataclass(frozen=True)
class SegmentationResult:
    """Labels and diagnostics from one :func:`identify_primary_objects` call."""

    labels: np.ndarray
    """2D integer label array, same shape as the input image; 0 is background."""

    small_removed_labels: np.ndarray
    """Labels after the small-object filter only (large objects still
    present), matching CellProfiler's ``objects.small_removed_segmented``."""

    unedited_labels: np.ndarray
    """Labels before border, size, count, or final hole filtering."""

    object_count: int
    threshold: float
    """The threshold value actually applied, after correction/clamping,
    in the same units as the (possibly rescaled) pixels this was run on."""

    threshold_map: np.ndarray | None = None
    """Corrected 2D Adaptive threshold array; None for Global thresholding."""

    guide_threshold: float | None = None
    """Corrected global guide used to constrain Adaptive thresholds."""


def _rescale_divisor(dtype: np.dtype) -> float:
    try:
        return _RESCALE_DIVISORS[dtype]
    except KeyError as error:
        raise TypeError(
            f"rescale=True does not support dtype {dtype}; pass rescale=False "
            "and interpret threshold values in the image's own raw units, "
            "or pre-convert the image to uint8/uint16/uint32 yourself"
        ) from error


def _resolve_threshold(
    threshold: str | float | int,
    array: np.ndarray,
    *,
    rescale: bool,
    is_integer_dtype: bool,
) -> tuple[str, float | None, np.ndarray | None]:
    """Map the public ``threshold`` argument to a native method/value/
    integer_pixels triple, validating the 'otsu_integer' + rescale combo."""
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        return "manual", float(threshold), None
    if threshold == "otsu":
        return "otsu", None, None
    if threshold == "otsu_integer":
        if rescale and is_integer_dtype:
            raise ValueError(
                "threshold='otsu_integer' requires rescale=False "
                "(rescaling turns integer pixels into floats, which "
                "defeats the exact-bincount integer histogram)"
            )
        if not is_integer_dtype:
            raise TypeError("threshold='otsu_integer' requires an integer dtype image")
        return "otsu_integer", None, np.ascontiguousarray(array, dtype=np.int64)
    if threshold == "otsu_three_class":
        return "otsu_three_class", None, None
    if threshold == "li":
        return "li", None, None
    if threshold == "robust_background":
        return "robust_background", None, None
    if threshold == "sauvola":
        return "sauvola", None, None
    raise ValueError(
        "threshold must be 'otsu', 'otsu_integer', 'otsu_three_class', 'li', "
        "'robust_background', 'sauvola', or a numeric value"
    )


def identify_primary_objects(
    image: object,
    *,
    mask: object | None = None,
    threshold: str | float | int = "otsu",
    rescale: bool = True,
    threshold_correction_factor: float = 1.0,
    threshold_range: tuple[float, float] | None = None,
    exclude_size: bool = False,
    min_diameter: float = _MIN_DIAMETER_DEFAULT,
    max_diameter: float = _MAX_DIAMETER_DEFAULT,
    exclude_border_objects: bool = False,
    maximum_object_count: int | None = None,
    fill_holes: str = "never",
    threshold_smoothing_scale: float = 0.0,
    assign_middle_to_foreground: bool = True,
    robust_background_lower_outlier_fraction: float = 0.05,
    robust_background_upper_outlier_fraction: float = 0.05,
    robust_background_averaging_method: str = "mean",
    robust_background_variance_method: str = "sd",
    robust_background_number_of_deviations: float = 2.0,
    log_transform: bool = False,
    threshold_scope: str = "global",
    adaptive_window_size: int = 50,
    unclump_method: str = "none",
    watershed_method: str = "none",
    smoothing_filter_size: float | None = None,
    maxima_suppression_size: float | None = None,
    low_resolution_maxima: bool = False,
) -> SegmentationResult:
    """Segment one 2D grayscale image into labeled primary objects.

    ``rescale`` (default ``True``) matches CellProfiler's own default
    image loading: integer-dtype images (``uint8``/``uint16``/``uint32``)
    are divided by their dtype's max value before thresholding, exactly
    like CellProfiler's ``Image(..., convert=True)``; float-dtype images
    are never rescaled (also matching CellProfiler). Set ``rescale=False``
    to operate on raw pixel values directly instead (matching this
    project's measurement API's own ``convert=False`` convention).

    ``threshold`` is ``"otsu"`` (two-class, global, 256-bin histogram --
    CellProfiler's default algorithm for any already-float pixel data,
    which is what every image becomes after rescaling), ``"otsu_integer"``
    (exact one-bin-per-value histogram; only valid with ``rescale=False``
    on an integer-dtype image -- matches CellProfiler's Otsu only if
    CellProfiler would also have skipped rescaling), ``"otsu_three_class"``
    (a 128-bin, 3-class histogram split; ``assign_middle_to_foreground``
    picks the lower split, i.e. the middle class counts as foreground,
    when ``True`` (default), or the upper split when ``False``, matching
    CellProfiler's ``"Foreground"``/``"Background"`` setting), ``"li"``
    (Li's iterative Minimum Cross-Entropy threshold, matching
    ``skimage.filters.threshold_li`` with CellProfiler's own tolerance
    formula), ``"robust_background"`` (a trimmed mean/median/mode --
    ``robust_background_averaging_method`` -- plus a multiple --
    ``robust_background_number_of_deviations`` -- of the standard
    deviation or MAD -- ``robust_background_variance_method`` --
    computed after discarding ``robust_background_lower_outlier_fraction``/
    ``robust_background_upper_outlier_fraction`` of the lowest/highest
    pixels; matches ``Threshold.get_threshold_robust_background``
    exactly, including its quirk where a zero lower-fraction also
    disables the upper trim), ``"sauvola"`` (a local mean/standard-
    deviation formula over a sliding window, only valid with
    ``threshold_scope="adaptive"`` -- matches
    ``skimage.filters.threshold_sauvola``; CellProfiler's own
    ``global_operation`` Choice does not even include Sauvola, so this
    raises if combined with ``threshold_scope="global"``), or a numeric
    manual threshold value, interpreted in the same units as the
    (possibly rescaled) pixels actually compared against it.

    ``threshold_scope`` (default ``"global"``) matches
    ``Threshold.threshold_scope``: ``"global"`` applies one scalar
    threshold to the whole image; ``"adaptive"`` instead computes a
    per-pixel threshold map by splitting the image into
    ``adaptive_window_size``-sized blocks (computing a block threshold
    with the same algorithm named by ``threshold`` -- or a direct
    sliding-window Sauvola computation, which has no block/spline
    step), blending the block grid across the image with a
    tensor-product B-spline (matching
    ``scipy.interpolate.RectBivariateSpline``), and finally clamping
    the result to ``[0.7, 1.5] * guide_threshold`` (the ordinary global
    threshold, used only as a bound; for ``"sauvola"`` this guide is
    computed via ``"li"``, matching CellProfiler's own shared branch).
    ``threshold_map`` (a same-shape array) and ``guide_threshold`` on
    the returned :class:`SegmentationResult` are ``None`` unless
    ``threshold_scope="adaptive"``; ``"manual"`` and ``"otsu_integer"``
    are not supported with adaptive scope (CellProfiler's own
    ``local_operation`` Choice does not include either).

    ``log_transform`` (default ``False``) matches CellProfiler's
    ``Threshold.log_transform`` setting, meaningful only when
    ``threshold`` is ``"otsu"``, ``"otsu_three_class"``, or ``"li"``
    (CellProfiler itself only exposes this setting for those global
    operations; it is silently ignored for ``"otsu_integer"``,
    ``"robust_background"``, and numeric manual thresholds, matching
    CellProfiler's own behavior). When enabled, the threshold is
    computed on a log-transformed, ``[0, 1]``-rescaled copy of the
    *entire* image (not just the masked subset) and mapped back before
    ``threshold_correction_factor``/clamping, exactly matching
    ``centrosome.threshold.log_transform``/``inverse_log_transform``.

    ``exclude_size``/``min_diameter``/``max_diameter`` match
    CellProfiler's ``IdentifyPrimaryObjects`` size filter: objects whose
    pixel area falls outside ``pi * diameter^2 / 4`` for the given
    diameter bounds are removed.

    ``exclude_border_objects`` (``IdentifyPrimaryObjects.exclude_border_objects``)
    removes any object touching the image border (or, if ``mask`` is
    given and nothing touches the image border directly, any object
    touching the mask's own border instead).

    ``maximum_object_count`` (``IdentifyPrimaryObjects.maximum_object_count``,
    only enforced when not ``None``) erases every object if the final
    count exceeds this value, matching CellProfiler's ``LIMIT_ERASE``.

    ``fill_holes`` (``IdentifyPrimaryObjects.fill_holes``) is one of
    ``"never"`` (default), ``"after_declump_only"`` (fills background
    holes enclosed by a single object, after border/size filtering), or
    ``"after_thresholding_and_declump"`` (also fills holes in the binary
    foreground mask before labeling, limited to area < ``max_diameter^2``).
    A background region bordered by two or more *different* objects (not
    fully enclosed by one) is left unfilled, matching
    ``centrosome.cpmorphology.fill_labeled_holes`` exactly, including
    its handling of a small object fully nested inside a larger one
    (absorbed into the enclosing label) and multi-level nested holes.

    ``threshold_smoothing_scale`` (``Threshold.threshold_smoothing_scale``,
    default ``0.0`` -- no smoothing) blurs the image used for the
    foreground/background comparison (not the value used to *compute*
    the threshold, which is always the unsmoothed image) with a Gaussian
    of ``sigma = threshold_smoothing_scale / 0.6744 / 2.0``, matching
    CellProfiler's ``apply_threshold`` exactly, including its
    mask-bleed-over correction near the image border.

    ``unclump_method``/``watershed_method``
    (``IdentifyPrimaryObjects.unclump_method``/``watershed_method``)
    separate touching objects after initial labeling and before
    border/size filtering -- matching ``separate_neighboring_objects``'s
    exact position in CellProfiler's pipeline. ``unclump_method`` is
    ``"none"`` (default; no separation), ``"intensity"`` (find local
    intensity maxima in a Gaussian-smoothed copy of the image), or
    ``"shape"`` (find local maxima of the Euclidean distance transform
    of the foreground, jittered with CellProfiler's own fixed-seed(0)
    legacy-RNG noise to break exact ties). ``watershed_method`` is
    ``"none"`` (default), ``"intensity"`` (watershed on ``1 - image``),
    ``"shape"`` (watershed on the negated distance transform -- reuses
    the *same* jittered transform computed above when
    ``unclump_method="shape"`` too, matching CellProfiler's own
    ``if distance_transformed_image is None`` check), or
    ``"propagate"`` (``centrosome.propagate.propagate``, a distinct
    non-watershed algorithm; ignores both image and distance-transform
    inputs, geometric distance only). Both must be set to something
    other than ``"none"`` for separation to actually run, matching
    CellProfiler's own ``UN_NONE``/``WA_NONE`` no-op check.

    ``smoothing_filter_size``/``maxima_suppression_size`` override
    CellProfiler's own automatic formulas (``2.35 * min_diameter / 3.5``
    and ``min_diameter / 1.5`` respectively) when given explicitly,
    matching ``automatic_smoothing``/``automatic_suppression`` set to
    ``False`` with an explicit ``Float`` Setting value.
    ``low_resolution_maxima`` (``IdentifyPrimaryObjects.low_res_maxima``)
    only takes effect when ``min_diameter > 10.0`` (CellProfiler's own
    gate); when both are true, maxima are found on a downsampled copy
    of the image for speed -- not yet implemented, raises if triggered.
    """
    from cp_anywhere import _native

    array = np.asarray(image)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    rows, cols = array.shape

    is_integer_dtype = array.dtype.kind in "iu"
    if rescale and is_integer_dtype:
        divisor = _rescale_divisor(array.dtype)
        pixels = np.ascontiguousarray(array, dtype=np.float64) / divisor
    else:
        pixels = np.ascontiguousarray(array, dtype=np.float64)

    method, threshold_value, integer_pixels = _resolve_threshold(
        threshold, array, rescale=rescale, is_integer_dtype=is_integer_dtype
    )

    if threshold_range is None:
        threshold_range = (
            (0.0, 1.0) if (rescale and is_integer_dtype) else (-np.inf, np.inf)
        )

    mask_array = None
    if mask is not None:
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=bool)
        if mask_array.shape != array.shape:
            raise ValueError("mask must have the image shape")

    raw = _native.identify_primary_objects(
        pixels,
        rows,
        cols,
        method,
        threshold_value=threshold_value,
        integer_pixels=integer_pixels,
        mask=mask_array,
        threshold_correction_factor=threshold_correction_factor,
        threshold_range_min=threshold_range[0],
        threshold_range_max=threshold_range[1],
        exclude_size=exclude_size,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        exclude_border_objects=exclude_border_objects,
        maximum_object_count=maximum_object_count,
        fill_holes=fill_holes,
        threshold_smoothing_scale=threshold_smoothing_scale,
        assign_middle_to_foreground=assign_middle_to_foreground,
        robust_background_lower_outlier_fraction=robust_background_lower_outlier_fraction,
        robust_background_upper_outlier_fraction=robust_background_upper_outlier_fraction,
        robust_background_averaging_method=robust_background_averaging_method,
        robust_background_variance_method=robust_background_variance_method,
        robust_background_number_of_deviations=robust_background_number_of_deviations,
        log_transform=log_transform,
        threshold_scope=threshold_scope,
        adaptive_window_size=adaptive_window_size,
        unclump_method=unclump_method,
        watershed_method=watershed_method,
        smoothing_filter_size=smoothing_filter_size,
        maxima_suppression_size=maxima_suppression_size,
        low_resolution_maxima=low_resolution_maxima,
    )
    return SegmentationResult(
        labels=raw["labels"].reshape(array.shape),
        small_removed_labels=raw["small_removed_labels"].reshape(array.shape),
        unedited_labels=raw["unedited_labels"].reshape(array.shape),
        object_count=int(raw["object_count"]),
        threshold=float(raw["threshold"]),
        threshold_map=(
            None
            if raw["threshold_map"] is None
            else raw["threshold_map"].reshape(array.shape)
        ),
        guide_threshold=raw["guide_threshold"],
    )


@dataclass(frozen=True)
class SecondaryObjectsResult:
    """Final and pre-filter secondary-object labels."""

    small_removed_labels: np.ndarray
    """Pre-filter labels in the original seed numbering, used by neighbors."""

    labels: np.ndarray
    """2D integer label array, same shape as the input image; 0 is background."""

    threshold: float | None
    """The secondary image's own applied threshold; ``None`` for
    ``method="distance_n"``, which never thresholds an image."""

    @property
    def unedited_labels(self) -> np.ndarray:
        """CellProfiler stores the same grown layer as unedited and small-removed."""
        return self.small_removed_labels


def identify_secondary_objects(
    image: object,
    primary_labels: object,
    *,
    primary_labels_in: object | None = None,
    method: str = "propagation",
    mask: object | None = None,
    threshold: str | float | int = "otsu",
    rescale: bool = True,
    threshold_correction_factor: float = 1.0,
    threshold_range: tuple[float, float] | None = None,
    threshold_smoothing_scale: float = 0.0,
    log_transform: bool = False,
    threshold_scope: str = "global",
    adaptive_window_size: int = 50,
    assign_middle_to_foreground: bool = True,
    robust_background_lower_outlier_fraction: float = 0.05,
    robust_background_upper_outlier_fraction: float = 0.05,
    robust_background_averaging_method: str = "mean",
    robust_background_variance_method: str = "sd",
    robust_background_number_of_deviations: float = 2.0,
    regularization_factor: float = 0.05,
    distance_to_dilate: float = 10.0,
    fill_holes: bool = False,
) -> SecondaryObjectsResult:
    """Grow a secondary object (e.g. ``"Cells"``) around each primary
    object (e.g. ``"Nuclei"``), matching CellProfiler 4.2.8's
    ``IdentifySecondaryObjects`` module.

    ``image`` is the secondary stain image (e.g. a cell-membrane or
    whole-cell marker); ``primary_labels`` is the final, kept primary
    object set (e.g. :attr:`SegmentationResult.labels` from
    :func:`identify_primary_objects`) -- a grown secondary object is
    kept only if its seed primary object survived there.

    ``primary_labels_in`` accepts the unedited primary labels, before
    size or border filtering. Pass :attr:`SegmentationResult.unedited_labels`
    from the upstream primary result. The Rust core retains edge-touching
    seeds and removes other seed pixels wherever ``primary_labels`` is
    zero, matching CellProfiler. Grown regions are mapped to retained
    primary IDs by overlap, so renumbered primaries stay aligned.
    If omitted, the kept primary array supplies the seeds; excluded
    edge objects cannot participate because their pixels are unavailable.

    ``method`` is ``"propagation"`` (the module's own default:
    weighted shortest-path region growing, bounded by a threshold of
    ``image``, steered by ``regularization_factor`` -- 0 favors image
    intensity, larger values favor Euclidean geometry) or
    ``"distance_n"`` (a fixed-radius Euclidean expansion bounded by
    ``distance_to_dilate``, with no image or threshold involved at
    all). ``"distance_b"``, ``"watershed_gradient"``, and
    ``"watershed_image"`` are explicit, not-yet-implemented gaps.

    ``threshold``/``rescale``/``threshold_correction_factor``/
    ``threshold_range``/``threshold_smoothing_scale``/``log_transform``/
    ``threshold_scope``/``adaptive_window_size``/
    ``assign_middle_to_foreground``/``robust_background_*`` all match
    :func:`identify_primary_objects`'s own identically-named
    parameters exactly (the secondary image's threshold uses the same
    machinery, with no size/border/declump filtering applied to it --
    matching ``IdentifySecondaryObjects``'s own plain threshold-and-
    binarize step). Ignored entirely for ``method="distance_n"``.

    ``fill_holes`` uses CellProfiler's labeled-hole graph, but changes
    only background pixels. Existing nested foreground labels survive.
    ``small_removed_labels`` retains the grown labels before filtering,
    in the original seed numbering, for downstream neighbor measurements.
    """
    from cp_anywhere import _native

    array = np.asarray(image)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    rows, cols = array.shape

    primary_array = _label_array(primary_labels, array.shape)
    labels_in_array = (
        primary_array
        if primary_labels_in is None
        else _label_array(primary_labels_in, array.shape)
    )

    if method not in ("propagation", "distance_n"):
        raise ValueError(
            "method must be 'propagation' or 'distance_n' "
            "('distance_b', 'watershed_gradient', and 'watershed_image' "
            "are explicit, not-yet-implemented gaps)"
        )

    is_integer_dtype = array.dtype.kind in "iu"
    if rescale and is_integer_dtype:
        divisor = _rescale_divisor(array.dtype)
        pixels = np.ascontiguousarray(array, dtype=np.float64) / divisor
    else:
        pixels = np.ascontiguousarray(array, dtype=np.float64)

    threshold_method, threshold_value, integer_pixels = _resolve_threshold(
        threshold, array, rescale=rescale, is_integer_dtype=is_integer_dtype
    )
    if threshold_range is None:
        threshold_range = (
            (0.0, 1.0) if (rescale and is_integer_dtype) else (-np.inf, np.inf)
        )

    mask_array = None
    if mask is not None:
        mask_array = np.ascontiguousarray(np.asarray(mask), dtype=bool)
        if mask_array.shape != array.shape:
            raise ValueError("mask must have the image shape")

    raw = _native.identify_secondary_objects(
        pixels,
        rows,
        cols,
        labels_in_array,
        primary_array,
        method,
        threshold_method,
        threshold_value=threshold_value,
        integer_pixels=integer_pixels,
        mask=mask_array,
        threshold_correction_factor=threshold_correction_factor,
        threshold_range_min=threshold_range[0],
        threshold_range_max=threshold_range[1],
        threshold_smoothing_scale=threshold_smoothing_scale,
        log_transform=log_transform,
        threshold_scope=threshold_scope,
        adaptive_window_size=adaptive_window_size,
        assign_middle_to_foreground=assign_middle_to_foreground,
        robust_background_lower_outlier_fraction=robust_background_lower_outlier_fraction,
        robust_background_upper_outlier_fraction=robust_background_upper_outlier_fraction,
        robust_background_averaging_method=robust_background_averaging_method,
        robust_background_variance_method=robust_background_variance_method,
        robust_background_number_of_deviations=robust_background_number_of_deviations,
        regularization_factor=regularization_factor,
        distance_to_dilate=distance_to_dilate,
        fill_holes=fill_holes,
    )
    return SecondaryObjectsResult(
        labels=raw["labels"].reshape(array.shape),
        small_removed_labels=raw["small_removed_labels"].reshape(array.shape),
        threshold=raw["threshold"],
    )


@dataclass(frozen=True)
class TertiaryObjectsResult:
    """Tertiary labels, preserving the larger object's label numbering."""

    labels: np.ndarray


def identify_tertiary_objects(
    primary_labels: object,
    secondary_labels: object,
    *,
    shrink_primary: bool = True,
) -> TertiaryObjectsResult:
    """Subtract smaller objects from larger objects on the same 2D grid.

    With ``shrink_primary=True`` (CellProfiler's default), retain the
    smaller objects' one-pixel, eight-connected outline. With ``False``,
    remove every smaller-object pixel. Output IDs are the larger
    objects' IDs, not newly numbered connected components. Inputs are
    never modified. Crop/resize metadata and parent measurements are
    not provided by this array API; grids must already match.
    """
    from cp_anywhere import _native

    primary = _tertiary_label_array(primary_labels)
    secondary = _tertiary_label_array(secondary_labels)
    if primary.shape != secondary.shape:
        raise ValueError("primary and secondary label arrays must have matching shapes")
    rows, cols = secondary.shape
    labels = _native.identify_tertiary_objects(
        primary, secondary, rows, cols, shrink_primary=shrink_primary
    )
    return TertiaryObjectsResult(labels=labels.reshape(secondary.shape))


def _tertiary_label_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("label arrays must be nonempty 2D arrays")
    if array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("label arrays must contain nonnegative integers")
    if np.any(array > np.iinfo(np.uint32).max):
        raise ValueError("label arrays must fit in uint32")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _label_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.uint32)
    if array.shape != shape:
        raise ValueError("label arrays must have the image shape")
    return array
