"""Type stub for the NumPy native measurement binding."""

from typing import Any

import numpy as np

def measure(
    shape: tuple[int, ...],
    images: dict[str, np.ndarray],
    objects: dict[str, np.ndarray],
    mask: np.ndarray | None,
    image_id: str,
    families: tuple[str, ...],
) -> dict[str, Any]: ...
def identify_primary_objects(
    pixels: np.ndarray,
    rows: int,
    cols: int,
    threshold_method: str,
    threshold_value: float | None = ...,
    integer_pixels: np.ndarray | None = ...,
    mask: np.ndarray | None = ...,
    threshold_correction_factor: float = ...,
    threshold_range_min: float = ...,
    threshold_range_max: float = ...,
    exclude_size: bool = ...,
    min_diameter: float = ...,
    max_diameter: float = ...,
    exclude_border_objects: bool = ...,
    maximum_object_count: int | None = ...,
    fill_holes: str | None = ...,
    threshold_smoothing_scale: float = ...,
    assign_middle_to_foreground: bool | None = ...,
    robust_background_lower_outlier_fraction: float = ...,
    robust_background_upper_outlier_fraction: float = ...,
    robust_background_averaging_method: str | None = ...,
    robust_background_variance_method: str | None = ...,
    robust_background_number_of_deviations: float = ...,
    log_transform: bool = ...,
    threshold_scope: str | None = ...,
    adaptive_window_size: int = ...,
    unclump_method: str | None = ...,
    watershed_method: str | None = ...,
    smoothing_filter_size: float | None = ...,
    maxima_suppression_size: float | None = ...,
    low_resolution_maxima: bool = ...,
) -> dict[str, Any]: ...
