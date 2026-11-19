"""Radial intensity distribution, implemented against CellProfiler 4.2.8's
``MeasureObjectIntensityDistribution`` module as the compatibility oracle.

Declared scope: 2D images, the module's default radial center ("These
objects" -- each object's own point farthest from its edge). The "Centers
of other objects" and "Edges of other objects" center choices need a
second, separately selected object set and are an explicit, documented
gap -- see ``crates/cp_anywhere/src/intensity_distribution.rs``'s own
doc comment for the exact algorithm this reproduces (geodesic distance
from each object's own farthest-from-edge seed, binned into concentric
rings, with optional Zernike intensity moments) and
``spec/cppipe-compatibility.md`` for the settings mapping.

    result = cp_anywhere.measure_intensity_distribution(
        dna_image, cell_labels, image_name="DNA",
    )
    result.columns["RadialDistribution_FracAtD_DNA_1of4"]

Every returned column is named exactly like real CellProfiler's own
``RadialDistribution_<feature>_<image_name>_<suffix>`` feature, one value
per object in ascending 1-based object ID order. A label image with no
objects at all reports every selected column as an empty array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_EXPECTED_NDIM = 2
_ZernikeOption = Literal["none", "magnitudes", "magnitudes_and_phase"]


@dataclass(frozen=True)
class IntensityDistributionResult:
    """Every selected ``RadialDistribution_*`` column, keyed by its exact
    real CellProfiler feature name."""

    columns: dict[str, np.ndarray]


def measure_intensity_distribution(
    image: object,
    labels: object,
    *,
    image_name: str,
    bin_count: int = 4,
    wants_scaled: bool = True,
    maximum_radius: float = 100.0,
    zernikes: _ZernikeOption = "none",
    zernike_degree: int = 9,
) -> IntensityDistributionResult:
    """Run CellProfiler's 2D radial intensity distribution on one image
    against one object set.

    ``bin_count``/``wants_scaled``/``maximum_radius`` are CellProfiler's
    own "Number of bins"/"Scale the bins?"/"Maximum radius" settings:
    with ``wants_scaled=True`` (the default) each object's own
    center-to-edge span is normalized to ``[0, 1)``; with ``False``,
    ``maximum_radius`` pixels make up one bin and an ``Overflow`` column
    reports whatever falls beyond ``bin_count`` bins. ``zernikes``
    selects which ``ZernikeMagnitude``/``ZernikePhase`` columns (if any)
    to add, up to and including radial degree ``zernike_degree``.
    """
    from cp_anywhere import _native

    array = np.ascontiguousarray(np.asarray(image), dtype=np.float64)
    if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
        raise ValueError("image must be a nonempty 2D array")
    label_array = np.ascontiguousarray(np.asarray(labels), dtype=np.uint32)
    if label_array.shape != array.shape:
        raise ValueError("labels must have the image shape")
    rows, cols = array.shape
    raw = _native.measure_intensity_distribution(
        array,
        label_array,
        rows,
        cols,
        image_name,
        bin_count=bin_count,
        wants_scaled=wants_scaled,
        maximum_radius=maximum_radius,
        zernikes=zernikes,
        zernike_degree=zernike_degree,
    )
    return IntensityDistributionResult(
        columns={name: np.asarray(values) for name, values in raw.items()}
    )
