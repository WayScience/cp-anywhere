"""Stage 0: an independently usable image-quality gate.

This measures basic technical acquisition quality -- focus, saturation/
clipping, and robust per-channel intensity statistics -- and returns a
``pass``/``review``/``fail`` decision per image. **This is not a cell
segmentation or biological-phenotype classifier**; it runs before any
segmentation and does not require object masks.

The ``ImageQuality_*`` metrics reuse this project's own
``MeasureImageQuality`` numeric kernel (see
:mod:`cp_anywhere._native.measure_image_quality` and
``crates/cp_anywhere/src/image_quality.rs``), the same kernel already
verified against real CellProfiler 4.2.8. A :class:`QcProfile` is
*calibration evidence*, not an invented universal standard: build one
from a pinned, published pipeline's own ``MeasureImageQuality``/
``FlagImage`` pair with :meth:`QcProfile.from_cppipe`, or supply
explicit per-channel bounds you calibrated yourself with
:meth:`QcProfile.from_criteria`. This module ships no built-in numeric
thresholds of its own.

The ``Intensity_*`` metrics (median, robust dispersion, extreme
quantiles, a simple background estimate, and the near-zero pixel
fraction) are new to this module -- they are not CellProfiler
measurements and carry no CellProfiler-parity claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from .cppipe import CppipeError, FlagCriterion, FlagImageStep, ImageFlag, load_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

__all__ = [
    "ImageQualityReport",
    "QcCriterion",
    "QcProfile",
    "QualityGateResult",
    "Reason",
    "Status",
    "assess",
    "assess_image",
]

_EXPECTED_NDIM = 2
_NEAR_ZERO_FRACTION = 0.01  # of the channel's own finite range, not a literature cutoff


class Status(IntEnum):
    """Ordered worst-wins: combining reasons keeps the highest status value."""

    PASS = 0
    REVIEW = 1
    FAIL = 2


@dataclass(frozen=True)
class Reason:
    """One decision input, not a complete explanation on its own."""

    kind: str
    """``\"structural\"`` (hard requirement), ``\"calibrated\"`` (a configured
    bound was outside range), or ``\"uncalibrated\"`` (the profile could not
    be evaluated, e.g. a required channel is missing)."""

    detail: str


@dataclass(frozen=True)
class QcCriterion:
    """One per-channel bound on a ``MeasureImageQuality`` metric name.

    ``metric`` is the bare metric name CellProfiler's ``MeasureImageQuality``
    uses (``\"PowerLogLogSlope\"``, ``\"FocusScore\"``, ``\"PercentMaximal\"``,
    ``\"PercentMinimal\"``, ``\"LocalFocusScore_<scale>\"``,
    ``\"Correlation_<scale>\"``), not the full ``ImageQuality_*_<channel>`` key.
    """

    channel: str
    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("QcCriterion needs a minimum, a maximum, or both")


@dataclass(frozen=True)
class QcProfile:
    """A versioned, documented set of per-channel quality bounds.

    Construct with :meth:`from_cppipe` or :meth:`from_criteria`, not
    directly -- the internal ``flag`` representation is an implementation
    detail shared with :class:`cp_anywhere.FlagImageStep`.
    """

    name: str
    version: str
    blur_scale: int
    flag: ImageFlag
    source: str = ""

    @classmethod
    def from_cppipe(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        version: str = "unpinned",
        source: str = "",
    ) -> QcProfile:
        """Build a profile from a real pipeline's calibrated thresholds.

        The pipeline must contain exactly one ``MeasureImageQuality`` step
        and exactly one ``FlagImage`` step with exactly one flag -- i.e.
        the same shape as a published QC-gate prefix (see
        ``scripts/validation/pedcp90_qc.md`` for one real example).
        """
        pipeline = load_pipeline(path)
        if len(pipeline.image_quality_steps) != 1:
            raise ValueError(
                "QcProfile.from_cppipe requires exactly one MeasureImageQuality step"
            )
        if (
            len(pipeline.flag_image_steps) != 1
            or len(pipeline.flag_image_steps[0].flags) != 1
        ):
            raise ValueError(
                "QcProfile.from_cppipe requires exactly one FlagImage flag"
            )
        quality_step = pipeline.image_quality_steps[0]
        (flag,) = pipeline.flag_image_steps[0].flags
        scales = quality_step.groups[0].blur_scales if quality_step.groups else ()
        return cls(
            name=name or str(path),
            version=version,
            blur_scale=scales[0] if scales else 0,
            flag=flag,
            source=source or str(path),
        )

    @classmethod
    def from_criteria(
        cls,
        *,
        name: str,
        version: str,
        criteria: Iterable[QcCriterion],
        combination: str = "any",
        blur_scale: int = 20,
        source: str = "",
    ) -> QcProfile:
        """Build a profile from explicit, caller-calibrated bounds.

        ``combination`` is ``\"any\"`` (fail if any criterion is outside its
        bound, CellProfiler's own ``FlagImage`` default) or ``\"all\"``.
        """
        if combination not in ("any", "all"):
            raise ValueError("combination must be 'any' or 'all'")
        criteria = tuple(criteria)
        if not criteria:
            raise ValueError("QcProfile.from_criteria requires at least one criterion")
        flag_criteria = tuple(
            FlagCriterion(f"ImageQuality_{c.metric}_{c.channel}", c.minimum, c.maximum)
            for c in criteria
        )
        flag = ImageFlag(
            name=f"{name}_QCFlag",
            all=combination == "all",
            skip=True,
            criteria=flag_criteria,
        )
        return cls(
            name=name, version=version, blur_scale=blur_scale, flag=flag, source=source
        )


@dataclass(frozen=True)
class ImageQualityReport:
    """One image's Stage 0 decision; preserved even when status is ``fail``."""

    image_id: str
    status: Status
    metrics: dict[str, float]
    criteria_failed: dict[str, bool]
    reasons: tuple[Reason, ...] = ()

    def row(self) -> dict[str, object]:
        """A flat record suitable for one row of a QC table."""
        return {
            "image_id": self.image_id,
            "status": self.status.name.lower(),
            "metrics": self.metrics,
            "criteria_failed": self.criteria_failed,
            "reasons": [{"kind": r.kind, "detail": r.detail} for r in self.reasons],
        }


def _structural_reasons(images: Mapping[str, object]) -> list[Reason]:
    if not images:
        return [Reason("structural", "no channels supplied")]
    reasons = []
    shape: tuple[int, ...] | None = None
    for name, value in images.items():
        array = np.asarray(value)
        if array.ndim != _EXPECTED_NDIM or 0 in array.shape:
            reasons.append(
                Reason("structural", f"channel {name!r} must be a nonempty 2D array")
            )
            continue
        if array.dtype.kind not in "iuf":
            reasons.append(
                Reason(
                    "structural",
                    f"channel {name!r} has unsupported dtype {array.dtype}",
                )
            )
            continue
        if not np.isfinite(array).all():
            reasons.append(
                Reason("structural", f"channel {name!r} has non-finite pixels")
            )
            continue
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            reasons.append(
                Reason(
                    "structural",
                    f"channel {name!r} shape {array.shape} does not match {shape}",
                )
            )
    return reasons


def _intensity_statistics(name: str, pixels: np.ndarray) -> dict[str, float]:
    """New, non-CellProfiler robust per-channel intensity statistics."""
    flat = pixels.ravel().astype(np.float64, copy=False)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    low, high = (float(q) for q in np.quantile(flat, (0.01, 0.99)))
    span = high - low if high > low else 1.0
    near_zero = float(np.mean(flat - low <= _NEAR_ZERO_FRACTION * span))
    return {
        f"Intensity_Median_{name}": median,
        f"Intensity_MAD_{name}": mad,
        f"Intensity_P01_{name}": low,
        f"Intensity_P99_{name}": high,
        f"Intensity_Background_{name}": low,
        f"Intensity_FractionNearZero_{name}": near_zero,
    }


def _channel_metrics(name: str, array: np.ndarray, blur_scale: int) -> dict[str, float]:
    from cp_anywhere import _native

    pixels = np.ascontiguousarray(array, dtype=np.float64)
    metrics = _native.measure_image_quality(
        pixels,
        name,
        [blur_scale] if blur_scale else [],
        True,  # saturation
        False,  # include_scaling: Stage 0 has no acquisition-scale input
        None,
        array.dtype == np.float32,
    )
    metrics.update(_intensity_statistics(name, pixels))
    return metrics


def assess_image(
    images: Mapping[str, object], profile: QcProfile, *, image_id: str = ""
) -> ImageQualityReport:
    """Assess one field's channels against ``profile``.

    ``images`` maps channel name to a 2D array; arrays are never modified.
    """
    structural = _structural_reasons(images)
    if structural:
        return ImageQualityReport(image_id, Status.FAIL, {}, {}, tuple(structural))

    metrics: dict[str, float] = {}
    for name, value in images.items():
        metrics.update(_channel_metrics(name, np.asarray(value), profile.blur_scale))

    try:
        results, skip = FlagImageStep((profile.flag,)).evaluate(metrics)
    except CppipeError as error:
        return ImageQualityReport(
            image_id, Status.REVIEW, metrics, {}, (Reason("uncalibrated", str(error)),)
        )

    failed = {name: bool(value) for name, value in results.items()}
    reasons = tuple(
        Reason("calibrated", f"{name} failed its configured bound")
        for name, value in failed.items()
        if value
    )
    status = Status.FAIL if (skip or any(failed.values())) else Status.PASS
    return ImageQualityReport(image_id, status, metrics, failed, reasons)


@dataclass(frozen=True)
class QualityGateResult:
    """Every assessed image's report, preserving ``pass``/``review``/``fail``."""

    reports: tuple[ImageQualityReport, ...]
    profile_name: str
    profile_version: str

    @property
    def accepted(self) -> tuple[str, ...]:
        """Image IDs eligible for the segmentation-method search."""
        return tuple(r.image_id for r in self.reports if r.status == Status.PASS)

    @property
    def rejected(self) -> tuple[str, ...]:
        """Image IDs excluded from automatic selection (status ``fail``)."""
        return tuple(r.image_id for r in self.reports if r.status == Status.FAIL)

    @property
    def review(self) -> tuple[str, ...]:
        """Image IDs needing a recorded manual override to enter selection."""
        return tuple(r.image_id for r in self.reports if r.status == Status.REVIEW)

    def accept_overrides(self, image_ids: Iterable[str]) -> tuple[str, ...]:
        """Return :attr:`accepted` plus explicitly overridden ``review`` IDs.

        Does not mutate stored reports; the override itself is the caller's
        own provenance to record alongside this result.
        """
        overridden = set(image_ids) & set(self.review)
        return tuple(dict.fromkeys((*self.accepted, *overridden)))

    def to_json(self) -> str:
        """A ``image-quality.json``-shaped payload: profile plus every report."""
        payload = {
            "profile": {"name": self.profile_name, "version": self.profile_version},
            "images": [report.row() for report in self.reports],
        }
        return json.dumps(payload, indent=2)


def assess(
    images: Mapping[str, Mapping[str, object]], profile: QcProfile
) -> QualityGateResult:
    """Assess every field in ``images`` (image ID -> channel name -> array)."""
    reports = tuple(
        assess_image(channels, profile, image_id=image_id)
        for image_id, channels in images.items()
    )
    return QualityGateResult(reports, profile.name, profile.version)
