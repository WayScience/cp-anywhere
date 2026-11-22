"""Stage 0 image-quality gate: pass/review/fail decisions, not segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import quality

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe/image_quality"


def _graded_field(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """Every pixel distinct, so PercentMaximal/PercentMinimal stay near 0."""
    return np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)


def _saturated_field(fraction: float, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """``fraction`` of pixels pinned at the image's own maximum value."""
    field = _graded_field(shape)
    saturated_count = round(fraction * field.size)
    flat = field.ravel()
    flat[:saturated_count] = flat.max()
    return flat.reshape(shape)


def test_pinned_pipeline_profile_matches_known_kept_and_skipped_fixtures() -> None:
    profile = quality.QcProfile.from_cppipe(FIXTURES / "quality_flag.cppipe")
    kept = {"DNA": np.load(FIXTURES / "quality_flag.kept.npy")}
    skipped = {"DNA": np.load(FIXTURES / "quality_flag.skipped.npy")}

    kept_report = quality.assess_image(kept, profile, image_id="kept")
    skipped_report = quality.assess_image(skipped, profile, image_id="skipped")

    assert kept_report.status == quality.Status.PASS
    assert not any(kept_report.criteria_failed.values())
    assert skipped_report.status == quality.Status.FAIL
    assert any(skipped_report.criteria_failed.values())
    # The profile's metrics are the ImageQuality_* keys MeasureImageQuality reports.
    assert any(
        key.startswith("ImageQuality_PowerLogLogSlope_") for key in kept_report.metrics
    )


def test_from_cppipe_rejects_a_pipeline_without_exactly_one_flag() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        quality.QcProfile.from_cppipe(FIXTURES.parent / "manual.cppipe")


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (_graded_field(), quality.Status.PASS),
        (_saturated_field(0.5), quality.Status.FAIL),
    ],
)
def test_explicit_criteria_profile_evaluates_a_bound(
    image: np.ndarray, expected: quality.Status
) -> None:
    profile = quality.QcProfile.from_criteria(
        name="synthetic",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=1.0)
        ],
    )
    report = quality.assess_image({"DNA": image}, profile, image_id="field-1")
    assert report.status == expected


def test_missing_required_channel_is_review_not_fail() -> None:
    profile = quality.QcProfile.from_criteria(
        name="synthetic",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="RNA", metric="PercentMaximal", maximum=1.0)
        ],
    )
    report = quality.assess_image({"DNA": _graded_field()}, profile, image_id="field-1")
    assert report.status == quality.Status.REVIEW
    assert report.reasons and report.reasons[0].kind == "uncalibrated"


@pytest.mark.parametrize(
    ("images", "match"),
    [
        ({}, "no channels"),
        ({"DNA": np.zeros((2, 2, 2))}, "2D"),
        ({"DNA": np.zeros((2, 2), dtype=np.complex128)}, "dtype"),
        ({"DNA": np.array([[1.0, np.nan]])}, "non-finite"),
        ({"DNA": np.zeros((2, 2)), "Actin": np.zeros((3, 3))}, "shape"),
    ],
)
def test_structural_problems_fail_closed(images: dict, match: str) -> None:
    profile = quality.QcProfile.from_criteria(
        name="synthetic",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=1.0)
        ],
    )
    report = quality.assess_image(images, profile, image_id="field-1")
    assert report.status == quality.Status.FAIL
    assert any(match in r.detail for r in report.reasons)


def test_assess_batch_groups_reports_and_exports_json(tmp_path: Path) -> None:
    profile = quality.QcProfile.from_criteria(
        name="synthetic",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=1.0)
        ],
    )
    result = quality.assess(
        {"a": {"DNA": _graded_field()}, "b": {"DNA": _saturated_field(0.5)}}, profile
    )
    assert result.accepted == ("a",)
    assert result.rejected == ("b",)
    path = tmp_path / "image-quality.json"
    path.write_text(result.to_json())
    decoded = json.loads(path.read_text())
    assert {row["image_id"] for row in decoded["images"]} == {"a", "b"}
    assert decoded["profile"] == {"name": "synthetic", "version": "v1"}


def test_images_do_not_mutate_the_callers_arrays() -> None:
    profile = quality.QcProfile.from_criteria(
        name="synthetic",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=1.0)
        ],
    )
    array = _graded_field()
    array.setflags(write=False)
    quality.assess_image({"DNA": array}, profile, image_id="field-1")
    assert array.flags.writeable is False
