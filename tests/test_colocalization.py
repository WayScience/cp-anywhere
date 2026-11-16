"""Oracle tests for :func:`cp_anywhere.measure_colocalization` against
real CellProfiler 4.2.8 fixtures (see ``fixtures/colocalization/`` and
``scripts/reference/generate_colocalization.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import measure_colocalization
from cp_anywhere.colocalization import FastCostesMode

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/colocalization"
_CASES = sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "manifest")
_IMAGES_OR_OBJECTS = {
    "Across entire image": (True, False),
    "Within objects": (False, True),
    "Both": (True, True),
}
_FAST_COSTES: dict[str, FastCostesMode] = {
    "Faster": "faster",
    "Fast": "fast",
    "Accurate": "accurate",
}


def _assert_matches(got: float, want: float) -> None:
    if math.isnan(want):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_colocalization_matches_a_fresh_cellprofiler_oracle_run(
    name: str,
) -> None:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    pixels = {
        image_name: np.array(rows, dtype=np.float64)
        for image_name, rows in fixture["pixels"].items()
    }
    masks = {
        image_name: np.array(rows, dtype=bool)
        for image_name, rows in fixture.get("masks", {}).items()
    }
    labels_by_object = {
        object_name: np.array(rows, dtype=np.uint32)
        for object_name, rows in fixture["labels"].items()
    }
    settings = fixture["settings"]
    wants_image, wants_object = _IMAGES_OR_OBJECTS[settings["images_or_objects"]]
    result = measure_colocalization(
        pixels,
        objects=next(iter(labels_by_object.values()), None),
        masks=masks or None,
        threshold_percent=settings["thr"],
        fast_costes=_FAST_COSTES[settings["fast_costes"]],
        costes_scale_max=_scale_max(fixture),
        do_corr_and_slope=settings["do_corr_and_slope"],
        do_manders=settings["do_manders"],
        do_rwc=settings["do_rwc"],
        do_overlap=settings["do_overlap"],
        do_costes=settings["do_costes"],
        wants_image_measurements=wants_image,
        wants_object_measurements=wants_object,
    )
    expected = fixture["expected"]
    assert set(result.image) == set(expected.get("Image", {}))
    for column, (want,) in expected.get("Image", {}).items():
        _assert_matches(result.image[column], want)
    object_scopes = [scope for scope in expected if scope != "Image"]
    if object_scopes:
        # One object set per run, exactly like the real module's own
        # per-object-set run_image_pair_objects calls.
        assert len(object_scopes) == len(labels_by_object)
        for object_name in object_scopes:
            per_object_result = measure_colocalization(
                pixels,
                objects=labels_by_object[object_name],
                masks=masks or None,
                threshold_percent=settings["thr"],
                fast_costes=_FAST_COSTES[settings["fast_costes"]],
                costes_scale_max=_scale_max(fixture),
                do_corr_and_slope=settings["do_corr_and_slope"],
                do_manders=settings["do_manders"],
                do_rwc=settings["do_rwc"],
                do_overlap=settings["do_overlap"],
                do_costes=settings["do_costes"],
                wants_image_measurements=False,
                wants_object_measurements=True,
            )
            assert set(per_object_result.objects) == set(expected[object_name])
            for column, values in expected[object_name].items():
                actual_values = per_object_result.objects[column]
                assert len(actual_values) == len(values)
                for got, want in zip(actual_values, values, strict=True):
                    _assert_matches(got, want)
    else:
        assert not result.objects


def _scale_max(fixture: dict) -> float:
    """``get_scale``'s resolved per-pair value: the max across both
    images' declared scales, defaulting to 255.0 when absent."""
    scales = fixture.get("scales", {})
    return float(max(scales.values())) if scales else 255.0


def test_colocalization_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        measure_colocalization({"DNA": np.zeros((5, 5)), "GFP": np.zeros((5, 6))})


def test_colocalization_rejects_a_single_image() -> None:
    with pytest.raises(ValueError, match="at least two images"):
        measure_colocalization({"DNA": np.zeros((5, 5))})


def test_colocalization_rejects_an_unknown_fast_costes_mode() -> None:
    with pytest.raises(ValueError, match="fast_costes"):
        measure_colocalization(
            {"DNA": np.zeros((5, 5)), "GFP": np.zeros((5, 5))},
            fast_costes="instant",  # ty: ignore[invalid-argument-type]
        )


def test_colocalization_rejects_object_measurements_without_labels() -> None:
    with pytest.raises(ValueError, match="wants_object_measurements"):
        measure_colocalization(
            {"DNA": np.zeros((5, 5)), "GFP": np.zeros((5, 5))},
            wants_object_measurements=True,
        )
