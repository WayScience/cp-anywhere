"""Exercise the classical and optional StarDist AutoSegment paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere import _native, autosegment, quality

_OVERRIDE_THRESHOLD_FACTOR = 1.1
_REPORT_SEED = 5
_FLOOR_SEARCH_UPPER = 0.05
_FLOOR_INERT = 0.038  # RxRx19a-reanalysis PR#18's calibrated floor
_FLOOR_BINDING = 0.9
_STARDIST_AVAILABLE = hasattr(_native, "stardist_segment")
_STARDIST_CENTERS = [(24, 24), (24, 100), (100, 24), (64, 64), (100, 100)]


def _blob_image(
    centers: list[tuple[int, int]],
    radius: float = 4.0,
    shape: tuple[int, int] = (48, 48),
) -> np.ndarray:
    """A synthetic field with one Gaussian bump per nucleus, plus noise floor."""
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    field = np.full(shape, 0.05, dtype=np.float64)
    for cy, cx in centers:
        field += np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius**2)))
    return np.clip(field, 0.0, 1.0).astype(np.float32)


_NUCLEI_CENTERS = [(12, 12), (12, 36), (36, 12), (36, 36)]


def _synthetic_field() -> dict[str, np.ndarray]:
    nuclei = _blob_image(_NUCLEI_CENTERS, radius=3.5)
    cells = _blob_image(_NUCLEI_CENTERS, radius=9.0)
    return {"DNA": nuclei, "AGP": cells}


def _lenient_profile() -> quality.QcProfile:
    # A trivially satisfiable bound, just to exercise the Stage 0 gate wiring.
    return quality.QcProfile.from_criteria(
        name="test",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=99.0)
        ],
    )


def _unsatisfiable_profile() -> quality.QcProfile:
    return quality.QcProfile.from_criteria(
        name="test",
        version="v1",
        criteria=[
            quality.QcCriterion(channel="DNA", metric="PercentMaximal", maximum=-1.0)
        ],
    )


def test_classical_pipeline_produces_three_consistent_compartments() -> None:
    field = _synthetic_field()
    pipeline = autosegment.ClassicalPipeline()
    outcome = pipeline.run(field["DNA"], field["AGP"])

    assert outcome.nuclei.shape == field["DNA"].shape
    assert outcome.cells.shape == outcome.cytoplasm.shape == outcome.nuclei.shape
    assert outcome.stats["nuclei"]["object_count"] == len(_NUCLEI_CENTERS)
    assert not np.any(outcome.cytoplasm[outcome.nuclei != 0])
    # Every surviving nucleus must have grown into exactly one cell ID.
    for nucleus_id in np.unique(outcome.nuclei)[1:]:
        cell_ids = np.unique(outcome.cells[outcome.nuclei == nucleus_id])
        assert cell_ids.tolist() == [nucleus_id]


def test_classical_pipeline_reports_reproducible_parameters() -> None:
    field = _synthetic_field()
    pipeline = autosegment.ClassicalPipeline(
        overrides={"nuclei_threshold_correction_factor": _OVERRIDE_THRESHOLD_FACTOR}
    )
    outcome = pipeline.run(field["DNA"], field["AGP"])
    assert (
        outcome.parameters["nuclei_threshold_correction_factor"]
        == _OVERRIDE_THRESHOLD_FACTOR
    )
    assert outcome.pipeline_name == "classical"


def test_classical_pipeline_defaults_to_and_searches_a_threshold_floor() -> None:
    params = autosegment.ClassicalPipeline().parameters()
    assert params["nuclei_threshold_range_min"] == 0.0
    lower, upper = autosegment._PARAMETER_SPACES["classical"][
        "nuclei_threshold_range_min"
    ]
    assert (lower, upper) == (0.0, _FLOOR_SEARCH_UPPER)
    sampled = autosegment._sample_parameters(
        42, autosegment._PARAMETER_SPACES["classical"]
    )
    assert 0.0 <= sampled["nuclei_threshold_range_min"] <= _FLOOR_SEARCH_UPPER


def test_classical_pipeline_applies_the_threshold_floor_when_it_binds() -> None:
    """A floor below the estimator's threshold is inert; above it, it binds.

    RxRx19a-reanalysis PR#18's nuclei threshold uses a 0.038 floor to keep
    perinuclear bleedthrough from passing a relative global threshold; the
    floor clamps the computed threshold regardless of the method.
    """
    field = _synthetic_field()
    pipeline = autosegment.ClassicalPipeline()
    baseline = pipeline.run(field["DNA"], field["AGP"])
    inert = autosegment.ClassicalPipeline(
        overrides={"nuclei_threshold_range_min": _FLOOR_INERT}
    ).run(field["DNA"], field["AGP"])
    binding = autosegment.ClassicalPipeline(
        overrides={"nuclei_threshold_range_min": _FLOOR_BINDING}
    ).run(field["DNA"], field["AGP"])

    assert (inert.nuclei != 0).sum() == (baseline.nuclei != 0).sum()
    assert (binding.nuclei != 0).sum() < (baseline.nuclei != 0).sum()
    assert np.unique(inert.nuclei).tolist() == np.unique(baseline.nuclei).tolist()
    assert binding.parameters["nuclei_threshold_range_min"] == _FLOOR_BINDING


def test_requesting_an_unknown_pipeline_name_is_a_documented_gap() -> None:
    with pytest.raises(NotImplementedError, match="deepflash"):
        autosegment.auto(
            {"a": _synthetic_field()},
            channels={"nuclei": "DNA", "cells": "AGP"},
            pipelines=["deepflash"],
            quality_profile=_lenient_profile(),
        )


@pytest.mark.skipif(_STARDIST_AVAILABLE, reason="checks the default native build")
def test_stardist_requires_the_native_feature_even_if_no_field_passes_qc() -> None:
    with pytest.raises(RuntimeError, match="stardist-model"):
        autosegment.auto(
            {"a": _synthetic_field()},
            channels={"nuclei": "DNA", "cells": "AGP"},
            pipelines=["classical", "stardist"],
            quality_profile=_unsatisfiable_profile(),
        )


@pytest.mark.skipif(not _STARDIST_AVAILABLE, reason="requires stardist-model")
@pytest.mark.parametrize("shape", [(128, 128), (127, 129)])
def test_stardist_detects_nuclei_and_preserves_the_image_grid(
    shape: tuple[int, int],
) -> None:
    dna, cells = _stardist_field(shape).values()
    result = autosegment.StarDistPipeline().run(dna, cells)

    assert result.pipeline_name == "stardist"
    assert result.nuclei.shape == result.cells.shape == result.cytoplasm.shape == shape
    assert result.nuclei.dtype == np.uint32
    ids = {int(result.nuclei[point]) for point in _STARDIST_CENTERS}
    assert ids == set(range(1, len(_STARDIST_CENTERS) + 1))
    assert set(np.unique(result.nuclei)) == ids | {0}
    nucleus_pixels = result.nuclei != 0
    np.testing.assert_array_equal(
        result.cells[nucleus_pixels], result.nuclei[nucleus_pixels]
    )
    assert not np.any(result.cytoplasm[nucleus_pixels])
    assert np.any(result.cytoplasm)


@pytest.mark.skipif(not _STARDIST_AVAILABLE, reason="requires stardist-model")
def test_stardist_can_be_selected_applied_and_saved(tmp_path: Path) -> None:
    images = {"field": _stardist_field((128, 128))}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["stardist"],
        quality_profile=_lenient_profile(),
        trials=2,
    )
    assert result.status == "selected"
    assert result.selected_pipeline_name == "stardist"
    candidate = result.candidates["stardist"]
    assert "prob_thresh" in candidate.parameters
    assert "nuclei_min_diameter" not in candidate.parameters
    reapplied = result.segment(images)["field"]
    np.testing.assert_array_equal(reapplied.nuclei, candidate.outcomes["field"].nuclei)
    destination = tmp_path / "config.json"
    result.save(destination)
    saved = json.loads(destination.read_text())
    assert saved["model"]["name"] == "2D_versatile_fluo"
    assert saved["model"]["sha256"] == autosegment._STARDIST_MODEL_SHA256
    report = tmp_path / "report.html"
    result.report(report)
    assert saved["model"]["sha256"] in report.read_text()


@pytest.mark.skipif(not _STARDIST_AVAILABLE, reason="requires stardist-model")
def test_stardist_and_classical_compete_on_the_same_field() -> None:
    field = _stardist_field((128, 128))
    result = autosegment.auto(
        {"field": field},
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical", "stardist"],
        quality_profile=_lenient_profile(),
        trials=1,
    )

    assert set(result.candidates) == {"classical", "stardist"}
    winner = max(result.candidates, key=lambda name: result.candidates[name].score)
    assert result.selected_pipeline_name == winner
    for candidate in result.candidates.values():
        outcome = candidate.outcomes["field"]
        assert outcome.nuclei.shape == field["DNA"].shape
        assert not np.any(outcome.cytoplasm[outcome.nuclei != 0])


@pytest.mark.skipif(not _STARDIST_AVAILABLE, reason="requires stardist-model")
def test_stardist_native_input_shape_must_match_declared_shape() -> None:
    image = np.zeros((8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="rows and cols"):
        _native.stardist_segment(image, 4, 8, b"", 32, 2, 2, 0.5, 0.3)


def _stardist_field(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, cols = np.indices(shape)
    dna = np.random.default_rng(7).normal(0.05, 0.02, size=shape)
    for row, col in _STARDIST_CENTERS:
        r2 = (rows - row) ** 2 + (cols - col) ** 2
        dna += 0.9 * np.exp(-r2 / (2 * 8.0**2))
    return {
        "DNA": np.clip(dna, 0, 1).astype(np.float32),
        "AGP": _blob_image(_STARDIST_CENTERS, radius=13.0, shape=shape),
    }


def test_requesting_a_gpu_device_is_a_documented_not_implemented_gap() -> None:
    with pytest.raises(NotImplementedError, match="cuda"):
        autosegment.auto(
            {"a": _synthetic_field()},
            channels={"nuclei": "DNA", "cells": "AGP"},
            quality_profile=_lenient_profile(),
            device="cuda",
        )


def test_auto_select_runs_stage0_through_stage2_and_picks_classical() -> None:
    images = {f"field-{i}": _synthetic_field() for i in range(3)}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical"],
        quality_profile=_lenient_profile(),
        trials=4,
        seed=7,
    )
    assert result.status == "selected"
    assert result.selected_pipeline_name == "classical"
    assert result.stage0.accepted == tuple(images)

    masks = result.segment(images)
    assert set(masks) == set(images)
    assert masks["field-0"].nuclei.shape == images["field-0"]["DNA"].shape


def test_auto_uses_disjoint_acquisition_groups_for_held_out_evidence() -> None:
    images = {name: _synthetic_field() for name in ("a1", "a2", "b1", "b2")}
    groups = {"a1": "plate-a", "a2": "plate-a", "b1": "plate-b", "b2": "plate-b"}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        quality_profile=_lenient_profile(),
        acquisition_groups=groups,
        trials=2,
    )
    assert result.status == "selected"
    phase_ids = {
        phase: {row.image_id for row in result.evaluations if row.phase == phase}
        for phase in ("screen", "tuning", "held_out")
    }
    assert phase_ids["screen"] == phase_ids["tuning"] == set(result.tuning_ids)
    assert phase_ids["held_out"] == set(result.validation_ids)
    assert {groups[name] for name in result.tuning_ids}.isdisjoint(
        groups[name] for name in result.validation_ids
    )
    assert all(row.runtime_seconds >= 0 for row in result.evaluations)


def test_auto_rejects_candidates_that_exceed_the_runtime_limit() -> None:
    result = autosegment.auto(
        {"a": _synthetic_field(), "b": _synthetic_field()},
        channels={"nuclei": "DNA", "cells": "AGP"},
        quality_profile=_lenient_profile(),
        max_field_seconds=1e-12,
        trials=2,
    )
    assert result.status == "review_required"
    assert result.evaluations
    assert all("runtime limit" in " ".join(row.failures) for row in result.evaluations)


def test_auto_select_is_reproducible_with_the_same_seed() -> None:
    images = {f"field-{i}": _synthetic_field() for i in range(3)}
    profile = _lenient_profile()
    runs = [
        autosegment.auto(
            images,
            channels={"nuclei": "DNA", "cells": "AGP"},
            pipelines=["classical"],
            quality_profile=profile,
            trials=4,
            seed=11,
        )
        for _ in range(2)
    ]
    assert runs[0].selected_parameters == runs[1].selected_parameters


def test_auto_select_excludes_failed_images_from_the_search() -> None:
    images = {f"field-{i}": _synthetic_field() for i in range(5)}
    images["bad"] = {
        "DNA": np.zeros((48, 48), dtype=np.float32),
        "AGP": np.zeros((48, 48), dtype=np.float32),
    }
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical"],
        quality_profile=_unsatisfiable_profile(),
        trials=2,
        seed=3,
    )
    assert result.status == "review_required"
    assert result.stage0.rejected == tuple(images)


def test_too_few_accepted_images_requests_review_without_running_stage1() -> None:
    images = {"field-0": _synthetic_field()}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical"],
        quality_profile=_lenient_profile(),
        trials=2,
        seed=1,
        minimum_accepted=2,
    )
    assert result.status == "review_required"
    assert result.candidates == {}


def test_save_report_and_review_round_trip(tmp_path: Path) -> None:
    images = {f"field-{i}": _synthetic_field() for i in range(3)}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical"],
        quality_profile=_lenient_profile(),
        trials=4,
        seed=_REPORT_SEED,
    )
    config_path = tmp_path / "segmentation-config.json"
    result.save(config_path)
    config = json.loads(config_path.read_text())
    assert config["selected_pipeline"] == "classical"
    assert config["seed"] == _REPORT_SEED

    report_path = tmp_path / "report.html"
    result.report(report_path)
    document = report_path.read_text()
    assert document.startswith("<!DOCTYPE html>")
    assert "Image-quality gate" in document
    assert "Decision summary" in document
    before_reproducibility = document.split("Reproducibility")[0]
    assert "http://" not in before_reproducibility
    assert "https://" not in before_reproducibility

    result.review.accept(reviewer="dave")
    review_path = tmp_path / "review.json"
    result.save_review(review_path)
    review = json.loads(review_path.read_text())
    assert review["review_mode"] == "optional"
    assert review["decisions"][0]["action"] == "accept"
    assert review["decisions"][0]["reviewer"] == "dave"


def test_required_review_mode_withholds_final_status(tmp_path: Path) -> None:
    images = {f"field-{i}": _synthetic_field() for i in range(3)}
    result = autosegment.auto(
        images,
        channels={"nuclei": "DNA", "cells": "AGP"},
        pipelines=["classical"],
        quality_profile=_lenient_profile(),
        trials=4,
        seed=_REPORT_SEED,
        review_mode="required",
    )
    assert result.status == "awaiting_approval"
    assert result.review.review_mode == "required"
