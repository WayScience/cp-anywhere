"""M1/M2/M3/M4: pipeline wrappers plus a two-stage AutoSegment selector.

M1 wraps this project's existing, oracle-verified classical primitives
(:func:`cp_anywhere.identify_primary_objects`,
:func:`cp_anywhere.identify_secondary_objects`,
:func:`cp_anywhere.identify_tertiary_objects`) into one named
``Nuclei -> Cells -> Cytoplasm`` pipeline. **No new segmentation
algorithm is implemented here.**

M2 uses the pretrained StarDist ``2D_versatile_fluo`` model for nuclei.
ONNX inference and polygon decoding run in Rust. Both candidates use
classical propagation and subtraction for cells and cytoplasm. StarDist
requires the opt-in ``stardist-model`` feature in the native extension.

The selector uses QC exclusion, hard constraints, and a seeded parameter
search over the requested candidates. It returns ``review_required`` if
none passes.

M4 implements Stage 3 (apply the selected configuration) and a reduced
Stage 4 (a self-contained, offline HTML report with the plan's section
structure, plus ``review.json`` provenance for accept/override/reject
decisions). It does not yet include response-surface plots, a failure
gallery, or the full literature technique catalog.

M5 (external scientific validation against expert-reviewed masks) is
out of scope for this module; see the plan's own Stage 1 validation
notes for what that would require.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native
from .quality import QcProfile, QualityGateResult, assess
from .segment import (
    identify_primary_objects,
    identify_secondary_objects,
    identify_tertiary_objects,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "AutoSegmentResult",
    "ClassicalPipeline",
    "EvaluationRow",
    "ReviewState",
    "SegmentationOutcome",
    "StarDistPipeline",
    "auto",
]


def _cp_anywhere_version() -> str:
    try:
        return _package_version("cp-anywhere")
    except PackageNotFoundError:
        return "unknown"


_DEFAULT_TRIALS = 16
_DEFAULT_TOP_K = 2
_REVIEW_MODES = ("optional", "required", "disabled")
_SUPPORTED_PIPELINES = ("classical", "stardist")
_MAX_CELL_COVERAGE = 0.98
# From the shipped model's own thresholds.json/config.json -- not invented.
_STARDIST_MODEL_NAME = "2D_versatile_fluo"
_STARDIST_MODEL_SHA256 = (
    "4d0c5663417544c88a6be2abaf09c4052f1202a71577425baa408d1a2d73741d"
)
_STARDIST_N_RAYS = 32
_STARDIST_GRID = (2, 2)
_IMAGE_NDIM = 2
_STARDIST_DEFAULT_PROB_THRESH = 0.479071463157368
_STARDIST_DEFAULT_NMS_THRESH = 0.3

# Stage 2's search space (plan section 6): sensible method-specific ranges,
# not a Cartesian grid. Keys match ClassicalPipeline's own override names.
CLASSICAL_PARAMETER_SPACE: dict[str, tuple[float, float]] = {
    "nuclei_threshold_correction_factor": (0.7, 1.3),
    "nuclei_min_diameter": (4.0, 16.0),
    "cells_threshold_correction_factor": (0.7, 1.3),
    "cells_regularization_factor": (0.01, 0.2),
    "cells_distance_to_dilate": (5.0, 20.0),
}


@dataclass(frozen=True)
class SegmentationOutcome:
    """One field's Nuclei/Cells/Cytoplasm labels plus reproducibility data."""

    nuclei: np.ndarray
    cells: np.ndarray
    cytoplasm: np.ndarray
    pipeline_name: str
    parameters: Mapping[str, float]
    stats: dict[str, dict[str, float]]


def _compartment_stats(labels: np.ndarray) -> dict[str, float]:
    """Fast, mask-only plausibility numbers -- not a morphology measurement."""
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    total = labels.size
    border = np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    border_touching = len(set(border.tolist()) - {0})
    return {
        "object_count": float(ids.size),
        "median_area": float(np.median(counts)) if counts.size else 0.0,
        "coverage_fraction": float(counts.sum() / total) if total else 0.0,
        "border_touching_count": float(border_touching),
    }


@dataclass(frozen=True)
class ClassicalPipeline:
    """CellProfiler-compatible Nuclei -> Cells -> Cytoplasm, one named pipeline."""

    name: str = "classical"
    overrides: dict[str, float] = field(default_factory=dict)

    def defaults(self) -> dict[str, float]:
        return {
            "nuclei_threshold_correction_factor": 1.0,
            "nuclei_min_diameter": 10.0,
            "cells_threshold_correction_factor": 1.0,
            "cells_regularization_factor": 0.05,
            "cells_distance_to_dilate": 10.0,
        }

    def parameters(self) -> dict[str, float]:
        merged = self.defaults()
        merged.update(self.overrides)
        return merged

    def run(self, nuclei_image: object, cell_image: object) -> SegmentationOutcome:
        """Run the full Nuclei -> Cells -> Cytoplasm chain on one field."""
        params = self.parameters()
        primary = identify_primary_objects(
            nuclei_image,
            threshold_correction_factor=params["nuclei_threshold_correction_factor"],
            min_diameter=params["nuclei_min_diameter"],
            unclump_method="shape",
            watershed_method="shape",
        )
        secondary = identify_secondary_objects(
            cell_image,
            primary.labels,
            primary_labels_in=primary.unedited_labels,
            method="propagation",
            threshold_correction_factor=params["cells_threshold_correction_factor"],
            regularization_factor=params["cells_regularization_factor"],
            distance_to_dilate=params["cells_distance_to_dilate"],
        )
        tertiary = identify_tertiary_objects(
            primary.labels, secondary.labels, shrink_primary=False
        )
        stats = {
            "nuclei": _compartment_stats(primary.labels),
            "cells": _compartment_stats(secondary.labels),
            "cytoplasm": _compartment_stats(tertiary.labels),
        }
        return SegmentationOutcome(
            nuclei=primary.labels,
            cells=secondary.labels,
            cytoplasm=tertiary.labels,
            pipeline_name=self.name,
            parameters=params,
            stats=stats,
        )


@cache
def _stardist_model_bytes() -> bytes:
    model_dir = Path(__file__).with_name("data") / "stardist"
    path = model_dir / f"{_STARDIST_MODEL_NAME}.onnx"
    model = path.read_bytes()
    if hashlib.sha256(model).hexdigest() != _STARDIST_MODEL_SHA256:
        raise RuntimeError(f"StarDist model checksum mismatch: {path}")
    return model


def _require_stardist() -> Callable[..., dict[str, Any]]:
    if not hasattr(_native, "stardist_segment"):
        raise RuntimeError(
            "The native extension does not include StarDist. "
            "Rebuild it with `maturin develop --features stardist-model`."
        )
    return _native.stardist_segment


def _percentile_normalize(image: np.ndarray) -> np.ndarray:
    """Scale a fluorescence image by its 1st and 99.8th percentiles."""
    values = image.astype(np.float64, copy=False)
    lo, hi = np.percentile(values, (1.0, 99.8))
    # Do not clip: pixels beyond the two percentiles retain their relative value.
    return np.ascontiguousarray((values - lo) / (hi - lo + 1e-20), dtype=np.float32)


@dataclass(frozen=True)
class StarDistPipeline:
    """Nuclei via the pretrained StarDist model; Cells/Cytoplasm stay classical.

    Requires the native extension to have been built with
    cp-anywhere's `stardist-model` Cargo feature (off by default --
    see the module docstring). Raises ``RuntimeError`` naming that gap
    if it was not, rather than silently falling back to the classical
    nuclei threshold.
    """

    name: str = "stardist"
    overrides: dict[str, float] = field(default_factory=dict)

    def defaults(self) -> dict[str, float]:
        return {
            "prob_thresh": _STARDIST_DEFAULT_PROB_THRESH,
            "nms_thresh": _STARDIST_DEFAULT_NMS_THRESH,
            "cells_threshold_correction_factor": 1.0,
            "cells_regularization_factor": 0.05,
            "cells_distance_to_dilate": 10.0,
        }

    def parameters(self) -> dict[str, float]:
        merged = self.defaults()
        merged.update(self.overrides)
        return merged

    def run(self, nuclei_image: object, cell_image: object) -> SegmentationOutcome:
        stardist_segment = _require_stardist()
        params = self.parameters()
        array = np.asarray(nuclei_image)
        if array.ndim != _IMAGE_NDIM or array.size == 0:
            raise ValueError("StarDist needs a nonempty 2D nuclei image")
        if array.dtype.kind not in "buif" or not np.isfinite(array).all():
            raise ValueError("StarDist needs a finite real-valued nuclei image")
        rows, cols = array.shape
        normalized = _percentile_normalize(array)
        raw = stardist_segment(
            normalized,
            rows,
            cols,
            _stardist_model_bytes(),
            _STARDIST_N_RAYS,
            _STARDIST_GRID[0],
            _STARDIST_GRID[1],
            params["prob_thresh"],
            params["nms_thresh"],
            2,
        )
        nuclei_labels = raw["labels"].reshape(rows, cols).astype(np.uint32)
        secondary = identify_secondary_objects(
            cell_image,
            nuclei_labels,
            primary_labels_in=nuclei_labels,
            method="propagation",
            threshold_correction_factor=params["cells_threshold_correction_factor"],
            regularization_factor=params["cells_regularization_factor"],
            distance_to_dilate=params["cells_distance_to_dilate"],
        )
        tertiary = identify_tertiary_objects(
            nuclei_labels, secondary.labels, shrink_primary=False
        )
        stats = {
            "nuclei": _compartment_stats(nuclei_labels),
            "cells": _compartment_stats(secondary.labels),
            "cytoplasm": _compartment_stats(tertiary.labels),
        }
        return SegmentationOutcome(
            nuclei=nuclei_labels,
            cells=secondary.labels,
            cytoplasm=tertiary.labels,
            pipeline_name=self.name,
            parameters=params,
            stats=stats,
        )


_PIPELINE_FACTORIES = {"classical": ClassicalPipeline, "stardist": StarDistPipeline}
_PARAMETER_SPACES: dict[str, dict[str, tuple[float, float]]] = {
    "classical": CLASSICAL_PARAMETER_SPACE,
    # StarDist's architecture/weights are fixed (pretrained); only its two
    # decision thresholds are tunable, within a narrow band around the
    # model's own calibrated defaults (plan section 6: "sensible,
    # method-specific ranges").
    "stardist": {
        "prob_thresh": (0.3, 0.7),
        "nms_thresh": (0.1, 0.5),
    },
}


def _hard_constraint_failures(outcome: SegmentationOutcome) -> list[str]:
    """Plan section 5.4: invalid labels, broken parent-child, unreasonable coverage."""
    failures = []
    if outcome.stats["nuclei"]["object_count"] == 0:
        failures.append("no nuclei detected")
    coverage = outcome.stats["cells"]["coverage_fraction"]
    if coverage == 0.0:
        failures.append("whole-cell mask is empty")
    elif coverage > _MAX_CELL_COVERAGE:
        failures.append("whole-cell mask covers nearly the entire image")
    nucleus_ids = set(np.unique(outcome.nuclei).tolist()) - {0}
    cell_ids = set(np.unique(outcome.cells).tolist()) - {0}
    if not nucleus_ids <= cell_ids:
        failures.append("a nucleus has no matching cell (broken parent-child link)")
    return failures


def _sample_parameters(
    seed: int, space: Mapping[str, tuple[float, float]]
) -> dict[str, float]:
    rng = random.Random(seed)
    return {name: rng.uniform(low, high) for name, (low, high) in sorted(space.items())}


@dataclass(frozen=True)
class EvaluationRow:
    """One candidate-pipeline run against one image in one selection phase."""

    image_id: str
    phase: str
    pipeline_name: str
    runtime_seconds: float
    failures: list[str]


@dataclass
class _Trial:
    parameters: Mapping[str, float]
    score: float
    outcomes: dict[str, SegmentationOutcome]
    failures: dict[str, list[str]]


def _score_trial(outcomes: Mapping[str, SegmentationOutcome]) -> float:
    """Higher is better: more detected objects, less degenerate coverage.

    A simple, documented ranking (plan section 7.2), not a literature-backed
    composite score.
    """
    counts = [o.stats["nuclei"]["object_count"] for o in outcomes.values()]
    coverage = [o.stats["cells"]["coverage_fraction"] for o in outcomes.values()]
    if not counts:
        return float("-inf")
    balance = 1.0 - abs(0.5 - (sum(coverage) / len(coverage)))
    return float(sum(counts) / len(counts)) * balance


def _split_tuning_validation(
    image_ids: Sequence[str],
    acquisition_groups: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Split accepted images into tuning and held-out sets (plan section 5).

    Disjointness is enforced at the acquisition-group level: every image
    sharing a group with a tuning image is kept out of held-out evidence,
    so plate-to-plate drift cannot leak between selection and validation.
    Without group metadata the split alternates the sorted image IDs,
    which guarantees only image-level disjointness.
    """
    if acquisition_groups:
        group_of = {
            image_id: acquisition_groups.get(image_id, image_id)
            for image_id in image_ids
        }
        tuning_groups = sorted(set(group_of.values()))[::2]
        tuning = sorted(
            image_id for image_id in image_ids if group_of[image_id] in tuning_groups
        )
        validation = sorted(
            image_id
            for image_id in image_ids
            if group_of[image_id] not in tuning_groups
        )
    else:
        ordered = sorted(image_ids)
        tuning = ordered[::2]
        validation = ordered[1::2]
    return tuning, validation


def _run_pipeline_on_sample(
    pipeline: ClassicalPipeline | StarDistPipeline,
    sample: Mapping[str, tuple[object, object]],
    *,
    phase: str,
    rows: list[EvaluationRow],
    max_field_seconds: float | None = None,
) -> tuple[dict[str, SegmentationOutcome], dict[str, list[str]]]:
    """Run one configuration over a sample, timing each image.

    Every run appends one ``EvaluationRow`` per image -- including images
    that fail a hard constraint or the optional ``max_field_seconds``
    runtime bound.
    """
    outcomes: dict[str, SegmentationOutcome] = {}
    failures: dict[str, list[str]] = {}
    for image_id, (nuclei_image, cell_image) in sample.items():
        started = time.perf_counter()
        outcome = pipeline.run(nuclei_image, cell_image)
        runtime_seconds = time.perf_counter() - started
        problems = _hard_constraint_failures(outcome)
        if max_field_seconds is not None and runtime_seconds > max_field_seconds:
            problems = [
                *problems,
                "runtime limit exceeded: "
                f"{runtime_seconds:.3g}s ran against the "
                f"{max_field_seconds:.3g}s maximum",
            ]
        rows.append(
            EvaluationRow(
                image_id=image_id,
                phase=phase,
                pipeline_name=outcome.pipeline_name,
                runtime_seconds=runtime_seconds,
                failures=problems,
            )
        )
        if problems:
            failures[image_id] = problems
            continue
        outcomes[image_id] = outcome
    return outcomes, failures


def _search_pipeline(  # noqa: PLR0913
    name: str,
    sample: Mapping[str, tuple[object, object]],
    *,
    trials: int,
    seed: int,
    phase: str,
    rows: list[EvaluationRow],
    max_field_seconds: float | None = None,
) -> _Trial | None:
    """Stage 2: a bounded, reproducible random search over one pipeline."""
    best: _Trial | None = None
    for trial_index in range(trials):
        # Evaluate the published StarDist defaults before tuning its thresholds.
        overrides = (
            {}
            if name == "stardist" and trial_index == 0
            else _sample_parameters(seed + trial_index, _PARAMETER_SPACES[name])
        )
        pipeline = _PIPELINE_FACTORIES[name](name=name, overrides=overrides)
        outcomes, failures = _run_pipeline_on_sample(
            pipeline,
            sample,
            phase=phase,
            rows=rows,
            max_field_seconds=max_field_seconds,
        )
        if not outcomes:
            continue
        score = _score_trial(outcomes)
        trial = _Trial(pipeline.parameters(), score, outcomes, failures)
        if best is None or trial.score > best.score:
            best = trial
    return best


@dataclass
class ReviewState:
    """Stage 4's human-judgment provenance (plan section 9)."""

    review_mode: str
    status: str
    decisions: list[dict[str, object]] = field(default_factory=list)

    def _record(self, action: str, **fields: object) -> None:
        if self.review_mode == "disabled":
            raise ValueError("review controls are disabled for this run")
        self.decisions.append(
            {
                "action": action,
                "timestamp": datetime.now(UTC).isoformat(),
                **fields,
            }
        )
        if action in ("accept", "override"):
            self.status = "accepted"
        elif action == "reject":
            self.status = "rejected"

    def accept(self, *, reviewer: str | None = None, reason: str = "") -> None:
        self._record("accept", reviewer=reviewer, reason=reason)

    def override(
        self, *, selected: str, reviewer: str | None = None, reason: str = ""
    ) -> None:
        self._record("override", selected=selected, reviewer=reviewer, reason=reason)

    def reject(self, *, reviewer: str | None = None, reason: str = "") -> None:
        self._record("reject", reviewer=reviewer, reason=reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "review_mode": self.review_mode,
            "status": self.status,
            "decisions": self.decisions,
        }


@dataclass
class AutoSegmentResult:
    """Everything Stage 0-2 produced, plus Stage 3/4 entry points."""

    status: str
    stage0: QualityGateResult
    selected_pipeline_name: str | None
    selected_parameters: Mapping[str, float] | None
    candidates: dict[str, _Trial]
    channels: dict[str, str]
    seed: int
    review: ReviewState
    cp_anywhere_version: str = field(default_factory=_cp_anywhere_version)
    evaluations: list[EvaluationRow] = field(default_factory=list)
    tuning_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()

    def segment(
        self, images: Mapping[str, Mapping[str, object]]
    ) -> dict[str, SegmentationOutcome]:
        """Stage 3: apply the selected configuration to (possibly new) images."""
        if self.selected_pipeline_name is None or self.selected_parameters is None:
            raise ValueError("no pipeline was selected; status is review_required")
        pipeline = _PIPELINE_FACTORIES[self.selected_pipeline_name](
            name=self.selected_pipeline_name,
            overrides=dict(self.selected_parameters),
        )
        results = {}
        for image_id, channels in images.items():
            nuclei_image = channels[self.channels["nuclei"]]
            cell_image = channels[self.channels["cells"]]
            results[image_id] = pipeline.run(nuclei_image, cell_image)
        return results

    def save(self, path: str | Path) -> None:
        """``segmentation-config.json``: selected pipeline, settings, provenance."""
        payload = {
            "cp_anywhere_version": self.cp_anywhere_version,
            "status": self.status,
            "selected_pipeline": self.selected_pipeline_name,
            "parameters": self.selected_parameters,
            "channels": self.channels,
            "seed": self.seed,
            "stage0_accepted": list(self.stage0.accepted),
            "stage0_rejected": list(self.stage0.rejected),
            "stage0_review": list(self.stage0.review),
        }
        if self.selected_pipeline_name == "stardist":
            payload["model"] = {
                "name": _STARDIST_MODEL_NAME,
                "sha256": _STARDIST_MODEL_SHA256,
            }
        Path(path).write_text(json.dumps(payload, indent=2))

    def save_review(self, path: str | Path) -> None:
        """``review.json``: review mode, status, decisions, audit trail."""
        Path(path).write_text(json.dumps(self.review.to_dict(), indent=2))

    def report(self, path: str | Path) -> None:
        """A self-contained, offline HTML report (plan section 9 structure)."""
        Path(path).write_text(_render_report(self))


def _escape(text: object) -> str:
    return html.escape(str(text))


def _render_report(result: AutoSegmentResult) -> str:
    stage0_rows = "".join(
        f"<tr><td>{_escape(r.image_id)}</td><td>{_escape(r.status.name.lower())}</td>"
        f"<td>{_escape(json.dumps(r.criteria_failed))}</td></tr>"
        for r in result.stage0.reports
    )
    candidate_rows = "".join(
        f"<tr><td>{_escape(name)}</td><td>{_escape(round(trial.score, 4))}</td>"
        f"<td>{_escape(json.dumps(trial.parameters))}</td></tr>"
        for name, trial in result.candidates.items()
    )
    model_note = ""
    if "stardist" in result.candidates:
        model_note = (
            f"<p>StarDist model: {_escape(_STARDIST_MODEL_NAME)}. "
            f"SHA-256: {_escape(_STARDIST_MODEL_SHA256)}.</p>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>cp-anywhere AutoSegment report</title></head>
<body>
<h1>cp-anywhere AutoSegment report</h1>

<h2>Image-quality gate</h2>
<p>Profile: {_escape(result.stage0.profile_name)}
({_escape(result.stage0.profile_version)}).
Accepted: {len(result.stage0.accepted)}. Rejected: {len(result.stage0.rejected)}.
Review: {len(result.stage0.review)}.</p>
<table><tr><th>Image</th><th>Status</th><th>Criteria failed</th></tr>
{stage0_rows}</table>

<h2>Decision summary</h2>
<p>Status: {_escape(result.status)}. Selected pipeline:
{_escape(result.selected_pipeline_name or "none")}.
Seed: {_escape(result.seed)}. Channels: {_escape(json.dumps(result.channels))}.</p>

<h2>Parameter search</h2>
<table><tr><th>Pipeline</th><th>Score</th><th>Parameters</th></tr>{candidate_rows}</table>

<h2>Human review</h2>
<p>Review mode: {_escape(result.review.review_mode)}.
Status: {_escape(result.review.status)}.
Decisions: {_escape(len(result.review.decisions))}.</p>

<h2>Reproducibility</h2>
<p>cp-anywhere {_escape(result.cp_anywhere_version)}. Seed {_escape(result.seed)}.</p>
{model_note}
<p>This report embeds no remote resources; see
cp-anywhere-autosegment-plan.md for the method catalog and literature
references not yet included in this reduced report.</p>
</body>
</html>
"""


def auto(  # noqa: C901, PLR0912, PLR0913
    images: Mapping[str, Mapping[str, object]],
    *,
    channels: Mapping[str, str],
    quality_profile: QcProfile,
    pipelines: Sequence[str] = ("classical",),
    top_k: int = _DEFAULT_TOP_K,
    trials: int = _DEFAULT_TRIALS,
    device: str = "cpu",
    seed: int = 42,
    minimum_accepted: int = 1,
    review_mode: str = "optional",
    acquisition_groups: Mapping[str, str] | None = None,
    max_field_seconds: float | None = None,
) -> AutoSegmentResult:
    """Stage 0 through Stage 2: assess quality, select, and tune a pipeline.

    ``channels`` maps the segmentation roles ``"nuclei"`` and ``"cells"``
    to channel names present in every image's own mapping.
    ``"classical"`` and ``"stardist"`` are implemented (see the module
    docstring for ``"stardist"``'s native-build requirement); any other
    name in ``pipelines`` raises ``NotImplementedError`` naming the
    specific gap. ``device`` only accepts ``"cpu"`` in this release --
    every underlying primitive is a native CPU Rust kernel; no GPU path
    exists to dispatch to.

    ``acquisition_groups`` optionally maps image IDs to acquisition
    groups (e.g. plates). The tuning and held-out splits never share a
    group, so held-out evidence always comes from acquisitions unseen
    during tuning (plan section 5); when it is omitted the split
    alternates the sorted image IDs instead. Every candidate run's
    per-image outcome, phase, and runtime is recorded on the result's
    ``evaluations``, and ``max_field_seconds`` optionally bounds each
    image's runtime: runs exceeding it fail with a "runtime limit"
    failure and the image is excluded from that candidate's evidence.
    """
    if device != "cpu":
        raise NotImplementedError(
            f'device={device!r} is not supported; only "cpu" is implemented'
        )
    if review_mode not in _REVIEW_MODES:
        raise ValueError(f"review_mode must be one of {_REVIEW_MODES}")
    unsupported = set(pipelines) - set(_SUPPORTED_PIPELINES)
    if unsupported:
        raise NotImplementedError(
            f"pipeline(s) {sorted(unsupported)!r} are not implemented in this "
            f"release. Only {_SUPPORTED_PIPELINES!r} are available."
        )
    if "stardist" in pipelines:
        _require_stardist()
    for role in ("nuclei", "cells"):
        if role not in channels:
            raise ValueError(f"channels must map {role!r} to a channel name")

    stage0 = assess(images, quality_profile)
    review = ReviewState(review_mode=review_mode, status="not_reviewed")
    tuning_ids, validation_ids = _split_tuning_validation(
        stage0.accepted, acquisition_groups
    )
    accepted_sample = {
        image_id: (
            images[image_id][channels["nuclei"]],
            images[image_id][channels["cells"]],
        )
        for image_id in stage0.accepted
    }
    tuning_sample = {image_id: accepted_sample[image_id] for image_id in tuning_ids}
    validation_sample = {
        image_id: accepted_sample[image_id] for image_id in validation_ids
    }

    def _result(
        status: str,
        *,
        selected_name: str | None = None,
        selected_parameters: Mapping[str, float] | None = None,
        candidates: dict[str, _Trial] | None = None,
        evaluations: list[EvaluationRow] | None = None,
    ) -> AutoSegmentResult:
        return AutoSegmentResult(
            status=status,
            stage0=stage0,
            selected_pipeline_name=selected_name,
            selected_parameters=selected_parameters,
            candidates=candidates or {},
            channels=dict(channels),
            seed=seed,
            review=review,
            tuning_ids=tuple(tuning_ids),
            validation_ids=tuple(validation_ids),
            evaluations=evaluations or [],
        )

    if len(stage0.accepted) < minimum_accepted:
        return _result("review_required")

    evaluations: list[EvaluationRow] = []
    candidates: dict[str, _Trial] = {}
    for name in pipelines:
        # Stage 1 screens each candidate's published defaults, Stage 2 tunes.
        screen_pipeline = _PIPELINE_FACTORIES[name](name=name, overrides={})
        _run_pipeline_on_sample(
            screen_pipeline,
            tuning_sample,
            phase="screen",
            rows=evaluations,
            max_field_seconds=max_field_seconds,
        )
        trial = _search_pipeline(
            name,
            tuning_sample,
            trials=trials,
            seed=seed,
            phase="tuning",
            rows=evaluations,
            max_field_seconds=max_field_seconds,
        )
        if trial is not None:
            candidates[name] = trial

    if not candidates:
        return _result("review_required", evaluations=evaluations)

    ranked = sorted(candidates.items(), key=lambda item: item[1].score, reverse=True)
    best_name, best_trial = ranked[: top_k or len(ranked)][0]

    status = "selected"
    if review_mode == "required":
        status = "awaiting_approval"

    # Stage 2 validation: rerun the tuned configuration on held-out
    # acquisitions so selection confidence rests on unseen evidence.
    if validation_sample:
        tuned_pipeline = _PIPELINE_FACTORIES[best_name](
            name=best_name, overrides=dict(best_trial.parameters)
        )
        held_out, _held_out_failures = _run_pipeline_on_sample(
            tuned_pipeline,
            validation_sample,
            phase="held_out",
            rows=evaluations,
            max_field_seconds=max_field_seconds,
        )
        if not held_out:
            return _result(
                "review_required", candidates=candidates, evaluations=evaluations
            )

    return _result(
        status,
        selected_name=best_name,
        selected_parameters=best_trial.parameters,
        candidates=candidates,
        evaluations=evaluations,
    )
