"""Run a translated pipeline on independent image sets."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import islice
from multiprocessing import get_context
from typing import TYPE_CHECKING

from .cppipe import Pipeline, load_pipeline
from .io import load_array

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from concurrent.futures import Future
    from pathlib import Path
    from typing import Literal

    import numpy as np

    from .api import MeasurementResult
    from .segment import (
        SecondaryObjectsResult,
        SegmentationResult,
        TertiaryObjectsResult,
    )


@dataclass(frozen=True)
class ImageSetResult:
    """One image set's labels and measurements, keyed by its caller-supplied ID."""

    image_id: str
    primary: dict[str, SegmentationResult]
    derived: dict[str, SecondaryObjectsResult | TertiaryObjectsResult]
    measurements: MeasurementResult | None


def _run_image_set(
    pipeline: Pipeline, image_id: str, sources: Mapping[str, str | Path | np.ndarray]
) -> ImageSetResult:
    primary, derived, result = pipeline.run(
        {name: load_array(source) for name, source in sources.items()}
    )
    if result is not None:
        # Pipeline.run identifies the field as "Image" in both the feature
        # table and its relationship endpoints. Rename them together.
        relationships = []
        for edge in result.relationships:
            updated = dict(edge)
            for key in ("first_image_id", "second_image_id"):
                if updated.get(key) == result.image_id:
                    updated[key] = image_id
            relationships.append(updated)
        result = replace(result, image_id=image_id, relationships=relationships)
    return ImageSetResult(image_id, primary, derived, result)


@contextmanager
def _image_set_context(image_id: str) -> Generator[None, None, None]:
    """Attach the image ID to errors in the caller, even across a process boundary."""
    try:
        yield
    except Exception as error:
        error.add_note(f"cp-anywhere image set: {image_id!r}")
        raise


def _result(image_id: str, future: Future[ImageSetResult]) -> ImageSetResult:
    with _image_set_context(image_id):
        return future.result()


def run_batch(
    pipeline: Pipeline | str | Path,
    image_sets: Mapping[str, Mapping[str, str | Path | np.ndarray]],
    *,
    workers: int = 1,
    backend: Literal["thread", "process"] = "thread",
) -> Generator[ImageSetResult, None, None]:
    """Run one translated pipeline per image set, in input order.

    At most ``workers`` image sets are in flight at once. A single image set
    always runs in the caller. Do not mutate the pipeline, mapping, or arrays
    while iterating. If you stop early, close the iterator to cancel queued work
    where possible and wait for running work to finish. Errors propagate with
    the image ID attached to the traceback.

    This function does not match files, group fields, estimate shared maps, or
    write outputs. Pass any shared calibration images by name with each set.
    For custom workflows, call :meth:`Pipeline.run` or individual modules.

    Args:
        pipeline: A loaded pipeline or path to a supported ``.cppipe`` file.
        image_sets: Image IDs mapped to named channel paths or arrays. An empty
            mapping yields no results.
        workers: Maximum workers. One runs serially, regardless of backend.
        backend: Use threads (default) or spawned processes. For processes,
            run from an importable Python script under a main guard. Paths avoid
            copying input arrays to workers. Results are copied back to the caller.

    Yields:
        Segmentation and measurement results with the caller's image ID.

    Raises:
        ValueError: If the backend, worker count, or image IDs are invalid.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if backend not in ("thread", "process"):
        raise ValueError("backend must be 'thread' or 'process'")
    if any(not isinstance(image_id, str) or not image_id for image_id in image_sets):
        raise ValueError("image IDs must be nonempty strings")
    if not image_sets:
        return
    translated = pipeline if isinstance(pipeline, Pipeline) else load_pipeline(pipeline)
    if workers == 1 or len(image_sets) == 1:
        for image_id, sources in image_sets.items():
            with _image_set_context(image_id):
                result = _run_image_set(translated, image_id, sources)
            yield result
        return

    inputs = iter(image_sets.items())
    limit = min(workers, len(image_sets))
    executor = (
        ProcessPoolExecutor(max_workers=limit, mp_context=get_context("spawn"))
        if backend == "process"
        else ThreadPoolExecutor(max_workers=limit)
    )

    def submit(
        image_id: str, sources: Mapping[str, str | Path | np.ndarray]
    ) -> tuple[str, Future[ImageSetResult]]:
        with _image_set_context(image_id):
            # Some mappings (such as MappingProxyType) cannot cross processes.
            return image_id, executor.submit(
                _run_image_set, translated, image_id, dict(sources)
            )

    try:
        pending = deque(submit(*item) for item in islice(inputs, limit))
        while pending:
            yield _result(*pending.popleft())
            if (item := next(inputs, None)) is not None:
                pending.append(submit(*item))
    finally:
        executor.shutdown(cancel_futures=True)
