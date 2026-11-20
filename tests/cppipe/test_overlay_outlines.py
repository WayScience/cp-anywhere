"""Replay the original OverlayOutlines settings and saved CellProfiler pixels."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.cppipe import CppipeError, load_pipeline

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cppipe/overlay_outlines"


def _inputs(planes: dict) -> dict[str, np.ndarray]:
    return {
        key: np.array(plane["values"], dtype=plane["dtype"]).reshape(plane["shape"])
        for key, plane in planes.items()
    }


def test_renders_cellprofiler_outline_images() -> None:
    oracle = json.loads((FIXTURE / "overlay.json").read_text(encoding="utf-8"))
    pipeline = load_pipeline(FIXTURE / "overlay.cppipe")
    images = _inputs(oracle["images"])
    labels = _inputs(oracle["labels"])
    working = dict(images)
    pipeline.render_overlays(working, labels)
    detailed = pipeline.run_detailed(images | labels)
    assert not detailed.skipped
    assert len(pipeline.overlay_outlines_steps) == len(oracle["outputs"])
    assert set(working) - images.keys() == oracle["outputs"].keys()
    assert detailed.derived_images.keys() == oracle["outputs"].keys()
    for name, expected in _inputs(oracle["outputs"]).items():
        assert working[name].dtype == np.float32
        np.testing.assert_array_equal(working[name], detailed.derived_images[name])
        np.testing.assert_allclose(
            working[name], expected, rtol=3 * float(np.finfo(np.float32).eps), atol=0
        )
    for name, original in _inputs(oracle["images"]).items():
        np.testing.assert_array_equal(working[name], original)
    for name, original in _inputs(oracle["labels"]).items():
        np.testing.assert_array_equal(labels[name], original)


@pytest.mark.parametrize("pixels", [np.array([[0.5]]), np.array([[-1]])])
def test_rejects_invalid_raw_label_arrays(pixels: np.ndarray) -> None:
    pipeline = load_pipeline(FIXTURE / "overlay.cppipe")
    with pytest.raises(CppipeError, match=r"OverlayOutlines.*Cells.*integers"):
        pipeline.render_overlays({"Cells": pixels}, {})


def test_rejects_forward_background_reference(tmp_path: Path) -> None:
    text = (FIXTURE / "overlay.cppipe").read_text(encoding="utf-8")
    source = tmp_path / "forward.cppipe"
    text = text.replace(
        "Select image on which to display outlines:DNA",
        "Select image on which to display outlines:RgbMax",
        1,
    )
    assert "Select image on which to display outlines:RgbMax" in text
    source.write_text(text, encoding="utf-8")
    with pytest.raises(CppipeError, match=r"OverlayOutlines.*RgbMax.*before"):
        load_pipeline(source)
