"""Replay the CellProfiler image order and ImageMath output."""

import json
from pathlib import Path

import numpy as np

from cp_anywhere import load_pipeline

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/cppipe/illumination/image_math_order.cppipe"
)


def test_image_math_runs_in_file_order() -> None:
    pipeline = load_pipeline(FIXTURE)
    inputs = {
        "Raw": np.array([[0.0, 0.25, 0.5], [0.75, 1.0, np.nan]], dtype=np.float32),
        "Flat": np.full((2, 3), 0.5, dtype=np.float32),
    }
    expected = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert not pipeline.skipped_modules
    assert pipeline.run(inputs) == ({}, {}, None)
    actual = pipeline.prepare_images(inputs)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    np.testing.assert_array_equal(inputs["Raw"], [[0, 0.25, 0.5], [0.75, 1, np.nan]])
