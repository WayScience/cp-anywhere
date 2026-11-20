"""Check threshold decisions against real CellProfiler FlagImage output."""

import json
from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.cppipe import CppipeError, load_pipeline

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/cppipe/image_quality"


@pytest.mark.parametrize("mode", ["any", "all"])
def test_flag_image_whole_image_measurements(mode: str) -> None:
    pipeline = load_pipeline(FIXTURES / f"flag_{mode}.cppipe")
    assert pipeline.skipped_modules == []
    (step,) = pipeline.flag_image_steps
    oracle = json.loads((FIXTURES / "flags.json").read_text())
    for (slope, percent), (flag, skip) in zip(
        oracle["inputs"], oracle["modes"][mode], strict=True
    ):
        assert step.evaluate(
            {"Slope": float("nan") if slope is None else slope, "Percent": percent}
        ) == ({"QC_Rejected": flag}, skip)
    with pytest.raises(CppipeError, match=r"FlagImage.*Percent"):
        step.evaluate({"Slope": -2.0})


@pytest.mark.parametrize("case", ["kept", "skipped"])
def test_quality_controls_later_image_preparation(case: str) -> None:
    pipeline = load_pipeline(FIXTURES / "quality_flag.cppipe")
    pixels = np.load(FIXTURES / f"quality_flag.{case}.npy")
    oracle = json.loads((FIXTURES / "quality_flag.json").read_text())[case]
    result = pipeline.run_detailed({"DNA": pixels}, scaling={"DNA": 4095})
    assert result.skipped == oracle["skipped"]
    assert result.measurements is not None
    columns = result.measurements.features["Image"]
    assert columns.keys() == oracle["columns"].keys()
    for name, value in oracle["columns"].items():
        # Use the same new-metric tolerance as the Rust oracle for the
        # different FFT implementations. The flag itself must match exactly.
        is_slope = name.startswith("ImageQuality_PowerLogLogSlope_")
        np.testing.assert_allclose(
            columns[name],
            value,
            rtol=2e-6 if is_slope else 1e-7,
            atol=2e-6 if is_slope else 1e-10,
            err_msg=name,
        )
    if result.skipped:
        assert result.derived_images == {}
    else:
        np.testing.assert_array_equal(result.derived_images["AfterQC"], pixels)
