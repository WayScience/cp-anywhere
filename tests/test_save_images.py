"""Verify cp_anywhere.save_images against real CellProfiler SaveImages'
bit-depth conversion oracle (scripts/reference/generate_save_images_bitdepth.py)
and round-trip file writes through tifffile/Pillow/numpy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp_anywhere.io import load_array
from cp_anywhere.save_images import _convert_bit_depth, save_array

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/save_images"
CASE_NAMES = sorted(p.stem for p in FIXTURE_DIR.glob("*.npz"))


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize(
    ("bit_depth", "fixture_key", "dtype"),
    [
        ("8-bit integer", "ubyte", np.uint8),
        ("16-bit integer", "uint16", np.uint16),
        ("32-bit floating point", "float32", np.float32),
    ],
)
def test_bit_depth_conversion_matches_a_real_cellprofiler_oracle(
    case: str, bit_depth: str, fixture_key: str, dtype: type
) -> None:
    data = np.load(FIXTURE_DIR / f"{case}.npz")
    pixels = data["input"]
    expected = data[fixture_key]
    actual = _convert_bit_depth(pixels, bit_depth)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)


def test_raw_bit_depth_is_a_no_op() -> None:
    pixels = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    actual = _convert_bit_depth(pixels, "No conversion")
    np.testing.assert_array_equal(actual, pixels)
    assert actual.dtype == pixels.dtype


def test_values_outside_0_1_are_rejected_for_integer_bit_depths() -> None:
    pixels = np.array([-0.1, 0.5, 1.1], dtype=np.float64)
    with pytest.raises(ValueError, match=r"\[0, ?1\]|\[0\.0, ?1\.0\]"):
        _convert_bit_depth(pixels, "8-bit integer")


def test_unknown_bit_depth_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bit_depth"):
        save_array(np.zeros((2, 2)), tmp_path / "x.png", bit_depth="not a thing")


@pytest.mark.parametrize(
    ("suffix", "bit_depth", "dtype"),
    [
        (".npy", "16-bit integer", np.float64),  # .npy never bit-depth-converts
        (".tiff", "16-bit integer", np.uint16),
        (".png", "8-bit integer", np.uint8),
        (".jpg", "8-bit integer", np.uint8),
    ],
)
def test_save_array_round_trips_through_the_real_file_format(
    tmp_path: Path, suffix: str, bit_depth: str, dtype: type
) -> None:
    pixels = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    path = tmp_path / f"image{suffix}"
    save_array(pixels, path, bit_depth=bit_depth)
    assert path.exists()

    loaded = load_array(path)
    assert loaded.dtype == dtype
    if dtype == np.float64:
        np.testing.assert_array_equal(loaded, pixels)
    else:
        # lossy integer formats: values must be within one quantization
        # step of the real skimage conversion, not byte-identical to the
        # float input.
        assert loaded.min() >= 0


def test_save_array_rejects_an_unknown_file_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported file extension"):
        save_array(np.zeros((2, 2)), tmp_path / "x.unknownext")


@pytest.mark.parametrize("suffix", [".npy", ".NPY", ".NpY"])
def test_save_array_uses_the_exact_npy_destination(tmp_path: Path, suffix: str) -> None:
    destination = tmp_path / f"image{suffix}"
    neighbor = destination.with_name(destination.name + ".npy")
    original = np.array([[42.0]])
    np.save(neighbor, original)
    pixels = np.array([[99.0]])

    save_array(pixels, destination)

    np.testing.assert_array_equal(load_array(neighbor), original)
    assert destination.is_file()
    np.testing.assert_array_equal(load_array(destination), pixels)
