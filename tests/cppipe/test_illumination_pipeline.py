"""Verify cp_anywhere.cppipe's CorrectIlluminationCalculate/Apply
translation against a real .cppipe pipeline and a fresh CellProfiler
4.2.8 oracle run of that exact combined pipeline (see
scripts/cppipe/generate_illumination_pipeline.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cp_anywhere import apply_illumination, calculate_illumination
from cp_anywhere.cppipe import CppipeError, SaveImagesStep, load_pipeline

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures/cppipe/illumination"


def test_illumination_calculate_apply_matches_a_fresh_oracle_run() -> None:
    name = "illumination_calculate_apply"
    pipeline = load_pipeline(FIXTURE_DIR / f"{name}.cppipe")

    assert len(pipeline.illumination_calculate_steps) == 1
    calculate_step = pipeline.illumination_calculate_steps[0]
    assert calculate_step.input_image == "DNA"
    assert calculate_step.output_image == "Illum"
    assert calculate_step.smoothing_method == "Gaussian Filter"

    assert len(pipeline.illumination_apply_steps) == 1
    apply_step = pipeline.illumination_apply_steps[0]
    assert len(apply_step.images) == 1
    assert apply_step.images[0].input_image == "DNA"
    assert apply_step.images[0].output_image == "DNACorrected"
    assert apply_step.images[0].illumination_function_image == "Illum"
    assert apply_step.images[0].divide_or_subtract == "Divide"

    image = np.load(FIXTURE_DIR / f"{name}.test_image.npy")
    expected = json.loads((FIXTURE_DIR / f"{name}.expected.json").read_text())
    _segmentations, _secondary, _measurement = pipeline.run({"DNA": image})
    # Pipeline.run() has no measurement steps here, so it only
    # exercises the illumination correction itself; reach the
    # corrected image through a fresh run with the same steps to
    # confirm against the real oracle (run() doesn't expose
    # intermediate images, only segmentation/measurement results).
    illum = calculate_illumination(
        image,
        intensity_choice=calculate_step.intensity_choice,
        block_size=calculate_step.block_size,
        dilate_objects=calculate_step.dilate_objects,
        object_dilation_radius=calculate_step.object_dilation_radius,
        smoothing_method=calculate_step.smoothing_method,
        automatic_object_width=calculate_step.automatic_object_width,
        object_width=calculate_step.object_width,
        size_of_smoothing_filter=calculate_step.size_of_smoothing_filter,
        rescale_option=calculate_step.rescale_option,
    )
    corrected = apply_illumination(
        image,
        illum,
        divide_or_subtract=apply_step.images[0].divide_or_subtract,
        truncate_low=apply_step.truncate_low,
        truncate_high=apply_step.truncate_high,
    )
    np.testing.assert_allclose(
        corrected, np.array(expected["corrected"]), rtol=1e-6, atol=1e-6
    )


def test_illumination_calculate_rejects_all_across_cycles() -> None:
    text = (FIXTURE_DIR / "illumination_calculate_apply.cppipe").read_text()
    text = text.replace(
        "Calculate function for each image individually, or based on all images?:Each",
        "Calculate function for each image individually, or based on all images?:"
        "All: Across cycles",
    )
    path = FIXTURE_DIR / "_tmp_all_across.cppipe"
    path.write_text(text)
    try:
        with pytest.raises(CppipeError, match="calculate_illumination_across"):
            load_pipeline(path)
    finally:
        path.unlink()


def test_illumination_calculate_rejects_convex_hull_and_splines() -> None:
    for method in ("Convex Hull", "Splines"):
        text = (FIXTURE_DIR / "illumination_calculate_apply.cppipe").read_text()
        text = text.replace(
            "Smoothing method:Gaussian Filter", f"Smoothing method:{method}"
        )
        path = FIXTURE_DIR / "_tmp_smoothing.cppipe"
        path.write_text(text)
        try:
            with pytest.raises(CppipeError, match="not yet implemented"):
                load_pipeline(path)
        finally:
            path.unlink()


def test_illumination_run_rejects_a_missing_input_image() -> None:
    pipeline = load_pipeline(FIXTURE_DIR / "illumination_calculate_apply.cppipe")
    with pytest.raises(CppipeError, match="CorrectIlluminationCalculate"):
        pipeline.run({})


def test_save_images_is_translated_and_writes_a_real_file(tmp_path: Path) -> None:
    pipeline = load_pipeline(FIXTURE_DIR / "save_images.cppipe")

    assert len(pipeline.save_images_steps) == 1
    step = pipeline.save_images_steps[0]
    assert step.image_name == "DNACorrected"
    assert step.file_name == "dna_corrected"
    assert step.file_format == "tiff"
    assert step.bit_depth == "16-bit integer"
    assert step.overwrite is True
    assert step.effective_bit_depth == "16-bit integer"

    image = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(4, 4)
    path = tmp_path / "out.tiff"
    step.write(image, path)
    assert path.exists()

    loaded = tifffile.imread(path)
    assert loaded.dtype == np.uint16


def test_save_images_overwrite_false_refuses_an_existing_file(tmp_path: Path) -> None:
    text = (FIXTURE_DIR / "save_images.cppipe").read_text()
    text = text.replace(
        "Overwrite existing files without warning?:Yes",
        "Overwrite existing files without warning?:No",
    )
    path = FIXTURE_DIR / "_tmp_no_overwrite.cppipe"
    path.write_text(text)
    try:
        pipeline = load_pipeline(path)
        step = pipeline.save_images_steps[0]
        assert step.overwrite is False
        existing = tmp_path / "out.tiff"
        existing.write_bytes(b"not a real tiff")
        with pytest.raises(CppipeError, match="already exists"):
            step.write(np.zeros((2, 2)), existing)
    finally:
        path.unlink()


def test_save_images_rejects_unsupported_file_name_methods_and_formats() -> None:
    for old, new in (
        (
            "Select method for constructing file names:Single name",
            "Select method for constructing file names:Sequential numbers",
        ),
        ("Saved file format:tiff", "Saved file format:h5"),
        (
            "Select the type of image to save:Image",
            "Select the type of image to save:Mask",
        ),
    ):
        text = (FIXTURE_DIR / "save_images.cppipe").read_text()
        assert old in text
        text = text.replace(old, new)
        path = FIXTURE_DIR / "_tmp_unsupported.cppipe"
        path.write_text(text)
        try:
            with pytest.raises(CppipeError, match="out-of-scope"):
                load_pipeline(path)
        finally:
            path.unlink()


def test_save_images_png_is_always_forced_to_8_bit() -> None:
    text = (FIXTURE_DIR / "save_images.cppipe").read_text()
    text = text.replace("Saved file format:tiff", "Saved file format:png")
    path = FIXTURE_DIR / "_tmp_png.cppipe"
    path.write_text(text)
    try:
        pipeline = load_pipeline(path)
        step = pipeline.save_images_steps[0]
        assert step.bit_depth == "16-bit integer"  # unchanged raw setting
        assert step.effective_bit_depth == "8-bit integer"  # CP's own override
    finally:
        path.unlink()


@pytest.mark.parametrize("suffix", [".npy", ".NPY", ".NpY"])
def test_save_images_npy_overwrite_policy_uses_the_exact_destination(
    tmp_path: Path, suffix: str
) -> None:
    step = SaveImagesStep("DNA", "image", "npy", "8-bit integer", False)
    destination = tmp_path / f"image{suffix}"
    neighbor = destination.with_name(destination.name + ".npy")
    original = np.array([[42.0]])
    np.save(neighbor, original)
    pixels = np.array([[99.0]])

    step.write(pixels, destination)

    np.testing.assert_array_equal(np.load(neighbor), original)
    assert destination.is_file()
    np.testing.assert_array_equal(np.load(destination), pixels)
    with pytest.raises(CppipeError, match="already exists"):
        step.write(original, destination)
    np.testing.assert_array_equal(np.load(destination), pixels)
    np.testing.assert_array_equal(np.load(neighbor), original)
