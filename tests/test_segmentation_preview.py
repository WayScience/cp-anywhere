"""Test the display pixels independently of the segmentation algorithm."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import cp_anywhere as cp


def test_preview_draws_inner_outlines_without_changing_inputs() -> None:
    image = np.full((5, 5), 30, dtype=np.uint16)
    image[0, 0], image[0, 1] = 0, 60
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 9
    original_image, original_labels = image.copy(), labels.copy()

    preview = cp.preview_segmentation(labels, image)

    expected = np.repeat(
        np.rint(image.astype(float) * (255 / 60)).astype(np.uint8)[..., None],
        3,
        axis=-1,
    )
    expected[1:4, 1:4] = (0, 255, 128)
    expected[2, 2] = (128, 128, 128)
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(preview, expected)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(labels, original_labels)


def test_preview_outlines_touching_objects_at_both_sides_of_the_boundary() -> None:
    labels = np.tile(np.array([4, 4, 4, 7, 7, 7], dtype=np.uint8), (3, 1))
    labels.setflags(write=False)

    preview = cp.preview_segmentation(labels)

    green = (0, 255, 128)
    black = (0, 0, 0)
    np.testing.assert_array_equal(
        preview[1], [green, black, green, green, black, green]
    )
    np.testing.assert_array_equal(preview[0], [green] * labels.shape[1])
    np.testing.assert_array_equal(preview[-1], preview[0])


def test_preview_label_colors_are_stable_for_sparse_large_ids() -> None:
    large = np.iinfo(np.uint64).max
    labels = np.array([[0, 9, large], [2, large, 9]], dtype=np.uint64)

    preview = cp.preview_segmentation(labels, mode="labels")

    np.testing.assert_array_equal(preview[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(preview[0, 1], preview[1, 2])
    np.testing.assert_array_equal(preview[0, 2], preview[1, 1])
    assert not np.array_equal(preview[0, 1], preview[0, 2])
    single = cp.preview_segmentation(labels[:1, 1:2], mode="labels")
    np.testing.assert_array_equal(preview[0, 1], single[0, 0])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([np.nan, -np.inf, np.inf], [0, 0, 0]),
        ([7.0, 7.0, 7.0], [0, 0, 0]),
        ([np.nan, -10.0, 10.0], [0, 0, 255]),
        ([-np.finfo(float).max, 0, np.finfo(float).max], [0, 128, 255]),
    ],
)
def test_preview_grayscale_has_safe_display_scaling(
    values: list[float], expected: list[int]
) -> None:
    image = np.array([values])
    preview = cp.preview_segmentation(np.zeros(image.shape, dtype=np.uint8), image)
    np.testing.assert_array_equal(
        preview[0], np.repeat(np.array(expected)[:, None], 3, axis=1)
    )


@pytest.mark.parametrize(
    ("labels", "image", "error", "message"),
    [
        (np.empty((0, 2), dtype=int), None, ValueError, "nonempty 2D"),
        (np.zeros((2, 2, 2), dtype=int), None, ValueError, "nonempty 2D"),
        (np.array([[0.0, 1.0]]), None, TypeError, "integer dtype"),
        (np.array([[-1, 1]]), None, ValueError, "nonnegative"),
        (np.array([[0, 1]]), np.zeros((1, 2, 3)), ValueError, "same 2D shape"),
        (np.array([[0, 1]]), np.array([[0j, 1j]]), TypeError, "numeric dtype"),
    ],
)
def test_preview_rejects_invalid_inputs(
    labels: np.ndarray, image: np.ndarray | None, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        cp.preview_segmentation(labels, image)


def test_preview_saves_the_exact_rgb_pixels_as_png(tmp_path: Path) -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[2:5, 2:5] = 255
    seg = cp.identify_primary_objects(image, threshold=0.5)
    preview = cp.preview_segmentation(seg.labels, image)
    target = tmp_path / "preview.png"

    cp.save_array(preview, target, bit_depth="No conversion")

    with Image.open(target) as saved:
        assert saved.mode == "RGB"
        np.testing.assert_array_equal(saved, preview)
