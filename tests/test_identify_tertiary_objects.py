"""Exercise the native tertiary-object API, including label preservation."""

import numpy as np
import pytest

import cp_anywhere


def test_default_shrink_removes_only_primary_interior() -> None:
    primary = np.zeros((5, 5), dtype=np.uint32)
    primary[1:4, 1:4] = 1
    secondary = np.full((5, 5), 7, dtype=np.uint32)
    expected = secondary.copy()
    expected[2, 2] = 0
    result = cp_anywhere.identify_tertiary_objects(primary, secondary)
    np.testing.assert_array_equal(result.labels, expected)
    np.testing.assert_array_equal(secondary, np.full((5, 5), 7, dtype=np.uint32))


def test_no_shrink_subtracts_all_primary_pixels_without_relabeling() -> None:
    primary = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint32)
    secondary = np.full((2, 3), 9, dtype=np.uint32)
    result = cp_anywhere.identify_tertiary_objects(
        primary, secondary, shrink_primary=False
    )
    np.testing.assert_array_equal(result.labels, [[9, 0, 9], [9, 0, 9]])


@pytest.mark.parametrize(
    "primary,secondary",
    [
        (np.zeros((2, 3)), np.zeros((3, 2))),
        (np.zeros((0, 2)), np.zeros((0, 2))),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2))),
        (np.array([[-1, 0], [0, 0]]), np.ones((2, 2), dtype=int)),
        (np.array([[0.5, 0], [0, 0]]), np.ones((2, 2), dtype=int)),
        (np.zeros((2, 2), dtype=int), np.full((2, 2), 2**32, dtype=np.uint64)),
    ],
)
def test_rejects_invalid_label_arrays(
    primary: np.ndarray, secondary: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        cp_anywhere.identify_tertiary_objects(primary, secondary)
