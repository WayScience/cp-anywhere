"""Display 2D segmentation results without changing the source arrays."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

_OUTLINE = (0, 255, 128)
_PALETTE = np.array(
    [
        (86, 180, 233),
        (230, 159, 0),
        (0, 158, 115),
        (204, 121, 167),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (180, 166, 242),
    ],
    dtype=np.uint8,
)
_IMAGE_DIMENSIONS = 2


def _grayscale(image: np.ndarray, shape: tuple[int, ...]) -> NDArray[np.uint8]:
    if image.shape != shape:
        raise ValueError("The grayscale image and labels must have the same 2D shape")
    if image.dtype.kind not in "uif":
        raise TypeError("The grayscale image must have a real numeric dtype")
    pixels = image.astype(np.float64, copy=False)
    valid = np.isfinite(pixels)
    gray = np.zeros(shape, dtype=np.uint8)
    values = pixels[valid]
    if values.size:
        low, high = float(values.min()), float(values.max())
        if low < high:
            if not np.isfinite(high - low):
                # Halving avoids overflow for ranges spanning both float64 extremes.
                values, low, high = values * 0.5, low * 0.5, high * 0.5
            gray[valid] = np.rint((values - low) / (high - low) * 255).astype(np.uint8)
    return gray


def _inner_outlines(labels: np.ndarray) -> NDArray[np.bool_]:
    edges = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical
    edges[[0, -1], :] = True
    edges[:, [0, -1]] = True
    return edges & (labels != 0)


def preview_segmentation(
    labels: np.ndarray,
    image: np.ndarray | None = None,
    *,
    mode: Literal["outlines", "labels"] = "outlines",
) -> NDArray[np.uint8]:
    """Return an 8-bit RGB preview of one 2D label image.

    Zero is background. Positive integers are object IDs and may have gaps.
    In ``outlines`` mode, draw green lines on the object side of boundaries.
    A boundary touches a different label or the image edge along a pixel side.
    Supply a same-shape grayscale ``image`` to see the outlines in context.
    The finite image range maps to black through white. Nonfinite pixels and
    constant images display as black. With no image, the background is black.

    In ``labels`` mode, omit ``image``. Each object gets a color based on its
    original ID, and the palette repeats. The background stays black.

    This function does not change the image or labels. Use the returned array
    for display or export, not as a substitute for the original labels.
    """
    labels = np.asarray(labels)
    if labels.ndim != _IMAGE_DIMENSIONS or not labels.size:
        raise ValueError("Labels must be a nonempty 2D array")
    if labels.dtype.kind not in "ui":
        raise TypeError("Labels must have an integer dtype")
    if labels.dtype.kind == "i" and np.any(labels < 0):
        raise ValueError("Labels must be nonnegative")
    if mode == "labels":
        if image is not None:
            raise ValueError("Omit the grayscale image in 'labels' mode")
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        foreground = labels != 0
        rgb[foreground] = _PALETTE[(labels[foreground] - 1) % len(_PALETTE)]
        return rgb
    if mode != "outlines":
        raise ValueError("Mode must be 'outlines' or 'labels'")
    if image is None:
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    else:
        gray = _grayscale(np.asarray(image), labels.shape)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[_inner_outlines(labels)] = _OUTLINE
    return rgb
