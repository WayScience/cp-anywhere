"""CellProfiler-compatible ``SaveImages`` -- write an image array to disk
the same way real CellProfiler's own ``SaveImages`` module does.

Scope: ``npy``/``tiff``/``png``/``jpeg`` (``FF_H5`` and the movie/stack
TIFF path, which read every cycle's frames via bioformats, are a
separate, not-yet-implemented follow-on -- see ``spec/illumination.md``).

Real CellProfiler's ``save_image`` (``cellprofiler.modules.saveimages``)
does the bit-depth conversion via ``skimage.util.img_as_ubyte``/
``img_as_uint``/``img_as_float32`` before handing the array to
``skimage.io.imsave``, EXCEPT for ``npy``, which calls ``numpy.save``
directly and bypasses bit-depth conversion entirely. The ``img_as_*``
formula (read from ``skimage/util/dtype.py``'s ``_convert``, the
``kind_in == 'f'`` branch with the default ``uniform=False``) is::

    scaled = pixels * imax_out            # imax_out = 255 or 65535
    scaled = numpy.rint(scaled)           # round-half-to-even
    scaled = numpy.clip(scaled, 0, imax_out)
    return scaled.astype(uint8_or_uint16)

verified byte-exact against a live CellProfiler 4.2.8 install in
scripts/reference/generate_save_images_bitdepth.py / fixtures/save_images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BIT_DEPTH_8 = "8-bit integer"
BIT_DEPTH_16 = "16-bit integer"
BIT_DEPTH_FLOAT = "32-bit floating point"
BIT_DEPTH_RAW = "No conversion"
_BIT_DEPTHS = (BIT_DEPTH_8, BIT_DEPTH_16, BIT_DEPTH_FLOAT, BIT_DEPTH_RAW)

_NPY_SUFFIXES = (".npy",)
_TIFF_SUFFIXES = (".tif", ".tiff")
_PILLOW_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def _convert_bit_depth(pixels: np.ndarray, bit_depth: str) -> np.ndarray:
    """Match ``skimage.util.img_as_ubyte``/``img_as_uint``/``img_as_float32``
    for the one case real ``SaveImages`` ever feeds them: a float array
    (this project's own image convention, see ``cp_anywhere.io``)."""
    if bit_depth == BIT_DEPTH_RAW:
        return pixels
    if bit_depth == BIT_DEPTH_FLOAT:
        return pixels.astype(np.float32)
    if bit_depth not in (BIT_DEPTH_8, BIT_DEPTH_16):
        msg = f"unknown bit_depth {bit_depth!r}; expected one of {_BIT_DEPTHS!r}"
        raise ValueError(msg)
    if pixels.min() < 0.0 or pixels.max() > 1.0:
        msg = (
            "SaveImages bit-depth conversion needs pixels in [0, 1] "
            f"(got min={pixels.min()!r}, max={pixels.max()!r})"
        )
        raise ValueError(msg)
    imax = 255 if bit_depth == BIT_DEPTH_8 else 65535
    dtype = np.uint8 if bit_depth == BIT_DEPTH_8 else np.uint16
    scaled = pixels * imax
    scaled = np.rint(scaled)
    scaled = np.clip(scaled, 0, imax)
    return scaled.astype(dtype)


def save_array(
    image: np.ndarray,
    path: str | Path,
    *,
    bit_depth: str = BIT_DEPTH_16,
) -> None:
    """Write ``image`` to ``path`` the way real CellProfiler's
    ``SaveImages`` module does, dispatching on the file suffix exactly
    like :func:`cp_anywhere.io.load_array` does for reading.

    ``bit_depth`` is one of ``"8-bit integer"``, ``"16-bit integer"``,
    ``"32-bit floating point"``, or ``"No conversion"`` (real
    CellProfiler's own setting values -- see ``BIT_DEPTH_*`` above).
    ``.npy`` bypasses ``bit_depth`` entirely, matching real
    CellProfiler's own ``FF_NPY`` branch (plain ``numpy.save``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _NPY_SUFFIXES:
        # A stream prevents NumPy from appending a lowercase .npy suffix.
        # Keep the actual destination identical to the caller's path.
        with path.open("wb") as stream:
            np.save(stream, image)
        return
    pixels = _convert_bit_depth(image, bit_depth)
    if suffix in _TIFF_SUFFIXES:
        import tifffile

        tifffile.imwrite(path, pixels)
        return
    if suffix in _PILLOW_SUFFIXES:
        from PIL import Image as PILImage

        PILImage.fromarray(pixels).save(path)
        return
    msg = (
        f"unsupported file extension {suffix!r}; use .npy, .tif/.tiff, "
        "or .png/.jpg/.jpeg/.bmp/.gif"
    )
    raise ValueError(msg)
