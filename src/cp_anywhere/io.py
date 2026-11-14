"""Load images and label arrays directly from common file formats.

This is the simple entry point for users with ordinary microscopy files
(TIFF/OME-TIFF, PNG, or saved NumPy arrays) who do not want to pre-convert
anything by hand. ``measure_files`` wraps :func:`cp_anywhere.api.measure`
and returns a tidy ``pandas.DataFrame`` instead of a ``MeasurementResult``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cp_anywhere.api import measure

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

_NPY_SUFFIXES = (".npy",)
_TIFF_SUFFIXES = (".tif", ".tiff")
_PILLOW_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def load_array(source: object) -> np.ndarray:
    """Load one image or label array from a file path, or pass an array through.

    Dispatches on file suffix: ``.npy`` via :func:`numpy.load`, ``.tif``/
    ``.tiff`` via ``tifffile``, and ``.png``/``.jpg``/``.jpeg``/``.bmp``/
    ``.gif`` via Pillow. An already-loaded :class:`numpy.ndarray` is
    returned unchanged.
    """
    if isinstance(source, np.ndarray):
        return source
    if not isinstance(source, (str, Path)):
        raise TypeError(
            "expected a file path (str/Path) or a numpy.ndarray, "
            f"got {type(source).__name__}"
        )
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in _NPY_SUFFIXES:
        return np.load(path, allow_pickle=False)
    if suffix in _TIFF_SUFFIXES:
        import tifffile

        return tifffile.imread(path)
    if suffix in _PILLOW_SUFFIXES:
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image)
    raise ValueError(
        f"unsupported file extension {suffix!r}; use .npy, .tif/.tiff, "
        "or .png/.jpg/.jpeg/.bmp/.gif, or pass an already-loaded array"
    )


def _as_named_map(value: object, default_name: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {default_name: value}


def measure_files(
    images: Mapping[str, object] | str | Path,
    objects: Mapping[str, object] | str | Path,
    *,
    mask: object | None = None,
    image_id: str = "Image",
    families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Measure images and objects given as file paths (or arrays), as a table.

    ``images``/``objects`` may each be a single path/array (named "DNA" and
    "Cells" respectively) or a mapping of channel/object-set name to
    path/array, exactly like :func:`cp_anywhere.api.measure` but accepting
    file paths for any supported format (see :func:`load_array`) in
    addition to already-loaded arrays. Returns one row per object per
    feature column (see :meth:`cp_anywhere.api.MeasurementResult.rows`).
    """
    import pandas as pd

    image_map = _as_named_map(images, "DNA")
    object_map = _as_named_map(objects, "Cells")
    loaded_images = {name: load_array(value) for name, value in image_map.items()}
    loaded_objects = {name: load_array(value) for name, value in object_map.items()}
    loaded_mask = load_array(mask) if mask is not None else None
    result = measure(
        loaded_images,
        loaded_objects,
        mask=loaded_mask,
        image_id=image_id,
        families=families,
    )
    return pd.DataFrame(result.rows())
