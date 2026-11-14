"""Type stub for the NumPy native measurement binding."""

from typing import Any

import numpy as np

def measure(
    shape: tuple[int, ...],
    images: dict[str, np.ndarray],
    objects: dict[str, np.ndarray],
    mask: np.ndarray | None,
    image_id: str,
    families: tuple[str, ...],
) -> dict[str, Any]: ...
