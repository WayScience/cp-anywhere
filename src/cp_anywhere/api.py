"""NumPy input validation and the public CP-Anywhere measurement result."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FAMILIES = ("image_intensity", "object_intensity", "neighbors")
MAX_LABEL = 1_000_000
MAX_IMAGE_DTYPE_ITEMSIZE = 8  # bytes; i.e. up to a 64-bit int/uint/float


@dataclass(frozen=True)
class MeasurementResult:
    """Typed feature columns and directed neighbor edges for one image."""

    image_id: str
    features: dict[str, dict[str, float | int | np.ndarray]]
    object_ids: dict[str, np.ndarray]
    relationships: list[dict[str, str | int]]

    def rows(self) -> list[dict[str, str | int | float]]:
        """Return a long table with one row per feature and object ID."""
        rows: list[dict[str, str | int | float]] = []
        for scope, columns in self.features.items():
            for name, values in columns.items():
                if scope == "Image":
                    rows.append(
                        {
                            "image_id": self.image_id,
                            "scope": scope,
                            "object_id": 0,
                            "feature": name,
                            "value": float(values),
                        }
                    )
                else:
                    for object_id, value in zip(
                        self.object_ids[scope], np.asarray(values), strict=True
                    ):
                        rows.append(
                            {
                                "image_id": self.image_id,
                                "scope": scope,
                                "object_id": int(object_id),
                                "feature": name,
                                "value": value.item(),
                            }
                        )
        return rows

    def to_csv(self, path: str | Path) -> None:
        """Write long-form feature rows with nonfinite values kept as strings."""
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("image_id", "scope", "object_id", "feature", "value"),
            )
            writer.writeheader()
            writer.writerows(self.rows())

    def to_arrow(self, path: str | Path) -> None:
        """Write every feature row as an Arrow IPC (Feather v2) file.

        One row per object per feature column, the same long-form shape as
        :meth:`rows`/:meth:`to_csv`, covering all feature data (every
        scope, object, and column) in a single typed Arrow table --
        ``image_id``/``scope``/``feature`` as strings, ``object_id`` as
        int64, ``value`` as float64 (NaN/Infinity preserved natively,
        unlike :meth:`to_json`'s string markers).
        """
        import pyarrow as pa
        from pyarrow import feather

        rows = self.rows()
        table = pa.table(
            {
                "image_id": pa.array(
                    [row["image_id"] for row in rows], type=pa.string()
                ),
                "scope": pa.array([row["scope"] for row in rows], type=pa.string()),
                "object_id": pa.array(
                    [row["object_id"] for row in rows], type=pa.int64()
                ),
                "feature": pa.array([row["feature"] for row in rows], type=pa.string()),
                "value": pa.array([row["value"] for row in rows], type=pa.float64()),
            }
        )
        feather.write_feather(table, str(path))

    def to_json(self, path: str | Path) -> None:
        """Write the reference-compatible feature and relationship structure."""

        def encode(value: object) -> object:
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return [encode(item) for item in value]
            if isinstance(value, np.generic):
                return encode(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return {
                    "nonfinite": "NaN"
                    if np.isnan(value)
                    else "+Infinity"
                    if value > 0
                    else "-Infinity"
                }
            return value

        payload = {
            "image_id": self.image_id,
            "features": encode(self.features),
            "object_ids": encode(self.object_ids),
            "relationships": self.relationships,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _image_array(value: object, shape: tuple[int, ...] | None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3) or 0 in array.shape:
        raise ValueError("images must have two or three nonzero dimensions")
    if shape is not None and array.shape != shape:
        raise ValueError("all inputs must have the same shape")
    if array.dtype.kind not in "iuf" or array.dtype.itemsize > MAX_IMAGE_DTYPE_ITEMSIZE:
        raise TypeError("images must have a supported real numeric dtype")
    return np.ascontiguousarray(array, dtype=np.float64)


def _label_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("all inputs must have the same shape")
    if array.dtype.kind not in "iu":
        raise TypeError("labels must have an integer dtype")
    if array.size and (np.min(array) < 0 or np.max(array) > MAX_LABEL):
        raise ValueError(f"labels must be between zero and {MAX_LABEL}")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _validate_families(families: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(FAMILIES if families is None else families)
    if len(selected) != len(set(selected)) or any(
        name not in FAMILIES for name in selected
    ):
        raise ValueError(f"families must be drawn from {FAMILIES}")
    return selected


def _convert_images(
    images: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    converted: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name, value in images.items():
        if not isinstance(name, str) or not name:
            raise ValueError("image names must be nonempty strings")
        array = _image_array(value, shape)
        shape = array.shape
        converted[name] = array
    if shape is None:
        raise ValueError("images and objects must be nonempty")
    return converted, shape


def _convert_objects(
    objects: Mapping[str, object], shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    if any(not isinstance(name, str) or not name for name in objects):
        raise ValueError("object names must be nonempty strings")
    return {name: _label_array(value, shape) for name, value in objects.items()}


def _convert_mask(mask: object | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    converted = np.asarray(mask)
    if converted.shape != shape or converted.dtype.kind != "b":
        raise ValueError("mask must be a boolean array with the image shape")
    return np.ascontiguousarray(converted)


def measure(
    images: Mapping[str, object],
    objects: Mapping[str, object],
    *,
    mask: object | None = None,
    image_id: str = "Image",
    families: Sequence[str] | None = None,
) -> MeasurementResult:
    """Measure 2D or 3D arrays with the selected CellProfiler 4.2.8 settings.

    Float64 C-contiguous images are borrowed by Rust. Other numeric images and
    strided arrays are converted once. Do not mutate inputs during this call.
    """
    from cp_anywhere import _native

    if not images or not objects:
        raise ValueError("images and objects must be nonempty")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    selected = _validate_families(families)
    converted_images, shape = _convert_images(images)
    converted_objects = _convert_objects(objects, shape)
    if "neighbors" in selected and "Cells" not in objects:
        raise ValueError("neighbors require a Cells object set")
    converted_mask = _convert_mask(mask, shape)
    raw = _native.measure(
        shape, converted_images, converted_objects, converted_mask, image_id, selected
    )
    features = raw["features"]
    if "Image" in features:
        features["Image"] = {
            name: value[0].item() for name, value in features["Image"].items()
        }
    return MeasurementResult(
        image_id, features, raw["object_ids"], raw["relationships"]
    )
