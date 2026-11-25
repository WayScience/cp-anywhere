"""Per-object blob export: JPEG image crops plus exact, compressed mask
shapes, written as real Parquet.

This is the object-level companion to :meth:`MeasurementResult's
to_parquet` feature exports: where ``to_parquet`` stores numeric
measurements, this module stores the visual evidence -- each object's
JPEG-encoded crop of the selected image channel(s) and its exact binary
mask, compressed with a simple, dependency-free RLE + zlib codec. The
mask blob is the accuracy guarantee: it decodes to the object's exact
owned pixels inside the recorded bounding box (pixel-for-pixel, lossless),
so an overlay drawn from it reconstructs the segmentation exactly; the
JPEG blob is the image evidence itself (lossy, quality-pinned, decoded by
any JPEG reader).

Declaring scope: this export is 2-D only (3-D label arrays raise
``NotImplementedError`` naming the gap); the stored mask covers the
object's bounding box, not the whole frame (an overlay composes it at
``(row_min, col_min)``); and JPEG encoding uses Pillow's quality knob
with a documented per-crop min-max intensity scaling.
"""

from __future__ import annotations

import io
import json
import uuid
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa


_RLE_ZLIB_LEVEL = 6
_SCALING_EPS = 1e-30
# The image-blob formats the object export supports (see
# encode_object_image): "jpeg" is the space-efficient default, "tiff"
# and "png" are the lossless alternatives.
SUPPORTED_IMAGE_FORMATS = ("jpeg", "tiff", "png")


def encode_mask_rle(mask: np.ndarray) -> bytes:
    """Compress one exact 2-D binary mask: zlib over bit-packed pixels.

    The blob carries no geometry -- the caller records the mask's shape
    (here: the object's bounding box) alongside it, and
    :func:`decode_mask_rle` restores the mask pixel-for-pixel. Raises
    ``ValueError`` for non-boolean or non-2-D input rather than silently
    encoding a lossy conversion.
    """
    array = np.asarray(mask)
    if array.ndim not in (2,):
        raise ValueError("mask must be 2-D")
    if array.dtype != np.bool_:
        raise ValueError("mask must be boolean (dtype bool)")
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return zlib.compress(packed.tobytes(), _RLE_ZLIB_LEVEL)


def decode_mask_rle(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode a :func:`encode_mask_rle` blob back to its exact boolean mask.

    ``shape`` is the (rows, cols) the mask covered. Raises ``ValueError``
    naming the expected and actual pixel counts if the blob does not
    decode to exactly that many pixels (e.g. it was written for a
    different shape or is corrupt), rather than returning a wrong-sized
    or silently-truncated mask.
    """
    rows, cols = shape
    packed = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    expected_packed = (rows * cols + 7) // 8
    if packed.size != expected_packed:
        raise ValueError(
            f"mask blob holds {packed.size} packed bytes, expected exactly "
            f"{expected_packed} for shape {(rows, cols)} -- the blob was "
            "written for a different shape or is corrupt"
        )
    bits = np.unpackbits(packed, bitorder="little")[: rows * cols]
    return bits.reshape(rows, cols).astype(bool)


def encode_labels_rle(labels: np.ndarray) -> bytes:
    """Compress one exact whole-frame label map: zlib over value/run-length
    pairs.

    The blob stores, per run of constant label id, the id and its length
    as int64 pairs -- lossless, decodeable without any extra dependency
    via :func:`decode_labels_rle`, and compact for real segmentations
    whose per-object pixel runs compress far below the raw 4 bytes/pixel.
    Raises ``ValueError`` for non-2-D input or label ids outside the
    range this codec can represent exactly (negative ids, or ids above
    int32 max -- cp_anywhere's own ``MAX_LABEL`` cap sits far below).
    """
    array = np.asarray(labels)
    if array.ndim not in (2,):
        raise ValueError("labels must be 2-D")
    values = array.astype(np.int64, copy=False)
    if values.size and values.min() < 0:
        raise ValueError("label ids must not be negative")
    if values.size and values.max() > np.iinfo(np.int32).max:
        raise ValueError("label ids above int32 max are too large for this codec")
    flat = values.reshape(-1)
    if flat.size == 0:
        raise ValueError("labels must be nonempty")
    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    run_values = flat[starts]
    run_lengths = np.diff(np.concatenate((starts, [flat.size])))
    payload = np.concatenate((run_values, run_lengths)).astype(np.int64)
    return zlib.compress(payload.tobytes(), _RLE_ZLIB_LEVEL)


def decode_labels_rle(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode a :func:`encode_labels_rle` blob back to its exact label map.

    ``shape`` is the (rows, cols) of the stored frame. Raises
    ``ValueError`` if the blob does not decode to exactly that many
    pixels or its run lengths are internally inconsistent (i.e. it was
    written for a different shape or is corrupt), rather than returning
    a wrong-sized or silently-truncated label map.
    """
    rows, cols = shape
    payload = np.frombuffer(zlib.decompress(blob), dtype=np.int64)
    if payload.size == 0 or payload.size % 2 != 0:
        raise ValueError("label blob is malformed (odd-length payload)")
    run_values = payload[: payload.size // 2]
    run_lengths = payload[payload.size // 2 :]
    if (run_lengths < 0).any():
        raise ValueError("label blob is malformed (negative run length)")
    if run_lengths.sum() != rows * cols:
        raise ValueError(
            f"label blob decodes to {int(run_lengths.sum())} pixels, expected "
            f"{rows * cols} for shape {(rows, cols)} -- the blob was written "
            "for a different shape or is corrupt"
        )
    repeated = np.repeat(run_values, run_lengths)
    return repeated.reshape(rows, cols).astype(np.uint32)


def encode_object_image_jpeg(channels: list[np.ndarray], *, quality: int) -> bytes:
    """JPEG-encode one object's crop from 1 (grayscale) or 3 (RGB) channels.

    Each channel's crop is scaled to 8-bit by its own min-max range
    (``rint((value - min) / max(max - min, eps) * 255)``, an all-constant
    crop maps to all-black -- documented semantics, the caller can always
    store the raw numbers in the feature columns instead). Raises
    ``ValueError`` for anything other than 1 or 3 identically-shaped
    channel crops.
    """
    if len(channels) not in (1, 3):
        raise ValueError(f"JPEG encoding needs 1 or 3 channels, got {len(channels)}")
    scaled: list[np.ndarray] = [
        _scale_to_uint8(np.asarray(channel)) for channel in channels
    ]
    first = scaled[0]
    for other in scaled[1:]:
        if other.shape != first.shape:
            raise ValueError("all channel crops must share one shape")
    from PIL import Image

    pixels = first if len(scaled) == 1 else np.stack(scaled, axis=-1)
    image = Image.fromarray(pixels)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def encode_object_image_tiff(channels: list[np.ndarray]) -> bytes:
    """Losslessly encode one object's crop as TIFF via tifffile.

    Any channel count (1 -> (H, W), N>1 -> (H, W, N) in list order), the
    source dtype preserved bit-for-bit (uint8/uint16 keep native bit
    depth instead of JPEG's forced 8-bit downscale; float dtypes store
    as IEEE floats), deflate-compressed inside the container. This is
    the channel-count-3+-compatible alternative to JPEG that stays
    exactly aligned with the exported mask shapes: both crop from the
    same bounding box, so the overlay never needs rescaling or
    re-normalizing.
    """
    if not channels:
        raise ValueError("TIFF encoding needs at least one channel")
    arrays = [np.asarray(channel) for channel in channels]
    first = arrays[0]
    for other in arrays[1:]:
        if other.shape != first.shape:
            raise ValueError("all channel crops must share one shape")
    pixels = first if len(arrays) == 1 else np.stack(arrays, axis=-1)
    import tifffile

    stream = io.BytesIO()
    tifffile.imwrite(
        stream,
        pixels,
        compression="zlib",
        photometric="minisblack" if len(arrays) == 1 else None,
    )
    return stream.getvalue()


def encode_object_image_png(channels: list[np.ndarray]) -> bytes:
    """Losslessly encode one object's crop as PNG via Pillow.

    Integer dtypes only (uint8/uint16, lossless); float crops raise
    ``ValueError`` naming the limitation -- use ``"tiff"`` for those.
    Channel counts 1-4 map onto PNG's native sample models (grayscale,
    gray+alpha, RGB, RGBA); higher counts must use ``"tiff"``.
    """
    arrays = [np.asarray(channel) for channel in channels]
    if not arrays:
        raise ValueError("PNG encoding needs at least one channel")
    for array in arrays:
        if array.dtype.kind not in "ui":
            raise ValueError(
                f"PNG crops must have an integer dtype (uint8/uint16); got "
                f"{array.dtype} -- use image_format='tiff' for float crops"
            )
    first = arrays[0]
    for other in arrays[1:]:
        if other.shape != first.shape:
            raise ValueError("all channel crops must share one shape")
    _PNG_MAX_CHANNELS = 4  # grayscale/gray+alpha/RGB/RGBA sample models
    if len(arrays) > _PNG_MAX_CHANNELS:
        raise ValueError(
            f"PNG supports at most 4 channels (grayscale/gray+alpha/RGB/"
            f"RGBA), got {len(arrays)} -- use image_format='tiff'"
        )
    from PIL import Image

    pixels = first if len(arrays) == 1 else np.stack(arrays, axis=-1)
    image = Image.fromarray(pixels)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _scale_to_uint8(channel: np.ndarray) -> np.ndarray:
    values = channel.astype(np.float64, copy=False)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= _SCALING_EPS:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.rint((values - lo) / (hi - lo) * 255.0)
    return scaled.clip(0, 255).astype(np.uint8)


def object_bboxes(
    labels: np.ndarray, object_ids: Sequence[int]
) -> dict[int, tuple[int, int, int, int, int]]:
    """Bounding boxes (row_min, row_max, col_min, col_max, area_pixels)
    for each requested id, found in one sorted pass (no per-object full
    scan). Raises ``ValueError`` naming the missing ids if any requested
    id is absent from the labels -- the export would otherwise silently
    write rows for objects that do not exist.
    """
    if labels.ndim not in (2,):
        raise NotImplementedError(
            f"3-D label arrays are not supported by the object blob export "
            f"(received ndim={labels.ndim}); 3-D object crops/masks are a "
            "declared gap"
        )
    flat = labels.reshape(-1)
    nonzero = np.flatnonzero(flat)
    ids_at = flat[nonzero]
    order = np.argsort(ids_at, kind="stable")
    sorted_ids = ids_at[order]
    rows_i, cols_i = np.unravel_index(nonzero[order], labels.shape)
    boxes: dict[int, tuple[int, int, int, int, int]] = {}
    start = 0
    total = sorted_ids.size
    while start < total:
        end = np.searchsorted(sorted_ids, sorted_ids[start], side="right")
        present = sorted_ids[start]
        if present in object_ids:
            where = slice(start, end)
            boxes[int(present)] = (
                int(rows_i[where].min()),
                int(rows_i[where].max()),
                int(cols_i[where].min()),
                int(cols_i[where].max()),
                int(end - start),
            )
        start = int(end)
    missing = [oid for oid in object_ids if oid not in boxes]
    if missing:
        raise ValueError(
            f"object id(s) {missing} are not present in this scope's labels"
        )
    return boxes


def build_object_rows(  # noqa: PLR0913 -- one row build pipeline, all real data
    image_id: str,
    images: Mapping[str, np.ndarray],
    objects: Mapping[str, np.ndarray],
    channels: Sequence[str],
    object_ids: Mapping[str, np.ndarray],
    *,
    image_format: str,
    jpeg_quality: int,
) -> list[dict[str, str | int | bytes]]:
    """Build one row per object per scope: ids, bbox, area, channel name,
    image crop bytes (encoded per ``image_format``), exact compressed
    mask bytes."""
    channel_names = list(channels)
    channel_label = "+".join(channel_names)
    image_column = f"image_{image_format}"
    rows: list[dict[str, str | int | bytes]] = []
    for scope, ids in object_ids.items():
        labels = np.asarray(objects[scope])
        for object_id in ids:
            oid = int(object_id)
            r0, r1, c0, c1, area = object_bboxes(labels, [oid])[oid]
            crops = [
                np.asarray(images[name])[r0 : r1 + 1, c0 : c1 + 1]
                for name in channel_names
            ]
            mask_crop = labels[r0 : r1 + 1, c0 : c1 + 1] == oid
            if image_format == "jpeg":
                blob = encode_object_image_jpeg(crops, quality=jpeg_quality)
            elif image_format == "tiff":
                blob = encode_object_image_tiff(crops)
            elif image_format == "png":
                blob = encode_object_image_png(crops)
            else:
                raise ValueError(
                    f"image_format must be one of {SUPPORTED_IMAGE_FORMATS}, "
                    f"got {image_format!r}"
                )
            rows.append(
                {
                    "image_id": image_id,
                    "scope": scope,
                    "object_id": oid,
                    "row_min": r0,
                    "row_max": r1,
                    "col_min": c0,
                    "col_max": c1,
                    "area_pixels": area,
                    "channels": channel_label,
                    image_column: blob,
                    "mask_rle": encode_mask_rle(mask_crop),
                }
            )
    return rows


def objects_parquet_table(
    rows: list[dict[str, str | int | bytes]], *, image_format: str = "jpeg"
) -> "pa.Table":
    """Typed Arrow table for the built rows (binary image/RLE columns)."""
    import pyarrow as pa

    image_column = f"image_{image_format}"

    if not rows:
        # still return a typed, empty table so callers can write an
        # empty-but-valid export
        rows = [
            {
                "image_id": "",
                "scope": "",
                "object_id": 0,
                "row_min": 0,
                "row_max": 0,
                "col_min": 0,
                "col_max": 0,
                "area_pixels": 0,
                "channels": "",
                image_column: b"",
                "mask_rle": b"",
            }
        ]
        table = pa.Table.from_pylist(rows)
        return table.slice(0, 0)
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("image_id", pa.string()),
                ("scope", pa.string()),
                ("object_id", pa.int64()),
                ("row_min", pa.int32()),
                ("row_max", pa.int32()),
                ("col_min", pa.int32()),
                ("col_max", pa.int32()),
                ("area_pixels", pa.int32()),
                ("channels", pa.string()),
                (image_column, pa.binary()),
                ("mask_rle", pa.binary()),
            ]
        ),
    )


def write_objects_parquet(
    rows: list[dict[str, str | int | bytes]],
    path: str | Path,
    *,
    image_format: str = "jpeg",
    if_exists: str,
    max_rows_per_file: int,
) -> None:
    """Write the object rows as a Parquet file (``.parquet`` suffix) or an
    accumulating Parquet dataset (a directory), mirroring
    :meth:`MeasurementResult.to_parquet`'s own conventions.
    """
    table = objects_parquet_table(rows, image_format=image_format)
    _write_table(table, path, if_exists=if_exists, max_rows_per_file=max_rows_per_file)


def _write_table(
    table: "pa.Table", path: str | Path, *, if_exists: str, max_rows_per_file: int
) -> None:
    """Write one Arrow table as a Parquet file (``.parquet`` suffix) or an
    accumulating Parquet dataset (a directory), mirroring
    :meth:`MeasurementResult.to_parquet`'s own conventions."""
    import pyarrow.dataset
    import pyarrow.parquet

    destination = Path(path)
    if destination.suffix == ".parquet":
        pyarrow.parquet.write_table(table, str(destination))
        return
    if if_exists not in ("append", "replace"):
        raise ValueError(f"if_exists must be 'append' or 'replace', got {if_exists!r}")
    if if_exists == "replace" and destination.exists():
        import shutil

        shutil.rmtree(destination)
    pyarrow.dataset.write_dataset(
        table,
        str(destination),
        format="parquet",
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_file,
        existing_data_behavior="overwrite_or_ignore",
        basename_template=f"{uuid.uuid4().hex}-{{i}}.parquet",
    )


def read_objects_parquet(path: str | Path) -> "pa.Table":
    """Read an objects Parquet export (single file or dataset directory)
    as one typed Arrow table. Raises the import-time error pyarrow gives
    for a missing/unreadable path; the row schema is documented in
    :meth:`MeasurementResult.to_objects_parquet`.
    """
    import pyarrow.dataset

    destination = Path(path)
    if destination.suffix == ".parquet":
        import pyarrow.parquet

        return pyarrow.parquet.read_table(str(destination))
    return pyarrow.dataset.dataset(str(destination), format="parquet").to_table()


def read_parquet_export(path: str | Path) -> "pa.Table":
    """Shared reader: a Parquet file or dataset directory to one typed
    Arrow table (used for both the objects and labels exports)."""
    import pyarrow.dataset

    destination = Path(path)
    if destination.suffix == ".parquet":
        import pyarrow.parquet

        return pyarrow.parquet.read_table(str(destination))
    return pyarrow.dataset.dataset(str(destination), format="parquet").to_table()


def read_labels_parquet(path: str | Path) -> "pa.Table":
    """Read a labels Parquet export (single file or dataset directory)
    as one typed Arrow table. The row schema is documented in
    :meth:`MeasurementResult.to_labels_parquet`.
    """
    return read_parquet_export(path)


def build_labels_rows(
    image_id: str,
    objects: Mapping[str, np.ndarray],
    object_ids: Mapping[str, np.ndarray],
) -> list[dict[str, str | int | bytes]]:
    """Build one row per measured scope: ids, frame shape, object count,
    and the exact whole-frame label map as compressed RLE bytes."""
    rows: list[dict[str, str | int | bytes]] = []
    for scope, ids in object_ids.items():
        labels = np.asarray(objects[scope])
        rows.append(
            {
                "image_id": image_id,
                "scope": scope,
                "rows": labels.shape[0],
                "cols": labels.shape[1],
                "object_count": len(ids),
                "labels_rle": encode_labels_rle(labels),
            }
        )
    return rows


def labels_parquet_table(rows: list[dict[str, str | int | bytes]]) -> "pa.Table":
    """Typed Arrow table for the whole-frame label rows (binary RLE column)."""
    import pyarrow as pa

    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("image_id", pa.string()),
                ("scope", pa.string()),
                ("rows", pa.int32()),
                ("cols", pa.int32()),
                ("object_count", pa.int32()),
                ("labels_rle", pa.binary()),
            ]
        ),
    )


def write_labels_parquet(
    rows: list[dict[str, str | int | bytes]],
    path: str | Path,
    *,
    if_exists: str,
    max_rows_per_file: int,
) -> None:
    """Write the whole-frame label rows as a Parquet file (``.parquet``
    suffix) or an accumulating Parquet dataset (a directory), mirroring
    :meth:`MeasurementResult.to_parquet`'s own conventions."""
    table = labels_parquet_table(rows)
    _write_table(table, path, if_exists=if_exists, max_rows_per_file=max_rows_per_file)


def _write_table_duckdb(
    table: "pa.Table", path: str | Path, table_name: str, *, if_exists: str
) -> None:
    """Create-or-append one Arrow table into a DuckDB database file, with
    :meth:`MeasurementResult.to_duckdb`'s own append/replace semantics."""
    import duckdb

    connection = duckdb.connect(str(path))
    try:
        cursor = connection.cursor()
        if if_exists == "replace":
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"').fetchall()
        exists = (
            len(
                cursor.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                    [table_name],
                ).fetchall()
            )
            > 0
        )
        cursor.register("_cp_anywhere_segmentation_rows", table)
        if exists:
            cursor.execute(
                f'INSERT INTO "{table_name}" '
                "SELECT * FROM _cp_anywhere_segmentation_rows"
            ).fetchall()
        else:
            cursor.execute(
                f'CREATE TABLE "{table_name}" AS '
                "SELECT * FROM _cp_anywhere_segmentation_rows"
            ).fetchall()
        cursor.unregister("_cp_anywhere_segmentation_rows")
        cursor.close()
    finally:
        connection.close()


def write_objects_duckdb(
    rows: list[dict[str, str | int | bytes]],
    path: str | Path,
    *,
    image_format: str = "jpeg",
    if_exists: str,
) -> None:
    """Write the object rows into a DuckDB database file's
    ``SegmentationObjects`` table (binary image/RLE blobs land as BLOB
    columns), with :meth:`MeasurementResult.to_duckdb`'s own
    append/replace semantics."""
    table = objects_parquet_table(rows, image_format=image_format)
    _write_table_duckdb(table, path, "SegmentationObjects", if_exists=if_exists)


def write_labels_duckdb(
    rows: list[dict[str, str | int | bytes]],
    path: str | Path,
    *,
    if_exists: str,
) -> None:
    """Write the whole-frame label rows into a DuckDB database file's
    ``SegmentationLabels`` table (the RLE blob lands as a BLOB column),
    with :meth:`MeasurementResult.to_duckdb`'s own append/replace
    semantics."""
    table = labels_parquet_table(rows)
    _write_table_duckdb(table, path, "SegmentationLabels", if_exists=if_exists)


def _metadata_note() -> str:
    """The encoding contract, suitable for embedding in the file's schema
    metadata so the blobs stay self-describing even when copied around."""
    return json.dumps(
        {
            "image_jpeg": (
                "JPEG per-object crop of the named channels; each channel "
                "scaled to 8-bit by its own min-max range (constant crops "
                "map to black); 1 channel = L, 3 channels = RGB in list order"
            ),
            "mask_rle": (
                "zlib-compressed bit-packed (packbits, little bitorder) "
                "row-major boolean mask over the bbox "
                "[row_min:row_max+1, col_min:col_max+1] -- decodes to the "
                "object's exact owned pixels"
            ),
        }
    )
