"""Tests for the per-object blob export: JPEG image crops plus exact,
compressed mask shapes overlayable on those crops, written as real
Parquet. Reuses the cppipe-coverage test image's Cells/Nuclei object
sets for real, non-trivial measurement data (not synthetic single-pixel
toys), following test_database_export.py's conventions.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.dataset
import pyarrow.parquet
import pytest
import tifffile
from PIL import Image as PILImage

from cp_anywhere.api import MeasurementResult, measure
from cp_anywhere.object_export import (
    decode_mask_rle,
    encode_mask_rle,
    read_objects_parquet,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"

# Named test constants (ruff PLR2004): the RLE codec's compression claim is
# only meaningful above zlib's container overhead, and the JPEG comparison
# tolerance bounds quantization noise, not a magic threshold.
_ZLIB_CONTAINER_BYTES = 128
_MIN_COMPRESSIBLE_PIXELS = 256
_RLE_BUDGET_RATIO = 0.25  # decoded pixels per stored byte, minimum claim
_JPEG_TOLERANCE_U8 = 16.0
_MASK_FILL_PROBABILITY = 0.5
_MASK_FLOOR_FRACTION = 0.98  # a flat mask must compress below 2% of raw bits


@pytest.fixture
def result() -> MeasurementResult:
    return measure(
        {
            "DNA": np.load(FIXTURE / "DNA.npy"),
            "Actin": np.load(FIXTURE / "Actin.npy"),
        },
        {
            "Cells": np.load(FIXTURE / "labels.npy"),
            "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
        },
        image_id="touching_gradient",
    )


@pytest.fixture
def labels_arrays() -> dict[str, np.ndarray]:
    return {
        "Cells": np.load(FIXTURE / "labels.npy"),
        "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
    }


# ---------------------------------------------------------------------------
# RLE shape codec: the accuracy guarantee underneath the export
# ---------------------------------------------------------------------------


def test_mask_rle_roundtrip_is_pixel_exact() -> None:
    rng = np.random.default_rng(7)
    for shape in [(13, 17), (64, 64), (1, 9), (40, 31)]:
        mask = rng.random(shape) > _MASK_FILL_PROBABILITY
        blob = encode_mask_rle(mask)
        assert decode_mask_rle(blob, mask.shape) is not None
        decoded = decode_mask_rle(blob, shape)
        assert np.array_equal(decoded, mask), shape
        # sanity: the codec must actually compress repetitive masks
        zeros = np.zeros((200, 300), dtype=bool)
        ones = np.ones((200, 300), dtype=bool)
        assert len(encode_mask_rle(zeros)) <= _ZLIB_CONTAINER_BYTES
        assert len(encode_mask_rle(ones)) <= _ZLIB_CONTAINER_BYTES
        # a flat mask must compress well past 1 bit per pixel once past
        # zlib's container overhead
        assert len(encode_mask_rle(zeros)) < _MASK_FLOOR_FRACTION * zeros.size / 8
        if mask.size >= _MIN_COMPRESSIBLE_PIXELS:
            assert len(blob) < mask.size * _RLE_BUDGET_RATIO


def test_mask_rle_rejects_non_boolean() -> None:
    with pytest.raises(ValueError, match="bool"):
        encode_mask_rle(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        encode_mask_rle(np.zeros((2, 4, 4), dtype=bool))


# ---------------------------------------------------------------------------
# Lossless image formats: TIFF and PNG alongside JPEG
# ---------------------------------------------------------------------------


def test_to_objects_parquet_tiff_crops_are_bit_exact(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.parquet"
    images = {
        "DNA": np.load(FIXTURE / "DNA.npy"),
        "Actin": np.load(FIXTURE / "Actin.npy"),
    }
    result.to_objects_parquet(
        destination,
        images=images,
        objects=labels_arrays,
        channels=["DNA"],
        image_format="tiff",
    )
    table = pyarrow.parquet.read_table(destination)
    assert "image_tiff" in table.column_names
    assert "image_jpeg" not in table.column_names
    dna = np.asarray(images["DNA"])
    checked = 0
    for row in table.to_pylist():
        crop = tifffile.imread(io.BytesIO(row["image_tiff"]))
        r0, r1 = row["row_min"], row["row_max"]
        c0, c1 = row["col_min"], row["col_max"]
        expected = dna[r0 : r1 + 1, c0 : c1 + 1]
        # TIFF is lossless: the crop decodes bit-for-bit to the source,
        # native bit depth preserved (no 8-bit downscale like JPEG)
        assert np.array_equal(crop, expected), row["object_id"]
        assert crop.shape == (r1 - r0 + 1, c1 - c0 + 1)
        checked += 1
    assert checked == table.num_rows > 0


def test_to_objects_parquet_tiff_multichannel_any_count(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    third = np.load(FIXTURE / "Actin.npy")[::-1].copy()
    fourth = np.load(FIXTURE / "DNA.npy").T.copy()
    images = {
        "DNA": np.load(FIXTURE / "DNA.npy"),
        "Actin": np.load(FIXTURE / "Actin.npy"),
        "ER": third,
        "RNA": fourth,
    }
    destination = tmp_path / "objects.parquet"
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA", "ER", "RNA"],
        image_format="tiff",
    )
    table = pyarrow.parquet.read_table(destination)
    assert table.column("channels").to_pylist() == ["DNA+ER+RNA"] * table.num_rows
    rna = np.asarray(images["RNA"])
    row = table.to_pylist()[0]
    crop = tifffile.imread(io.BytesIO(row["image_tiff"]))
    r0, r1, c0, c1 = row["row_min"], row["row_max"], row["col_min"], row["col_max"]
    assert crop.shape == (r1 - r0 + 1, c1 - c0 + 1, 3)
    assert np.array_equal(crop[:, :, 2], rna[r0 : r1 + 1, c0 : c1 + 1])


def test_to_objects_parquet_tiff_uint16_two_channel_stays_exact(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    # 16-bit sources keep their bit depth, and 2 channels are legal here
    # (JPEG rejects 2-channel encodings; TIFF has no such limit)
    dna16 = (np.load(FIXTURE / "DNA.npy") * 4095).astype(np.uint16)
    actin16 = (np.load(FIXTURE / "Actin.npy") * 4095).astype(np.uint16)
    destination = tmp_path / "objects.parquet"
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        destination,
        images={"DNA": dna16, "Actin": actin16},
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA", "Actin"],
        image_format="tiff",
    )
    table = pyarrow.parquet.read_table(destination)
    assert table.column("channels").to_pylist() == ["DNA+Actin"] * table.num_rows
    row = table.to_pylist()[0]
    crop = tifffile.imread(io.BytesIO(row["image_tiff"]))
    r0, r1, c0, c1 = row["row_min"], row["row_max"], row["col_min"], row["col_max"]
    expected = np.stack(
        [dna16[r0 : r1 + 1, c0 : c1 + 1], actin16[r0 : r1 + 1, c0 : c1 + 1]], axis=-1
    )
    assert crop.shape == expected.shape
    assert np.array_equal(crop, expected)


def test_to_objects_parquet_png_crops_are_lossless(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    dna16 = (np.load(FIXTURE / "DNA.npy") * 65535).astype(np.uint16)
    destination = tmp_path / "objects.parquet"
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        destination,
        images={"DNA": dna16},
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
        image_format="png",
    )
    table = pyarrow.parquet.read_table(destination)
    assert "image_png" in table.column_names
    row = table.to_pylist()[0]
    crop = np.asarray(PILImage.open(io.BytesIO(row["image_png"])))
    r0, r1, c0, c1 = row["row_min"], row["row_max"], row["col_min"], row["col_max"]
    assert crop.dtype == np.uint16
    assert np.array_equal(crop, dna16[r0 : r1 + 1, c0 : c1 + 1])


def test_to_objects_parquet_png_rejects_float(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="float"):
        result.select_scopes(["Nuclei"]).to_objects_parquet(
            tmp_path / "x.parquet",
            images={"DNA": np.load(FIXTURE / "DNA.npy")},
            objects={"Nuclei": labels_arrays["Nuclei"]},
            channels=["DNA"],
            image_format="png",
        )


def test_to_objects_parquet_unknown_format_rejected(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="image_format"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images={"DNA": np.load(FIXTURE / "DNA.npy")},
            objects=labels_arrays,
            channels=["DNA"],
            image_format="webp",
        )


def test_to_objects_parquet_jpeg_is_the_default_format(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.parquet"
    result.to_objects_parquet(
        destination,
        images={"DNA": np.load(FIXTURE / "DNA.npy")},
        objects=labels_arrays,
        channels=["DNA"],
    )
    table = pyarrow.parquet.read_table(destination)
    assert "image_jpeg" in table.column_names
    assert "image_tiff" not in table.column_names


# ---------------------------------------------------------------------------
# to_objects_parquet: JPEG blobs + exact shapes in real Parquet
# ---------------------------------------------------------------------------


def test_to_objects_parquet_writes_exact_crops_and_masks(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.parquet"
    result.to_objects_parquet(
        destination,
        images={"DNA": np.load(FIXTURE / "DNA.npy")},
        objects=labels_arrays,
        jpeg_quality=95,
    )
    table = pyarrow.parquet.read_table(destination)
    # schema: ids, bbox, area, channels, JPEG blob, exact shape blob
    names = set(table.column_names)
    assert {
        "image_id",
        "scope",
        "object_id",
        "row_min",
        "row_max",
        "col_min",
        "col_max",
        "area_pixels",
        "channels",
        "image_jpeg",
        "mask_rle",
    } <= names
    # one row per object id per scope, exactly the ids measurement kept
    expected = sum(len(result.object_ids[scope]) for scope in result.object_ids)
    assert table.num_rows == expected
    scopes = table.column("scope").to_pylist()
    assert set(scopes) == set(result.object_ids)

    # every row's mask blob decodes to the EXACT pixels its id owns inside
    # its bbox, and its JPEG blob decodes to the bbox-sized crop
    dna = np.load(FIXTURE / "DNA.npy")
    rows = table.to_pylist()
    checked = 0
    for row in rows:
        scope = row["scope"]
        object_id = row["object_id"]
        labels = labels_arrays[scope]
        r0, r1 = row["row_min"], row["row_max"]
        c0, c1 = row["col_min"], row["col_max"]
        ids_in_bbox = labels[r0 : r1 + 1, c0 : c1 + 1]
        owned = ids_in_bbox == object_id
        decoded = decode_mask_rle(row["mask_rle"], owned.shape)
        assert np.array_equal(decoded, owned), (scope, object_id)
        assert int(np.count_nonzero(owned)) == row["area_pixels"]
        crop = PILImage.open(io.BytesIO(row["image_jpeg"]))
        assert crop.size == (c1 - c0 + 1, r1 - r0 + 1)
        assert row["channels"] == "DNA"
        # JPEG is lossy: even at quality 95, sharp edges inside a
        # min-max-scaled crop can deviate by several levels. Assert the
        # crop still tracks its target intensities closely (well under
        # 16/255 ≈ 6%) rather than pixel-perfect.
        pixels = np.asarray(crop, dtype=np.float64)
        channel = dna[r0 : r1 + 1, c0 : c1 + 1].astype(np.float64)
        lo, hi = channel.min(), channel.max()
        if hi > lo:
            target = np.rint((channel - lo) / (hi - lo) * 255.0)
            assert np.abs(pixels - target).max() <= _JPEG_TOLERANCE_U8
        checked += 1
    assert checked == expected


def test_to_objects_parquet_rgb_composite_from_three_channels(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    third = np.load(FIXTURE / "Actin.npy")[::-1].copy()  # a distinct 3rd channel
    images = {
        "DNA": np.load(FIXTURE / "DNA.npy"),
        "Actin": np.load(FIXTURE / "Actin.npy"),
        "ER": third,
    }
    destination = tmp_path / "objects.parquet"
    result.to_objects_parquet(
        destination,
        images=images,
        objects=labels_arrays,
        channels=["DNA", "Actin", "ER"],
    )
    table = pyarrow.parquet.read_table(destination)
    assert table.column("channels").to_pylist() == ["DNA+Actin+ER"] * table.num_rows
    first = PILImage.open(io.BytesIO(table.column("image_jpeg").to_pylist()[0]))
    assert first.mode == "RGB"


def test_to_objects_parquet_two_channels_rejected(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="1 or 3"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images={
                "DNA": np.load(FIXTURE / "DNA.npy"),
                "Actin": np.load(FIXTURE / "Actin.npy"),
            },
            objects=labels_arrays,
            channels=["DNA", "Actin"],
        )


def test_to_objects_parquet_requires_channels_when_ambiguous(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="channels"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images={
                "DNA": np.load(FIXTURE / "DNA.npy"),
                "Actin": np.load(FIXTURE / "Actin.npy"),
            },
            objects=labels_arrays,
        )


def test_to_objects_parquet_unknown_channel_rejected(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not provided"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images={"DNA": np.load(FIXTURE / "DNA.npy")},
            objects={"Cells": labels_arrays["Cells"]},
            channels=["Membrane"],
        )


def test_to_objects_parquet_scope_must_have_labels(
    result: MeasurementResult, tmp_path: Path
) -> None:
    dna = {"DNA": np.load(FIXTURE / "DNA.npy")}
    cells = np.load(FIXTURE / "labels.npy")
    with pytest.raises(ValueError, match="object set"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images=dna,
            objects={"Cells": cells},
            channels=["DNA"],
        )
    with pytest.raises(ValueError, match="aligned"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images=dna,
            objects={
                "Cells": cells,
                "Nuclei": np.load(FIXTURE / "nuclei_labels.npy"),
                "Cytoplasm": cells,
            },
            channels=["DNA"],
        )


def test_to_objects_parquet_rejects_3d(
    result: MeasurementResult, tmp_path: Path
) -> None:
    labels_3d = np.load(FIXTURE / "labels.npy")[np.newaxis].repeat(3, axis=0)
    with pytest.raises(NotImplementedError, match="3-D"):
        result.to_objects_parquet(
            tmp_path / "x.parquet",
            images={"DNA": np.load(FIXTURE / "DNA.npy")},
            objects={"Cells": labels_3d, "Nuclei": labels_3d},
            channels=["DNA"],
        )


def test_to_objects_parquet_dataset_accumulates_and_replaces(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects_dataset"
    images = {"DNA": np.load(FIXTURE / "DNA.npy")}
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    first_table = pyarrow.dataset.dataset(destination, format="parquet").to_table()
    first_ids = first_table.column("image_id").to_pylist()
    # every call adds its scope's rows; no overwrite
    assert set(first_ids) == {"touching_gradient"}
    second = measure(
        images,
        {"Nuclei": labels_arrays["Nuclei"]},
        image_id="touching_gradient_2",
        families=["image_intensity", "object_intensity"],
    )
    second.to_objects_parquet(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
    )
    second_table = pyarrow.dataset.dataset(destination, format="parquet").to_table()
    second_ids = second_table.column("image_id").to_pylist()
    assert set(second_ids) == {"touching_gradient", "touching_gradient_2"}
    assert len(second_ids) == 2 * len(first_ids)
    result.select_scopes(["Nuclei"]).to_objects_parquet(
        destination,
        images=images,
        objects={"Nuclei": labels_arrays["Nuclei"]},
        channels=["DNA"],
        if_exists="replace",
    )
    table = pyarrow.dataset.dataset(destination, format="parquet").to_table()
    assert set(table.column("image_id").to_pylist()) == {"touching_gradient"}
    assert len(table) == len(first_ids)


def test_to_objects_parquet_rows_decode_with_the_reader_helper(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "objects.parquet"
    result.to_objects_parquet(
        destination,
        images={"DNA": np.load(FIXTURE / "DNA.npy")},
        objects=labels_arrays,
        channels=["DNA"],
    )
    table = read_objects_parquet(destination)
    assert table.num_rows == pyarrow.parquet.read_table(destination).num_rows
