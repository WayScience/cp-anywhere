"""Tests for whole-frame segmentation RLE storage: one row per image and
scope holding the complete label map as an exact, zlib-compressed
run-length-encoded blob, complementing the per-object masks/crops of
to_objects_parquet with a single-blob whole-segmentation export that
decodes pixel-for-pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.dataset
import pyarrow.parquet
import pytest

from cp_anywhere.api import MeasurementResult, measure
from cp_anywhere.object_export import (
    decode_labels_rle,
    encode_labels_rle,
    read_labels_parquet,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/synthetic/touching_gradient"

# Named test constants (ruff PLR2004): the fixture's scope count and the
# compression budget for a run-heavy plate are test assertions, not magic.
_EXPECTED_SCOPES = 2  # fixture measures Cells and Nuclei
_RLE_BUDGET_RATIO = 0.01  # stored bytes per raw 4-byte pixel, maximum
_MAX_LABEL_BYTE_TEST_ID = 2_500_000_000  # above int32 max, codec must refuse


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
# RLE label codec: whole-frame accuracy underneath the export
# ---------------------------------------------------------------------------


def test_labels_rle_roundtrip_is_pixel_exact() -> None:
    rng = np.random.default_rng(11)
    for shape, max_id in [((13, 17), 5), ((64, 64), 3), ((1, 99), 1), ((40, 31), 7)]:
        labels = rng.integers(0, max_id + 1, size=shape).astype(np.uint32)
        blob = encode_labels_rle(labels)
        decoded = decode_labels_rle(blob, shape)
        assert np.array_equal(decoded, labels), shape
        assert decoded.dtype == np.uint32
    # flat / zero frames are the best-case runs
    zeros = np.zeros((200, 300), dtype=np.uint32)
    assert decode_labels_rle(encode_labels_rle(zeros), (200, 300)).sum() == 0


def test_labels_rle_compresses_run_heavy_frames() -> None:
    # a dense-object plate stores as runs, far below 4 bytes/pixel
    labels = np.zeros((500, 600), dtype=np.uint32)
    for i, (r, c) in enumerate(
        [(50, 60), (50, 200), (120, 350), (300, 100), (420, 450)], start=1
    ):
        rr, cc = np.ogrid[:500, :600]
        disk = (rr - r) ** 2 + (cc - c) ** 2 <= 30**2
        labels[disk] = i
    blob = encode_labels_rle(labels)
    assert len(blob) < _RLE_BUDGET_RATIO * 4 * labels.size
    assert decode_labels_rle(blob, (500, 600)).sum() == labels.sum()


def test_labels_rle_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        encode_labels_rle(np.zeros((2, 4, 4), dtype=np.uint32))
    with pytest.raises(ValueError, match="negative"):
        encode_labels_rle(np.array([[-1, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="too large"):
        encode_labels_rle(np.array([[_MAX_LABEL_BYTE_TEST_ID]], dtype=np.int64))


def test_labels_rle_decode_rejects_mismatched_shape() -> None:
    blob = encode_labels_rle(np.zeros((4, 4), dtype=np.uint32))
    with pytest.raises(ValueError, match="decodes to"):
        decode_labels_rle(blob, (9, 9))


# ---------------------------------------------------------------------------
# to_labels_parquet: whole-frame label maps as real Parquet rows
# ---------------------------------------------------------------------------


def test_to_labels_parquet_writes_exact_frames(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "labels.parquet"
    result.to_labels_parquet(destination, objects=labels_arrays)
    table = pyarrow.parquet.read_table(destination)
    assert {
        "image_id",
        "scope",
        "rows",
        "cols",
        "object_count",
        "labels_rle",
    } <= set(table.column_names)
    assert table.num_rows == len(result.object_ids)
    # the fixture measures exactly two scopes (Cells, Nuclei)
    assert len(result.object_ids) == _EXPECTED_SCOPES
    rows = table.to_pylist()
    scopes = {row["scope"]: row for row in rows}
    assert set(scopes) == set(result.object_ids)
    for scope, labels in labels_arrays.items():
        row = scopes[scope]
        decoded = decode_labels_rle(row["labels_rle"], (row["rows"], row["cols"]))
        assert np.array_equal(decoded, labels), scope
        assert row["rows"] == labels.shape[0]
        assert row["cols"] == labels.shape[1]
        assert row["object_count"] == len(result.object_ids[scope])


def test_to_labels_parquet_select_scopes_subset(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "labels.parquet"
    result.select_scopes(["Nuclei"]).to_labels_parquet(
        destination, objects={"Nuclei": labels_arrays["Nuclei"]}
    )
    table = pyarrow.parquet.read_table(destination)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    decoded = decode_labels_rle(row["labels_rle"], (row["rows"], row["cols"]))
    assert np.array_equal(decoded, labels_arrays["Nuclei"])


def test_to_labels_parquet_dataset_accumulates_and_replaces(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "labels_dataset"
    result.select_scopes(["Nuclei"]).to_labels_parquet(
        destination, objects={"Nuclei": labels_arrays["Nuclei"]}
    )
    first_ids = set(
        pyarrow.dataset.dataset(destination, format="parquet")
        .to_table()
        .column("image_id")
        .to_pylist()
    )
    assert first_ids == {"touching_gradient"}
    second = measure(
        {"DNA": np.load(FIXTURE / "DNA.npy"), "Actin": np.load(FIXTURE / "Actin.npy")},
        {"Nuclei": labels_arrays["Nuclei"]},
        image_id="touching_gradient_2",
        families=["image_intensity", "object_intensity"],
    )
    second.to_labels_parquet(destination, objects={"Nuclei": labels_arrays["Nuclei"]})
    table = pyarrow.dataset.dataset(destination, format="parquet").to_table()
    assert set(table.column("image_id").to_pylist()) == {
        "touching_gradient",
        "touching_gradient_2",
    }
    result.select_scopes(["Nuclei"]).to_labels_parquet(
        destination, objects={"Nuclei": labels_arrays["Nuclei"]}, if_exists="replace"
    )
    table = pyarrow.dataset.dataset(destination, format="parquet").to_table()
    assert set(table.column("image_id").to_pylist()) == {"touching_gradient"}


def test_to_labels_parquet_scope_must_have_labels(
    result: MeasurementResult,
    labels_arrays: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="object set"):
        result.to_labels_parquet(
            tmp_path / "x.parquet", objects={"Cells": labels_arrays["Cells"]}
        )
    with pytest.raises(ValueError, match="aligned"):
        result.to_labels_parquet(
            tmp_path / "x.parquet",
            objects={
                "Cells": labels_arrays["Cells"],
                "Nuclei": labels_arrays["Nuclei"],
                "Cytoplasm": labels_arrays["Cells"],
            },
        )


def test_to_labels_parquet_rejects_3d(
    result: MeasurementResult, tmp_path: Path
) -> None:
    labels_3d = np.load(FIXTURE / "labels.npy")[np.newaxis].repeat(3, axis=0)
    with pytest.raises(NotImplementedError, match="3-D"):
        result.to_labels_parquet(
            tmp_path / "x.parquet",
            objects={"Cells": labels_3d, "Nuclei": labels_3d},
        )


def test_to_labels_parquet_reader_helper(
    result: MeasurementResult, labels_arrays: dict[str, np.ndarray], tmp_path: Path
) -> None:
    destination = tmp_path / "labels.parquet"
    result.to_labels_parquet(destination, objects=labels_arrays)
    table = read_labels_parquet(destination)
    assert table.num_rows == pyarrow.parquet.read_table(destination).num_rows
