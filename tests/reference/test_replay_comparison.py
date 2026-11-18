"""Cross-platform oracle replay accepts rounding, never changed science/inputs."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reference.replay import (
    verify_measurement_replay,
    verify_segmentation_replay,
)

ROOT = Path(__file__).resolve().parents[2]


def _json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False).encode()


def _npy(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _measurement_files() -> dict[str, bytes]:
    # These are actual macOS/Linux oracle outputs, differing by one float64 ULP.
    name = "fixtures/reference/volume_3d.json"
    files = {name: (ROOT / name).read_bytes()}
    files["spec/features.yaml"] = (ROOT / "spec/features.yaml").read_bytes()
    return files


def test_measurement_replay_accepts_platform_angle_rounding(tmp_path: Path) -> None:
    files = _measurement_files()
    _write(tmp_path, files)
    name = "fixtures/reference/volume_3d.json"
    actual = json.loads(files[name])
    actual["features"]["Cells"]["Neighbors_AngleBetweenNeighbors_Adjacent"][0] = (
        74.05460409907715
    )
    files[name] = _json(actual)
    assert verify_measurement_replay(tmp_path, files) == [name]


@pytest.mark.parametrize(
    "field",
    ["numeric", "count", "object_ids", "relationships", "nonfinite", "missing"],
)
def test_measurement_replay_rejects_scientific_changes(
    tmp_path: Path, field: str
) -> None:
    files = _measurement_files()
    _write(tmp_path, files)
    name = "fixtures/reference/volume_3d.json"
    actual = json.loads(files[name])
    if field == "numeric":
        actual["features"]["Cells"]["Neighbors_AngleBetweenNeighbors_Adjacent"][0] += (
            1e-4
        )
    elif field == "count":
        actual["features"]["Cells"]["Neighbors_NumberOfNeighbors_Adjacent"][0] += 1
    elif field == "object_ids":
        actual["object_ids"]["Cells"][0] += 1
    elif field == "relationships":
        actual["relationships"] = []
    elif field == "nonfinite":
        actual["features"]["Cells"]["Neighbors_AngleBetweenNeighbors_Adjacent"][0] = {
            "nonfinite": "NaN"
        }
    else:
        del actual["features"]["Cells"]["Neighbors_AngleBetweenNeighbors_Adjacent"]
    files[name] = _json(actual)
    with pytest.raises(AssertionError, match=r"volume_3d\.json"):
        verify_measurement_replay(tmp_path, files)


def _csv(value: str, *, object_id: str = "1") -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "image_id",
            "image_number",
            "object_scope",
            "object_id",
            "feature",
            "value_json",
        ]
    )
    writer.writerow(
        [
            "volume_3d",
            "1",
            "Cells",
            object_id,
            "Neighbors_AngleBetweenNeighbors_Adjacent",
            value,
        ]
    )
    return stream.getvalue().encode()


def test_measurement_csv_replay_compares_values_but_keeps_row_keys(
    tmp_path: Path,
) -> None:
    name = "fixtures/reference/objects.csv"
    files = _measurement_files() | {name: _csv("74.05460409907714")}
    _write(tmp_path, files)
    files[name] = _csv("74.05460409907715")
    assert verify_measurement_replay(tmp_path, files) == [name]
    files[name] = _csv("74.05460409907715", object_id="2")
    with pytest.raises(AssertionError, match="object_id"):
        verify_measurement_replay(tmp_path, files)


def _segmentation_files() -> dict[str, bytes]:
    folder = "fixtures/segmentation/platform_rounding"
    arrays = {
        f"{folder}/pixels.npy": _npy(np.array([[20, 200]], dtype=np.uint8)),
        f"{folder}/labels.npy": _npy(np.array([[0, 1]], dtype=np.int32)),
        f"{folder}/small_removed_labels.npy": _npy(np.array([[0, 1]], dtype=np.int32)),
        f"{folder}/threshold_map.npy": _npy(np.array([[0.24, 0.35]], dtype=np.float64)),
    }
    case = {
        "image_id": "platform_rounding",
        "settings": {"threshold_method": "li", "log_transform": True},
        "rescale": True,
        "threshold": 0.10339596744948397,
        "guide_threshold": 0.06994056857048067,
        "object_count": 1,
        "shape": [1, 2],
        "sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in arrays.items()
        },
    }
    manifest = {
        "source_commit": "2df090ff281cff80943fe679a1c3192fbda9cd6d",
        "cellprofiler_version": "4.2.8",
        "cases": [
            {
                "name": case["image_id"],
                "result": f"{folder}.json",
                "threshold": case["threshold"],
                "object_count": 1,
            }
        ],
    }
    return arrays | {
        f"{folder}.json": _json(case),
        "fixtures/segmentation/manifest.json": _json(manifest),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_segmentation_replay_accepts_only_numeric_outputs_with_valid_hashes(
    tmp_path: Path,
    dtype: type[np.floating],
) -> None:
    files = _segmentation_files()
    folder = "fixtures/segmentation/platform_rounding"
    map_path = f"{folder}/threshold_map.npy"
    files[map_path] = _npy(np.array([[0.24, 0.35]], dtype=dtype))
    case = json.loads(files[f"{folder}.json"])
    case["sha256"][map_path] = hashlib.sha256(files[map_path]).hexdigest()
    files[f"{folder}.json"] = _json(case)
    _write(tmp_path, files)
    name = f"{folder}.json"
    actual = json.loads(files[name])
    actual["threshold"] = 0.10339596608729282
    actual["guide_threshold"] = 0.06994056869901379
    # Captured Linux maximum map drift, not the much looser native-port tolerance.
    files[f"{folder}/threshold_map.npy"] = _npy(
        np.array([[0.24 + 6.931382839914235e-08, 0.35]], dtype=dtype)
    )
    actual["sha256"][f"{folder}/threshold_map.npy"] = hashlib.sha256(
        files[f"{folder}/threshold_map.npy"]
    ).hexdigest()
    files[name] = _json(actual)
    manifest = json.loads(files["fixtures/segmentation/manifest.json"])
    manifest["cases"][0]["threshold"] = actual["threshold"]
    files["fixtures/segmentation/manifest.json"] = _json(manifest)
    assert set(verify_segmentation_replay(tmp_path, files)) == {
        f"{folder}/threshold_map.npy",
        name,
        "fixtures/segmentation/manifest.json",
    }


@pytest.mark.parametrize(
    "field",
    [
        "threshold",
        "guide_threshold",
        "settings",
        "object_count",
        "shape",
        "labels.npy",
        "small_removed_labels.npy",
        "pixels.npy",
        "threshold_map.npy",
        "map_dtype",
        "map_shape",
        "map_nan",
        "hash",
        "manifest_count",
    ],
)
def test_segmentation_replay_rejects_changes_outside_rounding(
    tmp_path: Path, field: str
) -> None:
    files = _segmentation_files()
    _write(tmp_path, files)
    files = copy.copy(files)
    folder = "fixtures/segmentation/platform_rounding"
    name = f"{folder}.json"
    case = json.loads(files[name])
    if field in {"threshold", "guide_threshold"}:
        case[field] += 1e-6
    elif field == "settings":
        case["settings"]["log_transform"] = False
    elif field == "object_count":
        case[field] += 1
    elif field == "shape":
        case[field] = [2, 1]
    elif field == "hash":
        case["sha256"][f"{folder}/threshold_map.npy"] = "0" * 64
    elif field == "manifest_count":
        manifest = json.loads(files["fixtures/segmentation/manifest.json"])
        manifest["cases"][0]["object_count"] += 1
        files["fixtures/segmentation/manifest.json"] = _json(manifest)
    else:
        arrays = {
            "labels.npy": np.array([[1, 1]], dtype=np.int32),
            "small_removed_labels.npy": np.array([[1, 1]], dtype=np.int32),
            "pixels.npy": np.array([[21, 200]], dtype=np.uint8),
            "threshold_map.npy": np.array([[0.24 + 1e-6, 0.35]]),
            "map_dtype": np.array([[0.24, 0.35]], dtype=np.float32),
            "map_shape": np.array([[0.24], [0.35]]),
            "map_nan": np.array([[np.nan, 0.35]]),
        }
        filename = "threshold_map.npy" if field.startswith("map_") else field
        path = f"{folder}/{filename}"
        files[path] = _npy(arrays[field])
        case["sha256"][path] = hashlib.sha256(files[path]).hexdigest()
    files[name] = _json(case)
    with pytest.raises(AssertionError, match="fixtures/segmentation/"):
        verify_segmentation_replay(tmp_path, files)


@pytest.mark.parametrize("kind", ["measurement", "segmentation"])
def test_replay_leaves_input_bytes_and_provenance_strict(
    tmp_path: Path, kind: str
) -> None:
    files = (
        _measurement_files() if kind == "measurement" else _segmentation_files()
    ) | {"reference/pipelines/example.cppipe": b"original pipeline\n"}
    _write(tmp_path, files)
    files["reference/pipelines/example.cppipe"] = b"changed pipeline\n"
    verify = (
        verify_measurement_replay
        if kind == "measurement"
        else verify_segmentation_replay
    )
    with pytest.raises(AssertionError, match=r"example\.cppipe"):
        verify(tmp_path, files)
