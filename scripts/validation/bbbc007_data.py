"""Prepare immutable BBBC007 fields and independent calibration folds."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import tifffile

from scripts.validation.pilot_checks import (
    leave_one_out,
    manual_labels,
    normalized_grayscale,
)

ARCHIVES = {
    "images": "b7009e2fce0a3152a5c9adda916eaa699d09696f4bd02a7d05d12d041e30c6d1",
    "outlines": "6a5246f9a9d743d22eafdb409fae638a8461af97e9ff9c4a92f25eba236224d3",
}
CHANNEL_SUFFIXES = (
    ("_D_1UL", "DNA"),
    ("_F_2UL", "Actin"),
    ("d0", "DNA"),
    ("d1", "Actin"),
    ("d", "DNA"),
    ("f", "Actin"),
)


def sha256(path: Path) -> str:
    """Identify exact input bytes, including locally prepared arrays."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_archives(directory: Path) -> dict[str, Path]:
    """Fetch public CC0 archives only when absent, and verify pinned hashes."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, expected in ARCHIVES.items():
        name = f"BBBC007_v1_{kind}.zip"
        path = directory / name
        if not path.exists():
            url = f"https://data.broadinstitute.org/bbbc/BBBC007/{name}"
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        if sha256(path) != expected:
            raise ValueError(f"source archive checksum mismatch: {path}")
        paths[kind] = path
    return paths


def _field_key(path: str) -> tuple[str, str, str]:
    file = Path(path)
    for suffix, channel in CHANNEL_SUFFIXES:
        if file.stem.endswith(suffix):
            return file.parent.name, file.stem[: -len(suffix)], channel
    raise ValueError(f"unrecognized dataset channel: {path}")


def _store(directory: Path, name: str, values: np.ndarray) -> str:
    relative = f"arrays/{name}.npy"
    np.save(directory / relative, values, allow_pickle=False)
    return relative


def prepare_fields(archives: dict[str, Path], directory: Path) -> tuple[dict, list]:
    """Keep actual 2D grayscale pairs; explicitly report RGB exclusions.

    Six source fields have RGB storage with unequal channels. No channel
    stripping or luminance conversion is guessed for these measurements.
    """
    pairs = defaultdict(dict)
    with (
        zipfile.ZipFile(archives["images"]) as images,
        zipfile.ZipFile(archives["outlines"]) as outlines,
    ):
        for name in sorted(images.namelist()):
            if not name.lower().endswith(".tif") or "__MACOSX" in name:
                continue
            cohort, field, channel = _field_key(name)
            outline_name = name.replace("BBBC007_v1_images/", "BBBC007_v1_outlines/", 1)
            image_bytes = images.read(name)
            outline_bytes = outlines.read(outline_name)
            pairs[(cohort, field)][channel] = {
                "image": tifffile.imread(io.BytesIO(image_bytes)),
                "outline": tifffile.imread(io.BytesIO(outline_bytes)),
                "source": name,
                "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "outline_sha256": hashlib.sha256(outline_bytes).hexdigest(),
            }
    (directory / "arrays").mkdir(parents=True, exist_ok=True)
    cohorts = defaultdict(list)
    excluded = []
    expected_ndim = 2
    for (cohort, field), channels in sorted(pairs.items()):
        if set(channels) != {"DNA", "Actin"}:
            raise ValueError(f"unpaired source field: {cohort}/{field}")
        identifier = re.sub(r"[^a-zA-Z0-9]+", "-", f"{cohort}-{field}").strip("-")
        shapes = {
            channel: list(value["image"].shape) for channel, value in channels.items()
        }
        if any(value["image"].ndim != expected_ndim for value in channels.values()):
            excluded.append(
                {"id": identifier, "reason": "non-grayscale source", "shapes": shapes}
            )
            continue
        record = {
            "id": identifier,
            "cohort": cohort,
            "shape": shapes["DNA"],
            "sources": {},
        }
        for channel, value in channels.items():
            image = normalized_grayscale(value["image"])
            outline = np.asarray(value["outline"], dtype=bool)
            if outline.shape != image.shape:
                raise ValueError(f"annotation shape mismatch: {identifier}/{channel}")
            record[channel] = _store(directory, f"{identifier}-{channel}", image)
            record[f"{channel}_outline"] = _store(
                directory, f"{identifier}-{channel}-outline", outline
            )
            labels = manual_labels(outline)
            record[f"{channel}_manual_labels"] = _store(
                directory, f"{identifier}-{channel}-manual", labels
            )
            record["sources"][channel] = {
                key: value[key] for key in ("source", "source_sha256", "outline_sha256")
            }
        if shapes["DNA"] != shapes["Actin"]:
            raise ValueError(f"channel shape mismatch: {identifier}")
        cohorts[cohort].append(record)
    return dict(cohorts), excluded


def _derived_field(directory: Path, field: dict, identifier: str, arrays: dict) -> dict:
    record = {
        **field,
        "id": identifier,
        "reference_fields": {channel: field[channel] for channel in ("DNA", "Actin")},
    }
    for channel, array in arrays.items():
        record[channel] = _store(
            directory, f"{identifier}-{channel}", array.astype(np.float32)
        )
    return record


def stress_groups(directory: Path, fields: list[dict]) -> list[dict]:
    """Independent noisy flat fields and explicitly labeled empty/saturated inputs."""
    fields = sorted(fields, key=lambda field: field["id"])
    original = fields[0]
    shape = tuple(original["shape"])
    rows, cols = np.indices(shape, dtype=np.float64)
    x = cols / (shape[1] - 1)
    y = rows / (shape[0] - 1)
    gain = 0.9 - 0.5 * ((x - 0.3) ** 2 + (y - 0.6) ** 2)
    rng = np.random.default_rng(7924)
    calibration = []
    for index in range(3):
        identifier = f"independent-flat-{index}"
        record = {"id": identifier, "shape": list(shape)}
        for channel in ("DNA", "Actin"):
            pixels = np.clip(gain + rng.normal(0, 0.02, shape), 0, 1).astype(np.float32)
            record[channel] = _store(directory, f"{identifier}-{channel}", pixels)
        calibration.append(record)
    held_out = []
    for field in fields[:2]:
        arrays = {
            channel: np.load(directory / field[channel], allow_pickle=False) * gain
            for channel in ("DNA", "Actin")
        }
        held_out.append(
            _derived_field(directory, field, f"{field['id']}-vignette", arrays)
        )
    controlled = {
        "id": "controlled-noisy-calibration",
        "kind": "controlled_stress",
        "calibration": calibration,
        "validation": held_out,
        "illumination": {
            "smoothing_method": "Fit Polynomial",
            "automatic_object_width": "Automatic",
            "rescale_option": "No",
        },
        "description": (
            "Real held-out images with a simulated off-center quadratic gain; "
            "separate noisy flat-field calibration; not an instrument calibration."
        ),
    }
    real_calibration = fields[1:]
    empty = _derived_field(
        directory,
        original,
        f"{original['id']}-empty",
        {channel: np.zeros(shape, dtype=np.float32) for channel in ("DNA", "Actin")},
    )
    empty["skip_manual_evaluation"] = True
    saturated = _derived_field(
        directory,
        original,
        f"{original['id']}-saturated",
        {
            channel: np.clip(
                np.load(directory / original[channel], allow_pickle=False) * 4, 0, 1
            )
            for channel in ("DNA", "Actin")
        },
    )
    saturated["skip_manual_evaluation"] = True
    return [
        controlled,
        {
            "id": "stress-empty-field",
            "kind": "empty_stress",
            "calibration": real_calibration,
            "validation": [empty],
        },
        {
            "id": "stress-saturated-field",
            "kind": "saturated_stress",
            "calibration": real_calibration,
            "validation": [saturated],
        },
    ]


def prepare_inputs(data_directory: Path, directory: Path) -> Path:
    """Write a reproducible manifest; calibration and validation IDs are disjoint."""
    archives = source_archives(data_directory)
    cohorts, excluded = prepare_fields(archives, directory)
    groups = [
        fold
        for cohort, fields in sorted(cohorts.items())
        for fold in leave_one_out(re.sub(r"\W+", "-", cohort), fields)
    ]
    groups.extend(stress_groups(directory, cohorts["A9"]))
    manifest = {
        "dataset": "BBBC007v1",
        "source_url": "https://bbbc.broadinstitute.org/BBBC007",
        "license": "CC0 waiver, see dataset source page",
        "archives": {
            kind: {"name": path.name, "sha256": sha256(path)}
            for kind, path in archives.items()
        },
        "normalization": "uint8 divided by 255 then stored as float32 for both engines",
        "cohorts": {
            name: [field["id"] for field in fields] for name, fields in cohorts.items()
        },
        "excluded": excluded,
        "settings": {
            "illumination": {
                "intensity_choice": "Regular",
                "smoothing_method": "Gaussian Filter",
                "automatic_object_width": "Manually",
                "size_of_smoothing_filter": 128.0,
                "rescale_option": "Median",
            },
            "shape_scopes": ["Nuclei", "Cells"],
            "primary": {
                "threshold": "otsu",
                "min_diameter": 10.0,
                "max_diameter": 40.0,
                "exclude_size": True,
                "unclump_method": "shape",
                "watershed_method": "propagate",
            },
            "secondary": {
                "method": "propagation",
                "regularization_factor": 0.05,
                "fill_holes": True,
            },
            "tertiary": {"shrink_primary": False},
        },
        "groups": groups,
        "array_sha256": {
            str(path.relative_to(directory)): sha256(path)
            for path in sorted((directory / "arrays").glob("*.npy"))
        },
    }
    for group in groups:
        calibration = {field["id"] for field in group["calibration"]}
        held_out = {field["id"] for field in group["validation"]}
        if calibration & held_out:
            raise ValueError(f"calibration leakage: {group['id']}")
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
