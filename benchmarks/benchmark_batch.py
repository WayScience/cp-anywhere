"""Compare the public serial, thread, and process API on full BBBC007 fields."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Literal

import cp_anywhere as cp
from scripts.validation.bbbc007_data import prepare_fields, sha256, source_archives

PIPELINE = (
    Path(__file__).resolve().parents[1] / "fixtures/cppipe/tertiary/no_shrink.cppipe"
)


def _save(result: cp.ImageSetResult, directory: Path) -> None:
    if result.measurements is None:
        raise ValueError(f"No measurements for {result.image_id}")
    target = directory / result.image_id
    target.mkdir()
    result.measurements.to_parquet(target / "measurements.parquet")
    for name, objects in (result.primary | result.derived).items():
        cp.save_array(objects.labels, target / f"{name}.npy")


def _sample(
    pipeline: cp.Pipeline,
    fields: dict[str, dict[str, Path]],
    backend: Literal["thread", "process"],
    workers: int,
    directory: Path,
) -> tuple[float, dict[str, str]]:
    start = time.perf_counter()
    for result in cp.run_batch(pipeline, fields, workers=workers, backend=backend):
        _save(result, directory)
    seconds = time.perf_counter() - start
    hashes = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in directory.glob("*/*")
    }
    if len(hashes) != 4 * len(fields):
        raise ValueError("Expected three label arrays and one Parquet file per field")
    return seconds, hashes


def main() -> None:
    """Use pinned source archives; keep generated images and outputs out of Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="BBBC007 archive cache"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    configs: list[tuple[str, Literal["thread", "process"], int]] = [
        ("serial_1", "thread", 1),
        ("threads_2", "thread", 2),
        ("threads_4", "thread", 4),
        ("processes_2", "process", 2),
        ("processes_4", "process", 4),
    ]
    samples: dict[str, list[float]] = {}
    reference: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="cp-anywhere-batch-") as temporary:
        root = Path(temporary)
        cohorts, excluded = prepare_fields(source_archives(args.data_dir), root)
        fields = {
            field["id"]: {"DNA": root / field["DNA"], "Stain": root / field["Actin"]}
            for cohort in cohorts.values()
            for field in cohort
        }
        pipeline = cp.load_pipeline(PIPELINE)
        for trial in range(args.repeats + 1):
            # One warmup; rotate the timed cases so no mode always goes first.
            offset = trial % len(configs)
            for mode, backend, workers in configs[offset:] + configs[:offset]:
                with tempfile.TemporaryDirectory(dir=root) as output:
                    seconds, hashes = _sample(
                        pipeline, fields, backend, workers, Path(output)
                    )
                if reference is None:
                    reference = hashes
                elif hashes != reference:
                    raise ValueError(f"{mode} produced different outputs")
                if trial:
                    samples.setdefault(mode, []).append(seconds)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "field_ids": list(fields),
                "excluded_non_grayscale": len(excluded),
                "identical_files_per_run": len(reference or {}),
                "seconds": samples,
                "median_seconds": {
                    key: statistics.median(values) for key, values in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
