"""Command line measurement of saved image/array files."""

import argparse
from pathlib import Path

import numpy as np

from cp_anywhere.api import MeasurementResult, measure
from cp_anywhere.cppipe import load_pipeline
from cp_anywhere.io import load_array


def _write_result(result: MeasurementResult, args: argparse.Namespace) -> None:
    """Dispatch on --output's extension to the matching MeasurementResult writer."""
    output: Path = args.output
    if args.parquet_mode == "joined":
        # --output may be a single .parquet file or a directory (a
        # paginated Parquet dataset); to_parquet tells those apart itself.
        result.to_parquet(
            output,
            mode="joined",
            join=args.join,
            if_exists=args.if_exists,
            max_rows_per_file=args.max_rows_per_file,
        )
    elif args.parquet_mode in ("wide", "array"):
        # --output is a directory in these modes (one file per scope),
        # not a single extension-dispatched file; see to_parquet's docstring.
        result.to_parquet(output, mode=args.parquet_mode)
    elif output.suffix == ".json":
        result.to_json(output)
    elif output.suffix in (".arrow", ".feather"):
        result.to_arrow(output)
    elif output.suffix == ".parquet":
        result.to_parquet(output, mode="long")
    elif output.suffix == ".duckdb":
        result.to_duckdb(output, if_exists=args.if_exists)
    elif output.suffix in (".sqlite", ".sqlite3", ".db"):
        result.to_sqlite(output, if_exists=args.if_exists)
    else:
        result.to_csv(output)


def trigger() -> None:
    """Load channels and labels from image/array files and write feature rows."""
    parser = argparse.ArgumentParser(prog="cp-anywhere")
    parser.add_argument("--image", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--objects", action="append", metavar="NAME=PATH", default=[])
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--image-id", default="Image")
    parser.add_argument(
        "--family",
        action="append",
        choices=("image_intensity", "object_intensity", "neighbors"),
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        help=(
            "a real CellProfiler .cppipe file (IdentifyPrimaryObjects/"
            "Measure* module settings only; see "
            "spec/cppipe-compatibility.md). When given, --objects/"
            "--family are ignored: the pipeline's own translated "
            "settings and image/object names are used instead."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--if-exists",
        choices=("append", "replace"),
        default="append",
        help=(
            "for --output .duckdb/.sqlite files, or --parquet-mode "
            "joined with a directory --output: 'append' (default) adds "
            "this result's rows to whatever already exists; 'replace' "
            "drops/deletes and recreates it."
        ),
    )
    parser.add_argument(
        "--parquet-mode",
        choices=("long", "wide", "array", "joined"),
        default="long",
        help=(
            "for --output .parquet only: 'long' (default) writes one "
            "file with one row per object per feature. 'wide' and "
            "'array' both treat --output as a directory and write one "
            "Per_Image.parquet/Per_<ObjectName>.parquet file per scope "
            "-- 'wide' with one named column per feature (the "
            "traditional tabular shape), 'array' with every feature "
            "packed into a single fixed-size array column instead. "
            "'joined' writes one combined, CytoTable-style wide table "
            "(every object scope joined on object_id, columns prefixed "
            "by scope name) to a single file or, for a directory "
            "--output, a paginated Parquet dataset "
            "(see spec/database-export.md)."
        ),
    )
    parser.add_argument(
        "--join",
        choices=("outer", "inner"),
        default="outer",
        help=(
            "for --parquet-mode joined only: 'outer' (default) keeps "
            "every object_id seen in any scope; 'inner' keeps only "
            "object_ids present in every scope."
        ),
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=100_000,
        help=(
            "for --parquet-mode joined with a directory --output only: "
            "the row-count cap per written Parquet fragment file."
        ),
    )
    args = parser.parse_args()

    def load(items: list[str]) -> dict[str, np.ndarray]:
        arrays = {}
        for item in items:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                parser.error("each input must use NAME=PATH")
            arrays[name] = load_array(Path(path))
        return arrays

    if args.pipeline is not None:
        pipeline = load_pipeline(args.pipeline)
        _segmentations, _secondary, result = pipeline.run(load(args.image))
        if result is None:
            parser.error(
                "pipeline has no translated Measure* module; nothing to write "
                "(see --output's usual role; segmentation-only pipelines are "
                "not yet exposed on this CLI)"
            )
    else:
        if not args.objects:
            parser.error("--objects is required unless --pipeline is given")
        result = measure(
            load(args.image),
            load(args.objects),
            mask=load_array(args.mask) if args.mask else None,
            image_id=args.image_id,
            families=args.family,
        )
    _write_result(result, args)
