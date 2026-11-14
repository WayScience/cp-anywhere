"""Command line measurement of saved image/array files."""

import argparse
from pathlib import Path

import numpy as np

from cp_anywhere.api import measure
from cp_anywhere.io import load_array


def trigger() -> None:
    """Load channels and labels from image/array files and write feature rows."""
    parser = argparse.ArgumentParser(prog="cp-anywhere")
    parser.add_argument("--image", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--objects", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--image-id", default="Image")
    parser.add_argument(
        "--family",
        action="append",
        choices=("image_intensity", "object_intensity", "neighbors"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(items: list[str]) -> dict[str, np.ndarray]:
        arrays = {}
        for item in items:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                parser.error("each input must use NAME=PATH")
            arrays[name] = load_array(Path(path))
        return arrays

    result = measure(
        load(args.image),
        load(args.objects),
        mask=load_array(args.mask) if args.mask else None,
        image_id=args.image_id,
        families=args.family,
    )
    if args.output.suffix == ".json":
        result.to_json(args.output)
    elif args.output.suffix in (".arrow", ".feather"):
        result.to_arrow(args.output)
    else:
        result.to_csv(args.output)
