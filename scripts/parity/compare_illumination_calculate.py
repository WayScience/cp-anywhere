"""Compare the Rust-core CorrectIlluminationCalculate port against
every real oracle fixture in fixtures/illumination_calculate/."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from cp_anywhere import _native  # noqa: E402

FIXTURES = ROOT / "fixtures/illumination_calculate"


def main() -> None:
    mismatches = []
    checked = 0
    for path in sorted(FIXTURES.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text())
        settings = data["settings"]
        pixels = np.asarray(data["pixels"], dtype=np.float64)
        mask = None if data["mask"] is None else np.asarray(data["mask"], dtype=bool)
        rows, cols = pixels.shape
        try:
            actual = _native.calculate_illumination_each(
                np.ascontiguousarray(pixels.ravel()),
                rows,
                cols,
                mask=None if mask is None else np.ascontiguousarray(mask.ravel()),
                intensity_choice=settings["intensity_choice"],
                block_size=settings["block_size"],
                dilate_objects=settings["dilate_objects"],
                object_dilation_radius=settings["object_dilation_radius"],
                smoothing_method=settings["smoothing_method"],
                automatic_object_width=settings["automatic_object_width"],
                object_width=float(settings["object_width"]),
                size_of_smoothing_filter=float(settings["size_of_smoothing_filter"]),
                rescale_option=settings["rescale_option"],
            )
        except Exception as exc:
            mismatches.append((path.name, f"raised {exc!r}"))
            continue
        expected = np.asarray(data["expected"], dtype=np.float64).ravel()
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True):
            max_abs = np.nanmax(np.abs(actual - expected))
            mismatches.append((path.name, f"max abs diff {max_abs}"))
        checked += 1
    print(json.dumps({"checked": checked, "mismatches": len(mismatches)}, indent=2))
    for name, reason in mismatches[:20]:
        print(name, reason)


if __name__ == "__main__":
    main()
