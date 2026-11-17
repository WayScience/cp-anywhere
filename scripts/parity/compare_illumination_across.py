"""Compare cp_anywhere.calculate_illumination_across against every real
oracle fixture in fixtures/illumination_calculate_across/."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from cp_anywhere import calculate_illumination_across  # noqa: E402

FIXTURES = ROOT / "fixtures/illumination_calculate_across"


def main() -> None:
    mismatches = []
    checked = 0
    for path in sorted(FIXTURES.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text())
        settings = data["settings"]
        images = [np.asarray(img, dtype=np.float64) for img in data["images"]]
        masks = [
            None if m is None else np.asarray(m, dtype=bool) for m in data["masks"]
        ]
        try:
            actual = calculate_illumination_across(
                images,
                masks=masks,
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
        expected = np.asarray(data["expected"], dtype=np.float64)
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True):
            max_abs = np.nanmax(np.abs(actual - expected))
            mismatches.append((path.name, f"max abs diff {max_abs}"))
        checked += 1
    print(json.dumps({"checked": checked, "mismatches": len(mismatches)}, indent=2))
    for name, reason in mismatches[:20]:
        print(name, reason)


if __name__ == "__main__":
    main()
