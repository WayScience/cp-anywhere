# ty: ignore[import-not-found]
"""Capture 2D measurements from the pinned CellProfiler 4.2.8 module.

Run with the reference venv and the pinned checkout on PYTHONPATH. The
surveyed pipelines select all loader images. This isolated oracle selects
one supplied image directly; the numerical settings are unchanged.
"""

# ruff: noqa: E402, PLR2004
from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from subprocess import check_output

import cellprofiler
import cellprofiler_core.preferences
import numpy as np
from numpy.typing import NDArray

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.measureimagequality import O_SELECT, MeasureImageQuality
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

SOURCE = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/cp_anywhere/tests/image_quality_data/oracle.tsv"
CHANNEL = "Orig_DNA"
PINNED = {
    "cellprofiler-core": "4.2.8",
    "centrosome": "1.2.3",
    "numpy": "1.23.5",
    "scipy": "1.9.3",
}


def oracle(  # noqa: PLR0913
    pixels: NDArray[np.floating],
    mask: NDArray[np.bool_] | None,
    *,
    blur_scales: tuple[int, ...],
    saturation: bool,
    include_scaling: bool,
    scaling: int | None,
) -> dict[str, float]:
    """Run the module on a prepared image and return its actual output columns."""
    module = MeasureImageQuality()
    module.set_module_num(1)
    module.images_choice.value = O_SELECT
    group = module.image_groups[0]
    group.image_names.value = CHANNEL
    group.include_image_scalings.value = include_scaling
    group.check_blur.value = bool(blur_scales)
    group.check_saturation.value = saturation
    group.check_intensity.value = False
    group.calculate_threshold.value = False
    for index, scale in enumerate(blur_scales):
        if index:
            module.add_scale_group(group)
        group.scale_groups[index].scale.value = scale
        assert group.scale_groups[index].scale.value == scale

    pipeline = Pipeline()
    pipeline.add_module(module)
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    image_set.add(CHANNEL, Image(pixels, mask=mask, scale=scaling, convert=False))
    measurements = Measurements()
    try:
        workspace = Workspace(
            pipeline, module, image_set, ObjectSet(), measurements, image_sets
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            module.run(workspace)
        if "Image" not in measurements.get_object_names():
            return {}
        return {
            name: float(
                np.asarray(measurements.get_current_image_measurement(name)).item()
            )
            for name in sorted(measurements.get_feature_names("Image"))
            if name.startswith("ImageQuality_")
        }
    finally:
        measurements.close()


def cases() -> Iterator[
    tuple[
        str,
        NDArray[np.floating],
        NDArray[np.bool_] | None,
        str,
        str,
        tuple[int, ...],
        bool,
        bool,
        int | None,
    ]
]:
    """Yield deterministic input arrays and settings matching the Rust tests."""
    for name, shape, dtype, pattern, mask_kind, scales, sat, scaling_flag, scale in (
        ("options", (3, 5), "float32", "pattern", "none", (), True, True, 4095),
        (
            "options_zero_scale",
            (3, 5),
            "float64",
            "pattern",
            "none",
            (),
            False,
            True,
            0,
        ),
        (
            "nothing_selected",
            (3, 5),
            "float32",
            "pattern",
            "none",
            (),
            False,
            False,
            None,
        ),
        ("pedcp90", (64, 80), "float32", "pattern", "none", (20,), True, False, None),
        (
            "pedcp90_band",
            (64, 80),
            "float32",
            "disk_grid",
            "none",
            (20,),
            True,
            False,
            None,
        ),
        (
            "quantize_ties",
            (23, 40),
            "float32",
            "quantize_ties",
            "none",
            (7,),
            False,
            False,
            None,
        ),
        ("cardiac", (75, 83), "float32", "pattern", "none", (20,), True, True, 65535),
        (
            "masked64",
            (73, 65),
            "float64",
            "pattern",
            "holed",
            (20, 7),
            True,
            True,
            None,
        ),
        (
            "masked32",
            (99, 97),
            "float32",
            "pattern",
            "holed",
            (20, 7),
            True,
            True,
            None,
        ),
        (
            "empty_mask",
            (64, 64),
            "float32",
            "pattern",
            "empty",
            (20,),
            True,
            False,
            None,
        ),
        ("constant", (64, 64), "float32", "constant", "none", (20,), True, False, None),
        ("sparse", (48, 56), "float64", "impulse", "none", (20,), True, False, None),
        ("stripe", (64, 96), "float32", "stripe", "none", (20,), True, False, None),
        (
            "prime_stripe",
            (67, 89),
            "float32",
            "stripe",
            "none",
            (20,),
            True,
            False,
            None,
        ),
        ("one_ring", (24, 33), "float64", "pattern", "none", (20,), True, False, None),
        ("no_ring", (23, 65), "float32", "pattern", "none", (20,), True, False, None),
        ("singleton", (1, 39), "float32", "pattern", "none", (20,), True, False, None),
        (
            "nonfinite",
            (32, 33),
            "float32",
            "nonfinite",
            "none",
            (20,),
            True,
            False,
            None,
        ),
    ):
        r, c = np.indices(shape)
        if pattern == "constant":
            pixels = np.full(shape, 0.25)
        elif pattern == "impulse":
            pixels = (r == 4) * (c == 7) * 0.5
        elif pattern == "stripe":
            pixels = (c % 2) * 0.5
        elif pattern == "disk_grid":
            pixels = ((r % 41 - 18) ** 2 + (c % 47 - 20) ** 2 < 128) * 0.5
        elif pattern == "quantize_ties":
            levels = np.array(
                [0x3DC1BD6D, 0x3E800000, 0x3EF3AA97, 0x3F5B72EA], np.uint32
            ).view(np.float32)
            pixels = levels[(r + 2 * c + r * c) % len(levels)]
        else:
            pixels = ((39 * r + 11 * c + 7 * r * c) % 53 + (r + c) % 5) / 256
            if pattern == "nonfinite":
                pixels[0, 0] = np.nan
        mask = None
        if mask_kind == "holed":
            mask = ((r * 3 + c * 7) % 17 > 1) & ((r > 20) | (c > 21))
        if mask_kind == "empty":
            mask = np.zeros(shape, dtype=bool)
        yield (
            name,
            pixels.astype(dtype),
            mask,
            pattern,
            mask_kind,
            scales,
            sat,
            scaling_flag,
            scale,
        )


def main() -> None:
    """Write compact, dependency-free golden fixtures for Rust."""
    source = cellprofiler.__file__
    assert source is not None
    checkout = Path(source).resolve().parents[1]
    head = check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if head != SOURCE or any(version(key) != pin for key, pin in PINNED.items()):
        msg = "Use the pinned CellProfiler checkout and reference/requirements.lock"
        raise RuntimeError(msg)
    rows = [
        f"# CellProfiler {SOURCE}; NumPy {np.__version__}; SciPy {version('scipy')}; "
        f"centrosome {version('centrosome')}",
    ]
    for (
        name,
        pixels,
        mask,
        pattern,
        mask_kind,
        scales,
        sat,
        scaling_flag,
        scale,
    ) in cases():
        values = oracle(
            pixels,
            mask,
            blur_scales=scales,
            saturation=sat,
            include_scaling=scaling_flag,
            scaling=scale,
        )
        row = (
            name,
            *pixels.shape,
            pixels.dtype,
            pattern,
            mask_kind,
            ",".join(map(str, scales)) or "-",
            int(sat),
            int(scaling_flag),
            "-" if scale is None else scale,
            *(f"{name}={value!r}" for name, value in values.items()),
        )
        rows.append("\t".join(map(str, row)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
