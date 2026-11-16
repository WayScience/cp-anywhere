# ty: ignore[import-not-found]
"""Run real CellProfiler MeasureGranularity; capture 2D fixtures and Rust tests.

Compatibility adapter: CellProfiler 4.2.8 passes the removed ``selem`` keyword.
Only that keyword is renamed to ``footprint`` for erosion, dilation, and
reconstruction. All numeric work uses the installed, compiled SciPy/skimage
implementations. The reference installation is never modified.

Run with the reference environment's Python and CellProfiler on PYTHONPATH.
"""

# ruff: noqa: E402

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import cellprofiler_core.preferences
import numpy as np
import scipy
import skimage
import skimage.morphology

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules import measuregranularity
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/granularity"
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
ADAPTER = (
    "Rename only selem->footprint in skimage.morphology erosion/dilation/"
    "reconstruction; forward to installed implementations."
)


def install_compatibility_adapter() -> None:
    """Rename the removed ``selem`` kwarg; forward to the real compiled ops."""
    for name in ("erosion", "dilation", "reconstruction"):
        original: Callable[..., Any] = getattr(skimage.morphology, name)

        @functools.wraps(original)
        def compatible(
            *args: object, _original: Callable[..., Any] = original, **kwargs: object
        ) -> object:
            if "selem" in kwargs:
                kwargs["footprint"] = kwargs.pop("selem")
            return _original(*args, **kwargs)

        setattr(skimage.morphology, name, compatible)


_Case = tuple[
    str, np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[float, float, int, int]
]


def cases() -> Iterator[_Case]:
    """Yield 2D fixture pixels/masks/objects/settings for the live oracle."""
    for name, shape, settings, masked in (
        ("defaults", (65, 73), (0.25, 0.25, 10, 16), False),
        ("defaults_masked", (65, 73), (0.25, 0.25, 10, 16), True),
        ("full", (17, 19), (1.0, 1.0, 3, 7), False),
        ("full_masked", (17, 19), (1.0, 1.0, 3, 7), True),
        ("half", (33, 37), (0.5, 0.5, 3, 8), True),
        ("uneven", (19, 23), (0.63, 0.57, 2, 7), True),
        ("uneven_background_only", (19, 23), (1.0, 0.37, 2, 7), True),
        ("uneven_image_only", (19, 23), (0.73, 1.0, 2, 7), True),
        # NOTE: no singleton-dimension (1xN/Nx1/1x1) cases here. Real
        # CellProfiler 4.2.8's own MeasureGranularity.run_on_image_setting
        # divides by (orig_shape[dim] - 1) while remapping the reconstructed
        # image back to full resolution; any dimension of size 1 makes that
        # denominator zero and CellProfiler itself raises ZeroDivisionError,
        # unconditionally, before producing any oracle value -- confirmed
        # live against the pinned reference. This is a genuine real-CP input
        # domain limitation, not a cp-anywhere gap: there is no oracle value
        # to match, so no fixture is generated for it. See
        # docs/granularity.md for the declared scope.
    ):
        r, c = np.indices(shape)
        pixels = ((r * 17 + c * 13 + r * c * 7) % 41) / 80 + r / (10 * shape[0])
        pixels += (
            (r - shape[0] / 2) ** 2 + (c - shape[1] / 2) ** 2 < (min(shape) / 3) ** 2
        ) * 0.3
        mask = ((r * 3 + c * 7) % 17 > 1) if masked else np.ones(shape, bool)
        labels = np.where(c < shape[1] // 2, 1, 3).astype(np.uint32)
        labels[(r + c) % 7 == 0] = 0
        points = np.zeros(shape, np.uint32)
        points[shape[0] // 2, shape[1] // 2] = 1
        points[0, 0] = 4
        yield (
            name,
            pixels,
            mask,
            {"Cells": labels, "Points": points, "Empty": np.zeros(shape, np.uint32)},
            settings,
        )
    shape = (7, 9)
    for name, value, mask_value in (
        ("zero", 0.0, True),
        ("constant", 0.4, True),
        ("all_masked", 0.6, False),
        ("negative", -0.3, True),
    ):
        yield (
            name,
            np.full(shape, value),
            np.full(shape, mask_value),
            {"Cells": np.ones(shape, np.uint32)},
            (1.0, 1.0, 2, 4),
        )


def rust_float(value: float) -> str:
    """Format one f64 literal, spelling NaN/Infinity as Rust constants."""
    value = float(value)
    if np.isnan(value):
        return "f64::NAN"
    if np.isposinf(value):
        return "f64::INFINITY"
    if np.isneginf(value):
        return "f64::NEG_INFINITY"
    return repr(value)


def rust_vector(
    values: Iterable[Any], formatter: Callable[[Any], str] = rust_float
) -> str:
    """Format a ``Vec`` literal from an iterable of scalar values."""
    return "vec![" + ",".join(formatter(x) for x in values) + "]"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    install_compatibility_adapter()
    source = Path(measuregranularity.__file__)
    provenance = {
        "reference": f"CellProfiler 4.2.8 @ {SOURCE_COMMIT}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compatibility_adapter": ADAPTER,
        "scope": "2D only; real Module.run, real Measurements and pipeline dumps",
    }
    tests = ["// Generated by scripts/reference/generate_granularity.py.\n"]
    manifest = []
    for name, pixels, mask, object_arrays, settings in cases():
        module = measuregranularity.MeasureGranularity()
        module.images_list.value = "Input"
        module.wants_objects.value = bool(object_arrays)
        module.objects_list.value = ", ".join(object_arrays)
        module.set_module_num(1)
        keys = (
            "subsample_size",
            "image_sample_size",
            "element_size",
            "granular_spectrum_length",
        )
        for key, value in zip(keys, settings):
            getattr(module, key).value = value
        pipeline = Pipeline()
        pipeline.add_module(module)
        with (OUT / f"{name}.cppipe").open("w") as stream:
            pipeline.dump(stream, save_image_plane_details=False)
        image_sets = ImageSetList()
        image_set = image_sets.get_image_set(0)
        image_set.add("Input", Image(pixels, mask=mask, convert=False))
        object_set = ObjectSet()
        for object_name, labels in object_arrays.items():
            objects = Objects()
            objects.segmented = labels
            object_set.add_objects(objects, object_name)
        measurements = Measurements()
        module.run(
            Workspace(pipeline, module, image_set, object_set, measurements, image_sets)
        )
        features = [
            module.granularity_feature(i, "Input") for i in range(1, settings[3] + 1)
        ]
        expected_image = [
            float(measurements.get_current_image_measurement(f)) for f in features
        ]
        expected_objects = {
            obj: [
                measurements.get_current_measurement(obj, f).tolist() for f in features
            ]
            for obj in object_arrays
        }
        result = {
            **provenance,
            "settings": dict(zip(keys, settings)),
            "pixels": pixels.tolist(),
            "mask": mask.tolist(),
            "objects": {key: value.tolist() for key, value in object_arrays.items()},
            "expected": {"image": expected_image, "objects": expected_objects},
        }
        # Python's explicit NaN/Infinity spelling preserves nonfinite oracle values.
        (OUT / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=True) + "\n"
        )
        lines = [
            f"#[test] fn oracle_{name}() {{",
            f"let pixels = {rust_vector(pixels.ravel())};",
            f"let mask = {rust_vector(mask.ravel(), lambda v: str(bool(v)).lower())};",
        ]
        for idx, labels in enumerate(object_arrays.values()):
            lines.append(
                f"let labels_{idx}: Vec<u32> = {rust_vector(labels.ravel(), str)};"
            )
        objects = ",".join(
            f'("{obj}", labels_{idx}.as_slice())'
            for idx, obj in enumerate(object_arrays)
        )
        config = (
            f"&Config {{ subsample_size: {rust_float(settings[0])}, "
            f"image_sample_size: {rust_float(settings[1])}, "
            f"element_size: {settings[2]}, "
            f"granular_spectrum_length: {settings[3]} }}"
        )
        lines.append(
            f"let actual = measure(&pixels, &[{objects}], Some(&mask), "
            f"{pixels.shape[0]}, {pixels.shape[1]}, {config}).unwrap();"
        )
        image_assert = (
            f"assert_values(&actual.image, &{rust_vector(expected_image)}, "
            f'"{name}:image");'
        )
        lines.append(image_assert)
        for obj, scales in expected_objects.items():
            for scale, values in enumerate(scales):
                object_assert = (
                    f'assert_values(&actual.objects["{obj}"][{scale}], '
                    f'&{rust_vector(values)}, "{name}:{obj}:{scale + 1}");'
                )
                lines.append(object_assert)
        lines.append("}\n")
        tests.append("\n".join(lines))
        manifest.append(
            {
                "name": name,
                "shape": list(pixels.shape),
                "settings": dict(zip(keys, settings)),
            }
        )
        print(f"{name}: {len(expected_image)} scales, {len(expected_objects)} sets")
    (OUT / "oracle_tests.rs").write_text("\n".join(tests))
    (OUT / "manifest.json").write_text(
        json.dumps({**provenance, "cases": manifest}, indent=2) + "\n"
    )


if __name__ == "__main__":
    generate()
