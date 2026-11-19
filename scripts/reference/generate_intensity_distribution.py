# ty: ignore[import-not-found]
"""Run real CellProfiler MeasureObjectIntensityDistribution; capture fixtures.

Compatibility adapter: CellProfiler 4.2.8 passes the removed ``selem``
keyword to skimage.morphology functions in some code paths it shares with
other modules. Only that keyword is renamed to ``footprint``; all numeric
work still uses the installed, compiled SciPy/skimage implementations. The
reference installation is never modified. (This module itself doesn't call
skimage.morphology directly, but the adapter is installed defensively,
exactly like scripts/reference/generate_granularity.py.)

Scope: the module's DEFAULT center choice only (center_choice == "These
objects", i.e. center_object_name is None in the real source -- the radial
center is each object's own point farthest from its edge). The other two
center choices need a second, separately-selected object set and are an
explicit, documented gap in crates/cp_anywhere/src/intensity_distribution.rs.

Run with the reference environment's Python and CellProfiler on PYTHONPATH.
"""

# ruff: noqa: E402

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

import cellprofiler_core.preferences
import numpy as np
import scipy
import skimage

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules import measureobjectintensitydistribution as moid
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/intensity_distribution"
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
ADAPTER = (
    "Rename only selem->footprint in skimage.morphology erosion/dilation/"
    "reconstruction; forward to installed implementations. Installed "
    "defensively; this module does not call skimage.morphology directly."
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


class _Settings(TypedDict):
    """A `MeasureObjectIntensityDistribution` bin-count/Zernike
    configuration, matching `cp_anywhere::intensity_distribution::Config`."""

    bin_count: int
    wants_scaled: bool
    maximum_radius: float
    wants_zernikes: str
    zernike_degree: int


_Case = tuple[str, np.ndarray, dict[str, np.ndarray], _Settings]


def _ring_labels(shape: tuple[int, int]) -> np.ndarray:
    """A single annulus (hole in the middle): a genuinely non-convex shape."""
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    dist2 = (r - cy) ** 2 + (c - cx) ** 2
    outer = min(shape) / 2.0 - 1.0
    inner = outer / 2.2
    labels[(dist2 <= outer**2) & (dist2 >= inner**2)] = 1
    return labels


def _elongated_c_shape(shape: tuple[int, int], label: int) -> np.ndarray:
    """A concave, elongated "C" shape: non-convex, exercises irregular rings."""
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    dist2 = (r - cy) ** 2 + (c - cx) ** 2
    outer = min(shape) / 2.0 - 1.0
    inner = outer / 2.0
    mouth = c > cx
    labels[(dist2 <= outer**2) & (dist2 >= inner**2) & ~mouth] = label
    return labels


def cases() -> Iterator[_Case]:
    """Yield fixture pixels/objects/settings for the live oracle."""

    def pixels_for(shape: tuple[int, int]) -> np.ndarray:
        r, c = np.indices(shape)
        base = ((r * 11 + c * 7 + r * c * 3) % 29) / 29.0
        return base + 0.05

    # 1. CellProfiler's own defaults: scaled bins, 4 bins, Zernikes off.
    shape = (21, 23)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 10) ** 2 + (c - 11) ** 2 <= 8**2] = 1
    yield (
        "defaults",
        pixels_for(shape),
        {"Cells": labels},
        {
            "bin_count": 4,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_NONE,
            "zernike_degree": 9,
        },
    )

    # 2. Multiple objects: one touching the image border, one a small dot,
    # one a bigger disk -- all in one label image, with Zernikes on
    # (magnitudes and phase) at a small degree.
    shape = (30, 34)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 8) ** 2 + (c - 8) ** 2 <= 6**2] = 1  # interior disk
    labels[(r - 0) ** 2 + (c - 20) ** 2 <= 5**2] = 2  # touches top border
    labels[(r - 22) ** 2 + (c - 28) ** 2 <= 2**2] = 3  # small dot
    yield (
        "multi_object_border_and_small",
        pixels_for(shape),
        {"Cells": labels},
        {
            "bin_count": 5,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_MAGNITUDES_AND_PHASE,
            "zernike_degree": 4,
        },
    )

    # 3. A genuinely non-convex annulus (ring) object: binning must follow
    # the real geodesic propagation through the ring's own hole-free mask,
    # not a naive concentric-circle assumption.
    shape = (25, 25)
    yield (
        "ring_shape",
        pixels_for(shape),
        {"Cells": _ring_labels(shape)},
        {
            "bin_count": 6,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_MAGNITUDES,
            "zernike_degree": 6,
        },
    )

    # 4. Elongated concave "C" shape plus a second disk, unscaled bins with
    # a tight maximum_radius so the overflow bin actually gets populated.
    shape = (28, 28)
    c_shape = _elongated_c_shape(shape, 1)
    r, c = np.indices(shape)
    c_shape[(r - 5) ** 2 + (c - 5) ** 2 <= 3**2] = 2
    yield (
        "c_shape_unscaled_overflow",
        pixels_for(shape),
        {"Cells": c_shape},
        {
            "bin_count": 3,
            "wants_scaled": False,
            "maximum_radius": 4,
            "wants_zernikes": moid.Z_NONE,
            "zernike_degree": 9,
        },
    )

    # 5. Large bin count, high Zernike degree, single clean disk -- exercises
    # degree/bin-count scaling and the Zernike column-count formula.
    shape = (20, 20)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 10) ** 2 + (c - 10) ** 2 <= 8**2] = 1
    yield (
        "high_bins_high_degree",
        pixels_for(shape),
        {"Cells": labels},
        {
            "bin_count": 8,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_MAGNITUDES_AND_PHASE,
            "zernike_degree": 7,
        },
    )

    # 6. A label id *gap*: ids 1 and 3 present, id 2 entirely absent (zero
    # foreground pixels for that id) -- confirms real CP's own zero-division
    # behavior for a sparse label range, live, rather than assuming it.
    shape = (15, 15)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 4) ** 2 + (c - 4) ** 2 <= 2**2] = 1
    labels[(r - 10) ** 2 + (c - 10) ** 2 <= 2**2] = 3
    yield (
        "sparse_label_gap",
        pixels_for(shape),
        {"Cells": labels},
        {
            "bin_count": 3,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_MAGNITUDES,
            "zernike_degree": 2,
        },
    )

    # 7. An entirely empty label image: no objects at all.
    shape = (6, 7)
    yield (
        "no_objects",
        pixels_for(shape),
        {"Cells": np.zeros(shape, np.uint32)},
        {
            "bin_count": 3,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_MAGNITUDES_AND_PHASE,
            "zernike_degree": 3,
        },
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


def run_module(
    pixels: np.ndarray, labels: np.ndarray, settings: _Settings
) -> tuple[Measurements, moid.MeasureObjectIntensityDistribution]:
    module = moid.MeasureObjectIntensityDistribution()
    module.images_list.value = "Input"
    module.objects[0].object_name.value = "Cells"
    module.objects[0].center_choice.value = moid.C_SELF
    module.bin_counts[0].wants_scaled.value = settings["wants_scaled"]
    module.bin_counts[0].bin_count.value = settings["bin_count"]
    module.bin_counts[0].maximum_radius.value = settings["maximum_radius"]
    module.wants_zernikes.value = settings["wants_zernikes"]
    module.zernike_degree.value = settings["zernike_degree"]
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)

    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    image_set.add("Input", Image(pixels, convert=False))
    object_set = ObjectSet()
    objects = Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "Cells")
    measurements = Measurements()
    module.run(
        Workspace(pipeline, module, image_set, object_set, measurements, image_sets)
    )
    return measurements, module


def expected_columns(
    measurements: Measurements, module: moid.MeasureObjectIntensityDistribution
) -> dict[str, list[float]]:
    columns = {}
    for _obj_name, feature, _coltype in module.get_measurement_columns(None):
        values = measurements.get_current_measurement("Cells", feature)
        columns[feature] = [float(v) for v in np.asarray(values, dtype=float)]
    return columns


def zernike_rust_config(settings: _Settings) -> str:
    degree = settings["zernike_degree"]
    if settings["wants_zernikes"] == moid.Z_NONE:
        return "ZernikeOption::None"
    if settings["wants_zernikes"] == moid.Z_MAGNITUDES:
        return f"ZernikeOption::Magnitudes {{ degree: {degree} }}"
    return f"ZernikeOption::MagnitudesAndPhase {{ degree: {degree} }}"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    install_compatibility_adapter()
    source = Path(moid.__file__)
    provenance = {
        "reference": f"CellProfiler 4.2.8 @ {SOURCE_COMMIT}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compatibility_adapter": ADAPTER,
        "scope": (
            "2D only; center_choice == 'These objects' (default) only; "
            "real Module.run, real Measurements and pipeline dumps"
        ),
    }
    tests = ["// Generated by scripts/reference/generate_intensity_distribution.py.\n"]
    manifest = []
    total_scalars = 0
    for name, pixels, object_arrays, settings in cases():
        labels = object_arrays["Cells"]
        measurements, module = run_module(pixels, labels, settings)
        with (OUT / f"{name}.cppipe").open("w") as stream:
            module_pipeline = Pipeline()
            module.set_module_num(1)
            module_pipeline.add_module(module)
            module_pipeline.dump(stream, save_image_plane_details=False)
        expected = expected_columns(measurements, module)

        result = {
            **provenance,
            "settings": settings,
            "pixels": pixels.tolist(),
            "labels": labels.tolist(),
            "expected": expected,
        }
        (OUT / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=True) + "\n"
        )

        config = (
            f'Config {{ image_name: "Input".to_string(), '
            f"bin_count: {settings['bin_count']}, "
            f"wants_scaled: {str(settings['wants_scaled']).lower()}, "
            f"maximum_radius: {rust_float(settings['maximum_radius'])}, "
            f"zernikes: {zernike_rust_config(settings)} }}"
        )
        lines = [
            f"#[test] fn oracle_{name}() {{",
            f"let pixels = {rust_vector(pixels.ravel())};",
            f"let labels: Vec<u32> = {rust_vector(labels.ravel(), str)};",
            f"let config = {config};",
            f"let actual = measure(&pixels, &labels, {pixels.shape[0]}, "
            f"{pixels.shape[1]}, &config).unwrap();",
        ]
        for feature, values in sorted(expected.items()):
            lines.append(
                f'assert_values(&actual.columns["{feature}"], '
                f'&{rust_vector(values)}, "{name}:{feature}");'
            )
            total_scalars += len(values)
        lines.append("}\n")
        tests.append("\n".join(lines))
        manifest.append(
            {
                "name": name,
                "shape": list(pixels.shape),
                "settings": settings,
                "n_columns": len(expected),
            }
        )
        print(f"{name}: {len(expected)} columns")
    (OUT / "oracle_tests.rs").write_text("\n".join(tests))
    (OUT / "manifest.json").write_text(
        json.dumps({**provenance, "cases": manifest}, indent=2) + "\n"
    )
    print(f"TOTAL scalar comparisons: {total_scalars}")


def generate_float_precision() -> None:
    """Capture both storage dtypes using identical pixel values and geometry."""
    r, c = np.indices((11, 11))
    radius_squared = 16
    labels = ((r - 5) ** 2 + (c - 5) ** 2 <= radius_squared).astype(np.uint32)
    pixels = ((((17 * r + 11 * c + 3 * r * c) % 37) + 1) / 41).astype(np.float32)
    lines = [
        "// Generated by scripts/reference/generate_intensity_distribution.py.",
        "// CellProfiler 4.2.8, NumPy 1.23.5, SciPy 1.9.3; Image(convert=False).",
        "// Pixels have identical values; only the image storage dtype differs.",
        "#[test] fn oracle_radial_pixel_precision() {",
        f"let pixels = {rust_vector(pixels.ravel())};",
        f"let labels = {rust_vector(labels.ravel(), str)};",
    ]
    for bin_count in (4, 9):
        settings: _Settings = {
            "bin_count": bin_count,
            "wants_scaled": True,
            "maximum_radius": 100,
            "wants_zernikes": moid.Z_NONE,
            "zernike_degree": 9,
        }
        lines.append(
            f'let config = Config {{ image_name: "Input".into(), '
            f"bin_count: {bin_count}, wants_scaled: true, maximum_radius: 100.0, "
            "zernikes: ZernikeOption::None };"
        )
        for dtype in (np.float32, np.float64):
            measurements, module = run_module(pixels.astype(dtype), labels, settings)
            expected = expected_columns(measurements, module)
            measurements.close()
            precision = "Float32" if dtype == np.float32 else "Float64"
            lines.append(
                "let actual = measure_with_precision(&pixels, &labels, 11, 11, "
                f"&config, false, PixelPrecision::{precision}).unwrap();"
            )
            lines.append(f"assert_eq!(actual.columns.len(), {len(expected)});")
            lines.extend(
                f'assert_pilot_strict(actual.columns["{feature}"][0], '
                f'{rust_float(values[0])}, "{dtype.__name__}:{feature}");'
                for feature, values in sorted(expected.items())
            )
    lines.append("}")
    (OUT / "float_precision.rs").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    generate()
    generate_float_precision()
