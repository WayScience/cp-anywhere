# ty: ignore[import-not-found]
"""Run real CellProfiler MeasureColocalization; capture oracle fixtures.

Scope: the module's 2D case only (its own ``volumetric()`` returning ``True``
means it nominally also supports 3D; that case is an explicit, documented
gap, matching this project's house style for every other module ported so
far -- see ``crates/cp_anywhere_core/src/colocalization.rs``).

Run with the reference environment's Python and CellProfiler on PYTHONPATH.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

import cellprofiler_core.preferences

cellprofiler_core.preferences.set_headless()

import numpy as np
import scipy
from cellprofiler.modules import measurecolocalization as mc
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

OUT = Path(__file__).resolve().parents[2] / "fixtures/colocalization"
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"


class _Settings(TypedDict):
    """A `MeasureColocalization` configuration, matching
    `cp_anywhere_core::colocalization::Config`."""

    thr: float
    images_or_objects: str
    do_corr_and_slope: bool
    do_manders: bool
    do_rwc: bool
    do_overlap: bool
    do_costes: bool
    fast_costes: str


def _default_settings(
    *,
    thr: float = 15.0,
    images_or_objects: str = mc.M_IMAGES,
    do_corr_and_slope: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    fast_costes: str = mc.M_FASTER,
) -> _Settings:
    return {
        "thr": thr,
        "images_or_objects": images_or_objects,
        "do_corr_and_slope": do_corr_and_slope,
        "do_manders": do_manders,
        "do_rwc": do_rwc,
        "do_overlap": do_overlap,
        "do_costes": do_costes,
        "fast_costes": fast_costes,
    }


_Case = tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], _Settings]


def _noisy_image(
    shape: tuple[int, int], seed: int, base: np.ndarray | None = None
) -> np.ndarray:
    """A deterministic pseudo-random image in [0, 1], optionally correlated
    with ``base`` (same shape) via a fixed linear blend."""
    rng = np.random.RandomState(seed)
    noise = rng.uniform(size=shape)
    if base is None:
        return noise
    return np.clip(0.6 * base + 0.4 * noise, 0.0, 1.0)


def cases() -> Iterator[_Case]:
    """Yield fixture image dicts/object dicts/settings for the live oracle."""

    # 1. Two correlated/noisy images, every do_* flag on (defaults), thr=15.
    shape = (18, 20)
    img_a = _noisy_image(shape, seed=1)
    img_b = _noisy_image(shape, seed=2, base=img_a)
    yield (
        "defaults_two_images",
        {"DNA": img_a, "GFP": img_b},
        {},
        _default_settings(),
    )

    # 2-6. Each do_* flag independently toggled off, one at a time.
    flag_setters: dict[str, Callable[[], _Settings]] = {
        "do_corr_and_slope": lambda: _default_settings(do_corr_and_slope=False),
        "do_manders": lambda: _default_settings(do_manders=False),
        "do_rwc": lambda: _default_settings(do_rwc=False),
        "do_overlap": lambda: _default_settings(do_overlap=False),
        "do_costes": lambda: _default_settings(do_costes=False),
    }
    for flag, make_settings in flag_setters.items():
        shape = (16, 16)
        img_a = _noisy_image(shape, seed=10)
        img_b = _noisy_image(shape, seed=11, base=img_a)
        yield (
            f"flag_off_{flag}",
            {"DNA": img_a, "GFP": img_b},
            {},
            make_settings(),
        )

    # 7. Three images: exercises all C(3, 2) = 3 pairwise permutations.
    shape = (14, 15)
    img_a = _noisy_image(shape, seed=20)
    img_b = _noisy_image(shape, seed=21, base=img_a)
    img_c = _noisy_image(shape, seed=22, base=img_b)
    yield (
        "three_images_all_pairs",
        {"Red": img_a, "Green": img_b, "Blue": img_c},
        {},
        _default_settings(),
    )

    # 8. Disjoint masks: combined mask has zero True pixels -> the real
    # module's own `else: corr = numpy.NaN; ...` whole-image branch.
    shape = (10, 12)
    img_a = Image(
        _noisy_image(shape, seed=30),
        mask=np.zeros(shape, bool) | ([True] * 6 + [False] * 6),
        convert=False,
    )
    img_b = Image(
        _noisy_image(shape, seed=31),
        mask=np.zeros(shape, bool) | ([False] * 6 + [True] * 6),
        convert=False,
    )
    yield (
        "empty_mask_pair",
        {"DNA": img_a, "GFP": img_b},
        {},
        _default_settings(),
    )

    # 9-10. fast_costes = "Fast" / "Accurate" (both served by the real
    # module's own `linear_costes`; only the step logic differs).
    for mode in (mc.M_FAST, mc.M_ACCURATE):
        shape = (16, 16)
        img_a = _noisy_image(shape, seed=40)
        img_b = _noisy_image(shape, seed=41, base=img_a)
        yield (
            f"costes_{mode.lower()}",
            {"DNA": img_a, "GFP": img_b},
            {},
            _default_settings(
                do_corr_and_slope=False,
                do_manders=False,
                do_rwc=False,
                do_overlap=False,
                fast_costes=mode,
            ),
        )

    # 11. An explicit image.scale (16-bit-style) on one image only, to
    # confirm get_scale's max(scale_1, scale_2) / fallback-to-255 live.
    shape = (16, 16)
    img_a = Image(_noisy_image(shape, seed=50), convert=False, scale=65535)
    img_b = Image(
        _noisy_image(shape, seed=51, base=_noisy_image(shape, seed=50)), convert=False
    )
    yield (
        "costes_explicit_scale",
        {"DNA": img_a, "GFP": img_b},
        {},
        _default_settings(
            do_corr_and_slope=False,
            do_manders=False,
            do_rwc=False,
            do_overlap=False,
        ),
    )

    # 12. "Both": two images and real labels (multiple objects, different
    # sizes, one tiny/degenerate single-pixel object).
    shape = (20, 22)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 6) ** 2 + (c - 6) ** 2 <= 4**2] = 1  # medium disk
    labels[(r - 14) ** 2 + (c - 16) ** 2 <= 6**2] = 2  # large disk
    labels[10, 10] = 3  # tiny, single-pixel object
    img_a = _noisy_image(shape, seed=60)
    img_b = _noisy_image(shape, seed=61, base=img_a)
    yield (
        "objects_both_varied_sizes",
        {"DNA": img_a, "GFP": img_b},
        {"Cells": labels},
        _default_settings(images_or_objects=mc.M_IMAGES_AND_OBJECTS),
    )

    # 13b. "Both" with disjoint image masks: the *whole-image* mask (not the
    # per-object mask) is what gates the real module's own per-object NaN
    # branch -- confirmed live, not assumed (see colocalization.rs's doc
    # comment). Every flagged feature family becomes the same NaN array.
    shape = (12, 14)
    labels = np.zeros(shape, np.uint32)
    labels[2:5, 2:5] = 1
    labels[7:10, 7:10] = 2
    img_a = Image(
        _noisy_image(shape, seed=80),
        mask=np.zeros(shape, bool) | ([True] * 7 + [False] * 7),
        convert=False,
    )
    img_b = Image(
        _noisy_image(shape, seed=81),
        mask=np.zeros(shape, bool) | ([False] * 7 + [True] * 7),
        convert=False,
    )
    yield (
        "objects_whole_image_mask_empty",
        {"DNA": img_a, "GFP": img_b},
        {"Cells": labels},
        _default_settings(images_or_objects=mc.M_IMAGES_AND_OBJECTS),
    )

    # NOTE: a "per-object mask empty but whole-image mask nonempty" case
    # (object entirely inside a masked-out region, while some other region
    # of the image remains valid) was tried here and removed: real
    # CellProfiler 4.2.8 itself crashes on it with
    # `ValueError: zero-size array to reduction operation minimum which
    # has no identity` from `scipy.ndimage.mean(first_pixels, labels,
    # lrange)` when `first_pixels`/`labels` (the per-object-mask-filtered
    # arrays) are empty but `n_objects > 0`. There is no real-CP oracle
    # value to match here -- this is a confirmed, documented real-CP gap,
    # not a port bug; see colocalization.rs's module doc comment.

    # 13. "Within objects" only, with a sparse label *gap* (ids 1 and 3
    # present, id 2 entirely absent) -- confirms real CP's own scipy.ndimage
    # NaN-for-missing-label behavior live, rather than assuming it.
    shape = (15, 15)
    labels = np.zeros(shape, np.uint32)
    r, c = np.indices(shape)
    labels[(r - 4) ** 2 + (c - 4) ** 2 <= 2**2] = 1
    labels[(r - 10) ** 2 + (c - 10) ** 2 <= 2**2] = 3
    img_a = _noisy_image(shape, seed=70)
    img_b = _noisy_image(shape, seed=71, base=img_a)
    yield (
        "objects_within_sparse_label_gap",
        {"DNA": img_a, "GFP": img_b},
        {"Cells": labels},
        _default_settings(images_or_objects=mc.M_OBJECTS),
    )

    # 14. Zero objects: n_objects == 0 -> every feature family is an empty
    # (zero-length) array, independent of mask/pixel content.
    shape = (10, 10)
    labels = np.zeros(shape, np.uint32)
    img_a = _noisy_image(shape, seed=100)
    img_b = _noisy_image(shape, seed=101, base=img_a)
    yield (
        "objects_zero_objects",
        {"DNA": img_a, "GFP": img_b},
        {"Cells": labels},
        _default_settings(images_or_objects=mc.M_OBJECTS),
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
    images: dict[str, np.ndarray],
    objects: dict[str, np.ndarray],
    settings: _Settings,
) -> tuple[Measurements, mc.MeasureColocalization, list[str]]:
    module = mc.MeasureColocalization()
    image_names = list(images.keys())
    module.images_list.value = ", ".join(image_names)
    object_names = list(objects.keys())
    module.objects_list.value = ", ".join(object_names)
    module.thr.value = settings["thr"]
    module.images_or_objects.value = settings["images_or_objects"]
    module.do_all.value = True
    module.do_corr_and_slope.value = settings["do_corr_and_slope"]
    module.do_manders.value = settings["do_manders"]
    module.do_rwc.value = settings["do_rwc"]
    module.do_overlap.value = settings["do_overlap"]
    module.do_costes.value = settings["do_costes"]
    module.fast_costes.value = settings["fast_costes"]
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)

    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    for name, pixels in images.items():
        image = pixels if isinstance(pixels, Image) else Image(pixels, convert=False)
        image_set.add(name, image)
    object_set = ObjectSet()
    for name, label_array in objects.items():
        cp_objects = Objects()
        cp_objects.segmented = label_array
        object_set.add_objects(cp_objects, name)
    measurements = Measurements()
    module.run(
        Workspace(pipeline, module, image_set, object_set, measurements, image_sets)
    )
    return measurements, module, image_names


def expected_columns(
    measurements: Measurements,
    module: mc.MeasureColocalization,
    object_names: list[str],
) -> dict[str, dict[str, list[float]]]:
    expected: dict[str, dict[str, list[float]]] = {
        name: {} for name in ["Image", *object_names]
    }
    for obj_name, feature, _coltype in module.get_measurement_columns(None):
        value = measurements.get_current_measurement(obj_name, feature)
        expected[obj_name][feature] = [float(v) for v in np.atleast_1d(value)]
    return expected


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(mc.__file__)
    provenance = {
        "reference": f"CellProfiler 4.2.8 @ {SOURCE_COMMIT}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": (
            "2D only (volumetric() reports True but is out of scope here, "
            "matching this project's house style for every other module); "
            "real Module.run, real Measurements and pipeline dumps"
        ),
    }
    tests = ["// Generated by scripts/reference/generate_colocalization.py.\n"]
    manifest = []
    total_scalars = 0
    for name, images, objects, settings in cases():
        measurements, module, image_names = run_module(images, objects, settings)
        object_names = list(objects.keys())
        with (OUT / f"{name}.cppipe").open("w") as stream:
            module_pipeline = Pipeline()
            module.set_module_num(1)
            module_pipeline.add_module(module)
            module_pipeline.dump(stream, save_image_plane_details=False)
        expected = expected_columns(measurements, module, object_names)

        pixels_out = {
            n: (img.pixel_data if isinstance(img, Image) else img).tolist()
            for n, img in images.items()
        }
        masks_out = {
            n: img.mask.tolist()
            for n, img in images.items()
            if isinstance(img, Image) and img.has_mask
        }
        scales_out = {
            n: img.scale
            for n, img in images.items()
            if isinstance(img, Image) and img.scale is not None
        }
        result = {
            **provenance,
            "settings": settings,
            "image_names": image_names,
            "object_names": object_names,
            "pixels": pixels_out,
            "masks": masks_out,
            "scales": scales_out,
            "labels": {n: arr.tolist() for n, arr in objects.items()},
            "expected": expected,
        }
        (OUT / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=True) + "\n"
        )

        rows, cols = (
            pixels_out[image_names[0]].__len__(),
            len(pixels_out[image_names[0]][0]),
        )
        image_arrays = {
            n: (img.pixel_data if isinstance(img, Image) else img)
            for n, img in images.items()
        }
        mask_arrays = {
            n: img.mask
            for n, img in images.items()
            if isinstance(img, Image) and img.has_mask
        }
        wants_images = settings["images_or_objects"] in (
            mc.M_IMAGES,
            mc.M_IMAGES_AND_OBJECTS,
        )
        wants_objects = settings["images_or_objects"] in (
            mc.M_OBJECTS,
            mc.M_IMAGES_AND_OBJECTS,
        )
        lines = [f"#[test]\nfn oracle_{name}() {{"]
        image_bindings = []
        for idx, (img_name, arr) in enumerate(image_arrays.items()):
            lines.append(f"let img_{idx}: Vec<f64> = {rust_vector(arr.ravel())};")
            image_bindings.append(f'("{img_name}", img_{idx}.as_slice())')
        lines.append(
            f"let images: Vec<(&str, &[f64])> = vec![{','.join(image_bindings)}];"
        )
        mask_expr = "None"
        if mask_arrays:
            mask_bindings = []
            for idx, (img_name, arr) in enumerate(mask_arrays.items()):
                lines.append(
                    f"let mask_{idx}: Vec<bool> = "
                    f"{rust_vector(arr.ravel(), lambda v: str(bool(v)).lower())};"
                )
                mask_bindings.append(f'("{img_name}", mask_{idx}.as_slice())')
            lines.append(
                f"let masks: Vec<(&str, &[bool])> = vec![{','.join(mask_bindings)}];"
            )
            mask_expr = "Some(masks.as_slice())"
        labels_expr = "None"
        if object_names:
            labels_arr = objects[object_names[0]]
            lines.append(
                f"let labels: Vec<u32> = {rust_vector(labels_arr.ravel(), str)};"
            )
            labels_expr = "Some(labels.as_slice())"
        fast_costes_variant = {
            mc.M_FASTER: "Faster",
            mc.M_FAST: "Fast",
            mc.M_ACCURATE: "Accurate",
        }[settings["fast_costes"]]
        costes_scale_max = max(scales_out.values()) if scales_out else 255.0
        lines.append(
            "let config = Config {"
            f" threshold_percent: {rust_float(settings['thr'])},"
            f" fast_costes: FastCostesMode::{fast_costes_variant},"
            f" do_corr_and_slope: {str(settings['do_corr_and_slope']).lower()},"
            f" do_manders: {str(settings['do_manders']).lower()},"
            f" do_rwc: {str(settings['do_rwc']).lower()},"
            f" do_overlap: {str(settings['do_overlap']).lower()},"
            f" do_costes: {str(settings['do_costes']).lower()},"
            f" costes_scale_max: {rust_float(costes_scale_max)},"
            f" wants_image_measurements: {str(wants_images).lower()},"
            f" wants_object_measurements: {str(wants_objects).lower()},"
            " };"
        )
        lines.append(
            f"let actual = measure(&images, {rows}, {cols}, {mask_expr}, "
            f"{labels_expr}, &config).unwrap();"
        )
        for column, values in expected["Image"].items():
            lines.append(
                f'assert_values(&[actual.image["{column}"]], '
                f'&{rust_vector(values)}, "{name}:Image:{column}");'
            )
        for obj_name in object_names:
            for column, values in expected[obj_name].items():
                lines.append(
                    f'assert_values(&actual.objects["{column}"], '
                    f'&{rust_vector(values)}, "{name}:{obj_name}:{column}");'
                )
        lines.append("}\n")
        tests.append("\n".join(lines))

        scalars_this_case = sum(
            len(v) for scope in expected.values() for v in scope.values()
        )
        total_scalars += scalars_this_case
        manifest.append(
            {
                "name": name,
                "image_names": image_names,
                "object_names": object_names,
                "settings": settings,
                "scalars": scalars_this_case,
            }
        )
        print(f"{name}: {scalars_this_case} scalars")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {**provenance, "cases": manifest, "total_scalars": total_scalars},
            indent=2,
        )
        + "\n"
    )
    (OUT / "oracle_tests.rs").write_text("\n".join(tests) + "\n")
    print(f"TOTAL scalar comparisons: {total_scalars}")


if __name__ == "__main__":
    generate()
