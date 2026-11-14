#  ty: ignore[import-not-found]
"""Run original CellProfiler 4.2.8's IdentifyPrimaryObjects against fixed
array fixtures, as the segmentation compatibility oracle.

This script runs with the separate environment installed by
reference/install.sh (same pinned CellProfiler source as
scripts/reference/generate.py). It never imports cp_anywhere or
substitutes another segmentation engine.

Declared scope matches crates/cp_anywhere_segmentation's module docs:
Global Otsu (two-class) or Manual threshold, no smoothing, no
declumping/watershed, no hole-filling, no border exclusion. Every case
uses CellProfiler's default ``Image(..., convert=True)`` image loading
(the realistic default: integer-dtype pixels divided by their dtype's
max value before thresholding; CellProfiler's own ``Threshold`` module
hard-bounds ``manual_threshold``/``threshold_range`` to [0, 1] and
assumes this). See spec/segmentation-modules.md.
"""

# The reference API is dynamically typed, and fixture literals define the test cases.
# ruff: noqa: ANN001, ANN201, ANN003, E402

import hashlib
import json
from pathlib import Path

import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler_core.image import Image, ImageSetList, VanillaImage
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
OUT_DIR = ROOT / "fixtures/segmentation"


def configure_module(
    module,
    *,
    threshold_method,
    threshold_value=None,
    threshold_correction_factor=1.0,
    exclude_size=False,
    min_diameter=0,
    max_diameter=1e9,
    exclude_border_objects=False,
    maximum_object_count=None,
    fill_holes="never",
    threshold_smoothing_scale=0.0,
    assign_middle_to_foreground=True,
    robust_background_lower_outlier_fraction=0.05,
    robust_background_upper_outlier_fraction=0.05,
    robust_background_averaging_method="mean",
    robust_background_variance_method="sd",
    robust_background_number_of_deviations=2.0,
    log_transform=False,
    threshold_scope="global",
    adaptive_window_size=50,
    threshold_range=(0.0, 1.0),
):
    """Set exactly the declared-scope settings (see module docstring)."""
    module.use_advanced.value = True
    module.threshold.threshold_scope.value = {
        "global": "Global",
        "adaptive": "Adaptive",
    }[threshold_scope]
    module.threshold.adaptive_window_size.value = adaptive_window_size
    module.threshold.threshold_range.value = threshold_range
    module.threshold.threshold_smoothing_scale.value = threshold_smoothing_scale
    module.threshold.threshold_correction_factor.value = threshold_correction_factor
    module.threshold.log_transform.value = log_transform
    if threshold_method == "otsu":
        module.threshold.global_operation.value = "Otsu"
        module.threshold.two_class_otsu.value = "Two classes"
    elif threshold_method == "otsu_three_class":
        module.threshold.global_operation.value = "Otsu"
        module.threshold.two_class_otsu.value = "Three classes"
        module.threshold.assign_middle_to_foreground.value = (
            "Foreground" if assign_middle_to_foreground else "Background"
        )
    elif threshold_method == "manual":
        module.threshold.global_operation.value = "Manual"
        module.threshold.manual_threshold.value = threshold_value
    elif threshold_method == "sauvola":
        assert threshold_scope == "adaptive"
        module.threshold.local_operation.value = "Sauvola"
    elif threshold_method == "li":
        module.threshold.global_operation.value = "Minimum Cross-Entropy"
    elif threshold_method == "robust_background":
        module.threshold.global_operation.value = "Robust Background"
        module.threshold.lower_outlier_fraction.value = (
            robust_background_lower_outlier_fraction
        )
        module.threshold.upper_outlier_fraction.value = (
            robust_background_upper_outlier_fraction
        )
        averaging_choices = {"mean": "Mean", "median": "Median", "mode": "Mode"}
        module.threshold.averaging_method.value = averaging_choices[
            robust_background_averaging_method
        ]
        variance_choices = {
            "sd": "Standard deviation",
            "mad": "Median absolute deviation",
        }
        module.threshold.variance_method.value = variance_choices[
            robust_background_variance_method
        ]
        module.threshold.number_of_deviations.value = (
            robust_background_number_of_deviations
        )
    else:
        raise ValueError(f"unsupported threshold_method: {threshold_method}")
    if threshold_scope == "adaptive" and threshold_method != "sauvola":
        module.threshold.local_operation.value = module.threshold.global_operation.value
    # A setting accepted by .value is not proof that the correct operation runs.
    expected_operation = {
        "otsu": "Otsu",
        "otsu_three_class": "Otsu",
        "manual": "Manual",
        "li": "Minimum Cross-Entropy",
        "robust_background": "Robust Background",
        "sauvola": "Sauvola",
    }[threshold_method]
    assert module.threshold.threshold_operation == expected_operation
    assert (
        module.threshold.threshold_scope.value
        == {"global": "Global", "adaptive": "Adaptive"}[threshold_scope]
    )
    assert module.threshold.adaptive_window_size.value == adaptive_window_size
    assert (
        module.threshold.threshold_range.min,
        module.threshold.threshold_range.max,
    ) == tuple(threshold_range)
    assert (
        module.threshold.threshold_correction_factor.value
        == threshold_correction_factor
    )
    assert module.threshold.threshold_smoothing_scale.value == threshold_smoothing_scale
    assert module.threshold.log_transform.value == log_transform
    if threshold_method in ("otsu", "otsu_three_class"):
        assert module.threshold.two_class_otsu.value == (
            "Two classes" if threshold_method == "otsu" else "Three classes"
        )
        if threshold_method == "otsu_three_class":
            assert module.threshold.assign_middle_to_foreground.value == (
                "Foreground" if assign_middle_to_foreground else "Background"
            )
    if threshold_method == "manual":
        assert module.threshold.manual_threshold.value == threshold_value
    if threshold_method == "robust_background":
        assert (
            module.threshold.lower_outlier_fraction.value
            == robust_background_lower_outlier_fraction
        )
        assert (
            module.threshold.upper_outlier_fraction.value
            == robust_background_upper_outlier_fraction
        )
        assert (
            module.threshold.averaging_method.value
            == averaging_choices[robust_background_averaging_method]
        )
        assert (
            module.threshold.variance_method.value
            == variance_choices[robust_background_variance_method]
        )
        assert (
            module.threshold.number_of_deviations.value
            == robust_background_number_of_deviations
        )
    module.unclump_method.value = "None"
    module.watershed_method.value = "None"
    module.fill_holes.value = {
        "never": "Never",
        "after_declump_only": "After declumping only",
        "after_thresholding_and_declump": "After both thresholding and declumping",
    }[fill_holes]
    module.exclude_border_objects.value = exclude_border_objects
    module.exclude_size.value = exclude_size
    module.size_range.min = min_diameter
    module.size_range.max = max_diameter
    if maximum_object_count is not None:
        module.limit_choice.value = "Erase"
        module.maximum_object_count.value = maximum_object_count
    else:
        module.limit_choice.value = "Continue"


def run_segmentation(pixels, mask=None, **kwargs):
    """Run IdentifyPrimaryObjects in the declared scope; return labels/threshold.

    Uses CellProfiler's default ``convert=True`` image loading (the
    realistic default -- see module docstring), not this project's
    measurement-API's ``convert=False`` convention.
    """
    module = IdentifyPrimaryObjects()
    module.x_name.value = "image"
    module.y_name.value = "objects"
    configure_module(module, **kwargs)

    image = Image(pixels, mask=mask, convert=True, dimensions=2)
    image_set_list = ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.providers.append(VanillaImage("image", image))
    object_set = ObjectSet()
    measurements = Measurements()
    pipeline = Pipeline()
    workspace = Workspace(pipeline, module, image_set, object_set, measurements, None)
    module.run(workspace)
    threshold_map = None
    guide_threshold = None
    if module.threshold.threshold_scope.value == "Adaptive":
        threshold_map, _, guide_threshold = module.threshold.get_threshold(
            image, workspace
        )

    objects = object_set.get_objects("objects")
    threshold = measurements.get_current_measurement(
        "Image", "Threshold_FinalThreshold_objects"
    )
    return (
        np.asarray(objects.segmented, dtype=np.int32),
        float(np.atleast_1d(threshold)[0]),
        np.asarray(objects.small_removed_segmented, dtype=np.int32),
        threshold_map,
        guide_threshold,
    )


def fixture_cases():
    """Yield (name, pixels, kwargs) for every declared-scope fixture.

    Every ``pixels`` array here is the RAW (not pre-rescaled) array, the
    same thing a caller would pass to ``cp_anywhere.identify_primary_objects``
    -- ``run_segmentation`` feeds it to CellProfiler with ``convert=True``,
    exactly matching that function's default ``rescale=True`` behavior.
    """
    rng = np.random.default_rng(0)
    bimodal = (
        np.concatenate([rng.integers(10, 30, size=50), rng.integers(180, 220, size=50)])
        .reshape(10, 10)
        .astype(np.uint8)
    )
    yield "synthetic_otsu_bimodal", bimodal, {"threshold_method": "otsu"}

    manual_image = np.zeros((8, 8), dtype=np.uint8)
    manual_image[2:6, 2:6] = 200
    yield (
        "synthetic_manual_square",
        manual_image,
        {"threshold_method": "manual", "threshold_value": 0.5},
    )

    size_image = np.zeros((10, 12), dtype=np.uint8)
    size_image[0, 0] = 200  # 1-pixel object: too small
    size_image[3:8, 3:8] = 200  # 5x5=25-pixel object: kept
    yield (
        "synthetic_size_filter",
        size_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "exclude_size": True,
            "min_diameter": 3,
            "max_diameter": 100,
        },
    )

    empty_image = np.zeros((6, 6), dtype=np.uint8)
    yield (
        "synthetic_empty",
        empty_image,
        {"threshold_method": "manual", "threshold_value": 0.5},
    )

    real_cc0 = np.load(ROOT / "fixtures/real/real_cell_cc0/DNA.npy")
    yield "real_cell_cc0_dna_otsu", real_cc0, {"threshold_method": "otsu"}

    real_bbbc_dna = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield "real_bbbc007_dna_otsu", real_bbbc_dna, {"threshold_method": "otsu"}

    real_bbbc_actin = np.load(ROOT / "fixtures/real/real_bbbc007/Actin.npy")
    yield "real_bbbc007_actin_otsu", real_bbbc_actin, {"threshold_method": "otsu"}

    real_bbbc_filtered = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_otsu_size_filtered",
        real_bbbc_filtered,
        {
            "threshold_method": "otsu",
            "exclude_size": True,
            "min_diameter": 15,
            "max_diameter": 60,
        },
    )

    border_image = np.zeros((10, 10), dtype=np.uint8)
    border_image[0:3, 0:3] = 200  # touches top-left corner (border)
    border_image[5:8, 5:8] = 200  # fully interior
    yield (
        "synthetic_border_excluded",
        border_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "exclude_border_objects": True,
        },
    )

    real_bbbc_border = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_otsu_border_excluded",
        real_bbbc_border,
        {"threshold_method": "otsu", "exclude_border_objects": True},
    )

    limit_image = np.zeros((10, 10), dtype=np.uint8)
    limit_image[1, 1] = 200
    limit_image[5, 5] = 200
    limit_image[8, 8] = 200
    yield (
        "synthetic_limit_erase_exceeded",
        limit_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "maximum_object_count": 2,
        },
    )
    yield (
        "synthetic_limit_erase_within_bounds",
        limit_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "maximum_object_count": 5,
        },
    )

    hole_image = np.full((7, 7), 200, dtype=np.uint8)
    hole_image[0, :] = 0
    hole_image[-1, :] = 0
    hole_image[:, 0] = 0
    hole_image[:, -1] = 0
    hole_image[3, 3] = 0  # single interior background pixel: a hole
    yield (
        "synthetic_fill_holes_after_declump_only",
        hole_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "fill_holes": "after_declump_only",
        },
    )
    yield (
        "synthetic_fill_holes_after_thresholding_and_declump",
        hole_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "fill_holes": "after_thresholding_and_declump",
            "max_diameter": 100,
        },
    )

    real_bbbc_fill_holes = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_otsu_fill_holes",
        real_bbbc_fill_holes,
        {
            "threshold_method": "otsu",
            "fill_holes": "after_thresholding_and_declump",
            "max_diameter": 100,
        },
    )

    rng2 = np.random.default_rng(1)
    smoothing_image = (rng2.random((20, 20)) * 255).astype(np.uint8)
    yield (
        "synthetic_threshold_smoothing_manual",
        smoothing_image,
        {
            "threshold_method": "manual",
            "threshold_value": 0.5,
            "threshold_smoothing_scale": 2.0,
        },
    )

    real_bbbc_smoothing = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_otsu_threshold_smoothing",
        real_bbbc_smoothing,
        {"threshold_method": "otsu", "threshold_smoothing_scale": 1.5},
    )

    rng3 = np.random.default_rng(7)
    three_class_image = (
        np.concatenate(
            [
                rng3.normal(40, 8, 1000),
                rng3.normal(128, 8, 1000),
                rng3.normal(220, 8, 1000),
            ]
        )
        .clip(0, 255)
        .astype(np.uint8)
        .reshape(50, 60)
    )
    yield (
        "synthetic_otsu_three_class_foreground",
        three_class_image,
        {"threshold_method": "otsu_three_class", "assign_middle_to_foreground": True},
    )
    yield (
        "synthetic_otsu_three_class_background",
        three_class_image,
        {"threshold_method": "otsu_three_class", "assign_middle_to_foreground": False},
    )

    real_bbbc_three_class = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_otsu_three_class",
        real_bbbc_three_class,
        {"threshold_method": "otsu_three_class", "assign_middle_to_foreground": True},
    )

    rng4 = np.random.default_rng(3)
    li_image = (
        np.concatenate([rng4.normal(50, 12, 400), rng4.normal(180, 12, 400)])
        .clip(0, 255)
        .astype(np.uint8)
        .reshape(20, 40)
    )
    yield ("synthetic_li", li_image, {"threshold_method": "li"})

    real_bbbc_li = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield ("real_bbbc007_dna_li", real_bbbc_li, {"threshold_method": "li"})

    rng5 = np.random.default_rng(5)
    robust_image = (rng5.random((10, 8)) * 255).astype(np.uint8)
    yield (
        "synthetic_robust_background_mean_sd",
        robust_image,
        {"threshold_method": "robust_background"},
    )
    yield (
        "synthetic_robust_background_median_mad",
        robust_image,
        {
            "threshold_method": "robust_background",
            "robust_background_averaging_method": "median",
            "robust_background_variance_method": "mad",
        },
    )
    yield (
        "synthetic_robust_background_mode_sd",
        robust_image,
        {
            "threshold_method": "robust_background",
            "robust_background_averaging_method": "mode",
        },
    )
    yield (
        "synthetic_robust_background_custom_outliers",
        robust_image,
        {
            "threshold_method": "robust_background",
            "robust_background_lower_outlier_fraction": 0.1,
            "robust_background_upper_outlier_fraction": 0.1,
            "robust_background_number_of_deviations": 1.0,
        },
    )

    real_bbbc_robust = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_robust_background",
        real_bbbc_robust,
        {"threshold_method": "robust_background"},
    )

    rng6 = np.random.default_rng(11)
    log_image = (rng6.random((10, 10)) * 255).astype(np.uint8)
    yield (
        "synthetic_log_transform_otsu",
        log_image,
        {"threshold_method": "otsu", "log_transform": True},
    )
    yield (
        "synthetic_log_transform_li",
        log_image,
        {"threshold_method": "li", "log_transform": True},
    )
    yield (
        "synthetic_log_transform_otsu_three_class",
        log_image,
        {"threshold_method": "otsu_three_class", "log_transform": True},
    )

    sauvola_rng = np.random.default_rng(5028)
    sauvola_pixels = sauvola_rng.random((19, 27))
    sauvola_mask = np.ones(sauvola_pixels.shape, dtype=bool)
    sauvola_mask[:4, :7] = False
    sauvola_mask[8:12, 9:13] = False
    for window in (1, 7, 8, 30):
        yield (
            f"synthetic_adaptive_sauvola_window_{window}",
            sauvola_pixels,
            {
                "threshold_method": "sauvola",
                "threshold_scope": "adaptive",
                "adaptive_window_size": window,
                "mask": sauvola_mask,
            },
        )
    yield (
        "real_bbbc007_dna_adaptive_sauvola",
        real_bbbc_dna,
        {
            "threshold_method": "sauvola",
            "threshold_scope": "adaptive",
            "adaptive_window_size": 50,
        },
    )
    for method in ("otsu", "otsu_three_class", "li", "robust_background", "sauvola"):
        for edge, pixels, mask in (
            ("zero", np.zeros((9, 11)), sauvola_mask[:9, :11]),
            ("constant", np.full((9, 11), 0.375), None),
            ("empty_mask", np.full((9, 11), 0.8), np.zeros((9, 11), dtype=bool)),
            ("repeated_row", np.tile(np.linspace(0, 1, 11), (9, 1)), None),
        ):
            yield (
                f"synthetic_adaptive_{method}_{edge}",
                pixels,
                {
                    "threshold_method": method,
                    "threshold_scope": "adaptive",
                    "adaptive_window_size": 50,
                    "mask": mask,
                },
            )
    for method, log in (
        ("otsu", False),
        ("otsu", True),
        ("li", True),
        ("otsu_three_class", True),
        ("robust_background", False),
        ("sauvola", False),
    ):
        for bounds, factor in (
            ((0.2, 0.35), 1.3),
            ((0.65, 0.7), 0.5),
            ((0.42, 0.42), 2.0),
        ):
            yield (
                f"synthetic_adaptive_{method}_log_{log}_range_{bounds[0]}",
                sauvola_pixels,
                {
                    "threshold_method": method,
                    "threshold_scope": "adaptive",
                    "adaptive_window_size": 4,
                    "threshold_range": bounds,
                    "threshold_correction_factor": factor,
                    "log_transform": log,
                    "mask": sauvola_mask,
                    "threshold_smoothing_scale": 1.5,
                },
            )

    li_rng = np.random.default_rng(4928)
    local_li = li_rng.choice(
        np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]), (30, 42)
    )
    local_li[0, 0] = 0.100001
    local_mask = np.ones(local_li.shape, dtype=bool)
    local_mask[:6, :6] = (
        False  # empty block; excluded close gap still sets local Li tolerance
    )
    local_mask[11:16, 17:23] = False
    for log in (False, True):
        yield (
            f"synthetic_adaptive_li_masked_log_{log}",
            local_li,
            {
                "threshold_method": "li",
                "threshold_scope": "adaptive",
                "adaptive_window_size": 6,
                "mask": local_mask,
                "log_transform": log,
            },
        )
    for foreground in (False, True):
        multi = li_rng.random((30, 42))
        multi[:6, :6] = np.tile([0.1, 0.8], (6, 3))
        multi[:6, 6:12] = np.tile([0.0, 0.006, 1.0], (6, 2))
        yield (
            f"synthetic_adaptive_three_class_foreground_{foreground}",
            multi,
            {
                "threshold_method": "otsu_three_class",
                "threshold_scope": "adaptive",
                "adaptive_window_size": 6,
                "assign_middle_to_foreground": foreground,
            },
        )
    for averaging, variance in (("mean", "sd"), ("median", "mad"), ("mode", "sd")):
        yield (
            f"synthetic_adaptive_robust_{averaging}_{variance}",
            local_li,
            {
                "threshold_method": "robust_background",
                "threshold_scope": "adaptive",
                "adaptive_window_size": 6,
                "mask": local_mask,
                "robust_background_averaging_method": averaging,
                "robust_background_variance_method": variance,
                "robust_background_lower_outlier_fraction": 0.125,
                "robust_background_upper_outlier_fraction": 0.125,
                "robust_background_number_of_deviations": 0.75,
            },
        )

    adaptive_rng = np.random.default_rng(4828)
    for window in (7, 10, 14):
        pixels = adaptive_rng.random((31, 43)) * np.linspace(0.2, 1.0, 43)
        yield (
            f"synthetic_adaptive_otsu_window_{window}",
            pixels,
            {
                "threshold_method": "otsu",
                "threshold_scope": "adaptive",
                "adaptive_window_size": window,
            },
        )
    yield (
        "real_bbbc007_dna_adaptive_otsu",
        real_bbbc_dna,
        {
            "threshold_method": "otsu",
            "threshold_scope": "adaptive",
            "adaptive_window_size": 50,
        },
    )

    for method in ("li", "otsu_three_class", "robust_background"):
        yield (
            f"synthetic_global_constant_{method}",
            np.full((8, 10), 0.375, dtype=np.float64),
            {"threshold_method": method},
        )
        yield (
            f"synthetic_global_empty_mask_{method}",
            np.zeros((8, 10), dtype=np.float64),
            {"threshold_method": method, "mask": np.zeros((8, 10), dtype=bool)},
        )

    real_bbbc_log = np.load(ROOT / "fixtures/real/real_bbbc007/DNA.npy")
    yield (
        "real_bbbc007_dna_log_transform_otsu",
        real_bbbc_log,
        {"threshold_method": "otsu", "log_transform": True},
    )


def npy_bytes(array):
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    manifest = {
        "source_commit": SOURCE_COMMIT,
        "cellprofiler_version": "4.2.8",
        "cases": [],
    }
    files = {}
    for name, pixels, kwargs in fixture_cases():
        labels, threshold, small_removed, threshold_map, guide = run_segmentation(
            pixels, **kwargs
        )
        settings = {key: value for key, value in kwargs.items() if key != "mask"}
        folder = f"fixtures/segmentation/{name}"
        case_files = {
            f"{folder}/pixels.npy": npy_bytes(pixels),
            f"{folder}/labels.npy": npy_bytes(labels),
            f"{folder}/small_removed_labels.npy": npy_bytes(small_removed),
        }
        if threshold_map is not None:
            case_files[f"{folder}/threshold_map.npy"] = npy_bytes(threshold_map)
            assert np.isclose(np.mean(threshold_map), threshold, atol=1e-12)
        if kwargs.get("mask") is not None:
            case_files[f"{folder}/mask.npy"] = npy_bytes(kwargs["mask"])
        files.update(case_files)
        result = {
            "image_id": name,
            "settings": settings,
            "rescale": True,
            "threshold": threshold,
            "object_count": int(labels.max()),
            "shape": list(pixels.shape),
            "sha256": {
                path: hashlib.sha256(payload).hexdigest()
                for path, payload in case_files.items()
            },
        }
        if guide is not None:
            result["guide_threshold"] = float(guide)
        result_path = f"fixtures/segmentation/{name}.json"
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        files[result_path] = payload
        manifest["cases"].append(
            {
                "name": name,
                "result": result_path,
                "threshold": threshold,
                "object_count": result["object_count"],
            }
        )

    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    files["fixtures/segmentation/manifest.json"] = manifest_payload

    for relative_path, payload in files.items():
        path = ROOT / relative_path
        if args.verify:
            if not path.exists():
                raise SystemExit(f"missing reference replay file: {relative_path}")
            assert path.read_bytes() == payload, (
                f"Reference replay mismatch: {relative_path}"
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    print(
        json.dumps(
            {
                "mode": "verify" if args.verify else "write",
                "files": len(files),
                "cases": len(manifest["cases"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
