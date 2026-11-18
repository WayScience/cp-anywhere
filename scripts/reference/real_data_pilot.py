# ty: ignore[import-not-found]
"""Run a disjoint calibration/held-out pilot with pinned, real CellProfiler.

Use the Python 3.9 reference environment, with PYTHONPATH pointing at the pinned
CellProfiler source checkout. No cp_anywhere code or replacement algorithm is
imported. The two dumped pipelines are audit records of the actual configured
modules; this CLI supplies arrays and orchestrates the disjoint execution phases.
"""

# The reference API is dynamic and intentionally absent from the project venv.
# ruff: noqa: ANN401, E402

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import platform
import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cellprofiler
import cellprofiler_core.preferences
import numpy as np

cellprofiler_core.preferences.set_headless()

from cellprofiler.modules.correctilluminationapply import CorrectIlluminationApply
from cellprofiler.modules.correctilluminationcalculate import (
    CorrectIlluminationCalculate,
    CorrectIlluminationImageProvider,
)
from cellprofiler.modules.identifyprimaryobjects import IdentifyPrimaryObjects
from cellprofiler.modules.identifysecondaryobjects import IdentifySecondaryObjects
from cellprofiler.modules.identifytertiaryobjects import IdentifyTertiaryObjects
from cellprofiler.modules.measureimageintensity import MeasureImageIntensity
from cellprofiler.modules.measureobjectintensity import MeasureObjectIntensity
from cellprofiler.modules.measureobjectneighbors import (
    D_ADJACENT,
    MeasureObjectNeighbors,
)
from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace

SOURCE_COMMIT = "2df090ff281cff80943fe679a1c3192fbda9cd6d"
CHANNELS = ("DNA", "Actin")
DIMENSIONS = 2
SCOPES = ("Nuclei", "Cells", "Cytoplasm")
# cp_anywhere rejects size/shape on sparse tertiary IDs (empty cytoplasm
# regions), so the shape modules stay on the supported scopes. Cytoplasm
# intensity is still measured on both sides. See the pilot's limitations.
SHAPE_SCOPES = ("Nuclei", "Cells")
PREFIXES = ("Intensity_", "Location_", "AreaShape_", "Neighbors_")
# Identification-layer features produced by CellProfiler's object
# identification machinery rather than by any measurement module.
IDENTIFICATION_LAYER_FEATURES = (
    "Location_Center_X",
    "Location_Center_Y",
    "Location_Center_Z",
)
ILLUMINATION_CHOICES = {
    "intensity_choice": ("Regular",),
    "smoothing_method": ("Gaussian Filter", "Fit Polynomial"),
    "automatic_object_width": ("Manually", "Automatic"),
    "rescale_option": ("Median", "No"),
}
DEFAULTS = {
    "illumination": {
        "intensity_choice": "Regular",
        "smoothing_method": "Gaussian Filter",
        "automatic_object_width": "Manually",
        "size_of_smoothing_filter": 128.0,
        "rescale_option": "Median",
    },
    "primary": {
        "threshold": "otsu",
        "min_diameter": 10,
        "max_diameter": 40,
        "exclude_size": True,
        "unclump_method": "shape",
        "watershed_method": "propagate",
    },
    "secondary": {
        "method": "propagation",
        "regularization_factor": 0.05,
        "fill_holes": True,
    },
    "tertiary": {"shrink_primary": False},
}


def sha256(path: Path) -> str:
    """Hash actual file bytes, including source and normalized input arrays."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Keep real NaN and infinity measurements in this internal comparison data."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def validate_setting(section: str, key: str, value: Any, default: Any) -> None:
    """Check one supported setting's value without coercing invalid input."""
    choices = (
        ILLUMINATION_CHOICES.get(key, (default,))
        if section == "illumination"
        else (default,)
    )
    if isinstance(default, str) and value not in choices:
        raise ValueError(f"unsupported {section}.{key}: {value!r}")
    if isinstance(default, bool) and not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be a boolean")
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be a number")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{section}.{key} must be finite and nonnegative")


def configure_settings(supplied: dict[str, Any]) -> dict[str, Any]:
    """Reject unimplemented settings instead of silently running another method."""
    missing = set(DEFAULTS) - set(supplied)
    extra = set(supplied) - set(DEFAULTS) - {"shape_scopes"}
    if missing or extra:
        raise ValueError(
            "settings must contain illumination, primary, secondary, tertiary"
        )
    shape_scopes = supplied.get("shape_scopes", list(SHAPE_SCOPES))
    if (
        not isinstance(shape_scopes, list)
        or not shape_scopes
        or len(set(shape_scopes)) != len(shape_scopes)
        or set(shape_scopes) - set(SCOPES)
    ):
        raise ValueError(
            f"shape_scopes must be unique scopes from {SCOPES}, got {shape_scopes!r}"
        )
    settings: dict[str, Any] = {"shape_scopes": list(shape_scopes)}
    for section, defaults in DEFAULTS.items():
        if set(supplied[section]) != set(defaults):
            raise ValueError(f"unexpected or missing settings in {section}")
        settings[section] = dict(supplied[section])
        for key, default in defaults.items():
            validate_setting(section, key, settings[section][key], default)
    width = settings["illumination"]["size_of_smoothing_filter"]
    if width < 1 or int(width) != width:
        raise ValueError("size_of_smoothing_filter must be a positive integer")
    primary = settings["primary"]
    if not 0 < primary["min_diameter"] <= primary["max_diameter"]:
        raise ValueError("primary diameter range must be positive and ordered")
    return settings


def group_illumination(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge an optional stress-group override over global illumination settings."""
    if set(overrides) - set(base):
        raise ValueError("unexpected settings in group illumination override")
    merged = {**base, **overrides}
    for key, default in DEFAULTS["illumination"].items():
        validate_setting("illumination", key, merged[key], default)
    width = merged["size_of_smoothing_filter"]
    if width < 1 or int(width) != width:
        raise ValueError("size_of_smoothing_filter must be a positive integer")
    return merged


def slug(value: str) -> str:
    """Constrain manifest IDs to a single unambiguous output path component."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None
    ):
        raise ValueError(f"invalid output slug: {value!r}")
    return value


def validate_input(relative_path: str, root: Path) -> tuple[Path, tuple[int, ...]]:
    """Check a normalized array without modifying or normalizing its pixels."""
    relative = Path(relative_path)
    if relative.is_absolute() or relative.suffix != ".npy":
        raise ValueError("input channel paths must be relative .npy paths")
    path = (root / relative).resolve(strict=True)
    pixels = np.load(path, allow_pickle=False, mmap_mode="r")
    if pixels.ndim != DIMENSIONS or 0 in pixels.shape:
        raise ValueError(f"{path}: expected nonempty 2D shape")
    if pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError(f"{path}: expected finite normalized float32 pixels")
    if pixels.min() < 0 or pixels.max() > 1:
        raise ValueError(f"{path}: normalized pixels must lie in [0, 1]")
    return path, pixels.shape


def field_slug(value: str, ids: set[str], group_id: str) -> str:
    """Validate a field ID and avoid overwriting its group's illumination file."""
    field_id = slug(value)
    if field_id in ids:
        raise ValueError(f"{group_id}: field IDs must be unique and disjoint")
    if field_id == "illumination":
        raise ValueError("field ID 'illumination' is reserved")
    ids.add(field_id)
    return field_id


def validate_groups(groups: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Preflight shapes, dtypes, IDs, and disjoint paths before writing output."""
    if not groups:
        raise ValueError("at least one group is required")
    group_ids = set()
    records = []
    for group in groups:
        # Split isolation is per fold: leave-one-field-out necessarily reuses a
        # field for calibration in other independently accumulated folds.
        paths_by_phase: dict[str, set[Path]] = {
            "calibration": set(),
            "validation": set(),
        }
        group_id = slug(group["id"])
        if group_id in group_ids:
            raise ValueError(f"duplicate group ID: {group_id}")
        group_ids.add(group_id)
        ids = set()
        shape = None
        record: dict[str, Any] = {
            "id": group_id,
            "illumination": group.get("illumination", {}),
        }
        for phase in ("calibration", "validation"):
            if not group[phase]:
                raise ValueError(f"{group_id}: {phase} must not be empty")
            record[phase] = []
            for field in group[phase]:
                field_id = field_slug(field["id"], ids, group_id)
                field_record: dict[str, Any] = {"id": field_id}
                for channel in CHANNELS:
                    path, field_shape = validate_input(field[channel], root)
                    paths_by_phase[phase].add(path)
                    shape = shape or field_shape
                    if field_shape != shape:
                        raise ValueError(
                            f"{group_id}: incompatible shape {field_shape}, {shape}"
                        )
                    field_record[channel] = {
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                record[phase].append(field_record)
        assert shape is not None
        record["shape"] = list(shape)
        records.append(record)
        if paths_by_phase["calibration"] & paths_by_phase["validation"]:
            raise ValueError("calibration and validation input paths must be disjoint")
    return records


def set_threshold(module: Any) -> None:
    """Use the native pilot's explicit global, two-class Otsu defaults."""
    threshold = module.threshold
    threshold.threshold_scope.value = "Global"
    threshold.global_operation.value = "Otsu"
    threshold.two_class_otsu.value = "Two classes"
    threshold.threshold_smoothing_scale.value = 0.0
    threshold.threshold_correction_factor.value = 1.0
    threshold.threshold_range.value = (0.0, 1.0)
    threshold.log_transform.value = False
    assert threshold.threshold_operation == "Otsu"


def calibration_pipeline(settings: dict[str, Any]) -> Any:
    """Create the exact modules used by the accumulation providers."""
    pipeline = Pipeline()
    for index, channel in enumerate(CHANNELS, 1):
        module = CorrectIlluminationCalculate()
        module.set_module_num(index)
        module.image_name.value = f"{channel}_raw"
        module.illumination_image_name.value = f"{channel}_illumination"
        module.intensity_choice.value = "Regular"
        module.dilate_objects.value = False
        module.each_or_all.value = "All: Across cycles"
        for key, value in settings.items():
            getattr(module, key).value = (
                int(value) if key == "size_of_smoothing_filter" else value
            )
        pipeline.add_module(module)
    return pipeline


# One assignment per real CP setting keeps the configuration auditable.
def consumer_pipeline(settings: dict[str, Any]) -> Any:  # noqa: PLR0915
    """Build genuine producer-to-consumer modules, preserving CP object layers."""
    apply = CorrectIlluminationApply()
    del apply.images[:]
    for channel in CHANNELS:
        apply.add_image(can_delete=False)
        image = apply.images[-1]
        image.image_name.value = f"{channel}_raw"
        image.corrected_image_name.value = channel
        image.illum_correct_function_image_name.value = f"{channel}_illumination"
        image.divide_or_subtract.value = "Divide"
    apply.truncate_low.value = True
    apply.truncate_high.value = True

    primary = IdentifyPrimaryObjects()
    primary.x_name.value = "DNA"
    primary.y_name.value = "Nuclei"
    primary.use_advanced.value = True
    primary.size_range.min = settings["primary"]["min_diameter"]
    primary.size_range.max = settings["primary"]["max_diameter"]
    primary.exclude_size.value = settings["primary"]["exclude_size"]
    primary.exclude_border_objects.value = False
    primary.unclump_method.value = "Shape"
    primary.watershed_method.value = "Propagate"
    primary.automatic_smoothing.value = True
    primary.automatic_suppression.value = True
    primary.low_res_maxima.value = False
    primary.fill_holes.value = "Never"
    primary.limit_choice.value = "Continue"
    set_threshold(primary)

    secondary = IdentifySecondaryObjects()
    secondary.x_name.value = "Nuclei"
    secondary.y_name.value = "Cells"
    secondary.image_name.value = "Actin"
    secondary.method.value = "Propagation"
    secondary.regularization_factor.value = settings["secondary"][
        "regularization_factor"
    ]
    secondary.fill_holes.value = settings["secondary"]["fill_holes"]
    secondary.wants_discard_edge.value = False
    secondary.wants_discard_primary.value = False
    set_threshold(secondary)

    tertiary = IdentifyTertiaryObjects()
    tertiary.secondary_objects_name.value = "Cells"
    tertiary.primary_objects_name.value = "Nuclei"
    tertiary.subregion_objects_name.value = "Cytoplasm"
    tertiary.shrink_primary.value = settings["tertiary"]["shrink_primary"]

    image_intensity = MeasureImageIntensity()
    image_intensity.images_list.value = ", ".join(CHANNELS)
    image_intensity.wants_objects.value = False
    image_intensity.wants_percentiles.value = True
    image_intensity.percentiles.value = "10, 50, 90"
    object_intensity = MeasureObjectIntensity()
    object_intensity.images_list.value = ", ".join(CHANNELS)
    object_intensity.objects_list.value = ", ".join(SCOPES)
    size_shape = MeasureObjectSizeShape()
    size_shape.objects_list.value = ", ".join(settings["shape_scopes"])
    size_shape.calculate_zernikes.value = True
    size_shape.calculate_advanced.value = False
    neighbors = MeasureObjectNeighbors()
    neighbors.object_name.value = "Cells"
    neighbors.neighbors_name.value = "Cells"
    neighbors.distance_method.value = D_ADJACENT
    neighbors.wants_excluded_objects.value = False

    pipeline = Pipeline()
    for index, module in enumerate(
        (
            apply,
            primary,
            secondary,
            tertiary,
            image_intensity,
            object_intensity,
            size_shape,
            neighbors,
        ),
        1,
    ):
        module.set_module_num(index)
        pipeline.add_module(module)
    return pipeline


def dump_pipeline(pipeline: Any, path: Path) -> None:
    """Serialize actual settings and verify CellProfiler can reload them."""
    stream = io.StringIO()
    pipeline.dump(stream)
    payload = stream.getvalue()
    loaded = Pipeline()
    loaded.load(io.StringIO(payload))
    original = pipeline.modules()
    restored = loaded.modules()
    if len(original) != len(restored):
        raise RuntimeError("pipeline round trip lost modules")
    for before, after in zip(original, restored):
        if before.module_name != after.module_name or [
            setting.unicode_value for setting in before.settings()
        ] != [setting.unicode_value for setting in after.settings()]:
            raise RuntimeError(f"pipeline round trip changed {before.module_name}")
    path.write_text(payload)


def provenance(pipelines: tuple[Any, Any]) -> dict[str, Any]:
    """Fail closed on a wrong/modified source pin; record actual dependencies."""
    assert cellprofiler.__file__ is not None
    source_root = Path(cellprofiler.__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if commit != SOURCE_COMMIT or cellprofiler.__version__ != "4.2.8":
        raise RuntimeError(
            f"expected CellProfiler 4.2.8 @ {SOURCE_COMMIT}, got {commit}"
        )
    hashes = {}
    for pipeline in pipelines:
        for module in pipeline.modules():
            source = Path(inspect.getfile(type(module))).resolve()
            relative = source.relative_to(source_root).as_posix()
            pinned_bytes = subprocess.check_output(
                ["git", "-C", str(source_root), "show", f"{SOURCE_COMMIT}:{relative}"],
                timeout=10,
            )
            digest = sha256(source)
            if hashlib.sha256(pinned_bytes).hexdigest() != digest:
                raise RuntimeError(f"modified reference source: {source}")
            hashes[relative] = digest
    packages = (
        "cellprofiler-core",
        "centrosome",
        "numpy",
        "scipy",
        "scikit-image",
        "h5py",
        "mahotas",
    )
    return {
        "source_commit": commit,
        "source_root": str(source_root),
        "module_sha256": hashes,
        "versions": {
            "python": platform.python_version(),
            "cellprofiler": cellprofiler.__version__,
            **{package: version(package) for package in packages},
        },
        "execution": {
            "calibration": "Per-group/channel add_image then provide_image(None)",
            "validation": "Sequential real Module.run on shared Workspace state",
            "image_loading": "Input float32 pixels, Image(convert=False, dimensions=2)",
            "illumination_apply": "Divide, truncate_low=True, truncate_high=True",
            "thresholds": "Global two-class Otsu, smoothing=0, factor=1, range=[0,1]",
            "primary": (
                "Advanced; no hole filling or border/count exclusion; automatic maxima"
            ),
            "measurements": "Image percentiles 10/50/90; Zernike on, advanced off",
            "relationships": "Omitted; no inferred or invented topology",
            "grouping": "Acquisition membership is supplied by the input manifest",
        },
        "pipelines": {
            "calibration": "illumination.cppipe",
            "validation": "pipeline.cppipe",
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def calculate_illumination(group: dict[str, Any], pipeline: Any) -> dict[str, Any]:
    """Let CP accumulate only calibration fields; never include held-out images."""
    result = {}
    for channel, module in zip(CHANNELS, pipeline.modules()):
        provider = CorrectIlluminationImageProvider(f"{channel}_illumination", module)
        for field in group["calibration"]:
            pixels = np.load(field[channel]["path"], allow_pickle=False)
            provider.add_image(Image(pixels, convert=False, dimensions=2))
        result[channel] = provider.provide_image(None)
    return result


def measurements_on_shared_inputs(
    pipeline: Any, images: Any, objects: Any
) -> dict[str, Any]:
    """Run real CP measurement modules on float64 copies of the saved pixels.

    cp_anywhere's measurement API promotes inputs to float64. This separate
    oracle pass uses the same dtype, without changing the end-to-end pipeline
    measurements, which used CP's original float32 corrected images.
    """
    image_sets = ImageSetList()
    shared_images = image_sets.get_image_set(0)
    for channel in CHANNELS:
        shared_images.add(
            channel,
            Image(
                images.get_image(channel).pixel_data.astype(np.float64), convert=False
            ),
        )
    measurements = Measurements()
    try:
        for module in pipeline.modules():
            if module.module_name.startswith("Measure"):
                module.run(
                    Workspace(
                        pipeline,
                        module,
                        shared_images,
                        objects,
                        measurements,
                        image_sets,
                    )
                )
        return {
            scope: {
                name: np.asarray(
                    measurements.get_current_measurement(scope, name)
                ).tolist()
                for name in sorted(measurements.get_feature_names(scope))
                if name.startswith(PREFIXES)
            }
            for scope in ("Image", *SCOPES)
        }
    finally:
        measurements.close()


def run_field(
    field: dict[str, Any],
    illumination: dict[str, Any],
    pipeline: Any,
    destination: Path,
) -> dict[str, Any]:
    """Save actual corrected images, all requested label layers and CP features."""
    image_sets = ImageSetList()
    image_set = image_sets.get_image_set(0)
    for channel in CHANNELS:
        pixels = np.load(field[channel]["path"], allow_pickle=False)
        image_set.add(f"{channel}_raw", Image(pixels, convert=False, dimensions=2))
        image_set.add(f"{channel}_illumination", illumination[channel])
    objects = ObjectSet()
    measurements = Measurements()
    cells_small_removed = None
    identification_layer: dict[str, dict[str, Any]] = {}
    try:
        for module in pipeline.modules():
            workspace = Workspace(
                pipeline, module, image_set, objects, measurements, image_sets
            )
            module.run(workspace)
            if isinstance(module, IdentifySecondaryObjects):
                # MeasureObjectNeighbors later replaces the Cells Objects
                # with a fresh instance whose small_removed_segmented falls
                # back to segmented; capture the true secondary diagnostic
                # array before that happens.
                cells_small_removed = objects.get_objects(
                    "Cells"
                ).small_removed_segmented.copy()
        arrays = {
            f"{channel}_corrected": image_set.get_image(channel).pixel_data
            for channel in CHANNELS
        }
        arrays.update(
            {
                f"{scope}_labels": objects.get_objects(scope).segmented
                for scope in SCOPES
            }
        )
        nuclei = objects.get_objects("Nuclei")
        arrays["Nuclei_unedited"] = nuclei.unedited_segmented
        arrays["Nuclei_small_removed"] = nuclei.small_removed_segmented
        if cells_small_removed is None:
            raise RuntimeError("pipeline produced no secondary object set")
        arrays["Cells_small_removed"] = cells_small_removed
        features = {}
        object_ids = {}
        length_mismatches: dict[str, dict[str, dict[str, int]]] = {}
        for scope in ("Image", *SCOPES):
            features[scope] = {}
            if scope != "Image":
                object_ids[scope] = list(
                    range(1, int(arrays[f"{scope}_labels"].max()) + 1)
                )
            names = set(measurements.get_feature_names(scope))
            # Preserve identification-only measurements outside the bounded
            # module-feature comparison, including their actual oracle values.
            identification_layer[scope] = {
                name: np.asarray(
                    measurements.get_current_measurement(scope, name)
                ).tolist()
                for name in sorted(names & set(IDENTIFICATION_LAYER_FEATURES))
            }
            for feature in sorted(names - set(IDENTIFICATION_LAYER_FEATURES)):
                if feature.startswith(PREFIXES):
                    value = measurements.get_current_measurement(scope, feature)
                    features[scope][feature] = np.asarray(value).tolist()
                    if scope != "Image" and len(features[scope][feature]) != len(
                        object_ids[scope]
                    ):
                        # CP's regionprops-backed size/shape features can omit
                        # missing IDs in sparse tertiary labels. Preserve those
                        # genuine vectors, not padded or relabeled replacements.
                        length_mismatches.setdefault(scope, {})[feature] = {
                            "measurement_values": len(features[scope][feature]),
                            "object_ids": len(object_ids[scope]),
                        }
        np.savez_compressed(destination.with_suffix(".npz"), **arrays)
        write_json(
            destination.with_suffix(".json"),
            {
                "features": features,
                "features_on_shared_inputs": measurements_on_shared_inputs(
                    pipeline, image_set, objects
                ),
                "object_ids": object_ids,
                "measurement_length_mismatches": length_mismatches,
                "identification_layer_features": {
                    "status": "not_compared",
                    "reason": (
                        "CellProfiler's identification layer emits these; "
                        "cp_anywhere has no identification-module layer"
                    ),
                    "features": identification_layer,
                },
                "relationships": {
                    "status": "omitted",
                    "reason": "No relationship rows exported",
                },
            },
        )
        return {
            "id": field["id"],
            "object_counts": {scope: len(ids) for scope, ids in object_ids.items()},
        }
    finally:
        measurements.close()


def run(input_path: Path, output: Path) -> dict[str, Any]:
    """Validate the complete manifest before producing a new output directory."""
    input_path = input_path.resolve(strict=True)
    manifest = json.loads(input_path.read_text())
    settings = configure_settings(manifest["settings"])
    groups = validate_groups(manifest["groups"], input_path.parent)
    for group in groups:
        group["illumination"] = group_illumination(
            settings["illumination"], group["illumination"]
        )
    calibration = calibration_pipeline(settings["illumination"])
    consumers = consumer_pipeline(settings)
    source = provenance((calibration, consumers))
    source["pipelines"]["calibration_by_group"] = {
        group["id"]: f"{group['id']}/illumination.cppipe" for group in groups
    }
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dump_pipeline(calibration, output / "illumination.cppipe")
    dump_pipeline(consumers, output / "pipeline.cppipe")
    write_json(
        output / "provenance.json",
        {
            **source,
            "settings": settings,
            "input_manifest": str(input_path),
            "input_manifest_sha256": sha256(input_path),
            "groups": groups,
        },
    )
    outputs = []
    for group in groups:
        directory = output / group["id"]
        directory.mkdir()
        group_calibration = calibration_pipeline(group["illumination"])
        dump_pipeline(group_calibration, directory / "illumination.cppipe")
        illumination = calculate_illumination(group, group_calibration)
        np.savez_compressed(
            directory / "illumination.npz",
            **{channel: image.pixel_data for channel, image in illumination.items()},
        )
        fields = []
        for field in group["validation"]:
            entry = run_field(field, illumination, consumers, directory / field["id"])
            entry.update(
                {
                    "arrays": f"{group['id']}/{field['id']}.npz",
                    "measurements": f"{group['id']}/{field['id']}.json",
                }
            )
            fields.append(entry)
            print(f"{group['id']}/{field['id']}: {entry['object_counts']}", flush=True)
        outputs.append(
            {
                "id": group["id"],
                "illumination": f"{group['id']}/illumination.npz",
                "calibration_pipeline": f"{group['id']}/illumination.cppipe",
                "illumination_settings": group["illumination"],
                "validation": fields,
            }
        )
    summary = {
        "groups": outputs,
        "provenance": "provenance.json",
        "pipeline": "pipeline.cppipe",
        "calibration_pipeline": "illumination.cppipe",
    }
    write_json(output / "manifest.json", summary)
    return summary


def main() -> None:
    """CLI entry point; paths inside input JSON resolve from its parent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output.resolve())


if __name__ == "__main__":
    main()
