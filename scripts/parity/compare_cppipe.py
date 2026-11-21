"""Compare .cppipe plans and results in Python, native Rust, and Chromium WASM.

Run from the checkout after building the Python extension and the wasm-pack
web package and installing Playwright Chromium in examples/browser-vite:

    uv run --no-sync python scripts/parity/compare_cppipe.py

Uses every fixtures/cppipe pipeline and standalone measurement pipeline.
Standalone array fixtures also include masks and per-image Costes scales.
The pipeline API takes named arrays only, so these inputs are not replayed.
Specialized measurement steps also reject raw label inputs in Pipeline.run.
Their standalone plans and rejections are tested with the original planes.
SaveImages and ExportToDatabase are compared as descriptors, not file writes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from cp_anywhere import (
    CppipeError,
    Pipeline,
    TertiaryObjectsResult,
    load_pipeline,
)
from cp_anywhere.cppipe import PipelineRunResult

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures/cppipe"
SECONDARY = "secondary/identify_secondary_objects_propagation"
ILLUMINATION = "illumination/illumination_calculate_apply"
SAVE = "illumination/save_images"
EXPORT = "export_to_database/all_objects_overwrite_schema"
PRIMARY_CASES = (
    "adaptive_otsu",
    "adaptive_sauvola",
    "declump_intensity_intensity",
    "declump_shape_propagate_manual_smoothing",
    "fill_holes_border_limit",
    "li",
    "manual",
    "neighbors_excluded_border_object",
    "neighbors_expand_until_adjacent",
    "otsu_simple",
    "otsu_three_class_background",
    "robust_background",
    "segmentation_only_no_measure",
    "threshold_smoothing_log_transform",
)
REGRESSIONS = (
    "border_distance_n",
    "border_propagation",
    "bottom_border_distance_n",
    "bottom_border_propagation",
    "oversized_distance_n",
    "oversized_propagation",
    "renumbered_neighbors_distance_n",
    "renumbered_neighbors_propagation",
)
COMPARTMENTS = (
    "tertiary",
    "size_shape",
    "granularity",
    "intensity_distribution",
    "texture",
    "colocalization",
)
RADIAL_GROUPS = ("jump_groups", "scaled_then_unscaled", "unscaled_then_scaled")
SINGLE_MODULE_SHAPES = (
    "border",
    "diagonal",
    "empty",
    "multiple",
    "nested",
    "point",
    "rectangle",
    "ring",
)
DTYPES = (
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    images: dict[str, np.ndarray]
    # A diagnostic fragment from the Python test, not an entire error spelling.
    compile_error: str | None = None
    run_error: str | None = None
    scaling: dict[str, float] | None = None
    objects: dict[str, np.ndarray] | None = None
    object_diagnostics: dict[str, np.ndarray] | None = None

    def payload(self) -> dict:
        """Pack unnormalized C-order planes in descriptor order."""
        planes = [
            (role, name, plane)
            for role, inputs in (
                ("image", self.images),
                ("objects", self.objects or {}),
                ("object_diagnostics", self.object_diagnostics or {}),
            )
            for name, plane in inputs.items()
        ]
        return {
            "text": self.text,
            "images": [
                {
                    "name": name,
                    "shape": image.shape,
                    "dtype": image.dtype.name,
                    "role": role,
                    "scaling": (self.scaling or {}).get(name)
                    if role == "image"
                    else None,
                }
                for role, name, image in planes
            ],
            "pixels": [
                pixel
                for _, _, image in planes
                for pixel in image.astype(np.float64).ravel().tolist()
            ],
        }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fixture(name: str, images: dict[str, np.ndarray]) -> Case:
    return Case(name, _read(FIXTURES / f"{name}.cppipe"), images)


def _guard_inventory(directory: Path, names: set[str]) -> None:
    found = {
        path.relative_to(directory).with_suffix("").as_posix()
        for path in directory.rglob("*.cppipe")
    }
    if found != names:
        msg = (
            f"{directory}: missing fixtures {sorted(names - found)}; "
            f"fixtures without input mappings {sorted(found - names)}"
        )
        raise ValueError(msg)


def _primary_image(name: str) -> np.ndarray:
    # The Python importer tests use special images for these two neighbor cases.
    custom = {"neighbors_excluded_border_object", "neighbors_expand_until_adjacent"}
    file = f"{name}.test_image.npy" if name in custom else "test_image.npy"
    return np.load(FIXTURES / file)


def _compartments() -> list[Case]:
    dna = np.load(FIXTURES / f"{SECONDARY}.dna.npy")
    stain = np.load(FIXTURES / f"{SECONDARY}.stain.npy")
    images = {"DNA": dna, "Stain": stain}
    narrowed = {"DNA": dna, "Stain": stain.astype(np.float32).astype(np.float64)}
    cases = [
        _fixture(SECONDARY, images),
        _fixture("secondary/identify_secondary_objects_watershed_image", images),
    ]
    for family in COMPARTMENTS:
        inputs = images if family in {"tertiary", "size_shape"} else narrowed
        cases.extend(
            _fixture(f"{family}/{mode}", inputs) for mode in ("shrink", "no_shrink")
        )
    return cases + _radial_groups()


def _radial_groups() -> list[Case]:
    cases = []
    for case in RADIAL_GROUPS:
        name = f"intensity_distribution/{case}"
        oracle = json.loads(_read(FIXTURES / f"{name}.oracle.json"))
        inputs = {
            key: np.asarray(plane["values"], dtype=plane["dtype"])
            for key, plane in oracle["inputs"].items()
        }
        cases.append(_fixture(name, inputs))
    return cases


def _overlay_case() -> Case:
    """Use the raw typed backgrounds and label planes from CellProfiler."""
    name = "overlay_outlines/overlay"
    oracle = json.loads(_read(FIXTURES / f"{name}.json"))
    inputs = {
        key: np.asarray(plane["values"], dtype=plane["dtype"]).reshape(plane["shape"])
        for key, plane in (oracle["images"] | oracle["labels"]).items()
    }
    return _fixture(name, inputs)


def _fixture_cases() -> list[Case]:
    cases = [_fixture(name, {"DNA": _primary_image(name)}) for name in PRIMARY_CASES]
    for case in REGRESSIONS:
        name = f"secondary_regressions/pipeline_{case}"
        fixture = json.loads(_read(FIXTURES / f"{name}.json"))
        cases.append(
            _fixture(name, {k: np.asarray(v) for k, v in fixture["images"].items()})
        )
    for method in ("propagation", "watershed_image"):
        name = f"secondary_border_discard/{method}"
        fixture = json.loads(_read(FIXTURES / f"{name}.json"))
        images = {
            k: np.asarray(v, dtype=np.float32) for k, v in fixture["images"].items()
        }
        cases.append(_fixture(name, images))
    cases.extend(_compartments())
    cases.extend(
        (
            _fixture(
                ILLUMINATION,
                {"DNA": np.load(FIXTURES / f"{ILLUMINATION}.test_image.npy")},
            ),
            _fixture(SAVE, {}),
            _fixture(
                "illumination/image_math_order",
                {
                    "Raw": np.array([[0, 0.25, 0.5], [0.75, 1, 0]], dtype=np.float32),
                    "Flat": np.full((2, 3), 0.5, dtype=np.float32),
                },
            ),
        )
    )
    # The exporter generator appends descriptors to the complete shrink pipeline.
    tertiary_images = next(
        case.images for case in cases if case.name == "tertiary/shrink"
    )
    cases.extend(
        _fixture(name, tertiary_images)
        for name in (EXPORT, "export_to_database/select_objects_never_overwrite")
    )
    cases.append(_overlay_case())
    cases.extend(_quality_cases())
    _guard_inventory(FIXTURES, {case.name for case in cases})
    return cases


def _quality_cases() -> list[Case]:
    path = FIXTURES / "image_quality/quality_flag"
    return [
        Case(
            "image_quality/quality_flag",
            _read(path.with_suffix(".cppipe")),
            {"DNA": np.load(path.with_suffix(".kept.npy"))},
            scaling={"DNA": 4095},
        ),
        *(
            Case(
                f"image_quality/flag_{mode}",
                _read(path.with_name(f"flag_{mode}.cppipe")),
                {},
                run_error="FlagImage.*Slope",
            )
            for mode in ("any", "all")
        ),
    ]


def _shape_cases() -> list[Case]:
    directory = ROOT / "fixtures/size_shape"
    _guard_inventory(directory, set(SINGLE_MODULE_SHAPES))
    cases = []
    for name in SINGLE_MODULE_SHAPES:
        fixture = json.loads(_read(directory / f"{name}.json"))
        labels = np.asarray(fixture["labels"], dtype=np.uint32)
        cases.append(
            Case(
                f"standalone_size_shape/{name}",
                _read(directory / f"{name}.cppipe"),
                {"DNA": np.zeros(labels.shape), "Cells": labels},
            )
        )
    return cases


def _standalone_inputs(family: str, fixture: dict) -> dict[str, np.ndarray]:
    if family == "colocalization":
        return {
            **{
                name: np.asarray(rows, np.float64)
                for name, rows in fixture["pixels"].items()
            },
            **{
                name: np.asarray(rows, np.uint32)
                for name, rows in fixture["labels"].items()
            },
        }
    images: dict[str, np.ndarray] = {
        "Input": np.asarray(
            fixture["pixels"], dtype=fixture.get("pixel_dtype", "float64")
        )
    }
    if family == "granularity":
        images.update(
            {
                name: np.asarray(rows, np.uint32)
                for name, rows in fixture["objects"].items()
            }
        )
    else:
        images["Cells"] = np.asarray(fixture["labels"], np.uint32)
    return images


def _standalone_cases() -> list[Case]:
    cases = _shape_cases()
    modules = {
        "colocalization": "MeasureColocalization",
        "granularity": "MeasureGranularity",
        "intensity_distribution": "MeasureObjectIntensityDistribution",
        "texture": "MeasureTexture",
    }
    for family, module in modules.items():
        directory = ROOT / "fixtures" / family
        oracles = sorted(
            path for path in directory.glob("*.json") if path.stem != "manifest"
        )
        _guard_inventory(directory, {path.stem for path in oracles})
        for path in oracles:
            fixture = json.loads(_read(path))
            # Unlike the basic measurements, these modules resolve
            # object sets only from segmentation outputs.
            uses_objects = family != "colocalization" or bool(fixture["object_names"])
            cases.append(
                Case(
                    f"standalone_{family}/{path.stem}",
                    _read(path.with_suffix(".cppipe")),
                    _standalone_inputs(family, fixture),
                    run_error=rf"{module}.*needs object set" if uses_objects else None,
                )
            )
    return cases


def _substitute(text: str, old: str, new: str) -> str:
    if old not in text:
        msg = f"Mutation target {old!r} no longer exists"
        raise ValueError(msg)
    return text.replace(old, new, 1)


def _mutate(  # noqa: PLR0913 -- two distinct stages with independent diagnostics.
    case: Case,
    name: str,
    old: str,
    new: str,
    *,
    compile_error: str | None = None,
    run_error: str | None = None,
) -> Case:
    return replace(
        case,
        name=name,
        text=_substitute(case.text, old, new),
        compile_error=compile_error,
        run_error=run_error,
    )


def _revision_cases(cases: dict[str, Case]) -> list[Case]:
    revisions = (
        ("otsu_simple", "IdentifyPrimaryObjects", 11),
        ("tertiary/shrink", "IdentifyTertiaryObjects", 999),
        ("standalone_size_shape/rectangle", "MeasureObjectSizeShape", 999),
        ("granularity/shrink", "MeasureGranularity", 999),
        ("intensity_distribution/shrink", "MeasureObjectIntensityDistribution", 999),
        ("texture/shrink", "MeasureTexture", 999),
        ("colocalization/shrink", "MeasureColocalization", 999),
        (EXPORT, "ExportToDatabase", 999),
    )
    failures = []
    for source, module, revision in revisions:
        case = cases[source]
        match = re.search(
            rf"(?m)^{module}:\[.*?variable_revision_number:\d+", case.text
        )
        if match is None:
            msg = f"{source}: could not locate {module} revision"
            raise ValueError(msg)
        original = match.group()
        failures.append(
            _mutate(
                case,
                f"rejected/revision/{module}",
                original,
                f"{original.rpartition(':')[0]}:{revision}",
                compile_error=rf"{module}.*{revision}",
            )
        )
    return failures


def _rejected_settings(cases: dict[str, Case]) -> list[Case]:
    failures = (
        (
            "otsu_simple",
            "basic",
            "Use advanced settings?:Yes",
            "Use advanced settings?:No",
            "advanced settings",
        ),
        (
            "otsu_simple",
            "within_neighbors",
            "Method to determine neighbors:Adjacent",
            "Method to determine neighbors:Within",
            "distance method",
        ),
        (
            "secondary_border_discard/propagation",
            "discard_associated_primary_objects",
            "Discard the associated primary objects?:No",
            "Discard the associated primary objects?:Yes",
            "Discard the associated primary objects",
        ),
        (
            "intensity_distribution/shrink",
            "other_center",
            "Object to use as center?:These objects",
            "Object to use as center?:Centers of other objects",
            "Centers of other objects",
        ),
        ("texture/shrink", "scale_count", "Hidden:1", "Hidden:2", "MeasureTexture"),
        (
            "colocalization/shrink",
            "single_channel",
            "Select images to measure:DNA, Stain",
            "Select images to measure:DNA",
            "at least 2 selected images",
        ),
        (
            ILLUMINATION,
            "all_cycles",
            "or based on all images?:Each",
            "or based on all images?:All: Across cycles",
            "calculate_illumination_across",
        ),
        (
            ILLUMINATION,
            "first_cycle",
            "or based on all images?:Each",
            "or based on all images?:All: First cycle",
            "calculate_illumination_across",
        ),
        (
            ILLUMINATION,
            "convex_hull",
            "Smoothing method:Gaussian Filter",
            "Smoothing method:Convex Hull",
            "not yet implemented",
        ),
        (
            ILLUMINATION,
            "splines",
            "Smoothing method:Gaussian Filter",
            "Smoothing method:Splines",
            "not yet implemented",
        ),
        (
            SAVE,
            "sequential_names",
            "file names:Single name",
            "file names:Sequential numbers",
            "out-of-scope",
        ),
        (SAVE, "h5", "Saved file format:tiff", "Saved file format:h5", "out-of-scope"),
        (
            SAVE,
            "save_mask",
            "Select the type of image to save:Image",
            "Select the type of image to save:Mask",
            "out-of-scope",
        ),
        (EXPORT, "mysql", "Database type:SQLite", "Database type:MySQL", "MySQL"),
        (
            EXPORT,
            "combined_table",
            "a single object view?:One table per object type",
            "a single object view?:Single object table",
            "single object table or a single",
        ),
        (
            EXPORT,
            "table_prefix",
            "Add a prefix to table names?:No",
            "Add a prefix to table names?:Yes",
            "prefix",
        ),
    )
    return [
        _mutate(cases[source], f"rejected/{name}", old, new, compile_error=pattern)
        for source, name, old, new, pattern in failures
    ]


def _runtime_errors(cases: dict[str, Case]) -> list[Case]:
    return [
        replace(
            cases[ILLUMINATION],
            name="rejected/missing_illumination_input",
            images={},
            run_error="CorrectIlluminationCalculate.*DNA",
        ),
        replace(
            cases["otsu_simple"],
            name="rejected/missing_primary_image",
            images={},
            run_error="IdentifyPrimaryObjects.*DNA",
        ),
        _mutate(
            cases[SECONDARY],
            "rejected/missing_primary_objects",
            "Select the input objects:Nuclei",
            "Select the input objects:OtherNuclei",
            run_error="IdentifySecondaryObjects.*OtherNuclei",
        ),
        _mutate(
            cases["tertiary/shrink"],
            "rejected/missing_tertiary_objects",
            "Select the smaller identified objects:Nuclei",
            "Select the smaller identified objects:Missing",
            run_error="IdentifyTertiaryObjects.*Missing",
        ),
        _mutate(
            cases["granularity/shrink"],
            "rejected/missing_granularity_objects",
            "Nuclei, Cells",
            "Nuclei, Cells, Mystery",
            run_error="MeasureGranularity.*Mystery",
        ),
    ]


def _module(case: Case, module: str) -> str:
    return next(
        part for part in case.text.split("\n\n") if part.startswith(f"{module}:")
    )


def _measurement_modules(case: Case, image: str) -> str:
    return "\n\n".join(
        _substitute(
            _module(case, name),
            "Select images to measure:DNA",
            f"Select images to measure:{image}",
        )
        for name in ("MeasureImageIntensity", "MeasureObjectIntensity")
    )


def _integration_cases(cases: dict[str, Case]) -> list[Case]:
    primary = cases["otsu_simple"]
    header, _, body = primary.text.partition("\n\n")
    unknown = (
        "Images:[module_num:1|variable_revision_number:2|enabled:True]\n"
        "    Filter images?:No filtering"
    )
    illumination = cases[ILLUMINATION]
    measured = _measurement_modules(primary, "DNACorrected")
    label_shape = illumination.images["DNA"].shape
    illumination_text = _substitute(illumination.text, "ModuleCount:2", "ModuleCount:4")
    return [
        _image_texture_tie_case(
            cases["standalone_texture/float32_quantization_ties"],
        ),
        replace(
            primary,
            name="variant/skipped_images_module",
            text=f"{header}\n\n{unknown}\n\n{body}",
        ),
        replace(
            illumination,
            name="variant/measure_corrected_image",
            text=f"{illumination_text.rstrip()}\n\n{measured}\n",
            images={**illumination.images, "Cells": np.ones(label_shape, np.uint32)},
        ),
        _mutate(
            cases[SAVE],
            "variant/save_png",
            "Saved file format:tiff",
            "Saved file format:png",
        ),
        _mutate(
            cases[SAVE],
            "variant/save_no_overwrite",
            "Overwrite existing files without warning?:Yes",
            "Overwrite existing files without warning?:No",
        ),
    ]


def _image_texture_tie_case(case: Case) -> Case:
    text = _substitute(
        case.text, "Select objects to measure:Cells", "Select objects to measure:"
    )
    return replace(
        case,
        name="variant/image_texture_float32_ties",
        text=_substitute(
            text,
            "Measure whole images or objects?:Both",
            "Measure whole images or objects?:Images",
        ),
        run_error=None,
    )


def _dtype_cases(primary: Case) -> list[Case]:
    header = _substitute(
        primary.text.partition("\n\n")[0], "ModuleCount:4", "ModuleCount:2"
    )
    modules = _measurement_modules(primary, "DNA")
    text = f"{header}\n\n{modules}\n"
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint32)
    cases = []
    for name in DTYPES:
        dtype = np.dtype(name)
        if dtype.kind in "iu":
            limits = np.iinfo(name)
            # JavaScript's safe integer range also applies to the 64-bit dtypes.
            safe = 2**53 - 1
            lo = max(limits.min, -safe)
            hi = min(limits.max // 2 + 1, safe)
            pixels = np.array([[lo, 0], [hi, 16]], dtype=dtype)
        else:
            pixels = np.array([[0.0, 0.1], [0.375, 1.0]], dtype=dtype)
        cases.append(Case(f"dtype/{name}", text, {"DNA": pixels, "Cells": labels}))
    return cases


def _raw_tertiary_cases(cases: dict[str, Case]) -> list[Case]:
    inputs = []
    for mode in ("shrink", "no_shrink"):
        name = f"tertiary/{mode}"
        case = cases[name]
        fixture = json.loads(_read(FIXTURES / f"{name}.expected.json"))
        labels = {
            k: np.asarray(fixture["labels"][k], np.uint32) for k in ("Nuclei", "Cells")
        }
        header = _substitute(
            case.text.split("\n\n", 1)[0], "ModuleCount:4", "ModuleCount:2"
        )
        tertiary = _module(case, "IdentifyTertiaryObjects").replace(
            "module_num:3", "module_num:1", 1
        )
        intensity = _module(case, "MeasureObjectIntensity").replace(
            "module_num:4", "module_num:2", 1
        )
        text = f"{header}\n\n{tertiary}\n\n{intensity}\n"
        inputs.append(
            Case(f"variant/raw_tertiary/{mode}", text, {**case.images, **labels})
        )
    return inputs


def _explicit_object_cases(manual: Case) -> list[Case]:
    header, primary, *steps = manual.text.strip().split("\n\n")
    plugin = "RunCellpose:[module_num:1|variable_revision_number:3|enabled:True]"
    simple = "\n\n".join((header, plugin, *steps)) + "\n"
    excluded = _substitute(
        simple,
        "Consider objects discarded for touching image border?:No",
        "Consider objects discarded for touching image border?:Yes",
    )
    labels = np.zeros((5, 6), dtype=np.uint32)
    labels[1:3, 1:3] = 1
    diagnostic = labels.copy()
    diagnostic[1:3, 3:5] = 2
    images = {"DNA": np.ones(labels.shape, dtype=np.float32)}
    supplied = Case("objects/supplied", simple, images, objects={"Cells": labels})
    return [
        supplied,
        Case(
            "rejected/later_object_producer",
            "\n\n".join((header, steps[0], primary, *steps[1:])) + "\n",
            images,
            run_error=r"MeasureObjectIntensity.*Cells.*later",
        ),
        Case(
            "objects/geometry_without_matching_image",
            _read(ROOT / "fixtures/size_shape/rectangle.cppipe"),
            {"Unrelated": np.zeros((2, 3), dtype=np.float32)},
            objects={"Cells": labels},
        ),
        replace(
            supplied,
            name="objects/excluded_neighbors",
            text=excluded,
            object_diagnostics={"Cells": diagnostic},
        ),
        replace(
            supplied,
            name="rejected/missing_objects",
            objects=None,
            run_error=r"MeasureObjectIntensity.*Cells",
        ),
        replace(
            supplied,
            name="rejected/missing_diagnostics",
            text=excluded,
            run_error=r"MeasureObjectNeighbors.*Cells.*object_diagnostics",
        ),
    ]


def _cases() -> list[Case]:
    cases = _fixture_cases() + _standalone_cases()
    neighbor = ROOT / "fixtures/neighbors_distance/primary_mixed"
    oracle = json.loads(_read(neighbor.with_suffix(".json")))
    cases.append(
        Case(
            "neighbors_distance/primary_mixed",
            _read(neighbor.with_suffix(".cppipe")),
            {"DNA": np.asarray(oracle["pixels"], dtype=np.float32)},
        )
    )
    # The individual inventories check mappings; this also catches a new
    # fixture in a directory that is not yet listed by the harness.
    count = sum(1 for _ in (ROOT / "fixtures").rglob("*.cppipe"))
    if len(cases) != count:
        msg = f"Mapped {len(cases)} .cppipe fixtures, but found {count} under fixtures/"
        raise ValueError(msg)
    lookup = {case.name: case for case in cases}
    cases.extend(_integration_cases(lookup))
    cases.append(
        replace(
            lookup["image_quality/quality_flag"],
            name="variant/quality_flag_skips",
            images={
                "DNA": np.load(FIXTURES / "image_quality/quality_flag.skipped.npy")
            },
        )
    )
    cases.extend(_raw_tertiary_cases(lookup))
    cases.extend(_explicit_object_cases(lookup["manual"]))
    cases.extend(_dtype_cases(lookup["otsu_simple"]))
    cases.extend(_revision_cases(lookup))
    cases.extend(_rejected_settings(lookup))
    cases.extend(_runtime_errors(lookup))
    cases.append(
        Case(
            "rejected/not_a_pipeline",
            "just some random text\n",
            {},
            compile_error="not a CellProfiler",
        )
    )
    if len({case.name for case in cases}) != len(cases):
        msg = "Duplicate parity case names"
        raise ValueError(msg)
    return cases


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _json_value(value.ravel().tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}[str(value)]
    return value


def _expected_error(outcome: object, pattern: str, path: str) -> None:
    if isinstance(outcome, dict) and outcome.keys() == {"error"}:
        diagnostic = outcome["error"]
        if isinstance(diagnostic, str) and re.search(
            pattern, diagnostic, flags=re.I | re.S
        ):
            return
    msg = f"{path}: expected error matching {pattern!r}, got {repr(outcome)[:300]}"
    raise ValueError(msg)


def _python_result(case: Case, directory: Path) -> dict:
    source = directory / "source.cppipe"
    source.write_text(case.text, encoding="utf-8")
    try:
        pipeline = load_pipeline(source)
    except CppipeError as error:
        if case.compile_error is None:
            raise
        rejection = {"error": str(error)}
        _expected_error(rejection, case.compile_error, f"{case.name}: Python compile")
        return rejection
    if case.compile_error:
        msg = f"{case.name}: Python unexpectedly compiled"
        raise ValueError(msg)
    plan = _json_value(asdict(pipeline))
    try:
        result = pipeline.run_detailed(
            case.images,
            scaling=case.scaling,
            objects=case.objects,
            object_diagnostics=case.object_diagnostics,
        )
    except CppipeError as error:
        if case.run_error is None:
            raise
        rejection = {"error": str(error)}
        _expected_error(rejection, case.run_error, f"{case.name}: Python run")
        return {"plan": plan, **rejection}
    if case.run_error:
        msg = f"{case.name}: Python unexpectedly ran"
        raise ValueError(msg)
    return {
        "plan": plan,
        "result": _serialize_run(case, pipeline, result),
    }


def _serialize_run(
    case: Case,
    pipeline: Pipeline,
    run: PipelineRunResult,
) -> object:
    primary, derived, measured = run.primary, run.secondary, run.measurements
    secondary = {}
    tertiary = {}
    shapes = {name: labels.shape for name, labels in (case.objects or {}).items()}
    shapes.update({name: result.labels.shape for name, result in primary.items()})
    for name, result in derived.items():
        (tertiary if isinstance(result, TertiaryObjectsResult) else secondary)[name] = (
            asdict(result)
        )
        shapes[name] = result.labels.shape
    for step in () if run.skipped else pipeline.tertiary_steps:
        for name in (step.primary_object, step.secondary_object):
            if name in case.images and name not in shapes:
                shapes[name] = case.images[name].shape
    measurements = (
        None
        if measured is None
        else {
            "features": measured.features,
            "object_ids": measured.object_ids,
            "relationships": measured.relationships,
        }
    )
    return _json_value(
        {
            "primary": {name: asdict(value) for name, value in primary.items()},
            "secondary": secondary,
            "tertiary": tertiary,
            "measurements": measurements,
            "object_shapes": shapes,
            "derived_images": {
                name: {"shape": plane.shape, "values": plane}
                for name, plane in run.derived_images.items()
            },
            "skipped": run.skipped,
        }
    )


def _duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = dict(pairs)
    if len(keys) != len(pairs):
        msg = "Duplicate keys in runtime JSON"
        raise ValueError(msg)
    return keys


def _invalid_constant(value: str) -> None:
    msg = f"Bare nonfinite value in runtime JSON: {value}"
    raise ValueError(msg)


def _read_json(data: str | bytes) -> object:
    return json.loads(
        data, object_pairs_hook=_duplicate_keys, parse_constant=_invalid_constant
    )


def _array(value: object, count: int, runtime: str) -> list:
    if not isinstance(value, list) or len(value) != count:
        size = len(value) if isinstance(value, list) else type(value).__name__
        msg = f"{runtime}: expected {count} case results, got {size}"
        raise ValueError(msg)
    return value


def _compare(
    expected: object, actual: object, path: str, *, approximate: bool = False
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            msg = f"{path}: missing {missing}, extra {extra}"
            raise ValueError(msg)
        for key, value in expected.items():
            _compare(value, actual[key], f"{path}.{key}", approximate=approximate)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            msg = f"{path}: expected {len(expected)} entries, got {len(actual)}"
            raise ValueError(msg)
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            _compare(want, got, f"{path}[{index}]", approximate=approximate)
        return
    if (
        isinstance(expected, float)
        and isinstance(actual, (float, int))
        and not isinstance(actual, bool)
    ):
        # A JSON number need not contain a fractional part. Only Python fields
        # typed as floating point get a tolerance, and only in run output.
        if actual == expected or (
            approximate
            and ".Intensity_TotalArea_" not in path
            and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
        ):
            return
    elif type(expected) is type(actual) and expected == actual:
        return
    msg = f"{path}: expected {repr(expected)[:150]}, got {repr(actual)[:150]}"
    raise ValueError(msg)


def _browser_value(value: object, path: str) -> object:
    if isinstance(value, dict):
        if value.keys() == {"json"} and isinstance(value["json"], str):
            return _read_json(value["json"])
        if value.keys() == {"error"} and isinstance(value["error"], str):
            return value
    msg = f"{path}: expected a browser JSON string or an error"
    raise ValueError(msg)


def _check_native(case: Case, expected: dict, actual: object) -> None:
    path = f"native/{case.name}"
    if case.compile_error:
        _expected_error(actual, case.compile_error, f"{path}.compile")
        return
    if not case.run_error and isinstance(actual, dict) and "error" in actual:
        msg = f"{path}: unexpected rejection: {actual['error']}"
        raise ValueError(msg)
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        _compare(expected, actual, path)
        return
    _compare(expected["plan"], actual["plan"], f"{path}.plan")
    if case.run_error:
        _expected_error({"error": actual["error"]}, case.run_error, f"{path}.run")
    else:
        _compare(
            expected["result"], actual["result"], f"{path}.result", approximate=True
        )


def _check_browser(case: Case, expected: dict, actual: object) -> None:
    path = f"chromium/{case.name}"
    if not isinstance(actual, dict) or actual.keys() != {"compile", "run"}:
        msg = f"{path}: expected separate compile and run responses"
        raise ValueError(msg)
    compiled = _browser_value(actual["compile"], f"{path}.compile")
    ran = _browser_value(actual["run"], f"{path}.run")
    if case.compile_error:
        _expected_error(compiled, case.compile_error, f"{path}.compile")
    else:
        _compare(expected["plan"], compiled, f"{path}.compile")
    error = case.compile_error or case.run_error
    if error:
        _expected_error(ran, error, f"{path}.run")
    else:
        if isinstance(ran, dict) and "error" in ran:
            msg = f"{path}.run: unexpected rejection: {ran['error']}"
            raise ValueError(msg)
        _compare(expected["result"], ran, f"{path}.run", approximate=True)


def _run_native(data: str, count: int) -> list:
    run = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--release",
            "-p",
            "cp-anywhere-wasm",
            "--example",
            "cppipe_parity",
        ],
        input=data,
        encoding="utf-8",
        capture_output=True,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    if run.returncode:
        msg = f"cppipe_parity exited {run.returncode}:\n{run.stderr}"
        raise RuntimeError(msg)
    return _array(_read_json(run.stdout), count, "native")


def _run_chromium(data: str, count: int, directory: Path) -> list:
    source = directory / "cases.json"
    result = directory / "chromium.json"
    source.write_text(data, encoding="utf-8")
    subprocess.run(
        [
            "node",
            "examples/browser-vite/run-cppipe-check.mjs",
            str(source),
            str(result),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    return _array(_read_json(_read(result)), count, "chromium")


def _check_coverage(cases: list[Case], expected: list[dict]) -> None:
    plans = [row["plan"] for row in expected if "result" in row]
    missing = {key for key in plans[0] if not any(plan[key] for plan in plans)}
    if missing:
        msg = f"No successful case covers plan fields: {sorted(missing)}"
        raise ValueError(msg)
    measured = [row["result"]["measurements"] for row in expected if "result" in row]
    if not any(item and item["relationships"] for item in measured):
        msg = "Expected at least one real neighbor relationship"
        raise ValueError(msg)
    if not any(case.compile_error for case in cases) or not any(
        case.run_error for case in cases
    ):
        msg = "Missing compile or runtime error coverage"
        raise ValueError(msg)


def _run(cases: list[Case], directory: Path, *, full: bool = True) -> None:
    expected = [_python_result(case, directory) for case in cases]
    if full:
        _check_coverage(cases, expected)
    data = json.dumps(
        [case.payload() for case in cases], allow_nan=False, separators=(",", ":")
    )
    failures = []
    for check, results in (
        (_check_native, _run_native(data, len(cases))),
        (_check_browser, _run_chromium(data, len(cases), directory)),
    ):
        for case, want, got in zip(cases, expected, results, strict=True):
            try:
                check(case, want, got)
            except ValueError as error:
                failures.append(str(error))
    if failures:
        raise SystemExit(".cppipe parity failed:\n" + "\n".join(failures))


def main() -> None:
    """Run all cases; exit nonzero on any mismatch or missing runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radial-groups",
        action="store_true",
        help="run only the radial group fixtures",
    )
    args = parser.parse_args()
    cases = _radial_groups() if args.radial_groups else _cases()
    with tempfile.TemporaryDirectory(prefix="cp-anywhere-cppipe-parity-") as temp:
        _run(cases, Path(temp), full=not args.radial_groups)
    errors = sum(bool(case.compile_error or case.run_error) for case in cases)
    print(
        f".cppipe parity passed: {len(cases) - errors} successes, {errors} rejections "
        "(Python, native Rust, Chromium WASM)"
    )


if __name__ == "__main__":
    main()
