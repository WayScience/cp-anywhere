"""Import real CellProfiler ``.cppipe`` pipeline files, translating the
settings of every module ``cp_anywhere`` implements
(``IdentifyPrimaryObjects``, ``MeasureImageIntensity``,
``MeasureObjectIntensity``, ``MeasureObjectNeighbors``) into this
project's own configuration -- not a full pipeline execution engine.

The ``.cppipe`` text format (header lines, then one block per module:
``ModuleName:[attr:'val'|...]`` followed by indented, strictly
*positional* ``Label:Value`` setting lines -- the label text is
decorative and discarded, exactly as real CellProfiler's own loader
discards it) was reverse-engineered **from the real serializer**
(``cellprofiler_core.pipeline.io._v5.dump``, the pinned 4.2.8 source),
not guessed: every field order, separator, and value encoding below
was confirmed by generating real pipeline files with the pinned
CellProfiler oracle and reading their literal text. See
``spec/cppipe-compatibility.md`` for the full settings-mapping table
and every explicit gap.

This module has **no CellProfiler dependency** -- replicating that
would require re-implementing CellProfiler's own per-module
``upgrade_settings`` version-migration chains (years of legacy-format
branches per module), which is explicitly out of scope. Instead, this
importer accepts only the *current* ``variable_revision_number`` for
each supported module (the one the pinned 4.2.8 oracle itself writes)
and raises a clear, explicit error naming the unsupported revision
otherwise -- asking the user to re-open and re-save the pipeline in a
current CellProfiler first, rather than silently mis-translating an
older settings layout.

Pipeline input/organization modules (``Images``, ``Metadata``,
``NamesAndTypes``, ``Groups``, and anything else this module does not
explicitly translate) are recognized and skipped, not silently
dropped: see :attr:`Pipeline.skipped_modules`. This importer does not
replicate CellProfiler's own file-matching/grouping logic; the caller
supplies already-loaded image arrays by the same names the pipeline's
``IdentifyPrimaryObjects``/``Measure*`` modules reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cp_anywhere.api import _MAX_NEIGHBOR_DISTANCE, MeasurementResult, measure
from cp_anywhere.colocalization import measure_colocalization
from cp_anywhere.granularity import measure_granularity
from cp_anywhere.illumination import apply_illumination, calculate_illumination
from cp_anywhere.intensity_distribution import (
    _measure_intensity_distribution,
    _ZernikeOption,
)
from cp_anywhere.save_images import BIT_DEPTH_8, save_array
from cp_anywhere.segment import (
    SecondaryObjectsResult,
    identify_primary_objects,
    identify_secondary_objects,
    identify_tertiary_objects,
)
from cp_anywhere.texture import measure_texture

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from cp_anywhere.segment import (
        SegmentationResult,
        TertiaryObjectsResult,
    )

# `IdentifyPrimaryObjects.variable_revision_number` and the embedded
# `Threshold` sub-module's own `threshold_setting_version`, as written by
# the pinned CellProfiler 4.2.8 oracle (cellprofiler/modules/identifyprimaryobjects.py,
# cellprofiler/modules/threshold.py). A real `.cppipe` with either value
# different from this was saved by a different CellProfiler version and
# needs an upgrade chain this importer deliberately does not replicate.
_IPO_REVISION = 15
_THRESHOLD_SETTING_VERSION = 12
_MEASURE_OBJECT_INTENSITY_REVISION = 4
_MEASURE_IMAGE_INTENSITY_REVISION = 4
_MEASURE_OBJECT_NEIGHBORS_REVISION = 3
_ISO_REVISION = 10
_ITO_REVISION = 3
_MOSS_REVISION = 3
_MOSS_FIELDS = ("objects", "zernikes", "advanced")
_ITO_FIELDS = ("secondary", "primary", "output", "shrink")
_MG_REVISION = 4
_MG_FIELDS = (
    "images",
    "wants_objects",
    "objects",
    "subsample_size",
    "image_sample_size",
    "element_size",
    "granular_spectrum_length",
)
_MOID_REVISION = 6
_MOID_FIXED_FIELDS = 6
_MOID_GROUP_WIDTH = 3
_INTENSITY_DISTRIBUTION_ZERNIKES: dict[str, _ZernikeOption] = {
    "None": "none",
    "Magnitudes only": "magnitudes",
    "Magnitudes and phase": "magnitudes_and_phase",
}
_MT_REVISION = 7
# MeasureTexture's own fixed leading settings; the scale count (the
# "Hidden" setting) tells us how many trailing "Texture scale to
# measure" settings follow, one per configured scale group.
_MT_FIELDS = (
    "images",
    "objects",
    "gray_levels",
    "scale_count",
    "images_or_objects",
)
_MT_IMAGES_OR_OBJECTS = {
    "Images": (True, False),
    "Objects": (False, True),
    "Both": (True, True),
}
_MC_REVISION = 5
_MC_FIELDS = (
    "images",
    "threshold_percent",
    "images_or_objects",
    "objects",
    "do_all",
    "do_corr_and_slope",
    "do_manders",
    "do_rwc",
    "do_overlap",
    "do_costes",
    "fast_costes",
)
# "Run all metrics?" is UI-only -- real CellProfiler's own run() never
# reads it, only the five do_* flags it toggles, which are what this
# translator reads directly (see MeasureColocalization.run()).
_MC_IMAGES_OR_OBJECTS = {
    "Across entire image": (True, False),
    "Within objects": (False, True),
    "Both": (True, True),
}
_MC_FAST_COSTES = {
    "Faster": "faster",
    "Fast": "fast",
    "Accurate": "accurate",
}
_MIN_COLOCALIZATION_IMAGES = 2
_ICALC_REVISION = 2
# CorrectIlluminationCalculate's own fixed settings (confirmed against
# a real pipeline.dump() of the pinned oracle -- see
# scripts/reference/generate_illumination_calculate.py). The four
# save_average_image/average_image_name/save_dilated_image/
# dilated_image_name settings are consumed positionally but never
# read (this project has no "retain an intermediate image" concept);
# the six spline_* settings are likewise consumed positionally, only
# read far enough to raise a clear error when Splines is requested.
_ICALC_FIELDS = (
    "input_image",
    "output_image",
    "intensity_choice",
    "dilate_objects",
    "object_dilation_radius",
    "block_size",
    "rescale_option",
    "each_or_all",
    "smoothing_method",
    "automatic_object_width",
    "object_width",
    "size_of_smoothing_filter",
    "save_average_image",
    "average_image_name",
    "save_dilated_image",
    "dilated_image_name",
    "automatic_splines",
    "spline_mode",
    "spline_points",
    "spline_threshold",
    "spline_rescale",
    "spline_maxiter",
    "spline_convergence",
)
_ICALC_EACH_OR_ALL = {"Each", "All: First cycle", "All: Across cycles"}
_IAPPLY_REVISION = 5
_IAPPLY_SETTINGS_PER_IMAGE = 4
_IAPPLY_TRAILING_SETTINGS = 2
_ETD_REVISION = 28
# ExportToDatabase's own fixed leading settings (indices 0-38, confirmed
# against a real pipeline.dump() of the pinned oracle -- see
# scripts/reference/generate_export_to_database.py). The four
# "Hidden:" counts (image_group_count, group_field_count,
# filter_field_count, workspace_measurement_count) tell us how many
# trailing CellProfiler-Analyst-properties/workspace-file settings
# follow; this project implements none of that (see ExportToDatabaseStep's
# own docstring), so they are only used to validate the settings count,
# never read.
_ETD_FIELDS = (
    "db_type",
    "db_name",
    "want_table_prefix",
    "table_prefix",
    "directory",
    "save_cpa_properties",
    "db_host",
    "db_user",
    "db_password",
    "sqlite_file",
    "wants_agg_mean",
    "wants_agg_median",
    "wants_agg_std_dev",
    "wants_agg_mean_well",
    "wants_agg_median_well",
    "wants_agg_std_dev_well",
    "objects_choice",
    "objects_list",
    "max_column_size",
    "separate_object_tables",
    "properties_image_url_prepend",
    "want_image_thumbnails",
    "thumbnail_image_names",
    "auto_scale_thumbnail_intensities",
    "properties_plate_type",
    "properties_plate_metadata",
    "properties_well_metadata",
    "properties_export_all_image_defaults",
    "image_group_count",
    "group_field_count",
    "filter_field_count",
    "workspace_measurement_count",
    "experiment_name",
    "location_object",
    "properties_class_table_name",
    "wants_relationship_table_setting",
    "allow_overwrite",
    "wants_properties_image_url_prepend",
    "properties_classification_type",
)
_ETD_IMAGE_GROUP_SETTING_COUNT = 4
_ETD_GROUP_FIELD_SETTING_COUNT = 2
_ETD_FILTER_FIELD_SETTING_COUNT = 2
_ETD_WORKSPACE_MEASUREMENT_SETTING_COUNT = 9
_ETD_DB_TYPE_SQLITE = "SQLite"
_ETD_SEPARATE_OBJECT_TABLES_PER_OBJECT = "One table per object type"
_ETD_OBJECTS_CHOICE = {"All": "all", "None": "none", "Select...": "select"}
_ETD_OVERWRITE_TO_IF_EXISTS = {
    "Data and schema": "replace",
    # Real CellProfiler distinguishes "keep tables, overwrite the
    # data" from "ask before dropping tables (error if headless)" --
    # cp_anywhere's to_sqlite/to_duckdb have only "append"/"replace",
    # and neither prompts or errors on existing tables, so both
    # non-schema-dropping choices map to the same "append" here. A
    # documented simplification, not a faithful behavioral match.
    "Data only": "append",
    "Never": "append",
}


# Display-string -> cp_anywhere vocabulary, taken verbatim from the real
# module constants (cellprofiler/modules/identifyprimaryobjects.py's
# UN_*/WA_*/FH_*/LIMIT_* and cellprofiler/modules/threshold.py's
# TS_*/TM_*/O_*/RB_*), not guessed.
_UNCLUMP = {"None": "none", "Intensity": "intensity", "Shape": "shape"}
_WATERSHED = {
    "None": "none",
    "Intensity": "intensity",
    "Shape": "shape",
    "Propagate": "propagate",
}
_FILL_HOLES = {
    "Never": "never",
    "After declumping only": "after_declump_only",
    "After both thresholding and declumping": "after_thresholding_and_declump",
}
_THRESHOLD_SCOPE = {"Global": "global", "Adaptive": "adaptive"}
_THRESHOLD_OPERATION = {
    "Manual": "manual",
    "Otsu": "otsu",  # refined to "otsu_three_class" below via two_class_otsu
    "Minimum Cross-Entropy": "li",
    "Robust Background": "robust_background",
    "Sauvola": "sauvola",
}
_AVERAGING = {"Mean": "mean", "Median": "median", "Mode": "mode"}
_VARIANCE = {"Standard deviation": "sd", "Median absolute deviation": "mad"}
_YES_NO = {"Yes": True, "No": False}

# Positional settings() order for IdentifyPrimaryObjects at
# variable_revision_number=15 -- ImageSegmentation.settings() (x_name,
# y_name) + the module's own 14 settings + threshold_setting_version +
# Threshold.settings()[2:] (17 settings), 34 total. Confirmed against a
# real pipeline.dump() of the pinned oracle, not derived from reading
# settings() alone.
_IPO_FIELDS = (
    "x_name",
    "y_name",
    "size_range",
    "exclude_size",
    "exclude_border_objects",
    "unclump_method",
    "watershed_method",
    "smoothing_filter_size",
    "maxima_suppression_size",
    "low_res_maxima",
    "fill_holes",
    "automatic_smoothing",
    "automatic_suppression",
    "limit_choice",
    "maximum_object_count",
    "use_advanced",
    "threshold_setting_version",
    "threshold_scope",
    "global_operation",
    "threshold_smoothing_scale",
    "threshold_correction_factor",
    "threshold_range",
    "manual_threshold",
    "thresholding_measurement",
    "two_class_otsu",
    "log_transform",
    "assign_middle_to_foreground",
    "adaptive_window_size",
    "lower_outlier_fraction",
    "upper_outlier_fraction",
    "averaging_method",
    "variance_method",
    "number_of_deviations",
    "local_operation",
)

# Positional settings() order for IdentifySecondaryObjects at
# variable_revision_number=10 -- ImageSegmentation.settings() (x_name,
# y_name) + the module's own 8 settings + threshold_setting_version +
# Threshold.settings()[2:] (the same 18-field tail as _IPO_FIELDS[16:]),
# 28 total. Confirmed against a real pipeline.dump() of the pinned
# oracle, not derived from reading settings() alone.
_ISO_FIELDS = (
    "x_name",
    "y_name",
    "method",
    "image_name",
    "distance_to_dilate",
    "regularization_factor",
    "wants_discard_edge",
    "wants_discard_primary",
    "new_primary_objects_name",
    "fill_holes",
    *_IPO_FIELDS[16:],
)

# M_PROPAGATION/M_WATERSHED_G/M_WATERSHED_I/M_DISTANCE_N/M_DISTANCE_B,
# taken verbatim from cellprofiler/modules/identifysecondaryobjects.py.
_SECONDARY_METHOD = {
    "Propagation": "propagation",
    "Distance - N": "distance_n",
    "Watershed - Image": "watershed_image",
}


class CppipeError(ValueError):
    """An explicit, named incompatibility (not a parse failure) --
    e.g. an unsupported module revision, an out-of-scope setting value,
    or a settings combination outside ``spec/cppipe-compatibility.md``'s
    declared scope. Always names the offending module/setting/value;
    never a silent fallback."""


@dataclass(frozen=True)
class _RawModule:
    name: str
    variable_revision_number: int
    settings: tuple[str, ...]
    enabled: bool = True


def _parse_text(text: str) -> list[_RawModule]:
    """Parse the real ``.cppipe`` text format into raw, untranslated
    module blocks (generic: no per-module field knowledge here)."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("CellProfiler Pipeline:"):
        msg = "not a CellProfiler .cppipe file (missing header line)"
        raise CppipeError(msg)
    i = 1
    module_count = None
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip() == "":
            break
        key, _, value = line.partition(":")
        if key == "ModuleCount":
            module_count = int(value)
    modules: list[_RawModule] = []
    while i < len(lines) and (module_count is None or len(modules) < module_count):
        if lines[i].strip() == "":
            i += 1
            continue
        header = lines[i]
        i += 1
        name, _, attribute_string = header.partition(":")
        name = name.strip()
        match = re.search(r"variable_revision_number:(\d+)", attribute_string)
        if match is None:
            msg = f"module {name!r} is missing variable_revision_number"
            raise CppipeError(msg)
        revision = int(match.group(1))
        settings = []
        while i < len(lines) and lines[i].strip() != "":
            _, _, value = lines[i].partition(":")
            settings.append(value)
            i += 1
        enabled = (
            re.search(r"(?:\[|\|)enabled:False(?:\||\])", attribute_string) is None
        )
        modules.append(_RawModule(name, revision, tuple(settings), enabled))
    return modules


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CppipeError(message)


def _parse_range(text: str) -> tuple[float, float]:
    low, _, high = text.partition(",")
    return float(low), float(high)


@dataclass(frozen=True)
class SegmentationStep:
    """One translated ``IdentifyPrimaryObjects`` module: an input image
    name, the object name it writes, and the ``identify_primary_objects``
    kwargs (matching :func:`cp_anywhere.identify_primary_objects`'s own
    parameter names exactly, so ``**step.config`` works directly)."""

    input_image: str
    output_object: str
    config: dict[str, object]


@dataclass(frozen=True)
class SecondaryObjectsStep:
    """One translated ``IdentifySecondaryObjects`` module: the input
    (secondary stain) image name, the primary object name it grows
    around, the object name it writes, and the
    ``identify_secondary_objects`` kwargs (matching
    :func:`cp_anywhere.segment.identify_secondary_objects`'s own
    parameter names exactly, so ``**step.config`` works directly)."""

    input_image: str
    primary_object: str
    output_object: str
    config: dict[str, object]


@dataclass(frozen=True)
class TertiaryObjectsStep:
    """Subtract smaller objects from larger objects on the same grid."""

    primary_object: str
    secondary_object: str
    output_object: str
    shrink_primary: bool


@dataclass(frozen=True)
class GranularityStep:
    """One translated ``MeasureGranularity`` module: every requested
    image name (each measured independently, matching real
    CellProfiler), the object sets to also measure per-object spectra
    on, and ``cp_anywhere.measure_granularity``'s own keyword settings."""

    images: tuple[str, ...]
    objects: tuple[str, ...]
    subsample_size: float
    image_sample_size: float
    element_size: int
    granular_spectrum_length: int


@dataclass(frozen=True)
class RadialBinGroup:
    """One saved radial bin group, in pipeline order."""

    bin_count: int
    wants_scaled: bool
    maximum_radius: float


@dataclass(frozen=True)
class IntensityDistributionStep:
    """One revision 6 radial module with ordered object and bin groups.

    The original scalar bin settings describe the first group. Each object
    group uses its own objects as centers.
    """

    images: tuple[str, ...]
    objects: tuple[str, ...]
    bin_count: int
    wants_scaled: bool
    maximum_radius: float
    zernikes: _ZernikeOption
    zernike_degree: int
    additional_bin_groups: tuple[RadialBinGroup, ...] = ()


@dataclass(frozen=True)
class TextureStep:
    """One translated ``MeasureTexture`` module: every requested image and
    object name (each image measured against every object set, matching
    real CellProfiler's own full cross-product of feature columns), and
    ``cp_anywhere.measure_texture``'s own keyword settings. 2D only --
    real CellProfiler's 3D volumetric correspondence vectors (13
    directions instead of 4) are an explicit, documented gap."""

    images: tuple[str, ...]
    objects: tuple[str, ...]
    gray_levels: int
    scales: tuple[int, ...]
    wants_image_measurements: bool
    wants_object_measurements: bool


@dataclass(frozen=True)
class ColocalizationStep:
    """One translated ``MeasureColocalization`` module: every requested
    image (every ``i < j`` pair is measured, matching real
    CellProfiler's own ``get_image_pairs()``), at most one object set,
    and ``cp_anywhere.measure_colocalization``'s own keyword settings.
    2D only -- real CellProfiler's 3D volumetric case is an explicit,
    documented gap. "Run all metrics?" is UI-only and not represented
    here; the five ``do_*`` flags are read directly, matching real
    CellProfiler's own ``run()``."""

    images: tuple[str, ...]
    objects: tuple[str, ...]
    threshold_percent: float
    fast_costes: str
    do_corr_and_slope: bool
    do_manders: bool
    do_rwc: bool
    do_overlap: bool
    do_costes: bool
    wants_image_measurements: bool
    wants_object_measurements: bool


@dataclass(frozen=True)
class IlluminationCalculateStep:
    """One translated ``CorrectIlluminationCalculate`` module, ``"Each"``
    mode only: ``"All: First cycle"``/``"All: Across cycles"`` need the
    full image set list (every cycle's own image, not just one), which
    does not fit :meth:`Pipeline.run`'s single-image-set signature --
    an explicit, documented gap (use
    :func:`cp_anywhere.calculate_illumination_across` directly on your
    own loaded image sequence instead). ``Convex Hull`` and ``Splines``
    smoothing methods are separate, not-yet-implemented gaps (see
    :func:`cp_anywhere.calculate_illumination`); "Retain the
    averaged/dilated image?" has no equivalent here (this project has
    no "intermediate image" concept) and is silently not represented
    -- CellProfiler's own `SaveImages` module would be needed to save
    those, not this translator.
    """

    input_image: str
    output_image: str
    intensity_choice: str
    dilate_objects: bool
    object_dilation_radius: int
    block_size: int
    rescale_option: str
    smoothing_method: str
    automatic_object_width: str
    object_width: float
    size_of_smoothing_filter: float


@dataclass(frozen=True)
class ImageMathStep:
    """A single-input, unmasked ImageMath None operation."""

    input_image: str
    output_image: str
    input_factor: float
    output_factor: float
    addend: float
    truncate_low: bool
    truncate_high: bool
    replace_nan: bool


@dataclass(frozen=True)
class IlluminationApplyImage:
    """One of ``CorrectIlluminationApply``'s repeated image groups."""

    input_image: str
    output_image: str
    illumination_function_image: str
    divide_or_subtract: str


@dataclass(frozen=True)
class IlluminationApplyStep:
    """One translated ``CorrectIlluminationApply`` module: every
    configured image group (CellProfiler lets one module instance
    apply correction to several images at once, sharing the same
    truncate settings)."""

    images: tuple[IlluminationApplyImage, ...]
    truncate_low: bool
    truncate_high: bool


_SAVE_IMAGES_REVISION = 16
# SaveImages' own fixed 18-setting positional layout (confirmed against
# a real pipeline.dump() of the pinned oracle -- see
# scripts/cppipe/generate_save_images_pipeline.py). Settings not listed
# as a SaveImagesStep field below are consumed positionally to validate
# the settings count. Some settings apply only to rejected modes.
# Others (directory resolution, cycle timing, filename measurements,
# subfolders, and TIFF compression) are not executed by this importer.
_SAVE_IMAGES_FIELDS = (
    "save_image_or_figure",
    "image_name",
    "file_name_method",
    "file_image_name",
    "single_file_name",
    "number_of_digits",
    "wants_file_name_suffix",
    "file_name_suffix",
    "file_format",
    "pathname",
    "bit_depth",
    "overwrite",
    "when_to_save",
    "update_file_names",
    "create_subdirectories",
    "root_dir",
    "stack_axis",
    "tiff_compress",
)
_SAVE_IMAGES_FILE_FORMATS = {"tiff", "png", "jpeg", "npy"}


@dataclass(frozen=True)
class SaveImagesStep:
    """One translated ``SaveImages`` module.

    Only ``save_image_or_figure == "Image"`` (not ``"Mask"``,
    ``"Cropping"``, or ``"Movie/Stack"``, which need a cropping mask or
    multi-cycle frame-stacking concept this project does not have) and
    ``file_name_method == "Single name"`` (not ``"From image
    filename"``/``"Sequential numbers"``, which need per-image-set
    source-file/metadata tracking this project does not have) are
    supported -- explicit, documented gaps. ``"h5"`` (``FF_H5``) is a
    separate, not-yet-implemented gap (see
    :func:`cp_anywhere.save_images.save_array`'s own module docstring);
    ``"tiff"``/``"png"``/``"jpeg"``/``"npy"`` are supported.

    Like :class:`ExportToDatabaseStep`, this project never resolves a
    CellProfiler ``Directory`` setting's "Default Output/Input Folder"
    choice to an absolute path -- the caller supplies ``path`` to
    :meth:`write`, exactly as the caller supplies every image array to
    :meth:`Pipeline.run`. ``file_name`` is kept as the module's raw
    name setting. The directory setting is not retained, and
    ``write()`` uses only the caller-supplied destination.

    ``Pipeline.run()`` does not execute these save steps automatically.
    Cycle timing, filename measurements, subfolders, and TIFF compression
    settings are not implemented. See ``spec/illumination.md``.

    Real CellProfiler's own ``get_bit_depth()`` forces 8-bit for every
    file format except TIFF/HDF5, regardless of the ``bit_depth``
    setting's own value -- :attr:`effective_bit_depth` already applies
    that rule.
    """

    image_name: str
    file_name: str
    file_format: str
    bit_depth: str
    overwrite: bool

    @property
    def effective_bit_depth(self) -> str:
        """The bit depth real CellProfiler would actually apply --
        ``bit_depth`` is only honored for TIFF; every other supported
        format (PNG/JPEG/NPY) is always forced to 8-bit integer (NPY's
        own writer then ignores bit depth entirely regardless)."""
        if self.file_format == "tiff":
            return self.bit_depth
        return BIT_DEPTH_8

    def write(
        self, image: np.ndarray, path: str | Path, *, bit_depth: str | None = None
    ) -> None:
        """Write ``image`` to ``path`` via
        :func:`cp_anywhere.save_images.save_array`, using
        :attr:`effective_bit_depth` unless ``bit_depth`` overrides it.
        Honors this step's own ``overwrite`` setting."""
        destination = Path(path)
        if not self.overwrite and destination.exists():
            msg = (
                f"SaveImages: {destination} already exists and this "
                "module's Overwrite setting is No"
            )
            raise CppipeError(msg)
        save_array(image, destination, bit_depth=bit_depth or self.effective_bit_depth)


@dataclass(frozen=True)
class ExportToDatabaseStep:
    """One translated ``ExportToDatabase`` module. Only SQLite output is
    supported (no MySQL server connection of any kind), and only "one
    table per object type" (``separate_object_tables``'s own
    ``OT_PER_OBJECT`` -- real CellProfiler's default, "Single object
    table", combines every selected object's measurements into one row
    per object and is an explicit, documented gap here; use
    :meth:`MeasurementResult.to_parquet(mode="joined")
    <cp_anywhere.MeasurementResult.to_parquet>` instead if you want that
    shape). Table prefixes (``want_table_prefix``), the CellProfiler
    Analyst properties file, the object-relationship table, and the
    per-image aggregate (mean/median/std-dev) columns are also explicit,
    documented gaps -- none of ``to_sqlite``/``to_duckdb`` generate them.

    ``sqlite_file`` is the bare filename from the module's own setting:
    like every other translated step, this project never resolves a
    CellProfiler ``Directory`` setting's "Default Output/Input Folder"
    choice to an absolute path -- the caller supplies ``path`` to
    :meth:`write`, exactly as the caller supplies every image array to
    :meth:`Pipeline.run`.
    """

    sqlite_file: str
    objects_choice: str  # "all" | "none" | "select"
    objects: tuple[str, ...]
    if_exists: str  # "append" | "replace"

    def selected_scopes(self, result: MeasurementResult) -> tuple[str, ...]:
        """The ``result.features`` scopes this step would write."""
        if self.objects_choice == "none":
            return ("Image",) if "Image" in result.features else ()
        if self.objects_choice == "all":
            return tuple(result.features)
        return tuple(
            scope
            for scope in result.features
            if scope == "Image" or scope in self.objects
        )

    def write(self, result: MeasurementResult, path: str | Path) -> None:
        """Write ``result``'s selected scopes to a SQLite file at
        ``path`` via :meth:`MeasurementResult.to_sqlite`, honoring this
        step's ``objects_choice``/``objects`` selection and
        ``if_exists`` overwrite policy."""
        selected = result.select_scopes(self.selected_scopes(result))
        selected.to_sqlite(path, if_exists=self.if_exists)


@dataclass(frozen=True)
class MeasurementStep:
    """One translated ``Measure*`` module's requested image/object
    names and which ``cp_anywhere.measure`` family it activates."""

    module_name: str
    family: str
    images: tuple[str, ...]
    objects: tuple[str, ...]
    calculate_zernikes: bool = True
    calculate_advanced: bool = False
    wants_excluded_objects: bool = False
    """Whether neighbor geometry uses the object's small-removed layer.

    This matches CellProfiler's excluded-objects setting. Layer contents
    depend on the upstream segmentation module, not only this flag.
    """
    wants_expand: bool = False
    """Whether this ``neighbors`` step uses CellProfiler's "Expand until
    adjacent" distance method (``measure(..., neighbors_expand=...)``)
    instead of the default "Adjacent". Only meaningful when
    ``family == "neighbors"``.
    """
    neighbors_distance: int | None = None
    """Pixel distance for a "Within a specified distance" neighbors step."""


def _translate_threshold_tail(values: dict[str, str]) -> dict[str, object]:
    """Shared ``Threshold.settings()[2:]`` translation, reused by both
    ``IdentifyPrimaryObjects`` and ``IdentifySecondaryObjects`` (the
    same sub-module, same 18-field positional tail in both -- see
    ``_IPO_FIELDS``/``_ISO_FIELDS``). Returns
    ``identify_primary_objects``/``identify_secondary_objects``'s own
    shared keyword names (``threshold``, not ``threshold_method`` --
    both Python wrappers expose a single polymorphic ``threshold``
    parameter, not the raw CellProfiler setting name)."""
    _require(
        int(values["threshold_setting_version"]) == _THRESHOLD_SETTING_VERSION,
        "The embedded Threshold setting version="
        f"{values['threshold_setting_version']} is not supported (only "
        f"{_THRESHOLD_SETTING_VERSION} is)",
    )
    threshold_scope_cp = values["threshold_scope"]
    _require(
        threshold_scope_cp in _THRESHOLD_SCOPE,
        f"unsupported threshold_scope: {threshold_scope_cp!r}",
    )
    operation_text = (
        values["global_operation"]
        if threshold_scope_cp == "Global"
        else values["local_operation"]
    )
    _require(
        operation_text in _THRESHOLD_OPERATION,
        f"unsupported threshold operation: {operation_text!r} (e.g. "
        "'Measurement' is an explicit, documented gap)",
    )
    threshold_method = _THRESHOLD_OPERATION[operation_text]
    if threshold_method == "otsu" and values["two_class_otsu"] == "Three classes":
        threshold_method = "otsu_three_class"
    _require(
        values["averaging_method"] in _AVERAGING,
        f"unsupported averaging_method: {values['averaging_method']!r}",
    )
    _require(
        values["variance_method"] in _VARIANCE,
        f"unsupported variance_method: {values['variance_method']!r}",
    )

    config: dict[str, object] = {
        "threshold_correction_factor": float(values["threshold_correction_factor"]),
        "threshold_range": _parse_range(values["threshold_range"]),
        "threshold_smoothing_scale": float(values["threshold_smoothing_scale"]),
        "log_transform": _YES_NO[values["log_transform"]],
        "threshold_scope": _THRESHOLD_SCOPE[threshold_scope_cp],
        "adaptive_window_size": int(values["adaptive_window_size"]),
    }
    if threshold_method == "manual":
        config["threshold"] = float(values["manual_threshold"])
    else:
        config["threshold"] = threshold_method
    if threshold_method == "otsu_three_class":
        config["assign_middle_to_foreground"] = (
            values["assign_middle_to_foreground"] == "Foreground"
        )
    if threshold_method == "robust_background":
        config["robust_background_lower_outlier_fraction"] = float(
            values["lower_outlier_fraction"]
        )
        config["robust_background_upper_outlier_fraction"] = float(
            values["upper_outlier_fraction"]
        )
        config["robust_background_averaging_method"] = _AVERAGING[
            values["averaging_method"]
        ]
        config["robust_background_variance_method"] = _VARIANCE[
            values["variance_method"]
        ]
        config["robust_background_number_of_deviations"] = float(
            values["number_of_deviations"]
        )
    return config


def _translate_identify_primary_objects(raw: _RawModule) -> SegmentationStep:
    _require(
        raw.variable_revision_number == _IPO_REVISION,
        "IdentifyPrimaryObjects variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_IPO_REVISION} is); re-save the pipeline in a current "
        "CellProfiler first",
    )
    _require(
        len(raw.settings) == len(_IPO_FIELDS),
        f"IdentifyPrimaryObjects has {len(raw.settings)} settings, "
        f"expected {len(_IPO_FIELDS)} for revision {_IPO_REVISION}",
    )
    values = dict(zip(_IPO_FIELDS, raw.settings, strict=True))
    _require(
        _YES_NO[values["use_advanced"]],
        "IdentifyPrimaryObjects with 'Use advanced settings?' = No is not "
        "supported: CellProfiler's own basic mode silently overrides "
        "most of the saved settings at run time (see "
        "spec/cppipe-compatibility.md); toggle it to Yes and re-save",
    )
    _require(
        values["unclump_method"] in _UNCLUMP,
        f"unsupported unclump_method: {values['unclump_method']!r} (e.g. "
        "'Laplacian of Gaussian' is an explicit, documented gap)",
    )
    _require(
        values["fill_holes"] in _FILL_HOLES,
        f"unsupported fill_holes: {values['fill_holes']!r}",
    )
    _require(
        values["limit_choice"] in ("Continue", "Erase"),
        f"unsupported limit_choice: {values['limit_choice']!r}",
    )

    min_diameter, max_diameter = _parse_range(values["size_range"])
    automatic_smoothing = _YES_NO[values["automatic_smoothing"]]
    automatic_suppression = _YES_NO[values["automatic_suppression"]]

    config = _translate_threshold_tail(values)
    config.update(
        {
            "exclude_size": _YES_NO[values["exclude_size"]],
            "min_diameter": min_diameter,
            "max_diameter": max_diameter,
            "exclude_border_objects": _YES_NO[values["exclude_border_objects"]],
            "maximum_object_count": (
                int(values["maximum_object_count"])
                if values["limit_choice"] == "Erase"
                else None
            ),
            "fill_holes": _FILL_HOLES[values["fill_holes"]],
            "unclump_method": _UNCLUMP[values["unclump_method"]],
            "watershed_method": _WATERSHED[values["watershed_method"]],
            "smoothing_filter_size": (
                None if automatic_smoothing else float(values["smoothing_filter_size"])
            ),
            "maxima_suppression_size": (
                None
                if automatic_suppression
                else float(values["maxima_suppression_size"])
            ),
            "low_resolution_maxima": _YES_NO[values["low_res_maxima"]],
        }
    )

    return SegmentationStep(
        input_image=values["x_name"], output_object=values["y_name"], config=config
    )


def _translate_identify_secondary_objects(raw: _RawModule) -> SecondaryObjectsStep:
    _require(
        raw.variable_revision_number == _ISO_REVISION,
        "IdentifySecondaryObjects variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_ISO_REVISION} is); re-save the pipeline in a current "
        "CellProfiler first",
    )
    _require(
        len(raw.settings) == len(_ISO_FIELDS),
        f"IdentifySecondaryObjects has {len(raw.settings)} settings, "
        f"expected {len(_ISO_FIELDS)} for revision {_ISO_REVISION}",
    )
    values = dict(zip(_ISO_FIELDS, raw.settings, strict=True))
    _require(
        values["method"] in _SECONDARY_METHOD,
        f"IdentifySecondaryObjects method {values['method']!r} is an "
        "explicit, not-yet-implemented gap ('Watershed - Gradient' and "
        "'Distance - B' are not yet supported)",
    )
    _require(
        not _YES_NO[values["wants_discard_edge"]],
        "IdentifySecondaryObjects with 'Discard secondary objects "
        "touching the border of the image?' = Yes is an explicit, "
        "not-yet-implemented gap",
    )
    method = _SECONDARY_METHOD[values["method"]]

    config: dict[str, object] = {
        "method": method,
        "fill_holes": _YES_NO[values["fill_holes"]],
    }
    if method != "distance_n":
        config.update(_translate_threshold_tail(values))
        if method == "propagation":
            config["regularization_factor"] = float(values["regularization_factor"])
    else:
        config["distance_to_dilate"] = float(values["distance_to_dilate"])

    return SecondaryObjectsStep(
        input_image=values["image_name"],
        primary_object=values["x_name"],
        output_object=values["y_name"],
        config=config,
    )


def _translate_identify_tertiary_objects(raw: _RawModule) -> TertiaryObjectsStep:
    # Real revision-3 pipeline.dump(): larger, smaller, output, shrink.
    _require(
        raw.variable_revision_number == _ITO_REVISION,
        "IdentifyTertiaryObjects variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only 3 is)",
    )
    _require(
        len(raw.settings) == len(_ITO_FIELDS),
        f"IdentifyTertiaryObjects has {len(raw.settings)} settings, expected 4",
    )
    secondary, primary, output, shrink = raw.settings
    _require(shrink in _YES_NO, f"unsupported shrink_primary: {shrink!r}")
    return TertiaryObjectsStep(primary, secondary, output, _YES_NO[shrink])


def _translate_measure_object_size_shape(raw: _RawModule) -> MeasurementStep:
    _require(
        raw.variable_revision_number == _MOSS_REVISION,
        "MeasureObjectSizeShape variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only 3 is)",
    )
    _require(
        len(raw.settings) == len(_MOSS_FIELDS),
        f"MeasureObjectSizeShape has {len(raw.settings)} settings, expected 3",
    )
    objects, zernikes, advanced = raw.settings
    _require(
        zernikes in _YES_NO and advanced in _YES_NO,
        "MeasureObjectSizeShape requires Yes/No calculation flags",
    )
    return MeasurementStep(
        module_name="MeasureObjectSizeShape",
        family="object_size_shape",
        images=(),
        objects=_split_list(objects),
        calculate_zernikes=_YES_NO[zernikes],
        calculate_advanced=_YES_NO[advanced],
    )


def _split_list(text: str) -> tuple[str, ...]:
    """CellProfiler's ``ListSubscriber`` settings join multiple selected
    names with ``", "`` (confirmed in
    ``cellprofiler_core/setting/subscriber/list_subscriber/_list_subscriber.py``'s
    ``value.split(", ")``, and in a real saved pipeline's
    ``Select images to measure:DNA, Actin`` line)."""
    return tuple(item for item in text.split(", ") if item)


def _translate_measure_granularity(raw: _RawModule) -> GranularityStep:
    _require(
        raw.variable_revision_number == _MG_REVISION,
        "MeasureGranularity variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MG_REVISION} is)",
    )
    _require(
        len(raw.settings) == len(_MG_FIELDS),
        f"MeasureGranularity has {len(raw.settings)} settings, "
        f"expected {len(_MG_FIELDS)} for revision {_MG_REVISION}",
    )
    values = dict(zip(_MG_FIELDS, raw.settings, strict=True))
    _require(
        values["wants_objects"] in _YES_NO,
        "MeasureGranularity requires a Yes/No 'Measure within objects?' flag",
    )
    objects = _split_list(values["objects"]) if _YES_NO[values["wants_objects"]] else ()
    return GranularityStep(
        images=_split_list(values["images"]),
        objects=objects,
        subsample_size=float(values["subsample_size"]),
        image_sample_size=float(values["image_sample_size"]),
        element_size=int(values["element_size"]),
        granular_spectrum_length=int(values["granular_spectrum_length"]),
    )


def _translate_measure_object_intensity_distribution(
    raw: _RawModule,
) -> IntensityDistributionStep:
    _require(
        raw.variable_revision_number == _MOID_REVISION,
        "MeasureObjectIntensityDistribution variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MOID_REVISION} is)",
    )
    module = "MeasureObjectIntensityDistribution"
    _require(len(raw.settings) >= _MOID_FIXED_FIELDS, f"{module}: too few settings")
    images, object_count, bin_count, heatmap_count, zernikes, degree = raw.settings[
        :_MOID_FIXED_FIELDS
    ]
    nobjects = _radial_group_count("object", object_count)
    nbins = _radial_group_count("bin", bin_count)
    _require(
        _radial_group_count("heatmap", heatmap_count, minimum=0) == 0,
        f"{module}: heatmap groups are not supported",
    )
    bins_start = _MOID_FIXED_FIELDS + _MOID_GROUP_WIDTH * nobjects
    expected = bins_start + _MOID_GROUP_WIDTH * nbins
    _require(
        len(raw.settings) == expected,
        f"{module} has {len(raw.settings)} settings, expected {expected} for "
        f"{nobjects} object group(s) and {nbins} bin group(s)",
    )
    objects = []
    for offset in range(_MOID_FIXED_FIELDS, bins_start, _MOID_GROUP_WIDTH):
        name, center, _ = raw.settings[offset : offset + _MOID_GROUP_WIDTH]
        _require(
            center == "These objects",
            f"{module}: 'Object to use as center?' = {center!r} for {name!r} "
            "is not supported (only 'These objects' is)",
        )
        _require(name not in objects, f"{module}: duplicate object group {name!r}")
        objects.append(name)
    groups = _parse_radial_bins(raw.settings[bins_start:])
    _require(
        zernikes in _INTENSITY_DISTRIBUTION_ZERNIKES,
        f"{module}: unsupported 'Calculate intensity Zernikes?': {zernikes!r}",
    )
    first, *additional = groups
    return IntensityDistributionStep(
        images=_split_list(images),
        objects=tuple(objects),
        bin_count=first.bin_count,
        wants_scaled=first.wants_scaled,
        maximum_radius=first.maximum_radius,
        zernikes=_INTENSITY_DISTRIBUTION_ZERNIKES[zernikes],
        zernike_degree=int(degree),
        additional_bin_groups=tuple(additional),
    )


def _radial_group_count(name: str, text: str, *, minimum: int = 1) -> int:
    message = (
        f"MeasureObjectIntensityDistribution: {name} group count must be an "
        f"integer >= {minimum}, got {text!r}"
    )
    try:
        count = int(text)
    except ValueError as error:
        raise CppipeError(message) from error
    _require(count >= minimum, message)
    return count


def _parse_radial_bins(settings: tuple[str, ...]) -> list[RadialBinGroup]:
    module = "MeasureObjectIntensityDistribution"
    groups: list[RadialBinGroup] = []
    counts: set[int] = set()
    for offset in range(0, len(settings), _MOID_GROUP_WIDTH):
        scaled, count, radius = settings[offset : offset + _MOID_GROUP_WIDTH]
        _require(scaled in _YES_NO, f"{module}: 'Scale the bins?' requires Yes/No")
        try:
            bins = int(count)
            maximum_radius = float(radius)
        except ValueError as error:
            message = f"{module}: invalid bin count or maximum radius"
            raise CppipeError(message) from error
        _require(bins >= 1, f"{module}: bin count must be positive")
        _require(
            np.isfinite(maximum_radius) and (scaled == "Yes" or maximum_radius > 0),
            f"{module}: maximum radius must be finite and positive for unscaled bins",
        )
        _require(bins not in counts, f"{module}: duplicate bin count {bins}")
        counts.add(bins)
        groups.append(RadialBinGroup(bins, _YES_NO[scaled], maximum_radius))
    return groups


def _translate_measure_texture(raw: _RawModule) -> TextureStep:
    _require(
        raw.variable_revision_number == _MT_REVISION,
        f"MeasureTexture variable_revision_number={raw.variable_revision_number} "
        f"is not supported (only {_MT_REVISION} is)",
    )
    _require(
        len(raw.settings) >= len(_MT_FIELDS),
        f"MeasureTexture has {len(raw.settings)} settings, expected at "
        f"least {len(_MT_FIELDS)} for revision {_MT_REVISION}",
    )
    values = dict(zip(_MT_FIELDS, raw.settings, strict=False))
    scale_count = int(values["scale_count"])
    _require(
        len(raw.settings) == len(_MT_FIELDS) + scale_count,
        f"MeasureTexture has {len(raw.settings)} settings, expected "
        f"{len(_MT_FIELDS)} + {scale_count} scale(s) for revision {_MT_REVISION}",
    )
    _require(
        values["images_or_objects"] in _MT_IMAGES_OR_OBJECTS,
        "unsupported 'Measure whole images or objects?': "
        f"{values['images_or_objects']!r}",
    )
    wants_image, wants_object = _MT_IMAGES_OR_OBJECTS[values["images_or_objects"]]
    scales = tuple(int(s) for s in raw.settings[len(_MT_FIELDS) :])
    return TextureStep(
        images=_split_list(values["images"]),
        objects=_split_list(values["objects"]),
        gray_levels=int(values["gray_levels"]),
        scales=scales,
        wants_image_measurements=wants_image,
        wants_object_measurements=wants_object,
    )


def _translate_measure_colocalization(raw: _RawModule) -> ColocalizationStep:
    _require(
        raw.variable_revision_number == _MC_REVISION,
        f"MeasureColocalization variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MC_REVISION} is)",
    )
    _require(
        len(raw.settings) == len(_MC_FIELDS),
        f"MeasureColocalization has {len(raw.settings)} settings, "
        f"expected {len(_MC_FIELDS)} for revision {_MC_REVISION}",
    )
    values = dict(zip(_MC_FIELDS, raw.settings, strict=True))
    images = _split_list(values["images"])
    _require(
        len(images) >= _MIN_COLOCALIZATION_IMAGES,
        f"MeasureColocalization needs at least 2 selected images (got {len(images)})",
    )
    _require(
        values["images_or_objects"] in _MC_IMAGES_OR_OBJECTS,
        "unsupported 'Select where to measure correlation': "
        f"{values['images_or_objects']!r}",
    )
    wants_image, wants_object = _MC_IMAGES_OR_OBJECTS[values["images_or_objects"]]
    for flag in (
        "do_corr_and_slope",
        "do_manders",
        "do_rwc",
        "do_overlap",
        "do_costes",
    ):
        _require(
            values[flag] in _YES_NO,
            f"MeasureColocalization requires a Yes/No value for {flag!r}",
        )
    _require(
        values["fast_costes"] in _MC_FAST_COSTES,
        f"unsupported 'Method for Costes thresholding': {values['fast_costes']!r}",
    )
    return ColocalizationStep(
        images=images,
        objects=_split_list(values["objects"]) if wants_object else (),
        threshold_percent=float(values["threshold_percent"]),
        fast_costes=_MC_FAST_COSTES[values["fast_costes"]],
        do_corr_and_slope=_YES_NO[values["do_corr_and_slope"]],
        do_manders=_YES_NO[values["do_manders"]],
        do_rwc=_YES_NO[values["do_rwc"]],
        do_overlap=_YES_NO[values["do_overlap"]],
        do_costes=_YES_NO[values["do_costes"]],
        wants_image_measurements=wants_image,
        wants_object_measurements=wants_object,
    )


def _translate_correct_illumination_calculate(
    raw: _RawModule,
) -> IlluminationCalculateStep:
    _require(
        raw.variable_revision_number == _ICALC_REVISION,
        "CorrectIlluminationCalculate variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_ICALC_REVISION} is)",
    )
    _require(
        len(raw.settings) == len(_ICALC_FIELDS),
        f"CorrectIlluminationCalculate has {len(raw.settings)} settings, "
        f"expected {len(_ICALC_FIELDS)} for revision {_ICALC_REVISION}",
    )
    values = dict(zip(_ICALC_FIELDS, raw.settings, strict=True))
    _require(
        values["each_or_all"] in _ICALC_EACH_OR_ALL,
        f"unknown CorrectIlluminationCalculate each_or_all {values['each_or_all']!r}",
    )
    _require(
        values["each_or_all"] == "Each",
        f"CorrectIlluminationCalculate each_or_all={values['each_or_all']!r} needs "
        "the full image set list (every cycle's own image), which does not fit "
        "load_pipeline's single-image-set Pipeline.run(); call "
        "cp_anywhere.calculate_illumination_across(...) directly on your own "
        "loaded image sequence instead",
    )
    _require(
        values["smoothing_method"]
        in {
            "No smoothing",
            "Median Filter",
            "Gaussian Filter",
            "Fit Polynomial",
            "Smooth to Average",
        },
        f"CorrectIlluminationCalculate smoothing_method={values['smoothing_method']!r} "
        "is an explicit, out-of-scope gap (Convex Hull and Splines are not yet "
        "implemented)",
    )
    _require(
        values["dilate_objects"] in _YES_NO,
        "CorrectIlluminationCalculate requires a Yes/No "
        "'Dilate objects in the final averaged image?' flag",
    )
    return IlluminationCalculateStep(
        input_image=values["input_image"],
        output_image=values["output_image"],
        intensity_choice=values["intensity_choice"],
        dilate_objects=_YES_NO[values["dilate_objects"]],
        object_dilation_radius=int(values["object_dilation_radius"]),
        block_size=int(values["block_size"]),
        rescale_option=values["rescale_option"],
        smoothing_method=values["smoothing_method"],
        automatic_object_width=values["automatic_object_width"],
        object_width=float(values["object_width"]),
        size_of_smoothing_filter=float(values["size_of_smoothing_filter"]),
    )


def _translate_correct_illumination_apply(raw: _RawModule) -> IlluminationApplyStep:
    _require(
        raw.variable_revision_number == _IAPPLY_REVISION,
        "CorrectIlluminationApply variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_IAPPLY_REVISION} is)",
    )
    _require(
        len(raw.settings) % _IAPPLY_SETTINGS_PER_IMAGE == _IAPPLY_TRAILING_SETTINGS,
        f"CorrectIlluminationApply has {len(raw.settings)} settings, which is "
        "not a valid image-group count plus the trailing truncate_low/"
        "truncate_high settings",
    )
    image_count = len(raw.settings) // _IAPPLY_SETTINGS_PER_IMAGE
    images = []
    for i in range(image_count):
        offset = i * _IAPPLY_SETTINGS_PER_IMAGE
        input_image, output_image, illum_image, divide_or_subtract = raw.settings[
            offset : offset + _IAPPLY_SETTINGS_PER_IMAGE
        ]
        _require(
            divide_or_subtract in ("Divide", "Subtract"),
            "unknown CorrectIlluminationApply divide_or_subtract "
            f"{divide_or_subtract!r}",
        )
        images.append(
            IlluminationApplyImage(
                input_image=input_image,
                output_image=output_image,
                illumination_function_image=illum_image,
                divide_or_subtract=divide_or_subtract,
            )
        )
    truncate_low, truncate_high = raw.settings[
        image_count * _IAPPLY_SETTINGS_PER_IMAGE :
    ]
    _require(
        truncate_low in _YES_NO and truncate_high in _YES_NO,
        "CorrectIlluminationApply requires Yes/No truncate_low/truncate_high flags",
    )
    return IlluminationApplyStep(
        images=tuple(images),
        truncate_low=_YES_NO[truncate_low],
        truncate_high=_YES_NO[truncate_high],
    )


def _translate_save_images(raw: _RawModule) -> SaveImagesStep:
    _require(
        raw.variable_revision_number == _SAVE_IMAGES_REVISION,
        f"SaveImages variable_revision_number={raw.variable_revision_number} "
        f"is not supported (only {_SAVE_IMAGES_REVISION} is)",
    )
    _require(
        len(raw.settings) == len(_SAVE_IMAGES_FIELDS),
        f"SaveImages has {len(raw.settings)} settings, expected "
        f"{len(_SAVE_IMAGES_FIELDS)} for revision {_SAVE_IMAGES_REVISION}",
    )
    values = dict(zip(_SAVE_IMAGES_FIELDS, raw.settings, strict=True))
    _require(
        values["save_image_or_figure"] == "Image",
        "SaveImages save_image_or_figure="
        f"{values['save_image_or_figure']!r} is an explicit, out-of-scope "
        'gap (only "Image" is supported, not "Mask"/"Cropping"/'
        '"Movie/Stack")',
    )
    _require(
        values["file_name_method"] == "Single name",
        "SaveImages file_name_method="
        f"{values['file_name_method']!r} is an explicit, out-of-scope gap "
        '(only "Single name" is supported -- "From image filename"/'
        '"Sequential numbers" need per-image-set source-file/metadata '
        "tracking this project does not have)",
    )
    _require(
        values["file_format"] in _SAVE_IMAGES_FILE_FORMATS,
        f"SaveImages file_format={values['file_format']!r} is an explicit, "
        f"out-of-scope gap (only {sorted(_SAVE_IMAGES_FILE_FORMATS)!r} are "
        'supported, not "h5")',
    )
    _require(
        values["overwrite"] in _YES_NO,
        "SaveImages requires a Yes/No Overwrite flag",
    )
    return SaveImagesStep(
        image_name=values["image_name"],
        file_name=values["single_file_name"],
        file_format=values["file_format"],
        bit_depth=values["bit_depth"],
        overwrite=_YES_NO[values["overwrite"]],
    )


def _translate_export_to_database(raw: _RawModule) -> ExportToDatabaseStep:
    _require(
        raw.variable_revision_number == _ETD_REVISION,
        f"ExportToDatabase variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_ETD_REVISION} is)",
    )
    _require(
        len(raw.settings) >= len(_ETD_FIELDS),
        f"ExportToDatabase has {len(raw.settings)} settings, expected at "
        f"least {len(_ETD_FIELDS)} for revision {_ETD_REVISION}",
    )
    values = dict(zip(_ETD_FIELDS, raw.settings[: len(_ETD_FIELDS)], strict=True))
    expected_total = (
        len(_ETD_FIELDS)
        + _ETD_IMAGE_GROUP_SETTING_COUNT * int(values["image_group_count"])
        + 1  # properties_wants_groups
        + _ETD_GROUP_FIELD_SETTING_COUNT * int(values["group_field_count"])
        + 2  # properties_wants_filters, create_filters_for_plates
        + _ETD_FILTER_FIELD_SETTING_COUNT * int(values["filter_field_count"])
        + 1  # create_workspace_file
        + _ETD_WORKSPACE_MEASUREMENT_SETTING_COUNT
        * int(values["workspace_measurement_count"])
    )
    _require(
        len(raw.settings) == expected_total,
        f"ExportToDatabase has {len(raw.settings)} settings, expected "
        f"{expected_total} for revision {_ETD_REVISION} given its "
        "image/group/filter/workspace-measurement counts",
    )
    _require(
        values["db_type"] == _ETD_DB_TYPE_SQLITE,
        f"ExportToDatabase database type {values['db_type']!r} is an "
        "explicit, out-of-scope gap (only 'SQLite' is supported; this "
        "project never connects to a MySQL server)",
    )
    _require(
        values["separate_object_tables"] == _ETD_SEPARATE_OBJECT_TABLES_PER_OBJECT,
        "ExportToDatabase 'Create one table per object, a single object "
        f"table or a single object view?' = {values['separate_object_tables']!r} "
        "is an explicit, out-of-scope gap (only "
        f"{_ETD_SEPARATE_OBJECT_TABLES_PER_OBJECT!r} is supported; see "
        "ExportToDatabaseStep's own docstring for the "
        'to_parquet(mode="joined") alternative)',
    )
    _require(
        not _YES_NO[values["want_table_prefix"]],
        "ExportToDatabase 'Add a prefix to table names?' = Yes is an "
        "explicit, out-of-scope gap (to_sqlite/to_duckdb always use "
        "CellProfiler's own unprefixed Per_Image/Per_<ObjectName> names)",
    )
    _require(
        values["objects_choice"] in _ETD_OBJECTS_CHOICE,
        "unsupported 'Export measurements for all objects to the "
        f"database?': {values['objects_choice']!r}",
    )
    _require(
        values["allow_overwrite"] in _ETD_OVERWRITE_TO_IF_EXISTS,
        f"unsupported 'Overwrite without warning?': {values['allow_overwrite']!r}",
    )
    objects_choice = _ETD_OBJECTS_CHOICE[values["objects_choice"]]
    return ExportToDatabaseStep(
        sqlite_file=values["sqlite_file"],
        objects_choice=objects_choice,
        objects=(
            _split_list(values["objects_list"]) if objects_choice == "select" else ()
        ),
        if_exists=_ETD_OVERWRITE_TO_IF_EXISTS[values["allow_overwrite"]],
    )


def _translate_measure_object_intensity(raw: _RawModule) -> MeasurementStep:
    _require(
        raw.variable_revision_number == _MEASURE_OBJECT_INTENSITY_REVISION,
        "MeasureObjectIntensity variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MEASURE_OBJECT_INTENSITY_REVISION} is)",
    )
    images_text, objects_text = raw.settings
    return MeasurementStep(
        module_name="MeasureObjectIntensity",
        family="object_intensity",
        images=_split_list(images_text),
        objects=_split_list(objects_text),
    )


def _translate_measure_image_intensity(raw: _RawModule) -> MeasurementStep:
    _require(
        raw.variable_revision_number == _MEASURE_IMAGE_INTENSITY_REVISION,
        "MeasureImageIntensity variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MEASURE_IMAGE_INTENSITY_REVISION} is)",
    )
    images_text, wants_objects, _objects_text, _wants_pct, _pct = raw.settings
    _require(
        not _YES_NO[wants_objects],
        "MeasureImageIntensity with 'Measure the intensity only from "
        "areas enclosed by objects?' = Yes is an explicit, out-of-scope "
        "gap (object-restricted MeasureImageIntensity; see "
        "spec/compatibility.md)",
    )
    return MeasurementStep(
        module_name="MeasureImageIntensity",
        family="image_intensity",
        images=_split_list(images_text),
        objects=(),
    )


def _translate_measure_object_neighbors(raw: _RawModule) -> MeasurementStep:
    _require(
        raw.variable_revision_number == _MEASURE_OBJECT_NEIGHBORS_REVISION,
        "MeasureObjectNeighbors variable_revision_number="
        f"{raw.variable_revision_number} is not supported (only "
        f"{_MEASURE_OBJECT_NEIGHBORS_REVISION} is)",
    )
    (
        object_name,
        neighbors_name,
        distance_method,
        distance,
        wants_excluded_objects,
        *_rest,
    ) = raw.settings
    _require(
        object_name == neighbors_name,
        "MeasureObjectNeighbors is only supported for an object set against "
        f"itself (got {object_name!r} against {neighbors_name!r}); "
        "cross-object-set neighbors are an explicit, out-of-scope gap "
        "(see spec/compatibility.md)",
    )
    _require(
        distance_method
        in ("Adjacent", "Expand until adjacent", "Within a specified distance"),
        f"MeasureObjectNeighbors distance method {distance_method!r} is not supported",
    )
    radius = None
    if distance_method == "Within a specified distance":
        _require(
            re.fullmatch(r"\+?[0-9]+", distance) is not None,
            "Neighbor distance must be a positive integer",
        )
        try:
            radius = int(distance)
        except ValueError as exc:
            raise CppipeError("Neighbor distance does not fit in uint32") from exc
        _require(
            1 <= radius <= _MAX_NEIGHBOR_DISTANCE,
            "Neighbor distance must be a positive integer that fits in uint32",
        )
    return MeasurementStep(
        module_name="MeasureObjectNeighbors",
        family="neighbors",
        images=(),
        objects=(object_name,),
        wants_excluded_objects=_YES_NO[wants_excluded_objects],
        wants_expand=distance_method == "Expand until adjacent",
        neighbors_distance=radius,
    )


_IMAGE_MATH_REVISION = 5
_IMAGE_MATH_HEADER = 9
_IMAGE_MATH_OPERAND = 4
_IMAGE_NDIM = 2


def _translate_image_math(raw: _RawModule) -> ImageMathStep:
    _require(
        raw.variable_revision_number == _IMAGE_MATH_REVISION,
        f"ImageMath requires revision {_IMAGE_MATH_REVISION}",
    )
    values = list(raw.settings)
    operand_count, remainder = divmod(
        len(values) - _IMAGE_MATH_HEADER, _IMAGE_MATH_OPERAND
    )
    _require(
        operand_count >= 2 and not remainder,  # noqa: PLR2004 -- CellProfiler pads to two operands.
        "ImageMath revision 5 requires at least two operand blocks",
    )
    _require(
        values[0] == "None" and values[9] == "Image" and float(values[1]) == 1,
        "ImageMath supports only a single image with operation 'None' and exponent 1",
    )
    flags = values[4:8]
    _require(
        all(value in _YES_NO for value in flags), "ImageMath requires Yes/No flags"
    )
    input_factor, output_factor, addend = map(float, (values[11], values[2], values[3]))
    _require(
        all(isfinite(x) for x in (input_factor, output_factor, addend)),
        "ImageMath factors and addend must be finite",
    )
    _require(
        all(name.strip() and name != "None" for name in (values[10], values[8])),
        "ImageMath requires named input and output images",
    )
    return ImageMathStep(
        values[10],
        values[8],
        input_factor,
        output_factor,
        addend,
        _YES_NO[flags[0]],
        _YES_NO[flags[1]],
        _YES_NO[flags[2]],
    )


_TRANSLATORS = {
    "ImageMath": _translate_image_math,
    "IdentifyPrimaryObjects": _translate_identify_primary_objects,
    "MeasureObjectSizeShape": _translate_measure_object_size_shape,
    "MeasureGranularity": _translate_measure_granularity,
    "MeasureObjectIntensityDistribution": (
        _translate_measure_object_intensity_distribution
    ),
    "MeasureTexture": _translate_measure_texture,
    "MeasureColocalization": _translate_measure_colocalization,
    "ExportToDatabase": _translate_export_to_database,
    "CorrectIlluminationCalculate": _translate_correct_illumination_calculate,
    "CorrectIlluminationApply": _translate_correct_illumination_apply,
    "SaveImages": _translate_save_images,
    "IdentifySecondaryObjects": _translate_identify_secondary_objects,
    "IdentifyTertiaryObjects": _translate_identify_tertiary_objects,
    "MeasureObjectIntensity": _translate_measure_object_intensity,
    "MeasureImageIntensity": _translate_measure_image_intensity,
    "MeasureObjectNeighbors": _translate_measure_object_neighbors,
}


def _cp_converted_pixels(array: np.ndarray) -> np.ndarray:
    """Replicate real CellProfiler's ``Image(..., convert=True)`` pixel
    conversion exactly (cellprofiler_core ``_image.py``'s ``set_image``):
    unsigned integers are downcast to float32 then divided by their
    dtype's max (2^bits - 1); signed integers additionally shift by the
    dtype minimum and clip to [0, 1]; floats are only downcast to
    float32; booleans pass through unchanged. Returns float64 so the
    measurement cores receive their usual dtype."""
    if array.dtype == np.bool_:
        return np.asarray(array, dtype=np.float64)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32).astype(np.float64)
    info = np.iinfo(array.dtype)
    converted = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.signedinteger):
        # Real CP shifts by the dtype minimum, then divides by
        # (2^bits - 1) -- i.e. the full signed range -- and clips.
        converted -= np.float32(info.min)
        converted /= np.float32(info.max - info.min)
        converted = np.clip(converted, 0.0, 1.0)
    else:
        converted /= np.float32(info.max)
    return converted.astype(np.float64)


@dataclass
class Pipeline:
    """A ``.cppipe`` file's translated, ``cp_anywhere``-runnable steps.

    Not a full pipeline executor: file-matching/grouping modules
    (``Images``, ``Metadata``, ``NamesAndTypes``, ``Groups``, ...) are
    listed in :attr:`skipped_modules`, not run -- the caller supplies
    already-loaded image arrays by name via :meth:`run`.
    """

    segmentation_steps: list[SegmentationStep] = field(default_factory=list)
    secondary_steps: list[SecondaryObjectsStep] = field(default_factory=list)
    tertiary_steps: list[TertiaryObjectsStep] = field(default_factory=list)
    measurement_steps: list[MeasurementStep] = field(default_factory=list)
    granularity_steps: list[GranularityStep] = field(default_factory=list)
    intensity_distribution_steps: list[IntensityDistributionStep] = field(
        default_factory=list
    )
    texture_steps: list[TextureStep] = field(default_factory=list)
    colocalization_steps: list[ColocalizationStep] = field(default_factory=list)
    export_steps: list[ExportToDatabaseStep] = field(default_factory=list)
    illumination_calculate_steps: list[IlluminationCalculateStep] = field(
        default_factory=list
    )
    illumination_apply_steps: list[IlluminationApplyStep] = field(default_factory=list)
    save_images_steps: list[SaveImagesStep] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    image_math_steps: list[ImageMathStep] = field(default_factory=list)
    image_step_order: list[str] = field(default_factory=list)

    def prepare_images(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return inputs and prepared planes without modifying the caller's arrays."""
        all_images = dict(images)
        groups = {
            "CorrectIlluminationCalculate": iter(self.illumination_calculate_steps),
            "CorrectIlluminationApply": iter(self.illumination_apply_steps),
            "ImageMath": iter(self.image_math_steps),
        }
        order = self.image_step_order or (
            ["CorrectIlluminationCalculate"] * len(self.illumination_calculate_steps)
            + ["CorrectIlluminationApply"] * len(self.illumination_apply_steps)
            + ["ImageMath"] * len(self.image_math_steps)
        )
        for module in order:
            _prepare_image(all_images, module, next(groups[module]))
        return all_images

    # Keep the execution stages together so label provenance remains visible.
    def run(  # noqa: C901, PLR0912, PLR0915
        self, images: Mapping[str, np.ndarray]
    ) -> tuple[
        dict[str, SegmentationResult],
        dict[str, SecondaryObjectsResult | TertiaryObjectsResult],
        MeasurementResult | None,
    ]:
        """Prepare images, identify primary/secondary/tertiary objects, and measure."""
        segmentations: dict[str, SegmentationResult] = {}
        all_images = self.prepare_images(images)
        # Real CellProfiler's ImageSet holds every image converted by
        # Image(convert=True): integer pixels rescaled to [0, 1] float32,
        # float pixels downcast to float32 (see cellprofiler_core's
        # _image.py set_image). Measurement modules read THAT data, so
        # run() hands them this converted view; segmentation keeps the
        # raw arrays because identify_primary_objects applies the same
        # integer rescale itself while thresholding (its own
        # oracle-verified numerics). Corrected images (above) are
        # already in that converted domain, so re-converting them here
        # is a no-op.
        measurement_images = {
            name: _cp_converted_pixels(array) for name, array in all_images.items()
        }
        for step in self.segmentation_steps:
            if step.input_image not in all_images:
                msg = (
                    f"IdentifyPrimaryObjects needs image {step.input_image!r}, "
                    "not provided to run()"
                )
                raise CppipeError(msg)
            result = identify_primary_objects(
                all_images[step.input_image],
                **step.config,  # ty: ignore[invalid-argument-type]
            )
            segmentations[step.output_object] = result

        secondary_results: dict[
            str, SecondaryObjectsResult | TertiaryObjectsResult
        ] = {}
        for step in self.secondary_steps:
            if step.input_image not in all_images:
                msg = (
                    f"IdentifySecondaryObjects needs image {step.input_image!r}, "
                    "not provided to run()"
                )
                raise CppipeError(msg)
            if step.primary_object not in segmentations:
                msg = (
                    f"IdentifySecondaryObjects needs primary object set "
                    f"{step.primary_object!r}, not produced by a translated "
                    "IdentifyPrimaryObjects step in this same pipeline"
                )
                raise CppipeError(msg)
            primary = segmentations[step.primary_object]
            result = identify_secondary_objects(
                all_images[step.input_image],
                primary.labels,
                primary_labels_in=primary.unedited_labels,
                **step.config,  # ty: ignore[invalid-argument-type]
            )
            secondary_results[step.output_object] = result

        all_object_labels: dict[str, np.ndarray] = {
            name: result.labels for name, result in segmentations.items()
        }
        all_object_labels.update(
            {name: result.labels for name, result in secondary_results.items()}
        )

        for tertiary_step in self.tertiary_steps:
            for name in (tertiary_step.primary_object, tertiary_step.secondary_object):
                if name not in all_object_labels:
                    if name not in all_images:
                        raise CppipeError(
                            f"IdentifyTertiaryObjects needs object set {name!r}, "
                            "not produced or provided to run()"
                        )
                    all_object_labels[name] = all_images[name]
            tertiary_result = identify_tertiary_objects(
                all_object_labels[tertiary_step.primary_object],
                all_object_labels[tertiary_step.secondary_object],
                shrink_primary=tertiary_step.shrink_primary,
            )
            secondary_results[tertiary_step.output_object] = tertiary_result
            all_object_labels[tertiary_step.output_object] = tertiary_result.labels

        if (
            not self.measurement_steps
            and not self.granularity_steps
            and not self.intensity_distribution_steps
            and not self.texture_steps
            and not self.colocalization_steps
        ):
            return segmentations, secondary_results, None

        families = sorted(
            {
                step.family
                for step in self.measurement_steps
                if step.family not in ("neighbors", "object_size_shape")
            }
        )
        measure_images: dict[str, np.ndarray] = {}
        measure_objects: dict[str, np.ndarray] = {}
        for step in self.measurement_steps:
            for name in step.images:
                if name not in all_images:
                    msg = f"{step.module_name} needs image {name!r}, not provided"
                    raise CppipeError(msg)
                measure_images[name] = measurement_images[name]
            for name in step.objects:
                if name in all_object_labels:
                    measure_objects[name] = all_object_labels[name]
                elif name in all_images:
                    # A caller-supplied object set entered via the images
                    # dict is a raw label array, not ImageSet pixel data
                    # -- real CP never converts it, so neither do we.
                    measure_objects[name] = all_images[name]
                else:
                    msg = (
                        f"{step.module_name} needs object set {name!r}, not "
                        "produced by a translated IdentifyPrimaryObjects/"
                        "IdentifySecondaryObjects step or provided to run()"
                    )
                    raise CppipeError(msg)
        if not measure_images:
            # A pipeline with only MeasureObjectNeighbors has no image
            # inputs; measure() still requires at least one named image.
            measure_images = {
                name: next(iter(measurement_images.values())) for name in ("_unused",)
            }
        measured = (
            measure(measure_images, measure_objects, families=families)
            if families
            else None
        )
        masked_diagnostics: dict[str, np.ndarray] = {}
        fixed_scopes = {
            name
            for step in self.measurement_steps
            if step.family == "neighbors" and step.neighbors_distance is not None
            for name in step.objects
        }
        for step in self.measurement_steps:
            if step.family != "neighbors":
                continue
            for name in step.objects:
                diagnostic = None
                if name in segmentations:
                    diagnostic = segmentations[name].small_removed_labels
                elif name in secondary_results:
                    derived = secondary_results[name]
                    diagnostic = (
                        derived.small_removed_labels
                        if isinstance(derived, SecondaryObjectsResult)
                        else derived.labels
                    )
                kept = measure_objects[name]
                if (
                    not step.wants_excluded_objects
                    and name in fixed_scopes
                    and diagnostic is not None
                    and name not in masked_diagnostics
                ):
                    # CP mutates its diagnostic layer; carry that state only
                    # between neighbor steps and leave segmentation outputs intact.
                    masked_diagnostics[name] = np.where(kept, diagnostic, 0)
                if step.wants_excluded_objects or step.neighbors_distance is not None:
                    diagnostic = masked_diagnostics.get(name, diagnostic)
                else:
                    diagnostic = None
                if step.wants_excluded_objects and diagnostic is None:
                    raise CppipeError(
                        f"MeasureObjectNeighbors on {name!r} with excluded "
                        "objects requires a translated segmentation result"
                    )
                result = measure(
                    measure_images,
                    {name: kept},
                    families=["neighbors"],
                    neighbors_scopes=[name],
                    neighbors_expand=[name] if step.wants_expand else None,
                    neighbors_distances={name: step.neighbors_distance}
                    if step.neighbors_distance is not None
                    else None,
                    neighbors_full_objects={name: diagnostic}
                    if diagnostic is not None
                    else None,
                )
                if measured is None:
                    measured = result
                else:
                    for scope, columns in result.features.items():
                        measured.features.setdefault(scope, {}).update(columns)
                    measured.object_ids.update(result.object_ids)
                    measured.relationships.extend(result.relationships)
        # Keep each module's calculation flags and object selection independent.
        # Repeated modules overwrite the same named feature as in CellProfiler.
        for shape_step in self.measurement_steps:
            if shape_step.family != "object_size_shape":
                continue
            shape_result = measure(
                measure_images,
                {name: measure_objects[name] for name in shape_step.objects},
                families=["object_size_shape"],
                size_shape_calculate_zernikes=shape_step.calculate_zernikes,
                size_shape_calculate_advanced=shape_step.calculate_advanced,
            )
            if measured is None:
                measured = shape_result
            else:
                for scope, columns in shape_result.features.items():
                    measured.features.setdefault(scope, {}).update(columns)
                measured.object_ids.update(shape_result.object_ids)
        for granularity_step in self.granularity_steps:
            for image_name in granularity_step.images:
                if image_name not in all_images:
                    msg = (
                        f"MeasureGranularity needs image {image_name!r}, "
                        "not provided to run()"
                    )
                    raise CppipeError(msg)
                object_arrays: dict[str, np.ndarray] = {}
                for obj_name in granularity_step.objects:
                    if obj_name in all_object_labels:
                        object_arrays[obj_name] = all_object_labels[obj_name]
                    else:
                        msg = (
                            f"MeasureGranularity needs object set {obj_name!r}, "
                            "not produced by a translated IdentifyPrimaryObjects/"
                            "IdentifySecondaryObjects/IdentifyTertiaryObjects step"
                        )
                        raise CppipeError(msg)
                granularity_result = measure_granularity(
                    measurement_images[image_name],
                    objects=object_arrays,
                    subsample_size=granularity_step.subsample_size,
                    image_sample_size=granularity_step.image_sample_size,
                    element_size=granularity_step.element_size,
                    granular_spectrum_length=granularity_step.granular_spectrum_length,
                    _float32=True,
                )
                if measured is None:
                    measured = MeasurementResult(
                        image_id="Image", features={}, object_ids={}, relationships=[]
                    )
                image_features = measured.features.setdefault("Image", {})
                for scale_index, value in enumerate(granularity_result.image):
                    image_features[f"Granularity_{scale_index + 1}_{image_name}"] = (
                        float(value)
                    )
                for obj_name, scales in granularity_result.objects.items():
                    obj_features = measured.features.setdefault(obj_name, {})
                    for scale_index, values in enumerate(scales):
                        obj_features[f"Granularity_{scale_index + 1}_{image_name}"] = (
                            values
                        )
                    if obj_name not in measured.object_ids:
                        nobjects = len(scales[0]) if len(scales) else 0
                        measured.object_ids[obj_name] = np.arange(
                            1, nobjects + 1, dtype=np.uint32
                        )
        for intensity_step in self.intensity_distribution_steps:
            for image_name in intensity_step.images:
                if image_name not in all_images:
                    msg = (
                        "MeasureObjectIntensityDistribution needs image "
                        f"{image_name!r}, not provided to run()"
                    )
                    raise CppipeError(msg)
                for obj_name in intensity_step.objects:
                    if obj_name not in all_object_labels:
                        msg = (
                            "MeasureObjectIntensityDistribution needs "
                            f"object set {obj_name!r}, not produced by a "
                            "translated IdentifyPrimaryObjects/"
                            "IdentifySecondaryObjects/"
                            "IdentifyTertiaryObjects step"
                        )
                        raise CppipeError(msg)
                    columns = _run_radial_groups(
                        intensity_step,
                        image_name,
                        measurement_images[image_name],
                        all_object_labels[obj_name],
                        float32=True,
                    )
                    if measured is None:
                        measured = MeasurementResult(
                            image_id="Image",
                            features={},
                            object_ids={},
                            relationships=[],
                        )
                    obj_features = measured.features.setdefault(obj_name, {})
                    nobjects = 0
                    for column, values in columns.items():
                        obj_features[column] = values
                        nobjects = max(nobjects, len(values))
                    if obj_name not in measured.object_ids:
                        measured.object_ids[obj_name] = np.arange(
                            1, nobjects + 1, dtype=np.uint32
                        )
        for texture_step in self.texture_steps:
            scales = list(texture_step.scales)
            for image_name in texture_step.images:
                if image_name not in all_images:
                    msg = (
                        f"MeasureTexture needs image {image_name!r}, not "
                        "provided to run()"
                    )
                    raise CppipeError(msg)
                image = measurement_images[image_name].astype(np.float32)
                wrote_image = False
                for obj_name in texture_step.objects:
                    if obj_name not in all_object_labels:
                        msg = (
                            f"MeasureTexture needs object set {obj_name!r}, "
                            "not produced by a translated "
                            "IdentifyPrimaryObjects/IdentifySecondaryObjects/"
                            "IdentifyTertiaryObjects step"
                        )
                        raise CppipeError(msg)
                    wants_image_now = (
                        texture_step.wants_image_measurements and not wrote_image
                    )
                    texture_result = measure_texture(
                        image,
                        all_object_labels[obj_name],
                        image_name=image_name,
                        gray_levels=texture_step.gray_levels,
                        scales=scales,
                        wants_image_measurements=wants_image_now,
                        wants_object_measurements=(
                            texture_step.wants_object_measurements
                        ),
                    )
                    if measured is None:
                        measured = MeasurementResult(
                            image_id="Image",
                            features={},
                            object_ids={},
                            relationships=[],
                        )
                    if wants_image_now:
                        image_features = measured.features.setdefault("Image", {})
                        for column, value in texture_result.image.items():
                            image_features[column] = float(value)
                        wrote_image = True
                    if texture_step.wants_object_measurements:
                        obj_features = measured.features.setdefault(obj_name, {})
                        nobjects = 0
                        for column, values in texture_result.objects.items():
                            obj_features[column] = values
                            nobjects = max(nobjects, len(values))
                        if obj_name not in measured.object_ids:
                            measured.object_ids[obj_name] = np.arange(
                                1, nobjects + 1, dtype=np.uint32
                            )
                if texture_step.wants_image_measurements and not wrote_image:
                    # No object set paired this image (either "Images" with
                    # an empty objects_list, or "Both"/"Objects" with no
                    # configured objects): still measure the whole image.
                    dummy_labels = np.zeros(image.shape, dtype=np.uint32)
                    texture_result = measure_texture(
                        image,
                        dummy_labels,
                        image_name=image_name,
                        gray_levels=texture_step.gray_levels,
                        scales=scales,
                        wants_image_measurements=True,
                        wants_object_measurements=False,
                    )
                    if measured is None:
                        measured = MeasurementResult(
                            image_id="Image",
                            features={},
                            object_ids={},
                            relationships=[],
                        )
                    image_features = measured.features.setdefault("Image", {})
                    for column, value in texture_result.image.items():
                        image_features[column] = float(value)
        for colocalization_step in self.colocalization_steps:
            step_images = {}
            for image_name in colocalization_step.images:
                if image_name not in all_images:
                    msg = (
                        f"MeasureColocalization needs image {image_name!r}, "
                        "not provided to run()"
                    )
                    raise CppipeError(msg)
                step_images[image_name] = measurement_images[image_name]
            wrote_image = False
            for obj_name in colocalization_step.objects:
                if obj_name not in all_object_labels:
                    msg = (
                        f"MeasureColocalization needs object set {obj_name!r}, "
                        "not produced by a translated "
                        "IdentifyPrimaryObjects/IdentifySecondaryObjects/"
                        "IdentifyTertiaryObjects step"
                    )
                    raise CppipeError(msg)
                # Whole-image columns are identical for every object set,
                # so request them only once per step (the first call).
                wants_image_now = (
                    colocalization_step.wants_image_measurements and not wrote_image
                )
                fast_costes = colocalization_step.fast_costes
                colocalization_result = measure_colocalization(
                    step_images,
                    objects=all_object_labels[obj_name],
                    threshold_percent=colocalization_step.threshold_percent,
                    fast_costes=fast_costes,  # ty: ignore[invalid-argument-type]
                    do_corr_and_slope=colocalization_step.do_corr_and_slope,
                    do_manders=colocalization_step.do_manders,
                    do_rwc=colocalization_step.do_rwc,
                    do_overlap=colocalization_step.do_overlap,
                    do_costes=colocalization_step.do_costes,
                    wants_image_measurements=wants_image_now,
                    wants_object_measurements=True,
                    _float32=True,
                )
                if measured is None:
                    measured = MeasurementResult(
                        image_id="Image",
                        features={},
                        object_ids={},
                        relationships=[],
                    )
                if wants_image_now:
                    image_features = measured.features.setdefault("Image", {})
                    for column, value in colocalization_result.image.items():
                        image_features[column] = value
                    wrote_image = True
                obj_features = measured.features.setdefault(obj_name, {})
                nobjects = 0
                for column, values in colocalization_result.objects.items():
                    obj_features[column] = values
                    nobjects = max(nobjects, len(values))
                if obj_name not in measured.object_ids:
                    measured.object_ids[obj_name] = np.arange(
                        1, nobjects + 1, dtype=np.uint32
                    )
            if colocalization_step.wants_image_measurements and not wrote_image:
                # "Across entire image" with no object set (or one whose
                # step had no objects at all): still measure whole images.
                fast_costes = colocalization_step.fast_costes
                colocalization_result = measure_colocalization(
                    step_images,
                    threshold_percent=colocalization_step.threshold_percent,
                    fast_costes=fast_costes,  # ty: ignore[invalid-argument-type]
                    do_corr_and_slope=colocalization_step.do_corr_and_slope,
                    do_manders=colocalization_step.do_manders,
                    do_rwc=colocalization_step.do_rwc,
                    do_overlap=colocalization_step.do_overlap,
                    do_costes=colocalization_step.do_costes,
                    wants_image_measurements=True,
                    wants_object_measurements=False,
                )
                if measured is None:
                    measured = MeasurementResult(
                        image_id="Image",
                        features={},
                        object_ids={},
                        relationships=[],
                    )
                image_features = measured.features.setdefault("Image", {})
                for column, value in colocalization_result.image.items():
                    image_features[column] = value
        return segmentations, secondary_results, measured


def _prepare_image(
    images: dict[str, np.ndarray],
    module: str,
    step: IlluminationCalculateStep | IlluminationApplyStep | ImageMathStep,
) -> None:
    def pixels(name: str) -> np.ndarray:
        if name not in images:
            msg = f"{module} needs image {name!r}, not provided to run()"
            raise CppipeError(msg)
        return _cp_converted_pixels(images[name])

    if isinstance(step, IlluminationCalculateStep):
        images[step.output_image] = calculate_illumination(
            pixels(step.input_image),
            intensity_choice=step.intensity_choice,
            block_size=step.block_size,
            dilate_objects=step.dilate_objects,
            object_dilation_radius=step.object_dilation_radius,
            smoothing_method=step.smoothing_method,
            automatic_object_width=step.automatic_object_width,
            object_width=step.object_width,
            size_of_smoothing_filter=step.size_of_smoothing_filter,
            rescale_option=step.rescale_option,
        )
    elif isinstance(step, IlluminationApplyStep):
        for entry in step.images:
            images[entry.output_image] = apply_illumination(
                pixels(entry.input_image),
                pixels(entry.illumination_function_image),
                divide_or_subtract=entry.divide_or_subtract,
                truncate_low=step.truncate_low,
                truncate_high=step.truncate_high,
            )
    else:
        images[step.output_image] = _image_math(pixels(step.input_image), step)


def _image_math(pixels: np.ndarray, step: ImageMathStep) -> np.ndarray:
    if pixels.ndim != _IMAGE_NDIM or not pixels.size:
        msg = "ImageMath expects a nonempty 2D image"
        raise CppipeError(msg)
    result = pixels.astype(np.float32, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        if step.input_factor != 1:
            result *= np.float32(step.input_factor)
        if step.output_factor != 1:
            result *= np.float32(step.output_factor)
        if step.addend != 0:
            result += np.float32(step.addend)
    if step.truncate_low:
        result[result < 0] = 0
    if step.truncate_high:
        result[result > 1] = 1
    if step.replace_nan:
        result[np.isnan(result)] = 0
    return result


def _append_step(
    pipeline: Pipeline,
    step: (
        SegmentationStep
        | SecondaryObjectsStep
        | TertiaryObjectsStep
        | GranularityStep
        | IntensityDistributionStep
        | TextureStep
        | ColocalizationStep
        | ExportToDatabaseStep
        | IlluminationCalculateStep
        | IlluminationApplyStep
        | ImageMathStep
        | SaveImagesStep
        | MeasurementStep
    ),
) -> None:
    """Route one translated step onto its matching :class:`Pipeline`
    list, by type -- a plain ``Measure*`` translation (anything not one
    of the dedicated step types) falls through to ``measurement_steps``."""
    if isinstance(
        step,
        (
            IlluminationCalculateStep,
            IlluminationApplyStep,
            ImageMathStep,
            SaveImagesStep,
        ),
    ):
        _append_output_step(pipeline, step)
    elif isinstance(step, SegmentationStep):
        pipeline.segmentation_steps.append(step)
    elif isinstance(step, SecondaryObjectsStep):
        pipeline.secondary_steps.append(step)
    elif isinstance(step, TertiaryObjectsStep):
        pipeline.tertiary_steps.append(step)
    elif isinstance(step, GranularityStep):
        pipeline.granularity_steps.append(step)
    elif isinstance(step, IntensityDistributionStep):
        pipeline.intensity_distribution_steps.append(step)
    elif isinstance(step, TextureStep):
        pipeline.texture_steps.append(step)
    elif isinstance(step, ColocalizationStep):
        pipeline.colocalization_steps.append(step)
    elif isinstance(step, ExportToDatabaseStep):
        pipeline.export_steps.append(step)
    else:
        pipeline.measurement_steps.append(step)


def _append_output_step(
    pipeline: Pipeline,
    step: IlluminationCalculateStep
    | IlluminationApplyStep
    | ImageMathStep
    | SaveImagesStep,
) -> None:
    """The ``CorrectIlluminationCalculate``/``CorrectIlluminationApply``/
    ``SaveImages`` half of :func:`_append_step`'s dispatch, split out to
    keep that function's own branch count under the complexity-lint
    threshold."""
    if isinstance(step, IlluminationCalculateStep):
        pipeline.illumination_calculate_steps.append(step)
        pipeline.image_step_order.append("CorrectIlluminationCalculate")
    elif isinstance(step, IlluminationApplyStep):
        pipeline.illumination_apply_steps.append(step)
        pipeline.image_step_order.append("CorrectIlluminationApply")
    elif isinstance(step, ImageMathStep):
        pipeline.image_math_steps.append(step)
        pipeline.image_step_order.append("ImageMath")
    else:
        pipeline.save_images_steps.append(step)


_CALLER_MANAGED_MODULES = frozenset(
    {
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
        "LoadData",
        "CreateBatchFiles",
        "SaveImages",
        "ExportToDatabase",
        "ExportToSpreadsheet",
    }
)

if TYPE_CHECKING:
    _ModuleStatus = Literal[
        "disabled",
        "caller_managed",
        "unsupported_module",
        "unsupported_settings",
        "supported",
    ]


@dataclass(frozen=True)
class PipelineModuleCheck:
    """Preflight result for one module block, in one-based file order."""

    position: int
    name: str
    status: _ModuleStatus
    detail: str


def _inspect_module(position: int, raw: _RawModule) -> PipelineModuleCheck:
    def result(status: _ModuleStatus, detail: str) -> PipelineModuleCheck:
        return PipelineModuleCheck(position, raw.name, status, detail)

    if not raw.enabled:
        return result("disabled", "Disabled in the source file.")
    translator = _TRANSLATORS.get(raw.name)
    if translator is not None:
        try:
            translator(raw)
        except ValueError as error:
            return result("unsupported_settings", str(error))
    if raw.name in _CALLER_MANAGED_MODULES:
        return result(
            "caller_managed", "The caller handles input, grouping, or output."
        )
    if translator is None:
        return result(
            "unsupported_module", "No translator or executor for this module."
        )
    return result("supported", "Settings translated; execution was not checked.")


def inspect_pipeline(path: str | Path) -> tuple[PipelineModuleCheck, ...]:
    """Report each module's compatibility without reading image files or running it."""
    raw_modules = _parse_text(Path(path).read_text(encoding="utf-8-sig"))
    return tuple(_inspect_module(i, raw) for i, raw in enumerate(raw_modules, 1))


def _run_radial_groups(
    step: IntensityDistributionStep,
    image_name: str,
    image: np.ndarray,
    labels: np.ndarray,
    *,
    float32: bool = False,
) -> dict[str, np.ndarray]:
    first = RadialBinGroup(step.bin_count, step.wants_scaled, step.maximum_radius)
    columns: dict[str, np.ndarray] = {}
    for index, group in enumerate((first, *step.additional_bin_groups)):
        # CellProfiler caches normalized distances from the first bin group.
        # Each group still controls whether it writes the shared Overflow columns.
        result = _measure_intensity_distribution(
            image,
            labels,
            image_name=image_name,
            bin_count=group.bin_count,
            wants_scaled=first.wants_scaled,
            maximum_radius=first.maximum_radius,
            zernikes=step.zernikes if index == 0 else "none",
            zernike_degree=step.zernike_degree,
            emit_overflow=not group.wants_scaled,
            _float32=float32,
        )
        columns.update(result.columns)
    return columns


def load_pipeline(path: str | Path) -> Pipeline:
    """Parse a real CellProfiler ``.cppipe`` text file into a
    :class:`Pipeline`. Raises :class:`CppipeError` for any module
    setting outside ``spec/cppipe-compatibility.md``'s declared scope
    -- never a silent fallback or approximation.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    pipeline = Pipeline()
    for raw in _parse_text(text):
        translator = _TRANSLATORS.get(raw.name)
        if translator is None or not raw.enabled:
            pipeline.skipped_modules.append(raw.name)
            continue
        _append_step(pipeline, translator(raw))
    return pipeline
