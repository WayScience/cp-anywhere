"""Run the group-final illumination part of a CellProfiler .cppipe file.

The caller supplies one ordered group of named images at a time. This is
not a LoadData, metadata, grouping, or SaveImages engine. In particular,
this runner does not expose the running aggregate from earlier cycles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cp_anywhere.cppipe import (
    _ICALC_FIELDS,
    _ICALC_REVISION,
    _SAVE_IMAGES_FIELDS,
    _SAVE_IMAGES_REVISION,
    CppipeError,
    _cp_converted_pixels,
    _parse_text,
    _require,
)
from cp_anywhere.illumination import calculate_illumination_across

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from cp_anywhere.cppipe import _RawModule

_INPUT_MODULES = {"Images", "Metadata", "NamesAndTypes", "Groups", "LoadData"}
_RESIZE_FACTOR = "Resize by a fraction or multiple of the original size"
_RESIZE_OLD = 4
_RESIZE_CURRENT = 5
_RESIZE_OLD_COUNT = 10
_RESIZE_CURRENT_COUNT = 13
_IMAGE_NDIM = 2


@dataclass(frozen=True)
class _Resize:
    source: str
    target: str
    factors: tuple[float, float]  # y, x

    def run(self, pixels: np.ndarray) -> np.ndarray:
        scaled = tuple(
            size * f for size, f in zip(pixels.shape, self.factors, strict=True)
        )
        if not all(isfinite(size) and round(size) > 0 for size in scaled):
            raise ValueError(
                f"Resize {self.source!r} → {self.target!r} has invalid size"
            )
        return _resize_bilinear(pixels, (round(scaled[0]), round(scaled[1])))


def _gaussian_axis(image: np.ndarray, axis: int, sigma: float) -> np.ndarray:
    # scipy.ndimage's Gaussian kernel uses a 4-sigma radius, rounded to
    # the nearest integer, and stores the result of each pass as float32.
    radius = int(4 * sigma + 0.5)
    if radius == 0:
        return image
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    weights /= weights.sum()
    lines = np.moveaxis(image, axis, 0)
    length = lines.shape[0]
    padded = np.pad(lines, ((radius, radius), (0, 0)), mode="symmetric")
    acc = padded[radius : radius + length].astype(np.float64) * weights[radius]
    for i in range(radius):
        pair = padded[i : i + length].astype(np.float64)
        pair += padded[2 * radius - i : 2 * radius - i + length]
        acc += pair * weights[i]
    return np.moveaxis(acc.astype(np.float32), 0, axis)


def _sample_axis(size: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ndimage.zoom with grid_mode=True samples cell centers. Order 1 can
    # access at most one pixel beyond either edge; symmetric repeats it.
    pos = (np.arange(count, dtype=np.float64) + 0.5) * (size / count) - 0.5
    left = np.floor(pos).astype(np.intp)
    return left.clip(0, size - 1), (left + 1).clip(0, size - 1), pos - left


def _resize_bilinear(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """2D float32 path of the pinned oracle's symmetric skimage resize."""
    image = image.astype(np.float32, copy=False)
    filtered = image
    for axis, target in enumerate(shape):
        sigma = max(0.0, (image.shape[axis] / target - 1) / 2)
        filtered = _gaussian_axis(filtered, axis, sigma)
    top, bottom, wy = _sample_axis(image.shape[0], shape[0])
    left, right, wx = _sample_axis(image.shape[1], shape[1])
    acc = np.zeros(shape, dtype=np.float64)
    for ys, yw in ((top, 1 - wy), (bottom, wy)):
        for xs, xw in ((left, 1 - wx), (right, wx)):
            term = filtered[ys[:, None], xs] * yw[:, None]
            acc += term * xw
    # skimage clips to the input range, ignoring NaNs in that range.
    return acc.astype(np.float32).clip(np.nanmin(image), np.nanmax(image))


@dataclass(frozen=True)
class _Across:
    source: str
    target: str
    dilate_objects: bool
    object_dilation_radius: int
    smoothing_method: str
    automatic_object_width: str
    object_width: float
    size_of_smoothing_filter: float
    rescale_option: str

    def run(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if any(image.shape != images[0].shape for image in images):
            raise ValueError(
                f"{self.source!r} images must have the same shape in a group"
            )
        # CellProfiler accumulates workspace Image pixels in float32. The
        # standalone kernel sums in float64; the difference can move a
        # pixel across a median-filter histogram bin. For unmasked Regular
        # mode, pass the already averaged plane to reuse the same native
        # dilation, smoothing and rescaling without that precision change.
        total = np.zeros_like(images[0])
        for image in images:
            total += image
        average = (total.astype(np.float64) / len(images)).astype(np.float32)
        return calculate_illumination_across(
            [average],
            dilate_objects=self.dilate_objects,
            object_dilation_radius=self.object_dilation_radius,
            smoothing_method=self.smoothing_method,
            automatic_object_width=self.automatic_object_width,
            object_width=self.object_width,
            size_of_smoothing_filter=self.size_of_smoothing_filter,
            rescale_option=self.rescale_option,
        ).astype(np.float32)


def _resize_step(raw: _RawModule) -> _Resize:
    settings = raw.settings
    revision = raw.variable_revision_number
    if revision == _RESIZE_OLD:
        _require(
            len(settings) == _RESIZE_OLD_COUNT,
            f"Resize revision {revision} has invalid settings count",
        )
        # 4.2.8 upgrades revision 4 by repeating the old factor for X and Y.
        settings = (
            *settings[:4],
            settings[3],
            "1",
            *settings[4:6],
            "10",
            *settings[6:],
        )
    _require(
        revision in {_RESIZE_OLD, _RESIZE_CURRENT},
        f"Unsupported Resize revision {revision}",
    )
    _require(
        len(settings) == _RESIZE_CURRENT_COUNT,
        f"Resize revision {revision} has invalid settings count",
    )
    source, target, size_method, x, y, _, _, _, _, method, _, _, additional = settings
    _require(
        size_method == _RESIZE_FACTOR,
        f"Resize size method {size_method!r} is not supported",
    )
    _require(
        method == "Bilinear",
        f"Resize interpolation {method!r} is not supported; use Bilinear",
    )
    _require(additional == "0", "Resize additional images are not supported")
    factors = (float(y), float(x))
    _require(
        all(isfinite(f) and f > 0 for f in factors),
        "Resize factors must be finite and positive",
    )
    return _Resize(source, target, factors)


def _across_step(raw: _RawModule) -> _Across:
    revision = raw.variable_revision_number
    _require(
        revision == _ICALC_REVISION, f"Unsupported illumination revision {revision}"
    )
    _require(
        len(raw.settings) == len(_ICALC_FIELDS),
        "CorrectIlluminationCalculate has invalid settings count",
    )
    s: dict[str, str] = dict(zip(_ICALC_FIELDS, raw.settings, strict=True))
    _require(
        s["each_or_all"] == "All: Across cycles",
        "CorrectIlluminationCalculate requires All: Across cycles",
    )
    _require(
        s["intensity_choice"] == "Regular",
        "Plate illumination requires Regular intensity mode",
    )
    _require(
        s["save_average_image"] == s["save_dilated_image"] == "No",
        "CorrectIlluminationCalculate cannot retain average or dilated images",
    )
    choices = {
        "dilate_objects": {"Yes", "No"},
        "rescale_option": {"Yes", "No", "Median"},
        "smoothing_method": {
            "No smoothing",
            "Median Filter",
            "Gaussian Filter",
            "Fit Polynomial",
            "Smooth to Average",
        },
        "automatic_object_width": {"Automatic", "Manually", "Object size"},
    }
    for setting, allowed in choices.items():
        _require(
            s[setting] in allowed,
            f"Unsupported CorrectIlluminationCalculate {setting}={s[setting]!r}",
        )
    step = _Across(
        source=s["input_image"],
        target=s["output_image"],
        dilate_objects=s["dilate_objects"] == "Yes",
        object_dilation_radius=int(s["object_dilation_radius"]),
        smoothing_method=s["smoothing_method"],
        automatic_object_width=s["automatic_object_width"],
        object_width=float(s["object_width"]),
        size_of_smoothing_filter=float(s["size_of_smoothing_filter"]),
        rescale_option=s["rescale_option"],
    )
    _require(
        step.object_dilation_radius >= 0,
        "Illumination requires a nonnegative dilation radius",
    )
    _require(
        all(
            isfinite(v) and v >= 0
            for v in (step.object_width, step.size_of_smoothing_filter)
        ),
        "CorrectIlluminationCalculate filter sizes must be finite and nonnegative",
    )
    return step


def _output_name(raw: _RawModule) -> str:
    fields = _SAVE_IMAGES_FIELDS
    if raw.variable_revision_number == _SAVE_IMAGES_REVISION - 1:
        fields = fields[:-1]  # revision 16 appended TIFF compression
    else:
        _require(
            raw.variable_revision_number == _SAVE_IMAGES_REVISION,
            f"Unsupported SaveImages revision {raw.variable_revision_number}",
        )
    _require(len(raw.settings) == len(fields), "SaveImages has invalid settings count")
    values = dict(zip(fields, raw.settings, strict=True))
    _require(
        values["save_image_or_figure"] == "Image"
        and values["when_to_save"] == "Last cycle",
        "SaveImages declarations must select Image and Last cycle",
    )
    return values["image_name"]


def _steps(text: str) -> Iterator[_Resize | _Across | str]:
    """Translate computation; output declarations are yielded as names."""
    processing = False
    for raw in _parse_text(text):
        if not raw.enabled:
            continue
        if raw.name in _INPUT_MODULES:
            _require(not processing, f"{raw.name} is only supported before computation")
            continue
        processing = True
        try:
            if raw.name == "Resize":
                yield _resize_step(raw)
            elif raw.name == "CorrectIlluminationCalculate":
                yield _across_step(raw)
            elif raw.name == "SaveImages":
                yield _output_name(raw)
            else:
                raise CppipeError(f"Unsupported plate illumination module: {raw.name}")
        except ValueError as error:
            if isinstance(error, CppipeError):
                raise
            raise CppipeError(f"Invalid {raw.name} setting: {error}") from error


def _input_pixels(cycle: Mapping[str, object], index: int, name: str) -> np.ndarray:
    if name not in cycle:
        raise ValueError(f"cycle {index} is missing image {name!r}")
    pixels = np.asarray(cycle[name])
    if (
        pixels.ndim != _IMAGE_NDIM
        or 0 in pixels.shape
        or pixels.dtype.kind not in "iuf"
    ):
        raise ValueError(
            f"cycle {index} image {name!r} must be a nonempty 2D real numeric array"
        )
    # Match Pipeline.run's CellProfiler Image conversion for decoded arrays.
    return _cp_converted_pixels(pixels).astype(np.float32)


@dataclass(frozen=True)
class PlateIlluminationPipeline:
    """Ordered steps and names for a reusable, caller-managed group run.

    Run one group at a time with :meth:`run`. The plan does not read or
    save images and does not keep state between runs.
    """

    required_images: tuple[str, ...]
    output_images: tuple[str, ...]
    _steps: tuple[_Resize | _Across, ...]

    def run(self, image_sets: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
        """Return named float32 illumination images for one nonempty group.

        Each cycle supplies unmasked 2D grayscale arrays by name. Integer
        pixels are scaled across their dtype range. Floats are not rescaled
        on input; pass already-normalized float pixels to preserve custom
        camera scaling. Aggregation uses the supplied cycle order.
        """
        if not image_sets:
            raise ValueError("plate illumination requires a nonempty group")
        values: dict[str, list[np.ndarray] | np.ndarray] = {}
        uses = Counter(step.source for step in self._steps)
        uses.update(self.output_images)
        for step in self._steps:
            source = values.get(step.source)
            if source is None:
                images = (
                    _input_pixels(cycle, index, step.source)
                    for index, cycle in enumerate(image_sets, start=1)
                )
            elif isinstance(source, list):
                images = iter(source)
            else:
                if not isinstance(step, _Resize):
                    raise CppipeError(
                        f"Cannot accumulate group aggregate {step.source!r}"
                    )
                values[step.target] = step.run(source)
                uses[step.source] -= 1
                if not uses[step.source]:
                    del values[step.source]
                continue
            values[step.target] = (
                step.run(list(images))
                if isinstance(step, _Across)
                else [step.run(image) for image in images]
            )
            uses[step.source] -= 1
            if source is not None and not uses[step.source]:
                del values[step.source]
        return {name: np.asarray(values[name]) for name in self.output_images}


def load_plate_illumination_pipeline(path: str | Path) -> PlateIlluminationPipeline:
    """Load a focused illumination plan from a .cppipe file.

    Supports 2D factor-based bilinear Resize (revisions 4 and 5) and
    CorrectIlluminationCalculate (revision 2) in Regular, All: Across cycles mode.
    Resize can run before accumulation or on the final aggregate. A second
    calculation cannot consume an aggregate because CellProfiler would
    process the running result from each cycle instead.

    Disabled modules are ignored. Input modules do not configure loading.
    Last-cycle SaveImages declarations select outputs in file order. If
    none exist, the plan returns the terminal aggregate images in step
    order. The caller controls output file names and formats.
    """
    steps: list[_Resize | _Across] = []
    required: dict[str, None] = {}
    products: dict[str, bool] = {}  # name → whether it holds a group aggregate
    consumed: set[str] = set()
    outputs: dict[str, None] = {}
    for step in _steps(Path(path).read_text(encoding="utf-8")):
        if isinstance(step, str):
            _require(
                products.get(step, False),
                f"SaveImages selects {step!r}, not a group aggregate",
            )
            outputs[step] = None
            continue
        _require(bool(step.source and step.target), "Image names must not be empty")
        if step.source not in products:
            required[step.source] = None
        _require(
            step.target not in products and step.target not in required,
            f"Image {step.target!r} is already defined or used before it is produced",
        )
        is_aggregate = products.get(step.source, False)
        if isinstance(step, _Across):
            _require(
                not is_aggregate, f"Cannot accumulate group aggregate {step.source!r}"
            )
        products[step.target] = is_aggregate or isinstance(step, _Across)
        consumed.add(step.source)
        steps.append(step)
    names = tuple(outputs) or tuple(
        name
        for name, aggregate in products.items()
        if aggregate and name not in consumed
    )
    _require(bool(names), "Plate illumination needs a group aggregate output")
    return PlateIlluminationPipeline(tuple(required), names, tuple(steps))
