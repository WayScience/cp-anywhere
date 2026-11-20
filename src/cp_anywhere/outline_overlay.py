"""Settings and NumPy transport for the OverlayOutlines kernel."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cp_anywhere import _native

if TYPE_CHECKING:
    from collections.abc import Mapping

_NAMED_COLORS = {
    "black": "000000",
    "blue": "0000ff",
    "cyan": "00ffff",
    "aqua": "00ffff",
    "gray": "808080",
    "grey": "808080",
    "green": "008000",
    "lime": "00ff00",
    "magenta": "ff00ff",
    "fuchsia": "ff00ff",
    "orange": "ffa500",
    "purple": "800080",
    "red": "ff0000",
    "white": "ffffff",
    "yellow": "ffff00",
}
REVISION = 4
_FIXED_SETTINGS = 6


@dataclass(frozen=True)
class OutlineGroup:
    """One named object set and its RGB outline color."""

    objects_name: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class OverlayStep:
    """One ordered draw operation. A missing background creates a blank image."""

    input_image: str | None
    output_image: str
    display_mode: str
    line_mode: str
    brightness: str
    outlines: tuple[OutlineGroup, ...]

    def render(
        self,
        background: np.ndarray | None,
        labels: Mapping[str, np.ndarray],
        *,
        float64: bool,
    ) -> np.ndarray:
        """Return a new float32 image without changing input pixels or labels."""
        outlines = []
        for group in self.outlines:
            plane = labels[group.objects_name]
            if plane.dtype.kind not in "iu" or (
                plane.size
                and (plane.min() < 0 or plane.max() > np.iinfo(np.uint32).max)
            ):
                msg = (
                    f"object set {group.objects_name!r}: "
                    "labels must contain nonnegative integers that fit in uint32"
                )
                raise ValueError(msg)
            outlines.append((np.ascontiguousarray(plane, dtype=np.uint32), group.color))
        shape, values = _native.render_outline_overlay(
            None
            if background is None
            else np.ascontiguousarray(background, dtype=np.float64),
            outlines,
            (self.display_mode, self.line_mode, self.brightness),
            float64,
        )
        return values.reshape(shape)


def _required_name(name: str, field: str) -> str:
    if not name.strip() or name == "None":
        msg = f"requires a named {field}"
        raise ValueError(msg)
    return name


def _choice(value: str, field: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        msg = f"unsupported {field} {value!r}"
        raise ValueError(msg)
    return value.lower().replace(" ", "_")


def _color(value: str) -> tuple[int, int, int]:
    text = value.replace(" ", "")
    code = text[1:] if text.startswith("#") else _NAMED_COLORS.get(text.lower(), "")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", code):
        msg = f"unsupported outline color {value!r}; use #RRGGBB or a basic named color"
        raise ValueError(msg)
    return int(code[:2], 16), int(code[2:4], 16), int(code[4:], 16)


def translate_overlay(settings: tuple[str, ...]) -> OverlayStep:
    """Translate the positional settings of revision four."""
    if len(settings) < _FIXED_SETTINGS:
        msg = "requires six fixed settings followed by outline color/object pairs"
        raise ValueError(msg)
    blank, source, target, display, brightness, line, *groups = settings
    if not groups or len(groups) % 2:
        msg = "requires one or more outline color/object pairs"
        raise ValueError(msg)
    if blank not in ("Yes", "No"):
        msg = f"blank image requires a Yes/No value, got {blank!r}"
        raise ValueError(msg)
    return OverlayStep(
        None if blank == "Yes" else _required_name(source, "input image"),
        _required_name(target, "output image"),
        _choice(display, "display mode", ("Color", "Grayscale")),
        _choice(line, "line mode", ("Inner", "Outer", "Thick")),
        _choice(brightness, "brightness", ("Max of image", "Max possible")),
        tuple(
            OutlineGroup(_required_name(name, "outline object set"), _color(color))
            for color, name in zip(groups[::2], groups[1::2], strict=True)
        ),
    )


def _image_settings(step: dict) -> list[tuple[str, str]]:
    keys = ("input_image", "output_image", "illumination_function_image", "image_name")
    settings = [(key, step[key]) for key in keys if isinstance(step.get(key), str)]
    for key in ("images", "groups"):
        for entry in step.get(key, ()):
            if isinstance(entry, str):
                settings.append(("input_image", entry))
            elif isinstance(entry, dict):
                settings.extend(_image_settings(entry))
    return settings


def _require_prior(
    producers: dict[str, int], name: str, position: int, kind: str
) -> None:
    if producers.get(name, -1) >= position:
        msg = f"OverlayOutlines: {kind} {name!r} must be produced before its overlay"
        raise ValueError(msg)


def _check_consumers(
    outputs: dict[str, int], module: str, step: dict, position: int
) -> None:
    for _, name in _image_settings(step):
        owner = outputs.get(name)
        if owner is not None and (module != "SaveImages" or owner >= position):
            msg = f"OverlayOutlines: output image {name!r} cannot be used by {module}"
            raise ValueError(msg)


def validate_overlay_order(modules: list[tuple[str, dict]]) -> None:
    """Reject dependencies that would change when drawing overlays last."""
    outputs: dict[str, int] = {}
    image_producers: dict[str, int] = {}
    object_producers: dict[str, int] = {}
    for position, (module, step) in enumerate(modules):
        if module == "OverlayOutlines":
            output = step["output_image"]
            if output in outputs:
                msg = f"OverlayOutlines: duplicate output image {output!r}"
                raise ValueError(msg)
            outputs[output] = position
        image_producers.update(
            (name, position)
            for key, name in _image_settings(step)
            if key == "output_image"
        )
        if "output_object" in step:
            object_producers[step["output_object"]] = position

    for position, (module, step) in enumerate(modules):
        if module == "OverlayOutlines":
            if (source := step["input_image"]) is not None:
                _require_prior(image_producers, source, position, "background")
            for group in step["outlines"]:
                _require_prior(
                    object_producers, group["objects_name"], position, "object set"
                )
        else:
            _check_consumers(outputs, module, step, position)
