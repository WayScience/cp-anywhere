use std::collections::BTreeMap;

use super::{RawModule, Value};
use crate::outline_overlay::{Brightness, DisplayMode, LineMode, OutlineGroup, OverlayStep};

pub(super) fn overlay_outlines(raw: &RawModule<'_>) -> Result<Value, String> {
    let [blank, input, output, display, brightness, line, rest @ ..] = raw.settings.as_slice()
    else {
        return Err(raw.error("requires six fixed settings followed by outline color/object pairs"));
    };
    if rest.is_empty() || rest.len() % 2 != 0 {
        return Err(raw.error("requires one or more outline color/object pairs"));
    }
    let input_image = if raw.yes_no("blank image", blank)? {
        None
    } else {
        Some(name(raw, "input image", input)?)
    };
    let output_image = name(raw, "output image", output)?;
    let display_mode = match *display {
        "Color" => DisplayMode::Color,
        "Grayscale" => DisplayMode::Grayscale,
        _ => return Err(raw.error(format_args!("unsupported display mode {display:?}"))),
    };
    let brightness = match *brightness {
        "Max of image" => Brightness::MaxOfImage,
        "Max possible" => Brightness::MaxPossible,
        _ => return Err(raw.error(format_args!("unsupported brightness {brightness:?}"))),
    };
    let line_mode = match *line {
        "Inner" => LineMode::Inner,
        "Outer" => LineMode::Outer,
        "Thick" => LineMode::Thick,
        _ => return Err(raw.error(format_args!("unsupported line mode {line:?}"))),
    };
    let outlines = rest
        .chunks_exact(2)
        .map(|pair| {
            let color = color_rgb(pair[0]).ok_or_else(|| {
                raw.error(format_args!(
                    "unsupported outline color {:?}; use #RRGGBB or a basic named color",
                    pair[0]
                ))
            })?;
            Ok(OutlineGroup {
                objects_name: name(raw, "outline object set", pair[1])?,
                color,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    serde_json::to_value(OverlayStep {
        input_image,
        output_image,
        display_mode,
        brightness,
        line_mode,
        outlines,
    })
    .map_err(|e| raw.error(e))
}

fn name(raw: &RawModule<'_>, field: &str, value: &str) -> Result<String, String> {
    if value.trim().is_empty() || value == "None" {
        Err(raw.error(format_args!("requires a named {field}")))
    } else {
        Ok(value.to_owned())
    }
}

fn color_rgb(value: &str) -> Option<[u8; 3]> {
    let value = value.replace(' ', "");
    if let Some(hex) = value.strip_prefix('#') {
        if hex.len() != 6 || !hex.bytes().all(|digit| digit.is_ascii_hexdigit()) {
            return None;
        }
        return Some([
            u8::from_str_radix(&hex[0..2], 16).ok()?,
            u8::from_str_radix(&hex[2..4], 16).ok()?,
            u8::from_str_radix(&hex[4..6], 16).ok()?,
        ]);
    }
    Some(match value.to_ascii_lowercase().as_str() {
        "black" => [0, 0, 0],
        "blue" => [0, 0, 255],
        "cyan" | "aqua" => [0, 255, 255],
        "gray" | "grey" => [128, 128, 128],
        "green" => [0, 128, 0],
        "lime" => [0, 255, 0],
        "magenta" | "fuchsia" => [255, 0, 255],
        "orange" => [255, 165, 0],
        "purple" => [128, 0, 128],
        "red" => [255, 0, 0],
        "white" => [255, 255, 255],
        "yellow" => [255, 255, 0],
        _ => return None,
    })
}

const MODULE: &str = "OverlayOutlines";

// Illumination and quality can use nested image groups.
fn image_settings<'a>(step: &'a Value, visit: &mut impl FnMut(&str, &'a str)) {
    for key in [
        "input_image",
        "output_image",
        "illumination_function_image",
        "image_name",
    ] {
        if let Some(name) = step[key].as_str() {
            visit(key, name);
        }
    }
    for key in ["images", "groups"] {
        for entry in step[key].as_array().into_iter().flatten() {
            if let Some(name) = entry.as_str() {
                visit("input_image", name);
            } else {
                image_settings(entry, visit);
            }
        }
    }
}

/// The runner draws overlays last. Reject dependencies that this changes.
pub(super) fn validate_order<'a>(
    modules: impl Iterator<Item = (&'a str, &'a Value)> + Clone,
) -> Result<(), String> {
    let mut image_producers = BTreeMap::new();
    let mut object_producers = BTreeMap::new();
    let mut overlay_outputs = BTreeMap::new();
    for (position, (module, step)) in modules.clone().enumerate() {
        if module == MODULE {
            let output = step["output_image"].as_str().expect("translated overlay");
            if overlay_outputs.insert(output, position).is_some() {
                return Err(format!("{MODULE}: duplicate output image {output:?}"));
            }
        }
        image_settings(step, &mut |key, name| {
            if key == "output_image" {
                image_producers.insert(name, position);
            }
        });
        if let Some(name) = step["output_object"].as_str() {
            object_producers.insert(name, position);
        }
    }
    for (position, (module, step)) in modules.enumerate() {
        if module == MODULE {
            if let Some(name) = step["input_image"].as_str() {
                require_prior(&image_producers, name, position, "background")?;
            }
            for group in step["outlines"].as_array().into_iter().flatten() {
                let name = group["objects_name"].as_str().expect("translated outline");
                require_prior(&object_producers, name, position, "object set")?;
            }
        } else {
            let mut conflict = None;
            image_settings(step, &mut |_, name| {
                if overlay_outputs
                    .get(name)
                    .is_some_and(|&source| module != "SaveImages" || source >= position)
                    && conflict.is_none()
                {
                    conflict = Some(name);
                }
            });
            if let Some(name) = conflict {
                return Err(format!(
                    "{MODULE}: output image {name:?} cannot be used by {module}"
                ));
            }
        }
    }
    Ok(())
}

fn require_prior(
    producers: &BTreeMap<&str, usize>,
    name: &str,
    position: usize,
    kind: &str,
) -> Result<(), String> {
    if producers
        .get(name)
        .is_some_and(|&source| source >= position)
    {
        return Err(format!(
            "{MODULE}: {kind} {name:?} must be produced before its overlay"
        ));
    }
    Ok(())
}
