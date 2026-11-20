use std::borrow::Cow;

use serde_json::Value;

use crate::outline_overlay::{render, Background, Outline, OverlayStep, Precision};

use super::{Dtype, Plane, State};

pub(super) fn run(plan: &Value, state: &mut State) -> Result<(), String> {
    for value in plan["overlay_outlines_steps"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let step: OverlayStep = serde_json::from_value(value.clone())
            .map_err(|error| format!("OverlayOutlines: {error}"))?;
        let output = render_step(&step, state)?;
        state.derived.insert(step.output_image.clone());
        if let Some((_, plane)) = state
            .images
            .iter_mut()
            .find(|(name, _)| *name == step.output_image)
        {
            *plane = output;
        } else {
            state.images.push((step.output_image, output));
        }
    }
    Ok(())
}

fn render_step(step: &OverlayStep, state: &State) -> Result<Plane, String> {
    let image = step
        .input_image
        .as_deref()
        .map(|name| {
            state
                .image(name)
                .ok_or_else(|| format!("OverlayOutlines needs image {name:?}"))
        })
        .transpose()?;
    let converted = image.map(Plane::converted);
    let background = image
        .zip(converted.as_deref())
        .map(|(image, values)| Background {
            shape: &image.shape,
            values,
            precision: if matches!(image.dtype, Dtype::Bool) {
                Precision::Float64
            } else {
                Precision::Float32
            },
        });
    let labels = step
        .outlines
        .iter()
        .map(|group| {
            let name = &group.objects_name;
            if let Some((shape, labels)) = state.objects(name) {
                return Ok((shape, Cow::Borrowed(labels)));
            }
            let plane = state
                .image(name)
                .ok_or_else(|| format!("OverlayOutlines needs object set {name:?}"))?;
            let labels = plane
                .labels()
                .map_err(|error| format!("OverlayOutlines object set {name:?}: {error}"))?;
            Ok((plane.shape.as_slice(), Cow::Owned(labels)))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let outlines = step
        .outlines
        .iter()
        .zip(&labels)
        .map(|(group, (shape, labels))| Outline {
            shape,
            labels,
            color: group.color,
        })
        .collect::<Vec<_>>();
    let output = render(
        background,
        &outlines,
        step.display_mode,
        step.line_mode,
        step.brightness,
    )?;
    Ok(Plane {
        shape: output.shape,
        values: output.values,
        dtype: Dtype::Float32,
    })
}
