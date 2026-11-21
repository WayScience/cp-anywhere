//! JSON adapters for hosts without direct access to Rust result types.

use crate::Column;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{pixel_count, Dtype, PipelineInputs, Plane, RunResult};

/// Interpretation of a packed input plane.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InputRole {
    #[default]
    Image,
    Objects,
    ObjectDiagnostics,
}

/// A named row-major plane within a contiguous input buffer.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub scaling: Option<f64>,
}

/// A packed plane with an explicit image or object-label role.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub scaling: Option<f64>,
    #[serde(default)]
    pub role: InputRole,
}

/// Split a packed image-only buffer without changing its values.
pub fn unpack_images(
    images: Vec<ImageDescriptor>,
    pixels: &[f64],
) -> Result<Vec<(String, Plane)>, String> {
    let descriptors = images.into_iter().map(|image| InputDescriptor {
        name: image.name,
        shape: image.shape,
        dtype: image.dtype,
        scaling: image.scaling,
        role: InputRole::Image,
    });
    Ok(unpack_inputs(descriptors.collect(), pixels)?.images)
}

/// Split a packed input buffer by role without changing the plane values.
pub fn unpack_inputs(
    descriptors: Vec<InputDescriptor>,
    pixels: &[f64],
) -> Result<PipelineInputs, String> {
    let mut rest = pixels;
    let mut inputs = PipelineInputs::default();
    for InputDescriptor {
        name,
        shape,
        dtype,
        scaling,
        role,
    } in descriptors
    {
        let count = pixel_count(&shape).map_err(|error| format!("input {name:?}: {error}"))?;
        let (values, remaining) = rest
            .split_at_checked(count)
            .ok_or_else(|| format!("insufficient pixels for input {name:?}"))?;
        rest = remaining;
        if let Some(scaling) = scaling {
            if role != InputRole::Image {
                return Err(format!(
                    "scaling is only supported for image inputs: {name:?}"
                ));
            }
            inputs.scaling.insert(name.clone(), scaling);
        }
        let planes = match role {
            InputRole::Image => &mut inputs.images,
            InputRole::Objects => &mut inputs.objects,
            InputRole::ObjectDiagnostics => &mut inputs.object_diagnostics,
        };
        planes.push((
            name,
            Plane {
                shape,
                dtype,
                values: values.to_vec(),
            },
        ));
    }
    if !rest.is_empty() {
        return Err("pixel buffer has trailing values".into());
    }
    Ok(inputs)
}

fn float(value: f64) -> Value {
    if value.is_nan() {
        json!("NaN")
    } else if value == f64::INFINITY {
        json!("Infinity")
    } else if value == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(value)
    }
}

fn floats(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(float).collect())
}

impl RunResult {
    /// Encode the result with row-major arrays and named non-finite values.
    /// Absent thresholds/measurements remain JSON null.
    pub fn to_json_value(&self) -> Value {
        let primary: Map<_, _> = self
            .primary
            .iter()
            .map(|(name, result)| {
                (
                    name.clone(),
                    json!({
                        "labels": result.labels,
                        "small_removed_labels": result.small_removed_labels,
                        "unedited_labels": result.unedited_labels,
                        "object_count": result.object_count,
                        "threshold": float(result.threshold),
                        "threshold_map": result.threshold_map.as_deref().map(floats),
                        "guide_threshold": result.guide_threshold.map(float),
                    }),
                )
            })
            .collect();
        let secondary: Map<_, _> = self
            .secondary
            .iter()
            .map(|(name, result)| {
                (
                    name.clone(),
                    json!({
                        "labels": result.labels,
                        "small_removed_labels": result.small_removed_labels,
                        "threshold": result.threshold.map(float),
                    }),
                )
            })
            .collect();
        let tertiary: Map<_, _> = self
            .tertiary
            .iter()
            .map(|(name, result)| (name.clone(), json!({"labels": result.labels})))
            .collect();
        let derived_images: Map<_, _> = self
            .derived_images
            .iter()
            .map(|(name, plane)| {
                (
                    name.clone(),
                    json!({
                        "shape": plane.shape, "values": floats(&plane.values),
                    }),
                )
            })
            .collect();
        let measurements = self.measurements.as_ref().map(|table| {
            let features: Map<_, _> = table.features.iter().map(|(scope, columns)| {
                let encoded: Map<_, _> = columns.iter().map(|(name, column)| (name.clone(), match column {
                    Column::Float(values) if scope == "Image" && values.len() == 1 => float(values[0]),
                    Column::Int(values) if scope == "Image" && values.len() == 1 => json!(values[0]),
                    Column::Float(values) => floats(values),
                    Column::Int(values) => json!(values),
                })).collect();
                (scope.clone(), Value::Object(encoded))
            }).collect();
            let relationships: Vec<_> = table.relationships.iter().map(|(scope, first, second)| json!({
                "object_name": scope,
                "first_image_id": "Image",
                "first_object_id": first,
                "second_image_id": "Image",
                "second_object_id": second,
            })).collect();
            json!({"features": features, "object_ids": table.object_ids, "relationships": relationships})
        });
        json!({
            "skipped": self.skipped,
            "primary": primary,
            "secondary": secondary,
            "tertiary": tertiary,
            "derived_images": derived_images,
            "object_shapes": self.object_shapes,
            "measurements": measurements,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_image_descriptor_still_has_four_fields() {
        let descriptor = ImageDescriptor {
            name: "DNA".into(),
            shape: vec![1, 2],
            dtype: Dtype::Uint16,
            scaling: Some(65535.0),
        };
        let images = unpack_images(vec![descriptor], &[1.0, 65535.0]).unwrap();
        assert_eq!(images[0].0, "DNA");
        assert_eq!(images[0].1.values, [1.0, 65535.0]);
    }
}
