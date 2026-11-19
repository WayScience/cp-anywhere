//! JSON adapters for hosts without direct access to Rust result types.

use crate::Column;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{pixel_count, Dtype, Plane, RunResult};

/// A named row-major plane within a contiguous input buffer.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Split a packed input buffer into named arrays without changing their values.
pub fn unpack_images(
    images: Vec<ImageDescriptor>,
    pixels: &[f64],
) -> Result<Vec<(String, Plane)>, String> {
    let mut rest = pixels;
    let mut result = Vec::with_capacity(images.len());
    for ImageDescriptor { name, shape, dtype } in images {
        let count = pixel_count(&shape)?;
        let (values, remaining) = rest
            .split_at_checked(count)
            .ok_or_else(|| format!("insufficient pixels for image {name:?}"))?;
        rest = remaining;
        result.push((
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
    Ok(result)
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
            "primary": primary,
            "secondary": secondary,
            "tertiary": tertiary,
            "derived_images": derived_images,
            "object_shapes": self.object_shapes,
            "measurements": measurements,
        })
    }
}
