//! Thin wasm-bindgen binding for the platform-independent measurement core.
//!
//! The JSON input/output shape matches `MeasurementResult.to_json()` in
//! `src/cp_anywhere/api.py` exactly, so the same golden fixtures and the
//! same comparison logic (`scripts/parity/compare_cp_measure.py`) can be
//! reused from JavaScript or re-checked from Python after a browser run
//! saves its output.

use cp_anywhere_core::{measure as core_measure, Column, Families, Input};
use serde::Deserialize;
use serde_json::{Map, Value};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct WasmInput {
    shape: Vec<usize>,
    images: std::collections::BTreeMap<String, Vec<f64>>,
    objects: std::collections::BTreeMap<String, Vec<u32>>,
    mask: Option<Vec<bool>>,
    image_id: String,
    families: Vec<String>,
}

fn encode_float(value: f64) -> Value {
    if value.is_finite() {
        return serde_json::Number::from_f64(value)
            .map(Value::Number)
            .unwrap_or(Value::Null);
    }
    let mut marker = Map::new();
    let text = if value.is_nan() {
        "NaN"
    } else if value > 0.0 {
        "+Infinity"
    } else {
        "-Infinity"
    };
    marker.insert("nonfinite".into(), Value::String(text.into()));
    Value::Object(marker)
}

/// Measure typed-array inputs and return the reference-compatible JSON
/// structure as a string. `shape`, `images`, `objects`, `mask`, and
/// `families` arrive as real typed arrays / plain JS values through
/// wasm-bindgen; only the (small) output is JSON-encoded, matching the
/// Python `MeasurementResult.to_json()` shape byte-for-byte in meaning.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)] // The wasm-bindgen FFI boundary takes
                                     // plain scalars/typed arrays, not a
                                     // struct (see cp_anywhere_python's
                                     // equivalent PyO3 binding).
pub fn measure(
    shape: Vec<u32>,
    image_names: Vec<String>,
    images: Vec<f64>,
    object_names: Vec<String>,
    objects: Vec<u32>,
    mask: Option<Vec<u8>>,
    image_id: String,
    families: Vec<String>,
) -> Result<String, JsValue> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let size: usize = shape.iter().product();
    if image_names.is_empty() || images.len() != image_names.len() * size {
        return Err(JsValue::from_str(
            "image buffer length does not match shape and names",
        ));
    }
    if object_names.is_empty() || objects.len() != object_names.len() * size {
        return Err(JsValue::from_str(
            "object buffer length does not match shape and names",
        ));
    }
    let image_views: Vec<(&str, &[f64])> = image_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), &images[i * size..(i + 1) * size]))
        .collect();
    let object_views: Vec<(&str, &[u32])> = object_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), &objects[i * size..(i + 1) * size]))
        .collect();
    let mask_bools: Option<Vec<bool>> = mask.map(|bytes| bytes.iter().map(|&b| b != 0).collect());
    let mask_slice = mask_bools.as_deref();
    let selected = Families {
        image_intensity: families.iter().any(|f| f == "image_intensity"),
        object_intensity: families.iter().any(|f| f == "object_intensity"),
        neighbors: families.iter().any(|f| f == "neighbors"),
    };
    let input = Input {
        shape: &shape,
        images: &image_views,
        objects: &object_views,
        mask: mask_slice,
        families: selected,
    };
    let result = core_measure(&input).map_err(|e| JsValue::from_str(&e))?;

    let mut features = Map::new();
    for (scope, columns) in result.features {
        let mut group = Map::new();
        for (name, column) in columns {
            let value = match column {
                Column::Float(values) if scope == "Image" => encode_float(values[0]),
                Column::Int(values) if scope == "Image" => Value::from(values[0]),
                Column::Float(values) => {
                    Value::Array(values.into_iter().map(encode_float).collect())
                }
                Column::Int(values) => Value::Array(values.into_iter().map(Value::from).collect()),
            };
            group.insert(name, value);
        }
        features.insert(scope, Value::Object(group));
    }
    let mut object_ids = Map::new();
    for (scope, ids) in result.object_ids {
        object_ids.insert(
            scope,
            Value::Array(ids.into_iter().map(Value::from).collect()),
        );
    }
    let relationships: Vec<Value> = result
        .relationships
        .into_iter()
        .map(|(a, b)| {
            let mut row = Map::new();
            row.insert("first_image_id".into(), Value::String(image_id.clone()));
            row.insert("first_object_id".into(), Value::from(a));
            row.insert("second_image_id".into(), Value::String(image_id.clone()));
            row.insert("second_object_id".into(), Value::from(b));
            Value::Object(row)
        })
        .collect();
    let mut root = Map::new();
    root.insert("image_id".into(), Value::String(image_id));
    root.insert("features".into(), Value::Object(features));
    root.insert("object_ids".into(), Value::Object(object_ids));
    root.insert("relationships".into(), Value::Array(relationships));
    serde_json::to_string(&Value::Object(root)).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// JSON-string overload kept for small inputs / debugging from plain JS
/// without constructing typed arrays by hand.
#[wasm_bindgen(js_name = measureJson)]
pub fn measure_json(input: &str) -> Result<String, JsValue> {
    let parsed: WasmInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let shape_u32: Vec<u32> = parsed.shape.iter().map(|&v| v as u32).collect();
    let image_names: Vec<String> = parsed.images.keys().cloned().collect();
    let images: Vec<f64> = image_names
        .iter()
        .flat_map(|name| parsed.images[name].iter().copied())
        .collect();
    let object_names: Vec<String> = parsed.objects.keys().cloned().collect();
    let objects: Vec<u32> = object_names
        .iter()
        .flat_map(|name| parsed.objects[name].iter().copied())
        .collect();
    let mask = parsed
        .mask
        .map(|bits| bits.into_iter().map(u8::from).collect());
    measure(
        shape_u32,
        image_names,
        images,
        object_names,
        objects,
        mask,
        parsed.image_id,
        parsed.families,
    )
}
