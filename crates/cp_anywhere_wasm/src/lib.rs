//! Thin wasm-bindgen binding for the platform-independent measurement core.
//!
//! The JSON input/output shape matches `MeasurementResult.to_json()` in
//! `src/cp_anywhere/api.py` exactly, so the same golden fixtures and the
//! same comparison logic (`scripts/parity/compare_cp_measure.py`) can be
//! reused from JavaScript or re-checked from Python after a browser run
//! saves its output.
//!
//! Also exposes `identifyPrimaryObjects`, a wasm-bindgen binding for
//! `cp_anywhere_segmentation::identify_primary_objects` -- the same
//! oracle-verified CellProfiler-compatible segmentation the Python
//! binding (`cp_anywhere_python`) exposes, sharing its exact settings
//! surface via `cp_anywhere_segmentation::build_config` (not a separate,
//! driftable reimplementation of the threshold/declump string mapping).

use cp_anywhere_core::{measure as core_measure, Column, Families, Input};
use cp_anywhere_segmentation::{
    build_config, identify_primary_objects as core_segment, ConfigParams,
};
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

/// Result of `identifyPrimaryObjects`. Exposed as typed-array/scalar
/// getters, not a JSON string: the label arrays are the same size as
/// the input image (unlike `measure()`'s much smaller per-object
/// feature output), so JSON-encoding them would be wasteful.
#[wasm_bindgen]
pub struct SegmentationResult {
    labels: Vec<u32>,
    small_removed_labels: Vec<u32>,
    object_count: u32,
    threshold: f64,
    threshold_map: Option<Vec<f64>>,
    guide_threshold: Option<f64>,
}

#[wasm_bindgen]
impl SegmentationResult {
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Vec<u32> {
        self.labels.clone()
    }

    #[wasm_bindgen(getter, js_name = smallRemovedLabels)]
    pub fn small_removed_labels(&self) -> Vec<u32> {
        self.small_removed_labels.clone()
    }

    #[wasm_bindgen(getter, js_name = objectCount)]
    pub fn object_count(&self) -> u32 {
        self.object_count
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Per-pixel corrected thresholds (Adaptive scope only); `null` for
    /// Global scope, matching the Python binding's `threshold_map=None`.
    #[wasm_bindgen(getter, js_name = thresholdMap)]
    pub fn threshold_map(&self) -> Option<Vec<f64>> {
        self.threshold_map.clone()
    }

    #[wasm_bindgen(getter, js_name = guideThreshold)]
    pub fn guide_threshold(&self) -> Option<f64> {
        self.guide_threshold
    }
}

/// Segment typed-array pixel input using the real, oracle-verified
/// CellProfiler-compatible `identify_primary_objects` pipeline -- the
/// identical Rust core the Python binding (`cp_anywhere_python`) calls,
/// through the same `build_config`/`ConfigParams` settings surface (see
/// `cp_anywhere_segmentation::ConfigParams` for the full field list; any
/// field omitted from `config_json` uses CellProfiler's own equivalent
/// default, e.g. `threshold_method: "otsu"`). `pixels`/`mask` are plain
/// typed arrays (the bulk numeric data); `config_json` is a JSON object
/// carrying the many optional settings, matching `measureJson`'s own
/// "typed arrays for bulk data, JSON for small/sparse config" split.
#[wasm_bindgen(js_name = identifyPrimaryObjects)]
pub fn identify_primary_objects(
    pixels: Vec<f64>,
    rows: usize,
    cols: usize,
    mask: Option<Vec<u8>>,
    config_json: &str,
) -> Result<SegmentationResult, JsValue> {
    if pixels.len() != rows * cols {
        return Err(JsValue::from_str(
            "pixels length does not match rows * cols",
        ));
    }
    let params: ConfigParams = if config_json.trim().is_empty() {
        ConfigParams::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid config_json: {e}")))?
    };
    let config = build_config(&params).map_err(|e| JsValue::from_str(&e))?;
    let mask_bools: Option<Vec<bool>> = mask.map(|bytes| bytes.iter().map(|&b| b != 0).collect());
    let result = core_segment(&pixels, None, mask_bools.as_deref(), rows, cols, &config)
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(SegmentationResult {
        labels: result.labels,
        small_removed_labels: result.small_removed_labels,
        object_count: result.object_count,
        threshold: result.threshold,
        threshold_map: result.threshold_map,
        guide_threshold: result.guide_threshold,
    })
}

/// JSON-string overload for `identifyPrimaryObjects`, kept for parity
/// with `measureJson` (debugging / small inputs from plain JS without
/// constructing a typed array by hand). `input` carries `pixels`
/// (plain array, not base64/typed-array-in-JSON, matching `WasmInput`'s
/// own convention for small debug payloads), `rows`, `cols`, optional
/// `mask`, and `config` (the same shape `identifyPrimaryObjects`'s
/// `config_json` takes).
#[derive(Deserialize)]
struct WasmSegmentationInput {
    pixels: Vec<f64>,
    rows: usize,
    cols: usize,
    mask: Option<Vec<bool>>,
    #[serde(default)]
    config: ConfigParams,
}

#[wasm_bindgen(js_name = identifyPrimaryObjectsJson)]
pub fn identify_primary_objects_json(input: &str) -> Result<SegmentationResult, JsValue> {
    let parsed: WasmSegmentationInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if parsed.pixels.len() != parsed.rows * parsed.cols {
        return Err(JsValue::from_str(
            "pixels length does not match rows * cols",
        ));
    }
    let config = build_config(&parsed.config).map_err(|e| JsValue::from_str(&e))?;
    let result = core_segment(
        &parsed.pixels,
        None,
        parsed.mask.as_deref(),
        parsed.rows,
        parsed.cols,
        &config,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    Ok(SegmentationResult {
        labels: result.labels,
        small_removed_labels: result.small_removed_labels,
        object_count: result.object_count,
        threshold: result.threshold,
        threshold_map: result.threshold_map,
        guide_threshold: result.guide_threshold,
    })
}
