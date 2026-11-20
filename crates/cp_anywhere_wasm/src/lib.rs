//! Thin wasm-bindgen binding for the platform-independent measurement core.
//!
//! The JSON input/output shape matches `MeasurementResult.to_json()` in
//! `src/cp_anywhere/api.py` exactly, so the same golden fixtures and the
//! same comparison logic (`scripts/parity/compare_cp_measure.py`) can be
//! reused from JavaScript or re-checked from Python after a browser run
//! saves its output.
//!
//! Also exposes `identifyPrimaryObjects`, a wasm-bindgen binding for
//! `cp_anywhere::segmentation::identify_primary_objects` -- the same
//! oracle-verified CellProfiler-compatible segmentation the Python
//! binding (`cp_anywhere_python`) exposes, sharing its exact settings
//! surface via `cp_anywhere::segmentation::build_config` (not a separate,
//! driftable reimplementation of the threshold/declump string mapping).

mod cppipe;
mod illumination;

use cp_anywhere::segmentation::{
    build_config, identify_primary_objects as core_segment, ConfigParams,
};
use cp_anywhere::{measure_with_image_precision as core_measure, Column, Families, Input};
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
    #[serde(default)]
    neighbors_scopes: Vec<String>,
    #[serde(default)]
    neighbors_expand: Vec<String>,
    #[serde(default)]
    neighbors_distances: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    neighbors_full_objects: std::collections::BTreeMap<String, Vec<u32>>,
    #[serde(default)]
    float32_images: Vec<String>,
    size_shape_calculate_zernikes: Option<bool>,
    size_shape_calculate_advanced: Option<bool>,
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
#[wasm_bindgen(js_name = measureWithOptions)]
#[allow(clippy::too_many_arguments)] // The wasm-bindgen FFI boundary takes
                                     // plain scalars/typed arrays, not a
                                     // struct (see cp_anywhere_python's
                                     // equivalent PyO3 binding).
pub fn measure_with_options(
    shape: Vec<u32>,
    image_names: Vec<String>,
    images: Vec<f64>,
    object_names: Vec<String>,
    objects: Vec<u32>,
    mask: Option<Vec<u8>>,
    image_id: String,
    families: Vec<String>,
    neighbors_scopes: Vec<String>,
    neighbors_expand: Vec<String>,
    full_object_names: Vec<String>,
    full_objects: Vec<u32>,
    size_shape_calculate_zernikes: Option<bool>,
    size_shape_calculate_advanced: Option<bool>,
    neighbors_distances_json: Option<String>,
    float32_images: Option<Vec<String>>,
) -> Result<String, JsValue> {
    let distances: std::collections::BTreeMap<String, u32> = neighbors_distances_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .unwrap_or_default();
    measure_arrays(
        shape,
        image_names,
        images,
        object_names,
        objects,
        mask,
        image_id,
        families,
        neighbors_scopes,
        neighbors_expand,
        full_object_names,
        full_objects,
        size_shape_calculate_zernikes,
        size_shape_calculate_advanced,
        &distances,
        &float32_images.unwrap_or_default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_arrays(
    shape: Vec<u32>,
    image_names: Vec<String>,
    images: Vec<f64>,
    object_names: Vec<String>,
    objects: Vec<u32>,
    mask: Option<Vec<u8>>,
    image_id: String,
    families: Vec<String>,
    neighbors_scopes: Vec<String>,
    neighbors_expand: Vec<String>,
    full_object_names: Vec<String>,
    full_objects: Vec<u32>,
    size_shape_calculate_zernikes: Option<bool>,
    size_shape_calculate_advanced: Option<bool>,
    neighbors_distances: &std::collections::BTreeMap<String, u32>,
    float32_images: &[String],
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
    if full_objects.len() != full_object_names.len() * size {
        return Err(JsValue::from_str(
            "full object buffer length does not match shape and names",
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
    let full_object_views: Vec<(&str, &[u32])> = full_object_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), &full_objects[i * size..(i + 1) * size]))
        .collect();
    let mask_bools: Option<Vec<bool>> = mask.map(|bytes| bytes.iter().map(|&b| b != 0).collect());
    let mask_slice = mask_bools.as_deref();
    let selected = Families {
        image_intensity: families.iter().any(|f| f == "image_intensity"),
        object_intensity: families.iter().any(|f| f == "object_intensity"),
        neighbors: families.iter().any(|f| f == "neighbors"),
    };
    let neighbors_scope_refs: Vec<&str> = neighbors_scopes.iter().map(String::as_str).collect();
    let neighbors_expand_refs: Vec<&str> = neighbors_expand.iter().map(String::as_str).collect();
    let input = Input {
        shape: &shape,
        images: &image_views,
        objects: &object_views,
        mask: mask_slice,
        families: selected,
        neighbors_scopes: &neighbors_scope_refs,
        neighbors_expand: &neighbors_expand_refs,
        size_shape: families
            .iter()
            .any(|family| family == "object_size_shape")
            .then_some(cp_anywhere::SizeShapeOptions {
                calculate_zernikes: size_shape_calculate_zernikes.unwrap_or(true),
                calculate_advanced: size_shape_calculate_advanced.unwrap_or(false),
            }),
        neighbors_full_objects: &full_object_views,
    };
    let distances: Vec<_> = neighbors_distances
        .iter()
        .map(|(name, &r)| (name.as_str(), r))
        .collect();
    let float32_names: Vec<_> = float32_images.iter().map(String::as_str).collect();
    let result =
        core_measure(&input, &distances, &float32_names).map_err(|e| JsValue::from_str(&e))?;

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
        .map(|(scope, a, b)| {
            let mut row = Map::new();
            row.insert("object_name".into(), Value::String(scope));
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

/// Original eight-argument typed-array entry point. Keep existing callers
/// working; diagnostic neighbor layers and shape flags use measureWithOptions.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
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
    measure_with_options(
        shape,
        image_names,
        images,
        object_names,
        objects,
        mask,
        image_id,
        families,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        None,
        None,
        None,
    )
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
    let full_object_names: Vec<String> = parsed.neighbors_full_objects.keys().cloned().collect();
    let full_objects: Vec<u32> = full_object_names
        .iter()
        .flat_map(|name| parsed.neighbors_full_objects[name].iter().copied())
        .collect();
    let mask = parsed
        .mask
        .map(|bits| bits.into_iter().map(u8::from).collect());
    measure_arrays(
        shape_u32,
        image_names,
        images,
        object_names,
        objects,
        mask,
        parsed.image_id,
        parsed.families,
        parsed.neighbors_scopes,
        parsed.neighbors_expand,
        full_object_names,
        full_objects,
        parsed.size_shape_calculate_zernikes,
        parsed.size_shape_calculate_advanced,
        &parsed.neighbors_distances,
        &parsed.float32_images,
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
    unedited_labels: Vec<u32>,
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

    #[wasm_bindgen(getter, js_name = uneditedLabels)]
    pub fn unedited_labels(&self) -> Vec<u32> {
        self.unedited_labels.clone()
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
/// `cp_anywhere::segmentation::ConfigParams` for the full field list; any
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
        unedited_labels: result.unedited_labels,
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
        unedited_labels: result.unedited_labels,
        object_count: result.object_count,
        threshold: result.threshold,
        threshold_map: result.threshold_map,
        guide_threshold: result.guide_threshold,
    })
}

/// Input for `measureGranularityJson`: `pixels`/`mask` are plain arrays
/// (small-input JSON convention, matching `WasmInput`), `objects` maps
/// object-set name to its flattened `u32` label array (same shape as
/// `pixels`), and the remaining fields mirror
/// `cp_anywhere::granularity::Config` with CellProfiler's own
/// `MeasureGranularity` module defaults.
#[derive(Deserialize)]
struct WasmGranularityInput {
    pixels: Vec<f64>,
    #[serde(default)]
    float32: bool,
    rows: usize,
    cols: usize,
    mask: Option<Vec<bool>>,
    #[serde(default)]
    objects: std::collections::BTreeMap<String, Vec<u32>>,
    #[serde(default = "default_subsample_size")]
    subsample_size: f64,
    #[serde(default = "default_subsample_size")]
    image_sample_size: f64,
    #[serde(default = "default_element_size")]
    element_size: usize,
    #[serde(default = "default_granular_spectrum_length")]
    granular_spectrum_length: usize,
}

fn default_subsample_size() -> f64 {
    0.25
}

fn default_element_size() -> usize {
    10
}

fn default_granular_spectrum_length() -> usize {
    16
}

/// Run CellProfiler's 2D granularity spectrum on typed-array/JSON input --
/// the same oracle-verified `cp_anywhere::granularity` core the
/// Python binding (`cp_anywhere_python`) calls. Returns a JSON string
/// `{"image": [...], "objects": {"<name>": [[...], ...]}}` (one inner
/// array per scale, ascending 1-based object order), matching
/// `cp_anywhere.measure_granularity`'s own result shape.
#[wasm_bindgen(js_name = measureGranularityJson)]
pub fn measure_granularity_json(input: &str) -> Result<String, JsValue> {
    let parsed: WasmGranularityInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let size = parsed.rows * parsed.cols;
    if parsed.pixels.len() != size {
        return Err(JsValue::from_str(
            "pixels length does not match rows * cols",
        ));
    }
    let object_pairs: Vec<(&str, &[u32])> = parsed
        .objects
        .iter()
        .map(|(name, labels)| (name.as_str(), labels.as_slice()))
        .collect();
    let config = cp_anywhere::granularity::Config {
        subsample_size: parsed.subsample_size,
        image_sample_size: parsed.image_sample_size,
        element_size: parsed.element_size,
        granular_spectrum_length: parsed.granular_spectrum_length,
    };
    let measure = if parsed.float32 {
        cp_anywhere::granularity::measure_float32
    } else {
        cp_anywhere::granularity::measure
    };
    let result = measure(
        &parsed.pixels,
        &object_pairs,
        parsed.mask.as_deref(),
        parsed.rows,
        parsed.cols,
        &config,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let mut root = Map::new();
    root.insert(
        "image".into(),
        Value::Array(result.image.into_iter().map(encode_float).collect()),
    );
    let mut objects_out = Map::new();
    for (name, scales) in result.objects {
        let scales_json: Vec<Value> = scales
            .into_iter()
            .map(|scale| Value::Array(scale.into_iter().map(encode_float).collect()))
            .collect();
        objects_out.insert(name, Value::Array(scales_json));
    }
    root.insert("objects".into(), Value::Object(objects_out));
    Ok(Value::Object(root).to_string())
}

/// Input for `measureIntensityDistributionJson`: `pixels`/`labels` are
/// plain arrays (small-input JSON convention, matching `WasmInput`), and
/// the remaining fields mirror
/// `cp_anywhere::intensity_distribution::Config` with CellProfiler's
/// own `MeasureObjectIntensityDistribution` module defaults.
#[derive(Deserialize)]
struct WasmIntensityDistributionInput {
    pixels: Vec<f64>,
    #[serde(default)]
    float32: bool,
    labels: Vec<u32>,
    rows: usize,
    cols: usize,
    image_name: String,
    #[serde(default = "default_bin_count")]
    bin_count: usize,
    #[serde(default = "default_wants_scaled")]
    wants_scaled: bool,
    #[serde(default = "default_maximum_radius")]
    maximum_radius: f64,
    #[serde(default = "default_zernikes")]
    zernikes: String,
    #[serde(default = "default_zernike_degree")]
    zernike_degree: u32,
}

fn default_bin_count() -> usize {
    4
}

fn default_wants_scaled() -> bool {
    true
}

fn default_maximum_radius() -> f64 {
    100.0
}

fn default_zernikes() -> String {
    "none".to_string()
}

fn default_zernike_degree() -> u32 {
    9
}

/// Run CellProfiler's 2D radial intensity distribution on typed-array/
/// JSON input -- the same oracle-verified
/// `cp_anywhere::intensity_distribution` core the Python binding
/// (`cp_anywhere_python`) calls. Returns a JSON string
/// `{"<RadialDistribution_... column name>": [...], ...}`, one value per
/// object in ascending 1-based object ID order, matching
/// `cp_anywhere.measure_intensity_distribution`'s own result shape.
#[wasm_bindgen(js_name = measureIntensityDistributionJson)]
pub fn measure_intensity_distribution_json(input: &str) -> Result<String, JsValue> {
    use cp_anywhere::intensity_distribution::{Config, PixelPrecision, ZernikeOption};

    let parsed: WasmIntensityDistributionInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let size = parsed.rows * parsed.cols;
    if parsed.pixels.len() != size || parsed.labels.len() != size {
        return Err(JsValue::from_str(
            "pixels/labels length does not match rows * cols",
        ));
    }
    let zernikes = match parsed.zernikes.as_str() {
        "none" => ZernikeOption::None,
        "magnitudes" => ZernikeOption::Magnitudes {
            degree: parsed.zernike_degree,
        },
        "magnitudes_and_phase" => ZernikeOption::MagnitudesAndPhase {
            degree: parsed.zernike_degree,
        },
        other => {
            return Err(JsValue::from_str(&format!(
                "unsupported zernikes option: {other:?}"
            )))
        }
    };
    let config = Config {
        image_name: parsed.image_name,
        bin_count: parsed.bin_count,
        wants_scaled: parsed.wants_scaled,
        maximum_radius: parsed.maximum_radius,
        zernikes,
    };
    let result = cp_anywhere::intensity_distribution::measure_with_precision(
        &parsed.pixels,
        &parsed.labels,
        parsed.rows,
        parsed.cols,
        &config,
        !config.wants_scaled,
        if parsed.float32 {
            PixelPrecision::Float32
        } else {
            PixelPrecision::Float64
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let mut root = Map::new();
    for (name, values) in result.columns {
        root.insert(
            name,
            Value::Array(values.into_iter().map(encode_float).collect()),
        );
    }
    Ok(Value::Object(root).to_string())
}

/// Input for `measureTextureJson`: `pixels`/`labels` are plain arrays
/// (small-input JSON convention, matching `WasmInput`), and the
/// remaining fields mirror `cp_anywhere::texture::Config` with
/// CellProfiler's own `MeasureTexture` module defaults.
#[derive(Deserialize)]
struct WasmTextureInput {
    pixels: Vec<f64>,
    #[serde(default)]
    float32: bool,
    labels: Vec<u32>,
    rows: usize,
    cols: usize,
    image_name: String,
    #[serde(default = "default_texture_gray_levels")]
    gray_levels: u32,
    scales: Vec<u32>,
    #[serde(default = "default_wants_image_measurements")]
    wants_image_measurements: bool,
    #[serde(default = "default_wants_object_measurements")]
    wants_object_measurements: bool,
}

fn default_texture_gray_levels() -> u32 {
    256
}

fn default_wants_image_measurements() -> bool {
    true
}

fn default_wants_object_measurements() -> bool {
    true
}

/// Run CellProfiler's 2D Haralick texture measurements on typed-array/
/// JSON input -- the same oracle-verified `cp_anywhere::texture`
/// core the Python binding (`cp_anywhere_python`) calls. Returns a JSON
/// string `{"image": {"<Texture_... column name>": <float>, ...},
/// "objects": {"<column name>": [...one per object], ...}}`, matching
/// `cp_anywhere.measure_texture`'s own result shape.
/// Set `float32: true` when `pixels` contains samples from a float32 plane.
#[wasm_bindgen(js_name = measureTextureJson)]
pub fn measure_texture_json(input: &str) -> Result<String, JsValue> {
    use cp_anywhere::texture::{self, Config};

    let parsed: WasmTextureInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let size = parsed.rows * parsed.cols;
    if parsed.pixels.len() != size || parsed.labels.len() != size {
        return Err(JsValue::from_str(
            "pixels/labels length does not match rows * cols",
        ));
    }
    let config = Config {
        gray_levels: parsed.gray_levels,
        scales: parsed.scales,
        wants_image_measurements: parsed.wants_image_measurements,
        wants_object_measurements: parsed.wants_object_measurements,
    };
    let measure = if parsed.float32 {
        texture::measure_float32
    } else {
        texture::measure
    };
    let result = measure(
        &parsed.pixels,
        parsed.rows,
        parsed.cols,
        &parsed.labels,
        &parsed.image_name,
        &config,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let mut image_map = Map::new();
    for (name, value) in result.image {
        image_map.insert(name, encode_float(value));
    }
    let mut objects_map = Map::new();
    for (name, values) in result.objects {
        objects_map.insert(
            name,
            Value::Array(values.into_iter().map(encode_float).collect()),
        );
    }
    let mut root = Map::new();
    root.insert("image".to_string(), Value::Object(image_map));
    root.insert("objects".to_string(), Value::Object(objects_map));
    Ok(Value::Object(root).to_string())
}

/// Input for `measureColocalizationJson`: `images` is an ORDERED array
/// of `{name, pixels}` entries (small-input JSON convention) -- order
/// matters, because real CellProfiler's `get_image_pairs()` yields
/// `(images[i], images[j])` for `i < j`, and the asymmetric Correlation
/// and Slope columns name the pair in exactly that order. Plus the
/// rows/cols shape and `cp_anywhere::colocalization::Config`
/// fields with CellProfiler's own `MeasureColocalization` module
/// defaults.
#[derive(Deserialize)]
struct WasmNamedPixels {
    name: String,
    pixels: Vec<f64>,
}

#[derive(Deserialize)]
struct WasmColocalizationInput {
    images: Vec<WasmNamedPixels>,
    #[serde(default)]
    float32: bool,
    rows: usize,
    cols: usize,
    #[serde(default)]
    masks: Option<std::collections::BTreeMap<String, Vec<bool>>>,
    #[serde(default)]
    labels: Option<Vec<u32>>,
    #[serde(default = "default_threshold_percent")]
    threshold_percent: f64,
    #[serde(default = "default_fast_costes")]
    fast_costes: String,
    #[serde(default = "default_costes_scale_max")]
    costes_scale_max: f64,
    #[serde(default = "default_true")]
    do_corr_and_slope: bool,
    #[serde(default = "default_true")]
    do_manders: bool,
    #[serde(default = "default_true")]
    do_rwc: bool,
    #[serde(default = "default_true")]
    do_overlap: bool,
    #[serde(default = "default_true")]
    do_costes: bool,
    #[serde(default = "default_true")]
    wants_image_measurements: bool,
    #[serde(default = "default_colocalization_wants_objects")]
    wants_object_measurements: bool,
}

fn default_threshold_percent() -> f64 {
    15.0
}

fn default_fast_costes() -> String {
    "faster".to_string()
}

fn default_costes_scale_max() -> f64 {
    255.0
}

fn default_true() -> bool {
    true
}

fn default_colocalization_wants_objects() -> bool {
    false
}

/// Run CellProfiler's 2D pairwise colocalization on JSON input -- the
/// same oracle-verified `cp_anywhere::colocalization` core the
/// Python binding (`cp_anywhere_python`) calls. Returns a JSON string
/// `{"image": {"<Correlation_... column name>": <float>, ...},
/// "objects": {"<column name>": [...one per object], ...}}`, matching
/// `cp_anywhere.measure_colocalization`'s own result shape.
#[wasm_bindgen(js_name = measureColocalizationJson)]
pub fn measure_colocalization_json(input: &str) -> Result<String, JsValue> {
    use cp_anywhere::colocalization::{Config, FastCostesMode};

    let parsed: WasmColocalizationInput =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let size = parsed.rows * parsed.cols;
    if parsed.images.len() < 2 {
        return Err(JsValue::from_str("at least two images are required"));
    }
    let mut image_pairs: Vec<(&str, &[f64])> = Vec::with_capacity(parsed.images.len());
    let mut image_names: Vec<&str> = Vec::with_capacity(parsed.images.len());
    for image in &parsed.images {
        if image.pixels.len() != size {
            return Err(JsValue::from_str(&format!(
                "image '{}' length does not match rows * cols",
                image.name
            )));
        }
        image_names.push(image.name.as_str());
        image_pairs.push((image.name.as_str(), image.pixels.as_slice()));
    }
    if let Some(masks) = &parsed.masks {
        for (name, mask) in masks {
            if mask.len() != size {
                return Err(JsValue::from_str(&format!(
                    "mask '{name}' length does not match rows * cols"
                )));
            }
            if !image_names.contains(&name.as_str()) {
                return Err(JsValue::from_str(&format!(
                    "mask given for unknown image '{name}'"
                )));
            }
        }
    }
    if let Some(labels) = &parsed.labels {
        if labels.len() != size {
            return Err(JsValue::from_str(
                "labels length does not match rows * cols",
            ));
        }
    }
    let fast_costes = match parsed.fast_costes.as_str() {
        "faster" => FastCostesMode::Faster,
        "fast" => FastCostesMode::Fast,
        "accurate" => FastCostesMode::Accurate,
        other => {
            return Err(JsValue::from_str(&format!(
                "unsupported fast_costes option: {other:?}; use 'faster', 'fast', or 'accurate'"
            )))
        }
    };
    let config = Config {
        threshold_percent: parsed.threshold_percent,
        fast_costes,
        costes_scale_max: parsed.costes_scale_max,
        do_corr_and_slope: parsed.do_corr_and_slope,
        do_manders: parsed.do_manders,
        do_rwc: parsed.do_rwc,
        do_overlap: parsed.do_overlap,
        do_costes: parsed.do_costes,
        wants_image_measurements: parsed.wants_image_measurements,
        wants_object_measurements: parsed.wants_object_measurements,
    };
    let mask_pairs: Vec<(&str, &[bool])> = parsed
        .masks
        .as_ref()
        .map(|masks| {
            masks
                .iter()
                .map(|(name, mask)| (name.as_str(), mask.as_slice()))
                .collect()
        })
        .unwrap_or_default();
    let mask_option = if mask_pairs.is_empty() {
        None
    } else {
        Some(mask_pairs.as_slice())
    };
    let measure = if parsed.float32 {
        cp_anywhere::colocalization::measure_prepared_float32
    } else {
        cp_anywhere::colocalization::measure
    };
    let result = measure(
        &image_pairs,
        parsed.rows,
        parsed.cols,
        mask_option,
        parsed.labels.as_deref(),
        &config,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let mut image_map = Map::new();
    for (name, value) in result.image {
        image_map.insert(name, encode_float(value));
    }
    let mut objects_map = Map::new();
    for (name, values) in result.objects {
        objects_map.insert(
            name,
            Value::Array(values.into_iter().map(encode_float).collect()),
        );
    }
    let mut root = Map::new();
    root.insert("image".to_string(), Value::Object(image_map));
    root.insert("objects".to_string(), Value::Object(objects_map));
    Ok(Value::Object(root).to_string())
}
