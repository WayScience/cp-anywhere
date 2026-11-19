//! Typed-array illumination bindings; numerical calculation stays in the core.

use cp_anywhere::illumination::{
    self, Config, FilterSizeMethod, IntensityChoice, RescaleOption, SmoothingMethod,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CalculateConfig {
    intensity_choice: String,
    block_size: usize,
    dilate_objects: bool,
    object_dilation_radius: usize,
    smoothing_method: String,
    automatic_object_width: String,
    object_width: f64,
    size_of_smoothing_filter: f64,
    rescale_option: String,
}

impl Default for CalculateConfig {
    fn default() -> Self {
        Self {
            intensity_choice: "Regular".into(),
            block_size: 60,
            dilate_objects: false,
            object_dilation_radius: 1,
            smoothing_method: "Gaussian Filter".into(),
            automatic_object_width: "Automatic".into(),
            object_width: 10.0,
            size_of_smoothing_filter: 10.0,
            rescale_option: "Yes".into(),
        }
    }
}

impl CalculateConfig {
    fn into_core(self) -> Result<Config, JsValue> {
        let intensity_choice = match self.intensity_choice.as_str() {
            "Regular" => IntensityChoice::Regular,
            "Background" => IntensityChoice::Background,
            _ => return Err(JsValue::from_str("unknown intensity_choice")),
        };
        let smoothing_method = match self.smoothing_method.as_str() {
            "No smoothing" => SmoothingMethod::None,
            "Median Filter" => SmoothingMethod::MedianFilter,
            "Gaussian Filter" => SmoothingMethod::GaussianFilter,
            "Fit Polynomial" => SmoothingMethod::FitPolynomial,
            "Smooth to Average" => SmoothingMethod::ToAverage,
            "Convex Hull" | "Splines" => {
                return Err(JsValue::from_str(&format!(
                    "smoothing_method {:?} is not yet implemented",
                    self.smoothing_method
                )));
            }
            _ => return Err(JsValue::from_str("unknown smoothing_method")),
        };
        let automatic_object_width = match self.automatic_object_width.as_str() {
            "Automatic" => FilterSizeMethod::Automatic,
            "Object size" => FilterSizeMethod::ObjectSize,
            "Manually" => FilterSizeMethod::Manually,
            _ => return Err(JsValue::from_str("unknown automatic_object_width")),
        };
        let rescale_option = match self.rescale_option.as_str() {
            "Yes" => RescaleOption::Yes,
            "No" => RescaleOption::No,
            "Median" => RescaleOption::Median,
            _ => return Err(JsValue::from_str("unknown rescale_option")),
        };
        Ok(Config {
            intensity_choice,
            block_size: self.block_size,
            dilate_objects: self.dilate_objects,
            object_dilation_radius: self.object_dilation_radius,
            smoothing_method,
            automatic_object_width,
            object_width: self.object_width,
            size_of_smoothing_filter: self.size_of_smoothing_filter,
            rescale_option,
        })
    }
}

// Accept f64 at the JS boundary so fractional, negative, nonfinite and
// oversized dimensions are rejected, not silently coerced to unsigned ints.
fn shape(rows: f64, cols: f64, len: usize) -> Result<(usize, usize), JsValue> {
    if [rows, cols]
        .iter()
        .any(|v| !v.is_finite() || *v < 1.0 || v.fract() != 0.0 || *v > u32::MAX as f64)
    {
        return Err(JsValue::from_str(
            "rows and cols must be positive integer 2D dimensions within u32 range",
        ));
    }
    let (rows, cols) = (rows as usize, cols as usize);
    if rows.checked_mul(cols) != Some(len) {
        return Err(JsValue::from_str(
            "pixel buffer length does not match rows*cols",
        ));
    }
    Ok((rows, cols))
}

fn parse_config<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, JsValue> {
    // Serde's derived struct visitor also accepts arrays; the public config
    // contract accepts JSON objects only, including an empty object for defaults.
    if !json.trim_start().starts_with('{') {
        return Err(JsValue::from_str(
            "illumination config must be a JSON object",
        ));
    }
    serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("invalid illumination config: {e}")))
}

/// Calculate one 2D image's illumination function (CellProfiler "Each" mode).
/// Any nonzero mask byte is significant; absent mask preserves has_mask=False.
#[wasm_bindgen(js_name = calculateIllumination)]
pub fn calculate_illumination(
    pixels: &[f64],
    rows: f64,
    cols: f64,
    mask: Option<Vec<u8>>,
    config_json: &str,
) -> Result<Vec<f64>, JsValue> {
    let (rows, cols) = shape(rows, cols, pixels.len())?;
    if mask.as_ref().is_some_and(|m| m.len() != pixels.len()) {
        return Err(JsValue::from_str("mask length must match rows*cols"));
    }
    let config: CalculateConfig = parse_config(config_json)?;
    let config = config.into_core()?;
    let mask: Option<Vec<bool>> = mask.map(|m| m.into_iter().map(|v| v != 0).collect());
    illumination::calculate_each(pixels, mask.as_deref(), rows, cols, &config)
        .map_err(|e| JsValue::from_str(&e))
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ApplyConfig {
    divide_or_subtract: String,
    truncate_low: bool,
    truncate_high: bool,
}

impl Default for ApplyConfig {
    fn default() -> Self {
        Self {
            divide_or_subtract: "Divide".into(),
            truncate_low: true,
            truncate_high: true,
        }
    }
}

/// Apply a same-shaped 2D illumination function without epsilon substitution.
/// Comparisons deliberately preserve NaN and signed zero, unlike f64::max/min.
#[wasm_bindgen(js_name = applyIllumination)]
pub fn apply_illumination(
    pixels: &[f64],
    function: &[f64],
    rows: f64,
    cols: f64,
    config_json: &str,
) -> Result<Vec<f64>, JsValue> {
    shape(rows, cols, pixels.len())?;
    if function.len() != pixels.len() {
        return Err(JsValue::from_str(
            "image and illumination function must have equal dimensions (buffer lengths)",
        ));
    }
    let config: ApplyConfig = parse_config(config_json)?;
    let subtract = match config.divide_or_subtract.as_str() {
        "Divide" => false,
        "Subtract" => true,
        _ => {
            return Err(JsValue::from_str(
                "divide_or_subtract must be 'Divide' or 'Subtract'",
            ))
        }
    };
    Ok(pixels
        .iter()
        .zip(function)
        .map(|(&pixel, &illum)| {
            let mut value = if subtract {
                pixel - illum
            } else {
                pixel / illum
            };
            // Real CP subtraction always clips low, even with truncate_low=False.
            if (subtract || config.truncate_low) && value < 0.0 {
                value = 0.0;
            }
            if config.truncate_high && value > 1.0 {
                value = 1.0;
            }
            // Only the output Image construction introduces this round trip.
            value as f32 as f64
        })
        .collect())
}
