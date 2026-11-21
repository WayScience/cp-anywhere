//! Browser entry point for the same pipeline executor used by native Rust.
use cp_anywhere::pipeline::{unpack_inputs, InputDescriptor, Pipeline};
use wasm_bindgen::prelude::*;

/// Translate original `.cppipe` text to an inspectable plan.
#[wasm_bindgen]
pub fn compile_cppipe(text: &str) -> Result<String, JsValue> {
    Pipeline::parse(text)
        .map(|pipeline| pipeline.plan().to_string())
        .map_err(|error| JsValue::from_str(&error))
}

/// Run original `.cppipe` text over named, decoded row-major planes.
/// `images_json` describes the name, shape, dtype, and optional role of each plane.
/// Use `image` (default), `objects` (retained), or `object_diagnostics` (small-removed).
/// Only images accept scaling. `pixels` concatenates the planes in descriptor order.
#[wasm_bindgen]
pub fn run_cppipe(text: &str, images_json: &str, pixels: &[f64]) -> Result<String, JsValue> {
    CppipePipeline::new(text)?.run(images_json, pixels)
}

/// A parsed pipeline that can process multiple image sets in one WASM instance.
#[wasm_bindgen]
pub struct CppipePipeline(Pipeline);

#[wasm_bindgen]
impl CppipePipeline {
    #[wasm_bindgen(constructor)]
    pub fn new(text: &str) -> Result<CppipePipeline, JsValue> {
        Pipeline::parse(text)
            .map(Self)
            .map_err(|error| JsValue::from_str(&error))
    }

    /// Inspect translated settings, including explicit save/export descriptors.
    pub fn plan(&self) -> String {
        self.0.plan().to_string()
    }

    /// Process one field. This call does not modify the parsed pipeline.
    pub fn run(&self, images_json: &str, pixels: &[f64]) -> Result<String, JsValue> {
        run(&self.0, images_json, pixels).map_err(|error| JsValue::from_str(&error))
    }
}

fn run(pipeline: &Pipeline, images_json: &str, pixels: &[f64]) -> Result<String, String> {
    if !images_json.trim_start().starts_with('[') {
        return Err("image descriptors must be a JSON array".into());
    }
    let images: Vec<InputDescriptor> = serde_json::from_str(images_json)
        .map_err(|error| format!("invalid image descriptors: {error}"))?;
    let output = pipeline.run_with_inputs(unpack_inputs(images, pixels)?)?;
    Ok(output.to_json_value().to_string())
}
