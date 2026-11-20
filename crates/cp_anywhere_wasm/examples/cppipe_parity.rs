//! Run the same pipeline inputs passed to the browser parity checker.
#[cfg(target_family = "wasm")]
fn main() {}

#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cp_anywhere::pipeline::{unpack_images, ImageDescriptor, Pipeline};
    use serde::Deserialize;
    use serde_json::{json, Value};

    #[derive(Deserialize)]
    struct Case {
        text: String,
        images: Vec<ImageDescriptor>,
        pixels: Vec<f64>,
    }
    fn run(case: Case) -> Value {
        let pipeline = match Pipeline::parse(&case.text) {
            Ok(pipeline) => pipeline,
            Err(error) => return json!({"error": error}),
        };
        let scaling = case
            .images
            .iter()
            .filter_map(|image| Some((image.name.clone(), image.scaling?)))
            .collect();
        match unpack_images(case.images, &case.pixels)
            .and_then(|images| pipeline.run_with_scaling(images, &scaling))
        {
            Ok(result) => json!({"plan": pipeline.plan(), "result": result.to_json_value()}),
            Err(error) => json!({"plan": pipeline.plan(), "error": error}),
        }
    }
    let cases: Vec<Case> = serde_json::from_reader(std::io::stdin().lock())?;
    let results: Vec<_> = cases.into_iter().map(run).collect();
    serde_json::to_writer(std::io::stdout().lock(), &results)?;
    Ok(())
}
