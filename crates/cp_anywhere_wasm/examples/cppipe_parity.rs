//! Run the same pipeline inputs passed to the browser parity checker.
#[cfg(target_family = "wasm")]
fn main() {}

#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cp_anywhere::pipeline::{unpack_inputs, InputDescriptor, Pipeline};
    use serde::Deserialize;
    use serde_json::{json, Value};

    #[derive(Deserialize)]
    struct Case {
        text: String,
        images: Vec<InputDescriptor>,
        pixels: Vec<f64>,
    }
    fn run(case: Case) -> Value {
        let pipeline = match Pipeline::parse(&case.text) {
            Ok(pipeline) => pipeline,
            Err(error) => return json!({"error": error}),
        };
        match unpack_inputs(case.images, &case.pixels)
            .and_then(|inputs| pipeline.run_with_inputs(inputs))
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
