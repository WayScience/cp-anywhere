//! Strict regression for the pixel dtype used by real CellProfiler pipelines.
#![cfg(feature = "pipeline")]

use cp_anywhere::granularity::{self, Config};
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    shape: [usize; 2],
    subsample_size: f64,
    image_sample_size: f64,
    masked: bool,
    image: Vec<f64>,
    cells: Vec<Vec<f64>>,
}

fn assert_strict(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-10 + 1e-9 * e.abs(),
            "{context} [{i}]: actual {a}, expected {e}"
        );
    }
}

#[test]
fn pipeline_precision_matches_pinned_cellprofiler() {
    let fixture: Oracle = serde_json::from_str(include_str!(
        "../../../fixtures/granularity/float32/oracle.json"
    ))
    .unwrap();
    for case in fixture.cases {
        let [rows, cols] = case.shape;
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        let mut mask = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                pixels.push(((r * 17 + c * 13 + r * c * 7) % 1021 + 23) as f64 / 2048.0);
                labels.push(if (r + c) % 13 == 0 {
                    0
                } else {
                    1 + 2 * u32::from(r >= rows / 2) + u32::from(c >= cols / 2)
                });
                mask.push((r * 3 + c * 7) % 17 > 1);
            }
        }
        let config = Config {
            subsample_size: case.subsample_size,
            image_sample_size: case.image_sample_size,
            element_size: 10,
            granular_spectrum_length: 16,
        };
        let got = granularity::measure_float32(
            &pixels,
            &[("Cells", &labels)],
            case.masked.then_some(mask.as_slice()),
            rows,
            cols,
            &config,
        )
        .unwrap();
        assert_strict(&got.image, &case.image, &format!("{rows}x{cols} Image"));
        let cells = &got.objects["Cells"];
        assert_eq!(cells.len(), case.cells.len());
        for (scale, (a, e)) in cells.iter().zip(&case.cells).enumerate() {
            assert_strict(a, e, &format!("{rows}x{cols} Cells GS{}", scale + 1));
        }
    }
}
