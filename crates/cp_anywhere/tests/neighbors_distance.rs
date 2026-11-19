#![cfg(feature = "pipeline")]

use cp_anywhere::pipeline::{Dtype, Pipeline, Plane};
use cp_anywhere::{Column, ResultTable};
use serde_json::Value;

const PIPELINE: &str = include_str!("../../../fixtures/neighbors_distance/primary_mixed.cppipe");

fn compare(result: ResultTable, expected: &Value) {
    assert_eq!(
        serde_json::to_value(result.object_ids).unwrap(),
        expected["object_ids"]
    );
    assert_eq!(
        serde_json::to_value(result.relationships).unwrap(),
        expected["relationships"]
    );
    assert_eq!(
        result.features.len(),
        expected["features"].as_object().unwrap().len()
    );
    for (scope, columns) in result.features {
        assert_eq!(
            columns.len(),
            expected["features"][&scope].as_object().unwrap().len()
        );
        for (name, column) in columns {
            let values = expected["features"][&scope][&name].as_array().unwrap();
            match column {
                Column::Int(actual) => assert_eq!(
                    serde_json::to_value(actual).unwrap().as_array().unwrap(),
                    values,
                    "{name}"
                ),
                Column::Float(actual) => {
                    assert_eq!(actual.len(), values.len(), "{name}");
                    for (&a, b) in actual.iter().zip(values) {
                        if b["nonfinite"] == "NaN" {
                            assert!(a.is_nan(), "{name}: {a} != NaN");
                        } else {
                            let b = b.as_f64().unwrap();
                            assert!(
                                (a - b).abs() <= 1e-12 + 1e-12 * b.abs(),
                                "{name}: {a} != {b}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn executes_oracle_primary_to_multiple_neighbor_modes() {
    let pipeline = Pipeline::parse(PIPELINE).unwrap();
    let steps = pipeline.plan()["measurement_steps"].as_array().unwrap();
    assert_eq!(steps[0]["neighbors_distance"], 5);
    assert_eq!(steps[1]["neighbors_distance"], 1);
    assert_eq!(steps[2]["neighbors_distance"], 1);
    assert!(steps[3]["neighbors_distance"].is_null());
    assert_eq!(steps[4]["neighbors_distance"], 6);
    let expected: Value = serde_json::from_str(include_str!(
        "../../../fixtures/neighbors_distance/primary_mixed.json"
    ))
    .unwrap();
    let pixels: Vec<Vec<u16>> = serde_json::from_value(expected["pixels"].clone()).unwrap();
    let shape = vec![pixels.len(), pixels[0].len()];
    let output = pipeline
        .run(vec![(
            "DNA".into(),
            Plane {
                shape,
                values: pixels.into_iter().flatten().map(f64::from).collect(),
                dtype: Dtype::Uint16,
            },
        )])
        .unwrap();
    for (name, layers) in expected["outputs"].as_object().unwrap() {
        let labels = |kind| {
            serde_json::from_value::<Vec<Vec<u32>>>(layers[kind].clone())
                .unwrap()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        assert_eq!(output.primary[name].labels, labels("labels"));
        assert_eq!(
            output.primary[name].small_removed_labels,
            labels("full_labels")
        );
    }
    compare(output.measurements.unwrap(), &expected);
}

#[test]
fn reports_invalid_fixed_distances() {
    for distance in ["0", "-1", "1.5", "4294967296"] {
        let text = PIPELINE.replacen(
            "Neighbor distance:5",
            &format!("Neighbor distance:{distance}"),
            1,
        );
        let error = Pipeline::parse(&text).err().unwrap();
        assert!(error.contains("Neighbor distance"), "{error}");
    }
}
