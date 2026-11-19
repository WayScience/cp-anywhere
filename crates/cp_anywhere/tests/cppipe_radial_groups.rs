#![cfg(feature = "pipeline")]

use cp_anywhere::pipeline::{Dtype, Pipeline, Plane};
use cp_anywhere::Column;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Input {
    dtype: Dtype,
    values: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Oracle {
    inputs: BTreeMap<String, Input>,
    cytoplasm: Vec<Vec<u32>>,
    object_ids: BTreeMap<String, Vec<i64>>,
    features: BTreeMap<String, BTreeMap<String, Vec<Value>>>,
}

fn fixture(case: &str, extension: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(format!(
        "../../fixtures/cppipe/intensity_distribution/{case}.{extension}"
    )))
    .unwrap()
}

#[test]
fn radial_groups_match_cellprofiler() {
    for case in [
        "jump_groups",
        "scaled_then_unscaled",
        "unscaled_then_scaled",
    ] {
        let oracle: Oracle = serde_json::from_str(&fixture(case, "oracle.json")).unwrap();
        let pipeline = Pipeline::parse(&fixture(case, "cppipe")).unwrap();
        let inputs = oracle
            .inputs
            .into_iter()
            .map(|(name, Input { dtype, values })| {
                let shape = vec![values.len(), values[0].len()];
                let values = values.into_iter().flatten().collect();
                (
                    name,
                    Plane {
                        shape,
                        values,
                        dtype,
                    },
                )
            })
            .collect();
        let run = pipeline.run(inputs).unwrap();
        let cytoplasm: Vec<_> = oracle.cytoplasm.into_iter().flatten().collect();
        assert_eq!(run.tertiary["Cytoplasm"].labels, cytoplasm, "{case}");
        let table = run.measurements.unwrap();
        assert_eq!(table.object_ids, oracle.object_ids, "{case}");
        assert!(table.features.keys().eq(oracle.features.keys()), "{case}");
        for (scope, want) in oracle.features {
            let got = &table.features[&scope];
            assert!(got.keys().eq(want.keys()), "{case}: {scope}");
            for (name, expected) in want {
                let Column::Float(actual) = &got[&name] else {
                    panic!("{case}: {scope}.{name} must be a floating-point column");
                };
                assert_eq!(expected.len(), actual.len(), "{case}: {scope}.{name}");
                for (index, (want, &got)) in expected.iter().zip(actual).enumerate() {
                    let matches = if let Some(want) = want.as_f64() {
                        (want - got).abs() <= 1e-7 + 1e-5 * want.abs()
                    } else {
                        want["nonfinite"] == "NaN" && got.is_nan()
                    };
                    assert!(matches, "{case}: {scope}.{name}[{index}]: {want} != {got}");
                }
            }
        }
    }
}

#[test]
fn rejects_invalid_radial_groups() {
    let text = fixture("scaled_then_unscaled", "cppipe");
    for (from, to, diagnostic) in [
        (
            "Hidden:3\n    Hidden:3",
            "Hidden:2\n    Hidden:3",
            "settings",
        ),
        (
            "Hidden:3\n    Hidden:3",
            "Hidden:3\n    Hidden:-1",
            "group counts",
        ),
        ("Hidden:0", "Hidden:1", "heatmap"),
        (
            "Number of bins:3",
            "Number of bins:2",
            "duplicate bin count",
        ),
        ("Scale the bins?:No", "Scale the bins?:Maybe", "Yes/No"),
        ("Maximum radius:2", "Maximum radius:0", "maximum radius"),
    ] {
        assert!(text.contains(from));
        let input = text.replacen(from, to, 1);
        let Err(error) = Pipeline::parse(&input) else {
            panic!("must reject settings");
        };
        assert!(error.contains(diagnostic), "{error}");
    }
}
