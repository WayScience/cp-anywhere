//! Native Rust side of the cross-runtime batch regression.
#[cfg(target_family = "wasm")]
fn main() {}

#[cfg(not(target_family = "wasm"))]
fn main() {
    native::main();
}

#[cfg(not(target_family = "wasm"))]
mod native {
    use cp_anywhere::segmentation::{build_config, identify_primary_objects, ConfigParams};
    use cp_anywhere::{batch::run_batch, measure, Column, Families, Input};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use std::{fs, num::NonZeroUsize, path::Path};

    #[derive(Deserialize)]
    struct Field {
        image_id: String,
        shape: [usize; 2],
        pixels: String,
    }

    #[derive(Deserialize)]
    struct Fixture {
        fields: Vec<Field>,
        settings: ConfigParams,
        image_name: String,
        object_name: String,
        families: Vec<String>,
    }

    fn encode(value: f64) -> Value {
        if value.is_finite() {
            json!(value)
        } else {
            json!({"nonfinite": if value.is_nan() { "NaN" } else if value.is_sign_positive() { "+Infinity" } else { "-Infinity" }})
        }
    }

    fn run(fixture: &Fixture, field: &Field, directory: &Path) -> Value {
        let bytes = fs::read(directory.join(&field.pixels)).unwrap();
        assert_eq!(bytes.len(), field.shape.iter().product::<usize>() * 8);
        let pixels: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let [rows, cols] = field.shape;
        let config = build_config(&fixture.settings).unwrap();
        let seg = identify_primary_objects(&pixels, None, None, rows, cols, &config).unwrap();
        let has = |name: &str| fixture.families.iter().any(|family| family == name);
        let measured = measure(&Input {
            shape: &field.shape,
            images: &[(&fixture.image_name, &pixels)],
            objects: &[(&fixture.object_name, &seg.labels)],
            mask: None,
            families: Families {
                image_intensity: has("image_intensity"),
                object_intensity: has("object_intensity"),
                neighbors: has("neighbors"),
            },
            size_shape: None,
            neighbors_scopes: &[],
            neighbors_expand: &[],
            neighbors_full_objects: &[],
        })
        .unwrap();
        let features: serde_json::Map<String, Value> = measured
            .features
            .into_iter()
            .map(|(scope, cols)| {
                let cols: serde_json::Map<String, Value> = cols
                    .into_iter()
                    .map(|(name, col)| {
                        let value = match col {
                            Column::Int(v) if scope == "Image" => json!(v[0]),
                            Column::Float(v) if scope == "Image" => encode(v[0]),
                            Column::Int(v) => json!(v),
                            Column::Float(v) => Value::Array(v.into_iter().map(encode).collect()),
                        };
                        (name, value)
                    })
                    .collect();
                (scope, Value::Object(cols))
            })
            .collect();
        let relationships: Vec<_> = measured.relationships.into_iter().map(|(name, a, b)| json!({
        "object_name": name, "first_image_id": field.image_id, "second_image_id": field.image_id,
        "first_object_id": a, "second_object_id": b,
    })).collect();
        json!({
            "imageId": field.image_id, "labels": seg.labels, "smallRemovedLabels": seg.small_removed_labels,
            "uneditedLabels": seg.unedited_labels, "objectCount": seg.object_count, "threshold": seg.threshold,
            "measurements": {"image_id": field.image_id, "features": features,
                "object_ids": measured.object_ids, "relationships": relationships},
        })
    }

    pub(super) fn main() {
        let manifest = std::env::args_os().nth(1).expect("manifest path");
        let path = Path::new(&manifest);
        let fixture: Fixture = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let all = |workers| {
            run_batch(
                &fixture.fields,
                NonZeroUsize::new(workers).unwrap(),
                |field| run(&fixture, field, path.parent().unwrap()),
            )
        };
        let serial = all(1);
        assert_eq!(serial, all(2));
        println!("{}", json!(serial));
    }
}
