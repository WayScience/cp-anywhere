#![cfg(feature = "pipeline")]

use std::collections::BTreeMap;

use cp_anywhere::outline_overlay::{
    render, Background, Brightness, DisplayMode, LineMode, Outline, OverlayStep, Precision,
};
use cp_anywhere::pipeline::{unpack_images, ImageDescriptor, Pipeline};
use serde::Deserialize;
use serde_json::{json, Value};

const PIPE: &str = include_str!("../../../fixtures/cppipe/overlay_outlines/overlay.cppipe");
const ORACLE: &str = include_str!("../../../fixtures/cppipe/overlay_outlines/overlay.json");

#[test]
fn pipeline_returns_overlay_outputs_from_raw_and_chained_inputs() {
    let fixture: Oracle = serde_json::from_str(ORACLE).unwrap();
    let mut descriptors = Vec::new();
    let mut pixels = Vec::new();
    for (name, plane) in fixture.images {
        descriptors.push(json!({ "name": name, "shape": plane.shape, "dtype": plane.dtype }));
        pixels.extend(plane.values.iter().map(|value| {
            value
                .as_f64()
                .unwrap_or_else(|| f64::from(u8::from(value.as_bool().unwrap())))
        }));
    }
    for (name, plane) in fixture.labels {
        descriptors.push(json!({ "name": name, "shape": plane.shape, "dtype": plane.dtype }));
        pixels.extend(plane.values.into_iter().map(f64::from));
    }
    let descriptors: Vec<ImageDescriptor> =
        serde_json::from_value(Value::Array(descriptors)).unwrap();
    let result = Pipeline::parse(PIPE)
        .unwrap()
        .run(unpack_images(descriptors, &pixels).unwrap())
        .unwrap();
    assert_eq!(result.derived_images.len(), fixture.outputs.len());
    for (name, expected) in fixture.outputs {
        let actual = &result.derived_images[&name];
        assert_eq!(actual.shape, expected.shape, "{name}");
        assert_eq!(actual.values.len(), expected.values.len(), "{name}");
        for (&actual, reference) in actual.values.iter().zip(expected.values) {
            assert!((actual - reference).abs() <= 3.0 * f64::from(f32::EPSILON) * reference.abs());
        }
    }
}

#[derive(Deserialize)]
struct Plane<T> {
    shape: Vec<usize>,
    dtype: String,
    values: Vec<T>,
}

#[derive(Deserialize)]
struct Oracle {
    images: BTreeMap<String, Plane<Value>>,
    labels: BTreeMap<String, Plane<u32>>,
    outputs: BTreeMap<String, Plane<f64>>,
}

#[test]
fn renders_chained_cellprofiler_pixels() {
    let fixture: Oracle = serde_json::from_str(ORACLE).unwrap();
    let plan = Pipeline::parse(PIPE).unwrap();
    let steps: Vec<OverlayStep> =
        serde_json::from_value(plan.plan()["overlay_outlines_steps"].clone()).unwrap();
    let mut images: BTreeMap<String, Plane<f64>> = fixture
        .images
        .into_iter()
        .map(|(name, plane)| {
            (
                name,
                Plane {
                    shape: plane.shape,
                    dtype: plane.dtype,
                    values: plane
                        .values
                        .into_iter()
                        .map(|value| {
                            value
                                .as_f64()
                                .unwrap_or_else(|| u8::from(value.as_bool().unwrap()) as f64)
                        })
                        .collect(),
                },
            )
        })
        .collect();
    assert_eq!(steps.len(), fixture.outputs.len());
    for step in steps {
        let background = step.input_image.as_ref().map(|name| {
            let image = &images[name];
            Background {
                shape: &image.shape,
                values: &image.values,
                precision: if image.dtype == "bool" {
                    Precision::Float64
                } else {
                    Precision::Float32
                },
            }
        });
        let outlines: Vec<_> = step
            .outlines
            .iter()
            .map(|group| {
                let labels = &fixture.labels[&group.objects_name];
                Outline {
                    shape: &labels.shape,
                    labels: &labels.values,
                    color: group.color,
                }
            })
            .collect();
        let output = render(
            background,
            &outlines,
            step.display_mode,
            step.line_mode,
            step.brightness,
        )
        .unwrap();
        let expected = &fixture.outputs[&step.output_image];
        assert_eq!(expected.dtype, "float32");
        assert_eq!(output.shape, expected.shape, "{}", step.output_image);
        assert_eq!(
            output.values.len(),
            expected.values.len(),
            "{}",
            step.output_image
        );
        for (index, (&actual, &reference)) in output.values.iter().zip(&expected.values).enumerate()
        {
            assert_eq!(actual, actual as f32 as f64);
            // NumPy's float32 RGB dot product can fuse terms depending on its build.
            // Only RGB-to-gray arithmetic and the RGB pixels that inherit it
            // have tolerance. Painted RGB pixels and other backgrounds are exact.
            let rounded_gray = step.display_mode == DisplayMode::Grayscale
                || (step.output_image == "GrayToRgb"
                    && expected.values[(index / 3) * 3..(index / 3 + 1) * 3]
                        .windows(2)
                        .all(|pair| pair[0] == pair[1]));
            let tolerance = if rounded_gray {
                3.0 * f64::from(f32::EPSILON) * reference.abs()
            } else {
                0.0
            };
            assert!(
                (actual - reference).abs() <= tolerance,
                "{} sample {index}: {actual} != {reference}",
                step.output_image
            );
        }
        images.insert(
            step.output_image,
            Plane {
                shape: output.shape,
                dtype: "float32".into(),
                values: output.values,
            },
        );
    }
}

#[test]
fn translates_revision_four_positional_settings() {
    let parsed = Pipeline::parse(PIPE).unwrap();
    let steps = parsed.plan()["overlay_outlines_steps"].as_array().unwrap();
    assert_eq!(
        steps[0],
        json!({
            "input_image": null, "output_image": "SurveyColor", "display_mode": "color",
            "line_mode": "inner", "brightness": "max_of_image",
            "outlines": [{"objects_name": "Cells", "color": [0, 128, 0]}],
        })
    );
    assert_eq!(steps[1]["display_mode"], "grayscale");
    assert_eq!(steps[2]["input_image"], "DNA");
    assert_eq!(steps[2]["outlines"][1]["objects_name"], "Edges");
    assert_eq!(steps[5]["outlines"][0]["color"], json!([0, 128, 0]));
    assert_eq!(steps[7]["brightness"], "max_possible");
}

#[test]
fn rejects_invalid_settings_instead_of_silently_skipping() {
    for (from, to, error) in [
        (
            "variable_revision_number:4",
            "variable_revision_number:3",
            "variable_revision_number",
        ),
        (
            "How to outline:Inner",
            "How to outline:Subpixel",
            "line mode",
        ),
        (
            "Outline display mode:Color",
            "Outline display mode:RGBA",
            "display mode",
        ),
        (
            "brightness of outlines:Max of image",
            "brightness of outlines:Other",
            "brightness",
        ),
        (
            "outline color:#008000",
            "outline color:#ééé",
            "outline color",
        ),
        (
            "Select objects to display:Cells",
            "Select objects to display:None",
            "object",
        ),
        ("    Select objects to display:Cells\n", "", "pairs"),
        (
            "Select image on which to display outlines:DNA",
            "Select image on which to display outlines:RgbMax",
            "must be produced before",
        ),
    ] {
        assert!(PIPE.contains(from), "fixture should contain {from}");
        let actual = Pipeline::parse(&PIPE.replacen(from, to, 1)).unwrap_err();
        assert!(
            actual.contains("OverlayOutlines") && actual.contains(error),
            "{actual}"
        );
    }
}

#[test]
fn rejects_unsupported_or_malformed_planes() {
    let options = |background, outlines: &[Outline<'_>]| {
        render(
            background,
            outlines,
            DisplayMode::Color,
            LineMode::Inner,
            Brightness::MaxPossible,
        )
    };
    let good = Outline {
        shape: &[1, 1],
        labels: &[1],
        color: [255, 0, 0],
    };
    assert!(options(None, &[]).unwrap_err().contains("at least one"));
    for (shape, pixels) in [
        (&[1, 2, 4][..], &[0.0; 8][..]),
        (&[1, 1, 1][..], &[0.0][..]),
        (&[1, 2, 1, 3][..], &[0.0; 6][..]),
        (&[0, 1][..], &[][..]),
        (&[1, 2][..], &[0.0][..]),
        (&[usize::MAX, 2][..], &[][..]),
    ] {
        assert!(
            options(
                Some(Background {
                    shape,
                    values: pixels,
                    precision: Precision::Float32
                }),
                &[good]
            )
            .is_err(),
            "{shape:?}"
        );
    }
    for shape in [
        &[1, 1, 1][..],
        &[2, 1][..],
        &[0, 1][..],
        &[usize::MAX, 2][..],
    ] {
        assert!(
            options(None, &[Outline { shape, ..good }]).is_err(),
            "{shape:?}"
        );
    }
}
