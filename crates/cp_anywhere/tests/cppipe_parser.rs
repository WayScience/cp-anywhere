#![cfg(feature = "pipeline")]

use cp_anywhere::pipeline::Pipeline;
use serde_json::{json, Value};
use std::{fs, path::Path};

fn parse(text: &str) -> Result<Value, String> {
    Pipeline::parse(text).map(|pipeline| pipeline.plan().clone())
}

fn fixture(path: &str) -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(repo.join(path)).unwrap_or_else(|err| panic!("{path}: {err}"))
}

const MANUAL: &str = "fixtures/cppipe/manual.cppipe";
const ILLUMINATION: &str = "fixtures/cppipe/illumination/illumination_calculate_apply.cppipe";
const SAVE: &str = "fixtures/cppipe/illumination/save_images.cppipe";
const EXPORT: &str = "fixtures/cppipe/export_to_database/select_objects_never_overwrite.cppipe";

#[test]
fn matches_python_dataclass_for_real_pipelines() {
    // Each snapshot is json.dumps(dataclasses.asdict(load_pipeline(Path(path)))).
    let expected: Value =
        serde_json::from_str(include_str!("data/cppipe_plan_oracle.json")).unwrap();
    for (path, expected) in expected.as_object().unwrap() {
        let actual = parse(&fixture(path)).unwrap_or_else(|err| panic!("{path}: {err}"));
        assert_eq!(&actual, expected, "{path}");
    }
}

#[test]
fn header_and_block_parsing_preserves_positional_values_and_skipped_order() {
    let original = fixture(SAVE);
    let text = original.replace("ModuleCount:1", "ModuleCount:4").replace(
        "Enter single file name:dna_corrected",
        "Renamed label:  dna:corrected, x|y  ",
    ) + "\n\nImages:[variable_revision_number:4]\n    An ignored setting:x\n\nMystery:[variable_revision_number:19]\n\nImages:[variable_revision_number:1]\n\nIgnored even without a revision";
    let actual = parse(&text.replace('\n', "\r\n")).unwrap();
    assert_eq!(
        actual["skipped_modules"],
        json!(["Images", "Mystery", "Images"])
    );
    assert_eq!(
        actual["save_images_steps"][0]["file_name"],
        "  dna:corrected, x|y  "
    );
    assert_eq!(parse(&text.replace('\n', "\r")).unwrap(), actual);
    assert_eq!(parse(&text.replace('\n', "\u{2028}")).unwrap(), actual);
    assert_eq!(
        parse(&original.replace("ModuleCount:1\n", "")).unwrap(),
        parse(&original).unwrap()
    );
    assert_eq!(
        parse(&original.replace("ModuleCount:1", "ModuleCount:2")).unwrap(),
        parse(&original).unwrap()
    );
    let empty = parse(&original.replace("ModuleCount:1", "ModuleCount:0")).unwrap();
    assert_eq!(empty.as_object().unwrap().len(), 15);
    assert!(empty.as_object().unwrap().values().all(|v| *v == json!([])));
}

#[test]
fn rejects_bad_headers_and_bad_setting_counts() {
    assert!(parse("not a pipeline")
        .unwrap_err()
        .contains("missing header"));
    assert!(parse("CellProfiler Pipeline:\nModuleCount:abc\n\n")
        .unwrap_err()
        .contains("ModuleCount"));
    assert!(parse("CellProfiler Pipeline:\n\nUnknown:[]\n")
        .unwrap_err()
        .contains("variable_revision_number"));
    let manual = fixture(MANUAL);
    let text = manual.replace("    Select objects to measure:Cells\n", "");
    assert!(parse(&text).unwrap_err().contains("MeasureObjectIntensity"));
    let text = fixture(ILLUMINATION).replace("    Select the illumination function:Illum\n", "");
    assert!(parse(&text).unwrap_err().contains("image-group count"));
}

#[test]
fn rejects_known_unsupported_settings_in_real_blocks() {
    let cases = [
        (
            MANUAL,
            "variable_revision_number:15",
            "variable_revision_number:11",
            "variable_revision_number=11",
        ),
        (
            MANUAL,
            "Use advanced settings?:Yes",
            "Use advanced settings?:No",
            "advanced settings",
        ),
        (
            MANUAL,
            "Threshold setting version:12",
            "Threshold setting version:10",
            "Threshold setting version=10",
        ),
        (
            MANUAL,
            "Thresholding method:Manual",
            "Thresholding method:Measurement",
            "Measurement",
        ),
        (
            MANUAL,
            "Method to distinguish clumped objects:None",
            "Method to distinguish clumped objects:Laplacian of Gaussian",
            "Laplacian of Gaussian",
        ),
        (
            MANUAL,
            "Method to draw dividing lines between clumped objects:None",
            "Method to draw dividing lines between clumped objects:Unknown",
            "watershed_method",
        ),
        (
            MANUAL,
            "Select neighboring objects to measure:Cells",
            "Select neighboring objects to measure:OtherCells",
            "OtherCells",
        ),
        (
            MANUAL,
            "Method to determine neighbors:Adjacent",
            "Method to determine neighbors:Within X pixels",
            "Within X pixels",
        ),
        (
            MANUAL,
            "Measure the intensity only from areas enclosed by objects?:No",
            "Measure the intensity only from areas enclosed by objects?:Yes",
            "object-restricted",
        ),
        (
            EXPORT,
            "Discard secondary objects touching the border of the image?:No",
            "Discard secondary objects touching the border of the image?:Yes",
            "Discard secondary objects",
        ),
        (
            EXPORT,
            "Select the method to identify the secondary objects:Propagation",
            "Select the method to identify the secondary objects:Distance - B",
            "Distance - B",
        ),
        (
            EXPORT,
            "Database type:SQLite",
            "Database type:MySQL",
            "MySQL",
        ),
        (
            EXPORT,
            "One table per object type",
            "Single object table",
            "single object table or a single object view",
        ),
        (
            EXPORT,
            "Add a prefix to table names?:No",
            "Add a prefix to table names?:Yes",
            "Add a prefix",
        ),
        (
            EXPORT,
            "Workspace measurement count:1",
            "Workspace measurement count:2",
            "counts",
        ),
        (
            ILLUMINATION,
            "based on all images?:Each",
            "based on all images?:All: First cycle",
            "calculate_illumination_across",
        ),
        (
            ILLUMINATION,
            "Smoothing method:Gaussian Filter",
            "Smoothing method:Splines",
            "not yet implemented",
        ),
        (
            ILLUMINATION,
            "illumination function is applied:Divide",
            "illumination function is applied:Multiply",
            "divide_or_subtract",
        ),
        (
            SAVE,
            "Select the type of image to save:Image",
            "Select the type of image to save:Mask",
            "save_image_or_figure",
        ),
        (
            SAVE,
            "constructing file names:Single name",
            "constructing file names:Sequential numbers",
            "file_name_method",
        ),
        (
            SAVE,
            "Saved file format:tiff",
            "Saved file format:h5",
            "file_format",
        ),
        (
            "fixtures/intensity_distribution/defaults.cppipe",
            "center?:These objects",
            "center?:Centers of other objects",
            "Centers of other objects",
        ),
        (
            "fixtures/texture/defaults.cppipe",
            "Hidden:1",
            "Hidden:2",
            "scale(s)",
        ),
        (
            "fixtures/colocalization/defaults_two_images.cppipe",
            "Select images to measure:DNA, GFP",
            "Select images to measure:DNA",
            "at least 2 selected images",
        ),
    ];
    for (path, old, new, expected) in cases {
        let text = fixture(path);
        assert!(text.contains(old), "{path} does not contain {old:?}");
        let err = parse(&text.replacen(old, new, 1)).unwrap_err();
        assert!(
            err.contains(expected),
            "{path}: {err:?} does not contain {expected:?}"
        );
    }
}

#[test]
fn illumination_apply_supports_multiple_images_and_preserves_groups() {
    let text = fixture(ILLUMINATION).replace(
        "    Set output image values less than 0 equal to 0?:Yes",
        "    Select the input image:Actin\n    Name the output image:ActinCorrected\n    Select the illumination function:ActinIllum\n    Select how the illumination function is applied:Subtract\n    Set output image values less than 0 equal to 0?:No",
    );
    assert_eq!(
        parse(&text).unwrap()["illumination_apply_steps"],
        json!([{
            "images": [
                {"input_image": "DNA", "output_image": "DNACorrected", "illumination_function_image": "Illum", "divide_or_subtract": "Divide"},
                {"input_image": "Actin", "output_image": "ActinCorrected", "illumination_function_image": "ActinIllum", "divide_or_subtract": "Subtract"}
            ],
            "truncate_low": false,
            "truncate_high": true
        }])
    );
}
