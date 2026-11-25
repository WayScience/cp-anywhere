//! Runs the real pinned InstanSeg `single_channel_nuclei` ONNX exports
//! through tract and checks the results against the upstream
//! instanseg_torch 0.1.2 torchscript model's own outputs
//! (`tests/instanseg_fixtures/`, see that directory's `details.json` for
//! the generation parameters and hashes). This is the end-to-end oracle
//! for `segmentation::instanseg`: the fcn parity checks compare tract
//! against the real torch model; the full-path check compares the
//! complete native pipeline (padding + fcn + the postprocessing port)
//! against upstream's own `test-output_instance_segmentation` label
//! map, which must reproduce EXACTLY (integer labels).

// The gated module and the bundled ONNX files only exist under the
// opt-in `instanseg-model` feature; the default build compiles this
// file to an empty test binary.
#![cfg(feature = "instanseg-model")]

use cp_anywhere::segmentation::instanseg::{InstanSegConfig, InstanSegOnnxModel};
use std::path::{Path, PathBuf};

#[path = "support/npy.rs"]
mod npy;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/instanseg_fixtures")
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/cp_anywhere/data/instanseg")
}

fn model() -> InstanSegOnnxModel {
    let fcn = std::fs::read(data_dir().join("instanseg_fcn.onnx"))
        .expect("src/cp_anywhere/data/instanseg/instanseg_fcn.onnx must be present in the repo");
    let classifier = std::fs::read(data_dir().join("instanseg_pixel_classifier.onnx"))
        .expect("instanseg_pixel_classifier.onnx must be present in the repo");
    InstanSegOnnxModel::from_bytes(fcn, classifier)
}

/// NCHW fixture (1, C, H, W) -> (channels, rows, cols, data)
fn fixture_nchw(path: &Path) -> (usize, usize, usize, Vec<f32>) {
    let (shape, data) = npy::read_f32(path);
    assert_eq!(shape.len(), 4, "{path:?} must be 4-D");
    assert_eq!(shape[0], 1, "{path:?} must have batch size 1");
    (shape[1], shape[2], shape[3], data)
}

#[test]
fn fcn_onnx_reproduces_the_torchscript_model_on_the_oracle_input() {
    let dir = fixture_dir();
    let (_, rows, _, input) = fixture_nchw(&dir.join("fcn_input.npy"));
    let (channels, torch_rows, torch_cols, torch) = fixture_nchw(&dir.join("fcn_output_torch.npy"));

    let output = model()
        .run_fcn(&input, rows, cols_from(&input, rows))
        .unwrap();
    assert_eq!(output.channels, channels);
    assert_eq!(output.rows, torch_rows);
    assert_eq!(output.cols, torch_cols);
    let max_abs_diff = output
        .data
        .iter()
        .zip(torch.iter())
        .map(|(&ours, &theirs)| (ours - theirs).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-3,
        "fcn output diverges from the torchscript model by {max_abs_diff}"
    );
}

#[test]
fn fcn_onnx_reproduces_the_torchscript_model_on_the_probe_input() {
    let dir = fixture_dir();
    let (_, rows, _, input) = fixture_nchw(&dir.join("fcn_probe_input.npy"));
    let (channels, torch_rows, torch_cols, torch) =
        fixture_nchw(&dir.join("fcn_probe_output_torch.npy"));

    let output = model()
        .run_fcn(&input, rows, cols_from(&input, rows))
        .unwrap();
    assert_eq!(output.channels, channels);
    assert_eq!(output.rows, torch_rows);
    assert_eq!(output.cols, torch_cols);
    let max_abs_diff = output
        .data
        .iter()
        .zip(torch.iter())
        .map(|(&ours, &theirs)| (ours - theirs).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-3,
        "fcn output diverges from the torchscript model by {max_abs_diff}"
    );
}

#[test]
fn pixel_classifier_onnx_matches_the_torchscript_classifier() {
    let dir = fixture_dir();
    let input = npy::read_f32(&dir.join("classifier_probe_input.npy"));
    let torch = npy::read_f32(&dir.join("classifier_probe_torch.npy"));
    assert_eq!(input.0.len(), 2);
    let (rows, width) = (input.0[0], input.0[1]);

    let logits = model().run_pixel_classifier(&input.1, rows, width).unwrap();
    assert_eq!(logits.len(), torch.1.len());
    let max_abs_diff = logits
        .iter()
        .zip(torch.1.iter())
        .map(|(&ours, &theirs)| (ours - theirs).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-5,
        "classifier logits diverge from the torchscript model by {max_abs_diff}"
    );
}

#[test]
fn full_native_path_reproduces_the_upstream_label_oracle_exactly() {
    let dir = fixture_dir();
    let (_, rows, _, input) = fixture_nchw(&dir.join("fcn_input.npy"));
    let (oracle_shape, oracle) = npy::read_f32(&dir.join("labels_oracle.npy"));
    assert_eq!(
        (oracle_shape[2], oracle_shape[3]),
        (rows, cols_from(&input, rows))
    );

    let result = model()
        .segment(
            &input,
            rows,
            cols_from(&input, rows),
            &InstanSegConfig::default(),
        )
        .unwrap();

    // The oracle holds upstream's exact integer labels; the port must
    // reproduce them EXACTLY, not approximately.
    let mismatches: usize = result
        .labels
        .iter()
        .zip(oracle.iter())
        .filter(|(&ours, &theirs)| ours as f32 != theirs)
        .count();
    assert_eq!(
        mismatches, 0,
        "native InstanSeg labels diverge from upstream's oracle at {mismatches} pixels"
    );
}

#[test]
fn full_native_path_is_deterministic_on_a_non_oracle_shape() {
    let dir = fixture_dir();
    let (_, rows, _, input) = fixture_nchw(&dir.join("fcn_probe_input.npy"));
    let (cols, input) = (cols_from(&input, rows), input);

    let first = model().segment(&input, rows, cols, &InstanSegConfig::default());
    let second = model().segment(&input, rows, cols, &InstanSegConfig::default());
    let (first, second) = (first.unwrap(), second.unwrap());
    assert_eq!(first.labels, second.labels);
    assert_eq!(first.object_count, second.object_count);
    assert_eq!(first.labels.len(), rows * cols);
    // Upstream's own test image segments 26 objects on the oracle input;
    // the probe should still produce a sensible, non-degenerate result
    // without asserting a specific count.
    assert!(first.object_count > 0);
}

/// (1, 1, rows, cols) fixture rows -> cols
fn cols_from(input: &[f32], rows: usize) -> usize {
    input.len() / rows
}
