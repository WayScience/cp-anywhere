//! Decodes real `2D_versatile_fluo` network output (prob/dist arrays
//! produced by the actual pretrained StarDist model on a synthetic
//! 5-blob fluorescence field) and checks the result against the
//! authoritative Python `stardist` package's own `predict_instances`
//! output on the identical input (`tests/stardist_fixtures/`, see
//! `tests/stardist_fixtures/details.json` for the exact generation
//! parameters). This is the end-to-end oracle for
//! `segmentation::stardist::decode`: it is not comparing against a
//! hand-constructed fixture, it is comparing against the real
//! project's own reference implementation.

use cp_anywhere::segmentation::stardist::{decode, StarDistDecodeConfig, StarDistShapes};
use std::path::{Path, PathBuf};

#[path = "support/npy.rs"]
mod npy;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stardist_fixtures")
}

#[test]
fn decode_matches_the_real_stardist_package_on_real_network_output() {
    let dir = fixture_dir();
    let (prob_shape, prob) = npy::read_f32(&dir.join("prob.npy"));
    let (dist_shape, dist) = npy::read_f32(&dir.join("dist.npy"));
    let (label_shape, labels_ref) = npy::read_u32(&dir.join("labels.npy"));
    let details: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("details.json")).unwrap()).unwrap();

    assert_eq!(prob_shape, dist_shape[..2]);
    let n_rays = dist_shape[2];
    let shapes = StarDistShapes {
        out_rows: prob_shape[0],
        out_cols: prob_shape[1],
        n_rays,
        image_rows: label_shape[0],
        image_cols: label_shape[1],
    };
    let config = StarDistDecodeConfig {
        prob_thresh: details["prob_thresh"].as_f64().unwrap() as f32,
        nms_thresh: details["nms_thresh"].as_f64().unwrap() as f32,
        grid: (2, 2),
        border: 2,
    };

    let result = decode(&prob, &dist, &shapes, &config).unwrap();

    let ref_points: Vec<(f64, f64)> = details["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())
        })
        .collect();

    // Same detection count as the real stardist package.
    assert_eq!(
        result.detections.len(),
        ref_points.len(),
        "detection count must match stardist's predict_instances"
    );

    assert_eq!(result.labels.len(), labels_ref.len());
    // Match each detection by its grid center, then compare its own region.
    // A single foreground IoU would miss accidental merges or swapped labels.
    for (i, &(row, col)) in ref_points.iter().enumerate() {
        let (j, detection) = result
            .detections
            .iter()
            .enumerate()
            .find(|(_, d)| d.row == row && d.col == col)
            .unwrap_or_else(|| panic!("missing detection at ({row}, {col})"));
        let ref_prob = details["prob"][i].as_f64().unwrap() as f32;
        assert!((detection.prob - ref_prob).abs() < 1e-6);
        let (rust_id, ref_id) = ((j + 1) as u32, (i + 1) as u32);
        let (mut intersection, mut union) = (0, 0);
        for (&mine, &theirs) in result.labels.iter().zip(&labels_ref) {
            intersection += usize::from(mine == rust_id && theirs == ref_id);
            union += usize::from(mine == rust_id || theirs == ref_id);
        }
        let iou = intersection as f64 / union as f64;
        assert!(iou > 0.9, "mask IoU for ({row}, {col}) was only {iou}");
    }
}
