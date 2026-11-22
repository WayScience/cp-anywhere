//! Python binding for StarDist nuclei detection (M2). Only compiled
//! when `cp-anywhere`'s `stardist-model` feature is on; see
//! `crates/cp_anywhere/src/segmentation/stardist_model.rs` for why
//! the model is loaded from bytes, never a path.

use cp_anywhere::segmentation::stardist::{decode, StarDistDecodeConfig, StarDistShapes};
use cp_anywhere::segmentation::stardist_model::StarDistOnnxModel;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    image, rows, cols, model_bytes, n_rays, grid_rows, grid_cols,
    prob_thresh, nms_thresh, border=2,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stardist_segment<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f32>,
    rows: usize,
    cols: usize,
    model_bytes: Vec<u8>,
    n_rays: usize,
    grid_rows: usize,
    grid_cols: usize,
    prob_thresh: f32,
    nms_thresh: f32,
    border: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if image.shape() != [rows, cols] || !image.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "image must match rows and cols and be C-contiguous",
        ));
    }
    let image_slice = image.as_slice()?;
    let image_vec = image_slice.to_vec();

    let (labels, points, probs) = py
        .detach(move || {
            let model = StarDistOnnxModel::from_bytes(model_bytes, n_rays, (grid_rows, grid_cols));
            let raw = model.predict(&image_vec, rows, cols)?;
            let shapes = StarDistShapes {
                out_rows: raw.out_rows,
                out_cols: raw.out_cols,
                n_rays: raw.n_rays,
                image_rows: rows,
                image_cols: cols,
            };
            let config = StarDistDecodeConfig {
                prob_thresh,
                nms_thresh,
                grid: (grid_rows, grid_cols),
                border,
            };
            let result = decode(&raw.prob, &raw.dist, &shapes, &config)?;
            let points: Vec<f64> = result
                .detections
                .iter()
                .flat_map(|d| [d.row, d.col])
                .collect();
            let probs: Vec<f32> = result.detections.iter().map(|d| d.prob).collect();
            Ok::<_, String>((result.labels, points, probs))
        })
        .map_err(PyValueError::new_err)?;

    let object_count = probs.len();
    let root = PyDict::new(py);
    root.set_item("labels", PyArray1::from_vec(py, labels))?;
    root.set_item("points", PyArray1::from_vec(py, points))?;
    root.set_item("prob", PyArray1::from_vec(py, probs))?;
    root.set_item("object_count", object_count)?;
    Ok(root)
}
