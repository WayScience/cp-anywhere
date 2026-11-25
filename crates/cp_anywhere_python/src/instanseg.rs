//! Python binding for InstanSeg nuclei segmentation. Only compiled when
//! `cp-anywhere`'s `instanseg-model` feature is on; see
//! `crates/cp_anywhere/src/segmentation/instanseg.rs` for the upstream
//! reference (instanseg/instanseg v0.1.2), the ONNX-from-bytes loading
//! rationale, and the port's declared gaps.

use cp_anywhere::segmentation::instanseg::{InstanSegConfig, InstanSegOnnxModel};
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    image, rows, cols, fcn_bytes, classifier_bytes,
    mask_threshold=0.53, peak_distance=5, seed_threshold=0.1,
    overlap_threshold=0.5, window_size=32, min_size=20, cleanup_fragments=true,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn instanseg_segment<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f32>,
    rows: usize,
    cols: usize,
    fcn_bytes: Vec<u8>,
    classifier_bytes: Vec<u8>,
    mask_threshold: f32,
    peak_distance: usize,
    seed_threshold: f32,
    overlap_threshold: f32,
    window_size: usize,
    min_size: usize,
    cleanup_fragments: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if image.shape() != [rows, cols] || !image.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "image must match rows and cols and be C-contiguous",
        ));
    }
    let image_slice = image.as_slice()?;
    let image_vec = image_slice.to_vec();

    let config = InstanSegConfig {
        mask_threshold,
        peak_distance,
        seed_threshold,
        overlap_threshold,
        window_size,
        min_size,
        cleanup_fragments,
    };
    let (labels, object_count) = py
        .detach(move || {
            let model = InstanSegOnnxModel::from_bytes(fcn_bytes, classifier_bytes);
            let result = model.segment(&image_vec, rows, cols, &config)?;
            Ok::<_, String>((result.labels, result.object_count))
        })
        .map_err(PyValueError::new_err)?;

    let root = PyDict::new(py);
    root.set_item("labels", PyArray1::from_slice(py, &labels))?;
    root.set_item("object_count", object_count)?;
    root.set_item("rows", rows)?;
    root.set_item("cols", cols)?;
    Ok(root)
}
