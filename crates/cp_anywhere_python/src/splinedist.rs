//! Python binding for SplineDist nuclei segmentation. Only compiled when
//! `cp-anywhere`'s `splinedist-model` feature is on; see
//! `crates/cp_anywhere/src/segmentation/splinedist.rs` for the upstream
//! reference (uhlmanngroup/splinedist `bbbc038_10`), the ONNX-from-bytes
//! loading rationale, and the port's declared gaps.

use cp_anywhere::segmentation::splinedist::{
    decode, load_phi_basis, SplineDistOnnxModel, SplineDistThresholds, CONTOUR_SAMPLES,
    CONTROL_POINTS,
};
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (image, rows, cols, model_bytes, phi_bytes, real_rows=None, real_cols=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn splinedist_segment<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f32>,
    rows: usize,
    cols: usize,
    model_bytes: Vec<u8>,
    phi_bytes: Vec<u8>,
    real_rows: Option<usize>,
    real_cols: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if image.shape() != [rows, cols] || !image.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "image must match rows and cols and be C-contiguous",
        ));
    }
    let image_vec = image.as_slice()?.to_vec();
    let real_rows = real_rows.unwrap_or(rows).min(rows);
    let real_cols = real_cols.unwrap_or(cols).min(cols);
    let (labels, object_count) = py
        .detach(move || {
            let phi = load_phi_basis(&phi_bytes)?;
            let net = SplineDistOnnxModel::from_bytes(model_bytes)
                .run_network(&image_vec, rows, cols, real_rows, real_cols)?;
            let result = decode(
                &net.prob,
                &net.coord_channels,
                net.ch_rows,
                net.ch_cols,
                &phi,
                CONTOUR_SAMPLES,
                CONTROL_POINTS,
                &SplineDistThresholds::from_shipped_model(),
                real_rows,
                real_cols,
            )?;
            Ok::<_, String>((result.labels, result.object_count))
        })
        .map_err(PyValueError::new_err)?;

    let root = PyDict::new(py);
    root.set_item("labels", labels.into_pyarray(py))?;
    root.set_item("object_count", object_count)?;
    root.set_item("rows", real_rows)?;
    root.set_item("cols", real_cols)?;
    Ok(root)
}
