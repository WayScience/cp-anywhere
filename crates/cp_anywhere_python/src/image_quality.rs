use std::collections::BTreeMap;

use cp_anywhere::image_quality::{measure, Options};
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (pixels, name, blur_scales, saturation, include_scaling, scaling, float32))]
pub(crate) fn measure_image_quality(
    pixels: PyReadonlyArrayDyn<'_, f64>,
    name: &str,
    blur_scales: Vec<usize>,
    saturation: bool,
    include_scaling: bool,
    scaling: Option<f64>,
    float32: bool,
) -> PyResult<BTreeMap<String, f64>> {
    let values = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("image must be C-contiguous"))?;
    measure(
        pixels.shape(),
        values,
        None,
        name,
        Options {
            blur_scales: &blur_scales,
            saturation,
            include_scaling,
            scaling,
            float32,
        },
    )
    .map_err(PyValueError::new_err)
}
