//! Thin NumPy binding for the platform-independent measurement core.

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
fn measure<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    images: &Bound<'py, PyDict>,
    objects: &Bound<'py, PyDict>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    image_id: String,
    families: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut image_arrays = Vec::new();
    for (name, array) in images.iter() {
        image_arrays.push((
            name.extract::<String>()?,
            array.extract::<PyReadonlyArrayDyn<f64>>()?,
        ));
    }
    let mut object_arrays = Vec::new();
    for (name, array) in objects.iter() {
        object_arrays.push((
            name.extract::<String>()?,
            array.extract::<PyReadonlyArrayDyn<u32>>()?,
        ));
    }
    let image_views: Vec<(&str, &[f64])> = image_arrays
        .iter()
        .map(|(name, array)| {
            Ok((
                name.as_str(),
                array
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("image must be C-contiguous"))?,
            ))
        })
        .collect::<PyResult<_>>()?;
    let object_views: Vec<(&str, &[u32])> = object_arrays
        .iter()
        .map(|(name, array)| {
            Ok((
                name.as_str(),
                array
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?,
            ))
        })
        .collect::<PyResult<_>>()?;
    let mask_view = mask
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))
        })
        .transpose()?;
    let selected = cp_anywhere_core::Families {
        image_intensity: families.iter().any(|x| x == "image_intensity"),
        object_intensity: families.iter().any(|x| x == "object_intensity"),
        neighbors: families.iter().any(|x| x == "neighbors"),
    };
    let input = cp_anywhere_core::Input {
        shape: &shape,
        images: &image_views,
        objects: &object_views,
        mask: mask_view,
        families: selected,
    };
    let result = py
        .detach(|| cp_anywhere_core::measure(&input))
        .map_err(PyValueError::new_err)?;
    let root = PyDict::new(py);
    let features = PyDict::new(py);
    let ids = PyDict::new(py);
    let edges = PyList::empty(py);
    for (scope, columns) in result.features {
        let group = PyDict::new(py);
        for (name, column) in columns {
            match column {
                cp_anywhere_core::Column::Float(v) => {
                    group.set_item(name, PyArray1::from_vec(py, v))?
                }
                cp_anywhere_core::Column::Int(v) => {
                    group.set_item(name, PyArray1::from_vec(py, v))?
                }
            }
        }
        features.set_item(scope, group)?;
    }
    for (scope, values) in result.object_ids {
        ids.set_item(scope, PyArray1::from_vec(py, values))?;
    }
    for (a, b) in result.relationships {
        let row = PyDict::new(py);
        row.set_item("first_image_id", &image_id)?;
        row.set_item("first_object_id", a)?;
        row.set_item("second_image_id", &image_id)?;
        row.set_item("second_object_id", b)?;
        edges.append(row)?;
    }
    root.set_item("features", features)?;
    root.set_item("object_ids", ids)?;
    root.set_item("relationships", edges)?;
    root.set_item("image_id", image_id)?;
    Ok(root)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(measure, module)?)?;
    Ok(())
}
