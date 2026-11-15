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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    pixels, rows, cols, threshold_method,
    threshold_value=None, integer_pixels=None, mask=None,
    threshold_correction_factor=1.0,
    threshold_range_min=0.0, threshold_range_max=1.0,
    exclude_size=false, min_diameter=0.0, max_diameter=f64::INFINITY,
    exclude_border_objects=false, maximum_object_count=None,
    fill_holes=None,
    threshold_smoothing_scale=0.0,
    assign_middle_to_foreground=None,
    robust_background_lower_outlier_fraction=0.05,
    robust_background_upper_outlier_fraction=0.05,
    robust_background_averaging_method=None,
    robust_background_variance_method=None,
    robust_background_number_of_deviations=2.0,
    log_transform=false,
    threshold_scope=None, adaptive_window_size=50,
    unclump_method=None, watershed_method=None,
    smoothing_filter_size=None, maxima_suppression_size=None,
    low_resolution_maxima=false,
))]
fn identify_primary_objects<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f64>,
    rows: usize,
    cols: usize,
    threshold_method: String,
    threshold_value: Option<f64>,
    integer_pixels: Option<PyReadonlyArrayDyn<'py, i64>>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    threshold_correction_factor: f64,
    threshold_range_min: f64,
    threshold_range_max: f64,
    exclude_size: bool,
    min_diameter: f64,
    max_diameter: f64,
    exclude_border_objects: bool,
    maximum_object_count: Option<u32>,
    fill_holes: Option<String>,
    threshold_smoothing_scale: f64,
    assign_middle_to_foreground: Option<bool>,
    robust_background_lower_outlier_fraction: f64,
    robust_background_upper_outlier_fraction: f64,
    robust_background_averaging_method: Option<String>,
    robust_background_variance_method: Option<String>,
    robust_background_number_of_deviations: f64,
    log_transform: bool,
    threshold_scope: Option<String>,
    adaptive_window_size: usize,
    unclump_method: Option<String>,
    watershed_method: Option<String>,
    smoothing_filter_size: Option<f64>,
    maxima_suppression_size: Option<f64>,
    low_resolution_maxima: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use cp_anywhere_segmentation::{
        build_config, identify_primary_objects as segment, ConfigParams,
    };

    let pixels_slice = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?;
    let integer_pixels_vec: Option<Vec<i64>> = integer_pixels
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map(|s| s.to_vec())
                .map_err(|_| PyValueError::new_err("integer_pixels must be C-contiguous"))
        })
        .transpose()?;
    let mask_slice = mask
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))
        })
        .transpose()?;

    let config = build_config(&ConfigParams {
        threshold_method,
        threshold_value,
        threshold_correction_factor,
        threshold_range: (threshold_range_min, threshold_range_max),
        exclude_size,
        min_diameter,
        max_diameter,
        exclude_border_objects,
        maximum_object_count,
        fill_holes,
        threshold_smoothing_scale,
        assign_middle_to_foreground,
        robust_background_lower_outlier_fraction,
        robust_background_upper_outlier_fraction,
        robust_background_averaging_method,
        robust_background_variance_method,
        robust_background_number_of_deviations,
        log_transform,
        threshold_scope,
        adaptive_window_size,
        unclump_method,
        watershed_method,
        smoothing_filter_size,
        maxima_suppression_size,
        low_resolution_maxima,
    })
    .map_err(PyValueError::new_err)?;

    let result = py
        .detach(|| {
            segment(
                pixels_slice,
                integer_pixels_vec.as_deref(),
                mask_slice,
                rows,
                cols,
                &config,
            )
        })
        .map_err(PyValueError::new_err)?;

    let root = PyDict::new(py);
    root.set_item("labels", PyArray1::from_vec(py, result.labels))?;
    root.set_item(
        "small_removed_labels",
        PyArray1::from_vec(py, result.small_removed_labels),
    )?;
    root.set_item("object_count", result.object_count)?;
    root.set_item("threshold", result.threshold)?;
    match result.threshold_map {
        Some(values) => root.set_item("threshold_map", PyArray1::from_vec(py, values))?,
        None => root.set_item("threshold_map", py.None())?,
    }
    root.set_item("guide_threshold", result.guide_threshold)?;
    Ok(root)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(measure, module)?)?;
    module.add_function(wrap_pyfunction!(identify_primary_objects, module)?)?;
    Ok(())
}
