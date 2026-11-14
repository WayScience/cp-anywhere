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
) -> PyResult<Bound<'py, PyDict>> {
    use cp_anywhere_segmentation::{
        identify_primary_objects as segment, FillHoles, IdentifyPrimaryObjectsConfig,
        RobustAveraging, RobustBackgroundConfig, RobustVariance, ThresholdMethod,
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

    let method = match threshold_method.as_str() {
        "manual" => ThresholdMethod::Manual(threshold_value.ok_or_else(|| {
            PyValueError::new_err("threshold_value is required for threshold_method='manual'")
        })?),
        "otsu" => ThresholdMethod::OtsuTwoClass,
        "otsu_integer" => ThresholdMethod::OtsuTwoClassInteger,
        "otsu_three_class" => ThresholdMethod::OtsuThreeClass {
            assign_middle_to_foreground: assign_middle_to_foreground.unwrap_or(true),
        },
        "li" => ThresholdMethod::Li,
        "sauvola" => ThresholdMethod::Sauvola,
        "robust_background" => {
            let averaging_method = match robust_background_averaging_method.as_deref() {
                None | Some("mean") => RobustAveraging::Mean,
                Some("median") => RobustAveraging::Median,
                Some("mode") => RobustAveraging::Mode,
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported robust_background_averaging_method: {other:?}; \
                         use 'mean', 'median', or 'mode'"
                    )))
                }
            };
            let variance_method = match robust_background_variance_method.as_deref() {
                None | Some("sd") => RobustVariance::StandardDeviation,
                Some("mad") => RobustVariance::MedianAbsoluteDeviation,
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported robust_background_variance_method: {other:?}; \
                         use 'sd' or 'mad'"
                    )))
                }
            };
            ThresholdMethod::RobustBackground(RobustBackgroundConfig {
                lower_outlier_fraction: robust_background_lower_outlier_fraction,
                upper_outlier_fraction: robust_background_upper_outlier_fraction,
                averaging_method,
                variance_method,
                number_of_deviations: robust_background_number_of_deviations,
            })
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported threshold_method: {other:?}; use 'manual', 'otsu', \
                 'otsu_integer', 'otsu_three_class', 'li', 'robust_background', or 'sauvola'"
            )))
        }
    };
    let fill_holes = match fill_holes.as_deref() {
        None | Some("never") => FillHoles::Never,
        Some("after_declump_only") => FillHoles::AfterDeclumpOnly,
        Some("after_thresholding_and_declump") => FillHoles::AfterThresholdingAndDeclump,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unsupported fill_holes: {other:?}; use 'never', \
                 'after_declump_only', or 'after_thresholding_and_declump'"
            )))
        }
    };
    let config = IdentifyPrimaryObjectsConfig {
        threshold_method: method,
        threshold_correction_factor,
        threshold_range: (threshold_range_min, threshold_range_max),
        exclude_size,
        min_diameter,
        max_diameter,
        exclude_border_objects,
        maximum_object_count,
        fill_holes,
        threshold_smoothing_scale,
        log_transform,
        threshold_scope: match threshold_scope.as_deref() {
            None | Some("global") => cp_anywhere_segmentation::ThresholdScope::Global,
            Some("adaptive") => cp_anywhere_segmentation::ThresholdScope::Adaptive {
                window_size: adaptive_window_size,
            },
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unsupported threshold_scope: {other:?}; use 'global' or 'adaptive'"
                )))
            }
        },
    };

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
