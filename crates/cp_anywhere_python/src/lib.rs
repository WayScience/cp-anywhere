//! Thin NumPy binding for the platform-independent measurement core.

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (shape, images, objects, mask, image_id, families, neighbors_scopes=None, neighbors_expand=None, neighbors_full_objects=None, size_shape_calculate_zernikes=true, size_shape_calculate_advanced=false, neighbors_distances=None, float32_images=None))]
fn measure<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    images: &Bound<'py, PyDict>,
    objects: &Bound<'py, PyDict>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    image_id: String,
    families: Vec<String>,
    neighbors_scopes: Option<Vec<String>>,
    neighbors_expand: Option<Vec<String>>,
    neighbors_full_objects: Option<&Bound<'py, PyDict>>,
    size_shape_calculate_zernikes: bool,
    size_shape_calculate_advanced: bool,
    neighbors_distances: Option<std::collections::BTreeMap<String, u32>>,
    float32_images: Option<Vec<String>>,
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
    let mut full_object_arrays = Vec::new();
    if let Some(full_objects) = neighbors_full_objects {
        for (name, array) in full_objects.iter() {
            full_object_arrays.push((
                name.extract::<String>()?,
                array.extract::<PyReadonlyArrayDyn<u32>>()?,
            ));
        }
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
    let full_object_views: Vec<(&str, &[u32])> = full_object_arrays
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
    let selected = cp_anywhere::Families {
        image_intensity: families.iter().any(|x| x == "image_intensity"),
        object_intensity: families.iter().any(|x| x == "object_intensity"),
        neighbors: families.iter().any(|x| x == "neighbors"),
    };
    let neighbors_scopes = neighbors_scopes.unwrap_or_default();
    let neighbors_scope_refs: Vec<&str> = neighbors_scopes.iter().map(String::as_str).collect();
    let neighbors_expand = neighbors_expand.unwrap_or_default();
    let neighbors_expand_refs: Vec<&str> = neighbors_expand.iter().map(String::as_str).collect();
    let input = cp_anywhere::Input {
        shape: &shape,
        images: &image_views,
        objects: &object_views,
        mask: mask_view,
        families: selected,
        neighbors_scopes: &neighbors_scope_refs,
        neighbors_expand: &neighbors_expand_refs,
        size_shape: families
            .iter()
            .any(|family| family == "object_size_shape")
            .then_some(cp_anywhere::SizeShapeOptions {
                calculate_zernikes: size_shape_calculate_zernikes,
                calculate_advanced: size_shape_calculate_advanced,
            }),
        neighbors_full_objects: &full_object_views,
    };
    let neighbors_distances = neighbors_distances.unwrap_or_default();
    let distances: Vec<_> = neighbors_distances
        .iter()
        .map(|(name, &r)| (name.as_str(), r))
        .collect();
    let float32_images = float32_images.unwrap_or_default();
    let float32_names: Vec<_> = float32_images.iter().map(String::as_str).collect();
    let result = py
        .detach(|| cp_anywhere::measure_with_image_precision(&input, &distances, &float32_names))
        .map_err(PyValueError::new_err)?;
    let root = PyDict::new(py);
    let features = PyDict::new(py);
    let ids = PyDict::new(py);
    let edges = PyList::empty(py);
    for (scope, columns) in result.features {
        let group = PyDict::new(py);
        for (name, column) in columns {
            match column {
                cp_anywhere::Column::Float(v) => group.set_item(name, PyArray1::from_vec(py, v))?,
                cp_anywhere::Column::Int(v) => group.set_item(name, PyArray1::from_vec(py, v))?,
            }
        }
        features.set_item(scope, group)?;
    }
    for (scope, values) in result.object_ids {
        ids.set_item(scope, PyArray1::from_vec(py, values))?;
    }
    for (scope, a, b) in result.relationships {
        let row = PyDict::new(py);
        row.set_item("object_name", &scope)?;
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
    use cp_anywhere::segmentation::{
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
    root.set_item(
        "unedited_labels",
        PyArray1::from_vec(py, result.unedited_labels),
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

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    image, rows, cols, primary_labels_in, primary_kept_labels, method,
    threshold_method, threshold_value=None,
    integer_pixels=None, mask=None,
    threshold_correction_factor=1.0,
    threshold_range_min=0.0, threshold_range_max=1.0,
    threshold_smoothing_scale=0.0,
    log_transform=false,
    threshold_scope=None, adaptive_window_size=50,
    assign_middle_to_foreground=None,
    robust_background_lower_outlier_fraction=0.05,
    robust_background_upper_outlier_fraction=0.05,
    robust_background_averaging_method=None,
    robust_background_variance_method=None,
    robust_background_number_of_deviations=2.0,
    regularization_factor=0.05,
    distance_to_dilate=10.0,
    fill_holes=false,
))]
fn identify_secondary_objects<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f64>,
    rows: usize,
    cols: usize,
    primary_labels_in: PyReadonlyArrayDyn<'py, u32>,
    primary_kept_labels: PyReadonlyArrayDyn<'py, u32>,
    method: String,
    threshold_method: String,
    threshold_value: Option<f64>,
    integer_pixels: Option<PyReadonlyArrayDyn<'py, i64>>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    threshold_correction_factor: f64,
    threshold_range_min: f64,
    threshold_range_max: f64,
    threshold_smoothing_scale: f64,
    log_transform: bool,
    threshold_scope: Option<String>,
    adaptive_window_size: usize,
    assign_middle_to_foreground: Option<bool>,
    robust_background_lower_outlier_fraction: f64,
    robust_background_upper_outlier_fraction: f64,
    robust_background_averaging_method: Option<String>,
    robust_background_variance_method: Option<String>,
    robust_background_number_of_deviations: f64,
    regularization_factor: f64,
    distance_to_dilate: f64,
    fill_holes: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use cp_anywhere::segmentation::secondary::{
        identify_secondary_objects as segment_secondary, SecondaryMethod, SecondaryObjectsConfig,
    };
    use cp_anywhere::segmentation::{build_config, ConfigParams};

    let image_slice = image
        .as_slice()
        .map_err(|_| PyValueError::new_err("image must be C-contiguous"))?;
    let primary_labels_in_slice = primary_labels_in
        .as_slice()
        .map_err(|_| PyValueError::new_err("primary_labels_in must be C-contiguous"))?;
    let primary_kept_labels_slice = primary_kept_labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("primary_kept_labels must be C-contiguous"))?;
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

    let secondary_method = match method.as_str() {
        "propagation" => SecondaryMethod::Propagation,
        "distance_n" => SecondaryMethod::DistanceN,
        "watershed_image" => SecondaryMethod::WatershedImage,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported IdentifySecondaryObjects method: {other:?}; \
                 use 'propagation', 'distance_n', or 'watershed_image'"
            )))
        }
    };

    let threshold_config = build_config(&ConfigParams {
        threshold_method,
        threshold_value,
        threshold_correction_factor,
        threshold_range: (threshold_range_min, threshold_range_max),
        threshold_smoothing_scale,
        log_transform,
        threshold_scope,
        adaptive_window_size,
        assign_middle_to_foreground,
        robust_background_lower_outlier_fraction,
        robust_background_upper_outlier_fraction,
        robust_background_averaging_method,
        robust_background_variance_method,
        robust_background_number_of_deviations,
        ..ConfigParams::default()
    })
    .map_err(PyValueError::new_err)?;

    let config = SecondaryObjectsConfig {
        method: secondary_method,
        regularization_factor,
        distance_to_dilate,
        fill_holes,
        threshold_method: threshold_config.threshold_method,
        threshold_correction_factor: threshold_config.threshold_correction_factor,
        threshold_range: threshold_config.threshold_range,
        threshold_smoothing_scale: threshold_config.threshold_smoothing_scale,
        log_transform: threshold_config.log_transform,
        threshold_scope: threshold_config.threshold_scope,
    };

    let result = py
        .detach(|| {
            segment_secondary(
                image_slice,
                integer_pixels_vec.as_deref(),
                mask_slice,
                primary_labels_in_slice,
                primary_kept_labels_slice,
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
    match result.threshold {
        Some(value) => root.set_item("threshold", value)?,
        None => root.set_item("threshold", py.None())?,
    }
    Ok(root)
}

#[pyfunction]
#[pyo3(signature = (primary_labels, secondary_labels, rows, cols, shrink_primary=true))]
fn identify_tertiary_objects<'py>(
    py: Python<'py>,
    primary_labels: PyReadonlyArrayDyn<'py, u32>,
    secondary_labels: PyReadonlyArrayDyn<'py, u32>,
    rows: usize,
    cols: usize,
    shrink_primary: bool,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let primary = primary_labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("primary_labels must be C-contiguous"))?;
    let secondary = secondary_labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("secondary_labels must be C-contiguous"))?;
    let labels = py
        .detach(|| {
            cp_anywhere::segmentation::tertiary::identify_tertiary_objects(
                primary,
                secondary,
                cols,
                rows,
                shrink_primary,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, labels))
}

#[pyfunction]
#[pyo3(signature = (
    pixels, rows, cols, objects, mask=None,
    subsample_size=0.25, image_sample_size=0.25,
    element_size=10, granular_spectrum_length=16, *, float32=false,
))]
#[allow(clippy::too_many_arguments)]
fn measure_granularity<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f64>,
    rows: usize,
    cols: usize,
    objects: &Bound<'py, PyDict>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    subsample_size: f64,
    image_sample_size: f64,
    element_size: usize,
    granular_spectrum_length: usize,
    float32: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pixels_slice = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?;
    let mask_slice = mask
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))
        })
        .transpose()?;
    let mut object_names: Vec<String> = Vec::with_capacity(objects.len());
    let mut object_arrays: Vec<PyReadonlyArrayDyn<'py, u32>> = Vec::with_capacity(objects.len());
    for (key, value) in objects.iter() {
        object_names.push(key.extract()?);
        object_arrays.push(value.extract()?);
    }
    let mut object_slices: Vec<&[u32]> = Vec::with_capacity(object_arrays.len());
    for array in &object_arrays {
        object_slices.push(
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("object labels must be C-contiguous"))?,
        );
    }
    let object_pairs: Vec<(&str, &[u32])> = object_names
        .iter()
        .map(String::as_str)
        .zip(object_slices.iter().copied())
        .collect();
    let config = cp_anywhere::granularity::Config {
        subsample_size,
        image_sample_size,
        element_size,
        granular_spectrum_length,
    };
    let result = py
        .detach(|| {
            let measure = if float32 {
                cp_anywhere::granularity::measure_float32
            } else {
                cp_anywhere::granularity::measure
            };
            measure(pixels_slice, &object_pairs, mask_slice, rows, cols, &config)
        })
        .map_err(PyValueError::new_err)?;

    let root = PyDict::new(py);
    root.set_item("image", PyArray1::from_vec(py, result.image))?;
    let objects_out = PyDict::new(py);
    for (name, scales) in result.objects {
        let rows_py = PyList::empty(py);
        for scale in scales {
            rows_py.append(PyArray1::from_vec(py, scale))?;
        }
        objects_out.set_item(name, rows_py)?;
    }
    root.set_item("objects", objects_out)?;
    Ok(root)
}

#[pyfunction]
#[pyo3(signature = (
    pixels, rows, cols, mask=None,
    intensity_choice="Regular", block_size=60,
    dilate_objects=false, object_dilation_radius=1,
    smoothing_method="Gaussian Filter",
    automatic_object_width="Automatic", object_width=10.0,
    size_of_smoothing_filter=10.0,
    rescale_option="Yes",
))]
#[allow(clippy::too_many_arguments)]
fn calculate_illumination_each<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f64>,
    rows: usize,
    cols: usize,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    intensity_choice: &str,
    block_size: usize,
    dilate_objects: bool,
    object_dilation_radius: usize,
    smoothing_method: &str,
    automatic_object_width: &str,
    object_width: f64,
    size_of_smoothing_filter: f64,
    rescale_option: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use cp_anywhere::illumination::{
        Config, FilterSizeMethod, IntensityChoice, RescaleOption, SmoothingMethod,
    };
    let pixels_slice = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?;
    let mask_slice = mask
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))
        })
        .transpose()?;
    let intensity_choice = match intensity_choice {
        "Regular" => IntensityChoice::Regular,
        "Background" => IntensityChoice::Background,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown intensity_choice {other:?}; expected 'Regular' or 'Background'"
            )));
        }
    };
    let smoothing_method = match smoothing_method {
        "No smoothing" => SmoothingMethod::None,
        "Median Filter" => SmoothingMethod::MedianFilter,
        "Gaussian Filter" => SmoothingMethod::GaussianFilter,
        "Fit Polynomial" => SmoothingMethod::FitPolynomial,
        "Smooth to Average" => SmoothingMethod::ToAverage,
        "Convex Hull" => SmoothingMethod::ConvexHull,
        "Splines" => SmoothingMethod::Splines,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown smoothing_method {other:?}"
            )));
        }
    };
    let automatic_object_width = match automatic_object_width {
        "Automatic" => FilterSizeMethod::Automatic,
        "Object size" => FilterSizeMethod::ObjectSize,
        "Manually" => FilterSizeMethod::Manually,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown automatic_object_width {other:?}"
            )));
        }
    };
    let rescale_option = match rescale_option {
        "Yes" => RescaleOption::Yes,
        "No" => RescaleOption::No,
        "Median" => RescaleOption::Median,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown rescale_option {other:?}"
            )));
        }
    };
    let config = Config {
        intensity_choice,
        block_size,
        dilate_objects,
        object_dilation_radius,
        smoothing_method,
        automatic_object_width,
        object_width,
        size_of_smoothing_filter,
        rescale_option,
    };
    let result = py
        .detach(|| {
            cp_anywhere::illumination::calculate_each(pixels_slice, mask_slice, rows, cols, &config)
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (
    images, rows, cols,
    intensity_choice="Regular", block_size=60,
    dilate_objects=false, object_dilation_radius=1,
    smoothing_method="Gaussian Filter",
    automatic_object_width="Automatic", object_width=10.0,
    size_of_smoothing_filter=10.0,
    rescale_option="Yes",
))]
#[allow(clippy::too_many_arguments)]
fn calculate_illumination_across<'py>(
    py: Python<'py>,
    images: Vec<(
        PyReadonlyArrayDyn<'py, f64>,
        Option<PyReadonlyArrayDyn<'py, bool>>,
    )>,
    rows: usize,
    cols: usize,
    intensity_choice: &str,
    block_size: usize,
    dilate_objects: bool,
    object_dilation_radius: usize,
    smoothing_method: &str,
    automatic_object_width: &str,
    object_width: f64,
    size_of_smoothing_filter: f64,
    rescale_option: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use cp_anywhere::illumination::{
        Config, FilterSizeMethod, IntensityChoice, RescaleOption, SmoothingMethod,
    };
    let mut pixel_slices: Vec<&[f64]> = Vec::with_capacity(images.len());
    let mut mask_slices: Vec<Option<&[bool]>> = Vec::with_capacity(images.len());
    for (pixels, mask) in &images {
        pixel_slices.push(
            pixels
                .as_slice()
                .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?,
        );
        mask_slices.push(
            mask.as_ref()
                .map(|array| {
                    array
                        .as_slice()
                        .map_err(|_| PyValueError::new_err("mask must be C-contiguous"))
                })
                .transpose()?,
        );
    }
    let image_pairs: Vec<(&[f64], Option<&[bool]>)> = pixel_slices
        .iter()
        .copied()
        .zip(mask_slices.iter().copied())
        .collect();
    let intensity_choice = match intensity_choice {
        "Regular" => IntensityChoice::Regular,
        "Background" => IntensityChoice::Background,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown intensity_choice {other:?}; expected 'Regular' or 'Background'"
            )));
        }
    };
    let smoothing_method = match smoothing_method {
        "No smoothing" => SmoothingMethod::None,
        "Median Filter" => SmoothingMethod::MedianFilter,
        "Gaussian Filter" => SmoothingMethod::GaussianFilter,
        "Fit Polynomial" => SmoothingMethod::FitPolynomial,
        "Smooth to Average" => SmoothingMethod::ToAverage,
        "Convex Hull" => SmoothingMethod::ConvexHull,
        "Splines" => SmoothingMethod::Splines,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown smoothing_method {other:?}"
            )));
        }
    };
    let automatic_object_width = match automatic_object_width {
        "Automatic" => FilterSizeMethod::Automatic,
        "Object size" => FilterSizeMethod::ObjectSize,
        "Manually" => FilterSizeMethod::Manually,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown automatic_object_width {other:?}"
            )));
        }
    };
    let rescale_option = match rescale_option {
        "Yes" => RescaleOption::Yes,
        "No" => RescaleOption::No,
        "Median" => RescaleOption::Median,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown rescale_option {other:?}"
            )));
        }
    };
    let config = Config {
        intensity_choice,
        block_size,
        dilate_objects,
        object_dilation_radius,
        smoothing_method,
        automatic_object_width,
        object_width,
        size_of_smoothing_filter,
        rescale_option,
    };
    let result = py
        .detach(|| cp_anywhere::illumination::calculate_across(&image_pairs, rows, cols, &config))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(measure, module)?)?;
    module.add_function(wrap_pyfunction!(identify_primary_objects, module)?)?;
    module.add_function(wrap_pyfunction!(identify_secondary_objects, module)?)?;
    module.add_function(wrap_pyfunction!(identify_tertiary_objects, module)?)?;
    module.add_function(wrap_pyfunction!(measure_granularity, module)?)?;
    module.add_function(wrap_pyfunction!(measure_intensity_distribution, module)?)?;
    module.add_function(wrap_pyfunction!(measure_texture, module)?)?;
    module.add_function(wrap_pyfunction!(measure_colocalization, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_illumination_each, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_illumination_across, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    pixels, labels, rows, cols, image_name,
    bin_count=4, wants_scaled=true, maximum_radius=100.0,
    zernikes="none", zernike_degree=9, *, emit_overflow=None, float32=false,
))]
#[allow(clippy::too_many_arguments)]
fn measure_intensity_distribution<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f64>,
    labels: PyReadonlyArrayDyn<'py, u32>,
    rows: usize,
    cols: usize,
    image_name: String,
    bin_count: usize,
    wants_scaled: bool,
    maximum_radius: f64,
    zernikes: &str,
    zernike_degree: u32,
    emit_overflow: Option<bool>,
    float32: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use cp_anywhere::intensity_distribution::{Config, PixelPrecision, ZernikeOption};

    let pixels_slice = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?;
    let labels_slice = labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?;
    let zernike_option = match zernikes {
        "none" => ZernikeOption::None,
        "magnitudes" => ZernikeOption::Magnitudes {
            degree: zernike_degree,
        },
        "magnitudes_and_phase" => ZernikeOption::MagnitudesAndPhase {
            degree: zernike_degree,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported zernikes option: {other:?}; use 'none', \
                 'magnitudes', or 'magnitudes_and_phase'"
            )))
        }
    };
    let config = Config {
        image_name,
        bin_count,
        wants_scaled,
        maximum_radius,
        zernikes: zernike_option,
    };
    let result = py
        .detach(|| {
            cp_anywhere::intensity_distribution::measure_with_precision(
                pixels_slice,
                labels_slice,
                rows,
                cols,
                &config,
                emit_overflow.unwrap_or(!wants_scaled),
                if float32 {
                    PixelPrecision::Float32
                } else {
                    PixelPrecision::Float64
                },
            )
        })
        .map_err(PyValueError::new_err)?;

    let root = PyDict::new(py);
    for (name, values) in result.columns {
        root.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok(root)
}

#[pyfunction]
#[pyo3(signature = (
    pixels, labels, rows, cols, image_name,
    gray_levels, scales, wants_image_measurements, wants_object_measurements, float32=false,
))]
#[allow(clippy::too_many_arguments)]
fn measure_texture<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f64>,
    labels: PyReadonlyArrayDyn<'py, u32>,
    rows: usize,
    cols: usize,
    image_name: String,
    gray_levels: u32,
    scales: Vec<u32>,
    wants_image_measurements: bool,
    wants_object_measurements: bool,
    float32: bool,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    use cp_anywhere::texture::{self, Config};

    let pixels_slice = pixels
        .as_slice()
        .map_err(|_| PyValueError::new_err("pixels must be C-contiguous"))?;
    let labels_slice = labels
        .as_slice()
        .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?;
    let config = Config {
        gray_levels,
        scales,
        wants_image_measurements,
        wants_object_measurements,
    };
    let result = py
        .detach(|| {
            let measure = if float32 {
                texture::measure_float32
            } else {
                texture::measure
            };
            measure(pixels_slice, rows, cols, labels_slice, &image_name, &config)
        })
        .map_err(PyValueError::new_err)?;

    let image_dict = PyDict::new(py);
    for (name, value) in result.image {
        image_dict.set_item(name, value)?;
    }
    let object_dict = PyDict::new(py);
    for (name, values) in result.objects {
        object_dict.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok((image_dict, object_dict))
}

#[pyfunction]
#[pyo3(signature = (
    images, rows, cols, masks=None, labels=None,
    threshold_percent=15.0, fast_costes="faster", costes_scale_max=255.0,
    do_corr_and_slope=true, do_manders=true, do_rwc=true, do_overlap=true,
    do_costes=true, wants_image_measurements=true, wants_object_measurements=true,
    *, float32=false,
))]
#[allow(clippy::too_many_arguments)]
fn measure_colocalization<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyDict>,
    rows: usize,
    cols: usize,
    masks: Option<&Bound<'py, PyDict>>,
    labels: Option<PyReadonlyArrayDyn<'py, u32>>,
    threshold_percent: f64,
    fast_costes: &str,
    costes_scale_max: f64,
    do_corr_and_slope: bool,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    do_costes: bool,
    wants_image_measurements: bool,
    wants_object_measurements: bool,
    float32: bool,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    use cp_anywhere::colocalization::{Config, FastCostesMode};

    let mut image_names: Vec<String> = Vec::with_capacity(images.len());
    let mut image_arrays: Vec<PyReadonlyArrayDyn<'py, f64>> = Vec::with_capacity(images.len());
    for (key, value) in images.iter() {
        image_names.push(key.extract()?);
        image_arrays.push(value.extract()?);
    }
    let mut image_slices: Vec<&[f64]> = Vec::with_capacity(image_arrays.len());
    for array in &image_arrays {
        image_slices.push(
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("image arrays must be C-contiguous"))?,
        );
    }
    let image_pairs: Vec<(&str, &[f64])> = image_names
        .iter()
        .map(String::as_str)
        .zip(image_slices.iter().copied())
        .collect();

    let mut mask_names: Vec<String> = Vec::new();
    let mut mask_arrays: Vec<PyReadonlyArrayDyn<'py, bool>> = Vec::new();
    if let Some(mask_dict) = masks {
        for (key, value) in mask_dict.iter() {
            mask_names.push(key.extract()?);
            mask_arrays.push(value.extract()?);
        }
    }
    let mut mask_slices: Vec<&[bool]> = Vec::with_capacity(mask_arrays.len());
    for array in &mask_arrays {
        mask_slices.push(
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("mask arrays must be C-contiguous"))?,
        );
    }
    let mask_pairs: Vec<(&str, &[bool])> = mask_names
        .iter()
        .map(String::as_str)
        .zip(mask_slices.iter().copied())
        .collect();
    let mask_option = if mask_pairs.is_empty() {
        None
    } else {
        Some(mask_pairs.as_slice())
    };

    let labels_array = labels
        .as_ref()
        .map(|array| {
            array
                .as_slice()
                .map_err(|_| PyValueError::new_err("labels must be C-contiguous"))
        })
        .transpose()?;

    let fast_costes_mode = match fast_costes {
        "faster" => FastCostesMode::Faster,
        "fast" => FastCostesMode::Fast,
        "accurate" => FastCostesMode::Accurate,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported fast_costes option: {other:?}; use 'faster', 'fast', or 'accurate'"
            )))
        }
    };
    let config = Config {
        threshold_percent,
        fast_costes: fast_costes_mode,
        costes_scale_max,
        do_corr_and_slope,
        do_manders,
        do_rwc,
        do_overlap,
        do_costes,
        wants_image_measurements,
        wants_object_measurements,
    };
    let result = py
        .detach(|| {
            let measure = if float32 {
                cp_anywhere::colocalization::measure_prepared_float32
            } else {
                cp_anywhere::colocalization::measure
            };
            measure(&image_pairs, rows, cols, mask_option, labels_array, &config)
        })
        .map_err(PyValueError::new_err)?;

    let image_dict = PyDict::new(py);
    for (name, value) in result.image {
        image_dict.set_item(name, value)?;
    }
    let object_dict = PyDict::new(py);
    for (name, values) in result.objects {
        object_dict.set_item(name, PyArray1::from_vec(py, values))?;
    }
    Ok((image_dict, object_dict))
}
