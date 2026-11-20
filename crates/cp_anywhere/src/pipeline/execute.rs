use std::collections::{BTreeMap, BTreeSet};

use crate::illumination::{
    self, Config, FilterSizeMethod, IntensityChoice, RescaleOption, SmoothingMethod,
};
use crate::segmentation::secondary::{
    identify_secondary_objects, SecondaryMethod, SecondaryObjectsConfig,
};
use crate::segmentation::tertiary::identify_tertiary_objects;
use crate::segmentation::{
    build_config, identify_primary_objects, ConfigParams, IdentifyPrimaryObjectsConfig,
};
use serde::Deserialize;
use serde_json::Value;

use super::{measure, Dtype, Plane, RunResult, State, TertiaryObjectsResult};

fn steps<'a>(plan: &'a Value, key: &str) -> impl Iterator<Item = &'a Value> {
    plan[key].as_array().into_iter().flatten()
}

fn text<'a>(step: &'a Value, key: &str) -> Result<&'a str, String> {
    step[key]
        .as_str()
        .ok_or_else(|| format!("invalid pipeline setting {key}"))
}

fn boolean(step: &Value, key: &str) -> Result<bool, String> {
    step[key]
        .as_bool()
        .ok_or_else(|| format!("invalid pipeline setting {key}"))
}

fn number(step: &Value, key: &str) -> Result<f64, String> {
    step[key]
        .as_f64()
        .ok_or_else(|| format!("invalid pipeline setting {key}"))
}

fn size(step: &Value, key: &str) -> Result<usize, String> {
    step[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("invalid pipeline setting {key}"))
}

fn require_image<'a>(state: &'a State, module: &str, name: &str) -> Result<&'a Plane, String> {
    state
        .image(name)
        .ok_or_else(|| format!("{module} needs image {name:?}, not provided to run()"))
}

fn grid(shape: &[usize]) -> Result<(usize, usize), String> {
    match shape {
        &[rows, cols] => Ok((rows, cols)),
        _ => Err("expected a nonempty 2D array".into()),
    }
}

fn insert_image(state: &mut State, name: &str, shape: Vec<usize>, values: Vec<f64>) {
    let plane = Plane {
        shape,
        values,
        dtype: Dtype::Float64,
    };
    state.derived.insert(name.into());
    if let Some((_, current)) = state.images.iter_mut().find(|(key, _)| key == name) {
        *current = plane;
    } else {
        state.images.push((name.into(), plane));
    }
}

fn illumination_config(step: &Value) -> Result<Config, String> {
    Ok(Config {
        intensity_choice: match text(step, "intensity_choice")? {
            "Regular" => IntensityChoice::Regular,
            "Background" => IntensityChoice::Background,
            other => {
                return Err(format!(
                    "unsupported illumination intensity choice {other:?}"
                ))
            }
        },
        block_size: size(step, "block_size")?,
        dilate_objects: boolean(step, "dilate_objects")?,
        object_dilation_radius: size(step, "object_dilation_radius")?,
        smoothing_method: match text(step, "smoothing_method")? {
            "No smoothing" => SmoothingMethod::None,
            "Median Filter" => SmoothingMethod::MedianFilter,
            "Gaussian Filter" => SmoothingMethod::GaussianFilter,
            "Fit Polynomial" => SmoothingMethod::FitPolynomial,
            "Smooth to Average" => SmoothingMethod::ToAverage,
            other => {
                return Err(format!(
                    "unsupported illumination smoothing method {other:?}"
                ))
            }
        },
        automatic_object_width: match text(step, "automatic_object_width")? {
            "Automatic" => FilterSizeMethod::Automatic,
            "Object size" => FilterSizeMethod::ObjectSize,
            "Manually" => FilterSizeMethod::Manually,
            other => {
                return Err(format!(
                    "unsupported illumination filter size method {other:?}"
                ))
            }
        },
        object_width: number(step, "object_width")?,
        size_of_smoothing_filter: number(step, "size_of_smoothing_filter")?,
        rescale_option: match text(step, "rescale_option")? {
            "Yes" => RescaleOption::Yes,
            "No" => RescaleOption::No,
            "Median" => RescaleOption::Median,
            other => return Err(format!("unsupported illumination rescale option {other:?}")),
        },
    })
}

fn run_calculate(step: &Value, state: &mut State) -> Result<(), String> {
    let input = require_image(
        state,
        "CorrectIlluminationCalculate",
        text(step, "input_image")?,
    )?;
    let (rows, cols) = grid(&input.shape)?;
    let values = illumination::calculate_each(
        &input.converted(),
        None,
        rows,
        cols,
        &illumination_config(step)?,
    )?;
    insert_image(
        state,
        text(step, "output_image")?,
        input.shape.clone(),
        values,
    );
    Ok(())
}

fn run_apply(step: &Value, state: &mut State) -> Result<(), String> {
    let low = boolean(step, "truncate_low")?;
    let high = boolean(step, "truncate_high")?;
    for entry in steps(step, "images") {
        let input = require_image(
            state,
            "CorrectIlluminationApply",
            text(entry, "input_image")?,
        )?;
        let function = require_image(
            state,
            "CorrectIlluminationApply",
            text(entry, "illumination_function_image")?,
        )?;
        if input.shape != function.shape {
            return Err("image and illumination_function must have equal dimensions".into());
        }
        let subtract = match text(entry, "divide_or_subtract")? {
            "Subtract" => true,
            "Divide" => false,
            other => return Err(format!("unsupported illumination application {other:?}")),
        };
        let values = input
            .converted()
            .into_iter()
            .zip(function.converted())
            .map(|(p, f)| {
                let mut value = if subtract { p - f } else { p / f };
                if (subtract || low) && value < 0.0 {
                    value = 0.0;
                }
                if high && value > 1.0 {
                    value = 1.0;
                }
                value as f32 as f64
            })
            .collect();
        insert_image(
            state,
            text(entry, "output_image")?,
            input.shape.clone(),
            values,
        );
    }
    Ok(())
}

fn run_image_math(step: &Value, state: &mut State) -> Result<(), String> {
    let input = require_image(state, "ImageMath", text(step, "input_image")?)?;
    grid(&input.shape)?;
    let input_factor = number(step, "input_factor")?;
    let output_factor = number(step, "output_factor")?;
    let addend = number(step, "addend")?;
    let low = boolean(step, "truncate_low")?;
    let high = boolean(step, "truncate_high")?;
    let replace_nan = boolean(step, "replace_nan")?;
    let values = input
        .converted()
        .into_iter()
        .map(|pixel| {
            let mut pixel = pixel as f32;
            if input_factor != 1.0 {
                pixel *= input_factor as f32;
            }
            if output_factor != 1.0 {
                pixel *= output_factor as f32;
            }
            if addend != 0.0 {
                pixel += addend as f32;
            }
            if (low && pixel < 0.0) || (replace_nan && pixel.is_nan()) {
                pixel = 0.0;
            }
            if high && pixel > 1.0 {
                pixel = 1.0;
            }
            f64::from(pixel)
        })
        .collect();
    insert_image(
        state,
        text(step, "output_image")?,
        input.shape.clone(),
        values,
    );
    Ok(())
}

#[derive(Deserialize)]
struct QualityStep {
    all_loaded: bool,
    groups: Vec<QualityGroup>,
}

#[derive(Deserialize)]
struct QualityGroup {
    images: Vec<String>,
    blur_scales: Vec<usize>,
    saturation: bool,
    include_scaling: bool,
}

type ImageColumns = BTreeMap<String, crate::Column>;

fn run_quality(
    step: &Value,
    state: &State,
    loaded: &BTreeSet<String>,
    scaling: &BTreeMap<String, f64>,
    columns: &mut ImageColumns,
) -> Result<(), String> {
    use crate::image_quality::{measure, Options};
    let step = QualityStep::deserialize(step).map_err(|e| format!("MeasureImageQuality: {e}"))?;
    for group in step.groups {
        let names: Vec<_> = if step.all_loaded {
            loaded.iter().collect()
        } else {
            group.images.iter().collect()
        };
        for name in names {
            let plane = require_image(state, "MeasureImageQuality", name)?;
            let values = measure(
                &plane.shape,
                &plane.converted(),
                None,
                name,
                Options {
                    blur_scales: &group.blur_scales,
                    saturation: group.saturation,
                    include_scaling: group.include_scaling,
                    scaling: scaling
                        .get(name)
                        .copied()
                        .filter(|_| !state.derived.contains(name)),
                    float32: !matches!(plane.dtype, Dtype::Bool),
                },
            )?;
            columns.extend(
                values
                    .into_iter()
                    .map(|(name, value)| (name, crate::Column::Float(vec![value]))),
            );
        }
    }
    Ok(())
}

fn run_preprocessing(
    plan: &Value,
    state: &mut State,
    loaded: &BTreeSet<String>,
    scaling: &BTreeMap<String, f64>,
    columns: &mut ImageColumns,
) -> Result<bool, String> {
    let mut calculate = steps(plan, "illumination_calculate_steps");
    let mut apply = steps(plan, "illumination_apply_steps");
    let mut math = steps(plan, "image_math_steps");
    let mut quality = steps(plan, "image_quality_steps");
    let mut flags = steps(plan, "flag_image_steps");
    for module in steps(plan, "image_step_order") {
        let missing = || format!("missing preprocessing step {module}");
        match module.as_str() {
            Some("CorrectIlluminationCalculate") => {
                run_calculate(calculate.next().ok_or_else(missing)?, state)?
            }
            Some("CorrectIlluminationApply") => {
                run_apply(apply.next().ok_or_else(missing)?, state)?
            }
            Some("ImageMath") => run_image_math(math.next().ok_or_else(missing)?, state)?,
            Some("MeasureImageQuality") => {
                run_quality(
                    quality.next().ok_or_else(missing)?,
                    state,
                    loaded,
                    scaling,
                    columns,
                )?;
            }
            Some("FlagImage") => {
                use super::flag_image::{evaluate, Step};
                use crate::Column;
                let step = Step::deserialize(flags.next().ok_or_else(missing)?)
                    .map_err(|e| format!("FlagImage: {e}"))?;
                let (values, skip) = evaluate(&step, |name| match columns.get(name)? {
                    Column::Float(values) => values.first().copied(),
                    Column::Int(values) => values.first().map(|&value| value as f64),
                })?;
                columns.extend(
                    values
                        .into_iter()
                        .map(|(name, value)| (name, Column::Int(vec![value]))),
                );
                if skip {
                    return Ok(true);
                }
            }
            _ => return Err(format!("unknown preprocessing module {module}")),
        }
    }
    Ok(false)
}

fn segmentation_pixels(plane: &Plane) -> Result<std::borrow::Cow<'_, [f64]>, String> {
    use std::borrow::Cow;
    let divisor = match plane.dtype {
        Dtype::Uint8 => u8::MAX as f64,
        Dtype::Uint16 => u16::MAX as f64,
        Dtype::Uint32 => u32::MAX as f64,
        Dtype::Bool | Dtype::Float16 | Dtype::Float32 | Dtype::Float64 => {
            return Ok(Cow::Borrowed(&plane.values))
        }
        other => {
            return Err(format!(
                "rescale=True does not support dtype {other:?} for segmentation"
            ))
        }
    };
    // Python's segmentation wrappers divide raw integer pixels in f64, not
    // in the f32 conversion used for measurement and illumination.
    Ok(Cow::Owned(
        plane.values.iter().map(|p| p / divisor).collect(),
    ))
}

fn segmentation_config(
    step: &Value,
    plane: &Plane,
) -> Result<IdentifyPrimaryObjectsConfig, String> {
    let mut raw = step["config"].clone();
    let obj = raw
        .as_object_mut()
        .ok_or("missing segmentation configuration")?;
    // The secondary module's boolean hole filling setting is not the
    // primary module's three-way fill_holes enum.
    if obj.get("fill_holes").is_some_and(Value::is_boolean) {
        obj.remove("fill_holes");
    }
    if let Some(threshold) = obj.remove("threshold") {
        if threshold.is_number() {
            obj.insert("threshold_method".into(), "manual".into());
            obj.insert("threshold_value".into(), threshold);
        } else {
            obj.insert("threshold_method".into(), threshold);
        }
    }
    let mut config: ConfigParams = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    if step["config"]["threshold_range"].is_null() && plane.dtype.integer_range().is_none() {
        config.threshold_range = (f64::NEG_INFINITY, f64::INFINITY);
    }
    build_config(&config)
}

fn run_segmentation(plan: &Value, state: &mut State) -> Result<(), String> {
    for step in steps(plan, "segmentation_steps") {
        let image = require_image(state, "IdentifyPrimaryObjects", text(step, "input_image")?)?;
        let (rows, cols) = grid(&image.shape)?;
        let result = identify_primary_objects(
            &segmentation_pixels(image)?,
            None,
            None,
            rows,
            cols,
            &segmentation_config(step, image)?,
        )?;
        let name = text(step, "output_object")?.to_owned();
        let shape = image.shape.clone();
        state.primary_shapes.insert(name.clone(), shape.clone());
        state.shapes.insert(name.clone(), shape);
        state.primary.insert(name, result);
    }
    for step in steps(plan, "secondary_steps") {
        let image = require_image(
            state,
            "IdentifySecondaryObjects",
            text(step, "input_image")?,
        )?;
        let seed = text(step, "primary_object")?;
        let primary = state
            .primary
            .get(seed)
            .ok_or_else(|| format!("IdentifySecondaryObjects needs primary object set {seed:?}"))?;
        if state.primary_shapes[seed] != image.shape {
            return Err("primary labels must have the image shape".into());
        }
        let (rows, cols) = grid(&image.shape)?;
        let threshold = segmentation_config(step, image)?;
        let raw = &step["config"];
        let (method, regularization_factor, distance_to_dilate) = match text(raw, "method")? {
            "propagation" => (
                SecondaryMethod::Propagation,
                number(raw, "regularization_factor")?,
                10.0,
            ),
            "distance_n" => (
                SecondaryMethod::DistanceN,
                0.05,
                number(raw, "distance_to_dilate")?,
            ),
            "watershed_image" => (SecondaryMethod::WatershedImage, 0.05, 10.0),
            other => return Err(format!("unsupported secondary method {other:?}")),
        };
        let result = identify_secondary_objects(
            &segmentation_pixels(image)?,
            None,
            None,
            &primary.unedited_labels,
            &primary.labels,
            rows,
            cols,
            &SecondaryObjectsConfig {
                method,
                regularization_factor,
                distance_to_dilate,
                fill_holes: boolean(raw, "fill_holes")?,
                discard_edge: raw.get("discard_edge").is_some() && boolean(raw, "discard_edge")?,
                threshold_method: threshold.threshold_method,
                threshold_correction_factor: threshold.threshold_correction_factor,
                threshold_range: threshold.threshold_range,
                threshold_smoothing_scale: threshold.threshold_smoothing_scale,
                log_transform: threshold.log_transform,
                threshold_scope: threshold.threshold_scope,
            },
        )?;
        let name = text(step, "output_object")?.to_owned();
        state.shapes.insert(name.clone(), image.shape.clone());
        state.secondary.insert(name, result);
    }
    for step in steps(plan, "tertiary_steps") {
        let primary = text(step, "primary_object")?;
        let secondary = text(step, "secondary_object")?;
        for name in [primary, secondary] {
            if state.objects(name).is_none() {
                let image = state
                    .image(name)
                    .ok_or_else(|| format!("IdentifyTertiaryObjects needs object set {name:?}"))?;
                state.raw_labels.insert(name.into(), image.labels()?);
                // Keep the raw input shape alongside its label copy.
                let shape = state
                    .image(name)
                    .ok_or("missing label input")?
                    .shape
                    .clone();
                state.shapes.insert(name.into(), shape);
            }
        }
        let (shape, smaller) = state.objects(primary).ok_or("missing smaller object set")?;
        let (larger_shape, larger) = state
            .objects(secondary)
            .ok_or("missing larger object set")?;
        if shape != larger_shape {
            return Err("primary and secondary label arrays must have matching shapes".into());
        }
        let (rows, cols) = grid(shape)?;
        let labels = identify_tertiary_objects(
            smaller,
            larger,
            cols,
            rows,
            boolean(step, "shrink_primary")?,
        )?;
        let name = text(step, "output_object")?.to_owned();
        state.shapes.insert(name.clone(), shape.to_vec());
        state
            .tertiary
            .insert(name, TertiaryObjectsResult { labels });
    }
    Ok(())
}

pub(crate) fn run(
    plan: &Value,
    images: Vec<(String, Plane)>,
    scaling: &BTreeMap<String, f64>,
) -> Result<RunResult, String> {
    let mut names = BTreeSet::new();
    for (name, plane) in &images {
        if !names.insert(name.clone()) {
            return Err(format!("duplicate image name {name:?}"));
        }
        plane
            .validate()
            .map_err(|error| format!("image {name:?}: {error}"))?;
    }
    let mut state = State {
        images,
        primary: Default::default(),
        secondary: Default::default(),
        tertiary: Default::default(),
        raw_labels: Default::default(),
        shapes: Default::default(),
        primary_shapes: Default::default(),
        derived: Default::default(),
    };
    if scaling
        .iter()
        .any(|(name, value)| !names.contains(name) || !value.is_finite() || *value <= 0.0)
    {
        return Err("image scaling must name a supplied image and be finite and positive".into());
    }
    let mut columns = ImageColumns::new();
    let skipped = run_preprocessing(plan, &mut state, &names, scaling, &mut columns)?;
    let mut measurements = None;
    if !skipped {
        run_segmentation(plan, &mut state)?;
        measurements = measure::measure(plan, &state)?;
        super::overlay::run(plan, &mut state)?;
    }
    if !columns.is_empty() {
        let table = measurements.get_or_insert_with(|| crate::ResultTable {
            features: BTreeMap::new(),
            object_ids: BTreeMap::new(),
            relationships: Vec::new(),
        });
        table
            .features
            .entry("Image".into())
            .or_default()
            .extend(columns);
    }
    Ok(RunResult {
        skipped,
        primary: state.primary,
        secondary: state.secondary,
        tertiary: state.tertiary,
        object_shapes: state.shapes,
        derived_images: state
            .images
            .into_iter()
            .filter(|(name, _)| state.derived.contains(name))
            .collect(),
        measurements,
    })
}
