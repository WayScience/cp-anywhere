//! Measurement dispatch for the pipeline plan emitted by `cppipe.py`.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::{self as core, Column, Families, Input, ResultTable, SizeShapeOptions};
use serde::Deserialize;
use serde_json::Value;

use super::{pixel_count, Plane, State};

#[derive(Default, Deserialize)]
#[serde(default)]
struct MeasurementPlan {
    measurement_steps: Vec<MeasurementStep>,
    granularity_steps: Vec<GranularityStep>,
    intensity_distribution_steps: Vec<IntensityDistributionStep>,
    texture_steps: Vec<TextureStep>,
    colocalization_steps: Vec<ColocalizationStep>,
}

#[derive(Deserialize)]
struct MeasurementStep {
    module_name: String,
    family: Family,
    images: Vec<String>,
    objects: Vec<String>,
    #[serde(default = "default_true")]
    calculate_zernikes: bool,
    #[serde(default)]
    calculate_advanced: bool,
    #[serde(default)]
    wants_excluded_objects: bool,
    #[serde(default)]
    wants_expand: bool,
    #[serde(default)]
    neighbors_distance: Option<u32>,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Family {
    ImageIntensity,
    ObjectIntensity,
    ObjectSizeShape,
    Neighbors,
}

#[derive(Deserialize)]
struct GranularityStep {
    images: Vec<String>,
    objects: Vec<String>,
    subsample_size: f64,
    image_sample_size: f64,
    element_size: usize,
    granular_spectrum_length: usize,
}

#[derive(Deserialize)]
struct IntensityDistributionStep {
    images: Vec<String>,
    objects: Vec<String>,
    bin_count: usize,
    wants_scaled: bool,
    maximum_radius: f64,
    zernikes: RadialZernikes,
    zernike_degree: u32,
    #[serde(default)]
    additional_bin_groups: Vec<RadialBinGroup>,
}

#[derive(Deserialize)]
struct RadialBinGroup {
    bin_count: usize,
    wants_scaled: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum RadialZernikes {
    None,
    Magnitudes,
    MagnitudesAndPhase,
}

#[derive(Deserialize)]
struct TextureStep {
    images: Vec<String>,
    objects: Vec<String>,
    gray_levels: u32,
    scales: Vec<u32>,
    wants_image_measurements: bool,
    wants_object_measurements: bool,
}

#[derive(Deserialize)]
struct ColocalizationStep {
    images: Vec<String>,
    objects: Vec<String>,
    threshold_percent: f64,
    fast_costes: FastCostes,
    do_corr_and_slope: bool,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    do_costes: bool,
    wants_image_measurements: bool,
    // Python uses the object list itself to request object measurements.
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum FastCostes {
    Faster,
    Fast,
    Accurate,
}

pub(crate) fn measure(plan: &Value, state: &State) -> Result<Option<ResultTable>, String> {
    let plan = MeasurementPlan::deserialize(plan).map_err(|e| format!("measurement plan: {e}"))?;
    let mut table = measure_basic(&plan.measurement_steps, state)?;
    for step in &plan.granularity_steps {
        measure_granularity(step, state, &mut table)?;
    }
    for step in &plan.intensity_distribution_steps {
        measure_radial(step, state, &mut table)?;
    }
    for step in &plan.texture_steps {
        measure_texture(step, state, &mut table)?;
    }
    for step in &plan.colocalization_steps {
        measure_colocalization(step, state, &mut table)?;
    }
    Ok(table)
}

fn image<'a>(state: &'a State, module: &str, name: &str) -> Result<&'a Plane, String> {
    state
        .image(name)
        .ok_or_else(|| format!("{module} needs image {name:?}, not provided"))
}

fn object<'a>(
    state: &'a State,
    module: &str,
    name: &str,
) -> Result<(&'a [usize], &'a [u32]), String> {
    state.objects(name).ok_or_else(|| {
        format!(
            "{module} needs object set {name:?}, not produced by a translated segmentation step"
        )
    })
}

fn same_shape(
    module: &str,
    name: &str,
    actual: &[usize],
    expected: &[usize],
) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{module}: all inputs must have the same shape; {name:?} has {actual:?}, expected {expected:?}"))
    }
}

fn validate_labels(module: &str, name: &str, labels: &[u32], size: usize) -> Result<(), String> {
    if labels.len() != size {
        return Err(format!(
            "{module}: labels for {name:?} do not match the shape"
        ));
    }
    if labels.iter().any(|&id| id > core::MAX_LABEL) {
        return Err(format!(
            "{module}: labels for {name:?} must be between zero and {}",
            core::MAX_LABEL
        ));
    }
    Ok(())
}

fn get_table(table: &mut Option<ResultTable>) -> &mut ResultTable {
    table.get_or_insert_with(|| ResultTable {
        features: BTreeMap::new(),
        object_ids: BTreeMap::new(),
        relationships: Vec::new(),
    })
}

fn merge_core(table: &mut Option<ResultTable>, result: ResultTable) {
    if let Some(table) = table {
        for (scope, columns) in result.features {
            table.features.entry(scope).or_default().extend(columns);
        }
        table.object_ids.extend(result.object_ids);
        table.relationships.extend(result.relationships);
    } else {
        *table = Some(result);
    }
}

fn measure_basic(steps: &[MeasurementStep], state: &State) -> Result<Option<ResultTable>, String> {
    if steps.is_empty() {
        return Ok(None);
    }
    let mut families = Families::default();
    let mut images = Vec::new();
    let mut objects = BTreeMap::new();
    // Intensity families share the union of inputs. Neighbor and shape
    // settings are applied independently for each module.
    for step in steps {
        for name in &step.images {
            if !images.iter().any(|(key, _)| *key == name) {
                images.push((name.as_str(), image(state, &step.module_name, name)?));
            }
        }
        for name in &step.objects {
            if !objects.contains_key(name.as_str()) {
                let (shape, labels) = if let Some((shape, labels)) = state.objects(name) {
                    (shape, Cow::Borrowed(labels))
                } else if let Some(raw) = state.image(name) {
                    let labels = raw
                        .labels()
                        .map_err(|e| format!("{}: object set {name:?}: {e}", step.module_name))?;
                    (raw.shape.as_slice(), Cow::Owned(labels))
                } else {
                    return Err(format!(
                        "{} needs object set {name:?}, not produced or provided",
                        step.module_name
                    ));
                };
                objects.insert(name.as_str(), (shape, labels));
            }
        }
        match step.family {
            Family::ImageIntensity => families.image_intensity = true,
            Family::ObjectIntensity => families.object_intensity = true,
            Family::Neighbors | Family::ObjectSizeShape => (),
        }
    }

    // The core requires an image even for measurements that only use labels.
    // Python uses the first pipeline image in that case, under the name _unused.
    if images.is_empty() {
        let first = state
            .images
            .first()
            .ok_or("images and objects must be nonempty")?;
        images.push(("_unused", &first.1));
    }
    let shape = images[0].1.shape.as_slice();
    let size = pixel_count(shape)?;
    let pixels = images
        .iter()
        .map(|(name, plane)| {
            same_shape("measurements", name, &plane.shape, shape)?;
            Ok((*name, plane.converted()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let images: Vec<_> = pixels
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect();

    let all_objects: Vec<_> = objects
        .iter()
        .map(|(&name, (object_shape, labels))| {
            same_shape("measurements", name, object_shape, shape)?;
            Ok((name, labels.as_ref()))
        })
        .collect::<Result<_, String>>()?;
    let mut table = None;
    if families.image_intensity || families.object_intensity {
        table = Some(core::measure(&Input {
            shape,
            images: &images,
            objects: &all_objects,
            mask: None,
            families,
            size_shape: None,
            neighbors_scopes: &[],
            neighbors_expand: &[],
            neighbors_full_objects: &[],
        })?);
    }
    let mut masked_diagnostics = BTreeMap::<&str, Vec<u32>>::new();
    let fixed_scopes: BTreeSet<_> = steps
        .iter()
        .filter(|step| step.family == Family::Neighbors && step.neighbors_distance.is_some())
        .flat_map(|step| step.objects.iter().map(String::as_str))
        .collect();
    for step in steps.iter().filter(|step| step.family == Family::Neighbors) {
        for name in step.objects.iter().map(String::as_str) {
            let kept = objects[name].1.as_ref();
            let original = state
                .primary
                .get(name)
                .map(|r| r.small_removed_labels.as_slice())
                .or_else(|| state.tertiary.get(name).map(|r| r.labels.as_slice()))
                .or_else(|| {
                    state
                        .secondary
                        .get(name)
                        .map(|r| r.small_removed_labels.as_slice())
                });
            if step.wants_excluded_objects || fixed_scopes.contains(name) {
                if let Some(primary_shape) = state.primary_shapes.get(name) {
                    same_shape("MeasureObjectNeighbors", name, primary_shape, shape)?;
                }
                if let Some(labels) = original {
                    validate_labels("MeasureObjectNeighbors", name, labels, size)?;
                }
            }
            if !step.wants_excluded_objects && fixed_scopes.contains(name) {
                if let Some(labels) = original {
                    // The oracle masks its shared small-removed layer in place.
                    // Preserve that ordering between modules, not in our outputs.
                    masked_diagnostics.entry(name).or_insert_with(|| {
                        labels
                            .iter()
                            .zip(kept)
                            .map(|(&id, &retained)| if retained == 0 { 0 } else { id })
                            .collect()
                    });
                }
            }
            let diagnostic = if step.wants_excluded_objects || step.neighbors_distance.is_some() {
                masked_diagnostics.get(name).map(Vec::as_slice).or(original)
            } else {
                None
            };
            if step.wants_excluded_objects && diagnostic.is_none() {
                return Err(format!("MeasureObjectNeighbors on {name:?} with excluded objects requires a translated segmentation result"));
            }
            let diagnostics: Vec<_> = diagnostic
                .into_iter()
                .map(|labels| (name, labels))
                .collect();
            let distances: Vec<_> = step
                .neighbors_distance
                .into_iter()
                .map(|radius| (name, radius))
                .collect();
            let scopes = [name];
            let result = core::measure_with_neighbor_distances(
                &Input {
                    shape,
                    images: &images,
                    objects: &[(name, kept)],
                    mask: None,
                    families: Families::default(),
                    size_shape: None,
                    neighbors_scopes: &scopes,
                    neighbors_expand: if step.wants_expand { &scopes } else { &[] },
                    neighbors_full_objects: &diagnostics,
                },
                &distances,
            )?;
            merge_core(&mut table, result);
        }
    }
    for step in steps
        .iter()
        .filter(|step| step.family == Family::ObjectSizeShape)
    {
        let mut selected = BTreeSet::new();
        let step_objects: Vec<_> = step
            .objects
            .iter()
            .filter(|name| selected.insert(name.as_str()))
            .map(|name| (name.as_str(), objects[name.as_str()].1.as_ref()))
            .collect();
        let result = core::measure(&Input {
            shape,
            images: &images,
            objects: &step_objects,
            mask: None,
            families: Families::default(),
            size_shape: Some(SizeShapeOptions {
                calculate_zernikes: step.calculate_zernikes,
                calculate_advanced: step.calculate_advanced,
            }),
            neighbors_scopes: &[],
            neighbors_expand: &[],
            neighbors_full_objects: &[],
        })?;
        merge_core(&mut table, result);
    }
    Ok(table)
}

fn dimensions_2d(plane: &Plane, module: &str, name: &str) -> Result<(usize, usize), String> {
    match plane.shape.as_slice() {
        &[rows, cols] if rows != 0 && cols != 0 => Ok((rows, cols)),
        _ => Err(format!(
            "{module}: image {name:?} must be a nonempty 2D array"
        )),
    }
}

fn aligned_labels<'a>(
    state: &'a State,
    module: &str,
    name: &str,
    plane: &Plane,
) -> Result<&'a [u32], String> {
    let (shape, labels) = object(state, module, name)?;
    same_shape(module, name, shape, &plane.shape)?;
    validate_labels(module, name, labels, plane.values.len())?;
    Ok(labels)
}

fn merge_image(table: &mut Option<ResultTable>, columns: impl IntoIterator<Item = (String, f64)>) {
    get_table(table)
        .features
        .entry("Image".into())
        .or_default()
        .extend(
            columns
                .into_iter()
                .map(|(name, value)| (name, Column::Float(vec![value]))),
        );
}

fn merge_objects(
    table: &mut Option<ResultTable>,
    scope: &str,
    columns: impl IntoIterator<Item = (String, Vec<f64>)>,
) {
    let table = get_table(table);
    let features = table.features.entry(scope.to_owned()).or_default();
    let mut count = 0;
    for (name, values) in columns {
        count = count.max(values.len());
        features.insert(name, Column::Float(values));
    }
    // The specialized wrappers derive IDs from the result length, not the
    // labels. In particular, texture returns packed rows for sparse labels.
    table
        .object_ids
        .entry(scope.to_owned())
        .or_insert_with(|| (1..=count as i64).collect());
}

fn measure_granularity(
    step: &GranularityStep,
    state: &State,
    table: &mut Option<ResultTable>,
) -> Result<(), String> {
    const MODULE: &str = "MeasureGranularity";
    let config = core::granularity::Config {
        subsample_size: step.subsample_size,
        image_sample_size: step.image_sample_size,
        element_size: step.element_size,
        granular_spectrum_length: step.granular_spectrum_length,
    };
    for name in &step.images {
        let plane = image(state, MODULE, name)?;
        let (rows, cols) = dimensions_2d(plane, MODULE, name)?;
        let labels: Vec<_> = step
            .objects
            .iter()
            .map(|scope| Ok((scope.as_str(), aligned_labels(state, MODULE, scope, plane)?)))
            .collect::<Result<_, String>>()?;
        let result = core::granularity::measure_float32(
            &plane.converted(),
            &labels,
            None,
            rows,
            cols,
            &config,
        )
        .map_err(|e| format!("{MODULE}: {e}"))?;
        let feature = |(i, value)| (format!("Granularity_{}_{name}", i + 1), value);
        merge_image(table, result.image.into_iter().enumerate().map(feature));
        for (scope, scales) in result.objects {
            merge_objects(
                table,
                &scope,
                scales
                    .into_iter()
                    .enumerate()
                    .map(|(i, values)| (format!("Granularity_{}_{name}", i + 1), values)),
            );
        }
    }
    Ok(())
}

fn measure_radial(
    step: &IntensityDistributionStep,
    state: &State,
    table: &mut Option<ResultTable>,
) -> Result<(), String> {
    use core::intensity_distribution::{self as radial, PixelPrecision, ZernikeOption};
    const MODULE: &str = "MeasureObjectIntensityDistribution";
    let degree = step.zernike_degree;
    let zernikes = match step.zernikes {
        RadialZernikes::None => ZernikeOption::None,
        RadialZernikes::Magnitudes => ZernikeOption::Magnitudes { degree },
        RadialZernikes::MagnitudesAndPhase => ZernikeOption::MagnitudesAndPhase { degree },
    };
    for name in &step.images {
        let plane = image(state, MODULE, name)?;
        if step.objects.is_empty() {
            continue;
        }
        let (rows, cols) = dimensions_2d(plane, MODULE, name)?;
        let pixels = plane.converted();
        let mut config = radial::Config {
            image_name: name.clone(),
            bin_count: step.bin_count,
            wants_scaled: step.wants_scaled,
            maximum_radius: step.maximum_radius,
            zernikes,
        };
        for scope in &step.objects {
            let labels = aligned_labels(state, MODULE, scope, plane)?;
            config.bin_count = step.bin_count;
            config.zernikes = zernikes;
            let mut result = radial::measure_with_precision(
                &pixels,
                labels,
                rows,
                cols,
                &config,
                !config.wants_scaled,
                PixelPrecision::Float32,
            )
            .map_err(|e| format!("{MODULE}: {e}"))?;
            // CellProfiler uses the first group's cached distances for all
            // groups, but the last unscaled group writes the Overflow values.
            config.zernikes = ZernikeOption::None;
            for group in &step.additional_bin_groups {
                config.bin_count = group.bin_count;
                let group_result = radial::measure_with_precision(
                    &pixels,
                    labels,
                    rows,
                    cols,
                    &config,
                    !group.wants_scaled,
                    PixelPrecision::Float32,
                )
                .map_err(|e| format!("{MODULE}: {e}"))?;
                result.columns.extend(group_result.columns);
            }
            merge_objects(table, scope, result.columns);
        }
    }
    Ok(())
}

fn measure_texture(
    step: &TextureStep,
    state: &State,
    table: &mut Option<ResultTable>,
) -> Result<(), String> {
    use core::texture;
    const MODULE: &str = "MeasureTexture";
    let mut config = texture::Config {
        gray_levels: step.gray_levels,
        scales: step.scales.clone(),
        wants_image_measurements: false,
        wants_object_measurements: step.wants_object_measurements,
    };
    for name in &step.images {
        let plane = image(state, MODULE, name)?;
        if step.objects.is_empty() && !step.wants_image_measurements {
            continue;
        }
        let (rows, cols) = dimensions_2d(plane, MODULE, name)?;
        if config.scales.is_empty() {
            return Err(format!("{MODULE}: scales must not be empty"));
        }
        let pixels = plane.converted();
        config.wants_image_measurements = step.wants_image_measurements;
        if step.objects.is_empty() {
            config.wants_object_measurements = false;
            let dummy = vec![0; pixels.len()];
            let result = texture::measure_float32(&pixels, rows, cols, &dummy, name, &config)
                .map_err(|e| format!("{MODULE}: {e}"))?;
            merge_image(table, result.image);
        }
        for scope in &step.objects {
            let labels = aligned_labels(state, MODULE, scope, plane)?;
            let result = texture::measure_float32(&pixels, rows, cols, labels, name, &config)
                .map_err(|e| format!("{MODULE}: {e}"))?;
            get_table(table);
            if config.wants_image_measurements {
                merge_image(table, result.image);
                config.wants_image_measurements = false;
            }
            if config.wants_object_measurements {
                merge_objects(table, scope, result.objects);
            }
        }
    }
    Ok(())
}

fn measure_colocalization(
    step: &ColocalizationStep,
    state: &State,
    table: &mut Option<ResultTable>,
) -> Result<(), String> {
    use core::colocalization::{self, FastCostesMode};
    const MODULE: &str = "MeasureColocalization";
    let mut planes = Vec::new();
    let mut seen = BTreeSet::new();
    // Pairs must retain the order in the module's image list.
    for name in &step.images {
        if seen.insert(name) {
            planes.push((name.as_str(), image(state, MODULE, name)?));
        }
    }
    if step.objects.is_empty() && !step.wants_image_measurements {
        return Ok(());
    }
    if planes.len() < 2 {
        return Err(format!("{MODULE}: at least two images are required"));
    }
    let (first_name, first_plane) = planes[0];
    let (rows, cols) = dimensions_2d(first_plane, MODULE, first_name)?;
    let pixels: Vec<_> = planes
        .into_iter()
        .map(|(name, plane)| {
            same_shape(MODULE, name, &plane.shape, &first_plane.shape)?;
            Ok((name, plane.converted()))
        })
        .collect::<Result<_, String>>()?;
    let images: Vec<_> = pixels
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect();
    let mut config = colocalization::Config {
        threshold_percent: step.threshold_percent,
        fast_costes: match step.fast_costes {
            FastCostes::Faster => FastCostesMode::Faster,
            FastCostes::Fast => FastCostesMode::Fast,
            FastCostes::Accurate => FastCostesMode::Accurate,
        },
        costes_scale_max: 255.0, // Python wrapper default; the plan carries no scale.
        do_corr_and_slope: step.do_corr_and_slope,
        do_manders: step.do_manders,
        do_rwc: step.do_rwc,
        do_overlap: step.do_overlap,
        do_costes: step.do_costes,
        wants_image_measurements: step.wants_image_measurements,
        wants_object_measurements: !step.objects.is_empty(),
    };
    let run = |labels, config: &colocalization::Config| {
        colocalization::measure_prepared_float32(&images, rows, cols, None, labels, config)
            .map_err(|e| format!("{MODULE}: {e}"))
    };
    if step.objects.is_empty() {
        merge_image(table, run(None, &config)?.image);
    }
    for scope in &step.objects {
        let labels = aligned_labels(state, MODULE, scope, first_plane)?;
        let result = run(Some(labels), &config)?;
        if config.wants_image_measurements {
            merge_image(table, result.image);
            config.wants_image_measurements = false;
        }
        merge_objects(table, scope, result.objects);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Dtype, Plane, TertiaryObjectsResult};
    use crate::segmentation::SegmentationResult;
    use serde_json::json;

    fn test_state() -> State {
        let mut labels = vec![0.0; 64];
        for row in 2..4 {
            labels[row * 8 + 2..row * 8 + 4].fill(1.0);
            labels[row * 8 + 4..row * 8 + 6].fill(2.0);
        }
        State {
            images: vec![
                (
                    "Z".into(),
                    Plane {
                        shape: vec![8, 8],
                        values: (0..64).map(f64::from).collect(),
                        dtype: Dtype::Uint8,
                    },
                ),
                (
                    "Input".into(),
                    Plane {
                        shape: vec![8, 8],
                        values: labels,
                        dtype: Dtype::Uint16,
                    },
                ),
            ],
            primary: Default::default(),
            secondary: Default::default(),
            tertiary: Default::default(),
            raw_labels: Default::default(),
            shapes: Default::default(),
            primary_shapes: Default::default(),
            derived: Default::default(),
        }
    }

    #[test]
    fn merges_specialized_families_in_the_same_table() {
        let mut state = test_state();
        state.images.push((
            "A".into(),
            Plane {
                shape: vec![8, 8],
                values: vec![0.0; 64],
                dtype: Dtype::Uint8,
            },
        ));
        state
            .raw_labels
            .insert("Input".into(), state.images[1].1.labels().unwrap());
        state.shapes.insert("Input".into(), vec![8, 8]);
        let plan = json!({
            "granularity_steps": [{"images":["Z"], "objects":["Input"], "subsample_size":1.0, "image_sample_size":1.0, "element_size":1, "granular_spectrum_length":2}],
            "intensity_distribution_steps": [{"images":["Z"], "objects":["Input"], "bin_count":2, "wants_scaled":true, "maximum_radius":100.0, "zernikes":"magnitudes_and_phase", "zernike_degree":2}],
            "texture_steps": [{"images":["Z"], "objects":["Input"], "gray_levels":8, "scales":[1], "wants_image_measurements":true, "wants_object_measurements":true}],
            "colocalization_steps": [{"images":["Z", "A"], "objects":["Input"], "threshold_percent":15.0, "fast_costes":"faster", "do_corr_and_slope":true, "do_manders":false, "do_rwc":false, "do_overlap":false, "do_costes":false, "wants_image_measurements":true, "wants_object_measurements":true}]
        });
        let out = measure(&plan, &state).unwrap().unwrap();
        let image = &out.features["Image"];
        let objects = &out.features["Input"];
        assert!(matches!(&objects["Granularity_2_Z"], Column::Float(v) if v == &[0.0, 0.0]));
        assert!(
            matches!(&objects["RadialDistribution_FracAtD_Z_1of2"], Column::Float(v) if v == &[0.800000011920929, 0.7959184050559998])
        );
        assert!(
            matches!(&image["Texture_Contrast_Z_1_00_8"], Column::Float(v) if v == &[0.017857142857142856])
        );
        assert!(
            matches!(&objects["Correlation_Correlation_Z_A"], Column::Float(v) if v.len() == 2 && v.iter().all(|x| x.is_nan()))
        );
        assert!(
            matches!(&image["Correlation_Correlation_Z_A"], Column::Float(v) if v.len() == 1 && v[0].is_nan())
        );
        assert!(!image.contains_key("Correlation_Correlation_A_Z"));
        assert_eq!(out.object_ids["Input"], [1, 2]);
    }

    #[test]
    fn measures_raw_labels_without_rescaling_and_keeps_neighbor_integer_columns() {
        let mut state = test_state();
        let plan = json!({"measurement_steps": [
            {"module_name":"MeasureObjectSizeShape", "family":"object_size_shape", "images":[], "objects":["Input"], "calculate_zernikes":false},
            {"module_name":"MeasureObjectNeighbors", "family":"neighbors", "images":[], "objects":["Input"], "wants_expand":true},
            {"module_name":"MeasureObjectIntensity", "family":"object_intensity", "images":["Z"], "objects":["Input"]}
        ]});
        let out = measure(&plan, &state).unwrap().unwrap();
        let columns = &out.features["Input"];
        assert!(
            matches!(&columns["Neighbors_NumberOfNeighbors_Expanded"], Column::Int(v) if v == &[1, 1])
        );
        assert!(matches!(&columns["AreaShape_Area"], Column::Float(v) if v == &[4.0, 4.0]));
        assert!(
            matches!(&columns["Intensity_MeanIntensity_Z"], Column::Float(v) if v == &[0.0882352963089943, 0.09607843309640884])
        );
        assert!(!columns
            .keys()
            .any(|name| name.starts_with("AreaShape_Zernike")));
        assert_eq!(out.object_ids["Input"], [1, 2]);
        assert_eq!(
            out.relationships,
            [("Input".into(), 1, 2), ("Input".into(), 2, 1)]
        );

        // A transposed shape with the same element count is not aligned.
        state.images[1].1.shape = vec![4, 16];
        assert!(measure(&plan, &state).unwrap_err().contains("same shape"));
        state.images[1].1.shape = vec![8, 8];
        state.images[1].1.dtype = Dtype::Float64;
        assert!(measure(&plan, &state).unwrap_err().contains("integer"));
    }

    #[test]
    fn checks_shape_of_primary_neighbor_diagnostics_when_shadowed() {
        let mut state = test_state();
        let labels = state.images[1].1.labels().unwrap();
        state.primary.insert(
            "Input".into(),
            SegmentationResult {
                labels: labels.clone(),
                small_removed_labels: labels.clone(),
                unedited_labels: labels.clone(),
                object_count: 2,
                threshold: 0.0,
                threshold_map: None,
                guide_threshold: None,
            },
        );
        state.primary_shapes.insert("Input".into(), vec![4, 16]);
        state
            .tertiary
            .insert("Input".into(), TertiaryObjectsResult { labels });
        state.shapes.insert("Input".into(), vec![8, 8]);
        let plan = json!({"measurement_steps": [
            {"module_name":"MeasureObjectNeighbors", "family":"neighbors", "images":[], "objects":["Input"], "wants_excluded_objects":true}
        ]});
        assert!(measure(&plan, &state).unwrap_err().contains("same shape"));
    }
}
