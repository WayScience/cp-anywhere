use super::{list, RawModule};
use serde_json::{json, Value};
use std::collections::HashSet;

// Every Measure* module that shares Python's MeasurementStep gets all of
// its dataclass fields, including defaults unrelated to its own family.
#[derive(Default)]
struct MeasurementFlags {
    size_shape: Option<(bool, bool)>,
    wants_excluded_objects: bool,
    wants_expand: bool,
}

fn measurement_step(
    raw: &RawModule<'_>,
    family: &str,
    images: &[&str],
    objects: &[&str],
    flags: MeasurementFlags,
) -> Value {
    let (calculate_zernikes, calculate_advanced) = flags.size_shape.unwrap_or((true, false));
    json!({
        "module_name": raw.name,
        "family": family,
        "images": images,
        "objects": objects,
        "calculate_zernikes": calculate_zernikes,
        "calculate_advanced": calculate_advanced,
        "wants_excluded_objects": flags.wants_excluded_objects,
        "wants_expand": flags.wants_expand,
        "neighbors_distance": null,
    })
}

pub(super) fn object_intensity(raw: &RawModule<'_>) -> Result<Value, String> {
    let [images, objects] = raw.exact()?;
    Ok(measurement_step(
        raw,
        "object_intensity",
        &list(images),
        &list(objects),
        MeasurementFlags::default(),
    ))
}

pub(super) fn image_intensity(raw: &RawModule<'_>) -> Result<Value, String> {
    let [images, wants_objects, _objects, _wants_pct, _pct] = raw.exact()?;
    if raw.yes_no("wants_objects", wants_objects)? {
        return Err(raw.error("'Measure the intensity only from areas enclosed by objects?' = Yes is out of scope (object-restricted MeasureImageIntensity)"));
    }
    Ok(measurement_step(
        raw,
        "image_intensity",
        &list(images),
        &[],
        MeasurementFlags::default(),
    ))
}

pub(super) fn neighbors(raw: &RawModule<'_>) -> Result<Value, String> {
    // Python reads the first five settings and ignores any visual output settings.
    let [object_name, neighbors_name, distance_method, distance, wants_excluded_objects] =
        raw.leading()?;
    if object_name != neighbors_name {
        return Err(raw.error(format_args!("only an object set against itself is supported (got {object_name:?} against {neighbors_name:?})")));
    }
    let wants_expand = raw.choice(
        "distance method",
        distance_method,
        &[
            ("Adjacent", false),
            ("Expand until adjacent", true),
            ("Within a specified distance", false),
        ],
    )?;
    let mut step = measurement_step(
        raw,
        "neighbors",
        &[],
        &[object_name],
        MeasurementFlags {
            wants_excluded_objects: raw.yes_no("wants_excluded_objects", wants_excluded_objects)?,
            wants_expand,
            ..MeasurementFlags::default()
        },
    );
    if distance_method == "Within a specified distance" {
        let radius = raw.int("Neighbor distance", distance)?;
        if !matches!(radius.as_u64(), Some(n) if (1..=u64::from(u32::MAX)).contains(&n)) {
            return Err(
                raw.error("Neighbor distance must be a positive integer that fits in uint32")
            );
        }
        step["neighbors_distance"] = json!(radius);
    }
    Ok(step)
}

pub(super) fn size_shape(raw: &RawModule<'_>) -> Result<Value, String> {
    let [objects, zernikes, advanced] = raw.exact()?;
    let zernikes = raw.yes_no("calculate_zernikes", zernikes)?;
    let advanced = raw.yes_no("calculate_advanced", advanced)?;
    Ok(measurement_step(
        raw,
        "object_size_shape",
        &[],
        &list(objects),
        MeasurementFlags {
            size_shape: Some((zernikes, advanced)),
            ..MeasurementFlags::default()
        },
    ))
}

pub(super) fn granularity(raw: &RawModule<'_>) -> Result<Value, String> {
    let [images, wants_objects, objects, subsample_size, image_sample_size, element_size, granular_spectrum_length] =
        raw.exact()?;
    let wants_objects = raw.yes_no("'Measure within objects?'", wants_objects)?;
    Ok(json!({
        "images": list(images),
        "objects": if wants_objects { list(objects) } else { Vec::new() },
        "subsample_size": raw.float("subsample_size", subsample_size)?,
        "image_sample_size": raw.float("image_sample_size", image_sample_size)?,
        "element_size": raw.int("element_size", element_size)?,
        "granular_spectrum_length": raw.int("granular_spectrum_length", granular_spectrum_length)?,
    }))
}

pub(super) fn intensity_distribution(raw: &RawModule<'_>) -> Result<Value, String> {
    const FIXED: usize = 6;
    const WIDTH: usize = 3;
    let [images, object_count, bin_count, heatmap_count, zernikes, zernike_degree] =
        raw.leading::<FIXED>()?;
    let nobjects = raw.count("object group count", object_count)?;
    let nbins = raw.count("bin group count", bin_count)?;
    if nobjects < 1 || nbins < 1 {
        return Err(raw.error("object and bin group counts must be positive"));
    }
    if raw.count("heatmap group count", heatmap_count)? != 0 {
        return Err(raw.error("heatmap groups are not supported"));
    }
    let groups = &raw.settings[FIXED..];
    // Compare counts before slicing; malformed counts cannot overflow on wasm32.
    if groups.len() % WIDTH != 0
        || nobjects.checked_add(nbins) != i64::try_from(groups.len() / WIDTH).ok()
    {
        return Err(raw.count_error(format_args!(
            "{FIXED} + {WIDTH} * ({nobjects} object group(s) + {nbins} bin group(s))"
        )));
    }
    let (object_groups, bin_groups) = groups.split_at(nobjects as usize * WIDTH);
    let mut objects = Vec::new();
    let mut object_names = HashSet::new();
    for group in object_groups.chunks_exact(WIDTH) {
        let name = group[0];
        let center = group[1];
        if center != "These objects" {
            return Err(raw.error(format_args!("'Object to use as center?' = {center:?} for {name:?} is not supported (only 'These objects' is)")));
        }
        if !object_names.insert(name) {
            return Err(raw.error(format_args!("duplicate object group {name:?}")));
        }
        objects.push(name);
    }
    let mut bins = Vec::new();
    let mut counts = HashSet::new();
    for group in bin_groups.chunks_exact(WIDTH) {
        let scaled = raw.yes_no("'Scale the bins?'", group[0])?;
        let count = raw.int("bin count", group[1])?;
        let radius = raw.float("maximum radius", group[2])?;
        let count = count
            .as_u64()
            .filter(|&count| count > 0)
            .ok_or_else(|| raw.error("bin count must be positive"))?;
        if !scaled && radius <= 0.0 {
            return Err(raw.error("maximum radius must be finite and positive for unscaled bins"));
        }
        if !counts.insert(count) {
            return Err(raw.error(format_args!("duplicate bin count {count}")));
        }
        bins.push(json!({"bin_count": count, "wants_scaled": scaled, "maximum_radius": radius}));
    }
    let zernikes = raw.choice(
        "'Calculate intensity Zernikes?'",
        zernikes,
        &[
            ("None", "none"),
            ("Magnitudes only", "magnitudes"),
            ("Magnitudes and phase", "magnitudes_and_phase"),
        ],
    )?;
    Ok(json!({
        "images": list(images), "objects": objects,
        "bin_count": bins[0]["bin_count"], "wants_scaled": bins[0]["wants_scaled"],
        "maximum_radius": bins[0]["maximum_radius"], "additional_bin_groups": bins[1..],
        "zernikes": zernikes, "zernike_degree": raw.int("zernike_degree", zernike_degree)?,
    }))
}

pub(super) fn texture(raw: &RawModule<'_>) -> Result<Value, String> {
    const FIXED: usize = 5;
    let [images, objects, gray_levels, scale_count, images_or_objects] = raw.leading::<FIXED>()?;
    let scale_count = raw.count("scale_count", scale_count)?;
    let scales = &raw.settings[FIXED..];
    if usize::try_from(scale_count).ok() != Some(scales.len()) {
        return Err(raw.count_error(format_args!("{FIXED} + {scale_count} scale(s)")));
    }
    let (wants_image, wants_object) = raw.choice(
        "'Measure whole images or objects?'",
        images_or_objects,
        &[
            ("Images", (true, false)),
            ("Objects", (false, true)),
            ("Both", (true, true)),
        ],
    )?;
    let scales = scales
        .iter()
        .map(|s| raw.int("texture scale", s))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "images": list(images), "objects": list(objects),
        "gray_levels": raw.int("gray_levels", gray_levels)?, "scales": scales,
        "wants_image_measurements": wants_image, "wants_object_measurements": wants_object,
    }))
}

pub(super) fn colocalization(raw: &RawModule<'_>) -> Result<Value, String> {
    // The UI-only "Run all metrics?" toggle is ignored. Use each saved flag.
    let [images, threshold_percent, images_or_objects, objects, _do_all, do_corr_and_slope, do_manders, do_rwc, do_overlap, do_costes, fast_costes] =
        raw.exact()?;
    let images = list(images);
    if images.len() < 2 {
        return Err(raw.error(format_args!(
            "needs at least 2 selected images (got {})",
            images.len()
        )));
    }
    let (wants_image, wants_object) = raw.choice(
        "'Select where to measure correlation'",
        images_or_objects,
        &[
            ("Across entire image", (true, false)),
            ("Within objects", (false, true)),
            ("Both", (true, true)),
        ],
    )?;
    let do_corr_and_slope = raw.yes_no("do_corr_and_slope", do_corr_and_slope)?;
    let do_manders = raw.yes_no("do_manders", do_manders)?;
    let do_rwc = raw.yes_no("do_rwc", do_rwc)?;
    let do_overlap = raw.yes_no("do_overlap", do_overlap)?;
    let do_costes = raw.yes_no("do_costes", do_costes)?;
    let fast_costes = raw.choice(
        "'Method for Costes thresholding'",
        fast_costes,
        &[
            ("Faster", "faster"),
            ("Fast", "fast"),
            ("Accurate", "accurate"),
        ],
    )?;
    Ok(json!({
        "images": images, "objects": if wants_object { list(objects) } else { Vec::new() },
        "threshold_percent": raw.float("threshold_percent", threshold_percent)?,
        "fast_costes": fast_costes, "do_corr_and_slope": do_corr_and_slope, "do_manders": do_manders,
        "do_rwc": do_rwc, "do_overlap": do_overlap, "do_costes": do_costes,
        "wants_image_measurements": wants_image, "wants_object_measurements": wants_object,
    }))
}
