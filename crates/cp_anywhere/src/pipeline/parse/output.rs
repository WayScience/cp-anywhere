use super::{list, RawModule};
use serde_json::{json, Value};

pub(super) fn illumination_calculate(raw: &RawModule<'_>) -> Result<Value, String> {
    // The remaining settings describe retained intermediate images and spline parameters.
    let [input_image, output_image, intensity_choice, dilate_objects, object_dilation_radius, block_size, rescale_option, each_or_all, smoothing_method, automatic_object_width, object_width, size_of_smoothing_filter, ..] =
        raw.exact::<23>()?;
    match each_or_all {
        "Each" => (),
        "All: First cycle" | "All: Across cycles" => return Err(raw.error(format_args!(
            "each_or_all={each_or_all:?} needs the full image set list; call cp_anywhere.calculate_illumination_across(...) instead"
        ))),
        _ => return Err(raw.error(format_args!("unknown each_or_all {each_or_all:?}"))),
    }
    if !matches!(
        smoothing_method,
        "No smoothing"
            | "Median Filter"
            | "Gaussian Filter"
            | "Fit Polynomial"
            | "Smooth to Average"
    ) {
        return Err(raw.error(format_args!("smoothing_method={smoothing_method:?} is out of scope (Convex Hull and Splines are not yet implemented)")));
    }
    let dilate_objects = raw.yes_no(
        "'Dilate objects in the final averaged image?'",
        dilate_objects,
    )?;
    Ok(json!({
        "input_image": input_image, "output_image": output_image,
        "intensity_choice": intensity_choice, "dilate_objects": dilate_objects,
        "object_dilation_radius": raw.int("object_dilation_radius", object_dilation_radius)?,
        "block_size": raw.int("block_size", block_size)?, "rescale_option": rescale_option,
        "smoothing_method": smoothing_method, "automatic_object_width": automatic_object_width,
        "object_width": raw.float("object_width", object_width)?,
        "size_of_smoothing_filter": raw.float("size_of_smoothing_filter", size_of_smoothing_filter)?,
    }))
}

pub(super) fn illumination_apply(raw: &RawModule<'_>) -> Result<Value, String> {
    let [groups @ .., truncate_low, truncate_high] = raw.settings.as_slice() else {
        return Err(raw
            .error("invalid image-group count plus trailing truncate_low/truncate_high settings"));
    };
    let groups = groups.chunks_exact(4);
    if !groups.remainder().is_empty() {
        return Err(raw.error(format_args!("has {} settings, which is not a valid image-group count plus the trailing truncate_low/truncate_high settings", raw.settings.len())));
    }
    let mut images = Vec::with_capacity(groups.len());
    for group in groups {
        let [input_image, output_image, illum_image, mode] = *group else {
            unreachable!("chunks_exact(4) only yields four settings");
        };
        if !matches!(mode, "Divide" | "Subtract") {
            return Err(raw.error(format_args!("unknown divide_or_subtract {mode:?}")));
        }
        images.push(json!({
            "input_image": input_image, "output_image": output_image,
            "illumination_function_image": illum_image, "divide_or_subtract": mode,
        }));
    }
    Ok(json!({
        "images": images,
        "truncate_low": raw.yes_no("truncate_low", truncate_low)?,
        "truncate_high": raw.yes_no("truncate_high", truncate_high)?,
    }))
}

pub(super) fn save_images(raw: &RawModule<'_>) -> Result<Value, String> {
    let [save_mode, image_name, file_name_method, _file_image_name, file_name, _number_of_digits, _wants_suffix, _suffix, file_format, _pathname, bit_depth, overwrite, ..] =
        raw.exact::<18>()?;
    if save_mode != "Image" {
        return Err(raw.error(format_args!(
            "save_image_or_figure={save_mode:?} is out-of-scope (only 'Image' is supported)"
        )));
    }
    if file_name_method != "Single name" {
        return Err(raw.error(format_args!("file_name_method={file_name_method:?} is out-of-scope (only 'Single name' is supported)")));
    }
    if !matches!(file_format, "tiff" | "png" | "jpeg" | "npy") {
        return Err(raw.error(format_args!("file_format={file_format:?} is out-of-scope (only tiff, png, jpeg, and npy are supported)")));
    }
    Ok(json!({
        "image_name": image_name, "file_name": file_name, "file_format": file_format,
        "bit_depth": bit_depth, "overwrite": raw.yes_no("Overwrite", overwrite)?,
    }))
}

pub(super) fn export_to_database(raw: &RawModule<'_>) -> Result<Value, String> {
    const FIXED: usize = 39;
    let leading = raw.leading::<FIXED>()?;
    // Positional fields 28–31 count repeated CellProfiler Analyst settings.
    let [image_groups, group_fields, filter_fields, workspace_measurements] =
        [leading[28], leading[29], leading[30], leading[31]];
    let counts = [
        ("image_group_count", image_groups, 4),
        ("group_field_count", group_fields, 2),
        ("filter_field_count", filter_fields, 2),
        ("workspace_measurement_count", workspace_measurements, 9),
    ];
    // Four fixed switches are interleaved among the groups. Only the total is used.
    let mut expected = FIXED as i128 + 4;
    for (field, count, settings_per_group) in counts {
        expected += i128::from(raw.count(field, count)?) * settings_per_group;
    }
    if raw.settings.len() as i128 != expected {
        return Err(raw.error(format_args!("has {} settings, expected {expected} for revision {} given its image/group/filter/workspace-measurement counts", raw.settings.len(), raw.revision)));
    }
    let db_type = leading[0];
    if db_type != "SQLite" {
        return Err(raw.error(format_args!(
            "database type {db_type:?} is out of scope (only 'SQLite' is supported)"
        )));
    }
    let separate_object_tables = leading[19];
    if separate_object_tables != "One table per object type" {
        return Err(raw.error(format_args!("'Create one table per object, a single object table or a single object view?' = {separate_object_tables:?} is out of scope (only 'One table per object type' is supported)")));
    }
    if raw.yes_no("'Add a prefix to table names?'", leading[2])? {
        return Err(raw.error("'Add a prefix to table names?' = Yes is out of scope"));
    }
    let objects_choice = raw.choice(
        "'Export measurements for all objects to the database?'",
        leading[16],
        &[("All", "all"), ("None", "none"), ("Select...", "select")],
    )?;
    let if_exists = raw.choice(
        "'Overwrite without warning?'",
        leading[36],
        &[
            ("Data and schema", "replace"),
            ("Data only", "append"),
            ("Never", "append"),
        ],
    )?;
    Ok(json!({
        "sqlite_file": leading[9], "objects_choice": objects_choice,
        "objects": if objects_choice == "select" { list(leading[17]) } else { Vec::new() },
        "if_exists": if_exists,
    }))
}
