use super::RawModule;
use serde_json::{json, Value};

pub(super) fn primary(raw: &RawModule<'_>) -> Result<Value, String> {
    let [input_image, output_object, size_range, exclude_size, exclude_border_objects, unclump, watershed, smoothing_filter_size, maxima_suppression_size, low_res_maxima, fill_holes, automatic_smoothing, automatic_suppression, limit_choice, maximum_object_count, use_advanced, tail @ ..] =
        raw.exact::<34>()?;
    if !raw.yes_no("use_advanced", use_advanced)? {
        return Err(raw.error("'Use advanced settings?' = No overrides saved settings at run time; toggle it to Yes and re-save"));
    }

    let unclump = raw.choice(
        "unclump_method",
        unclump,
        &[
            ("None", "none"),
            ("Intensity", "intensity"),
            ("Shape", "shape"),
        ],
    )?;
    let fill_holes = raw.choice(
        "fill_holes",
        fill_holes,
        &[
            ("Never", "never"),
            ("After declumping only", "after_declump_only"),
            (
                "After both thresholding and declumping",
                "after_thresholding_and_declump",
            ),
        ],
    )?;
    let erase = raw.choice(
        "limit_choice",
        limit_choice,
        &[("Continue", false), ("Erase", true)],
    )?;
    let [min_diameter, max_diameter] = raw.range("size_range", size_range)?;
    let automatic_smoothing = raw.yes_no("automatic_smoothing", automatic_smoothing)?;
    let automatic_suppression = raw.yes_no("automatic_suppression", automatic_suppression)?;
    let mut config = threshold(raw, tail)?;

    config.insert(
        "exclude_size",
        json!(raw.yes_no("exclude_size", exclude_size)?),
    );
    config.insert("min_diameter", json!(min_diameter));
    config.insert("max_diameter", json!(max_diameter));
    config.insert(
        "exclude_border_objects",
        json!(raw.yes_no("exclude_border_objects", exclude_border_objects)?),
    );
    config.insert(
        "maximum_object_count",
        json!(if erase {
            Some(raw.int("maximum_object_count", maximum_object_count)?)
        } else {
            None
        }),
    );
    config.insert("fill_holes", json!(fill_holes));
    config.insert("unclump_method", json!(unclump));
    config.insert(
        "watershed_method",
        json!(raw.choice(
            "watershed_method",
            watershed,
            &[
                ("None", "none"),
                ("Intensity", "intensity"),
                ("Shape", "shape"),
                ("Propagate", "propagate"),
            ]
        )?),
    );
    config.insert(
        "smoothing_filter_size",
        json!(if automatic_smoothing {
            None
        } else {
            Some(raw.float("smoothing_filter_size", smoothing_filter_size)?)
        }),
    );
    config.insert(
        "maxima_suppression_size",
        json!(if automatic_suppression {
            None
        } else {
            Some(raw.float("maxima_suppression_size", maxima_suppression_size)?)
        }),
    );
    config.insert(
        "low_resolution_maxima",
        json!(raw.yes_no("low_res_maxima", low_res_maxima)?),
    );
    Ok(json!({ "input_image": input_image, "output_object": output_object, "config": config }))
}

pub(super) fn secondary(raw: &RawModule<'_>) -> Result<Value, String> {
    let [primary_object, output_object, method, input_image, distance_to_dilate, regularization_factor, wants_discard_edge, _wants_discard_primary, _new_primary_objects_name, fill_holes, tail @ ..] =
        raw.exact::<28>()?;
    let method = raw.choice(
        "method",
        method,
        &[
            ("Propagation", "propagation"),
            ("Distance - N", "distance_n"),
        ],
    )?;
    if raw.yes_no("wants_discard_edge", wants_discard_edge)? {
        return Err(raw.error("'Discard secondary objects touching the border of the image?' = Yes is not yet implemented"));
    }
    let fill_holes = raw.yes_no("fill_holes", fill_holes)?;
    // Distance - N uses the seed geometry only; its saved Threshold tail is ignored.
    let mut config = if method == "propagation" {
        let mut config = threshold(raw, tail)?;
        config.insert(
            "regularization_factor",
            json!(raw.float("regularization_factor", regularization_factor)?),
        );
        config
    } else {
        BTreeMap::from([(
            "distance_to_dilate",
            json!(raw.float("distance_to_dilate", distance_to_dilate)?),
        )])
    };
    config.insert("method", json!(method));
    config.insert("fill_holes", json!(fill_holes));
    Ok(
        json!({ "input_image": input_image, "primary_object": primary_object,
        "output_object": output_object, "config": config }),
    )
}

pub(super) fn tertiary(raw: &RawModule<'_>) -> Result<Value, String> {
    // Revision 3 saves the larger (secondary) object before the smaller (primary).
    let [secondary_object, primary_object, output_object, shrink] = raw.exact()?;
    Ok(
        json!({ "primary_object": primary_object, "secondary_object": secondary_object,
        "output_object": output_object, "shrink_primary": raw.yes_no("shrink_primary", shrink)? }),
    )
}

use std::collections::BTreeMap;

// Both segmentation modules embed the same 18 settings from Threshold.
fn threshold(
    raw: &RawModule<'_>,
    tail: [&str; 18],
) -> Result<BTreeMap<&'static str, Value>, String> {
    let [version, scope, global_operation, smoothing_scale, correction_factor, range, manual_threshold, _thresholding_measurement, two_class_otsu, log_transform, assign_middle, adaptive_window_size, lower_outlier_fraction, upper_outlier_fraction, averaging_method, variance_method, number_of_deviations, local_operation] =
        tail;
    if raw.int("threshold_setting_version", version)?.as_i64() != Some(12) {
        return Err(raw.error(format_args!(
            "embedded Threshold setting version={version} is not supported (only 12 is)"
        )));
    }
    let (scope, operation) = match scope {
        "Global" => ("global", global_operation),
        "Adaptive" => ("adaptive", local_operation),
        _ => return Err(raw.error(format_args!("unsupported threshold_scope: {scope:?}"))),
    };
    let method = raw.choice(
        "threshold operation",
        operation,
        &[
            ("Manual", "manual"),
            ("Otsu", "otsu"),
            ("Minimum Cross-Entropy", "li"),
            ("Robust Background", "robust_background"),
            ("Sauvola", "sauvola"),
        ],
    )?;
    // The reference checks these choices even when the threshold is not Robust Background.
    let averaging_method = raw.choice(
        "averaging_method",
        averaging_method,
        &[("Mean", "mean"), ("Median", "median"), ("Mode", "mode")],
    )?;
    let variance_method = raw.choice(
        "variance_method",
        variance_method,
        &[
            ("Standard deviation", "sd"),
            ("Median absolute deviation", "mad"),
        ],
    )?;
    let mut config = BTreeMap::from([
        (
            "threshold_correction_factor",
            json!(raw.float("threshold_correction_factor", correction_factor)?),
        ),
        (
            "threshold_range",
            json!(raw.range("threshold_range", range)?),
        ),
        (
            "threshold_smoothing_scale",
            json!(raw.float("threshold_smoothing_scale", smoothing_scale)?),
        ),
        (
            "log_transform",
            json!(raw.yes_no("log_transform", log_transform)?),
        ),
        ("threshold_scope", json!(scope)),
        (
            "adaptive_window_size",
            json!(raw.int("adaptive_window_size", adaptive_window_size)?),
        ),
    ]);
    let threshold = match method {
        "manual" => json!(raw.float("manual_threshold", manual_threshold)?),
        "otsu" if two_class_otsu == "Three classes" => {
            config.insert(
                "assign_middle_to_foreground",
                json!(assign_middle == "Foreground"),
            );
            json!("otsu_three_class")
        }
        "robust_background" => {
            config.insert(
                "robust_background_lower_outlier_fraction",
                json!(raw.float("lower_outlier_fraction", lower_outlier_fraction)?),
            );
            config.insert(
                "robust_background_upper_outlier_fraction",
                json!(raw.float("upper_outlier_fraction", upper_outlier_fraction)?),
            );
            config.insert(
                "robust_background_averaging_method",
                json!(averaging_method),
            );
            config.insert("robust_background_variance_method", json!(variance_method));
            config.insert(
                "robust_background_number_of_deviations",
                json!(raw.float("number_of_deviations", number_of_deviations)?),
            );
            json!(method)
        }
        _ => json!(method),
    };
    config.insert("threshold", threshold);
    Ok(config)
}
