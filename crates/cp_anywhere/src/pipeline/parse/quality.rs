use super::{json, RawModule, Value};

pub(super) fn image_quality(raw: &RawModule<'_>) -> Result<Value, String> {
    let [choice, count, rest @ ..] = raw.settings.as_slice() else {
        return Err(raw.error("missing image group count"));
    };
    let all_loaded = match *choice {
        "All loaded images" => true,
        "Select..." => false,
        _ => return Err(raw.error(format_args!("unsupported image choice {choice:?}"))),
    };
    let count = group_count(raw, "image group count", count)?;
    let (sizes, mut settings) = rest
        .split_at_checked(count.saturating_mul(2))
        .ok_or_else(|| raw.error("incomplete image group counts"))?;
    if all_loaded && count != 1 {
        return Err(raw.error("All loaded images requires one image group"));
    }
    let mut groups = Vec::new();
    for pair in sizes.chunks_exact(2) {
        let scales = group_count(raw, "blur scale count", pair[0])?;
        let thresholds = group_count(raw, "threshold count", pair[1])?;
        let size = 7usize
            .saturating_add(scales)
            .saturating_add(thresholds.saturating_mul(5));
        let (group, tail) = settings
            .split_at_checked(size)
            .ok_or_else(|| raw.error("incomplete image quality group"))?;
        settings = tail;
        let images: Vec<_> = if all_loaded {
            Vec::new()
        } else {
            group[0]
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect()
        };
        if !all_loaded && images.is_empty() {
            return Err(raw.error("Select... requires named images"));
        }
        let include_scaling = raw.yes_no("include image scaling", group[1])?;
        let blur_scales = if raw.yes_no("calculate blur", group[2])? {
            group[3..3 + scales]
                .iter()
                .map(|scale| group_count(raw, "blur scale", scale))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let saturation = raw.yes_no("calculate saturation", group[3 + scales])?;
        let intensity = raw.yes_no("calculate intensity", group[4 + scales])?;
        let thresholds = raw.yes_no("calculate thresholds", group[5 + scales])?;
        if intensity || thresholds {
            return Err(raw.error("intensity and threshold metrics are not supported"));
        }
        groups.push(json!({"images": images, "include_scaling": include_scaling,
            "blur_scales": blur_scales, "saturation": saturation}));
    }
    if !settings.is_empty() {
        return Err(raw.error("unexpected settings after the image quality groups"));
    }
    Ok(json!({"all_loaded": all_loaded, "groups": groups}))
}

fn group_count(raw: &RawModule<'_>, field: &str, value: &str) -> Result<usize, String> {
    let count = raw.count(field, value)?;
    usize::try_from(count)
        .ok()
        .filter(|n| (1..=u32::MAX as usize).contains(n))
        .ok_or_else(|| raw.error(format_args!("{field} must be a positive 32-bit integer")))
}
