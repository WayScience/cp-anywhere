//! Translate CellProfiler 4.2.8 pipeline text into the JSON shape of
//! `dataclasses.asdict(cp_anywhere.cppipe.load_pipeline(...))`.

use serde_json::{json, Number, Value};
use std::collections::BTreeMap;

mod measurement;
mod output;
mod segmentation;

const PIPELINE_FIELDS: [&str; 15] = [
    "segmentation_steps",
    "secondary_steps",
    "tertiary_steps",
    "measurement_steps",
    "granularity_steps",
    "intensity_distribution_steps",
    "texture_steps",
    "colocalization_steps",
    "export_steps",
    "illumination_calculate_steps",
    "illumination_apply_steps",
    "save_images_steps",
    "skipped_modules",
    "image_math_steps",
    "image_step_order",
];

type Translator = fn(&RawModule<'_>) -> Result<Value, String>;

/// Parse a `.cppipe` document without running it or resolving any paths.
///
/// Unknown module names are kept in `skipped_modules`, in file order. Known
/// modules with unsupported revisions or settings return an error.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut pipeline: BTreeMap<_, Vec<Value>> = PIPELINE_FIELDS
        .into_iter()
        .map(|field| (field, Vec::new()))
        .collect();
    for raw in raw_modules(text)? {
        if !raw.enabled {
            pipeline
                .entry("skipped_modules")
                .or_default()
                .push(json!(raw.name));
            continue;
        }
        let (revision, field, translate): (&str, &str, Translator) = match raw.name {
            "IdentifyPrimaryObjects" => ("15", "segmentation_steps", segmentation::primary),
            "IdentifySecondaryObjects" => ("10", "secondary_steps", segmentation::secondary),
            "IdentifyTertiaryObjects" => ("3", "tertiary_steps", segmentation::tertiary),
            "MeasureImageIntensity" => ("4", "measurement_steps", measurement::image_intensity),
            "MeasureObjectIntensity" => ("4", "measurement_steps", measurement::object_intensity),
            "MeasureObjectNeighbors" => ("3", "measurement_steps", measurement::neighbors),
            "MeasureObjectSizeShape" => ("3", "measurement_steps", measurement::size_shape),
            "MeasureGranularity" => ("4", "granularity_steps", measurement::granularity),
            "MeasureObjectIntensityDistribution" => (
                "6",
                "intensity_distribution_steps",
                measurement::intensity_distribution,
            ),
            "MeasureTexture" => ("7", "texture_steps", measurement::texture),
            "MeasureColocalization" => ("5", "colocalization_steps", measurement::colocalization),
            "ExportToDatabase" => ("28", "export_steps", output::export_to_database),
            "CorrectIlluminationCalculate" => (
                "2",
                "illumination_calculate_steps",
                output::illumination_calculate,
            ),
            "CorrectIlluminationApply" => {
                ("5", "illumination_apply_steps", output::illumination_apply)
            }
            "SaveImages" => ("16", "save_images_steps", output::save_images),
            "ImageMath" => ("5", "image_math_steps", output::image_math),
            _ => {
                pipeline
                    .entry("skipped_modules")
                    .or_default()
                    .push(json!(raw.name));
                continue;
            }
        };
        if raw.revision != revision {
            return Err(raw.error(format_args!(
                "variable_revision_number={} is not supported (only {revision} is); re-save the pipeline in a current CellProfiler first",
                raw.revision,
            )));
        }
        pipeline.entry(field).or_default().push(translate(&raw)?);
        if matches!(
            field,
            "image_math_steps" | "illumination_calculate_steps" | "illumination_apply_steps"
        ) {
            pipeline
                .entry("image_step_order")
                .or_default()
                .push(json!(raw.name));
        }
    }
    Ok(json!(pipeline))
}

struct RawModule<'a> {
    name: &'a str,
    revision: &'a str,
    settings: Vec<&'a str>,
    enabled: bool,
}

fn raw_modules(text: &str) -> Result<Vec<RawModule<'_>>, String> {
    let mut lines = pipeline_lines(text.strip_prefix('\u{feff}').unwrap_or(text)).peekable();
    if !lines
        .next()
        .is_some_and(|line| line.starts_with("CellProfiler Pipeline:"))
    {
        return Err("not a CellProfiler .cppipe file (missing header line)".into());
    }

    let mut module_count = None;
    for line in lines.by_ref().take_while(|line| !line.trim().is_empty()) {
        let (key, value) = line.split_once(':').unwrap_or((line, ""));
        if key == "ModuleCount" {
            module_count = Some(
                value
                    .trim()
                    .parse::<i128>()
                    .map_err(|_| format!("invalid ModuleCount: {value:?}"))?,
            );
        }
    }

    let mut modules = Vec::new();
    // ModuleCount limits parsing; it need not equal the number of blocks.
    while module_count.is_none_or(|count| (modules.len() as i128) < count) {
        let Some(header) = lines.by_ref().find(|line| !line.trim().is_empty()) else {
            break;
        };
        let (name, attributes) = header.split_once(':').unwrap_or((header, ""));
        let name = name.trim();
        let revision = revision_number(attributes)
            .ok_or_else(|| format!("module {name:?} is missing variable_revision_number"))?;
        let mut settings = Vec::new();
        while let Some(line) = lines.next_if(|line| !line.trim().is_empty()) {
            // Labels are decorative. Split only once and keep the value verbatim.
            settings.push(line.split_once(':').map_or("", |(_, value)| value));
        }
        modules.push(RawModule {
            name,
            revision,
            settings,
            enabled: !attributes
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split('|')
                .any(|attribute| attribute == "enabled:False"),
        });
    }
    Ok(modules)
}

fn pipeline_lines(mut text: &str) -> impl Iterator<Item = &str> {
    // Python's splitlines handles CRLF as one separator, and accepts CR alone.
    std::iter::from_fn(move || {
        if text.is_empty() {
            return None;
        }
        if let Some(i) = text.find([
            '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
            '\u{2029}',
        ]) {
            let (line, rest) = text.split_at(i);
            let sep = rest.chars().next().unwrap();
            text = &rest[sep.len_utf8()..];
            if sep == '\r' {
                text = text.strip_prefix('\n').unwrap_or(text);
            }
            Some(line)
        } else {
            Some(std::mem::take(&mut text))
        }
    })
}

fn revision_number(attributes: &str) -> Option<&str> {
    const KEY: &str = "variable_revision_number:";
    attributes.match_indices(KEY).find_map(|(start, _)| {
        let raw = &attributes[start + KEY.len()..];
        let end = raw
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(raw.len());
        let digits = &raw[..end];
        if digits.is_empty() {
            return None;
        }
        let normalized = digits.trim_start_matches('0');
        Some(if normalized.is_empty() {
            "0"
        } else {
            normalized
        })
    })
}

impl<'a> RawModule<'a> {
    fn error(&self, reason: impl std::fmt::Display) -> String {
        format!("{}: {reason}", self.name)
    }

    fn count_error(&self, expected: impl std::fmt::Display) -> String {
        self.error(format_args!(
            "has {} settings, expected {expected} for revision {}",
            self.settings.len(),
            self.revision,
        ))
    }

    fn exact<const N: usize>(&self) -> Result<[&'a str; N], String> {
        self.settings
            .as_slice()
            .try_into()
            .map_err(|_| self.count_error(N))
    }

    fn leading<const N: usize>(&self) -> Result<[&'a str; N], String> {
        self.settings
            .first_chunk::<N>()
            .copied()
            .ok_or_else(|| self.count_error(format_args!("at least {N}")))
    }

    fn yes_no(&self, field: &str, value: &str) -> Result<bool, String> {
        match value {
            "Yes" => Ok(true),
            "No" => Ok(false),
            _ => Err(self.error(format_args!(
                "{field} requires a Yes/No value, got {value:?}"
            ))),
        }
    }

    fn int(&self, field: &str, value: &str) -> Result<Number, String> {
        let text = value.trim();
        text.parse::<i64>()
            .map(Number::from)
            .or_else(|_| text.parse::<u64>().map(Number::from))
            .map_err(|_| self.error(format_args!("{field} requires an integer, got {value:?}")))
    }

    fn count(&self, field: &str, value: &str) -> Result<i64, String> {
        value
            .trim()
            .parse()
            .map_err(|_| self.error(format_args!("{field} requires an integer, got {value:?}")))
    }

    fn float(&self, field: &str, value: &str) -> Result<f64, String> {
        value
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite())
            .ok_or_else(|| {
                self.error(format_args!(
                    "{field} requires a finite number, got {value:?}"
                ))
            })
    }

    fn range(&self, field: &str, value: &str) -> Result<[f64; 2], String> {
        let (low, high) = value.split_once(',').unwrap_or((value, ""));
        Ok([self.float(field, low)?, self.float(field, high)?])
    }

    fn choice<T: Copy>(
        &self,
        field: &str,
        value: &str,
        options: &[(&str, T)],
    ) -> Result<T, String> {
        options
            .iter()
            .find_map(|&(label, translated)| (label == value).then_some(translated))
            .ok_or_else(|| self.error(format_args!("unsupported {field}: {value:?}")))
    }
}

fn list(text: &str) -> Vec<&str> {
    // CellProfiler ListSubscriber joins names with a comma and one space.
    text.split(", ").filter(|name| !name.is_empty()).collect()
}
