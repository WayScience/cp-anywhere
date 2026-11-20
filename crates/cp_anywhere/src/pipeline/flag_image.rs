//! Whole-image threshold flags used by the CellProfiler pipeline runner.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Step {
    flags: Vec<Flag>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Flag {
    name: String,
    all: bool,
    skip: bool,
    criteria: Vec<Criterion>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Criterion {
    measurement: String,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

pub(super) fn parse(settings: &[&str]) -> Result<Step, String> {
    fn count(value: &str) -> Result<usize, String> {
        value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| "expected a positive group count".into())
    }

    fn bound(enabled: &str, value: &str) -> Result<Option<f64>, String> {
        match enabled {
            "No" => Ok(None),
            "Yes" => value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .map(Some)
                .ok_or_else(|| "threshold must be finite".into()),
            _ => Err("expected a Yes/No threshold switch".into()),
        }
    }

    let [flag_count, remaining @ .., last] = settings else {
        return Err("missing flag count or last-cycle setting".into());
    };
    if *last != "No" {
        return Err("requires 'Ignore flag skips on last cycle?' = No".into());
    }
    let flag_count = count(flag_count)?;
    let mut rest = remaining;
    let mut flags = Vec::new();
    for _ in 0..flag_count {
        let [criteria_count, category, name, combination, skip, tail @ ..] = rest else {
            return Err("incomplete flag group".into());
        };
        rest = tail;
        let all = match *combination {
            "Flag if all fail" => true,
            "Flag if any fail" => false,
            _ => return Err(format!("unsupported combination {combination:?}")),
        };
        let skip = match *skip {
            "Yes" => true,
            "No" => false,
            _ => return Err("expected a Yes/No skip switch".into()),
        };
        if category.trim().is_empty() || name.trim().is_empty() {
            return Err("flag category and name must not be empty".into());
        }
        let mut criteria = Vec::new();
        for _ in 0..count(criteria_count)? {
            let [source, _, measurement, wants_min, minimum, wants_max, maximum, _, _, _, _, tail @ ..] =
                rest
            else {
                return Err("incomplete measurement group".into());
            };
            rest = tail;
            if *source != "Whole-image measurement" {
                return Err(format!("unsupported flag source {source:?}"));
            }
            if measurement.trim().is_empty() || *measurement == "None" {
                return Err("flag requires a named measurement".into());
            }
            criteria.push(Criterion {
                measurement: (*measurement).to_owned(),
                minimum: bound(wants_min, minimum)?,
                maximum: bound(wants_max, maximum)?,
            });
        }
        flags.push(Flag {
            name: format!("{category}_{name}"),
            all,
            skip,
            criteria,
        });
    }
    if !rest.is_empty() {
        return Err("unexpected settings after the flag groups".into());
    }
    Ok(Step { flags })
}

type FlagValues = Vec<(String, i64)>;

pub(super) fn evaluate(
    step: &Step,
    mut measurement: impl FnMut(&str) -> Option<f64>,
) -> Result<(FlagValues, bool), String> {
    let mut values = FlagValues::new();
    let mut skip = false;
    for flag in &step.flags {
        let mut failed = flag.all;
        for criterion in &flag.criteria {
            let value = values
                .iter()
                .rev()
                .find(|(name, _)| *name == criterion.measurement)
                .map(|(_, value)| *value as f64)
                .or_else(|| measurement(&criterion.measurement))
                .ok_or_else(|| {
                    format!(
                        "FlagImage needs a prior image measurement {:?}",
                        criterion.measurement
                    )
                })?;
            let outside = criterion.minimum.is_some_and(|min| value < min)
                || criterion.maximum.is_some_and(|max| value > max);
            failed = if flag.all {
                failed && outside
            } else {
                failed || outside
            };
        }
        skip |= flag.skip && failed;
        values.push((flag.name.clone(), i64::from(failed)));
    }
    Ok((values, skip))
}

#[cfg(test)]
mod tests {
    use super::{evaluate, parse};
    use serde_json::json;

    #[test]
    fn matches_cellprofiler_428_whole_image_flags() {
        // Reference: real FlagImage.run on injected whole-image measurements.
        let cases = [
            (-3.0, 4.0, false, false),
            (-2.0, 4.001, true, false),
            (-3.001, 1.0, true, false),
            (-3.001, 4.001, true, true),
            (f64::NAN, 1.0, false, false),
        ];
        for all in [false, true] {
            let settings = [
                "1",
                "2",
                "QC",
                "Rejected",
                if all {
                    "Flag if all fail"
                } else {
                    "Flag if any fail"
                },
                "Yes",
                "Whole-image measurement",
                "None",
                "Slope",
                "Yes",
                "-3",
                "Yes",
                "-1",
                "Elsewhere...|",
                "rules.txt",
                "",
                "No",
                "Whole-image measurement",
                "None",
                "Percent",
                "No",
                "0",
                "Yes",
                "4",
                "Elsewhere...|",
                "rules.txt",
                "",
                "No",
                "No",
            ];
            let step = parse(&settings).unwrap();
            for (slope, percent, any_fail, all_fail) in cases {
                let expected = if all { all_fail } else { any_fail };
                let (flags, skip) = evaluate(&step, |name| match name {
                    "Slope" => Some(slope),
                    "Percent" => Some(percent),
                    _ => None,
                })
                .unwrap();
                assert_eq!(flags, [("QC_Rejected".into(), i64::from(expected))]);
                assert_eq!(skip, expected);
            }
            assert!(evaluate(&step, |name| (name == "Slope").then_some(-2.0)).is_err());
            assert_eq!(
                serde_json::to_value(&step).unwrap()["flags"][0]["name"],
                json!("QC_Rejected")
            );
        }
    }
}
