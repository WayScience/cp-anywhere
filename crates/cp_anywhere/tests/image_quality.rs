use cp_anywhere::image_quality::{measure, Options};

fn check_oracle(select: impl Fn(&str) -> bool, select_key: impl Fn(&str) -> bool) {
    let mut failures = Vec::new();
    let mut seen = 0;
    for row in include_str!("image_quality_data/oracle.tsv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split('\t').collect();
        let name = fields[0];
        if !select(name) {
            continue;
        }
        seen += 1;
        let height: usize = fields[1].parse().unwrap();
        let width: usize = fields[2].parse().unwrap();
        let mut pixels = Vec::new();
        let mut mask = Vec::new();
        for r in 0..height {
            for c in 0..width {
                pixels.push(match fields[4] {
                    "constant" => 0.25,
                    "impulse" => f64::from(r == 4 && c == 7) * 0.5,
                    "stripe" => (c % 2) as f64 * 0.5,
                    "disk_grid" => {
                        let dy = (r % 41) as i32 - 18;
                        let dx = (c % 47) as i32 - 20;
                        f64::from(dy * dy + dx * dx < 128) * 0.5
                    }
                    "quantize_ties" => f64::from(f32::from_bits(
                        [0x3dc1_bd6d, 0x3e80_0000, 0x3ef3_aa97, 0x3f5b_72ea]
                            [(r + 2 * c + r * c) % 4],
                    )),
                    "nonfinite" if r == 0 && c == 0 => f64::NAN,
                    "nonfinite" | "pattern" => {
                        ((39 * r + 11 * c + 7 * r * c) % 53 + (r + c) % 5) as f64 / 256.0
                    }
                    pattern => panic!("unexpected oracle pattern {pattern}"),
                });
                mask.push(match fields[5] {
                    "empty" => false,
                    "holed" => (r * 3 + c * 7) % 17 > 1 && (r > 20 || c > 21),
                    "none" => true,
                    kind => panic!("unexpected oracle mask {kind}"),
                });
            }
        }
        let scales: Vec<usize> = match fields[6] {
            "-" => Vec::new(),
            scales => scales.split(',').map(|s| s.parse().unwrap()).collect(),
        };
        let actual = measure(
            &[height, width],
            &pixels,
            (fields[5] != "none").then_some(mask.as_slice()),
            "Orig_DNA",
            Options {
                float32: fields[3] == "float32",
                blur_scales: &scales,
                saturation: fields[7] == "1",
                include_scaling: fields[8] == "1",
                scaling: (fields[9] != "-").then(|| fields[9].parse().unwrap()),
            },
        )
        .unwrap();
        let actual: std::collections::BTreeMap<_, _> = actual
            .into_iter()
            .filter(|(key, _)| select_key(key))
            .collect();
        let expected: Vec<_> = fields[10..]
            .iter()
            .map(|entry| entry.split_once('=').unwrap())
            .filter(|(key, _)| select_key(key))
            .collect();
        if actual.len() != expected.len() {
            failures.push(format!(
                "{name}: wrong column count; actual columns {actual:?}"
            ));
        }
        for (key, expected) in expected {
            let Some(&actual) = actual.get(key) else {
                failures.push(format!("{name}: missing {key}"));
                continue;
            };
            let expected: f64 = expected.parse().unwrap();
            let tolerance = if key.contains("PowerLogLogSlope") {
                // SciPy uses a complex32 FFT for f32 inputs; the standalone
                // Rust FFT accumulates in f64 before rounding spectral power.
                2e-6 + expected.abs() * 2e-6
            } else {
                1e-12 + expected.abs() * 1e-9
            };
            if !(actual == expected
                || (actual.is_nan() && expected.is_nan())
                || (actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= tolerance))
            {
                failures.push(format!("{name}/{key}: {actual:?} != {expected:?}"));
            }
        }
    }
    assert!(seen > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn agrees_with_cellprofiler_radial_power_spectrum() {
    check_oracle(
        |name| !name.starts_with("options") && name != "nothing_selected",
        |key| key.contains("PowerLogLogSlope"),
    );
}

#[test]
fn agrees_with_cellprofiler_focus_and_correlation() {
    check_oracle(
        |name| !name.starts_with("options") && name != "nothing_selected",
        |key| !key.contains("PowerLogLogSlope"),
    );
}

#[test]
fn agrees_with_cellprofiler_saturation_and_loader_scaling() {
    check_oracle(
        |name| name.starts_with("options") || name == "nothing_selected",
        |_| true,
    );
}

#[test]
fn rejects_inconsistent_plane_or_mask_sizes() {
    let opts = Options::default();
    assert!(measure(&[2, 2, 1], &[0.0; 4], None, "DNA", opts).is_err());
    assert!(measure(&[2, 0], &[], None, "DNA", opts).is_err());
    assert!(measure(&[usize::MAX, 2], &[], None, "DNA", opts).is_err());
    assert!(measure(&[2, 2], &[0.0; 3], None, "DNA", opts).is_err());
    assert!(measure(&[2, 2], &[0.0; 4], Some(&[true; 3]), "DNA", opts).is_err());
    assert!(measure(
        &[2, 2],
        &[0.0; 4],
        None,
        "DNA",
        Options {
            blur_scales: &[0],
            ..opts
        },
    )
    .is_err());
}
