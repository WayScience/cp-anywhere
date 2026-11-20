//! CellProfiler OverlayOutlines on 2D label planes and grayscale or RGB images.
//!
//! Drawing does not treat the image edge as background. Touching object IDs
//! remain distinct, and grayscale brightness uses the pre-draw channel maximum,
//! not the maximum after RGB-to-gray conversion.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum DisplayMode {
    Color,
    Grayscale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LineMode {
    Inner,
    Outer,
    Thick,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Brightness {
    MaxOfImage,
    MaxPossible,
}

/// Image-plan settings. A missing input image means a blank background.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct OverlayStep {
    pub input_image: Option<String>,
    pub output_image: String,
    pub display_mode: DisplayMode,
    pub line_mode: LineMode,
    pub brightness: Brightness,
    pub outlines: Vec<OutlineGroup>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct OutlineGroup {
    pub objects_name: String,
    pub color: [u8; 3],
}

/// Arithmetic precision of the prepared background. Use Float64 for boolean
/// input, which `img_as_float` promotes to double precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    Float32,
    Float64,
}

#[derive(Clone, Copy, Debug)]
pub struct Background<'a> {
    pub shape: &'a [usize],
    pub values: &'a [f64],
    pub precision: Precision,
}

#[derive(Clone, Copy, Debug)]
pub struct Outline<'a> {
    pub shape: &'a [usize],
    pub labels: &'a [u32],
    pub color: [u8; 3],
}

#[derive(Debug)]
pub struct OverlayImage {
    pub shape: Vec<usize>,
    /// Row-major pixels (interleaved RGB if color), rounded to float32.
    pub values: Vec<f64>,
}

/// Draw ordered outlines over a prepared 2D background, or over a black image
/// sized to the first outline. Later groups replace earlier groups at overlaps.
/// Label planes are cropped or zero-padded at their bottom/right edges.
/// Volumes, non-RGB multichannel images, and overlapping labels within a single
/// object set are outside this dense 2D interface.
pub fn render(
    background: Option<Background<'_>>,
    outlines: &[Outline<'_>],
    display_mode: DisplayMode,
    line_mode: LineMode,
    brightness: Brightness,
) -> Result<OverlayImage, String> {
    let first = outlines
        .first()
        .ok_or("OverlayOutlines requires at least one outline group")?;
    for outline in outlines {
        let [height, width] = *outline.shape else {
            return Err("OverlayOutlines requires 2D label planes".into());
        };
        validate_buffer(height, width, 1, outline.labels.len(), "label")?;
    }
    let (height, width, channels, precision) = match background {
        Some(image) => {
            let (height, width, channels) = match *image.shape {
                [height, width] => (height, width, 1),
                [height, width, 3] => (height, width, 3),
                _ => return Err("OverlayOutlines requires a 2D grayscale or RGB background".into()),
            };
            validate_buffer(height, width, channels, image.values.len(), "background")?;
            (height, width, channels, image.precision)
        }
        None => (first.shape[0], first.shape[1], 1, Precision::Float64),
    };
    let count = height * width;
    let output_channels = if display_mode == DisplayMode::Color {
        3
    } else {
        1
    };
    let output_len = count
        .checked_mul(output_channels)
        .ok_or("OverlayOutlines output dimensions overflow")?;
    let narrow = |sample: f64| match precision {
        Precision::Float32 => f64::from(sample as f32),
        Precision::Float64 => sample,
    };
    let gray_line = match (display_mode, brightness, background) {
        (DisplayMode::Grayscale, Brightness::MaxOfImage, Some(image)) => {
            image
                .values
                .iter()
                .map(|&value| narrow(value))
                .reduce(|a, b| {
                    // NumPy maximum propagates NaN; f64::max ignores a single NaN.
                    if a.is_nan() || a > b {
                        a
                    } else {
                        b
                    }
                })
                .expect("validated nonempty background")
        }
        _ => 1.0,
    };
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_len)
        .map_err(|_| "OverlayOutlines output is too large")?;
    // Build the final image directly. Gray brightness is measured on the
    // original RGB background, but conversion to gray follows drawing in CP.
    for index in 0..count {
        let rgb = if let Some(image) = background {
            if channels == 3 {
                let offset = index * 3;
                [
                    narrow(image.values[offset]),
                    narrow(image.values[offset + 1]),
                    narrow(image.values[offset + 2]),
                ]
            } else {
                [narrow(image.values[index]); 3]
            }
        } else {
            [0.0; 3]
        };
        if display_mode == DisplayMode::Color {
            values.extend(rgb.map(as_output));
        } else {
            values.push(gray(rgb, precision));
        }
    }
    for outline in outlines {
        let line = if display_mode == DisplayMode::Color {
            outline
                .color
                .map(|channel| as_output(f64::from(channel) / 255.0))
        } else {
            [gray([gray_line; 3], precision); 3]
        };
        for row in 0..height {
            for col in 0..width {
                if is_boundary(*outline, height, width, row, col, line_mode) {
                    let offset = (row * width + col) * output_channels;
                    values[offset..offset + output_channels]
                        .copy_from_slice(&line[..output_channels]);
                }
            }
        }
    }
    let mut shape = vec![height, width];
    if output_channels == 3 {
        shape.push(3);
    }
    Ok(OverlayImage { shape, values })
}

fn validate_buffer(
    height: usize,
    width: usize,
    channels: usize,
    len: usize,
    kind: &str,
) -> Result<(), String> {
    if height == 0
        || width == 0
        || height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(channels))
            != Some(len)
    {
        return Err(format!(
            "OverlayOutlines {kind} dimensions must match a nonempty pixel buffer"
        ));
    }
    Ok(())
}

fn as_output(value: f64) -> f64 {
    f64::from(value as f32)
}

fn gray([red, green, blue]: [f64; 3], precision: Precision) -> f64 {
    // Coefficients and source-dtype arithmetic from scikit-image rgb2gray.
    match precision {
        Precision::Float32 => {
            let [red, green, blue] = [red as f32, green as f32, blue as f32];
            f64::from((red * 0.2125 + green * 0.7154) + blue * 0.0721)
        }
        Precision::Float64 => as_output((red * 0.2125 + green * 0.7154) + blue * 0.0721),
    }
}

fn is_boundary(
    outline: Outline<'_>,
    height: usize,
    width: usize,
    row: usize,
    col: usize,
    mode: LineMode,
) -> bool {
    let label_at = |row, col| {
        if row < outline.shape[0] && col < outline.shape[1] {
            outline.labels[row * outline.shape[1] + col]
        } else {
            0
        }
    };
    let label = label_at(row, col);
    if mode == LineMode::Inner && label == 0 {
        return false;
    }
    // Ignore neighbors outside the image (the reflected stencil adds no labels).
    let differs = (row > 0 && label_at(row - 1, col) != label)
        || (row + 1 < height && label_at(row + 1, col) != label)
        || (col > 0 && label_at(row, col - 1) != label)
        || (col + 1 < width && label_at(row, col + 1) != label);
    if !differs || mode != LineMode::Outer || label == 0 {
        return differs;
    }
    // Outer mode keeps a foreground boundary only if two nonzero object
    // labels meet in its 8-neighborhood (including diagonal contacts).
    (row.saturating_sub(1)..=(row + 1).min(height - 1)).any(|y| {
        (col.saturating_sub(1)..=(col + 1).min(width - 1)).any(|x| {
            let other = label_at(y, x);
            other != 0 && other != label
        })
    })
}
