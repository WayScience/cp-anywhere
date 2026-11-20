//! 2D CellProfiler 4.2.8 image quality measurements.
//!
//! Numerical reference: `MeasureImageQuality` at CellProfiler
//! `2df090ff281cff80943fe679a1c3192fbda9cd6d` and centrosome 1.2.3.
use std::{borrow::Cow, collections::BTreeMap};
/// Options for the image quality metrics enabled by the surveyed pipelines.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options<'a> {
    /// An empty slice disables blur measurements. Otherwise, each scale is in pixels.
    pub blur_scales: &'a [usize],
    /// Measure the percentage of included pixels at each observed extreme.
    pub saturation: bool,
    /// Report the acquisition scale, even when it is not available.
    pub include_scaling: bool,
    /// Acquisition scale supplied by the loader, not inferred from the pixels.
    pub scaling: Option<f64>,
    /// The prepared pixels represent a float32 CellProfiler image.
    pub float32: bool,
}
/// Measure a prepared, row-major grayscale plane with shape `[height, width]`.
/// A `true` mask entry includes its pixel. Keys are full `ImageQuality_*`
/// measurement names. Missing or zero acquisition scales are reported as NaN.
/// The spectral slope can differ from CellProfiler when fitted bands are
/// dominated by FFT rounding error. It is not a bit-exact PocketFFT port.
pub fn measure(
    shape: &[usize],
    pixels: &[f64],
    mask: Option<&[bool]>,
    image_name: &str,
    options: Options<'_>,
) -> Result<BTreeMap<String, f64>, String> {
    let &[height, width] = shape else {
        return Err("MeasureImageQuality requires a 2D grayscale plane".into());
    };
    if height == 0 || width == 0 || height.checked_mul(width) != Some(pixels.len()) {
        return Err(
            "MeasureImageQuality pixel buffer length must match positive dimensions".into(),
        );
    }
    if mask.is_some_and(|mask| mask.len() != pixels.len()) {
        return Err("MeasureImageQuality mask length must match pixel buffer length".into());
    }
    if options.blur_scales.contains(&0) {
        return Err("MeasureImageQuality blur scales must be positive".into());
    }
    let mut result = BTreeMap::new();
    let mut put = |metric, scale, value| {
        let key = match scale {
            Some(scale) => format!("ImageQuality_{metric}_{image_name}_{scale}"),
            None => format!("ImageQuality_{metric}_{image_name}"),
        };
        result.insert(key, value);
    };
    if options.include_scaling {
        put(
            "Scaling",
            None,
            options.scaling.filter(|&x| x != 0.0).unwrap_or(f64::NAN),
        );
    }
    if !options.saturation && options.blur_scales.is_empty() {
        return Ok(result);
    }
    let pixels = if options.float32 {
        Cow::Owned(
            pixels
                .iter()
                .map(|&value| f64::from(value as f32))
                .collect(),
        )
    } else {
        Cow::Borrowed(pixels)
    };
    let included = match mask {
        Some(mask) => Cow::Owned(
            pixels
                .iter()
                .zip(mask)
                .filter_map(|(&p, &keep)| keep.then_some(p))
                .collect(),
        ),
        None => Cow::Borrowed(pixels.as_ref()),
    };
    if options.saturation {
        let (maximum, minimum) = saturation(&included);
        put("PercentMaximal", None, maximum);
        put("PercentMinimal", None, minimum);
    }
    if !options.blur_scales.is_empty() {
        put("FocusScore", None, focus_score(&included, options.float32));
        put(
            "PowerLogLogSlope",
            None,
            power_log_log_slope(&pixels, &included, mask, height, width, options.float32)?,
        );
        let quantized = quantize(&pixels, &included, options.float32);
        for &scale in options.blur_scales {
            put(
                "LocalFocusScore",
                Some(scale),
                local_focus_score(&pixels, mask, height, width, scale),
            );
            put(
                "Correlation",
                Some(scale),
                correlation(&quantized, mask, width, scale),
            );
        }
    }
    Ok(result)
}
fn saturation(pixels: &[f64]) -> (f64, f64) {
    if pixels.is_empty() || pixels.iter().any(|p| p.is_nan()) {
        return (0.0, 0.0);
    }
    let (min, max) = extrema(pixels);
    let percent =
        |value| 100.0 * pixels.iter().filter(|&&p| p == value).count() as f64 / pixels.len() as f64;
    (percent(max), percent(min))
}
fn extrema(pixels: &[f64]) -> (f64, f64) {
    pixels
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &value| {
            (min.min(value), max.max(value))
        })
}

fn focus_score(included: &[f64], float32: bool) -> f64 {
    if float32 {
        let mut pixels: Vec<_> = included.iter().map(|&p| p as f32).collect();
        let mean = crate::numpy_sum_f32(&pixels) / pixels.len() as f32;
        if mean > 0.0 {
            pixels.iter_mut().for_each(|x| {
                let delta = *x - mean;
                *x = delta * delta;
            });
            return f64::from(crate::numpy_sum_f32(&pixels))
                / (pixels.len() as f64 * f64::from(mean));
        }
    } else {
        let mean = mean(included);
        if mean > 0.0 {
            return variance(included, mean) / mean;
        }
    }
    0.0
}

fn local_focus_score(
    pixels: &[f64],
    mask: Option<&[bool]>,
    height: usize,
    width: usize,
    scale: usize,
) -> f64 {
    let rows = height.div_ceil(scale);
    let cols = width.div_ceil(scale);
    let mut sums = vec![0.0; rows * cols];
    let mut counts = vec![0; sums.len()];
    let grid = (0..height).flat_map(|r| {
        // Distribute cells evenly rather than leaving a narrow final cell.
        let row = (r as f64 * rows as f64 / height as f64) as usize;
        (0..width).map(move |c| row * cols + (c as f64 * cols as f64 / width as f64) as usize)
    });
    for (i, (label, &value)) in grid.clone().zip(pixels).enumerate() {
        if mask.is_none_or(|m| m[i]) {
            sums[label] += value;
            counts[label] += 1;
        }
    }
    let means: Vec<_> = sums
        .into_iter()
        .zip(&counts)
        .map(|(s, &n)| s / n as f64)
        .collect();
    let mut squared = vec![0.0; means.len()];
    for (i, (label, &value)) in grid.zip(pixels).enumerate() {
        if mask.is_none_or(|m| m[i]) {
            let delta = value - means[label];
            squared[label] += delta * delta;
        }
    }
    let local_var: Vec<_> = squared
        .into_iter()
        .zip(counts)
        .zip(means)
        .filter_map(|((sum, count), mean)| {
            (mean != 0.0 && !mean.is_nan()).then_some(sum / (count as f64 * mean))
        })
        .collect();
    let median = median(&mut local_var.clone(), 2.0, f64::total_cmp).unwrap_or(f64::NAN);
    if median.is_finite() && median > 0.0 {
        variance(&local_var, mean(&local_var)) / median
    } else {
        0.0
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64], mean: f64) -> f64 {
    values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / values.len() as f64
}

fn median<T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Div<Output = T>>(
    values: &mut [T],
    two: T,
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> Option<T> {
    if values.is_empty() || values.iter().any(|&x| x.partial_cmp(&x).is_none()) {
        return None;
    }
    let mid = values.len() / 2;
    let even = values.len() % 2 == 0;
    let (lower, &mut upper, _) = values.select_nth_unstable_by(mid, &compare);
    Some(if even {
        (*lower.iter().max_by(|a, b| compare(a, b)).unwrap() + upper) / two
    } else {
        upper
    })
}

const LEVELS: usize = 8;

fn quantize(pixels: &[f64], included: &[f64], float32: bool) -> Vec<u8> {
    let (min, max) = extrema(included);
    // SciPy sorts NaNs last for extrema. Centrosome uses a divisor of 1
    // when max > min is false, including when the maximum is NaN.
    let range = if max > min && !included.iter().any(|x| x.is_nan()) {
        if float32 {
            f64::from(max as f32 - min as f32)
        } else {
            max - min
        }
    } else {
        1.0
    };
    pixels
        .iter()
        .map(|&p| {
            // SciPy extrema have the image dtype, so Centrosome subtracts
            // the minimum in that dtype before dividing by a float64 range.
            let offset = if float32 {
                f64::from(p as f32 - min as f32)
            } else {
                p - min
            };
            // Centrosome casts the floored bin to a signed byte before clipping.
            let level = (offset / range * LEVELS as f64).floor();
            if level.is_finite() {
                ((level % 256.0) as i32 as i8).clamp(0, LEVELS as i8 - 1) as u8
            } else {
                0
            }
        })
        .collect()
}

fn correlation(quantized: &[u8], mask: Option<&[bool]>, width: usize, scale: usize) -> f64 {
    if scale >= width {
        return 0.0;
    }
    let mut histogram = [[0usize; LEVELS]; LEVELS];
    let mut pairs = 0;
    for (r, row) in quantized.chunks_exact(width).enumerate() {
        for c in 0..width - scale {
            let i = r * width + c;
            if mask.is_none_or(|m| m[i] && m[i + scale]) {
                histogram[row[c] as usize][row[c + scale] as usize] += 1;
                pairs += 1;
            }
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    let mut px = [0.0; LEVELS];
    let mut py = [0.0; LEVELS];
    let mut cross = [0.0; LEVELS];
    for (x, row) in histogram.iter().enumerate() {
        let mut products = [0.0; LEVELS];
        let probabilities = row.map(|n| n as f64 / pairs as f64);
        px[x] = sum_bins(&probabilities);
        for (y, p) in probabilities.into_iter().enumerate() {
            py[y] += p;
            products[y] = ((x + 1) * (y + 1)) as f64 * p;
        }
        cross[x] = sum_bins(&products);
    }
    let (mx, sx) = marginal_moments(px);
    let (my, sy) = marginal_moments(py);
    let value = (sum_bins(&cross) - mx * my) / (sx * sy);
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn sum_bins([a, b, c, d, e, f, g, h]: &[f64; LEVELS]) -> f64 {
    ((a + b) + (c + d)) + ((e + f) + (g + h))
}

fn marginal_moments(mut marginal: [f64; LEVELS]) -> (f64, f64) {
    let total = sum_bins(&marginal);
    marginal.iter_mut().for_each(|p| *p /= total);
    let average = sum_bins(&std::array::from_fn(|i| (i + 1) as f64 * marginal[i]));
    let variance = sum_bins(&std::array::from_fn(|i| {
        let delta = (i + 1) as f64 - average;
        delta * delta * marginal[i]
    }));
    (average, variance.sqrt())
}

fn power_log_log_slope(
    pixels: &[f64],
    included: &[f64],
    mask: Option<&[bool]>,
    height: usize,
    width: usize,
    float32: bool,
) -> Result<f64, String> {
    // Centrosome reflects the radius grid at the last pixel, not the FFT
    // period. All four corner bins have label 1 and are excluded, as are
    // labels >= floor(min(height, width) / 8).
    let limit = height.min(width) / 8;
    if limit <= 3 || included.is_empty() || included.iter().any(|x| !x.is_finite()) {
        return Ok(0.0);
    }
    let (min, max) = extrema(included);
    if min == max {
        return Ok(0.0);
    }
    let mut centered = if float32 {
        let mut field: Vec<_> = pixels.iter().map(|&p| p as f32).collect();
        if let Some(mask) = mask {
            let masked: Vec<_> = included.iter().map(|&p| p as f32).collect();
            let average = crate::numpy_sum_f32(&masked) / masked.len() as f32;
            for (p, &keep) in field.iter_mut().zip(mask) {
                if !keep {
                    *p = average;
                }
            }
        }
        let average = crate::numpy_sum_f32(&field) / field.len() as f32;
        let mut deviations: Vec<_> = field.iter().map(|&p| (p - average).abs()).collect();
        if let Some(mad) = median(&mut deviations, 2.0, f32::total_cmp) {
            field.iter_mut().for_each(|p| *p /= mad);
        }
        let average = crate::numpy_sum_f32(&field) / field.len() as f32;
        field
            .into_iter()
            .map(|p| Complex::real(f64::from(p - average)))
            .collect::<Vec<_>>()
    } else {
        let mut field = pixels.to_vec();
        if let Some(mask) = mask {
            let average = mean(included);
            for (p, &keep) in field.iter_mut().zip(mask) {
                if !keep {
                    *p = average;
                }
            }
        }
        let average = mean(&field);
        let mut deviations: Vec<_> = field.iter().map(|&p| (p - average).abs()).collect();
        if let Some(mad) = median(&mut deviations, 2.0, f64::total_cmp) {
            field.iter_mut().for_each(|p| *p /= mad);
        }
        let average = mean(&field);
        field
            .into_iter()
            .map(|p| Complex::real(p - average))
            .collect::<Vec<_>>()
    };
    // A zero MAD makes background normalization nonfinite; SciPy then
    // returns a spectrum with NaN magnitude and CP reports zero.
    if centered.iter().any(|p| !p.re.is_finite()) {
        return Ok(0.0);
    }
    fft_2d(&mut centered, height, width)?;
    let mut magnitude = vec![0.0; limit];
    let mut power = vec![0.0; limit];
    for (y, row) in centered.chunks_exact(width).enumerate() {
        let dy = y.min(height - 1 - y) as f64;
        if dy >= (limit - 1) as f64 {
            continue;
        }
        for (x, p) in row.iter().enumerate() {
            let dx = x.min(width - 1 - x) as f64;
            let label = (dx * dx + dy * dy).sqrt() as usize + 1;
            if (2..limit).contains(&label) {
                let mag = p.re.hypot(p.im);
                let (mag, pwr) = if float32 {
                    let mag = mag as f32;
                    (f64::from(mag), f64::from(mag * mag))
                } else {
                    (mag, mag * mag)
                };
                magnitude[label] += mag;
                power[label] += pwr;
            }
        }
    }
    if magnitude.iter().sum::<f64>().is_nan() {
        return Ok(0.0);
    }
    let count = magnitude.iter().filter(|&&m| m > 0.0).count();
    if count <= 1 {
        return Ok(0.0);
    }
    let points: Vec<_> = magnitude
        .into_iter()
        .zip(power)
        .enumerate()
        .filter_map(|(label, (mag, pwr))| {
            let log_power = pwr.ln();
            (mag > 0.0 && log_power.is_finite()).then(|| ((label as f64).ln(), log_power))
        })
        .collect();
    // Least-squares fit with intercept, using centered coordinates. A
    // single finite row has the minimum-norm solution used by SciPy.
    Ok(match points.as_slice() {
        [] => 0.0,
        &[(x, y)] => x * y / (x * x + 1.0),
        _ => {
            let mx = points.iter().map(|&(x, _)| x).sum::<f64>() / points.len() as f64;
            let my = points.iter().map(|&(_, y)| y).sum::<f64>() / points.len() as f64;
            let covariance = points
                .iter()
                .map(|&(x, y)| (x - mx) * (y - my))
                .sum::<f64>();
            let variance = points
                .iter()
                .map(|&(x, _)| (x - mx) * (x - mx))
                .sum::<f64>();
            covariance / variance
        }
    })
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    fn phase(angle: f64) -> Self {
        let (im, re) = angle.sin_cos();
        Self { re, im }
    }
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

fn fft_2d(field: &mut [Complex], height: usize, width: usize) -> Result<(), String> {
    let rows = Fft::new(width)?;
    let columns = Fft::new(height)?;
    let mut input = vec![Complex::default(); height];
    let mut output = vec![Complex::default(); height.max(width)];
    let mut scratch = vec![Complex::default(); rows.scratch_len.max(columns.scratch_len)];
    for row in field.chunks_exact_mut(width) {
        rows.transform(row, 1, &mut output[..width], &mut scratch);
        row.copy_from_slice(&output[..width]);
    }
    for x in 0..width {
        for (row, dst) in field.chunks_exact(width).zip(&mut input) {
            *dst = row[x];
        }
        columns.transform(&input, 1, &mut output[..height], &mut scratch);
        for (row, &src) in field.chunks_exact_mut(width).zip(&output) {
            row[x] = src;
        }
    }
    Ok(())
}

struct Fft {
    scratch_len: usize,
    algorithm: FftAlgorithm,
}

enum FftAlgorithm {
    Radix2(Vec<Complex>),
    Factor {
        radix: usize,
        roots: Vec<Complex>,
        inner: Box<Fft>,
    },
    Bluestein {
        roots: Vec<Complex>,
        chirp: Vec<Complex>,
        kernel_fft: Vec<Complex>,
    },
}

impl Fft {
    fn new(n: usize) -> Result<Self, String> {
        if n.is_power_of_two() {
            return Ok(Self {
                scratch_len: 0,
                algorithm: FftAlgorithm::Radix2(roots(n, n / 2)),
            });
        }
        if let Some(radix) = [2, 3, 5, 7]
            .into_iter()
            .find(|factor| n % factor == 0 && n > *factor)
        {
            let inner = Box::new(Self::new(n / radix)?);
            return Ok(Self {
                scratch_len: n.max(inner.scratch_len),
                algorithm: FftAlgorithm::Factor {
                    radix,
                    roots: roots(n, n),
                    inner,
                },
            });
        }
        // Bluestein reduces an arbitrary-length DFT to a linear convolution
        // of chirps, evaluated with a power-of-two FFT. No padding of the
        // image or resampling of its frequency bins is involved.
        let convolution_len = n
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| "MeasureImageQuality FFT dimension is too large".to_owned())?;
        let roots = roots(convolution_len, convolution_len / 2);
        let chirp: Vec<_> = (0..n)
            .map(|i| {
                let phase = ((i as u128 * i as u128) % (2 * n) as u128) as f64;
                Complex::phase(-std::f64::consts::PI * phase / n as f64)
            })
            .collect();
        let mut kernel_fft = vec![Complex::default(); convolution_len];
        kernel_fft[0] = chirp[0].conj();
        for (i, &value) in chirp.iter().enumerate().skip(1) {
            kernel_fft[i] = value.conj();
            kernel_fft[convolution_len - i] = value.conj();
        }
        radix2(&mut kernel_fft, &roots);
        Ok(Self {
            scratch_len: convolution_len,
            algorithm: FftAlgorithm::Bluestein {
                roots,
                chirp,
                kernel_fft,
            },
        })
    }

    fn transform(
        &self,
        input: &[Complex],
        stride: usize,
        out: &mut [Complex],
        scratch: &mut [Complex],
    ) {
        let n = out.len();
        let values = input.iter().step_by(stride).take(n);
        // A constant line has no non-DC frequency components.
        if values.clone().all(|x| *x == input[0]) {
            out.fill(Complex::default());
            out[0] = input[0].mul(Complex::real(n as f64));
            return;
        }
        match &self.algorithm {
            FftAlgorithm::Radix2(roots) => {
                for (dst, &src) in out.iter_mut().zip(values) {
                    *dst = src;
                }
                radix2(out, roots);
            }
            FftAlgorithm::Factor {
                radix,
                roots,
                inner,
            } => {
                let size = n / radix;
                for (i, band) in out.chunks_exact_mut(size).enumerate() {
                    inner.transform(&input[i * stride..], stride * radix, band, scratch);
                }
                for (k, dst) in scratch[..n].iter_mut().enumerate() {
                    let j = k % size;
                    *dst = out[j];
                    for i in 1..*radix {
                        // Reduce phase before multiplication so an otherwise
                        // impractical 32-bit allocation cannot wrap an index.
                        let phase = (i as u64 * k as u64 % n as u64) as usize;
                        *dst = dst.add(out[i * size + j].mul(roots[phase]));
                    }
                }
                out.copy_from_slice(&scratch[..n]);
            }
            FftAlgorithm::Bluestein {
                roots,
                chirp,
                kernel_fft,
            } => {
                let tmp = &mut scratch[..self.scratch_len];
                tmp.fill(Complex::default());
                for ((dst, &src), &w) in tmp.iter_mut().zip(values).zip(chirp) {
                    *dst = src.mul(w);
                }
                radix2(tmp, roots);
                for (p, &kernel) in tmp.iter_mut().zip(kernel_fft) {
                    *p = p.mul(kernel).conj();
                }
                radix2(tmp, roots);
                let normalize = Complex::real(1.0 / tmp.len() as f64);
                for ((dst, &src), &w) in out.iter_mut().zip(tmp.iter()).zip(chirp) {
                    *dst = src.conj().mul(normalize).mul(w);
                }
            }
        }
    }
}

fn roots(n: usize, len: usize) -> Vec<Complex> {
    (0..len)
        .map(|k| {
            // Exact axis twiddles keep periodic signals from creating spurious
            // power merely because sin(pi) rounded to a nonzero value.
            if k == 0 {
                Complex::real(1.0)
            } else if n % 2 == 0 && k == n / 2 {
                Complex::real(-1.0)
            } else if n % 4 == 0 && k == n / 4 {
                Complex { re: 0.0, im: -1.0 }
            } else if n % 4 == 0 && k == n - n / 4 {
                Complex { re: 0.0, im: 1.0 }
            } else {
                Complex::phase(-std::f64::consts::TAU * k as f64 / n as f64)
            }
        })
        .collect()
}

fn radix2(values: &mut [Complex], roots: &[Complex]) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    let shift = usize::BITS - n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut width = 2;
    loop {
        let half = width / 2;
        let spacing = n / width;
        for group in values.chunks_exact_mut(width) {
            let (lower, upper) = group.split_at_mut(half);
            for (i, (a, b)) in lower.iter_mut().zip(upper).enumerate() {
                let product = b.mul(roots[i * spacing]);
                *b = a.sub(product);
                *a = a.add(product);
            }
        }
        if width == n {
            break;
        }
        width *= 2;
    }
}
