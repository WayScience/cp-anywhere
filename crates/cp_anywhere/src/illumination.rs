//! CellProfiler 4.2.8 `CorrectIlluminationCalculate` ("Each" mode),
//! oracle-verified against a pinned reference CellProfiler/centrosome/
//! scipy/skimage install (see `scripts/reference/generate_illumination_
//! calculate.py` and `fixtures/illumination_calculate/*.json`).
//!
//! From-scratch Rust port -- no `scipy`/`skimage`/`centrosome` dependency
//! -- of `correctilluminationcalculate.py`'s `run()` (`EA_EACH` branch):
//! `preprocess_image_for_averaging` -> `apply_dilation` ->
//! `apply_smoothing` -> `apply_scaling`.
//!
//! Covers 5 of the module's 7 smoothing methods: `No smoothing`,
//! `Median Filter`, `Gaussian Filter`, `Fit Polynomial`, and
//! `Smooth to Average`. `Convex Hull` (a multi-pass geometric
//! transform, `centrosome.filter.convex_hull_transform`) and `Splines`
//! (an iterative 2D background-classification spline fit,
//! `centrosome.bg_compensate.backgr`) are explicit, documented,
//! not-yet-implemented gaps -- [`calculate_each`] returns a clear
//! [`Err`] naming them; see `spec/illumination.md`.
//!
//! ## The float32 round-trip contract (confirmed empirically, not guessed)
//!
//! Real CellProfiler's `cellprofiler_core.image.Image.__init__` always
//! takes `convert=True` by default, and `set_image` *unconditionally*
//! downcasts to `float32` before storing -- even for already-floating
//! input (`img = img.astype(numpy.float32)` runs regardless of dtype;
//! only `convert=False` or a `bool` dtype skip it). Every pipeline
//! stage here that constructs a *new* `Image(...)` therefore loses
//! precision to `float32` at that exact point, and the NEXT stage
//! consumes the already-rounded values. A stage that instead returns
//! its input unchanged (`SM_NONE`, `rescale_option == "No"`, dilation
//! disabled, `Regular` intensity choice with no mask) constructs no
//! new `Image` and introduces no extra rounding. This was confirmed by
//! checking real fixture outputs against an exact `as f32 as f64`
//! round-trip, not inferred from reading `_image.py` alone.

const ROBUST_FACTOR: f64 = 0.02;

/// `IC_REGULAR` / `IC_BACKGROUND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntensityChoice {
    Regular,
    Background,
}

/// `SM_*` smoothing method constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingMethod {
    None,
    MedianFilter,
    GaussianFilter,
    FitPolynomial,
    ToAverage,
    ConvexHull,
    Splines,
}

/// `FI_*` smoothing-filter-size method constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterSizeMethod {
    Automatic,
    ObjectSize,
    Manually,
}

/// Rescale option: `"Yes"` / `"No"` / `RE_MEDIAN` ("Median").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RescaleOption {
    Yes,
    No,
    Median,
}

/// Settings mirroring `CorrectIlluminationCalculate`'s module settings
/// (the subset relevant to `"Each"` mode and the 5 implemented
/// smoothing methods).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub intensity_choice: IntensityChoice,
    /// "Block size" (`IC_BACKGROUND` only).
    pub block_size: usize,
    pub dilate_objects: bool,
    /// "Dilation radius" (`dilate_objects` only).
    pub object_dilation_radius: usize,
    pub smoothing_method: SmoothingMethod,
    pub automatic_object_width: FilterSizeMethod,
    /// "Approximate object diameter" (`FI_OBJECT_SIZE` only).
    pub object_width: f64,
    /// "Smoothing filter size" (`FI_MANUALLY` only).
    pub size_of_smoothing_filter: f64,
    pub rescale_option: RescaleOption,
}

fn round_trip_f32(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = *v as f32 as f64;
    }
}

/// `fixup_scipy_ndimage_result`-equivalent `centrosome.cpmorphology.block`
/// partition, used only by the `IC_BACKGROUND` preprocessing step.
fn block_labels(rows: usize, cols: usize, block_size: usize) -> (Vec<i64>, usize, usize) {
    let block_size = block_size.max(1);
    let i_max = (rows / block_size).max(1);
    let j_max = (cols / block_size).max(1);
    let mult_i = i_max as f64 / rows as f64;
    let mult_j = j_max as f64 / cols as f64;
    let mut labels = vec![0i64; rows * cols];
    for r in 0..rows {
        let bi = (r as f64 * mult_i) as i64;
        for c in 0..cols {
            let bj = (c as f64 * mult_j) as i64;
            labels[r * cols + c] = bi * j_max as i64 + bj;
        }
    }
    (labels, i_max, j_max)
}

/// `preprocess_image_for_averaging`'s `IC_BACKGROUND` branch: minimum
/// pixel intensity within each block of `block_size x block_size`,
/// excluding masked-out pixels from every block's minimum and leaving
/// masked-out output pixels at `0.0` (matching `min_block`'s
/// zero-initialization and the `labels[~mask] = -1` exclusion).
fn block_minimum(
    pixels: &[f64],
    mask: &[bool],
    rows: usize,
    cols: usize,
    block_size: usize,
) -> Vec<f64> {
    let (labels, i_max, j_max) = block_labels(rows, cols, block_size);
    let bins = i_max * j_max;
    let mut minima = vec![f64::INFINITY; bins];
    for idx in 0..rows * cols {
        if !mask[idx] {
            continue;
        }
        let bin = labels[idx] as usize;
        if pixels[idx] < minima[bin] {
            minima[bin] = pixels[idx];
        }
    }
    let mut out = vec![0.0; rows * cols];
    for idx in 0..rows * cols {
        if mask[idx] {
            let bin = labels[idx] as usize;
            // An empty block (no unmasked pixel landed in it) has no
            // real scipy.ndimage.minimum value; 0.0 matches no real
            // fixture here (every block has >=1 unmasked pixel in the
            // oracle-verified cases) but avoids propagating +inf.
            out[idx] = if minima[bin].is_finite() {
                minima[bin]
            } else {
                0.0
            };
        }
    }
    out
}

/// `centrosome.smooth.smooth_with_function_and_mask`: apply a linear
/// (convolution-like) `function` to the image with masked pixels
/// zeroed out, then divide by the same function applied to the mask
/// itself (as 0.0/1.0), recovering the effect of smoothing from only
/// the significant pixels.
fn smooth_with_function_and_mask(
    pixels: &[f64],
    mask: &[bool],
    function: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    let mask_f64: Vec<f64> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
    let bleed_over = function(&mask_f64);
    let masked_image: Vec<f64> = pixels
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m { v } else { 0.0 })
        .collect();
    let smoothed = function(&masked_image);
    smoothed
        .iter()
        .zip(&bleed_over)
        .map(|(&s, &b)| s / (b + f64::EPSILON))
        .collect()
}

/// `centrosome.smooth.circular_gaussian_kernel`: a `(2*radius+1)^2`
/// Gaussian kernel restricted to the circle of the given `radius`,
/// normalized to sum to 1.
fn circular_gaussian_kernel(sd: f64, radius: usize) -> (Vec<f64>, usize) {
    let size = 2 * radius + 1;
    let r = radius as f64;
    let mut kernel = vec![0.0; size * size];
    for di in 0..size {
        let fi = di as f64 - r;
        for dj in 0..size {
            let fj = dj as f64 - r;
            let ni = fi / r;
            let nj = fj / r;
            if ni * ni + nj * nj <= 1.0 {
                let si = fi * r / sd / r;
                let sj = fj * r / sd / r;
                kernel[di * size + dj] = (-(si * si + sj * sj) / (2.0 * sd * sd)).exp();
            }
        }
    }
    let sum: f64 = kernel.iter().sum();
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    (kernel, size)
}

/// Zero-padded ("constant", `cval=0`) 2D correlation with a
/// `(2*radius+1)^2` kernel -- equivalent to `scipy.ndimage.convolve`
/// for this project's only caller, [`circular_gaussian_kernel`], which
/// is symmetric under `(i, j) -> (-i, -j)` so convolution and
/// correlation coincide.
fn correlate_constant_zero(
    image: &[f64],
    rows: usize,
    cols: usize,
    kernel: &[f64],
    ksize: usize,
) -> Vec<f64> {
    let kr = (ksize / 2) as isize;
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for ki in 0..ksize {
                let ii = r as isize + ki as isize - kr;
                if ii < 0 || ii >= rows as isize {
                    continue;
                }
                for kj in 0..ksize {
                    let jj = c as isize + kj as isize - kr;
                    if jj < 0 || jj >= cols as isize {
                        continue;
                    }
                    acc += image[ii as usize * cols + jj as usize] * kernel[ki * ksize + kj];
                }
            }
            out[r * cols + c] = acc;
        }
    }
    out
}

/// `scipy.ndimage.gaussian_filter(image, sigma, mode="constant", cval=0)`:
/// separable 1D Gaussian, truncated at `truncate=4.0` standard
/// deviations (`radius = floor(4*sigma + 0.5)`), zero-padded border.
/// Confirmed by direct comparison against real scipy output, not
/// inferred from its docstring alone.
fn gaussian_filter_constant_zero(image: &[f64], rows: usize, cols: usize, sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut weights = Vec::with_capacity((2 * radius + 1) as usize);
    for k in -radius..=radius {
        weights.push((-0.5 * (k as f64 / sigma).powi(2)).exp());
    }
    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
    // Horizontal pass (along columns), then vertical pass (along rows).
    let mut tmp = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (offset, &w) in (-radius..=radius).zip(&weights) {
                let jj = c as isize + offset;
                if jj >= 0 && jj < cols as isize {
                    acc += image[r * cols + jj as usize] * w;
                }
            }
            tmp[r * cols + c] = acc;
        }
    }
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (offset, &w) in (-radius..=radius).zip(&weights) {
                let ii = r as isize + offset;
                if ii >= 0 && ii < rows as isize {
                    acc += tmp[ii as usize * cols + c] * w;
                }
            }
            out[r * cols + c] = acc;
        }
    }
    out
}

/// `centrosome.smooth.fit_polynomial`: least-squares fit of
/// `A*x^2 + B*y^2 + C*x*y + D*x + E*y + F` to pixels where
/// `mask & (pixel_data > 0)`, evaluated over the whole image, then
/// clipped to `[0, 1]`. Solved via the normal equations (a 6x6 linear
/// system) rather than SVD-based `scipy.linalg.lstsq`; the two agree
/// to this project's parity tolerance for every well-conditioned
/// oracle-verified case.
fn fit_polynomial(pixels: &[f64], mask: &[bool], rows: usize, cols: usize) -> Vec<f64> {
    let fit_mask: Vec<bool> = pixels
        .iter()
        .zip(mask)
        .map(|(&p, &m)| m && p > 0.0)
        .collect();
    if !fit_mask.iter().any(|&m| m) {
        return pixels.to_vec();
    }
    // Columns: [x, y, x^2, y^2, x*y, 1].
    let mut ata = [[0.0_f64; 6]; 6];
    let mut atb = [0.0_f64; 6];
    for r in 0..rows {
        let x = r as f64;
        for c in 0..cols {
            if !fit_mask[r * cols + c] {
                continue;
            }
            let y = c as f64;
            let row = [x, y, x * x, y * y, x * y, 1.0];
            let b = pixels[r * cols + c];
            for i in 0..6 {
                atb[i] += row[i] * b;
                for j in 0..6 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let coeffs = solve_6x6(ata, atb);
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let x = r as f64;
        for c in 0..cols {
            let y = c as f64;
            let row = [x, y, x * x, y * y, x * y, 1.0];
            let mut v = 0.0;
            for i in 0..6 {
                v += coeffs[i] * row[i];
            }
            out[r * cols + c] = v.clamp(0.0, 1.0);
        }
    }
    out
}

/// Gaussian elimination with partial pivoting for a 6x6 linear system
/// (the normal equations `A^T A x = A^T b` from [`fit_polynomial`]).
fn solve_6x6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> [f64; 6] {
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        if diag.abs() < 1e-300 {
            continue;
        }
        for row in (col + 1)..6 {
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            let pivot_row = a[col];
            for (k, &pivot_value) in pivot_row.iter().enumerate().skip(col) {
                a[row][k] -= factor * pivot_value;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 6];
    for col in (0..6).rev() {
        let diag = a[col][col];
        let mut v = b[col];
        for (k, coeff) in x.iter().enumerate().skip(col + 1) {
            v -= a[col][k] * coeff;
        }
        x[col] = if diag.abs() < 1e-300 { 0.0 } else { v / diag };
    }
    x
}

/// `centrosome.cpmorphology.strel_disk`: a boolean disk footprint of
/// the given integer radius.
fn strel_disk(radius: usize) -> (Vec<bool>, usize) {
    let size = 2 * radius + 1;
    let r2 = (radius * radius) as i64;
    let mut strel = vec![false; size * size];
    for di in 0..size {
        let fi = di as i64 - radius as i64;
        for dj in 0..size {
            let fj = dj as i64 - radius as i64;
            if fi * fi + fj * fj <= r2 {
                strel[di * size + dj] = true;
            }
        }
    }
    (strel, size)
}

/// `skimage.filters.median(data, strel, behavior="rank")`: a sliding
/// rank-median over the disk footprint, considering only in-bounds
/// footprint cells at the border ("valid window", confirmed by direct
/// comparison against real skimage output -- no padding/reflection).
/// Ties at an even window size take the upper-median
/// (`sorted[len / 2]`, confirmed the same way).
fn rank_median_disk(values: &[u32], rows: usize, cols: usize, radius: usize) -> Vec<u32> {
    let (strel, size) = strel_disk(radius);
    let r = radius as isize;
    let mut out = vec![0u32; rows * cols];
    let mut window = Vec::with_capacity(size * size);
    for row in 0..rows {
        for col in 0..cols {
            window.clear();
            for ki in 0..size {
                let ii = row as isize + ki as isize - r;
                if ii < 0 || ii >= rows as isize {
                    continue;
                }
                for kj in 0..size {
                    if !strel[ki * size + kj] {
                        continue;
                    }
                    let jj = col as isize + kj as isize - r;
                    if jj < 0 || jj >= cols as isize {
                        continue;
                    }
                    window.push(values[ii as usize * cols + jj as usize]);
                }
            }
            window.sort_unstable();
            out[row * cols + col] = window[window.len() / 2];
        }
    }
    out
}

/// `smooth_plane`'s per-method dispatch, over one 2D plane.
///
/// `is_f32` tracks whether `pixels` already lost precision to an
/// earlier `float32` round-trip (preprocessing with a mask, or
/// dilation): [`SmoothingMethod::MedianFilter`]'s `pixel_data * 65535`
/// multiply happens in whatever precision `pixel_data` already has in
/// real CellProfiler, and a value that never went through an `Image`
/// round-trip stays full `f64` precision there (confirmed by a real
/// mismatch when this was forced to always multiply in `f32`).
fn smooth_plane(
    pixels: &[f64],
    mask: &[bool],
    rows: usize,
    cols: usize,
    method: SmoothingMethod,
    filter_size: f64,
    is_f32: bool,
) -> Result<Vec<f64>, String> {
    let sigma = filter_size / 2.35;
    match method {
        SmoothingMethod::None => Ok(pixels.to_vec()),
        SmoothingMethod::FitPolynomial => Ok(fit_polynomial(pixels, mask, rows, cols)),
        SmoothingMethod::GaussianFilter => {
            Ok(smooth_with_function_and_mask(pixels, mask, |plane| {
                gaussian_filter_constant_zero(plane, rows, cols, sigma)
            }))
        }
        SmoothingMethod::MedianFilter => {
            let filter_sigma = (sigma + 0.5).max(1.0) as usize;
            let rescaled: Vec<u32> = pixels
                .iter()
                .zip(mask)
                .map(|(&p, &m)| {
                    // See the `is_f32` doc comment above: only
                    // multiply in `f32` precision when `pixels`
                    // already lost precision to an earlier round-trip.
                    let scaled = if is_f32 {
                        ((p as f32) * 65535.0_f32) as i64
                    } else {
                        (p * 65535.0) as i64
                    };
                    let scaled = scaled.clamp(0, u32::MAX as i64) as u32;
                    if m {
                        scaled
                    } else {
                        0
                    }
                })
                .collect();
            let filtered = rank_median_disk(&rescaled, rows, cols, filter_sigma);
            Ok(filtered
                .iter()
                .map(|&v| v as f32 as f64 / 65535.0)
                .collect())
        }
        SmoothingMethod::ToAverage => {
            let (sum, count) = pixels
                .iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .fold((0.0, 0usize), |(s, n), (&p, _)| (s + p, n + 1));
            let mean = if count > 0 {
                sum / count as f64
            } else {
                f64::NAN
            };
            Ok(vec![mean; rows * cols])
        }
        SmoothingMethod::ConvexHull => Err(
            "CorrectIlluminationCalculate smoothing_method = 'Convex Hull' is not yet \
             implemented (centrosome.filter.convex_hull_transform is a multi-pass \
             geometric algorithm, a separate, documented follow-on; see spec/illumination.md)"
                .to_string(),
        ),
        SmoothingMethod::Splines => Err(
            "CorrectIlluminationCalculate smoothing_method = 'Splines' is not yet \
             implemented (centrosome.bg_compensate.backgr is an iterative 2D spline \
             background fit, a separate, documented follow-on; see spec/illumination.md)"
                .to_string(),
        ),
    }
}

fn smoothing_filter_size(
    rows: usize,
    cols: usize,
    method: FilterSizeMethod,
    object_width: f64,
    manual_size: f64,
) -> f64 {
    match method {
        FilterSizeMethod::Manually => manual_size,
        FilterSizeMethod::ObjectSize => object_width * 2.35 / 3.5,
        FilterSizeMethod::Automatic => (rows.max(cols) as f64 / 40.0).min(30.0),
    }
}

/// `apply_scaling`: rescale so pixel intensities are all `>= 1`
/// (`"Yes"`, using the 2nd-percentile-like `ROBUST_FACTOR` minimum) or
/// divide by the median (`"Median"`); `"No"` is the identity.
fn apply_scaling(pixels: &[f64], mask: &[bool], option: RescaleOption) -> Vec<f64> {
    if option == RescaleOption::No {
        return pixels.to_vec();
    }
    let mut sorted: Vec<f64> = pixels
        .iter()
        .zip(mask)
        .filter(|(&p, &m)| p > 0.0 && m)
        .map(|(&p, _)| p)
        .collect();
    if sorted.is_empty() {
        return pixels.to_vec();
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    // "Yes" also clamps every pixel below the robust minimum up to it
    // before dividing (`pixel_data[pixel_data < robust_minimum] =
    // robust_minimum`), so clamped pixels end up at exactly `1.0`
    // rather than some smaller fraction; "Median" does not clamp.
    let robust_minimum = match option {
        RescaleOption::Yes => {
            let idx = (sorted.len() as f64 * ROBUST_FACTOR) as usize;
            sorted[idx]
        }
        RescaleOption::Median => sorted[sorted.len() / 2],
        RescaleOption::No => unreachable!(),
    };
    if robust_minimum == 0.0 {
        return pixels.to_vec();
    }
    pixels
        .iter()
        .map(|&p| {
            let clamped = if option == RescaleOption::Yes {
                p.max(robust_minimum)
            } else {
                p
            };
            clamped / robust_minimum
        })
        .collect()
}

/// `preprocess_image_for_averaging`: one image's contribution to the
/// (possibly averaged-across-many-images) working image, plus
/// whether constructing it already lost precision to a `float32`
/// round-trip (`IC_BACKGROUND` always; `IC_REGULAR` only when a mask
/// was supplied) -- reused identically by both `"Each"` mode
/// ([`calculate_each`]) and `"All"` mode accumulation
/// ([`calculate_across`]), matching real CellProfiler's own single
/// `preprocess_image_for_averaging` method.
fn preprocess_for_averaging(
    pixels: &[f64],
    mask: &[bool],
    has_mask: bool,
    rows: usize,
    cols: usize,
    config: &Config,
) -> (Vec<f64>, bool) {
    match config.intensity_choice {
        IntensityChoice::Regular => {
            if has_mask {
                let mut masked: Vec<f64> = pixels
                    .iter()
                    .zip(mask)
                    .map(|(&p, &m)| if m { p } else { 0.0 })
                    .collect();
                round_trip_f32(&mut masked);
                (masked, true)
            } else {
                (pixels.to_vec(), false)
            }
        }
        IntensityChoice::Background => {
            let mut values = block_minimum(pixels, mask, rows, cols, config.block_size);
            round_trip_f32(&mut values);
            (values, true)
        }
    }
}

/// `apply_dilation` -> `apply_smoothing` -> `apply_scaling`, given an
/// already-averaged working image (one preprocessed image for
/// `"Each"` mode, or the sum/count average of many for `"All"` mode)
/// and the combined mask that applies to it. `is_f32` is whether
/// `avg` already lost precision to an earlier round-trip (see
/// [`preprocess_for_averaging`] and the module docs).
fn finish_pipeline(
    mut avg: Vec<f64>,
    mask: &[bool],
    rows: usize,
    cols: usize,
    config: &Config,
    mut is_f32: bool,
) -> Result<Vec<f64>, String> {
    // apply_dilation
    if config.dilate_objects {
        let radius = config.object_dilation_radius.max(1);
        let (kernel, ksize) = circular_gaussian_kernel(radius as f64, radius * 3);
        let dilated = smooth_with_function_and_mask(&avg, mask, |plane| {
            correlate_constant_zero(plane, rows, cols, &kernel, ksize)
        });
        avg = dilated;
        round_trip_f32(&mut avg);
        is_f32 = true;
    }

    // apply_smoothing
    let smoothed = if config.smoothing_method == SmoothingMethod::None {
        avg
    } else {
        let filter_size = smoothing_filter_size(
            rows,
            cols,
            config.automatic_object_width,
            config.object_width,
            config.size_of_smoothing_filter,
        );
        let mut result = smooth_plane(
            &avg,
            mask,
            rows,
            cols,
            config.smoothing_method,
            filter_size,
            is_f32,
        )?;
        round_trip_f32(&mut result);
        result
    };

    // apply_scaling
    let scaled = if config.rescale_option == RescaleOption::No {
        smoothed
    } else {
        let mut result = apply_scaling(&smoothed, mask, config.rescale_option);
        round_trip_f32(&mut result);
        result
    };

    Ok(scaled)
}

/// Calculate the illumination function for a single image ("Each"
/// mode): `preprocess_image_for_averaging` -> `apply_dilation` ->
/// `apply_smoothing` -> `apply_scaling`, with real CellProfiler's own
/// `float32` round-trip at every stage that constructs a new `Image`
/// (see the module docs).
///
/// `mask = None` matches real CellProfiler's `Image.has_mask == False`
/// (an implicit all-significant mask that changes which branches
/// introduce a round-trip, not the numeric result of any one branch).
pub fn calculate_each(
    pixels: &[f64],
    mask: Option<&[bool]>,
    rows: usize,
    cols: usize,
    config: &Config,
) -> Result<Vec<f64>, String> {
    if pixels.len() != rows * cols {
        return Err("pixel buffer length does not match rows*cols".to_string());
    }
    let has_mask = mask.is_some();
    let all_true = vec![true; rows * cols];
    let mask = mask.unwrap_or(&all_true);
    let (avg, is_f32) = preprocess_for_averaging(pixels, mask, has_mask, rows, cols, config);
    finish_pipeline(avg, mask, rows, cols, config, is_f32)
}

/// Calculate the illumination function across many image sets ("All:
/// First cycle" / "All: Across cycles" mode): accumulate every
/// image's own [`preprocess_for_averaging`] contribution into a
/// per-pixel sum and mask-count (CellProfiler's own
/// `CorrectIlluminationImageProvider.add()`), divide to get the
/// average (CellProfiler's own `calculate_image()`, which
/// unconditionally constructs a new `Image` -- always a `float32`
/// round-trip, unlike `"Each"` mode's `Regular`+no-mask special
/// case), then run the same `apply_dilation` -> `apply_smoothing` ->
/// `apply_scaling` pipeline as `"Each"` mode.
///
/// `images` is `(pixels, mask)` per image set, in any order (the sum
/// is commutative); an empty `images` is an error, matching real
/// CellProfiler's own `ZeroDivisionError`-shaped failure for a
/// zero-image group (this project raises a clear [`Err`] instead).
pub fn calculate_across(
    images: &[(&[f64], Option<&[bool]>)],
    rows: usize,
    cols: usize,
    config: &Config,
) -> Result<Vec<f64>, String> {
    if images.is_empty() {
        return Err("calculate_across needs at least one image".to_string());
    }
    let size = rows * cols;
    let mut sum = vec![0.0_f64; size];
    let mut count = vec![0u32; size];
    let all_true = vec![true; size];
    for &(pixels, mask) in images {
        if pixels.len() != size {
            return Err("pixel buffer length does not match rows*cols".to_string());
        }
        let has_mask = mask.is_some();
        let mask = mask.unwrap_or(&all_true);
        if mask.len() != size {
            return Err("mask buffer length does not match rows*cols".to_string());
        }
        let (contribution, _) =
            preprocess_for_averaging(pixels, mask, has_mask, rows, cols, config);
        for idx in 0..size {
            if mask[idx] {
                sum[idx] += contribution[idx];
                count[idx] += 1;
            }
        }
    }
    let mut avg = vec![0.0_f64; size];
    let mut combined_mask = vec![false; size];
    for idx in 0..size {
        if count[idx] > 0 {
            avg[idx] = sum[idx] / count[idx] as f64;
            combined_mask[idx] = true;
        }
    }
    round_trip_f32(&mut avg);
    finish_pipeline(avg, &combined_mask, rows, cols, config, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_values(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9_f64.max(e.abs() * 1e-6),
                "{label}[{idx}]: got {a}, want {e}"
            );
        }
    }

    #[test]
    fn regular_no_mask_no_smoothing_no_rescale_is_identity() {
        let pixels = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let config = Config {
            intensity_choice: IntensityChoice::Regular,
            block_size: 6,
            dilate_objects: false,
            object_dilation_radius: 2,
            smoothing_method: SmoothingMethod::None,
            automatic_object_width: FilterSizeMethod::Automatic,
            object_width: 10.0,
            size_of_smoothing_filter: 10.0,
            rescale_option: RescaleOption::No,
        };
        let actual = calculate_each(&pixels, None, 2, 3, &config).unwrap();
        assert_values(&actual, &pixels, "identity");
    }

    #[test]
    fn convex_hull_and_splines_are_explicit_errors() {
        let pixels = vec![0.1; 9];
        for method in [SmoothingMethod::ConvexHull, SmoothingMethod::Splines] {
            let config = Config {
                intensity_choice: IntensityChoice::Regular,
                block_size: 6,
                dilate_objects: false,
                object_dilation_radius: 2,
                smoothing_method: method,
                automatic_object_width: FilterSizeMethod::Automatic,
                object_width: 10.0,
                size_of_smoothing_filter: 10.0,
                rescale_option: RescaleOption::No,
            };
            assert!(calculate_each(&pixels, None, 3, 3, &config).is_err());
        }
    }
}
